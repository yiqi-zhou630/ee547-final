"""
Model Training Script
Train a Cross-Encoder model using SciEntsBank dataset for automatic scoring
Supports multi-task learning: regression score (0-5) and 5-class classification labels 
(correct, contradictory, partially correct but incomplete, irrelevant, non-domain)

Data Splitting Strategy:
- Exclude test_ud (unseen domain) subset
- Merge test_ua and test_uq, shuffle and split into:
  * Training portion: for fine-tuning and testing the model
  * Hold-out portion: for subsequent system testing phase, simulating real student submission data
"""

import os
# Set environment variables to avoid transformers importing tensorflow (not needed for this project)
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_TORCH'] = '0'  # Ensure PyTorch is used
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizers parallel warnings

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, DatasetDict
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SciEntsBankDataset(Dataset):
    """SciEntsBank Dataset Wrapper Class"""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 512,
        use_features: bool = False
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_features = use_features
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Build input: concatenate "question + reference answer" with "student answer"
        question = item['question']
        reference_answer = item['reference_answer']
        student_answer = item['student_answer']
        
        # Concatenation format: [CLS] question [SEP] reference_answer [SEP] student_answer [SEP]
        text_a = f"{question} {reference_answer}"
        text_b = student_answer
        
        # Tokenize
        encoded = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels (directly use 5-way labels)
        label_5way = item['label']
        
        # Convert to numerical score (0-5)
        score = self._label_to_score(label_5way)
        
        result = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0),
            'label_class': torch.tensor(label_5way, dtype=torch.long),  # Directly use 5-way labels
            'label_score': torch.tensor(score, dtype=torch.float32)
        }
        
        # Optional: add interpretable features
        if self.use_features:
            features = self._extract_features(question, reference_answer, student_answer)
            result['features'] = torch.tensor(features, dtype=torch.float32)
        
        return result
    
    def _get_label_name(self, label_5way: int) -> str:
        """Get 5-way label name"""
        # 5-way label mapping:
        # 0=correct
        # 1=contradictory
        # 2=partially_correct_incomplete
        # 3=irrelevant
        # 4=non_domain
        label_names = [
            'correct',           # 0: correct
            'contradictory',     # 1: contradictory
            'partially_correct_incomplete',  # 2: partially correct but incomplete
            'irrelevant',        # 3: irrelevant
            'non_domain'         # 4: non-domain
        ]
        return label_names[label_5way] if 0 <= label_5way < len(label_names) else 'unknown'
    
    def _label_to_score(self, label_5way: int) -> float:
        """Convert label to score (0-5)"""
        # Map labels to scores
        score_mapping = {
            0: 5.0,  # correct -> 5 points
            1: 1.0,  # contradictory -> 1 point
            2: 3.0,  # partially_correct_incomplete -> 3 points
            3: 0.5,  # irrelevant -> 0.5 points
            4: 0.0   # non_domain -> 0 points
        }
        return score_mapping.get(label_5way, 0.0)
    
    def _extract_features(
        self,
        question: str,
        reference_answer: str,
        student_answer: str
    ) -> List[float]:
        """Extract interpretable features"""
        # Keyword overlap rate
        ref_words = set(reference_answer.lower().split())
        student_words = set(student_answer.lower().split())
        keyword_overlap = len(ref_words & student_words) / max(len(ref_words), 1)
        
        # Answer length ratio
        length_ratio = len(student_answer) / max(len(reference_answer), 1)
        
        # Sentence similarity (simple vocabulary overlap)
        all_words = ref_words | student_words
        if len(all_words) > 0:
            jaccard_similarity = len(ref_words & student_words) / len(all_words)
        else:
            jaccard_similarity = 0.0
        
        # Fluency (simple heuristic: based on punctuation and length)
        fluency = min(len(student_answer.split()) / 10.0, 1.0)  # Normalize to 0-1
        
        return [keyword_overlap, length_ratio, jaccard_similarity, fluency]


class MultiTaskCrossEncoder(nn.Module):
    """Multi-task Cross-Encoder Model"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        num_classes: int = 5,  # 5-class classification: correct, contradictory, partially correct but incomplete, irrelevant, non-domain
        use_features: bool = False,
        feature_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_features = use_features
        
        # Load pretrained cross-encoder model
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # If using feature fusion, add feature dimensions
        classifier_input_size = hidden_size
        if use_features:
            classifier_input_size += feature_dim
        
        # Classification head (5-way: correct, contradictory, partially_correct_incomplete, irrelevant, non_domain)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Regression head (0-5 score)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output 0-1, then multiply by 5 to get 0-5
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get [CLS] vector
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Feature fusion
        if self.use_features and features is not None:
            cls_embedding = torch.cat([cls_embedding, features], dim=1)
        
        # Classification output
        logits = self.classifier(cls_embedding)
        
        # Regression output (0-5)
        score = self.regressor(cls_embedding) * 5.0
        
        return {
            'logits': logits,
            'score': score.squeeze(-1)
        }


class MultiTaskTrainer(Trainer):
    """Custom Trainer supporting multi-task loss"""
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # Classification loss weight
        self.beta = beta    # Regression loss weight
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute multi-task loss
        
        Args:
            model: Model
            inputs: Input data
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (optional parameter passed by newer transformers versions)
        """
        # Convert inputs to regular dict (if it's a special type like BatchEncoding)
        if hasattr(inputs, 'keys'):
            inputs_dict = {k: inputs[k] for k in inputs.keys()}
        else:
            inputs_dict = dict(inputs) if isinstance(inputs, dict) else inputs
        
        # Get labels
        labels_class = inputs_dict.get('label_class')
        labels_score = inputs_dict.get('label_score')
        
        # If labels are not in inputs, there's a data format issue
        if labels_class is None or labels_score is None:
            # Try to get from inputs attributes
            if hasattr(inputs, 'label_class'):
                labels_class = inputs.label_class
            if hasattr(inputs, 'label_score'):
                labels_score = inputs.label_score
            
            if labels_class is None or labels_score is None:
                raise ValueError(
                    f"Missing labels in inputs. Available keys: {list(inputs_dict.keys()) if isinstance(inputs_dict, dict) else 'not a dict'}. "
                    "Make sure the dataset returns 'label_class' and 'label_score'."
                )
        
        # Get features (if present)
        features = inputs_dict.get('features', None)
        
        # Prepare model inputs
        model_inputs = {
            'input_ids': inputs_dict['input_ids'],
            'attention_mask': inputs_dict['attention_mask'],
        }
        if 'token_type_ids' in inputs_dict:
            model_inputs['token_type_ids'] = inputs_dict['token_type_ids']
        
        # Forward pass
        outputs = model(**model_inputs, features=features)
        
        # Compute loss
        loss_cls = self.criterion_cls(outputs['logits'], labels_class)
        loss_reg = self.criterion_reg(outputs['score'], labels_score)
        
        # Weighted combination
        loss = self.alpha * loss_cls + self.beta * loss_reg
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to support multi-task output"""
        has_labels = 'label_class' in inputs
        inputs = self._prepare_inputs(inputs)
        
        # Get labels
        labels_class = inputs.pop('label_class', None)
        labels_score = inputs.pop('label_score', None)
        features = inputs.pop('features', None)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids'),
                features=features
            )
            
            loss = None
            if has_labels:
                loss_cls = self.criterion_cls(outputs['logits'], labels_class)
                loss_reg = self.criterion_reg(outputs['score'], labels_score)
                loss = self.alpha * loss_cls + self.beta * loss_reg
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Prepare prediction results
        logits = outputs['logits'].detach()
        scores = outputs['score'].detach()
        
        if has_labels:
            labels = {
                'label_class': labels_class.detach(),
                'label_score': labels_score.detach()
            }
        else:
            labels = None
        
        return (loss, {'logits': logits, 'score': scores}, labels)


def load_and_preprocess_data(
    dataset_name: str = "nkazi/SciEntsBank",
    holdout_ratio: float = 0.3,
    random_seed: int = 42
) -> DatasetDict:
    """
    Load and preprocess SciEntsBank dataset
    
    Args:
        dataset_name: Dataset name
        holdout_ratio: Ratio of hold-out portion for subsequent system testing
        random_seed: Random seed for data shuffling and splitting
    
    Returns:
        Processed dataset dictionary containing:
        - train: Original training set + part of merged test set
        - eval: Validation set for model evaluation
        - holdout: Hold-out portion for subsequent system testing
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Exclude test_ud (unseen domain) subset
    logger.info("Excluding test_ud subset (unseen domain)")
    
    # Merge test_ua and test_uq
    logger.info("Merging test_ua and test_uq subsets")
    from datasets import concatenate_datasets
    test_combined = concatenate_datasets([dataset['test_ua'], dataset['test_uq']])
    
    # Shuffle data
    logger.info(f"Shuffling merged test data (random seed: {random_seed})")
    test_combined = test_combined.shuffle(seed=random_seed)
    
    # Calculate split point
    total_size = len(test_combined)
    holdout_size = int(total_size * holdout_ratio)
    train_size = total_size - holdout_size
    
    logger.info(f"Total size of merged test data: {total_size}")
    logger.info(f"Training portion size: {train_size} ({1-holdout_ratio:.1%})")
    logger.info(f"Hold-out portion size: {holdout_size} ({holdout_ratio:.1%})")
    
    # Split data
    test_train = test_combined.select(range(train_size))
    test_holdout = test_combined.select(range(train_size, total_size))
    
    # Merge original training set with training portion of test set
    train_final = concatenate_datasets([dataset['train'], test_train])
    
    # Create new dataset dictionary
    # Use part of test_ua as validation set (for evaluation during training)
    eval_size = min(500, len(test_train) // 4)  # Use 25% of training portion or at most 500 samples as validation set
    eval_dataset = test_train.select(range(eval_size))
    
    new_dataset = DatasetDict({
        'train': train_final,
        'eval': eval_dataset,
        'holdout': test_holdout
    })
    
    logger.info("Dataset processing completed")
    logger.info(f"Final training set size: {len(new_dataset['train'])}")
    logger.info(f"Validation set size: {len(new_dataset['eval'])}")
    logger.info(f"Hold-out portion size: {len(new_dataset['holdout'])}")
    
    return new_dataset


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    # Trainer's compute_metrics receives EvalPrediction object
    # Contains predictions and label_ids attributes
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # predictions is our custom dict format {'logits': ..., 'score': ...}
    if isinstance(predictions, dict):
        logits = predictions.get('logits')
        scores = predictions.get('score')
    elif isinstance(predictions, tuple):
        logits, scores = predictions
    else:
        # If just an array, assume it's logits
        logits = predictions
        scores = None
    
    # labels is also dict format {'label_class': ..., 'label_score': ...}
    if isinstance(labels, dict):
        true_classes = labels.get('label_class')
        true_scores = labels.get('label_score')
    elif isinstance(labels, tuple):
        true_classes, true_scores = labels
    else:
        true_classes = labels
        true_scores = None
    
    # Convert to numpy arrays
    if torch.is_tensor(logits):
        logits = logits.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    if torch.is_tensor(true_classes):
        true_classes = true_classes.cpu().numpy()
    if torch.is_tensor(true_scores):
        true_scores = true_scores.cpu().numpy()
    
    # Ensure numpy arrays
    logits = np.array(logits)
    if scores is not None:
        scores = np.array(scores)
    true_classes = np.array(true_classes)
    if true_scores is not None:
        true_scores = np.array(true_scores)
    
    # Get predicted classes
    if len(logits.shape) > 1:
        pred_classes = np.argmax(logits, axis=-1)
    else:
        pred_classes = logits
    
    metrics = {
        'accuracy': float(accuracy_score(true_classes, pred_classes)),
        'f1_macro': float(f1_score(true_classes, pred_classes, average='macro')),
        'f1_weighted': float(f1_score(true_classes, pred_classes, average='weighted'))
    }
    
    # Regression metrics
    if scores is not None and true_scores is not None:
        metrics['mse'] = float(mean_squared_error(true_scores, scores))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train multi-task cross-encoder model')
    parser.add_argument('--model_name', type=str, 
                       default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                       help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Model output directory')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Classification loss weight')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Regression loss weight')
    parser.add_argument('--use_features', action='store_true',
                       help='Whether to use interpretable features')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Steps to save checkpoint')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Steps to evaluate')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Steps to log')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data (use 5-class classification, exclude test_ud, merge test_ua and test_uq)
    dataset = load_and_preprocess_data(holdout_ratio=0.3, random_seed=42)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultiTaskCrossEncoder(
        model_name=args.model_name,
        num_classes=5,  # 5-class classification: correct, contradictory, partially correct but incomplete, irrelevant, non-domain
        use_features=args.use_features
    )
    
    # Create datasets
    train_dataset = SciEntsBankDataset(
        dataset['train'],
        tokenizer,
        max_length=args.max_length,
        use_features=args.use_features
    )
    
    eval_dataset = SciEntsBankDataset(
        dataset['eval'],
        tokenizer,
        max_length=args.max_length,
        use_features=args.use_features
    )
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(eval_dataset)}")
    logger.info(f"Hold-out portion size: {len(dataset['holdout'])}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",  # Newer versions use eval_strategy instead of evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False  # Keep all columns, including label_class and label_score
    )
    
    # Create Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        alpha=args.alpha,
        beta=args.beta,
        processing_class=tokenizer,  # Newer versions use processing_class instead of tokenizer
        compute_metrics=compute_metrics
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"Model saved to: {final_model_path}")
    
    # Evaluate validation set
    logger.info("Evaluating validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation set evaluation results: {eval_results}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Save hold-out dataset information (for subsequent system testing)
    holdout_info = {
        'holdout_size': len(dataset['holdout']),
        'holdout_path': os.path.join(args.output_dir, "holdout_dataset")
    }
    dataset['holdout'].save_to_disk(holdout_info['holdout_path'])
    logger.info(f"Hold-out dataset saved to: {holdout_info['holdout_path']}")
    
    with open(os.path.join(args.output_dir, "holdout_info.json"), "w") as f:
        json.dump(holdout_info, f, indent=2)


if __name__ == "__main__":
    main()

