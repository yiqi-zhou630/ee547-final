"""
Model Inference Script
Load trained model for prediction
"""

import os
# Set environment variables to avoid transformers importing tensorflow (not needed for this project)
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_TORCH'] = '0'  # Ensure PyTorch is used

import argparse
import torch
from transformers import AutoTokenizer
try:
    from .train import MultiTaskCrossEncoder
except ImportError:
    from model_training.train import MultiTaskCrossEncoder
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringModel:
    """Scoring Model Wrapper Class"""
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        use_features: bool = False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_features = use_features
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        # First try to load base model name from saved config
        config_path = f"{model_path}/config.json"
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_model_name = config.get('_name_or_path', model_path)
        else:
            base_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        self.model = MultiTaskCrossEncoder(
            model_name=base_model_name,
            num_classes=5,  # 5-class classification: correct, contradictory, partially correct but incomplete, irrelevant, non-domain
            use_features=use_features
        )
        
        # Load weights - try different formats
        loaded = False
        
        # Try pytorch_model.bin first
        model_file = f"{model_path}/pytorch_model.bin"
        if os.path.exists(model_file):
            try:
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                loaded = True
                logger.info("Loaded model from pytorch_model.bin")
            except Exception as e:
                logger.warning(f"Failed to load pytorch_model.bin: {e}")
        
        # Try model.safetensors
        if not loaded:
            model_file = f"{model_path}/model.safetensors"
            if os.path.exists(model_file):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                    # Load with strict=False to ignore missing/extra keys
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys}")
                    loaded = True
                    logger.info("Loaded model from model.safetensors")
                except Exception as e:
                    logger.error(f"Failed to load model.safetensors: {e}")
                    raise
        
        if not loaded:
            raise FileNotFoundError(f"No valid model file found in {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model moved to {self.device} and set to eval mode")
        
        # 5-class category names
        self.class_names = [
            'correct',                        # 0: correct
            'contradictory',                  # 1: contradictory
            'partially_correct_incomplete',   # 2: partially correct but incomplete
            'irrelevant',                     # 3: irrelevant
            'non_domain'                      # 4: non-domain
        ]
    
    def predict(
        self,
        question: str,
        reference_answer: str,
        student_answer: str
    ) -> Dict[str, float]:
        """
        Predict score and class for student answer
        
        Args:
            question: Question text
            reference_answer: Reference answer
            student_answer: Student answer
        
        Returns:
            Dictionary containing score and class
        """
        # Build input
        text_a = f"{question} {reference_answer}"
        text_b = student_answer
        
        # Tokenize
        encoded = self.tokenizer(
            text_a,
            text_b,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_type_ids = encoded.get('token_type_ids', torch.zeros_like(input_ids)).to(self.device)
        
        # Extract features (if used)
        features = None
        if self.use_features:
            features = self._extract_features(question, reference_answer, student_answer)
            features = torch.tensor([features], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                features=features
            )
        
        # Get results
        logits = outputs['logits']
        score = outputs['score'].item()
        
        # Get class
        predicted_class = torch.argmax(logits, dim=-1).item()
        class_name = self.class_names[predicted_class]
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        class_probs = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        
        return {
            'score': round(score, 2),
            'class': class_name,
            'class_probabilities': class_probs
        }
    
    def _extract_features(
        self,
        question: str,
        reference_answer: str,
        student_answer: str
    ) -> list:
        """Extract interpretable features"""
        # Keyword overlap rate
        ref_words = set(reference_answer.lower().split())
        student_words = set(student_answer.lower().split())
        keyword_overlap = len(ref_words & student_words) / max(len(ref_words), 1)
        
        # Answer length ratio
        length_ratio = len(student_answer) / max(len(reference_answer), 1)
        
        # Jaccard similarity
        all_words = ref_words | student_words
        if len(all_words) > 0:
            jaccard_similarity = len(ref_words & student_words) / len(all_words)
        else:
            jaccard_similarity = 0.0
        
        # Fluency
        fluency = min(len(student_answer.split()) / 10.0, 1.0)
        
        return [keyword_overlap, length_ratio, jaccard_similarity, fluency]


def main():
    parser = argparse.ArgumentParser(description='Use trained model for inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Model path')
    parser.add_argument('--question', type=str,
                       help='Question text')
    parser.add_argument('--reference_answer', type=str,
                       help='Reference answer')
    parser.add_argument('--student_answer', type=str,
                       help='Student answer')
    parser.add_argument('--use_features', action='store_true',
                       help='Whether to use interpretable features')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model: {args.model_path}")
    model = ScoringModel(
        model_path=args.model_path,
        use_features=args.use_features
    )
    
    # If input is provided, make prediction
    if args.question and args.reference_answer and args.student_answer:
        result = model.predict(
            args.question,
            args.reference_answer,
            args.student_answer
        )
        
        print("\nPrediction Results:")
        print(f"Score: {result['score']}/5.0")
        print(f"Class: {result['class']}")
        print(f"Class Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    else:
        print("Please provide --question, --reference_answer, --student_answer arguments for prediction")
        print("\nExample usage:")
        print("python inference.py --model_path ./outputs/final_model \\")
        print("  --question 'How did you separate the salt from the water?' \\")
        print("  --reference_answer 'The water was evaporated, leaving the salt.' \\")
        print("  --student_answer 'By letting it sit in a dish for a day.'")


if __name__ == "__main__":
    main()

