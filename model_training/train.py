"""
模型训练脚本
使用 SciEntsBank 数据集训练跨编码器模型进行自动评分
支持多任务学习：回归分数（0-5）和分类标签（Correct/Partial/Incorrect）
"""

import os
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SciEntsBankDataset(Dataset):
    """SciEntsBank数据集包装类"""
    
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
        
        # 构建输入：将"题目+标准答案"与"学生答案"拼接
        question = item['question']
        reference_answer = item['reference_answer']
        student_answer = item['student_answer']
        
        # 拼接格式：[CLS] question [SEP] reference_answer [SEP] student_answer [SEP]
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
        
        # 获取标签
        label_5way = item['label']
        
        # 转换为3-way标签：Correct/Partial/Incorrect
        label_3way = self._convert_to_3way(label_5way)
        
        # 转换为数值分数（0-5）
        score = self._label_to_score(label_5way)
        
        result = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0),
            'label_class': torch.tensor(label_3way, dtype=torch.long),
            'label_score': torch.tensor(score, dtype=torch.float32)
        }
        
        # 可选：添加可解释特征
        if self.use_features:
            features = self._extract_features(question, reference_answer, student_answer)
            result['features'] = torch.tensor(features, dtype=torch.float32)
        
        return result
    
    def _convert_to_3way(self, label_5way: int) -> int:
        """将5-way标签转换为3-way标签"""
        # 5-way: 0=correct, 1=contradictory, 2=partially_correct_incomplete, 3=irrelevant, 4=non_domain
        # 3-way: 0=Correct, 1=Partial, 2=Incorrect
        # 根据项目需求：Correct, Partial, Incorrect
        mapping = {
            0: 0,  # correct -> Correct
            1: 2,  # contradictory -> Incorrect
            2: 1,  # partially_correct_incomplete -> Partial
            3: 2,  # irrelevant -> Incorrect
            4: 2   # non_domain -> Incorrect
        }
        return mapping.get(label_5way, 2)
    
    def _label_to_score(self, label_5way: int) -> float:
        """将标签转换为0-5的分数"""
        # 根据标签映射到分数
        score_mapping = {
            0: 5.0,  # correct -> 5分
            1: 1.0,  # contradictory -> 1分
            2: 3.0,  # partially_correct_incomplete -> 3分
            3: 0.5,  # irrelevant -> 0.5分
            4: 0.0   # non_domain -> 0分
        }
        return score_mapping.get(label_5way, 0.0)
    
    def _extract_features(
        self,
        question: str,
        reference_answer: str,
        student_answer: str
    ) -> List[float]:
        """提取可解释特征"""
        # 关键词重合率
        ref_words = set(reference_answer.lower().split())
        student_words = set(student_answer.lower().split())
        keyword_overlap = len(ref_words & student_words) / max(len(ref_words), 1)
        
        # 答案长度比例
        length_ratio = len(student_answer) / max(len(reference_answer), 1)
        
        # 句子相似度（简单的词汇重叠）
        all_words = ref_words | student_words
        if len(all_words) > 0:
            jaccard_similarity = len(ref_words & student_words) / len(all_words)
        else:
            jaccard_similarity = 0.0
        
        # 流畅度（简单的启发式：基于标点符号和长度）
        fluency = min(len(student_answer.split()) / 10.0, 1.0)  # 归一化到0-1
        
        return [keyword_overlap, length_ratio, jaccard_similarity, fluency]


class MultiTaskCrossEncoder(nn.Module):
    """多任务跨编码器模型"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        num_classes: int = 3,
        use_features: bool = False,
        feature_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_features = use_features
        
        # 加载预训练的cross-encoder模型
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # 如果使用特征融合，增加特征维度
        classifier_input_size = hidden_size
        if use_features:
            classifier_input_size += feature_dim
        
        # 分类头（3-way: Correct/Partial/Incorrect）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 回归头（0-5分数）
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出0-1，然后乘以5得到0-5
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 编码
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS]向量
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 特征融合
        if self.use_features and features is not None:
            cls_embedding = torch.cat([cls_embedding, features], dim=1)
        
        # 分类输出
        logits = self.classifier(cls_embedding)
        
        # 回归输出（0-5）
        score = self.regressor(cls_embedding) * 5.0
        
        return {
            'logits': logits,
            'score': score.squeeze(-1)
        }


class MultiTaskTrainer(Trainer):
    """自定义Trainer，支持多任务损失"""
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # 分类损失权重
        self.beta = beta    # 回归损失权重
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取标签
        labels_class = inputs.pop('label_class')
        labels_score = inputs.pop('label_score')
        
        # 获取特征（如果存在）
        features = inputs.pop('features', None)
        
        # 前向传播
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids'),
            features=features
        )
        
        # 计算损失
        loss_cls = self.criterion_cls(outputs['logits'], labels_class)
        loss_reg = self.criterion_reg(outputs['score'], labels_score)
        
        # 加权组合
        loss = self.alpha * loss_cls + self.beta * loss_reg
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """重写prediction_step以支持多任务输出"""
        has_labels = 'label_class' in inputs
        inputs = self._prepare_inputs(inputs)
        
        # 获取标签
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
        
        # 准备预测结果
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
    use_3way: bool = True
) -> DatasetDict:
    """加载并预处理SciEntsBank数据集"""
    logger.info(f"正在加载数据集: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    if use_3way:
        # 转换为3-way标签
        from datasets import ClassLabel
        
        def convert_to_3way(example):
            label_5way = example['label']
            # 映射：correct->0, contradictory->1, 其他->2
            if label_5way == 0:  # correct
                example['label'] = 0
            elif label_5way == 1:  # contradictory
                example['label'] = 1
            else:  # partially_correct_incomplete, irrelevant, non_domain
                example['label'] = 2
            return example
        
        dataset = dataset.map(convert_to_3way)
        # 注意：这里保持原始标签值，实际转换在Dataset类中进行
        # 数据集中的label仍然是5-way，我们在Dataset中转换
    
    logger.info("数据集加载完成")
    logger.info(f"训练集大小: {len(dataset['train'])}")
    logger.info(f"测试集大小: {len(dataset['test_ua'])}")
    
    return dataset


def compute_metrics(eval_pred):
    """计算评估指标"""
    # Trainer的compute_metrics接收EvalPrediction对象
    # 包含predictions和label_ids属性
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # predictions是我们自定义的字典格式 {'logits': ..., 'score': ...}
    if isinstance(predictions, dict):
        logits = predictions.get('logits')
        scores = predictions.get('score')
    elif isinstance(predictions, tuple):
        logits, scores = predictions
    else:
        # 如果只是数组，假设是logits
        logits = predictions
        scores = None
    
    # labels也是字典格式 {'label_class': ..., 'label_score': ...}
    if isinstance(labels, dict):
        true_classes = labels.get('label_class')
        true_scores = labels.get('label_score')
    elif isinstance(labels, tuple):
        true_classes, true_scores = labels
    else:
        true_classes = labels
        true_scores = None
    
    # 转换为numpy数组
    if torch.is_tensor(logits):
        logits = logits.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    if torch.is_tensor(true_classes):
        true_classes = true_classes.cpu().numpy()
    if torch.is_tensor(true_scores):
        true_scores = true_scores.cpu().numpy()
    
    # 确保是numpy数组
    logits = np.array(logits)
    if scores is not None:
        scores = np.array(scores)
    true_classes = np.array(true_classes)
    if true_scores is not None:
        true_scores = np.array(true_scores)
    
    # 获取预测类别
    if len(logits.shape) > 1:
        pred_classes = np.argmax(logits, axis=-1)
    else:
        pred_classes = logits
    
    metrics = {
        'accuracy': float(accuracy_score(true_classes, pred_classes)),
        'f1_macro': float(f1_score(true_classes, pred_classes, average='macro')),
        'f1_weighted': float(f1_score(true_classes, pred_classes, average='weighted'))
    }
    
    # 回归指标
    if scores is not None and true_scores is not None:
        metrics['mse'] = float(mean_squared_error(true_scores, scores))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练多任务跨编码器模型')
    parser.add_argument('--model_name', type=str, 
                       default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                       help='预训练模型名称')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='模型输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='分类损失权重')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='回归损失权重')
    parser.add_argument('--use_features', action='store_true',
                       help='是否使用可解释特征')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='预热步数')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='保存检查点的步数')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='评估步数')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='日志记录步数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    dataset = load_and_preprocess_data(use_3way=True)
    
    # 加载tokenizer和模型
    logger.info(f"正在加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultiTaskCrossEncoder(
        model_name=args.model_name,
        num_classes=3,
        use_features=args.use_features
    )
    
    # 创建数据集
    train_dataset = SciEntsBankDataset(
        dataset['train'],
        tokenizer,
        max_length=args.max_length,
        use_features=args.use_features
    )
    
    eval_dataset = SciEntsBankDataset(
        dataset['test_ua'],
        tokenizer,
        max_length=args.max_length,
        use_features=args.use_features
    )
    
    # 训练参数
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
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none"
    )
    
    # 创建Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        alpha=args.alpha,
        beta=args.beta,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"模型已保存到: {final_model_path}")
    
    # 评估
    logger.info("开始评估...")
    eval_results = trainer.evaluate()
    logger.info(f"评估结果: {eval_results}")
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()

