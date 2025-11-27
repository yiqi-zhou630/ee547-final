"""
模型推理脚本
用于加载训练好的模型进行预测
"""

import argparse
import os
import torch
from transformers import AutoTokenizer
from train import MultiTaskCrossEncoder
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringModel:
    """评分模型包装类"""
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        use_features: bool = False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_features = use_features
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        # 首先尝试从保存的配置加载基础模型名称
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
            num_classes=3,
            use_features=use_features
        )
        
        # 加载权重
        model_file = f"{model_path}/pytorch_model.bin"
        if not os.path.exists(model_file):
            model_file = f"{model_path}/model.safetensors"
            # 如果使用safetensors，需要使用safetensors库
            if os.path.exists(model_file):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                    self.model.load_state_dict(state_dict, strict=False)
                except ImportError:
                    logger.warning("safetensors not installed, trying alternative loading method")
                    # 尝试直接加载整个模型
                    self.model = torch.load(f"{model_path}/model.pt", map_location=self.device)
        else:
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 类别名称
        self.class_names = ['Correct', 'Partial', 'Incorrect']
    
    def predict(
        self,
        question: str,
        reference_answer: str,
        student_answer: str
    ) -> Dict[str, float]:
        """
        预测学生答案的分数和类别
        
        Args:
            question: 题目文本
            reference_answer: 标准答案
            student_answer: 学生答案
        
        Returns:
            包含分数和类别的字典
        """
        # 构建输入
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
        
        # 移动到设备
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_type_ids = encoded.get('token_type_ids', torch.zeros_like(input_ids)).to(self.device)
        
        # 提取特征（如果使用）
        features = None
        if self.use_features:
            features = self._extract_features(question, reference_answer, student_answer)
            features = torch.tensor([features], dtype=torch.float32).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                features=features
            )
        
        # 获取结果
        logits = outputs['logits']
        score = outputs['score'].item()
        
        # 获取类别
        predicted_class = torch.argmax(logits, dim=-1).item()
        class_name = self.class_names[predicted_class]
        
        # 获取概率
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
        """提取可解释特征"""
        # 关键词重合率
        ref_words = set(reference_answer.lower().split())
        student_words = set(student_answer.lower().split())
        keyword_overlap = len(ref_words & student_words) / max(len(ref_words), 1)
        
        # 答案长度比例
        length_ratio = len(student_answer) / max(len(reference_answer), 1)
        
        # Jaccard相似度
        all_words = ref_words | student_words
        if len(all_words) > 0:
            jaccard_similarity = len(ref_words & student_words) / len(all_words)
        else:
            jaccard_similarity = 0.0
        
        # 流畅度
        fluency = min(len(student_answer.split()) / 10.0, 1.0)
        
        return [keyword_overlap, length_ratio, jaccard_similarity, fluency]


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--question', type=str,
                       help='题目文本')
    parser.add_argument('--reference_answer', type=str,
                       help='标准答案')
    parser.add_argument('--student_answer', type=str,
                       help='学生答案')
    parser.add_argument('--use_features', action='store_true',
                       help='是否使用可解释特征')
    
    args = parser.parse_args()
    
    # 加载模型
    logger.info(f"正在加载模型: {args.model_path}")
    model = ScoringModel(
        model_path=args.model_path,
        use_features=args.use_features
    )
    
    # 如果提供了输入，进行预测
    if args.question and args.reference_answer and args.student_answer:
        result = model.predict(
            args.question,
            args.reference_answer,
            args.student_answer
        )
        
        print("\n预测结果:")
        print(f"分数: {result['score']}/5.0")
        print(f"类别: {result['class']}")
        print(f"类别概率:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    else:
        print("请提供 --question, --reference_answer, --student_answer 参数进行预测")
        print("\n示例用法:")
        print("python inference.py --model_path ./outputs/final_model \\")
        print("  --question 'How did you separate the salt from the water?' \\")
        print("  --reference_answer 'The water was evaporated, leaving the salt.' \\")
        print("  --student_answer 'By letting it sit in a dish for a day.'")


if __name__ == "__main__":
    main()

