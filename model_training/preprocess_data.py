"""
数据预处理脚本
用于预处理SciEntsBank数据集，转换为训练所需的格式
"""

import argparse
from datasets import load_dataset, DatasetDict, ClassLabel
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_3way(dataset: DatasetDict) -> DatasetDict:
    """将5-way标签转换为3-way标签"""
    def convert_label(example):
        label_5way = example['label']
        # 映射规则：
        # 0 (correct) -> 0 (Correct)
        # 1 (contradictory) -> 1 (Contradictory/Partial)
        # 2 (partially_correct_incomplete) -> 1 (Partial)
        # 3 (irrelevant) -> 2 (Incorrect)
        # 4 (non_domain) -> 2 (Incorrect)
        
        if label_5way == 0:
            new_label = 0  # Correct
        elif label_5way == 1:
            new_label = 1  # Contradictory -> Partial
        elif label_5way == 2:
            new_label = 1  # Partial
        else:
            new_label = 2  # Incorrect
        
        example['label_3way'] = new_label
        return example
    
    dataset = dataset.map(convert_label)
    
    # 更新label列
    dataset = dataset.cast_column('label_3way', ClassLabel(names=['Correct', 'Partial', 'Incorrect']))
    
    return dataset


def convert_to_2way(dataset: DatasetDict) -> DatasetDict:
    """将5-way标签转换为2-way标签"""
    def convert_label(example):
        label_5way = example['label']
        # 只有correct是1，其他都是0
        new_label = 1 if label_5way == 0 else 0
        example['label_2way'] = new_label
        return example
    
    dataset = dataset.map(convert_label)
    dataset = dataset.cast_column('label_2way', ClassLabel(names=['Incorrect', 'Correct']))
    
    return dataset


def print_label_distribution(dataset: DatasetDict, label_col: str = 'label'):
    """打印标签分布"""
    for split_name in dataset:
        print(f"\n{split_name}:")
        num_examples = 0
        label_names = dataset[split_name].features[label_col].names
        
        for label_name in label_names:
            label_id = dataset[split_name].features[label_col].str2int(label_name)
            count = dataset[split_name][label_col].count(label_id)
            print(f"  {label_name}: {count}")
            num_examples += count
        
        print(f"  total: {num_examples}")


def main():
    parser = argparse.ArgumentParser(description='预处理SciEntsBank数据集')
    parser.add_argument('--dataset_name', type=str, default='nkazi/SciEntsBank',
                       help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='输出目录')
    parser.add_argument('--label_scheme', type=str, choices=['5way', '3way', '2way'],
                       default='3way', help='标签方案')
    parser.add_argument('--save_local', action='store_true',
                       help='是否保存到本地')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    logger.info(f"正在加载数据集: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    # 打印原始5-way标签分布
    logger.info("原始5-way标签分布:")
    print_label_distribution(dataset, 'label')
    
    # 转换标签
    if args.label_scheme == '3way':
        dataset = convert_to_3way(dataset)
        logger.info("\n转换后3-way标签分布:")
        print_label_distribution(dataset, 'label_3way')
    elif args.label_scheme == '2way':
        dataset = convert_to_2way(dataset)
        logger.info("\n转换后2-way标签分布:")
        print_label_distribution(dataset, 'label_2way')
    
    # 保存到本地
    if args.save_local:
        save_path = Path(args.output_dir) / f"SciEntsBank_{args.label_scheme}"
        dataset.save_to_disk(str(save_path))
        logger.info(f"\n数据集已保存到: {save_path}")
        
        # 保存统计信息
        stats = {}
        for split_name in dataset:
            stats[split_name] = {
                'size': len(dataset[split_name]),
                'label_distribution': {}
            }
            label_col = f'label_{args.label_scheme}' if args.label_scheme != '5way' else 'label'
            label_names = dataset[split_name].features[label_col].names
            for label_name in label_names:
                label_id = dataset[split_name].features[label_col].str2int(label_name)
                count = dataset[split_name][label_col].count(label_id)
                stats[split_name]['label_distribution'][label_name] = count
        
        with open(Path(args.output_dir) / f"stats_{args.label_scheme}.json", "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"统计信息已保存到: {Path(args.output_dir) / f'stats_{args.label_scheme}.json'}")


if __name__ == "__main__":
    main()

