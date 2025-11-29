"""
Data Preprocessing Script
Preprocess SciEntsBank dataset and convert to format required for training

Data Splitting Strategy (consistent with train.py):
- Exclude test_ud (unseen domain) subset
- Merge test_ua and test_uq, shuffle and split into:
  * Training portion: for fine-tuning and testing the model
  * Hold-out portion: for subsequent system testing phase
"""

import argparse
from datasets import load_dataset, DatasetDict, ClassLabel, concatenate_datasets
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_3way(dataset: DatasetDict) -> DatasetDict:
    """Convert 5-way labels to 3-way labels"""
    def convert_label(example):
        label_5way = example['label']
        # Mapping rules:
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
    
    # Update label column
    dataset = dataset.cast_column('label_3way', ClassLabel(names=['Correct', 'Partial', 'Incorrect']))
    
    return dataset


def convert_to_2way(dataset: DatasetDict) -> DatasetDict:
    """Convert 5-way labels to 2-way labels"""
    def convert_label(example):
        label_5way = example['label']
        # Only correct is 1, others are 0
        new_label = 1 if label_5way == 0 else 0
        example['label_2way'] = new_label
        return example
    
    dataset = dataset.map(convert_label)
    dataset = dataset.cast_column('label_2way', ClassLabel(names=['Incorrect', 'Correct']))
    
    return dataset


def print_label_distribution(dataset: DatasetDict, label_col: str = 'label'):
    """Print label distribution"""
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


def prepare_training_data(
    dataset: DatasetDict,
    holdout_ratio: float = 0.3,
    random_seed: int = 42
) -> DatasetDict:
    """
    Prepare training data (consistent with logic in train.py)
    
    Args:
        dataset: Original dataset
        holdout_ratio: Ratio of hold-out portion
        random_seed: Random seed
    
    Returns:
        Processed dataset dictionary
    """
    # Exclude test_ud (unseen domain) subset
    logger.info("Excluding test_ud subset (unseen domain)")
    
    # Merge test_ua and test_uq
    logger.info("Merging test_ua and test_uq subsets")
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
    
    # Create validation set (use 25% of training portion or at most 500 samples)
    eval_size = min(500, len(test_train) // 4)
    eval_dataset = test_train.select(range(eval_size))
    
    # Create new dataset dictionary
    new_dataset = DatasetDict({
        'train': train_final,
        'eval': eval_dataset,
        'holdout': test_holdout
    })
    
    return new_dataset


def main():
    parser = argparse.ArgumentParser(description='Preprocess SciEntsBank dataset')
    parser.add_argument('--dataset_name', type=str, default='nkazi/SciEntsBank',
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory')
    parser.add_argument('--label_scheme', type=str, choices=['5way', '3way', '2way'],
                       default='5way', help='Label scheme (default 5way, consistent with training)')
    parser.add_argument('--holdout_ratio', type=float, default=0.3,
                       help='Ratio of hold-out portion (default 0.3)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed (default 42)')
    parser.add_argument('--save_local', action='store_true',
                       help='Whether to save to local')
    parser.add_argument('--prepare_training', action='store_true',
                       help='Whether to prepare data according to training requirements (exclude test_ud, merge test_ua and test_uq)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    # Print original 5-way label distribution
    logger.info("Original 5-way label distribution:")
    print_label_distribution(dataset, 'label')
    
    # If prepare_training is specified, prepare data according to training requirements
    if args.prepare_training:
        logger.info("\nPreparing data according to training requirements...")
        dataset = prepare_training_data(dataset, args.holdout_ratio, args.random_seed)
        logger.info("\nProcessed data distribution:")
        print_label_distribution(dataset, 'label')
    
    # Convert labels (if needed)
    if args.label_scheme == '3way':
        dataset = convert_to_3way(dataset)
        logger.info("\nConverted 3-way label distribution:")
        print_label_distribution(dataset, 'label_3way')
    elif args.label_scheme == '2way':
        dataset = convert_to_2way(dataset)
        logger.info("\nConverted 2-way label distribution:")
        print_label_distribution(dataset, 'label_2way')
    elif args.label_scheme == '5way':
        logger.info("\nUsing 5-way labels (no conversion)")
    
    # Save to local
    if args.save_local:
        scheme_suffix = f"_{args.label_scheme}" if args.label_scheme != '5way' else ""
        training_suffix = "_training" if args.prepare_training else ""
        save_path = Path(args.output_dir) / f"SciEntsBank{scheme_suffix}{training_suffix}"
        dataset.save_to_disk(str(save_path))
        logger.info(f"\nDataset saved to: {save_path}")
        
        # Save statistics
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
        
        stats_file = Path(args.output_dir) / f"stats_{args.label_scheme}{training_suffix}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()

