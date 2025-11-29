# Model Training Guide

This module implements a multi-task learning model based on Cross-Encoder for automatic assessment of student short-answer questions. The model outputs both a numerical score (0-5) and a 5-class label (correct, contradictory, partially correct but incomplete, irrelevant, non-domain).

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Parameters](#training-parameters)
- [Model Output](#model-output)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd model_training
pip install -r requirements.txt
```

### Step 2: Train the Model

**Method 1: Use Default Parameters (Simplest)**

```bash
python train.py
```

**Method 2: Custom Parameters**

```bash
python train.py \
    --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --output_dir ./outputs \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --alpha 0.5 \
    --beta 0.5 \
    --use_features
```

**Method 3: Use Shell Script**

```bash
chmod +x run_training.sh
./run_training.sh
```

### Step 3: Model Inference

After training, use the trained model for inference:

```bash
python inference.py \
    --model_path ./outputs/final_model \
    --question "How did you separate the salt from the water?" \
    --reference_answer "The water was evaporated, leaving the salt." \
    --student_answer "By letting it sit in a dish for a day."
```

### Step 4: View Results

After training, the model and results are saved in:
- **Model files**: `./outputs/final_model/`
- **Evaluation results**: `./outputs/eval_results.json`
- **Hold-out dataset**: `./outputs/holdout_dataset/` (for subsequent system testing)

---

## 🏗️ Model Architecture

### Core Features

- **Base Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Architecture**: Cross-Encoder that jointly encodes "question + reference answer" and "student answer" after concatenation
- **Input Format**: Triple (question text, reference answer, student answer)
- **Encoding Format**: `[CLS] question [SEP] reference_answer [SEP] student_answer [SEP]`

### Multi-Task Output

The model uses two task heads:

1. **Classification Head**: Outputs 5-way classification
   - 0: correct
   - 1: contradictory
   - 2: partially_correct_incomplete
   - 3: irrelevant
   - 4: non_domain

2. **Regression Head**: Outputs a numerical score from 0-5

### Loss Function

```
loss = α * CrossEntropy(classification) + β * MSE(regression)
```

Where:
- `α`: Classification loss weight (default 0.5)
- `β`: Regression loss weight (default 0.5)

### Interpretable Features (Optional)

If `--use_features` is enabled, the model will incorporate the following features:

1. **Keyword Overlap Rate**: Vocabulary overlap between reference answer and student answer
2. **Answer Length Ratio**: Ratio of student answer length to reference answer length
3. **Jaccard Similarity**: Similarity based on vocabulary sets
4. **Fluency**: Heuristic metric based on answer length

---

## 📊 Dataset

### Dataset Information

- **Source**: [SciEntsBank](https://huggingface.co/datasets/nkazi/SciEntsBank) (Hugging Face)
- **Size**: Approximately 10.8K samples
- **Original Split**: 
  - Training set: 4.97K
  - Test sets: test_ua (540), test_uq (733), test_ud (4.56K)

### Data Splitting Strategy

To simplify training and focus on familiar scientific domain questions, we adopt the following strategy:

1. **Exclude test_ud (unseen domain) subset**: Not used for training or testing
2. **Merge test_ua and test_uq**: Combine the two test subsets
3. **Shuffle data**: Shuffle the merged data using random seed 42
4. **Data Split**:
   - **Training portion** (70%): Merged with the original training set for fine-tuning and testing the model
   - **Hold-out portion** (30%): Used for subsequent system testing phase, simulating real student submission data

### Label Scheme

This project directly uses 5-way labels without label conversion:

| Label ID | Label Name | Description |
|----------|------------|-------------|
| 0 | correct | Student answer is completely correct |
| 1 | contradictory | Student answer contradicts the reference answer |
| 2 | partially_correct_incomplete | Student answer is partially correct but incomplete |
| 3 | irrelevant | Student answer is irrelevant to the question |
| 4 | non_domain | Student answer is outside the domain scope |

### Data Preprocessing (Optional)

Data preprocessing is **automatically completed** during training, no manual execution required.

If you need to preprocess data separately (e.g., pre-download data, view data distribution), you can use:

```bash
# View data distribution
python preprocess_data.py --prepare_training

# Preprocess and save to local
python preprocess_data.py \
    --prepare_training \
    --label_scheme 5way \
    --save_local
```

---

## ⚙️ Training Parameters

### Main Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--model_name` | Pretrained model name | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `--output_dir` | Model output directory | `./outputs` |
| `--batch_size` | Batch size | 16 |
| `--learning_rate` | Learning rate | 2e-5 |
| `--num_epochs` | Number of training epochs | 5 |
| `--max_length` | Maximum sequence length | 512 |
| `--alpha` | Classification loss weight | 0.5 |
| `--beta` | Regression loss weight | 0.5 |
| `--use_features` | Whether to use interpretable features | False |

### Advanced Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--warmup_steps` | Number of warmup steps | 100 |
| `--save_steps` | Steps to save checkpoint | 500 |
| `--eval_steps` | Steps to evaluate | 500 |
| `--logging_steps` | Steps to log | 100 |

### Parameter Tuning Suggestions

- **Out of Memory**: Reduce `batch_size` or `max_length`
- **Slow Training**: Use GPU, or reduce `max_length`
- **Model Not Converging**: Adjust `learning_rate`, increase `num_epochs`, or adjust `alpha` and `beta`

---

## 📤 Model Output

### Output Format

Model inference output contains the following information:

```python
{
    'score': 3.5,  # Numerical score from 0-5
    'class': 'partially_correct_incomplete',  # One of the 5-class labels
    'class_probabilities': {
        'correct': 0.1,
        'contradictory': 0.05,
        'partially_correct_incomplete': 0.7,
        'irrelevant': 0.1,
        'non_domain': 0.05
    }
}
```

### Evaluation Metrics

The following metrics are calculated during training:

**Classification Metrics**:
- **Accuracy**: Accuracy rate
- **F1-macro**: Macro-averaged F1 score
- **F1-weighted**: Weighted F1 score

**Regression Metrics**:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

---

## 🔧 Troubleshooting

### 1. Out of Memory

**Solutions**:
- Reduce `batch_size` (e.g., from 16 to 8)
- Reduce `max_length` (e.g., from 512 to 256)
- Use gradient accumulation

### 2. Slow Training Speed

**Solutions**:
- Use GPU (highly recommended)
- Reduce `max_length`
- Mixed precision training is automatically enabled (fp16)

### 3. Model Not Converging

**Solutions**:
- Adjust learning rate (try `1e-5` or `3e-5`)
- Increase number of training epochs (`num_epochs`)
- Adjust loss weights (`alpha` and `beta`)

### 4. TensorFlow Import Error

**Error Message**: `ImportError: dlopen(...libtensorflow_cc.2.dylib...)`

**Cause**: The transformers library may attempt to import tensorflow in some cases, but tensorflow installation has issues

**Solutions**:
- **Solution 1 (Recommended)**: Uninstall tensorflow (not needed for this project)
  ```bash
  pip uninstall tensorflow tensorflow-macos tensorflow-metal
  ```
- **Solution 2**: The code automatically sets environment variables to avoid importing tensorflow. If problems persist, you can manually set:
  ```bash
  export TRANSFORMERS_NO_TF=1
  python train.py
  ```

### 5. First Run Issues

**Common Issues**:
- **Network Connection**: First run will automatically download dataset and pretrained model, requires stable network connection
- **Disk Space**: Ensure sufficient disk space (at least 2GB)
- **Dependency Installation**: Ensure all dependencies are correctly installed

### 6. Disk Space Insufficient

**Error Message**: `RuntimeError: [enforce fail at inline_container.cc:769] . PytorchStreamWriter failed writing file`

**Cause**: Insufficient disk space when saving checkpoints

**Solutions**:
- **Solution 1**: Free up disk space (recommended)
  ```bash
  # Check disk usage
  df -h .
  
  # Remove old checkpoints
  rm -rf ./outputs/checkpoint-*
  ```
- **Solution 2**: Increase `--save_steps` to reduce checkpoint frequency
  ```bash
  python train.py --save_steps 2000
  ```
- **Solution 3**: Modify `save_total_limit` in training arguments to keep fewer checkpoints

---

## 📁 File Structure

```
model_training/
├── train.py              # Training script (main file)
├── inference.py          # Inference script
├── preprocess_data.py    # Data preprocessing script (optional)
├── config.yaml           # Configuration file
├── requirements.txt      # Dependency package list
├── run_training.sh       # Quick training script
├── model_train.md        # Chinese documentation
├── README_model_training.md  # This document (English)
└── __init__.py          # Package initialization file
```

---

## 📝 Training Workflow

1. **Data Loading**: Load SciEntsBank dataset from Hugging Face
2. **Data Preprocessing**: 
   - Exclude test_ud subset
   - Merge test_ua and test_uq
   - Shuffle data and split into training portion and hold-out portion
   - Tokenization
   - Feature extraction (optional)
3. **Model Training**: 
   - Train using MultiTaskTrainer
   - Calculate multi-task loss (5-class classification + regression)
   - Periodic evaluation and saving
4. **Model Saving**: Save final model, tokenizer, and hold-out dataset information

---

## ⚠️ Important Notes

1. **GPU Recommended**: GPU is recommended for training, CPU training is slow
2. **Memory Requirements**: At least 16GB RAM recommended
3. **First Run**: Will automatically download dataset and pretrained model, requires network connection
4. **Loss Weights**: Can adjust `alpha` and `beta` values according to task requirements
5. **Feature Fusion**: Enabling `--use_features` can improve interpretability but may slightly increase training time
6. **Hold-out Dataset**: After training, the hold-out dataset will be saved in `./outputs/holdout_dataset/` for subsequent system testing

---

## 📚 Citation

If using the SciEntsBank dataset, please cite:

```bibtex
@inproceedings{dzikovska2013semeval,
  title = {{S}em{E}val-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge},
  author = {Dzikovska, Myroslava and Nielsen, Rodney and Brew, Chris and Leacock, Claudia and Giampiccolo, Danilo and Bentivogli, Luisa and Clark, Peter and Dagan, Ido and Dang, Hoa Trang},
  year = 2013,
  booktitle = {Second Joint Conference on Lexical and Computational Semantics ({SEM}), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation ({S}em{E}val 2013)},
  pages = {263--274},
}
```

