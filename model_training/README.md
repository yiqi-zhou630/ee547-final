# 模型训练模块

本模块实现了基于跨编码器（Cross-Encoder）的多任务学习模型，用于自动评估学生简答题。

## 模型架构

- **基础模型**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **架构**: 跨编码器（Cross-Encoder），将"题目+标准答案"和"学生答案"拼接后共同编码
- **多任务输出**:
  - 回归任务：输出0-5的数值分数
  - 分类任务：输出3-way标签（Correct/Partial/Incorrect）

## 数据集

使用 [SciEntsBank](https://huggingface.co/datasets/nkazi/SciEntsBank) 数据集进行训练。

### 标签方案

- **5-way**: correct, contradictory, partially_correct_incomplete, irrelevant, non_domain
- **3-way**: Correct, Partial, Incorrect（本项目默认使用）
- **2-way**: Correct, Incorrect

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据预处理（可选）

```bash
python preprocess_data.py \
  --dataset_name nkazi/SciEntsBank \
  --label_scheme 3way \
  --save_local
```

### 2. 训练模型

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

### 3. 模型推理

```bash
python inference.py \
  --model_path ./outputs/final_model \
  --question "How did you separate the salt from the water?" \
  --reference_answer "The water was evaporated, leaving the salt." \
  --student_answer "By letting it sit in a dish for a day."
```

## 训练参数说明

- `--model_name`: 预训练模型名称
- `--output_dir`: 模型输出目录
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数
- `--max_length`: 最大序列长度
- `--alpha`: 分类损失权重（默认0.5）
- `--beta`: 回归损失权重（默认0.5）
- `--use_features`: 是否使用可解释特征（关键词重合率、句子相似度等）

## 损失函数

模型使用加权组合的损失函数：

```
loss = α * CrossEntropy(分类) + β * MSE(回归)
```

其中：
- `α`: 分类损失权重
- `β`: 回归损失权重

## 可解释特征（可选）

如果启用 `--use_features`，模型会融合以下特征：

1. **关键词重合率**: 标准答案和学生答案的词汇重叠度
2. **答案长度比例**: 学生答案与标准答案的长度比
3. **Jaccard相似度**: 基于词汇集合的相似度
4. **流畅度**: 基于答案长度的启发式流畅度指标

## 输出格式

模型输出包含：

```python
{
    'score': 3.5,  # 0-5的数值分数
    'class': 'Partial',  # Correct/Partial/Incorrect
    'class_probabilities': {
        'Correct': 0.1,
        'Partial': 0.7,
        'Incorrect': 0.2
    }
}
```

## 评估指标

训练过程中会计算以下指标：

- **分类指标**: Accuracy, F1-macro, F1-weighted
- **回归指标**: MSE, RMSE

## 文件结构

```
model_training/
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── preprocess_data.py    # 数据预处理脚本
├── config.yaml           # 配置文件
├── requirements.txt       # 依赖包
└── README.md            # 本文档
```

## 注意事项

1. 首次运行会自动下载SciEntsBank数据集和预训练模型
2. 建议使用GPU进行训练，CPU训练速度较慢
3. 可以根据实际情况调整损失函数权重（alpha和beta）
4. 可解释特征可以提高模型的可解释性，但可能会略微增加训练时间

