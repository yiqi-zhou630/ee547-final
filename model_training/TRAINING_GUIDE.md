# 模型训练指南

## 概述

本项目实现了一个基于跨编码器（Cross-Encoder）的多任务学习模型，用于自动评估学生简答题。模型同时输出数值分数（0-5）和分类标签（Correct/Partial/Incorrect）。

## 核心特性

### 1. 模型架构

- **基础模型**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **输入格式**: 三元组（题目文本、标准答案、学生答案）
- **编码方式**: 将"题目+标准答案"与"学生答案"拼接后共同输入Transformer
- **输出**:
  - 回归分数：0-5的连续值
  - 分类标签：Correct/Partial/Incorrect（3-way）

### 2. 多任务学习

模型使用两个任务头：
- **分类头**: 输出3-way分类（Correct/Partial/Incorrect）
- **回归头**: 输出0-5的分数

### 3. 损失函数

```
loss = α * CrossEntropy(分类) + β * MSE(回归)
```

其中：
- `α`: 分类损失权重（默认0.5）
- `β`: 回归损失权重（默认0.5）

### 4. 可解释特征（可选）

如果启用 `--use_features`，模型会融合以下特征：
1. **关键词重合率**: 标准答案和学生答案的词汇重叠度
2. **答案长度比例**: 学生答案与标准答案的长度比
3. **Jaccard相似度**: 基于词汇集合的相似度
4. **流畅度**: 基于答案长度的启发式指标

## 数据集

使用 **SciEntsBank** 数据集：
- **来源**: Hugging Face (`nkazi/SciEntsBank`)
- **大小**: 约10.8K条数据
- **分割**: 
  - 训练集: 4.97K
  - 测试集: test_ua (540), test_uq (733), test_ud (4.56K)

### 标签转换

原始数据集使用5-way标签：
- 0: correct
- 1: contradictory
- 2: partially_correct_incomplete
- 3: irrelevant
- 4: non_domain

转换为3-way标签：
- 0 (correct) → 0 (Correct)
- 1 (contradictory) → 2 (Incorrect)
- 2 (partially_correct_incomplete) → 1 (Partial)
- 3 (irrelevant) → 2 (Incorrect)
- 4 (non_domain) → 2 (Incorrect)

## 快速开始

### 1. 安装依赖

```bash
cd model_training
pip install -r requirements.txt
```

### 2. 训练模型

**方式一：使用Python脚本**

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

**方式二：使用Shell脚本**

```bash
./run_training.sh \
    --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --output_dir ./outputs \
    --batch_size 16 \
    --num_epochs 5 \
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

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name` | 预训练模型名称 | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `--output_dir` | 模型输出目录 | `./outputs` |
| `--batch_size` | 批次大小 | 16 |
| `--learning_rate` | 学习率 | 2e-5 |
| `--num_epochs` | 训练轮数 | 5 |
| `--max_length` | 最大序列长度 | 512 |
| `--alpha` | 分类损失权重 | 0.5 |
| `--beta` | 回归损失权重 | 0.5 |
| `--use_features` | 是否使用可解释特征 | False |
| `--warmup_steps` | 预热步数 | 100 |
| `--save_steps` | 保存检查点的步数 | 500 |
| `--eval_steps` | 评估步数 | 500 |

## 评估指标

训练过程中会计算以下指标：

### 分类指标
- **Accuracy**: 准确率
- **F1-macro**: 宏平均F1分数
- **F1-weighted**: 加权F1分数

### 回归指标
- **MSE**: 均方误差
- **RMSE**: 均方根误差

## 模型输出格式

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

## 训练流程

1. **数据加载**: 从Hugging Face加载SciEntsBank数据集
2. **数据预处理**: 
   - Tokenization
   - 标签转换（5-way → 3-way）
   - 特征提取（可选）
3. **模型训练**: 
   - 使用MultiTaskTrainer进行训练
   - 计算多任务损失
   - 定期评估和保存
4. **模型保存**: 保存最终模型和tokenizer

## 注意事项

1. **GPU推荐**: 建议使用GPU进行训练，CPU训练速度较慢
2. **内存要求**: 建议至少16GB内存
3. **首次运行**: 会自动下载数据集和预训练模型，需要网络连接
4. **损失权重**: 可以根据任务需求调整`alpha`和`beta`的值
5. **特征融合**: 启用`--use_features`可以提高可解释性，但可能略微增加训练时间

## 故障排除

### 1. 内存不足

- 减小`batch_size`
- 减小`max_length`
- 使用梯度累积

### 2. 训练速度慢

- 使用GPU
- 减小`max_length`
- 使用混合精度训练（已启用fp16）

### 3. 模型不收敛

- 调整学习率
- 增加训练轮数
- 调整损失权重（alpha和beta）

## 文件结构

```
model_training/
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── preprocess_data.py    # 数据预处理脚本
├── config.yaml           # 配置文件
├── requirements.txt      # 依赖包
├── README.md            # 文档
├── TRAINING_GUIDE.md    # 本指南
├── run_training.sh       # 快速训练脚本
└── __init__.py          # 包初始化文件
```

## 引用

如果使用SciEntsBank数据集，请引用：

```bibtex
@inproceedings{dzikovska2013semeval,
  title = {{S}em{E}val-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge},
  author = {Dzikovska, Myroslava and Nielsen, Rodney and Brew, Chris and Leacock, Claudia and Giampiccolo, Danilo and Bentivogli, Luisa and Clark, Peter and Dagan, Ido and Dang, Hoa Trang},
  year = 2013,
  booktitle = {Second Joint Conference on Lexical and Computational Semantics ({SEM}), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation ({S}em{E}val 2013)},
  pages = {263--274},
}
```

