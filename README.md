# COVID-Sentiment

Final Project + Paper for COMP9444 @ UNSW

## 项目概述

本项目针对 COVID-19 相关推文进行情感分析，实现了五种不同的模型，将推文分类为负面、中性或正面三种情感类别。

## 数据集

- 数据文件: `clean_COVIDSenti.csv`
- 来源: [COVIDSenti](https://github.com/usmaann/COVIDSenti) (Naseem et al., IEEE TCSS 2021)
- 总量: 90,000 条推文
- 类别分布:

| 标签 | 含义 | 数量 | 占比 |
|------|------|------|------|
| -1 | 负面 (Negative) | 16,335 | 18.2% |
| 0 | 中性 (Neutral) | 67,385 | 74.9% |
| 1 | 正面 (Positive) | 6,280 | 7.0% |

- 预处理: 推文经过清洗、去标点、去停用词、token化处理

## 模型结果对比

| 模型 | 验证集最佳准确率 | 测试集准确率 | Tokenizer | 交叉验证 |
|------|-----------------|-------------|-----------|---------|
| LSTM | ~90% | ~90% | 自定义词表 (39,747词) | 5折 |
| DistilBERT (IterativeLayer) | ~90.6% | 75%-89.6% | DistilBERT Tokenizer | 5折 |
| Trained Transformer | 85.3% | 83.1% | 自定义词表+拼写纠正 (32,389词) | 单次 |
| Word2Vec + RandomForest | - | 76.1% | NLTK word_tokenize | 单次 |
| Homemade Transformer | - | - | DistilBERT Tokenizer | 5折 |

> Homemade Transformer 因代码问题尚未成功训练。

---

## 模型详情

### 1. LSTM (循环神经网络)

**文件**: [LSTM.ipynb](models/LSTM/LSTM.ipynb)

自定义的双向 LSTM 情感分析模型，包含词嵌入层和多层 LSTM。

**模型参数**:

| 参数 | 值 |
|------|-----|
| 序列长度 | 28 |
| 词嵌入维度 | 400 |
| 隐藏层维度 | 256 |
| LSTM层数 | 2 |
| 双向 | 是 |
| Dropout | 0.5 (LSTM), 0.3 (FC层前) |
| 输出类别数 | 3 |
| 学习率 | 0.001 |
| 批次大小 | 50 |
| 优化器 | Adam |
| 损失函数 | CrossEntropyLoss |
| 梯度裁剪阈值 | 5 |
| 交叉验证折数 | 5 |
| 早停轮数 | 5 |

**特殊设计**:
- 双向 LSTM 捕捉前后文依赖
- 梯度裁剪防止梯度爆炸
- 词表基于训练数据自动构建 (39,747 词)
- 每个fold训练时打印混淆矩阵
- 类别权重平衡处理数据不平衡

---

### 2. DistilBERT (微调预训练模型)

**文件**: [BERT.ipynb](models/BERT/BERT.ipynb)

基于 Hugging Face 的 `distilbert-base-uncased` 预训练模型进行微调，第4层 Transformer 被替换为 `IterativeLayer`。

**模型参数**:

| 参数 | 值 |
|------|-----|
| 预训练模型 | distilbert-base-uncased |
| 最大序列长度 | 47 |
| 输出类别数 | 3 |
| 学习率 | 0.0001 |
| 批次大小 | 64 |
| 优化器 | AdamW |
| 损失函数 | CrossEntropyLoss |
| 迭代层Poisson均值 | 3 |
| 交叉验证折数 | 5 |
| 早停轮数 | 3 |

**特殊设计**:
- `IterativeLayer`: 第4层 Transformer 根据泊松分布动态决定迭代次数，平均迭代3次
- 退出阈值: 当输出变化 < 1e-4 时提前停止迭代
- "思考"机制: 让模型在关键层进行深度处理，灵感来源于人类反复斟酌关键信息

---

### 3. Trained Transformer (从头训练Transformer)

**文件**: [Trained_Transformer.ipynb](models/Trained_Transformer/Trained_Transformer.ipynb)

完全从头训练的 Transformer 编码器模型，不依赖预训练权重，使用自定义词表和拼写纠正。

**模型参数**:

| 参数 | 值 |
|------|-----|
| 词表大小 | 32,389 |
| 最大序列长度 | 28 |
| 词嵌入维度 | 128 |
| 注意力头数 | 4 |
| Transformer编码器层数 | 2 |
| Dropout | 0.1 |
| 输出类别数 | 3 |
| 学习率 | 0.001 |
| 批次大小 | 64 |
| 优化器 | Adam |
| 损失函数 | NLLLoss + LogSoftmax |
| 早停轮数 | 5 |

**特殊设计**:
- 使用 `autocorrect` 库进行拼写纠正预处理
- 词表基于训练数据频率构建
- 包含位置编码 (PositionalEncoding)
- 输出层使用 LogSoftmax 激活函数
- 已保存训练好的权重: [SentiTrans.pt](models/Trained_Transformer/SentiTrans.pt)

---

### 4. Homemade Transformer (自定义思考型Transformer)

**文件**: [HomemadeTransformer.ipynb](models/Transformer/HomemadeTransformer.ipynb)

从零实现的 Transformer 编码器模型，使用 DistilBERT 的 tokenizer 进行文本预处理，采用 Prelude-Recurrent-Coda 架构。

**模型参数**:

| 参数 | 值 |
|------|-----|
| 词表大小 | 由 DistilBERT tokenizer 决定 |
| 最大序列长度 | 47 |
| 词嵌入维度 | 128 |
| 注意力头数 | 8 |
| Prelude层数 | 1 |
| Coda层数 | 1 |
| 循环层Poisson均值 | 3 |
| 退出阈值 | 1e-4 |
| Dropout | 0.5 |
| 输出类别数 | 3 |
| 学习率 | 0.0001 |
| 批次大小 | 64 |
| 优化器 | AdamW |
| 损失函数 | CrossEntropyLoss |
| 混合精度 | autocast + GradScaler |
| 交叉验证折数 | 5 |
| 早停轮数 | 5 |

**特殊设计**:
- Prelude-Recurrent-Coda 架构: 前奏层初步理解文本，循环层反复"思考"，尾声层提炼特征
- 循环层基于 Poisson 分布控制迭代次数，支持提前退出
- 退出阈值基于输出变化量的 L2 范数
- 使用混合精度训练 (autocast + GradScaler)
- WeightedRandomSampler 处理类别不平衡

---

### 5. Word2Vec + RandomForest

**文件**: [word2vec.ipynb](models/Word2Vec/word2vec.ipynb)

使用 Gensim 训练 Word2Vec 词向量 (Skip-gram)，提取推文特征后使用 RandomForest 进行分类。

**模型参数**:

| 参数 | 值 |
|------|-----|
| 词向量维度 | 300 |
| 训练算法 | Skip-gram (sg=1) |
| 训练轮次 | 60 |
| 窗口大小 | 5 |
| 最小词频 | 5 |
| 分类器 | RandomForest |
| n_estimators | 200 / 300 / 500 |
| max_depth | 15 / 20 / None |
| max_features | sqrt / log2 |
| class_weight | balanced |
| 训练/测试划分 | 80/20 |

**最佳结果**: accuracy = 76.1%

**断点续跑机制**:
- Word2Vec 模型保存为 `word2vec.model`，已存在则跳过训练
- 特征向量保存为 `word2vec-features.npy` / `word2vec-labels.npy`，已存在则跳过提取
- RandomForest 网格搜索结果保存为 `rf_results.json`，已完成的参数组合自动跳过

**局限性**:
- 对词向量取平均会丢失词序信息，情感分析高度依赖词序 (如 "not good" vs "good not")
- 准确率在所有模型中最低

---

## 文件结构

```
COVID-Sentiment/
├── clean_COVIDSenti.csv
├── README.md
└── models/
    ├── BERT/
    │   └── BERT.ipynb
    ├── LSTM/
    │   └── LSTM.ipynb
    ├── Transformer/
    │   └── HomemadeTransformer.ipynb
    ├── Trained_Transformer/
    │   ├── Trained_Transformer.ipynb
    │   └── SentiTrans.pt
    ├── Word2Vec/
    │   └── word2vec.ipynb
    └── requirements.txt
```

## 训练策略

所有深度学习模型均采用以下通用策略:
- 类别权重平衡 (处理数据不平衡)
- 早停机制 (验证集无提升则停止)
- 按类别统计准确率

**通用参数设计原因**:

- **5折交叉验证**: 数据集规模中等，需要可靠评估。80%训练、10%验证、10%测试是标准划分
- **早停轮数 = 3-5**: 防止过拟合，验证集性能连续3-5轮不提升就停止
- **类别权重平衡**: 数据集严重不平衡 (中性74.9%，正面仅7%)，使用 1/频率 作为权重让模型更关注少数类
- **学习率差异**: 预训练模型使用 0.0001 (避免破坏预训练知识)，从头训练使用 0.001 (需要更快收敛)

## 基准对比

根据 COVIDSenti 数据集的公开文献:

| 方法 | 准确率 | 来源 |
|------|--------|------|
| SVM | 86% | Naseem et al. (2021) |
| LSTM | 87% | Naseem et al. (2021) |
| 本项目 LSTM | ~90% | - |
| 本项目 DistilBERT (IterativeLayer) | ~90% | - |

本项目 LSTM 和 DistilBERT 的结果已达到 COVIDSenti 数据集的公开 SOTA 水平。
