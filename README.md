# COVID-Sentiment
Final Project + Paper for COMP9444 @ UNSW

## 项目概述
本项目针对COVID-19相关推文进行情感分析，实现了四种不同的深度学习模型，将推文分类为负面、中性或正面三种情感类别。

## 数据集
- 数据文件: `clean_COVIDSenti.csv`
- 类别: 3类 (负面=0, 中性=1, 正面=2)
- 预处理: 推文经过清洗和token化处理

## 模型架构与参数

### 1. DistilBERT (微调预训练模型)
**文件**: [BERT.ipynb](models/BERT/BERT.ipynb)

基于Hugging Face的`distilbert-base-uncased`预训练模型进行微调。

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
| 早停轮数 | 5 |

**特殊设计**:
- 第4层Transformer层被替换为`IterativeLayer`，支持基于Poisson分布的动态迭代次数
- 使用早停机制防止过拟合

**参数设计原因**:

**最大序列长度 = 47**: 通过分析COVID-19推文数据集得出，推文token化后最大长度为47个token。BERT类模型的注意力机制复杂度为O(n²)，序列长度直接影响计算量，47确保不丢失任何信息的同时避免浪费计算资源。

**学习率 = 0.0001**: 微调预训练模型时使用较小的学习率，避免"灾难性遗忘"（破坏预训练学到的知识）。比从头训练的学习率(0.001)小10倍是常见做法。

**Poisson均值 = 3**: 这是"BERT with Thinking"架构的核心创新。第4层Transformer会根据Poisson(3)分布动态决定迭代次数，平均迭代3次，让模型在关键层进行"深度思考"。配合退出阈值(当输出变化<阈值时提前停止)，实现自适应计算，灵感来源于人类思考问题时会反复斟酌某些关键信息。

**批次大小 = 64**: 平衡内存使用和训练稳定性。批次越大，梯度估计越准确，训练越稳定，但受限于内存，64是常见的折中值。

---

### 2. LSTM (循环神经网络)
**文件**: [LSTM_Covid.py](models/LSTM/LSTM_Covid.py)

自定义的LSTM情感分析模型，包含词嵌入层和多层LSTM。

**模型参数**:
| 参数 | 值 |
|------|-----|
| 序列长度 | 28 |
| 词嵌入维度 | 400 |
| 隐藏层维度 | 256 |
| LSTM层数 | 2 |
| Dropout概率 | 0.5 (LSTM), 0.3 (FC层前) |
| 输出类别数 | 3 |
| 学习率 | 0.001 |
| 批次大小 | 50 |
| 优化器 | Adam |
| 损失函数 | CrossEntropyLoss |
| 梯度裁剪阈值 | 5 |
| 交叉验证折数 | 5 |
| 早停轮数 | 5 |

**特殊设计**:
- 使用梯度裁剪防止梯度爆炸
- 词表大小基于训练数据自动构建
- 每个fold训练时打印混淆矩阵

**参数设计原因**:

**序列长度 = 28**: 与DistilBERT不同，LSTM使用自定义tokenizer，词粒度更大。28个词足以覆盖大多数推文。LSTM按顺序处理序列，过长会导致梯度消失/爆炸，比BERT的47短是因为词级别比subword级别包含更多信息。

**词嵌入维度 = 400**: 较大的嵌入维度可以捕捉更丰富的语义信息。400维是经验值，比常见的300维(GloVe)稍大，更大的维度可以区分更多词汇的细微差别，但会增加参数量和过拟合风险。

**隐藏层维度 = 256**: LSTM的隐藏状态大小，代表模型的"记忆容量"。越大能记住更长的上下文依赖，但计算复杂度为O(hidden_dim²)，需要权衡。

**LSTM层数 = 2**: 多层LSTM可以学习层次化特征。第1层捕捉局部特征(词组、短语)，第2层捕捉全局特征(句子级别语义)。层数过多会导致梯度问题，2层是常见选择。

**Dropout = 0.5 (LSTM), 0.3 (FC层前)**: 防止过拟合的正则化技术。0.5是LSTM的标准dropout值(论文推荐)，LSTM内部有门控机制，本身有一定抗过拟合能力。FC层前用0.3是因为全连接层更容易过拟合。

**梯度裁剪阈值 = 5**: RNN/LSTM特有的梯度爆炸问题解决方案。RNN反向传播时梯度会指数级增长，裁剪到5以内保证训练稳定性，这是训练RNN的必备技巧。

---

### 3. Homemade Transformer (自定义Transformer)
**文件**: [HomemadeTransformer.ipynb](models/Transformer/HomemadeTransformer.ipynb)

从零实现的Transformer编码器模型，使用DistilBERT的tokenizer进行文本预处理。

**模型参数**:
| 参数 | 值 |
|------|-----|
| 词表大小 | 由tokenizer决定 |
| 最大序列长度 | 47 |
| 词嵌入维度 | 128 |
| 注意力头数 (nhead) | 8 |
| Prelude层数 | 1 |
| Coda层数 | 1 |
| 循环层Poisson均值 | 3 |
| Dropout | 0.5 |
| 输出类别数 | 3 |
| 学习率 | 0.0001 |
| 批次大小 | 64 |
| 优化器 | AdamW |
| 损失函数 | CrossEntropyLoss |
| 交叉验证折数 | 5 |
| 早停轮数 | 5 |

**特殊设计**:
- 包含位置编码 (PositionalEncoding)
- 中间层使用Poisson分布控制迭代次数，支持提前退出机制
- 退出阈值基于输出变化量的L2范数

**参数设计原因**:

**注意力头数 = 8**: Transformer的核心超参数。8头意味着模型可以同时关注输入的不同方面，例如：头1关注情感词，头2关注否定词，头3关注程度副词。128维嵌入 / 8头 = 每头16维，是合理的分配。原始Transformer论文使用8头。

**Prelude层数 = 1, Coda层数 = 1**: 这是"思考型Transformer"的架构设计。Prelude(前奏)负责初步理解输入文本，Recurrent(循环)是中间的"思考"层进行迭代处理，Coda(尾声)负责最终提炼特征。1+1的结构让模型有足够能力但不复杂。

**Poisson均值 = 3 (循环层)**: 与DistilBERT相同的"思考"机制。让模型在关键位置反复处理信息，对于复杂情感(如讽刺、双重否定)，多次迭代有帮助。退出机制避免不必要的计算。

**Dropout = 0.5**: 从头训练的Transformer更容易过拟合，比预训练模型(0.1)大很多。因为没有预训练知识的正则化效果，需要更强的随机失活来防止记忆训练数据。

---

### 4. Trained Transformer (从头训练Transformer)
**文件**: [Trained_Transformer.ipynb](models/Trained_Transformer/Trained_Transformer.ipynb)

完全从头训练的Transformer模型，不依赖预训练权重，使用自定义词表。

**模型参数**:
| 参数 | 值 |
|------|-----|
| 词表大小 | ~32,389 |
| 最大序列长度 | 28 |
| 词嵌入维度 | 128 |
| 注意力头数 (nhead) | 4 |
| Transformer编码器层数 | 2 |
| Dropout | 0.1 |
| 输出类别数 | 3 |
| 学习率 | 0.001 |
| 批次大小 | 64 |
| 优化器 | Adam |
| 损失函数 | NLLLoss (配合LogSoftmax) |
| 早停轮数 | 5 |

**特殊设计**:
- 使用autocorrect库进行拼写纠正
- 词表基于训练数据频率构建
- 输出层使用LogSoftmax激活函数

**参数设计原因**:

**词表大小 ≈ 32,389**: 基于训练数据统计得出，只包含数据集中实际出现的词。比预训练tokenizer的词表小，拼写纠正后减少了词表膨胀。

**注意力头数 = 4**: 比Homemade Transformer少。4头是较小的配置，适合从头训练。头数少意味着每头需要学习更多功能，计算量更小，适合资源受限的情况。

**Transformer编码器层数 = 2**: README中提到"uses 8 total attention mechanisms"。2层 × 4头 = 8个注意力机制。层数少是因为从头训练数据量不足，深层Transformer需要大量数据才能发挥优势。

**Dropout = 0.1**: 相对较低的dropout，可能是因为模型本身参数不多(2层)。过大的dropout会阻碍学习，配合其他正则化手段(如早停)使用。

**损失函数 = NLLLoss + LogSoftmax**: 数学上等价于CrossEntropyLoss。分开写可能是为了调试或修改方便，在某些框架中分开实现更灵活。

**使用autocorrect拼写纠正**: 推文包含大量拼写错误，拼写纠正可以减少词表大小(同一词的不同拼写合并)，提高模型泛化能力，对从头训练的模型尤其重要(没有预训练知识)。

---

## 重要文件说明
- DistilBERT微调: [BERT.ipynb](models/BERT/BERT.ipynb)
- 自定义Transformer: [HomemadeTransformer.ipynb](models/Transformer/HomemadeTransformer.ipynb)
- 从头训练Transformer: [Trained_Transformer.ipynb](models/Trained_Transformer/Trained_Transformer.ipynb)
- LSTM模型: [LSTM_Covid.py](models/LSTM/LSTM_Covid.py)
- 训练好的Transformer权重: [SentiTrans.pt](models/SentiTrans.pt)

## 训练策略
所有模型均采用以下通用策略:
- 5折交叉验证
- 类别权重平衡 (处理数据不平衡)
- 早停机制 (验证集5轮无提升则停止)
- 按类别统计准确率

**通用参数设计原因**:

**5折交叉验证**: 数据集规模中等，需要可靠评估。每份数据都会被用作测试集一次，减少评估方差。80%训练、10%验证、10%测试是标准划分。

**早停轮数 = 5**: 防止过拟合的通用策略。验证集性能5轮不提升就停止，避免在训练集上过度拟合，节省计算时间。

**类别权重平衡**: COVID-19推文数据集类别不平衡。使用1/频率作为权重，让模型更关注少数类，防止模型偏向多数类。

**学习率差异**: 预训练模型使用0.0001(小，避免破坏预训练知识)，从头训练使用0.001(大，需要更快收敛)。
