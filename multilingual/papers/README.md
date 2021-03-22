# 论文笔记

## overview

conventional method (language dependent)：

- network structure

- multi-task learning
- layer sharing
- residual learning
- knowledge distillation
- data augmentation

E2E method(language independent):

​	a single model for multi-lingual task



## Multilingual Speech Recognition with Self-Attention Structured Parameterization

### 目标

多语言流式ASR

### 对比方法



### 创新点

提出两个新方法来包含LID（语种id）信息

### 本文方法

#### 模型结构

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/LID%20in%20the%20multi-headed%20attention%20layer.png" style="zoom:80%;" />

#### 利用Language ID 

共有三部分：

Language Embedding Concatenation

- one-hot 向量表示language ID
- LID向量和input of multi-headed attention拼接
- 相当于给普通的mha多了一个language-specific bias项

Language Specific Attention Head

- 将mha分为共享部分和language-specific部分
- language-specific部分的权重仅使用对应语种数据训练
- 最终输出由两部分产生的heads拼接在一起

Language Dependent Attention Span

- 本方法使用left-context in self-attention来具备流式能力
- 通过function生成mask，来控制attention范围
- 不同语种的attention拥有不同的mask，及不同的attention范围

### 实验配置

#### 数据

本文使用的5种语言，是相互易懂的（同一语系），含有重叠的字母。

