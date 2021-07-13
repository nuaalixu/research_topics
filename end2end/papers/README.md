# 论文笔记

## Attention机制

### 概述

以对齐的视角看attention：

![preview](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/bVbwad5)

attention过程分为三步：
$$
\mathbf{z} = \sum_{i=1}^n\alpha_{i}\mathbf{y}_i \\
\alpha_{i} = align(e_{i})=\frac{exp(e_i)}{\sum_{i^\prime=1}exp(e_{i\prime})} \\
e_{i}=score(\mathbf{c},\mathbf{y}_i)
$$




以QKV模型视角看

![preview](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/bVbwael)

也分为三步：
$$
e_i=score(q,\mathbf{k}_i) \\
\alpha_i = softmax(e_i) \\
\mathbf{c}=\sum_i\alpha_i\mathbf{v}_i
$$


attention由以下三步构成：

- score function：度量query向量和key向量的相似性
- alignment function：计算attention weight，通常使用softmax进行归一化
- context vector function：根据attention weight得到输出向量

修改以上三步内容，可以构成不同的attention。

### Attention类型

从计算区域、所用信息、结构层次和模型等方面对Attention的形式进行归类。

**1. 计算区域**

根据Attention的计算区域，可以分成以下几种：

1）**Soft Attention**，这是比较常见的Attention方式，对所有key求权重概率，每个key都有一个对应的权重，是一种全局的计算方式（也可以叫Global Attention）。这种方式比较理性，参考了所有key的内容，再进行加权。但是计算量可能会比较大一些。

2）**Hard Attention**，这种方式是直接精准定位到某个key，其余key就都不管了，相当于这个key的概率是1，其余key的概率全部是0。因此这种对齐方式要求很高，要求一步到位，如果没有正确对齐，会带来很大的影响。另一方面，因为不可导，一般需要用强化学习的方法进行训练。（或者使用gumbel softmax之类的）

3）**Local Attention**，这种方式其实是以上两种方式的一个折中，对一个窗口区域进行计算。先用Hard方式定位到某个地方，以这个点为中心可以得到一个窗口区域，在这个小区域内用Soft方式来算Attention。

**2. 所用信息**

假设我们要对一段原文计算Attention，这里原文指的是我们要做attention的文本，那么所用信息包括内部信息和外部信息，内部信息指的是原文本身的信息，而外部信息指的是除原文以外的额外信息。

1）**General Attention**，这种方式利用到了外部信息，常用于需要构建两段文本关系的任务，query一般包含了额外信息，根据外部query对原文进行对齐。

2）**Local Attention**，这种方式只使用内部信息，key和value以及query只和输入原文有关，在self attention中，key=value=query。既然没有外部信息，那么在原文中的每个词可以跟该句子中的所有词进行Attention计算，相当于寻找原文内部的关系。

**3. 结构层次**

结构方面根据是否划分层次关系，分为单层attention，多层attention和多头attention：

1）**单层Attention**，这是比较普遍的做法，用一个query对一段原文进行一次attention。

2）**多层Attention**，一般用于文本具有层次关系的模型，假设我们把一个document划分成多个句子，在第一层，我们分别对每个句子使用attention计算出一个句向量（也就是单层attention）；在第二层，我们对所有句向量再做attention计算出一个文档向量（也是一个单层attention），最后再用这个文档向量去做任务。

3）**多头Attention**，这是Attention is All You Need中提到的multi-head attention，用到了多个query对一段原文进行了多次attention，每个query都关注到原文的不同部分，相当于重复做多次单层attention，最后再把这些结果拼接起来。

### score函数

以下介绍几种不同的score函数。

additive attention/Bahdanau attention:

利用feedforward  neural network作为score函数，softmax归一化
$$
e_{i,j} =\mathbf{w}^Ttanh(\mathbf{W}\mathbf{s}_{t-1} + \mathbf{V}\mathbf{h}_i+b)
$$


location-based attention：

引入前一时刻alignment的信息。
$$
\alpha_i=attend(s_{i-1},\alpha_{i-1})
$$
具体实现，可以通过对$\alpha_{i-1}$施加卷积：
$$
e_{i,j} = \mathbf{w}^Ttanh(\mathbf{W}\mathbf{s}_{t-1} + \mathbf{V}\mathbf{h}_i+\mathbf{U}f_{i,j}+b) \\
f_i = \mathbf{F}*\alpha_{i-1}
$$


Scaled Dot-Product：

和点积attention很像，只是增加了scale factor。因为当输入较大时，softmax函数的梯度趋近于零，所以需要对输入进行缩小。
$$
score(\mathbf{s}_t,\mathbf{h}_i)=\frac{\mathbf{s}_t^T\mathbf{h}_i}{\sqrt{n}}
$$


## END-TO-END ATTENTION-BASED LARGE VOCABULARY SPEECH RECOGNITION

### A pooling over time BiRNN:

> We found that for our decoder such representation is overly precise and contains much redundant information.

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/a%20pooling%20over%20time%20BiRNN.png" style="zoom:67%;" />

所谓pooling over time，其实就是第二层跳帧。

### Encoder-Decoder Architecture

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/Attention-based%20Recurrent%20Sequence%20Generator.png" style="zoom: 67%;" />

> making the attention location-aware, that is using $\alpha_t$ in the equations defining $\alpha_t$, is crucial for reliable behaviour on long input sequences.

## Listen, Attend and Spell

### 与CTC的区别

> The network produces character sequences without making any independence assumptions between the characters. 

跟CTC模型相比，本质区别是没有每帧独立性假设。

CTC的帧独立假设：
$$
P(\mathbf{y}|\mathbf{x}) = \prod_{i}(y_i|\mathbf{x})
$$
encoder-decoder 架构：
$$
P(\mathbf{y}|\mathbf{x}) = \prod_{i}{P(y_i|\mathbf{x},y_{<i})}
$$

### Model Framework

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/LAS%20model%20structure.png" style="zoom:80%;" />

### Listen

encoder使用Bidirectional Long Short Term Memory RNN (pBLSTM) with a pyramid structure. 

pBLSTM用于降低attention模块的输入长度（时间轴），减少计算量。

”pyramid“体现在后面几层BLSTM的时间步越来越少。

encoder用于提取高阶representation序列：$\mathbf{h}$
$$
h^j_i = BLSTM(h^j_{i-1}, h^{j-1}_i) \\
h^j_i = pBLSTM(h^j_{i-1}, [h^{j-1}_{2i}, h^{j-1}_{2i+1}])
$$
其中，i,j分别表示time step和 layer。

### Attend and Spell

总体公式：

三步骤：生成$c_i$，生成$s_i$，生成P
$$
c_i = AttentionContext(s_i,\mathbf{h}) \\
s_i = RNN(s_{i-1}, y_{i-1}, c_{i-1}) \\
P(y_i|\mathbf{x}, y_{<i}) = CharacterDistribution(s_i, c_i)
$$

其中，CharacterDistribution 是一个MLP

$c_i$的生成公式：
$$
e_{i,u} = <\phi(s_i), \psi(h_u)> \\
\alpha_{i,u} = \frac{exp(e_{i,u})}{\sum_{u}{exp(e_{i,u})}} \\
c_i = \sum_{u}{\alpha_{i,u}h_{u}}
$$
可以选择不同的函数生成$e_{i,u}$

### Learning

训练trick：

* 故意选择错误的预测输入到下一时刻
* 发现pretrain无用
* 发现multi-task无用

### Decoding and Rescoring

$$
s(\mathbf{y}|\mathbf{x}) = \frac{P(\mathbf{y}|\mathbf{x})}{|y|_c} + \lambda\log{P_{LM}(\mathbf{y})}
$$

第一部分是e2e模型的分数，因为短句子上模型偏差更小，所以e2e模型分数根据句子长度规整

### Experiment

***oracle WER***:* By oracle WER is meant WER of the path, stored in the generated lattice, which best matches the utterance regardless the scores. In other words, oracle WER gives us a bound on how well can we get by tuning scores in a given lattice*

#### Effects of Utterance Length

太长和太短效果均不好。

可能解决办法：location-based prior

#### Word Frequency

通常rare word的召回率比高频词差。但是也受到声学发音独特性的影响。

#### Interesting Decoding Examples

同样的音频，LAS能够转成大相径庭的文本，比如“triple a”和“aaa”。因为没有基于帧独立假设。

LAS能够有效识别包含重复词的音频。因为LAS采用content-based attention，所以预期应该无法正确attention重复词。实际上，LAS做到了正确attention。

## Sequence Transduction with Recurrent Neural Networks

### 问题

机器学习任务很多可以抽象为序列转换问题。

模型需要找到可变序列（长度、内容）的不变表达（representation）。

从表象抓住本质。

### 通常解决方法

普通RNN模型（此处特指用于seq2seq任务的RNN）：

* 对齐（alignment）问题
  * 训练需要帧对齐
  * output长度与input长度一致
* 帧独立假设

RNN-CTC：

* 帧独立假设
* output长度必须小于input长度

### 新方法

RNNT：

* output理论上任意长度
* ouput帧之间有关联

#### 符号注释

输入序列：
$$
\mathbf{x}=(x_1,x_2,...,x_T),\mathbf{x}\in \mathcal{X}^*
$$
输出序列：
$$
\mathbf{y}=(y_1,y_2,...,y_U),\mathbf{y}\in\mathcal{Y^*}
$$
$\varnothing$表示输出为空：
$$
\overline{\mathcal{Y}}^*=\mathcal{Y^*}\cup\varnothing
$$
$\mathbf{a}$是输入和输出间的alignment：
$$
\mathbf{a}\in\overline{\mathcal{Y}}^*
$$


条件分布：
$$
Pr(\mathbf{y}\in\mathcal{Y^*}|\mathbf{x})=\sum_{\mathbf{a}\in\mathcal{\beta^{-1}(y)}}Pr(\mathbf{a}|\mathbf{x})
$$

#### 模型整体架构

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/RNN-Transducer.png)

#### Prediction Network

预测网络采用一个next-step-prediction式的RNN，起到“语言模型”的作用。



带输出层的RNN的公式应当如下：
$$
h_u=\mathcal{H}(W_{ih}\hat{\mathbf{y}}_u+W_{hh}h_{u-1}+b_h)\\
g_u=W_{ho}h_u+b_o
$$
本文中转换函数$\mathcal{H}$采用了LSTM，而不是传统激活函数。

$(y_1,y_2,...,y_u)$输入，输出$(g_1,g_2,...,g_u)$

#### Transcription Network

转录网络对输入序列编码，起到”声学模型“的作用。

本文采用Bi-RNN来获取上下文信息。

$(x_1,x_2,...,x_T)$输入，输出$(f_1,f_2,...,f_T)$

#### Joint Network/Output Distribution

可以用简单的MLP实现，本文使用如下公式：
$$
h(k,t,u)=exp(f^k_t+g^k_u)\\
Pr(k\in\mathcal{\overline{Y}}|t,u)=\frac{h(k,t,u)}{\sum_{k'\in\mathcal{\overline{Y}}}h(k',t,u)}
$$
$f$和$g$均为K+1维，k表示向量的第k个元素

简化注释，定义：
$$
y(t,u)\equiv{Pr(y_{u+1}|t,u)}\\
\varnothing(t,u)\equiv{Pr(\varnothing|t,u)}
$$
下图，从左下到右上，每一条路径对应着x和y之间的一种可能的alignment：

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/output%20probability%20lattice.png" style="zoom:80%;" />

输出的目标同样是
$$
Pr(\mathbf{y}|\mathbf{x})
$$
其是所有可能alignment的概率和。

利用动态规划的前后向算法求和。

#### Test

beam search

## TRIGGERED ATTENTION FOR END-TO-END SPEECH RECOGNITION

### motivation

受限于没有frame-by-frame的对齐能力，传统的attention-based decoder不适用于流式识别。

原因重点不在于encoder，而在于decoder的attention是全局的。

### innovation

本文重点在于改进decoder的attention机制，使其变成 frame-synchronous。

从ctc的output找到best path，其中非blk的时刻作为decode的trigger。

decoder的attention范围仅从历史看到触发的当前时刻$t$，+ delay。

> During training, forced alignment of the CTC output sequence is used to derive the
> time instants of the triggering. 
>
> During decoding, uncertainties of the CTC trained trigger model are taken into account to generate alternative trigger sequences and output sequences, respectively.

#### model architecture

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/imgimage-20210524190530380.png" alt="image-20210524190530380" style="zoom:80%;" />

### experiment

#### baseline

full sequence attention model。

#### results

triggered attention比full sequcence attention (label synchronous decoding)好。

不同的attention对delay（look ahead frame）需求不同，不是越长越好。

## MINIMUM WORD ERROR RATE TRAINING FOR ATTENTION-BASED SEQUENCE-TO-SEQUENCE MODELS
### Method

**MWER通用公式**
$$
\mathcal{L}_{wer}(\bold{x}, \bold{y}^*)=\mathbb{E}[\mathcal{W}(\bold{y},\bold{y}^*)]=\sum_{\bold{y}}P(\bold{y}|\bold{x})\mathcal{W}(\bold{y},\bold{y}^*)
$$
其实计算的是WER的期望，期望 = 概率 x WER值。

期望计算理论上是变量所有取值的和，但是实现困难，所以有以下两种估计方法。

**随机采样实现**
$$
\mathcal{L}(\bold{x},\bold{y}^*)\approx\mathcal{L}^{sample}_{werr}(\bold{x},\bold{y}^*)=\frac{1}{N}\sum_{\bold{y}_i\sim P(\bold{y}|\bold{x})}\mathcal{W}(\bold{y}_i,\bold{y}^*)
$$
随机变量的取值控制在有限次。

**NBest实现**																					

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/imgimage-20210712205150345.png" alt="image-20210712205150345" style="zoom:80%;" />

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/imgimage-20210712205218845.png" alt="image-20210712205218845" style="zoom:80%;" />

概率值做了归一化，$\widehat{W}$指的wer平均值，实际上对梯度无影响。

**训练loss**

多目标loss
$$
\mathcal{L}^{N-best} = \sum_{(\bold{x},\bold{y}^*)}\mathcal{L}_{werr}^{N-best}(\bold{x},\bold{y}^*) + \lambda\mathcal{L}_{CE}
$$
sequence级别loss和frame级别loss联合，有利于训练稳定

