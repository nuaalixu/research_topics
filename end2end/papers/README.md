# 论文笔记

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

第一部分是e2e模型的分数，因为短句子上模型偏差更小，所以e2e模型分数根据句子长度规整55

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