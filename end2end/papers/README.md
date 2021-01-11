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

### 其他e2e方法的区别

之前的e2e方法：CTC 和 enc-dec with attention

CTC的区别

> The network produces character sequences without making any independence assumptions between the characters. 

跟CTC模型相比，本质区别是没有每帧独立性假设。

enc-dec with attention

之前仅在phoneme上建模，没有真正的e2e

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

公式：
$$
c_i = AttentionContext(s_i,\mathbf{h}) \\
s_i = RNN(s_{i-1}, y_{i-1}, c_{i-1}) \\
P(y_i|\mathbf{x}, y_{<i}) = CharacterDistribution(s_i, c_i)
$$
