# 论文笔记

## END-TO-END ATTENTION-BASED LARGE VOCABULARY SPEECH RECOGNITION

### A pooling over time BiRNN:

> We found that for our decoder such representation is overly precise and contains much redundant information.

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/a%20pooling%20over%20time%20BiRNN.png" style="zoom:67%;" />

所谓pooling over time，其实就是第二层跳帧。

### Encoder-Decoder Architecture

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/Attention-based%20Recurrent%20Sequence%20Generator.png" style="zoom: 67%;" />

> making the attention location-aware, that is using $\alpha_t$ in the equations defining $\alpha_t$, is crucial for reliable behaviour on long input sequences.

## Attention-Based Models for Speech Recognition

