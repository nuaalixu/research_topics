# 读书笔记
## Autoregressive product of multi-frame predictions can improve the accuracy of hybrid models
>In this paper the predictions at multiple time points were trained independently and a simple geometric average was used at test time.

![p1](https://github.com/nuaalixu/picBed/raw/master/PicGo/Auto-regressive%20product%20model%20for%20speech%20recognition.png)

*自回归模型指基于目标变量历史数据的组合对目标变量进行预测*

训练时，多帧输入拼帧，对应多帧label。故输出层有多个独立的softmax组成

解码时，当前帧的预测由多个时刻的预测**几何平均**求得。类似model combination，但是这里只有一个模型，是不同时刻的值。



## MULTIFRAME DEEP NEURAL NETWORKS FOR ACOUSTIC MODELING

> This paper presents a novel approach to training DNNs for hybrid systems which compares advantageously in terms of decoding complexity at equivalent accuracy to the standard approach

*帧同步模型，baseline，每帧输入对应一帧 prediction*

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/frame%20synchronous%20approach.png)

*帧异步模型，缺失 prediction 为复制*

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/frame%20asynchronous%20approach.png)

*multiframe 模型，输出层扩展，多个prediction共用相同隐层

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/multiframe%20approach.png)

训练时，input low frame rate，label frame rate不变。

解码时，同训练。

## Multi-Frame Cross-Entropy Training for Convolutional Neural Networks in Speech Recognition
> To adapt modern CNNs to ASR, three aspects need some care.
>
> Firstly, before the output layer, the convolutional pathway feeds into a stack of fully connected (FC) layers.
>
> Secondly, in the domain of speech, [3] says to avoid padding the convolutional layers in the time dimension, as this would prevent efficient full-utterance processing at test time.
>
> Thirdly, for efficient sequence prediction with CNNs, we additionally replace pooling in time with time-dilated convolutions

MFCE: 将多个相邻帧的CE loss联立

对于 full utterance cross-entropy training 来说，无效