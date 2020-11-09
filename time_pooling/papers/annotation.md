# 论文笔记

## Dense Prediction on Sequences with Time-Dilated Convolutions for Speech Recognition

> strided pooling: allowing to access more context on higher feature maps, while reducing the spatial resolution
>
> With pooling, the receptive field in time of the CNN can be larger than the same network without pooling

pooling 扩大了感受野，但是降低了分辨率

### Time-dilated convolutions

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/time-dilated%20convolution.png)

图b使用普通pooling，扩大了感受野，但是降低了分辨率

图c使用time-dilated convolution，扩大感受野的同时保持原分辨率。其中，浅色部分的计算与图b相同，多出了深色部分，才保持分辨率不变

time-dialted convolution 本质上是在卷积核内部增加skip，卷积核的感受野扩大，卷积的计算量不变。

## Advances in Very Deep Convolutional Neural Networks for LVCSR

> The advantage is that higher layers in the network are able to access more context and can learn useful invariants in time
>
> The disadvantage is that the resolution is reduced with which neighboring but different CD states can be distinguished, which could possibly hurt performance.

### Exploring pooling and padding in time

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/different%20versions%20of%20the%2010-layers%20CNN.png)

>  We compared three model variants, and discussed the importance of time-padding and time-pooling.
> • Architecture (a) with pooling performs better then (b) and (c) without pooling.
> • Only architecture (c) without padding or pooling allows for batch normalization and efficient convolutional processing of full utterances.

### Efficient convolution over full utterances

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/two%20ways%20of%20evaluatiing%20a%20full%20utterance.png)

**A** 重复计算太多

**B** 减少了重复计算