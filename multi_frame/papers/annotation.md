# 读书笔记
## Autoregressive product of multi-frame predictions can improve the accuracy of hybrid models
>In this paper the predictions at multiple time points were trained independently and a simple geometric average was used at test time.

![pic1](https://github.com/nuaalixu/picBed/raw/master/PicGo/Auto-regressive%20product%20model%20for%20speech%20recognition.png)

*自回归模型指基于目标变量历史数据的组合对目标变量进行预测*

训练时，多帧输入拼帧，对应多帧label。故输出层有多个独立的softmax组成

解码时，当前帧的预测由多个时刻的预测**几何平均**求得。类似model combination，但是这里只有一个模型，是不同时刻的值。