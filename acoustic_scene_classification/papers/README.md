# 论文笔记

## A Convolutional Neural Network Approach for Acoustic Scene Classification

### 问题

声学场景分类

### 之前解决方法

GMM、SVM、tree bagger classifiers等

### 本文方法

基于CNN

#### 输入和输出

输入：segment切成多个短音频组成sequence

输出：每个短音频的分类得分，它们的得分平均数最大值对应的类别是segment的分类
$$
c^* = \mathop{argmax}_{c}[\frac{1}{M}\sum^{M}_{i=1}y^{(i)}_{c}]
$$


#### 特征提取

1. 求log-mel spectrogram（mel-scale filter bank后取log）
2. 均值和标准差规整
3. 切成不重叠的短音频

#### 模型结构

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/CNN%20in%20ASC.png)

conv（5x5） + max pooling （5x5）+ conv （5x5）+ max pooling（4x全时间轴）+ fully connective

#### 正则化和训练

> Batch normalization, introduced in [26], is a technique that addresses the issue described by Shimoidara et al. [27], known as **internal covariate shift**.

BN能够加速收敛

训练方法：

* non-full training
  * 将training data分为training（80%）和validation（20%）
  * randomly shuffle和time-shift，保证训练sequence多样性
  * validtaion控制训练epoch次数
* full training
  * 使用全部training data，retrain
  * 训练epoch数跟non-full一样

sequence 长度 3s最佳

场景混淆

> This may indicate that our model is relying more on the background noise of the sequence rather then on acoustic event occurrences. 

### remark

CNN应用于ASC的基础型文章。同时也提到了以往ASC的方法，比如GMM、SVM和tree bagger classifiers。