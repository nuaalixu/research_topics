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

## A TWO-STAGE APPROACH TO DEVICE-ROBUST ACOUSTIC SCENE CLASSIFICATION

### 本文方法

特点：

* two-stage
* data augmentation
* model fusion

#### Two-Stage Classification Procedure

第一步：粗分类，in-door、out-door、transporation三个场景

第二部：细分类，十个场景

设输入$x$，预测结果为$Class(x)$:
$$
Class(x)=\mathop{argmax}_{q,(p{\in}C^1,q{\in}C^2,p{\supset}q)}F^1_p(x)*F^2_q(x)
$$
其中$p{\supset}q$表示p是q的超集，比如transportation是bus、tram和metro的超集。

#### Ensemble

使用三种不同的CNN结构，训练模型，Resnet，FCNN，fsFCNN

#### Augmentation

generate extra data：

* 

not：

* mixup
* random cropping
* specaugment
* 

## ACOUSTIC SCENE CLASSIFICATION USING DEEP RESIDUAL NETWORKS WITH LATE FUSION OF SEPARATED HIGH AND LOW FREQUENCY PATHS

### 本文方法

图像和声谱图不同：

* 图像中，物体A会被物体B完全挡住，仅显示物体B。声谱图中，重叠的声音A和声音B，会相互作用。
* 图像中，物体A在不同位置，都代表物体A。声谱图中，不同位置代表不同的意义。
* 声谱图中，某一时间点的频率特征可能是非局部的，比如“谐波”。

故，声谱图的时间轴和频率轴，需要区别对待。

#### 模型结构

全卷积结构

1*1卷积的作用：

* 降维，减少feature map的channel数量
* 加入非线性，卷积层之后经过激励层
* 也可以用于pixel-wise像素分割

global average pooling（用于替换）：

* 将feature map逐channel求均值，输出维度为channel数的向量
* native to 卷积，建立分类和feature map间的直接关联
* 无新增参数，所以避免了该层的过拟合
* 保留空间信息，故对input的空间变换具有鲁棒性

#### Augmentation

mixup：两个训练样本对合成一个新的样本对
$$
X = {\lambda}X_1+(1-\lambda)X_2 \\
y={\lambda}y_1 + (1-\lambda)y_2
$$
其中，$\lambda$由beta分布$Be(\alpha,\alpha)$采样而来

时间轴crop：即沿时间轴随机抽取连续的400维作为训练样本

#### 训练

learning rate schedule：warm restart，即每隔一段时间重新设置learning rate 到maximum（$10^{-1}$)，然后衰减（$10^{-5}$）

## ACOUSTIC SCENE CLASSIFICATION FOR MISMATCHED RECORDING DEVICES USING HEATED-UP SOFTMAX AND SPECTRUM CORRECTION

## 问题

ASC训练和测试数据的设备不匹配，导致性能下降。

## 传统方法

ensemble techniques：averaging，weighted averaging，ensemble selection，random forests，sanpshot averageing。

domain adaptation and transfer learning。

regulation and data augmentation：mixup，specaugment，temporal cropping。

spectrum correction

## 本文方法

heated-up softmax embedding 

spectrum correction

focal loss

CNN

#### 模型结构

#### spectrum corrction

用于

步骤：

* 需要aligned recording（不同设备录同一条音频）

* 在spectrogram上做规整（而不是fbank）

* 第一步，计算correction coefficients

  * aligned recording 一对

  * 其中一条作为reference device的频谱，另一条是目标设备X的频谱
  * reference device的频谱除以目标设备X的频谱（逐元素），获取一个系数
  * 选择多对，计算多个修正系数，求均值，即为设备X的修正系数

* 第二步，设备X的频谱乘以设备X的修正系数，得到修正后的频谱。

#### focal loss

普通的ce loss：
$$
CE(p,y)=-\sum^C_{j=1}y_ilog(p_j)
$$
focal loss的思想是，通过动态缩放ce loss，来减轻分类正确sample的权重，从而更关注难分类（易错）的sample。
$$
FL(p,y)=-\sum^C_{j=1}(1-p_j)^{\gamma}y_ilog(p_j)
$$
其中，通过缩放系数$（1-p_j)^{\gamma}$来控制ce loss的focus到易错类。