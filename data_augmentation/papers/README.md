# SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition

**The augmentation policy consists of :**

**warping the features**

图像扭曲操作：借助“锚点”，通过前后“锚点”的变化，来扭曲图像。

六个固定点，一个随机点。随机点在谱图中间，沿时间轴随机取。

**masking blocks of frequency channels**

频率范围限制，整张图mask

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/frequency%20mask.png)

**masking blocks of time steps.**

时间范围限制，整张图mask

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/time%20mask.png)

> The log mel spectrograms are normalized to have zero mean value, and thus setting the masked value to zero is equivalent to setting it to the mean value.

[对应代码](https://github.com/DemisEom/SpecAugment)

# MixSpeech: Data Augmentation for Low-resource Automatic Speech Recognition
## Mixup Recap
加权组合输入和label。
$$
X_{mix}=\lambda X_i + (1-\lambda)X_j\\
Y_{mix}=\lambda Y_i + (1-\lambda)Y_j
$$
其中\lambda的选择服从Beta分布，$\lambda \sim Beta(\alpha,\alpha)$
较难用于序列生成任务，如ASR，原因：
1.label长度不一致；
2.label是离散的，直接相加有问题；
## MixSpeech
将label的mixup调整为loss的mixup。
$$
\mathcal{L}_i = \mathcal{L}(X_{mix},Y_i)\\
\mathcal{L}_j = \mathcal{L}(X_{mix},Y_j)\\
\mathcal{L}_{mix} = \lambda \mathcal{L}_i+(1-\lambda)\mathcal{L}_j
$$