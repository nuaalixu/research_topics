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

> The log mel spectrograms are normalized to have zero mean value, and thus setting the masked value
> to zero is equivalent to setting it to the mean value.

[对应代码](https://github.com/DemisEom/SpecAugment)