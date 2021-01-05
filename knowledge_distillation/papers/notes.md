# 论文笔记

## Distilling the Knowledge in a Neural Network

本质上是一种“知识”的迁移。

> Once the cumbersome model has been trained, we can then use a different kind of training, which we call “distillation” to transfer the knowledge from the cumbersome model to a small model that is more suitable for deployment.

**soft targets 方法**

soft targets 包含的信息更多，“熵”更大。

provide much more information per training case than hard targets and much less variance in the gradient between training cases

所以需要less data and higher learning rate

问题：当soft target的置信度很高时，其他incorrect的概率接近与零，直接计算，对于交叉熵 loss的影响很小

解决思路一：取log，log能够向下规整

**如何生成soft target**

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/distillation%2520loss.png" style="zoom:80%;" />

*logit* $z_i$ *：the input of the final softmax*

*T: 温度，用于平滑soft target，通常>=1，=1时等价于普通softmax*



**Distillation Loss**

output 和 soft target 的交叉熵损失函数（因为Teacher Model不更新，所以KLD 等价于CE）

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/image-20201104201340596.png" alt="image-20201104201340596" style="zoom:80%;" />

所以梯度相当于普通hard target 交叉熵损失函数梯度的$1/T^2$，在联合loss中需要将soft target loss 放大$T^2$