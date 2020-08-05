# 论文笔记

## Training Augmentation with Adversarial Examples for Robust Speech Recognition
> In this work, we propose data augmentation using adversarial examples for robust acoustic modeling.

*FGSM，即 fast gradient sign method，利用network对input的梯度，正向增长，使loss变大*

FGSM生成adversarial example：

![](https://github.com/nuaalixu/picBed/raw/master/PicGo/FGSM%20generate%20adversarial%20example.png)

sign() 表示只取符号±

T/S 进一步优化，联合loss：

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/combining%20TS%20training%20with%20data%20augmentation.png))

公式前半部分，*y*是clean data在 student model 对应的posterior，意味着让student model 学习 clean 和 noise 的domain-invariant representation

公式后半部分，即常规的T/S loss

## STUDENT-TEACHER NETWORK LEARNING WITH ENHANCED FEATURES

> This paper proposes a new student-teacher learning scheme for noise robust ASR, where the teacher model uses enhanced features while the student model uses noisy features during training.

两种T/S方法：

![](https://github.com/nuaalixu/picBed/raw/master/PicGo/two%20TS%20learning%20approaches.png)

常规方法，模型结构不同；

新方法，训练数据不同（parallel data），（模型结构亦可不同）

soft label 比 hard label 效果好：可能是noisy data 的 Viterbi alignments不够准确

normal-enhanced pairs 效果甚至比normal-clean pairs更好，因为前者的数据更parallel，后者录制时存在偏差。

## Adversarial Regularization for Attention Based End-to-End Robust Speech Recognition
> The essence behind AT and VAT is the same: seek a “worst” spot around the current data point, and then optimize the model using this “worst” data we just found.

### adversarial examples:

![](https://github.com/nuaalixu/picBed/raw/master/PicGo/definition%20of%20adversarial%20example.png)

### adversarial training:

![](https://github.com/nuaalixu/picBed/raw/master/PicGo/adversarial%20training.png)

两部分loss都是基于ground truth *y*的

训练时，先 *x* forward，不更新参数，在 



### virtual adversarial training

*LDS(local distribution smoothness): 模型输出相对于输入的顺滑度，用KL散度度量*

KL散度：

![](https://github.com/nuaalixu/picBed/raw/master/PicGo/KLD%20of%20VAT.png)

LDS：

![](https://github.com/nuaalixu/picBed/raw/master/PicGo/LDS%20of%20VAT.png)

loss ：

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/loss%20of%20VAT.png)

LDS旨在于衡量输入扰动很小时，模型输出的变化程度。

loss的后半部分，旨在于抑制这种变化，使其相对于输入smooth

### AT vs. VAT

不同点:

1. 扰动生成方式不同，有监督和无监督
2. loss 不同。AT旨在于减少扰动后模型输出和ground truth的损失，VAT旨在于减少扰动后和扰动前模型输出之间的损失

### augmentation 和 regularization （包括AT，VAT）：

相同点：

1. augmentation 和 AT 类似，均为两个相对于ground truth 的 CE loss

不同点：

1. augmentation 和 AT 不同在于，前者$x$ 和 $\widehat{x}$ 的loss无关联，前后分别更新两次模型；后者$x$ 和 $\widehat{x}$ 的loss联立，同时更新一次模型
2. augmentation和VAT明显不同，VAT loss 新增了衡量 smoothness 的部分

