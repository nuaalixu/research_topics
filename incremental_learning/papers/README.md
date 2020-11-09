# 论文笔记

## Incremental Learning for End-to-End Automatic Speech Recognition

incremental learning要克服的问题：catastrophic forgetting

**incremental learning 主要分为三类:**

1. without using old data
2. using synthetic data
3. using exemplars

**公式说明incremental learning，fine-tuning，retrain：**

*$D_1, D_2$分别表示new dataset 和 old dataset*, $D_0 = D_1 \cup D_2$

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/formula%20of%20cremental%20learning.png" style="zoom:80%;" />

**incremental learning 与 fine-tuning 的主要不同：**

- fine-tuning 只要求在 new task 上的性能
- incremental 既要求在 new task 上性能好，还要求在 old dataset 上与原模型性能近似

**retrain:**

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/formula%20of%20retrain.png" style="zoom:80%;" />

**方法图解**:

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/a%20schematic%20representation%20of%20incremental%20learning%20for%20e2e%20ASR.png" style="zoom:80%;" />

**loss**：

ctc loss + distilling loss（KL散度）



## SERIL:  Noise Adaptive Speech Enhancement using Regularization-based Incremental Learning

> In this paper, we propose a regularization-based incremental learning strategy for adapting DL-based SE models to new environments (speakers and noise types) while handling the
> catastrophic forgetting issue.

***Speech Enhancement:*** *The objective of speech enhancement (SE) is to transform low-quality speech signals into enhanced-quality speech signals.*



## Incremental Classifier Learning with Generative Adversarial Networks



## Learning without Forgetting

notation:

$\theta_s$: 模型的共享参数（隐层）

$\theta_o$: 模型的old 输出层

$\theta_n$: 模型的new输出层

三种更新$\theta_n$的方法：

**Feature Extraction**: $\theta_s$, $\theta_o$不变，只更新$\theta_n$

**Fine-tuning**: $\theta_s, \theta_n$ 均更新，$\theta_o$不变，通常小学习率

**Joint Training**: $\theta_s, \theta_o, \theta_n$均更新

![](https://github.com/nuaalixu/picBed/raw/master/PicGo/learning without forgetting.png)

基于 Knowledge Distillation，修改multi-loss，包含:

KLD loss（old data），保持old task的能力

CE（loss），学习new task能力

正则化项（weight decay），平衡old、new task能力

![](https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/Procedure for Learning without Forgetting.png)

## iCaRL: Incremental Classifier and Representation Learning

> **iCaRL**：incremental classifier and representation learning
>
> a practical strategy for simultaneously learning classifiers and a feature representation in
> the class-incremental setting.

## Incremental Classifier Learning with Generative Adversarial Networks

incremental learning 的典型方法：保留代表性的old sample，并且使用distillation 正则化。

该方法的问题：

1. loss 的效率不高
2. old data和 new data不平衡
3. 挑选出的代表性的old sample有局限性
4. old data可能完全无法获取

本文提出了：

1. 新的loss function，联合CE 和 distillation
2. 简单的方法解决新旧数据的平衡问题
3. 利用GAN生成代表性的old sample

loss function，包含distillation loss（KLD)和CE loss：

$$L = \Lambda L_d + (1 - \Lambda) L_c$$

利用GANs 生成exemplars，和new data 一起训练。

