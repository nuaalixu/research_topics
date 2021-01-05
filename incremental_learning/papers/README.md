# 论文笔记

## PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning

incremental learning要克服的问题：catastrophic forgetting

> incremental learning is different from transfer learning in that we aim to have good performance in both old and new classes.

现有的增量学习方法: 

1. reusing a limited amount of previous training data [33,3]; 

2. learning to generate the training data [17,36]; 
3. extending the architecture for new phases of data [39,22]; 
4. using a sub-network for each phase [7,11]; 
5. constraining the model divergence as it evolves [18,25,1,23,33,3].



## Incremental Learning for End-to-End Automatic Speech Recognition

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

### Remark

通过distillation loss来保留旧任务的信息。

> This very strategy can cause issues if the data for the new task belongs to a distribution different from that of prior tasks

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

## Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting

*Sequential Transfer Learning*:

​	learns source tasks and target tasks in sequence, and transfers knowledge from source tasks to improve the models' performance on target tasks

* pretraining
* adaptation
  - fine-tuning
  - feature extraction: freezing some weights

*Multi-task Learning*:

​	learns multiple tasks simutaneously

*Elastic Weight Consolidation(EWC)*: 正则手段，限制对原任务重要的参数，调整其他的参数

本文方法包括两种机制：

- Pretraining Simulation
- Objective Shifting

### Pretraining Simulation

使用了一系列的近似优化方法将pretraining tasks的训练目标Loss_S近似推导为一个不依赖于pretraining data而只依赖于pretrained model的二次惩罚项。
论文中的公式推导借鉴并基于EWC方法，通过使用Laplaces方法和模型参数之间的独立性假设，对于pretraining tasks的训练目标$Loss_S$进行近似优化的完整过程。

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/pretraining%20simulation.png" style="zoom:80%;" />

### Objective Shifting

 to allow the objective function to gradually shift to $Loss_T$ with the **annealing coefficient**.

将普通multi-task loss中的常数权重$\lambda$替换为关于时间的退火函数$\lambda(t)$：

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/objective%20shifting.png" style="zoom:80%;" />

退火函数：

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/annealing%20function.png" style="zoom:80%;" />

其中，$k$和$t_0$分别控制退火率和时间步的超参。

### Algorithm

将本算法放在优化器部分实现，结合Adam，构建RecAdam

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/Algorithm%201%20Adam%20and%20RecAdam.png" style="zoom:80%;" />

### Remark

本质上，通过初始权重，作为正则项，保存older task信息。

通过退火系数，权衡新旧任务的比重。

## PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning

### Remark

模型结构，每个task拥有专属的parameters，和一个对应的mask，mask用于pruning。同时，各task拥有shared parameters。

训练时，各task按顺序依次训练。每个task训练时，先用mask“剪枝”，相当于固定其他task的专属参数和共享参数，再调整本task的专属参数。

推理时，通过每个task对应的mask，来”屏蔽”其他task的专属参数，以使当前模型“转换”为该task的专属模型。

Pruning的意义在于，保持模型性能不下降的条件下，减少每个task需要的模型参数量。