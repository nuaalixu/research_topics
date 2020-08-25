# 论文笔记

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