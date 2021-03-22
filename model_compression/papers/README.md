# 论文笔记

## DEEP COMPRESSION: COMPRESSING DEEP NEURAL NETWORKS WITH PRUNING, TRAINED QUANTIZATION AND HUFFMAN CODING

### 问题

large network需要消耗过多存储和能量，以致于难以部署在移动设备上。

### 本文方法

三步：剪枝-量化-huffman编码

#### Network Pruning

流程：正常模型训练-消除低于阈值的权重-retrain网络

剪枝后，权重矩阵变稀疏（sparse）。

使用compressed sparse row（CSR）或compressed sparse column（CSC）格式储存。

只储存非0值。

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/representing%20the%20matrix%20sparsity.png" style="zoom:80%;" />

为了进一步压缩，储存index的相对值，而不是绝对值。

#### Trained Quantization and Weight Sharing

权重共享即将权重矩阵分成k个簇，每个簇共享同一个值。

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/weight%20sharing.png" style="zoom:80%;" />

设给定k个簇，那么需要$log_2(k)$个位来表示全部的index。设网络权重矩阵有n个元素，每个元素值用b位表示。那么权重共享完成的压缩比为：
$$
r=\frac{nb}{nlog_2(k)+kb}
$$
将权重分类，即聚类，本文使用1-D k-means聚类方法

#### Huffman Coding

一种数据压缩方法，将出现频率高的数据，用较短的bit编码。主要目的是根据使用频率来最大化节省字符（编码）的存储空间。

### Remark

多种方法的组合。

related work部分介绍了模型压缩的多种方法，可用于该领域入门学习。