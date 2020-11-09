# 论文笔记

## SMALL-FOOTPRINT KEYWORD SPOTTING USING DEEP NEURAL NETWORKS

**Keyword Spotting (KWS)**: aims at detecting predefined keywords in an audio stream.

**DeepKWS:** A deep neural network is trained to directly predict the keyword(s) or subword units of the keyword(s) followed by a posterior handling method producing a final confidence score.

advantage: In contrast with the HMM approach, this system does not require a sequence search algorithm (decoding)

### **传统的HMM结构：**

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/HMM%20topoloy%20for%20KWS.png" style="zoom:67%;" />

filler: non-keyword

### DeepKWS 结构：

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/frameworkofDeepKWS.png" style="zoom:67%;" />

建模单元：word /subword，建模单元影响输出层的size，也要考虑对应数据是否稀疏

### **Posterior Handling**:

Posterior smoothing:

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/formula%20of%20posterior%20smoothing.png" style="zoom:67%;" />

Confidence:

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/formula%20of%20confidence.png" style="zoom:67%;" />

如果以word建模，key phrase为唤醒词，无法解决顺序问题。