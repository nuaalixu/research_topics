# Notatioon

## END-TO-END AUTOMATIC SPEECH RECOGNITION INTEGRATED WITH CTC-BASED VOICE ACTIVITY DETECTION
### Motivation

E2E-ASR要求输入音频是segmented的短音频，需要VAD，额外的VAD模块弊端：

1. 需要额外的训练
2. 超参数需要精调，不直观。

## Innovation

将vad模块和E2E-ASR模型集成在一起，利用encoder的ctc输出non-blank和blank后验概率来做segment。

## Method

> The E2E model was mainly based on the hybrid CTC/attention architecture and trained on the segmented data provided by the datasets.

说明训练不变，推理时使用该内嵌vad方法。

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/imgimage-20210709120044722.png" alt="image-20210709120044722" style="zoom:80%;" />

## Experiment

比外置VAD方法更准；

