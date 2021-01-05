# 论文笔记

## Tacotron: Towards end-To-end speech synthesis

> We have proposed Tacotron, an integrated end-to-end generative TTS model that takes a character sequence as input and outputs the corresponding spectrogram. With

TTS is a large-scale inverse problem: a highly compressed source (text) is “decompressed” into audio.

### TTS 传统架构

a text frontend extracting various linguistic features

a duration model

an acoustic feature prediction model

a complex signal-processing-based vocoder

### MODEL ARCHITECTURE

e2e：trained on <text, audio> pairs with minimal human annotation. 

**Tacotron directly predicts raw spectrogram**

输入是char（embedding)，e2e网络的target是mel频谱，之后会接后处理网络，其target是线性频谱。

We use a simple L1 loss for both seq2seq decoder (mel-scale spectrogram) and post-processing net (linear-scale spectrogram). The two losses have equal weights. 

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/model%20architecture%20of%20tacotron.png" style="zoom:80%;" />

#### CBHG MODULE

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/CBHG.png" style="zoom:80%;" />

CB：1-D conv bank，一维卷积组，每组卷积kernel size递增，相当于1gram、2gram、3gram...

H：highway network

<img src="https://raw.githubusercontent.com/nuaalixu/picBed/master/PicGo/highwav%20networks.png" style="zoom:80%;" />

本文配置，4 layers of FC-128-ReLU

G： bi-GRU

> We found that this CBHG-based encoder not only reduces overfitting, but also makes fewer mispronunciations than a standard multi-layer RNN encoder

#### DECODER

An important trick we discovered was predicting multiple, non-overlapping output frames at each decoder step.

#### POST-PROCESSING NET AND WAVEFORM SYNTHESIS

**post-processing nets** convert the seq2seq target to a target that can be synthesized into waveforms

**Griffin-Lim algorithm**  to synthesize waveform from the predicted spectrogram. 