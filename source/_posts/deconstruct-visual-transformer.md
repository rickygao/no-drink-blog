---
title: 解构 Visual Transformer
date: 2020-11-25 12:00:00
tags: [Computer Vision, Transformer]
mathjax: true
---

本文针对 [:paperclip: arXiv](https://arxiv.org/abs/2006.03677) 的早期版本，与最新版本会有所出入。

## 知识背景

### Attention (Retrieval)

文中所有的公式都是从下式派生，

$$
\mathrm{Attention}\left(Q,K,V\right)=\mathrm{softmax}\left(QK^T\right)V,
$$

其中 $Q\in\mathbb{R}^{N_Q\times d_K}$、$K\in\mathbb{R}^{N_V\times d_K}$、$V\in\mathbb{R}^{N_V\times d_V}$、$R\in\mathbb{R}^{N_Q\times d_V}$。

<!-- more -->

具体解释为：

- $Q$ 即 Query，代表 $N_Q$ 个想要检索的目标，每行是一个 $d_K$ 维的查询关键字。例如，可以按类型将电影 embed 到一个关键字空间中，每个分量可以代表其具有的不同属性；
- $K$ 即 Key，代表 $N_V$ 个被检索的候选者，每行是一个 $d_K$ 维的查询关键字。是所有被检索的候选者在关键字空间中的表示；
- $V$ 即 Value，代表 $N_V$ 个被检索的候选者，每行是一个 $d_V$ 维的值。例如，将电影的内容记录为一个向量，是真正想要取回的内容；
- $M=\mathrm{softmax}\left(QK^T\right)\in\mathbb{R}^{N_Q\times N_V}$ 即 Match，根据内积计算相似度并归一化，对于每个检索目标，相似度高的候选者将取得较高的值。

有时 $Q$、$K$、$V$ 并不直接存在，而是由已有的值进行变换得到，例如：

$$
\mathrm{Attention}\left(QW_Q,KW_K,VW_V\right)=\mathrm{softmax}\left(\left(QW_Q\right){\left(KW_K\right)}^T\right)\left(VW_V\right).
$$

此时，通常 $K=V$ 或者它们的空间维度一致。例如检索电影时，电影的内容和类型是混杂在一起的信息，并具有很多冗余，需要利用 $W_K$、$W_V$ 对其进行变换，分别得到类型矩阵和内容矩阵；同时检索的查询关键字也存在冗余，或者并非一个直接的查询，需要 $W_Q$ 进行变换。

### Multi-Head

本文中没有详细介绍（但是提到使用了该技术），使用该技术可以显著减少 Attention 的计算开销。

$$
\mathrm{MultiHead}\left(Q,K,V\right)=\mathrm{concat}\left(\mathrm{head}_i\right)W_O,
$$

其中，$\mathrm{head}_i=\mathrm{Attention}\left(Q{W_Q}_i,K{W_K}_i,V{W_V}_i\right)$，$i=1,2,\dots,h$。这将大型矩阵乘法划分在 $h$ 个子空间计算，最后使用 $W_O$ 进行组合。

## 视觉模块

### Filter-Based Tokenizer (Static Tokenizer)

「利用静态的 Queries 从像素特征图检索 Tokens 信息」

由下式开始变换，

$$
R=\mathrm{softmax}\left(QK^T\right)V,
$$

令 $K=\bar{X}W_K$、$V=\bar{X}W_V$ 且 $\bar{X}\in\mathbb{R}^{HW\times C}$ 是由原特征图将空间维度 flatten 得到，$X_K$、$X_V$ 分别是从像素特征空间变换为关键字空间和值空间（也即 Token 的特征空间）的线性变换。意为从原特征图中利用静态的查询 $Q$ 检索信息，有

$$\begin{aligned}
T
&=\mathrm{softmax}\left(Q{\left(\bar{X}W_K\right)}^T\right)\left(\bar{X}W_V\right)\\
&=\mathrm{softmax}\left(\left(QW_K^T\right)\bar{X}^T\right)\left(\bar{X}W_V\right).
\end{aligned}$$

若取 $d_K=C$、$d_V=C_T$，并记 $QW_K^T=W_A$ 得原文公式，

$$
T=\mathrm{softmax}\left(W_A\bar{X}^T\right)\left(\bar{X}W_V\right)=A^TV,
$$

可学习的参数为 $W_{A}$、$W_V$。

### Recurrent Tokenizer (Dynamic Tokenizer)

「利用已有的 Tokens 得到 Queries 从像素特征图检索新的 Tokens 信息」

是 Filter-based Tokenizer 的变种，此时使用的 $Q$ 并非静态的，而是已有 Tokens 进行变换得到，即 $Q=T_{in}W_{Q}$，有

$$\begin{aligned}
T_{out}
&=\mathrm{softmax}\left(\left(T_{in}W_Q\right){\left(\bar{X}_{in}W_K\right)}^T\right)\left(\bar{X}_{in}W_V\right)\\
&=\mathrm{softmax}\left(\left(T_{in}W_QW_K^T\right)\bar{X}_{in}^T\right)\left(\bar{X}_{in}W_V\right),
\end{aligned}$$

记 $W_QW_K^T=W_{T\to W_A}$ 得

$$\begin{aligned}
T_{out}
&=\mathrm{softmax}\left(\left(T_{in}W_{T\to W_A}\right)\bar{X}_{in}^T\right)\left(\bar{X}_{in}W_V\right)\\
&=\mathrm{StaticTokenizer}\left(T_{in}W_{T\to W_A},\bar{X}_{in}\right),
\end{aligned}$$

可学习的参数为 $W_{T\to W_A}$、$W_V$，原文形式是只更新一半的 Tokens。

### Position Encoder

​「利用 Tokenizer 的匹配结果，从静态的 Position Encodings 中检索位置信息」

保持 Tokenizer 的 Match 不变，令 $V=W_{A\to P}\in\mathbb{R}^{\bar{H}\bar{W}\times C_P}$，是从位置编码 $W_{A\to P}$ 中按权重取回值（我理解是为了降维或者固定位置编码的维数的考虑，进行了下采样），得

$$
P=\mathrm{downsample}{\left(A\right)}^TW_{A\to P},
$$

可学习的参数为 $W_{A\to P}$。

### Transformer (Self-Attention)

「Tokens 进行自源的信息检索」

对于

$$
R=\mathrm{softmax}\left(\left(QW_Q\right){\left(KW_K\right)}^T\right)\left(VW_V\right),
$$

令 $Q=K=V=T_{in}$，改记 $W_Q=Q$、$W_K=K$、$W_V=V$，并一般取 $d_K=C_T/2$ 有

$$
T_{out}=\mathrm{softmax}\left(\left(T_{in}Q\right){\left(T_{in}K\right)}^T\right)\left(T_{in}V\right).
$$

再加上残差结构，即

$$
T_{out}=T_{in}+\mathrm{softmax}\left(\left(T_{in}Q\right){\left(T_{in}K\right)}^T\right)\left(T_{in}V\right),
$$

可学习的参数为 $Q$、$K$、$V$。

### Projector

「利用像素特征图从 Tokens 中检索信息，从而对特种图进行了语义的增强」

对于

$$
R=\mathrm{softmax}\left(\left(QW_Q\right){\left(KW_K\right)}^T\right)\left(VW_V\right),
$$

令 $Q=\bar{X}$、$K=V=T_{out}$ 且 $W_Q=Q_{T\to X}$、$W_K=K_{T\to X}$、$W_V=V_{T\to V}$，并一般取 $d_K=C_T/2$（原文这三个变换矩阵的维度有误），有

$$
\bar{X}_{out}=\mathrm{softmax}\left(\left(\bar{X}_{in}Q_{T\to X}\right)
{\left(T_{out}K_{T\to X}\right)}^T\right)\left(T_{out}V_{T\to X}\right).
$$

同样地再加上残差结构，即

$$
X_{out}=X_{in}+\mathrm{softmax}\left(\left(\bar{X}_{in}Q_{T\to X}\right)
{\left(T_{out}K_{T\to X}\right)}^T\right)\left(T_{out}V_{T\to X}\right),
$$

可学习的参数为 $Q_{T\to X}$、$K_{T\to X}$、$V_{T\to X}$。

### 模块总结

| Name                   | Query           | Key         | Value       | Result              |
| ---------------------- | --------------- | ----------- | ----------- | ------------------- |
| Static Tokenizer (ST)  | Static          | Feature Map | Feature Map | Tokens              |
| Dynamic Tokenizer (DT) | Tokens          | Feature Map | Feature Map | Refined Tokens      |
| Position Encoder (PE)  | Static / Tokens | Tokens      | Static      | Position Encodings  |
| Transformer (Tr)       | Tokens          | Tokens      | Tokens      | Transformed Tokens  |
| Projector (Pr)         | Feature Map     | Tokens      | Tokens      | Refined Feature Map |

## 模型构建

1. Classification: 使用 ST、Tr、DT、Tr、……、DT、Tr 结构代替了 Resnet 中的 Stage 5。
2. Semantic Segmentation: 使用 ST、Tr、Pr 结构代替了 FPN 中的 Lateral Conv 和 Downsample、Down Conv。（依据原文描述，原文的示意图可能有误，标出了 downsample）
