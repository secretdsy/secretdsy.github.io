---
title: "[논문 리뷰] Attention Is All You Need (Transformer)"
date: 2025-04-15 15:30:00 +0900
categories: [Paper]
tags: [Transformer, Attention, NLP]
math: true
toc: true
---

## 📝 논문 정보

- **제목**: Attention Is All You Need  
- **저자**: Vaswani et al.  
- **학회/연도**: NeurIPS 2017  
- **링크**: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)

---

## 1. Introduction

NLP 분야는 **Transformer**가 등장한 이후, 엄청난 발전을 이루었다. 특히 이 모델은 Language를 넘어 Vision, Vision-Language Models(VLM) 등 다양한 분야로 확장되면서, 딥러닝의 기본 골격을 다시 쓰는 데 큰 역할을 했다.  
Transformer가 나오기 전까지 주류였던 모델들은 대부분 순차적 구조를 기반으로 한 RNN 계열 모델(RNN, LSTM, GRU 등)이었다. **이들은 시간 순서에 따라 입력을 처리하는 특성 때문에 병렬화가 어렵고, 긴 문장에서 과거 정보를 기억하는 데 한계가 있었다.**  
본 논문은 이러한 기존 모델들의 구조적 한계를 해결하기 위해, 순차적 처리를 전혀 사용하지 않고 전적으로 Attention 메커니즘에만 기반한 Transformer 아키텍처를 제안하였다. 이로써 NLP 뿐만 아니라 다양한 분야에서 병렬화가 가능하고 장기 의존성 문제를 극복한 모델들이 빠르게 등장하기 시작했다.  
이번 글에서는 원 논문에 맞추어 NLP 분야를 중심으로 Transformer의 구조와 동작 방식을 살펴보고자 한다.

---

## 2. Limitations of Traditional RNN-based Models

**기존 모델들은 어떤 문제가 있었을까?**  
Recurrent Neural Network(RNN)은 시퀀스 데이터를 다루기 위해 고안된 구조로, 현재 입력과 과거 정보를 함께 고려하여 출력을 생성한다. 그러나 이 구조는 다음과 같은 한계를 가진다.  

- 순차적 처리(Sequential Processing): 현재 입력을 처리하기 위해 이전 시점의 출력을 반드시 기다려야 하므로, 병렬 처리가 불가능하다.
- 장기 의존성 문제(Long-Term Dependency Problem): 입력 시퀀스가 길어질수록 과거 정보가 소실되어, 먼 과거의 정보를 잘 기억하지 못한다.  

![RNN-short](/assets/2025-04-15-transformer/0-1.png)  
겨우 3개의 단어로 이루어진 문장이라 할지라도, 이전 시점의 출력이 나올때까지 **기다린 후에** 다음 시점의 출력을 계산할 수 있다.  

![RNN-long](/assets/2025-04-15-transformer/0-2.png)  
또한, "I love ... you" 문장에서 "love"와 "you" 사이에 엄청 많은 단어가 있는 문장이라면, "I" 와 "love" 에 대한 정보가 "you"까지 전달이 잘 되지 않는 **Vanishing Gradient** 문제가 생긴다.  

**Vanishing Gradient 문제는 왜 생길까?**  
딥러닝의 어원은 layer가 여러개, 즉 깊게 쌓는다고해서 딥러닝이라고 한다. 그리고 layer를 깊게 쌓는 것에 의미를 가지려면 비선형성을 유지하며 쌓아야하는데 (선형적이면 여러 layer를 쌓을 필요가 없음. 예를 들어, 첫 번째 layer는 모든 값을 2배로, 두 번째 layer는 모든 값을 3배로 하는 모델과 layer가 하나지만 모든 값을 6배 하는 모델과 다를 바가 없음), 보통 이런 비선형성은 흔히 아는 sigmoid, tanh, ReLU 와 같은 activation function을 통해 유지할 수 있다.  
그런데 이렇게 깊게 쌓은 구조는 학습을 위해 backpropagation 연산을 하는 과정에서 문제가 생긴다. sigmoid, tanh, ReLU 함수들의 gradient 값 범위는 아래와 같다.  
- sigmoid: (0, 0.25]
- tanh: (0, 1]
- ReLU: 0 or 1

이러한 값들이 여러번 곱해진다면 0에 가까운 값이 나올 수 밖에 없고, 이러한 이유로 층이 깊으면 **Vanishing Gradient** 문제가 생기게 된다. 그리고 NLP에서는 문장이 길면 길수록 수 많은 RNN 셀을 거쳐서 전달이 되기 때문에 층이 깊어지는 것과 비슷하게 작용하여 Vanishing Gradient가 발생한다.

이를 개선하기 위해 Long Short-Term Memory(LSTM)과 Gated Recurrent Unit(GRU) 같은 모델이 제안되었지만, 근본적으로 "순차적 구조"를 완전히 벗어나지는 못했다.

---

## 3. Self-Attention

**Self-Attention은 뭘까?**  
Attention은 사전적으로는 "참조, 집중, 주목, 주의"를 뜻한다. 그리고 앞에 self가 붙었으니 "자기 자신 참조, 집중, 주목, 주의" 정도가 될 것이다. 그리고 입력이 문장이라면 "자기 자신"은 입력 문장이 된다.  

"The animal didn't cross the road because it was too tired."  

예를 들어 위 문장이 있을 때, **"it"** 은 문장에 있는 모든 단어들 중에 **"animal"** 과 더 관련이 있다고 판단하도록 하는 것이다.  

이제 Attention 수식을 보며 설명하려고 한다.  

$$
Attention(Q, K, V) = softmax{\left(\frac{QK^T}{\sqrt{d_k}}\right)}V
$$

$Q, K, V, \sqrt{d_k}$ 각각이 뜻하는 바는 다음과 같다.  
- **$Q$ (Query)**: 궁금한 단어  
- **$K$ (Key)**: 비교할 단어  
- **$V$ (Value)**: $K$와 짝지어진 정보  
- **$\sqrt{d_k}$ (dimension of key vector)**: $QK^T$ 값이 너무 커지는 것을 방지하기 위한 scaling factor  

수식의 우변을 보면 $softmax$(확률 값)과 $V$를 곱하게 되어있다.  
즉, $Q$ 와 $K$의 유사도를 기반으로 $V$의 가중치를 정한다고 할 수 있다.  

$$
Output_i = \sum_{j=1}^{n}Attention(i, j) \cdot V_j
$$

그리고 두 번째 수식은 Output vector를 계산하는 식이고, "입력 문장 전체를 참조하여 더 관련이 있을 법한 단어를 반영" 한다고 볼 수 있다.  

예를 들어 $i = 1, n = 3$ 일 때, $Output_1$ 은 아래 세개의 vector의 합이다.  
- 첫번째 단어와 첫번째 단어가 얼마나 유사한지를 계산하고, **첫번째 단어**와 짝지어진 정보를 가중치만큼 반영  
- 첫번째 단어와 두번째 단어와 얼마나 유사한지를 계산하고, **두번째 단어**와 짝지어진 정보를 가중치만큼 반영  
- 첫번째 단어와 세번째 단어와 얼마나 유사한지를 계산하고, **세번째 단어**와 짝지어진 정보를 가중치만큼 반영  

Self-Attention은 입력 시퀀스 내 각 단어가 다른 모든 단어를 얼마나 **"Attend"** 해야 하는지를 계산하는 메커니즘이다. 각 단어는 Query, Key, Value로 변환된 후, Query와 Key의 유사도를 계산하고, 이를 바탕으로 Value의 가중합을 구한다. 그 결과, 각 단어는 문장 전체를 참조하여 의미를 재구성한 새로운 벡터로 표현된다.  
**Multi-Head Attention은 말 그대로 Attention이 여러 개 있는 형태인데, 왜 여러 개를 썼을까?**  
하나의 Attention만 사용한다면, 모든 관계를 동일한 방식으로만 학습하게 된다. Transformer에서는 이를 보완하기 위해 여러 개의 Attention Head를 병렬로 적용하는 Multi-Head Self-Attention 방식을 도입했다.  
각 Head는 서로 다른 부분에 주목할 수 있도록 학습되며, 서로 다른 관계(문법적 관계, 의미적 유사성 등)를 포착할 수 있다. 이후 이 Head들의 출력을 하나로 합쳐 최종 Attention 출력을 만든다.  
Multi-Head 구조는 마치 CNN에서 Feature map을 추출할 때 필터를 하나만 쓰는 것이 아니라, 다수 채널의 필터를 사용하는 것과 유사하다고 볼 수 있다.

---

## 4. Transformer Architecture

![Transformer Architecture](/assets/2025-04-15-transformer/1.png)  
_(이미지 기준 왼쪽은 Encoder, 오른쪽은 Decoder)_  

아주 간략하게 요약하면, Encoder는 입력 문장을 벡터로 변환하고, Decoder는 벡터를 출력 문장으로 바꿔주는 역할을 한다.  

Encoder부터 살펴보면 아래와 같은 순서로 입력된다.  
Input + Posional Encoding -> Multi-Head Attention -> Feed Forward  
**[Multi-Head Attention -> Feed Forward]** 부분은 그림에서 * N 표시가 되어있는데, 논문에서는 6번 반복해서 쌓았다고 한다.  

**Decoder의 Multi-Head Attention에는 왜 Masked가 붙어 있을까?**  
Masked Multi-Head Attention은 학습 시에 미래 시점의 단어를 알지 못하도록 하는 역할을 한다. 예를 들어 "I love"라는 단어가 생성된 상황에서는 "you"를 예측할 때 "you"라는 단어를 미리 볼 수 없도록 하기 위해 Self-Attention을 계산할 때, 현재 시점("I love") 이후의 단어들("you")에 대해 Attention score를 강제로 $-\infty$ 로 설정한다. 이렇게 되면 softmax를 취한 결과, 이후 단어에 대한 Attention 확률은 0이 되어버린다. 즉, 모델이 미래 단어를 전혀 참조할 수 없게 만드는 것이다.

이 과정을 간단히 요약하면 다음과 같다.  
- Self-Attention: 입력 시퀀스 전체를 참조 가능
- Masked Self-Attention: 미래 시점은 Masking하여 현재 시점까지의 단어만 참조 가능

덕분에 Decoder는 Auto-Regressive(순차 생성) 방식으로 문장을 하나씩 생성할 수 있다. Encoder와 마찬가지로, Masked Attention 구조를 Multi-Head로 확장하여 다양한 관점에서 과거 토큰들의 정보를 동시에 고려할 수 있도록 했다.  

**Posinal Encoding(PE)은 왜 필요할까?**  
사실 이 논문을 처음 보는 것도 아니고, 이번에 다시 보면서 가장 크게 얻은 부분이 바로 PE다.  
물론 혼자서 깨달은 것이 아님을 먼저 밝히며 내 친구이자 멘토인 [Yedarm Seong](https://github.com/mybirth0407)에게 감사를 표한다.  

**우선, PE는 단순히 각 Embedding의 위치 정보를 전달하기 위한 장치다.**  

Transformer는 PE없이는 RNN, LSTM과 다르게 순서를 알 수가 없다. **그런데 나는 이말이 제일 이해가 안갔다!**  
"왜 순서를 알 수가 없지? 분명 Input Embedding을 만들 때 순서를 지켜서 만들 수 있을텐데", 그리고 "순서를 지키면 당연히 모델이 Input Embedding의 순차적인 정보를 이해할 수 있을 것"이라고 생각했다.  
그런데 수식을 다시 살펴보면 $QK^T$ 내적할 때 문장에서 K가 Q앞에 있는지, 뒤에 있는지 알려주는 부분이 있나? 당연히 없다...  

만약 아래 내용을 더 빨리 알았다면, Input Embedding을 만들 때 순서를 지키고 말고 생각할 것 없이 PE가 왜 필요한지 더 빨리 알았을 것 같다.

Multi-layer Perceptron(MLP)에서 Input Layer의 Patch 단위로 순서를 섞어도 결국 매우 비슷한 Output을 낸다고 한다. 이 내용은 [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf)의 결과를 참고한 내용이다.  

![MLP-Mixer Results](/assets/2025-04-15-transformer/2.png)  

그럼 이제 다시 입력으로 "I love you"라는 문장이 있을 때 "Love I you", "You I love" 등 여러 조합을 만들 수 있고, 각각의 단어가 Patch라고 보면 MLP-Mixer에서 증명한 내용에 의해 어떤 조합으로 들어와도 결과가 매우 비슷하다.  
근데 이건 MLP 얘기 아니냐 할 수 있다. MLP 얘기가 맞다. 근데 Transformer도 PE가 없다면, 그리고 Self-Attention이 없다면, 더 넓은 관점에서는 MLP랑 다를 바가 없다고 볼 수 있다.  

**이 논문에서 사용한 PE는 어떤 방식일까?**  
sin/cos PE, Rotary PE, Relative PE 등 다양한 방법이 있다고 한다. 이번 글에서는 논문에서 사용한 방식인 sin/cos PE에 대해서만 설명하고자 한다.  

sin/cos PE의 수식은 아래와 같다.  
$$
PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})
$$
$$
PE_{pos,2i+1} = cos(pos/10000^{2i/d_{model}})
$$

$ pos, i, d_{model} $ 은 각각 아래의 의미로 쓰인다.
- $ pos $: 문장 내에서 몇 번째 토큰인지
- $ i $: PE Embedding 에서 몇 번째인지 ($ 0 \leq i < d_{model} / 2 $)
- $ d_{model} $: Input Embedding과 동일한 차원

예를 들어 $ pos = 3, i = 0, d_{model} = 512 $라면, **입력 문장에서 3번째 토큰의 512차원인 vector 중에 0, 1 index의 값**을 뜻하고, 아래와 같은 방식으로 계산된다.

$$
PE_{(3, 0)} = sin(3/10000^{0/512})
$$
$$
PE_{(3, 1)} = cos(3/10000^{0/512})
$$

**그런데 왜 $ d_{model} $ 이 Input Embedding과 동일한 차원일까?**  
Architecture를 다시 살펴보면, Attention 연산을 하기 전에 Input Embedding과 PE를 더해서 전달해준다. 덧셈 연산이 가능하려면 차원이 동일해야하기 때문에 Input Embedding과 동일한 차원을 사용한다.  

**sin/cos PE의 특징은 무엇일까?**  
**우선, $ pos, i $ 변화에 따른 그래프를 살펴보자.**  

![pos-3-4](/assets/2025-04-15-transformer/3.png)  
_(그림 3. pos=3, pos=4 일때 차원 변화에 따른 PE값 비교)_  

![pos-3-500](/assets/2025-04-15-transformer/4.png)  
_(그림 4. pos=3, pos=500 일때 차원 변화에 따른 PE값 비교)_  

sin/cos PE의 특징 중 하나는 가까운 위치에 있는 토큰끼리는 차이가 적고, 먼 위치에 있는 토큰끼리는 차이가 크다. 그림 3에 있는 두 그래프는 매우 유사하고, 그림 4에 있는 두 그래프는 차이가 상대적으로 커보인다.  
따라서 거리가 가깝고 멀고에 대한 상대적인 거리도 판별이 가능하다.

![dim](/assets/2025-04-15-transformer/5.png)  
_(그림 5. 차원별 pos 변화에 따른 PE값)_  

각 차원 별로 pos 변화에 따른 PE값을 살펴보자.  
dim = 0, 10일 때는 주기가 매우 짧다. 그리고 dim = 300, 511일 때는 변화가 없는 것처럼 보이는데 주기가 차원의 크기($d_{model} = 512$)보다 커져서 변화량이 매우 작을 뿐 변화가 없는 것은 아니다.  

**sin/cos PE는 다음과 같은 장점이 있다.**  
- 학습 파라미터가 없기 때문에 학습 시간에 큰 영향을 끼치지 않는다.
- 학습 중에 보지 못한 길이의 문장도 대응 가능하다.

**하지만 sin/cos PE는 다음과 같은 단점이 있다.**  
- 차원이 매우 큰 경우, 차원의 크기보다 주기가 더 작은 경우가 생기기 때문에 같거나 매우 유사한 PE 벡터가 생길 수 있다.
- 감정 분석과 같은 특정 태스크에서 **중요한 위치(감정 표현이 잦은 문장 맨 앞, 맨 뒤)** 를 학습할 수 없다.
- 단순히 수학적으로 위치를 특정 위치가 문법적으로 어떤 역할을 하는지는 구분하기 힘들다. (예를 들어 주어 동사는 보통 붙어다니는데 두 PE 벡터는 인접해있기 때문에 유사함)

---

## 5. Summary

Transformer는 Attention 메커니즘을 적용하여 RNN 계열 모델의 한계인 순차적인 구조, 장기 의존성 문제를 해결했다. BERT, GPT 등 현재 거의 모든 SOTA 모델들이 Transformer 구조를 기반으로 만들어졌으며, 8년이 지난 지금까지도 Transformer 구조를 사용하고 있다.  
이 논문에서 제안한 Transformer는 단순히 모델 구조의 혁신뿐 아니라, 딥러닝 패러다임 자체를 전환시킨 모델이라는 점에서 큰 의의가 있다. 후속 연구에서 NLP 뿐만 아니라 VISION 분야에서도 Transformer 구조를 활용하면서 한 단계 더 발전할 수 있는 계기가 되었다.    

---

## 🔗 Reference

[1] [Learning Representations by Back-Propagating Errors (RNN)](https://www.nature.com/articles/323533a0)  
[2] [Long Short-Term Memory (LSTM)](https://www.bioinf.jku.at/publications/older/2604.pdf)  
[3] [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (GRU)](https://arxiv.org/pdf/1406.1078.pdf)  
[4] [MLP-Mixer](https://arxiv.org/pdf/2105.01601)  
