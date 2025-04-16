---
title: "[논문 리뷰] Attention Is All You Need"
date: 2025-04-15 15:30:00 +0900
categories: [Paper]
tags: [Transformer, Attention, NLP]
math: true
toc: true
layout: single
---

## 📝 논문 정보

- **제목**: Attention Is All You Need  
- **저자**: Vaswani et al.  
- **학회/연도**: NeurIPS 2017  
- **링크**: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)

---

## 1. Introduction

이 논문에서 제안한 Transformer가 나온 이후로 Language 분야는 엄청난 발전을 이뤘다. 그 이후에는 Vision, Vision-Language Models(VLM) 등 많은 분야에서 활용하기 시작하면서 많은 분야의 성능 향상에 큰 기여를 했다. 우선 이번 글에서는 논문에 나온대로 Language 분야를 기준으로 설명하려한다.  

기존에 주를 이루던 Language 모델들은 RNN, LSTM 기반이었는데 몇 가지 문제점이 있었다.  
- 순차적으로 처리해서 병렬화 불가능  
- 긴 문장에서 과거 정보가 잘 사라짐  

이 논문에서는 Self-Attention을 도입하여 위 문제를 해결했고, 이어서 Self-Attention, Transformer Architecture에 대해 자세히 설명하려한다.  

---

## 2. Self-Attention
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

정리하자면, Self-Attention은 입력 시퀀스 내 각 단어가 다른 모든 단어를 얼마나 "Attend" 해야 하는지를 계산하는 메커니즘이다. 이 과정에서 각 단어는 Query, Key, Value로 변환된 후, Query와 Key의 유사도를 계산하고, 이를 바탕으로 Value의 가중합을 구한다. 그 결과, 각 단어는 문장 전체를 참조하여 의미를 재구성한 새로운 벡터로 표현된다.  

---

## 3. Transformer Architecture
![Transformer Architecture](/assets/images/2025-04-15-transformer/1.png)  
_(이미지 기준 왼쪽은 Encoder, 오른쪽은 Decoder)_  

아주 간략하게 요약하면, Encoder는 입력 문장을 벡터로 변환하고, Decoder는 벡터를 출력 문장으로 바꿔주는 역할을 한다.  

Encoder부터 살펴보면 아래와 같은 순서로 입력된다.  
Input + Posional Encoding -> Multi-Head Attention -> Feed Forward  
**[Multi-Head Attention -> Feed Forward]** 부분은 그림에서 * N 표시가 되어있는데, 논문에서는 6번 반복해서 쌓았다고 한다.  
그리고 Decoder에서는 **Masked-Multi-Head Attention**이 추가되어 있는데, 이 부분은 학습 시에 미래 시점의 단어를 알지 못하도록 하는 역할을 한다.  

**Multi-Head Attention은 말 그대로 Attention이 여러 개 있는 형태인데, 왜 여러 개를 썼을까?**  
이유는 각각의 Attention이 다양한 정보를 학습하게 하기위해서다. 예를 들어 첫 번째 Attention은 주어와 동사의 관계를 파악하고, 두 번째 Attention은 근처에 있는 단어들간의 관계를 파악하는 등 여러 정보를 학습하기를 **기대**하고 여러 개의 Attention을 사용한 것이다. 이는 CNN에서 Feature map을 추출할 때 필터를 하나만 쓰는 것이 아니라, 다수 채널의 필터를 사용하는 것과 유사하다고 볼 수 있다.

**Posinal Encoding(PE)은 왜 필요할까?**  
사실 이 논문을 처음 보는 것도 아니고, 이번에 다시 보면서 가장 크게 얻은 부분이 바로 PE다.  
물론 혼자서 깨달은 것이 아님을 먼저 밝히며 내 친구이자 멘토인 [Yedarm Seong](https://github.com/mybirth0407)에게 감사를 표한다.  

**우선, PE는 단순히 각 Embedding의 위치 정보를 전달하기 위한 장치다.**  

Transformer는 PE없이는 RNN, LSTM과 다르게 순서를 알 수가 없다. **그런데 나는 이말이 제일 이해가 안갔다!**  
"왜 순서를 알 수가 없지? 분명 Input Embedding을 만들 때 순서를 지켜서 만들 수 있을텐데", 그리고 "순서를 지키면 당연히 모델이 Input Embedding의 순차적인 정보를 이해할 수 있을 것"이라고 생각했다.  
그런데 수식을 다시 살펴보면 $QK^T$ 내적할 때 문장에서 K가 Q앞에 있는지, 뒤에 있는지 알려주는 부분이 있나? 당연히 없다...  

만약 아래 내용을 더 빨리 알았다면, Input Embedding을 만들 때 순서를 지키고 말고 생각할 것 없이 PE가 왜 필요한지 더 빨리 알았을 것 같다.

Multi-layer Perceptron(MLP)에서 Input Layer의 Patch 단위로 순서를 섞어도 결국 매우 비슷한 Output을 낸다고 한다. 이 내용은 [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf)의 결과를 참고한 내용이다.  

![MLP-Mixer Results](/assets/images/2025-04-15-transformer/2.png)  

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

![pos-3-4](/assets/images/2025-04-15-transformer/3.png)  
_(그림 3. pos=3, pos=4 일때 차원 변화에 따른 PE값 비교)_  

![pos-3-500](/assets/images/2025-04-15-transformer/4.png)  
_(그림 4. pos=3, pos=500 일때 차원 변화에 따른 PE값 비교)_  

sin/cos PE의 특징 중 하나는 가까운 위치에 있는 토큰끼리는 차이가 적고, 먼 위치에 있는 토큰끼리는 차이가 크다. 그림 3에 있는 두 그래프는 매우 유사하고, 그림 4에 있는 두 그래프는 차이가 상대적으로 커보인다.  
따라서 거리가 가깝고 멀고에 대한 상대적인 거리도 판별이 가능하다.

![dim](/assets/images/2025-04-15-transformer/5.png)  
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

## 4. Contribution
이 논문의 핵심적인 기여는 다음과 같다.
- RNN 계열 모델의 한계인 순차적 구조를 Attention mechanism 적용으로 완전히 벗어남  
- BERT, GPT 등 현재 거의 모든 SOTA 모델들이 Transformer 아키텍처를 기반으로 만들어짐
- sin/cos 기반의 Positional Encoding 방식 제안

이 논문에서 제안한 Transformer는 단순히 모델 구조의 혁신뿐 아니라, 딥러닝 패러다임 자체를 전환시킨 모델이라는 점에서 큰 의의가 있다.  

---

## 🔗 Reference
[1] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)  
[2] [MLP-Mixer](https://arxiv.org/pdf/2105.01601)