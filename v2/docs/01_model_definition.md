# 01. v2 Model Definition

---

## 1. 모델 정체성

v2 모델은 HateXplain 기반 3-class 혐오표현 탐지 모델이다.

```text
Classes:
0 = hatespeech
1 = offensive
2 = normal
```

핵심 메시지는 다음과 같다.

```text
단어 단서뿐 아니라 맥락 단서까지 함께 학습한 모델이
baseline 대비 분류 성능과 판단 투명성을 개선하는지 검증한다.
```

여기서 중요한 점은 “단어는 중요하지 않다”가 아니다. 혐오표현 탐지에서 단어 단서는 여전히 강한 신호다. 문제는 baseline이 비하어와 표면 단서에 과의존할 수 있다는 점이고, v2는 단어 신호를 유지하면서 맥락 신호를 추가로 학습하는지를 검증한다.

---

## 2. 입력 원칙

모델 입력은 텍스트뿐이다.

| 정보 | 모델 입력 | 학습 supervision | 평가/분석 |
|---|---:|---:|---:|
| `post_tokens` | O | - | O |
| VADER 4d | O, C/D 조건 | - | O |
| `label` | X | O | O |
| `rationale_mask` | X | O, Attention Loss | O |
| `target` | X | 선택, Aux Loss | O |
| `source` | X | X | O |
| `agreement` | X | X | O |

`source`, `target`, `agreement`는 모델 입력으로 넣지 않는다. 이 값들은 subgroup 분석과 사후 검증에만 사용한다.

---

## 3. 모델 구조

### 3.1 공통 encoder

두 사전학습 family를 사용한다.

```text
BERT: bert-base-uncased
RoBERTa: roberta-base
```

각 encoder의 `[CLS]` 또는 pooled representation을 가져와 MLP head에 연결한다.

### 3.2 VADER feature

C/D 조건에서는 텍스트에서 VADER sentiment score를 계산한다.

```text
VADER columns = [pos, neg, neu, compound]
```

이 4차원 벡터를 encoder representation에 concatenate한다.

```text
Without VADER: 768d -> MLP
With VADER:    768d + 4d = 772d -> MLP
```

VADER는 텍스트에서 파생되는 deterministic feature이므로 “모델 입력 단일 소스” 원칙과 충돌하지 않는다.

### 3.3 Classification head

모든 transformer 조건에서 MLP head를 통일한다.

```text
hidden -> Dropout -> Linear(input_dim, 256) -> ReLU -> Linear(256, 3)
```

이렇게 해야 VADER 유무 또는 Attention Loss 유무 외의 구조 차이를 줄일 수 있다.

---

## 4. 8조건 Ablation Matrix

### 4.1 BERT family

| 조건 | Attention Loss | VADER | 설명 |
|---|---:|---:|---|
| A_B | X | X | BERT baseline |
| B_B | O | X | BERT + rationale-aware attention loss |
| C_B | X | O | BERT + VADER |
| D_B | O | O | BERT + attention loss + VADER |

### 4.2 RoBERTa family

| 조건 | Attention Loss | VADER | 설명 |
|---|---:|---:|---|
| A_R | X | X | RoBERTa baseline |
| B_R | O | X | RoBERTa + rationale-aware attention loss |
| C_R | X | O | RoBERTa + VADER |
| D_R | O | O | RoBERTa + attention loss + VADER |

---

## 5. 손실 함수

기본 손실은 3-class classification loss다.

```text
L_cls = CrossEntropy(label, prediction)
```

Attention Loss 조건에서는 human rationale mask와 CLS attention을 정렬한다.

```text
L_attn = BCE(CLS attention distribution, rationale mask)
```

전체 손실은 다음과 같다.

```text
L_total = L_cls + alpha * L_attn
```

Target auxiliary loss는 메인 8조건에는 포함하지 않는다. 필요한 경우 D_B의 부가 실험으로만 분리한다.

```text
L_total_aux = L_cls + alpha * L_attn + beta * L_target
```

---

## 6. Hyperparameter 통제

새 15-seed 실험에서는 조건별 튜닝값을 섞지 않는다.

권장값:

```python
BERT_COMMON = {
    "learning_rate": 2e-5,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 5,
}

ROBERTA_COMMON = {
    "learning_rate": 2e-5,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 5,
}
```

Attention Loss alpha는 B_B에서 선택한 값을 전체 attention 조건에 동일 적용한다.

```text
B_B, D_B, B_R, D_R use the same alpha
```

이 원칙이 깨지면 ablation 원인 해석이 불가능해진다.

---

## 7. 기대되는 주장

강하게 검증하고 싶은 주장은 다음이다.

```text
D_B가 A_B보다 높은 Macro F1을 보이는가?
D_B가 B_B 또는 C_B보다 높은가?
Attention Loss와 VADER가 결합될 때 상호보완 효과가 있는가?
이 효과가 BERT와 RoBERTa family에서 일관적인가?
```

반대로 다음 주장은 데이터가 뒷받침할 때만 쓴다.

```text
VADER 단독으로 성능이 유의하게 오른다.
Attention Loss 단독으로 성능이 유의하게 오른다.
RoBERTa가 BERT보다 항상 강하다.
```

