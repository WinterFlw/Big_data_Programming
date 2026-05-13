# 12. Code Commenting Guide

> 목적: v2 코드에 주석을 어떻게 달지 정한다. 팀원이 에이전트를 쓰더라도 같은 기준으로 설명을 남기게 하기 위한 문서다.

---

## 1. 기본 원칙

v2 코드는 단순히 돌아가는 코드가 아니라 팀원이 읽고 나눠 맡을 코드다.
따라서 주석은 아래를 설명해야 한다.

```text
이 파일이 왜 존재하는가?
이 함수가 어느 stage의 책임인가?
어떤 산출물 계약을 지키는가?
서버 실행 비용을 줄이기 위해 어떤 안전장치가 있는가?
다음 담당자가 어디를 구현해야 하는가?
```

---

## 2. 좋은 주석

좋은 주석은 의도를 설명한다.

```python
# The full benchmark is 8 x 15 = 120 GPU runs. Keep subset selectors here so
# server smoke tests can run A_B/D_B seed 42 before the expensive full run.
```

좋은 주석은 위험을 설명한다.

```python
# Primary XAI must reuse the same sample IDs across seeds. Resampling per seed
# would make seed-stability metrics meaningless.
```

좋은 주석은 다음 구현 지점을 설명한다.

```python
# Next implementation point:
#   - call the v2-local train_neural_model adapter
#   - redirect checkpoints into unit.run_dir
#   - honor --resume before launching a GPU job
```

---

## 3. 나쁜 주석

아래처럼 코드 그대로를 반복하는 주석은 피한다.

```python
# Increment i by 1
i += 1
```

아래처럼 근거 없이 안심시키는 주석도 피한다.

```python
# This definitely works.
```

---

## 4. v2에서 주석이 반드시 필요한 곳

```text
server-expensive stage
resume/force/only-failed logic
condition x seed mapping
output path decision
statistics assumption
XAI sampling rule
checkpoint loading/saving
v2/runtime experiment_core.py adapter boundary
```

---

## 5. 에이전트에게 줄 주석 지시

팀원이 에이전트를 쓸 때 아래 문장을 추가한다.

```text
코드를 수정할 때는 함수의 목적, 서버 실행 비용과 관련된 안전장치, output path가 왜 그렇게 정해졌는지 주석으로 남겨주세요.
단순히 코드 한 줄을 한국어로 번역하는 주석은 피하고, 다음 담당자가 실수하지 않도록 의도와 제약을 설명해주세요.
```

---

## 6. 리뷰 기준

리뷰어는 아래를 확인한다.

```text
비싼 서버 작업 앞에 smoke/dry-run 의도가 설명되어 있는가?
run_id output root를 지키는 이유가 설명되어 있는가?
statistics/XAI의 가정이 코드 근처에 적혀 있는가?
다음 구현자가 이어받을 위치가 명확한가?
```
