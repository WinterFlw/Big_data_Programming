# 24. 연구 파이프라인 — 모델의 흐름과 철학

> 이 문서는 파일·디렉토리·CI 같은 엔지니어링이 아니라, **모델이 무엇을 왜 학습하는지**, 연구가 어떤 질문에서 출발해 어떻게 답하는지를 담는다.
> GitHub은 ```mermaid 코드블록을 자동 렌더링한다. 마지막 업데이트: 2026-05-19

---

## 0. 한 줄 — 무엇을 묻고 무엇으로 답하나

> 비하어에 기대는 모델은 맥락 혐오를 놓친다.
> **단어 단서를 유지하면서 맥락 단서까지 함께 학습한 모델**이 분류 성능과 판단 투명성 모두에서 나아짐을, 통제된 ablation과 자동 XAI로 입증한다.

---

## 1. 연구 서사 — 문제에서 결론까지

```mermaid
flowchart LR
    problem[문제<br/>비하어 매칭은<br/>맥락 혐오를 놓침] --> diagnose[진단 H1<br/>베이스 BERT가<br/>단어에 과의존]
    diagnose --> treat[처방<br/>Attention Loss<br/>+ VADER]
    treat --> verify[검증 H2·H3<br/>8조건 ablation<br/>+ 자동 XAI 4축]
    verify --> conclude[결론<br/>단어 신호 유지<br/>+ 맥락 단서 추가 학습]
```

---

## 2. 베이스 모델의 문제 — 왜 고쳐야 하나

```mermaid
flowchart LR
    sent["문장: 'those people should go back'"] --> base[베이스 BERT]
    base --> miss["비하어 없음 → normal 오판"]
    slur["문장: 'all X must die'"] --> base
    base --> easy["비하어 die 매칭 → hate 정답"]
    miss --> issue[단어 단서에 과의존<br/>맥락 단서 활용 부족]
    easy --> issue
```

베이스는 비하어가 있으면 잘 맞히고, 비하어 없는 맥락 혐오는 놓친다. **단어 자체가 신호가 아닌 게 아니라, 단어에 *과의존*하는 게 문제.**

---

## 3. 두 처방 — 서로 다른 층위에 개입

```mermaid
flowchart LR
    text[입력 텍스트] --> v[VADER 감성 4d<br/>입력층 개입] --> model[모델]
    text --> model
    rationale[인간 rationale<br/>학습 손실층 개입] --> model
    model --> learn[단어 + 맥락<br/>함께 학습]
```

| 처방 | 어느 층위 | 무엇을 가르치나 | 근거 |
|---|---|---|---|
| **VADER 감성 피처** | 입력층 (concat) | "이 문장의 감정 온도" — 맥락 단서 | Cheng 2022 선행연구 기반 사전 가설 |
| **Attention Loss** | 학습 손실층 | "어디를 봐야 하는가" — 인간 근거 토큰에 정렬 | Mathew 2021 rationale을 평가→학습으로 |

두 처방은 직교한다 — VADER는 모델 구조, Attention Loss는 학습 손실. 그래서 ablation으로 따로 떼어 측정할 수 있다.

---

## 4. 모델이 실제로 학습하는 것 (D_B 기준)

```mermaid
flowchart LR
    text[텍스트] --> bert[BERT가 문맥 표현 학습] --> meaning["[CLS] 문장 의미 768d"]
    text --> senti[VADER가 감정 신호 추출] --> meaning2[감성 4d]
    meaning --> fuse[의미 + 감성 결합]
    meaning2 --> fuse
    fuse --> judge[혐오 / 공격 / 일반 판단]
    rationale[인간 근거] --> guide[판단의 시선을<br/>근거 토큰에 정렬] --> judge
```

모델은 세 신호를 함께 본다 — **BERT의 문맥 의미**, **VADER의 감정 온도**, 그리고 학습 중 **인간 근거가 가르치는 시선의 방향**. 추론할 땐 텍스트만 있으면 된다 (rationale은 학습 때만).

---

## 5. 8조건 ablation — 두 처방을 따로 떼어 측정

```mermaid
flowchart LR
    AB["A_B 베이스<br/>처방 없음"] --> BB["B_B<br/>+ Attention Loss"]
    AB --> CB["C_B<br/>+ VADER"]
    BB --> DB["D_B<br/>+ 둘 다"]
    CB --> DB
```

```mermaid
flowchart LR
    q1["Attention Loss 효과?"] --> a1["B_B - A_B"]
    q2["VADER 효과?"] --> a2["C_B - A_B"]
    q3["둘이 시너지?"] --> a3["D_B - B_B - C_B + A_B"]
    q4["RoBERTa도 같나?"] --> a4["A_R~D_R 반복"]
```

같은 데이터·시드·하이퍼파라미터에서 **한 번에 한 처방만 바꿔** 주효과·상호작용·강건성을 분리한다. 이게 "통제된 ablation".

---

## 6. 무엇이 좋아졌는지 — 두 가지 차원으로 본다

```mermaid
flowchart LR
    DB[D_B 개선 모델] --> perf[분류 성능<br/>Macro F1 향상?]
    DB --> trans[판단 투명성<br/>맥락 단서를 보는가?]
    perf --> h2[H2 검증]
    trans --> h3[H3 검증]
```

성능만 좋아진 게 아니라 **모델이 판단하는 방식**도 바뀌었는지 본다 — 그게 이 연구의 핵심 주장.

---

## 7. XAI 4축 — 모델이 단어에 기대나, 맥락을 보나

```mermaid
flowchart LR
    ask["모델은 무엇을 보고 판단하나?"] --> ax1["축1 Attribution<br/>어떤 토큰을 지목하나"]
    ax1 --> ax2["축2 Faithfulness<br/>그 토큰이 진짜 근거인가"]
    ax2 --> ax3["축3 Context Learning ★<br/>한두 단어 집중 vs 여러 토큰 분산"]
    ax3 --> ax4["축4 Plausibility<br/>인간 근거와 닮았나"]
```

3축 Context Learning이 본 연구의 결정 카드 — **인간이 만든 비하어 목록 같은 카테고리에 기대지 않고**, 모델 내부 토큰 동역학만으로 맥락 학습을 정량화한다.

---

## 8. 단어 의존 → 맥락 의존, 어떻게 드러나나

```mermaid
flowchart LR
    ab["A_B 단어 의존"] --> ci_hi["CI 높음<br/>소수 토큰 집중"]
    ab --> is_lo["IS 낮음<br/>토큰 독립적"]
    ab --> mss_sm["MSS 작음<br/>1~2 토큰이면 충분"]
    db["D_B 맥락 학습"] --> ci_lo["CI 낮음<br/>여러 토큰 분산"]
    db --> is_hi["IS 높음<br/>토큰 시너지"]
    db --> mss_lg["MSS 큼<br/>여러 토큰 필요"]
```

**비유** — A_B는 단거리 선수(혼자 뛴다), D_B는 축구팀(11명 패스로 골). 단어 의존 모델은 토큰 하나하나가 독립이고, 맥락 의존 모델은 토큰들이 함께 작동한다.

---

## 9. 가설 검증 사슬 — H1에서 H3까지

```mermaid
flowchart LR
    h1["H1 진단<br/>A_B 단어 과의존 확인"] --> h2["H2 개선<br/>D_B 분류 성능 향상<br/>ΔF1 > 0"]
    h2 --> h3["H3 입증<br/>D_B 자동 XAI 4축<br/>맥락 학습 통계 유의"]
    h3 --> robust["RoBERTa에서도 일관"]
    robust --> claim["주장 성립<br/>단어 + 맥락 함께 학습"]
```

H1 진단에서 출발해 H2(성능)·H3(판단 방식)를 거치고, RoBERTa에서도 같은 패턴이 나오면 주장이 강건해진다. **단일 지표가 아니라 다축 일관성으로 판정** — 한 지표만 좋으면 우연일 수 있으니까.

---

## 10. 전체를 한 흐름으로

```mermaid
flowchart LR
    data[HateXplain<br/>텍스트 + 인간 근거] --> train["학습<br/>BERT + VADER + Attention Loss"]
    train --> models[8조건 모델]
    models --> measure["측정<br/>분류 성능 + 자동 XAI 4축"]
    measure --> answer["답<br/>단어 신호 유지하며<br/>맥락 단서까지 학습"]
```

---

## 부록 — 절대 하지 않는 주장

- "혐오는 단어가 아니라 맥락이다" (이분법 과장) — 단어도 신호다. *과의존*이 문제일 뿐.
- "XAI 진단으로 VADER를 골랐다" — VADER는 Cheng 2022 선행연구 기반 사전 가설. XAI는 사후 검증.
- "순환적 프레임워크 / 피드백 루프" — 본 연구는 "가설 → 통제된 ablation → XAI 사후 검증"의 단방향 과학적 검증.

---

문서 끝.
