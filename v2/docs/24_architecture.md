# 24. v2 아키텍처 — Mermaid 다이어그램

> HateSpeachStudy v2_15seed 파이프라인 구조. 모든 다이어그램은 **좌→우 직선 흐름**으로 큰 줄기가 한눈에 보이도록 그렸다.
> GitHub은 ```mermaid 코드블록을 자동 렌더링한다.
> 마지막 업데이트: 2026-05-19

---

## 1. 메인 파이프라인 (큰 줄기 한 줄)

```mermaid
flowchart LR
    plan --> data --> benchmark --> aggregate --> xai --> bundle --> report --> dashboard
```

곁가지는 따로:

```mermaid
flowchart LR
    benchmark[benchmark] --> agg[aggregate 통계]
    benchmark --> xai[xai 3종]
    agg --> bundle[xai-bundle]
    xai --> bundle
    bundle --> report[report]
    bundle --> dash[dashboard]
    agg --> report
```

---

## 2. 모듈 레이어 (CLI → pipeline → runtime)

```mermaid
flowchart LR
    runsh[run.sh] --> cli[cli.py] --> runner[runner.py]
    runner --> pipe[pipeline/<br/>orchestration 12파일]
    pipe --> adapter[training_adapter<br/>· xai adapter]
    adapter --> runtime[runtime/<br/>experiment_core · experiment_xai]
    daily[scripts/daily.sh] --> gate[gate_check.py]
    ci[GitHub Actions] --> daily
```

---

## 3. 학습 흐름 (데이터 → 8조건 → 산출물)

```mermaid
flowchart LR
    raw[HateXplain<br/>19,192건] --> split[Stratified split<br/>70/10/20] --> cond[8조건 x 15seed<br/>= 120 unit] --> train[train_neural_model<br/>AMP fp16] --> out[checkpoint.pt<br/>+ metrics + history.csv]
```

8조건 매트릭스:

```mermaid
flowchart LR
    bert[BERT] --> AB[A_B 베이스] & BB[B_B +Attn] & CB[C_B +VADER] & DB[D_B +Attn+VADER]
    roberta[RoBERTa] --> AR[A_R] & BR[B_R] & CR[C_R] & DR[D_R]
```

---

## 4. BERT + VADER 모델 내부 (forward 한 줄)

```mermaid
flowchart LR
    text[텍스트] --> tok[Tokenizer] --> bert[BERT 12-layer] --> cls["[CLS] 768d"] --> cat["⊕ concat 772d"] --> mlp["MLP 772→256"] --> head[Main Head 256→3] --> logits[logits 3-class]
    text --> vader[VADER 4d] --> cat
```

A/B 조건은 VADER 없이 768d 직행. C/D 조건만 `⊕ concat`으로 772d.

---

## 5. 손실 계산 (출력 → 3개 손실 → 합산)

```mermaid
flowchart LR
    logits[logits] --> lcls[L_cls<br/>CrossEntropy]
    attn["CLS attention"] --> lattn[α·L_attn<br/>BCE vs rationale]
    auxlogits[target_logits] --> ltgt[β·L_target<br/>BCE]
    lcls --> total[L_total]
    lattn --> total
    ltgt --> total
    total --> back[backward → AdamW]
```

`L_total = L_cls + α·L_attn + β·L_target` — L_attn은 B/D 조건, L_target은 D_B 부가만.

---

## 6. XAI 4축 흐름 (checkpoint → 메트릭)

```mermaid
flowchart LR
    ckpt[checkpoint.pt] --> sample[stratified sample<br/>seed 무관] --> shap[SHAP / LIME] --> axis[4축 12지표] --> metric[seed_level_metrics<br/>+ token_highlight.html]
```

4축:

```mermaid
flowchart LR
    shap[SHAP/LIME] --> a1[1 Attribution] --> a2[2 Faithfulness] --> a3[3 Context Learning ★] --> a4[4 Plausibility]
```

(축은 병렬 계산이지만 읽기 순서대로 나열 — 3축 Context Learning이 본 연구 결정 카드.)

---

## 7. 5인 역할 + Git 흐름

```mermaid
flowchart LR
    s2[2번 서버 학습] -->|git push| repo[(GitHub main)]
    repo -->|git pull| s1[1번 Gate]
    repo -->|git pull| s3[3번 통계]
    repo -->|git pull| s4[4번 XAI]
    repo -->|git pull| s5[5번 발표]
    repo --> ci[CI ✓/✗]
```

---

## 8. CI 자동화 흐름

```mermaid
flowchart LR
    push[git push] --> trigger[GitHub Actions] --> daily[daily.sh 실행] --> check[8조건 점검] --> result[✓ 초록 / ✗ 빨강]
```

---

## 9. Stage별 입출력 (INPUT → stage → OUTPUT)

```mermaid
flowchart LR
    cfg[configs.json] --> plan[plan] --> mani[manifest.json]
    mani --> bench[benchmark] --> runs[runs/.../<br/>metrics·history·checkpoint]
    runs --> agg[aggregate] --> csv[benchmark·anova CSV]
    runs --> xai[xai-*] --> xout[xai/primary·deep·ablation]
    csv --> bundle[xai-bundle] --> ev[evidence_bundle 15파일]
    xout --> bundle
    ev --> rep[report] --> final[final_report.md/docx]
    ev --> dash[dashboard] --> html[index.html]
```

| Stage | INPUT | OUTPUT |
|---|---|---|
| plan | `configs/v2_15seed.json` | `manifest.json`, `execution_status.csv` |
| benchmark | manifest + data split | `metrics.json` + `history.csv`(28컬럼) + `checkpoint.pt` |
| aggregate | `runs/*/metrics.json` | `benchmark_summary.csv` + `paired_tests.csv` + `anova_*.csv` |
| xai-primary | `checkpoints/*.pt` | `seed_level_metrics.csv`(18컬럼) + `seed_stability.csv` |
| xai-deep | `checkpoints/*.pt` | `case_summary.csv` + `token_highlight.html` + `cases/*.png` |
| xai-ablation | `checkpoints/*.pt` | `xai_ablation_metrics.csv` |
| xai-bundle | xai 산출물 + benchmark CSV | `evidence_bundle/` 15파일 |
| report | benchmark CSV + bundle | `final_report.md/docx` |
| dashboard | benchmark + xai | `dashboard/index.html` |

---

## 10. BERT + VADER (C_B / D_B) STEP 0~4 상세

학습 전체를 단계별 좌→우로. 텐서 shape: `B`=batch · `L`=seq(128) · `768`=BERT · `4`=VADER · `256`=MLP · `3`=labels.

**STEP 0~1 — VADER 추출 + 입력 구성**

```mermaid
flowchart LR
    text[text] --> vsa[polarity_scores] --> vp[vader 4d]
    ptok[post_tokens] --> tk[tokenizer] --> iid[input_ids B,L]
    tk --> am[attention_mask B,L]
    tk --> rm[rationale_mask B,L]
```

**STEP 2 — forward**

```mermaid
flowchart LR
    iid[input_ids] --> bert[BERT 12-layer] --> cls["[CLS] B,768"]
    vp[vader B,4] --> cat["⊕ B,772"]
    cls --> cat --> mlp["Linear 772→256 + ReLU"] --> mh[Main Head → logits B,3]
    mlp --> ah[Aux Head → target B,T<br/>D_B 부가만]
```

**STEP 3~4 — 손실 + 출력**

```mermaid
flowchart LR
    logits[logits] --> lc[L_cls] --> tot[L_total]
    rm[rationale] --> la[α·L_attn] --> tot
    tot --> bw[backward AMP] --> ck[checkpoint + history.csv]
```

C_B는 `α·L_attn` 빼고 `L_total = L_cls`. D_B는 포함.

---

## 11. 입출력 검증 게이트 (어디서 막나)

```mermaid
flowchart LR
    plan[plan] --> g1{manifest OK?} -->|"NO"| stop1[STOP]
    g1 -->|"YES"| bench[benchmark] --> g2{unit 완료?}
    g2 -->|"failed"| fail[failed_runs.csv]
    g2 -->|"completed"| gate{Gate 6/6?}
    gate -->|"NO"| stop2[STOP — fix 멘션]
    gate -->|"YES"| go[GO — full 학습]
```

| Stage | 검증 |
|---|---|
| plan | 조건 오타·seed 중복·키 누락 → STOP |
| benchmark | metrics+history+config 3개 다 있어야 completed |
| aggregate / xai | 빈 입력이면 헤더만 CSV (graceful) |
| Full Run Gate | 6조건 자동 (`gate_check.py`) — 6/6만 GO |

---

## 12. 디렉토리 구조

```mermaid
flowchart LR
    v2[v2/] --> pipeline[pipeline/]
    v2 --> runtime[runtime/]
    v2 --> scripts[scripts/]
    v2 --> docs[docs/]
    v2 --> outputs[outputs/experiments/]
    docs --> rg[role_guides/ 5인 Word]
    docs --> at[agent_tasks/ 00~24]
```

---

문서 끝.
