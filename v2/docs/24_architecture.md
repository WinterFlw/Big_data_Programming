# 24. v2 아키텍처 — Mermaid 다이어그램

> HateSpeachStudy v2_15seed 파이프라인의 전체 구조를 Mermaid로 정리한다.
> GitHub은 ```mermaid 코드블록을 자동 렌더링하므로, 이 문서를 GitHub에서 열면 그림으로 보인다.
> 마지막 업데이트: 2026-05-19

---

## 1. End-to-End 파이프라인 Stage 흐름

`run.sh e2e <stage>` → `pipeline/cli.py` → `pipeline/runner.py`가 각 stage를 호출한다.

```mermaid
flowchart LR
    plan[plan<br/>manifest 생성] --> data[data<br/>split 검증]
    data --> benchmark[benchmark<br/>8조건 x 15seed 학습]
    benchmark --> aggregate[aggregate<br/>통계 집계]
    benchmark --> xaiP[xai-primary<br/>A_B vs D_B]
    benchmark --> xaiD[xai-deep<br/>median seed 사례]
    benchmark --> xaiA[xai-ablation<br/>8조건 경량]
    aggregate --> bundle[xai-bundle<br/>evidence 통합]
    xaiP --> bundle
    xaiD --> bundle
    xaiA --> bundle
    bundle --> report[report<br/>final_report.md/docx]
    bundle --> dashboard[dashboard<br/>index.html]
    aggregate --> report

    status[status<br/>실행 상태 점검] -.매일.-> benchmark

    classDef gate fill:#fef3c7,stroke:#d97706,stroke-width:2px
    classDef heavy fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px
    class benchmark heavy
    class plan,status gate
```

---

## 2. 모듈 아키텍처 — 4개 레이어

`pipeline/`은 얇은 orchestration, `runtime/`은 무거운 모델 코드. adapter가 둘을 잇는다.

```mermaid
flowchart TD
    subgraph CLI["진입점"]
        runsh[run.sh e2e]
        cli[pipeline/cli.py]
        runsh --> cli
    end

    subgraph PIPE["pipeline/ — orchestration (얇은 코드)"]
        runner[runner.py<br/>stage 총괄]
        manifest[manifest.py]
        artifacts[artifacts.py]
        schema[schema.py]
        paths[paths.py]
        trainAdp[training_adapter.py]
        stats[statistics.py]
        xai[xai.py + xai_sampling.py]
        xaiBundle[xai_bundle.py]
        reporting[reporting.py]
    end

    subgraph RUNTIME["runtime/ — 무거운 모델 코드"]
        expCore[experiment_core.py<br/>학습 루프 + AMP]
        expXai[experiment_xai.py<br/>SHAP/LIME/4축]
        utils[utils.py<br/>seed/device/metrics]
    end

    subgraph OPS["scripts/ + CI"]
        daily[scripts/daily.sh]
        gate[scripts/gate_check.py]
        ci[.github/workflows/preflight.yml]
    end

    cli --> runner
    runner --> manifest & artifacts & stats & xai & xaiBundle & reporting
    manifest --> schema
    artifacts --> paths
    runner --> trainAdp
    trainAdp --> expCore
    xai --> expXai
    expCore --> utils
    expXai --> utils
    daily --> gate
    ci --> daily

    classDef pipe fill:#ede9fe,stroke:#8b5cf6
    classDef rt fill:#dbeafe,stroke:#1d4ed8
    classDef ops fill:#dcfce7,stroke:#16a34a
    class runner,manifest,artifacts,schema,paths,trainAdp,stats,xai,xaiBundle,reporting pipe
    class expCore,expXai,utils rt
    class daily,gate,ci ops
```

---

## 3. 8조건 Ablation 매트릭스 + 학습 흐름

`benchmark` stage가 2×2×2 조건 × 15 seed = 120 unit을 학습한다.

```mermaid
flowchart TD
    data[HateXplain 19,192건<br/>3-class + rationale + target] --> split[Stratified split<br/>Train 70 / Val 10 / Test 20]

    split --> matrix{8조건 ablation 매트릭스}

    matrix --> AB["A_B: BERT 베이스"]
    matrix --> BB["B_B: BERT + Attn Loss"]
    matrix --> CB["C_B: BERT + VADER"]
    matrix --> DB["D_B: BERT + Attn + VADER"]
    matrix --> AR["A_R: RoBERTa 베이스"]
    matrix --> BR["B_R: RoBERTa + Attn"]
    matrix --> CR["C_R: RoBERTa + VADER"]
    matrix --> DR["D_R: RoBERTa + Attn + VADER"]

    AB & BB & CB & DB & AR & BR & CR & DR --> seeds["x 15 seed = 120 unit"]
    seeds --> train["train_neural_model<br/>AMP fp16 + cudnn 결정성"]
    train --> outp["metrics.json + history.csv 28컬럼<br/>+ predictions.csv + checkpoint.pt"]

    classDef base fill:#f1f5f9,stroke:#64748b
    classDef improved fill:#fef3c7,stroke:#d97706
    class AB,AR base
    class DB,DR improved
```

손실: `L_total = L_cls + α·L_attn (+ β·L_target)` — α는 B_B에서 그리드 결정 후 D_B/B_R/D_R 동일 적용.

---

## 4. XAI 4축 12지표 구조

`xai-primary/deep/ablation`이 자동 XAI 4축을 계산한다. 인간 라벨 의존을 최소화한 것이 본 연구 결정 카드.

```mermaid
flowchart TD
    ckpt["checkpoint.pt<br/>(A_B vs D_B)"] --> sample["seed 무관 stratified sample<br/>primary 200 / deep 500 / ablation 50"]
    sample --> shap["SHAP / LIME<br/>(CPU 강제)"]

    shap --> axis1["축 1 Attribution<br/>SHAP · LIME · LOO"]
    shap --> axis2["축 2 Faithfulness<br/>Comprehensiveness · Sufficiency · MSS"]
    shap --> axis3["축 3 Context Learning ★<br/>CI · IS · Attention Rollout"]
    shap --> axis4["축 4 Plausibility<br/>Token F1 · IOU · Overlap@5"]

    axis1 & axis2 & axis3 & axis4 --> cache[".cache/ JSON 캐싱"]
    cache --> seedMetric["seed_level_metrics.csv<br/>18컬럼"]
    cache --> sampleMetric["sample_level_metrics.csv"]
    seedMetric --> stability["seed_stability.csv<br/>top-k Jaccard · Spearman"]

    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:2px
    class axis3 decision
```

판정: H3 = 축 3 (CI ↓, IS ↑, MSS ↑)이 통계 유의 + RoBERTa 일관 → 맥락 학습 입증.

---

## 5. 5인 역할 + 산출물 + Git 흐름

서버 실행은 2번 단독, 나머지는 git pull로 결과를 받는다.

```mermaid
flowchart TD
    subgraph SERVER["NVIDIA 서버 (2번만 SSH)"]
        r2["2번 학습 실행<br/>모든 run.sh e2e 명령"]
        r2 --> artifacts["outputs/experiments/v2_15seed/<br/>metrics · CSV · JSON · history.csv"]
    end

    artifacts -->|git push| repo[(GitHub repo<br/>main)]

    repo -->|push마다 자동| ci["CI preflight<br/>✓ / ✗"]

    repo -->|git pull| r1["1번 E2E Gate<br/>daily.sh + GO/STOP"]
    repo -->|git pull| r3["3번 통계<br/>paired t · CI · effect size"]
    repo -->|git pull| r4["4번 XAI<br/>토큰 sanity · 학술 인용"]
    repo -->|git pull| r5["5번 발표<br/>PPT 2종 · 오분류 · Q&A · 시각화 6장"]

    r3 --> p["발표 p17·p18"]
    r4 --> p2["발표 p1·p4·p5·p6"]
    r1 -.D7~D10 페어.-> r5
    r5 --> final["중간발표 PPT + 기말발표 PPT (6/10)"]

    classDef server fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px
    classDef person fill:#ede9fe,stroke:#8b5cf6
    classDef out fill:#dcfce7,stroke:#16a34a
    class r2 server
    class r1,r3,r4,r5 person
    class final,p,p2 out
```

---

## 6. 데이터 흐름 — 모델 입력 단일 소스 원칙

모델은 텍스트(post_tokens)만 입력으로 받는다. rationale·target은 학습 supervision으로만 사용.

```mermaid
flowchart LR
    text["post_tokens<br/>(텍스트)"] --> tok["Tokenizer<br/>WordPiece/BPE"]
    text --> vader["VADER<br/>4d 감성 (C/D 조건)"]
    tok --> bert["BERT/RoBERTa<br/>[CLS] 768d"]
    bert --> concat{"⊕ concat"}
    vader --> concat
    concat --> mlp["MLP<br/>772 또는 768 → 256 → 3"]
    mlp --> mainH["Main Head<br/>3-class"]
    mlp --> auxH["Aux Head<br/>target multi-label<br/>(D_B 부가만)"]

    rationale["rationale 마스크"] -.L_attn supervision.-> mainH
    target["target 라벨"] -.L_target supervision.-> auxH

    mainH --> pred["hate / offensive / normal"]

    classDef inp fill:#dcfce7,stroke:#16a34a
    classDef sup fill:#fee2e2,stroke:#dc2626,stroke-dasharray: 4 4
    class text,vader inp
    class rationale,target sup
```

추론 시: 텍스트만 주어져도 동작 (Aux Head 무시, rationale/target 불필요).

---

## 7. CI 자동화 흐름

push마다 GitHub Actions가 preflight를 자동 실행한다.

```mermaid
flowchart LR
    push["git push origin main"] --> trigger["GitHub Actions<br/>preflight.yml trigger"]
    trigger --> deps["minimal deps 설치<br/>numpy/pandas/scipy/statsmodels"]
    deps --> dailysh["scripts/daily.sh 실행"]
    dailysh --> checks["compile + config + CLI<br/>+ dry-run + scaffolding<br/>+ gate_check 6조건"]
    checks --> result{"결과"}
    result -->|통과| green["✓ 초록"]
    result -->|실패| red["✗ 빨강 → 1번이 로그 확인"]

    classDef pass fill:#dcfce7,stroke:#16a34a
    classDef fail fill:#fee2e2,stroke:#dc2626
    class green pass
    class red fail
```

---

## 8. Stage별 입출력 명세 (INPUT → 처리 → OUTPUT)

각 stage가 정확히 무엇을 받아서 무엇을 내놓는지.

```mermaid
flowchart TD
    cfg[("configs/v2_15seed.json")] --> S_plan

    subgraph S_plan["plan"]
        direction TB
        P1["manifest 생성 + 검증<br/>조건 8개 · seed 15개 확인"]
    end
    S_plan --> O_plan[("manifest.json<br/>execution_status.csv<br/>plan_status.json<br/>+ 디렉토리 트리")]

    rawdata[("data/dataset.json<br/>post_id_divisions.json")] --> S_data
    subgraph S_data["data"]
        D1["split 존재성 · hash · 분포 검증"]
    end
    S_data --> O_data[("split_profile marker")]

    O_plan --> S_bench
    O_data --> S_bench
    subgraph S_bench["benchmark — 8조건 x 15seed = 120 unit"]
        B1["train_neural_model<br/>AMP fp16 + cudnn 결정성"]
    end
    S_bench --> O_bench[("runs/&lt;cond&gt;/seed_&lt;n&gt;/<br/>metrics.json · history.csv 28컬럼<br/>run_config.json · predictions.csv<br/>stdout/stderr.log + checkpoint.pt")]

    O_bench --> S_agg
    subgraph S_agg["aggregate"]
        A1["paired t-test · Holm · Cohen dz<br/>bootstrap CI · ANOVA 2/3-way"]
    end
    S_agg --> O_agg[("benchmark_runs/summary.csv<br/>paired_tests(_holm).csv<br/>anova_2way_bert/roberta.csv<br/>anova_3way.csv")]

    O_bench --> S_xai
    subgraph S_xai["xai-primary / deep / ablation"]
        X1["SHAP/LIME (CPU) + 4축 12지표<br/>seed 무관 sample + 캐싱"]
    end
    S_xai --> O_xai[("xai/primary/*.csv · xai/deep/*<br/>token_highlight.html · cases/*.png<br/>xai/ablation/*.csv · .cache/")]

    O_agg --> S_bundle
    O_xai --> S_bundle
    subgraph S_bundle["xai-bundle"]
        BD1["통계 확증 claim만 추출<br/>source_artifacts 필수"]
    end
    S_bundle --> O_bundle[("evidence_bundle/ 15파일<br/>xai_claims.json · dashboard_bundle.json<br/>token_attributions.jsonl")]

    O_agg --> S_report
    O_bundle --> S_report
    subgraph S_report["report + dashboard"]
        R1["markdown/docx 표 자동 채움<br/>HTML 카드"]
    end
    S_report --> O_report[("reports/final_report.md/docx<br/>dashboard/index.html")]

    classDef io fill:#dcfce7,stroke:#16a34a
    classDef stage fill:#dbeafe,stroke:#1d4ed8
    class cfg,rawdata,O_plan,O_data,O_bench,O_agg,O_xai,O_bundle,O_report io
```

### 입출력 요약표

| Stage | INPUT | 처리 | OUTPUT |
|---|---|---|---|
| **plan** | `configs/v2_15seed.json` | manifest 생성·검증, 디렉토리 생성 | `manifest.json`, `execution_status.csv`, `plan_status.json` |
| **data** | `data/dataset.json` + `post_id_divisions.json` | split 존재·hash·분포 검증 | split_profile marker |
| **benchmark** | manifest + data split | 120 unit 학습 (AMP, cudnn) | unit별 `metrics.json` + `history.csv`(28컬럼) + `run_config.json` + `predictions.csv` + `checkpoint.pt` |
| **aggregate** | `runs/*/metrics.json` 120개 | paired t / Holm / Cohen dz / bootstrap CI / ANOVA | `benchmark_summary.csv` + `paired_tests(_holm).csv` + `anova_*.csv` 3종 |
| **xai-primary** | `checkpoints/*.pt` (A_B, D_B × 15seed) | SHAP/LIME + 4축 12지표 | `seed_level_metrics.csv`(18컬럼) + `sample_level_metrics.csv` + `paired_xai_tests.csv` + `seed_stability.csv` |
| **xai-deep** | `checkpoints/*.pt` (median seed) | 대표 case 분석 + 토큰 하이라이트 | `case_summary.csv` + `token_highlight.html` + `cases/*.png` |
| **xai-ablation** | `checkpoints/*.pt` (8조건 median) | 8조건 4축 경량 비교 | `xai_ablation_metrics.csv` (11컬럼) |
| **xai-bundle** | 위 xai 산출물 + benchmark CSV | claim 추출 (통계 확증만) | `evidence_bundle/` 15파일 |
| **report** | benchmark CSV + evidence_bundle | 표·claim 자동 삽입 | `final_report.md/docx` |
| **dashboard** | benchmark + xai summary | HTML 카드 렌더 | `dashboard/index.html` |

---

## 9. 학습이 하는 일 — benchmark stage 내부 상세

`train_neural_model`이 한 (condition, seed) unit에 대해 수행하는 흐름.

```mermaid
flowchart TD
    inp["post_tokens (텍스트만)"] --> tok["Tokenizer<br/>BERT WordPiece / RoBERTa BPE<br/>max_len 128"]
    tok --> enc["BERT/RoBERTa 12-layer<br/>→ [CLS] 768d"]

    inp --> vader["VADER SentimentIntensityAnalyzer<br/>→ pos/neg/neu/compound 4d"]

    enc --> concat{"C/D 조건?"}
    vader --> concat
    concat -->|"C/D (VADER 사용)"| d772["⊕ → 772d"]
    concat -->|"A/B (VADER 미사용)"| d768["768d"]
    d772 --> mlp["Dropout → Linear(→256) → ReLU"]
    d768 --> mlp
    mlp --> mainH["Main Head Linear(256→3)"]
    mlp --> auxH["Aux Head Linear(256→T)<br/>D_B 부가 실험만"]

    mainH --> lcls["L_cls = CrossEntropy<br/>(class weight balanced)"]
    rationale["rationale 마스크"] --> lattn["α·L_attn = BCE<br/>(CLS attention vs rationale)<br/>B/D 조건만"]
    target["target 라벨"] --> ltgt["β·L_target = BCE<br/>D_B 부가만"]

    lcls --> total["L_total = L_cls + α·L_attn + β·L_target"]
    lattn --> total
    ltgt --> total

    total --> bw["backward — GradScaler fp16<br/>grad clip max_norm=1.0"]
    bw --> opt["AdamW + linear warmup 0.1"]
    opt --> epoch{"epoch 끝?"}
    epoch -->|"진행"| tok
    epoch -->|"종료"| save["best val Macro F1 → checkpoint.pt 저장<br/>매 epoch history.csv 28컬럼 기록"]

    classDef inp fill:#dcfce7,stroke:#16a34a
    classDef sup fill:#fee2e2,stroke:#dc2626
    classDef loss fill:#fef3c7,stroke:#d97706
    class inp,vader inp
    class rationale,target sup
    class lcls,lattn,ltgt,total loss
```

**핵심 원칙**: 모델 입력은 **텍스트(post_tokens)만**. rationale·target은 손실 함수의 supervision 신호로만 쓰이고 모델 입력에 안 들어감. VADER 4d는 텍스트에서 자동 계산되는 파생 피처라 "텍스트만 입력" 원칙 위배 아님.

---

## 9.1 BERT + VADER 조건 (C_B / D_B) 전체 상세 — STEP 0~4

C/D 조건은 `HybridTextDataset` + `TransformerConditionClassifier(use_vader=True)`를 쓴다. 텐서 shape 단위로 추적.

> 변수: `B`=batch(64) · `L`=seq(max 128) · `768`=BERT hidden · `4`=VADER · `256`=MLP hidden · `3`=labels · `T`=num_targets(~24, D_B 부가만)

```mermaid
flowchart TD
    subgraph S0["STEP 0 — VADER 사전 추출 (학습 전 1회)"]
        t0["text"] --> vsa["SentimentIntensityAnalyzer<br/>.polarity_scores()"]
        vsa --> vp["vader_features.pkl<br/>(N, 4) — pos·neg·neu·compound"]
    end

    subgraph S1["STEP 1 — HybridTextDataset (입력 구성)"]
        pt["post_tokens"] --> tk["tokenizer<br/>is_split_into_words=True<br/>truncation max_len=128"]
        tk --> iid["input_ids (B,L)"]
        tk --> am["attention_mask (B,L)"]
        tk --> wi["word_ids()"]
        rr["rationale 원본"] --> wi
        wi --> rm["rationale_mask (B,L)<br/>단어→서브워드 복제"]
        vp --> vf["vader (B,4)"]
        lb["labels (B,)"]
        tg["target_multilabel (B,T)"]
    end

    subgraph S2["STEP 2 — TransformerConditionClassifier.forward (use_vader=True)"]
        iid --> enc["BERT 12-layer encoder<br/>output_attentions=True (D조건)"]
        am --> enc
        enc --> cls["pooler_output / [CLS]<br/>(B, 768)"]
        enc --> at["attentions[-1]<br/>(B, heads, L, L)"]
        cls --> cc["⊕ concat<br/>(B,768) + (B,4) = (B,772)"]
        vf --> cc
        cc --> dp["Dropout(0.1)"]
        dp --> hd["Linear(772 → 256) + ReLU<br/>(B, 256)"]
        hd --> mh["Main Head<br/>Linear(256→3) → logits (B,3)"]
        hd --> ah["Aux Head<br/>Linear(256→T) → target_logits (B,T)<br/>D_B 부가만"]
    end

    subgraph S3["STEP 3 — 손실 계산"]
        mh --> lc["L_cls = CrossEntropy<br/>class weight balanced"]
        lb --> lc
        at --> ca["cls_attention<br/>mean(heads)[:,0,:] = (B,L)"]
        ca --> la["α·L_attn = BCE<br/>cls_attention vs rationale_mask<br/>(D 조건만)"]
        rm --> la
        ah --> lt["β·L_target = BCE_with_logits<br/>(D_B 부가만)"]
        tg --> lt
        lc --> tot["L_total = L_cls + α·L_attn + β·L_target"]
        la -.D 조건.-> tot
        lt -.D_B 부가.-> tot
    end

    subgraph S4["STEP 4 — 역전파 + 출력"]
        tot --> bw["GradScaler fp16 backward<br/>clip_grad_norm 1.0 → AdamW + warmup"]
        bw --> ep{"epoch 끝?"}
        ep -->|진행| tk
        ep -->|종료| out["checkpoint.pt (best val Macro F1)<br/>history.csv 28컬럼 / metrics.json<br/>predictions.csv"]
        mh --> pred["추론 출력<br/>hate / offensive / normal 확률"]
    end

    classDef pre fill:#e0e7ff,stroke:#6366f1
    classDef inp fill:#dcfce7,stroke:#16a34a
    classDef sup fill:#fee2e2,stroke:#dc2626
    classDef loss fill:#fef3c7,stroke:#d97706
    classDef model fill:#dbeafe,stroke:#1d4ed8
    class t0,vsa,vp pre
    class iid,am,vf,cls,cc inp
    class rr,rm,lb,tg sup
    class lc,la,lt,tot loss
    class enc,hd,mh,ah model
```

**C_B vs D_B 차이 (위 다이어그램에서)**

| 항목 | C_B (BERT+VADER) | D_B (BERT+VADER+Attn) |
|---|---|---|
| VADER concat | ✓ (B,772) | ✓ (B,772) |
| `output_attentions` | False | True |
| `α·L_attn` 점선 | 없음 | CLS attention ↔ rationale BCE |
| `β·L_target` 점선 | 없음 | D_B 부가 실험만 (듀얼 헤드) |
| `L_total` | `L_cls` | `L_cls + α·L_attn (+ β·L_target)` |

색 범례: 보라=VADER 사전추출 / 초록=모델 입력 / 빨강=supervision(입력 아님) / 파랑=모델 레이어 / 노랑=손실.

---

## 10. 입출력 검증 게이트 — 어디서 무엇을 막나

각 stage가 잘못된 입력을 어떻게 걸러내고 출력을 어떻게 보장하는지.

```mermaid
flowchart TD
    plan["plan"] --> v1{"manifest 검증"}
    v1 -->|"조건 오타 / seed 중복 / 키 누락"| stop1["STOP — validate_manifest 에러"]
    v1 -->|"통과"| ok1["120 unit 계획 확정"]

    ok1 --> bench["benchmark"]
    bench --> v2{"unit_status 판정"}
    v2 -->|"metrics+history+config 다 있음"| completed["completed"]
    v2 -->|"stderr에 fatal marker"| failed["failed → failed_runs.csv"]
    v2 -->|"그 외"| planned["planned"]

    completed --> agg["aggregate"]
    agg --> v3{"빈 입력?"}
    v3 -->|"completed 0개"| emptyCsv["헤더만 CSV (graceful)"]
    v3 -->|"completed 있음"| realCsv["실제 통계 row"]

    completed --> xai["xai-primary/deep/ablation"]
    xai --> v4{"checkpoint 존재?"}
    v4 -->|"없음"| skipXai["빈 CSV graceful skip"]
    v4 -->|"있음"| realXai["SHAP/LIME 실행"]

    realXai --> v5{"sample 결정성"}
    v5 -->|"md5 불일치"| stop5["STOP — seed 의존 버그"]
    v5 -->|"md5 일치"| okXai["4축 메트릭 산출"]

    realCsv --> gate["Full Run Gate 6조건"]
    okXai --> gate
    gate --> v6{"6/6 PASS?"}
    v6 -->|"FAIL"| stopGate["STOP — fix 책임자 멘션"]
    v6 -->|"PASS"| go["GO — full 120 학습"]

    classDef stop fill:#fee2e2,stroke:#dc2626
    classDef pass fill:#dcfce7,stroke:#16a34a
    class stop1,stop5,stopGate,failed stop
    class ok1,completed,go,okXai pass
```

검증 규칙 요약:

| Stage | 입력 검증 | 출력 보장 |
|---|---|---|
| plan | `validate_manifest` — 조건 오타·seed 중복·키 누락 시 STOP | 120 unit `execution_status.csv` |
| benchmark | unit별 `metrics+history+config` 3개 다 있어야 completed | `failed_runs.csv` / `completed_runs.csv` 자동 분리 |
| aggregate | completed 0개여도 헤더만 CSV (graceful) | schema 컬럼 고정 — downstream 안 깨짐 |
| xai-* | checkpoint 없으면 빈 CSV skip | sample md5 일치 검사 (seed 무관성) |
| xai-bundle | 통계 미확증 결과는 strong claim 금지 | 모든 claim에 `source_artifacts` 필수 |
| Full Run Gate | 6조건 자동 점검 (`gate_check.py`) | 6/6 PASS만 full 학습 GO |

---

## 부록 — 디렉토리 구조

```mermaid
flowchart TD
    v2["v2/"] --> pipeline["pipeline/<br/>orchestration 12파일"]
    v2 --> runtime["runtime/<br/>모델 코드 7파일"]
    v2 --> scripts["scripts/<br/>daily.sh · gate_check.py"]
    v2 --> configs["configs/<br/>v2_15seed.json"]
    v2 --> docs["docs/<br/>설계·역할·agent_tasks"]
    v2 --> gh[".github/workflows/<br/>preflight.yml"]
    v2 --> outputs["outputs/experiments/v2_15seed/<br/>benchmark · xai · reports · dashboard"]

    docs --> roleGuides["role_guides/<br/>5인 Word 가이드"]
    docs --> agentTasks["agent_tasks/<br/>00~23 에이전트 지시서"]
```

---

문서 끝.
