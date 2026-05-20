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
