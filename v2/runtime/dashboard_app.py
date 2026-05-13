"""
HateSpeachStudy -- FastAPI Dashboard
=====================================
단일 파일 FastAPI 앱으로, Chart.js 시각화와 함께 HTML 대시보드를 제공한다.
v2/outputs 디렉토리에서 데이터를 읽어 10개 탭으로 구성된 종합 대시보드를 표시한다.

실행: cd v2/runtime && python3 dashboard_app.py
접속: http://localhost:8501
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
RUNTIME_DIR = Path(__file__).resolve().parent
BASE = RUNTIME_DIR.parent
OUTPUTS = BASE / "outputs"
REPORTS = OUTPUTS / "reports"
BEST_MODELS_PATH = REPORTS / "best_models.json"
V2_PLAYGROUND_ORDER = ["A_B", "B_B", "C_B", "D_B", "A_R", "B_R", "C_R", "D_R", "D_B+Target"]
COMPAT_PLAYGROUND_ORDER = ["BERT+MLP", "BERT-base", "BERT+VADER", "RoBERTa+VADER"]

# experiment_core.py, experiment_xai.py를 임포트하기 위해 경로 추가
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

app = FastAPI(title="HateSpeachStudy Dashboard")

# ---------------------------------------------------------------------------
# Playground: 모델 로딩 + 추론 (lazy loading -- 최초 요청 시에만 로드)
# ---------------------------------------------------------------------------
_playground_models: dict = {}  # 캐시: 한 번 로드하면 재사용
_playground_device = None


def _load_best_models_registry() -> dict:
    if not BEST_MODELS_PATH.exists():
        return {}
    try:
        return json.loads(BEST_MODELS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else BASE / path


def _available_playground_models() -> list[str]:
    registry = _load_best_models_registry()
    ordered = [name for name in [*V2_PLAYGROUND_ORDER, *COMPAT_PLAYGROUND_ORDER] if name in registry]
    return [
        name
        for name in ordered
        if (checkpoint_path := _resolve_path(registry[name].get("checkpoint_path"))) is not None
        and checkpoint_path.exists()
    ]


def _get_device():
    """MPS > CUDA > CPU 순으로 자동 감지"""
    global _playground_device
    if _playground_device is not None:
        return _playground_device
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _playground_device = torch.device("mps")
    elif torch.cuda.is_available():
        _playground_device = torch.device("cuda")
    else:
        _playground_device = torch.device("cpu")
    print(f"[playground] device = {_playground_device}", flush=True)
    return _playground_device


def _load_playground_model(model_key: str):
    """best_models.json에 기록된 v2 checkpoint에서 모델 로드 (lazy, 캐시됨)."""
    if model_key in _playground_models:
        return _playground_models[model_key]

    import torch
    from experiment_core import (
        TransformerCLSClassifier,
        TransformerMLPClassifier,
        HybridSentimentClassifier,
        TransformerConditionClassifier,
    )
    from transformers import AutoTokenizer

    device = _get_device()

    registry = _load_best_models_registry()
    if model_key not in registry:
        return None

    record = registry[model_key]
    checkpoint_path = _resolve_path(record.get("checkpoint_path"))
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    hyperparams = record.get("hyperparams", {})
    dropout = float(hyperparams.get("dropout") or 0.1)
    hidden_dim = int(hyperparams.get("mlp_hidden") or 256)
    target_labels = hyperparams.get("target_labels") or []

    if model_key == "BERT-base":
        model_name = "bert-base-uncased"
        model = TransformerCLSClassifier(model_name=model_name, dropout=dropout)
        model_type = "transformer"
    elif model_key == "BERT+VADER":
        model_name = "bert-base-uncased"
        model = HybridSentimentClassifier(model_name=model_name, dropout=dropout, hidden_dim=hidden_dim)
        model_type = "hybrid"
    elif model_key == "RoBERTa+VADER":
        model_name = "roberta-base"
        model = HybridSentimentClassifier(model_name=model_name, dropout=dropout, hidden_dim=hidden_dim)
        model_type = "hybrid"
    elif model_key == "BERT+MLP":
        model_name = "bert-base-uncased"
        model = TransformerMLPClassifier(model_name=model_name, dropout=dropout, hidden_dim=hidden_dim)
        model_type = "mlp"
    elif model_key in V2_PLAYGROUND_ORDER:
        model_name = "roberta-base" if model_key.endswith("_R") else "bert-base-uncased"
        use_vader = model_key.startswith(("C_", "D_"))
        model = TransformerConditionClassifier(
            model_name=model_name,
            use_vader=use_vader,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_targets=len(target_labels) if record.get("use_target_aux") else 0,
        )
        model_type = "hybrid" if use_vader else "mlp"
    else:
        return None

    print(f"[playground] Loading {model_key}...", flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # sdpa attention은 output_attentions=True를 지원하지 않음
    # eager attention으로 전환하여 Attention Heatmap이 정상 작동하도록 함
    if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        model.encoder.config._attn_implementation = "eager"
        # 각 레이어의 self-attention도 eager로 전환
        for layer in model.encoder.encoder.layer:
            layer.attention.self._attn_implementation = "eager"
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    bundle = {
        "model": model,
        "tokenizer": tokenizer,
        "model_type": model_type,
        "device": device,
    }
    _playground_models[model_key] = bundle
    print(f"[playground] {model_key} loaded on {device}", flush=True)
    return bundle


def _predict_single(bundle: dict, text: str, run_lime: bool = False) -> dict:
    """단일 텍스트 추론 -> {label, probabilities, vader_scores, attention, lime}"""
    import torch
    import numpy as np

    device = bundle["device"]
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    # 토크나이징
    encoded = tokenizer(
        [text], truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # VADER 점수
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    vader_scores = {"pos": vs["pos"], "neg": vs["neg"], "neu": vs["neu"], "compound": vs["compound"]}

    # ---- Attention 추출 ----
    token_attention = []
    with torch.no_grad():
        # encoder에 output_attentions=True를 넘겨서 attention weight 추출
        # return_dict=True를 명시해야 .attentions 속성이 정상 반환됨
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        # 마지막 레이어의 attention: (batch, num_heads, seq_len, seq_len)
        attentions = getattr(encoder_outputs, "attentions", None)
        if attentions and len(attentions) > 0:
            last_attention = attentions[-1]
            # 모든 head 평균 -> (batch, seq_len, seq_len)
            avg_attention = last_attention.mean(dim=1)
            # [CLS] 토큰(index 0)이 다른 토큰에 주는 attention -> (seq_len,)
            cls_attention = avg_attention[0, 0, :].cpu().numpy()
            # 전체 토큰간 attention 행렬 저장 (토큰 상호작용 히트맵용)
            full_attention_matrix = avg_attention[0].cpu().numpy()
        else:
            # attention을 가져올 수 없는 경우 균등 분포로 대체
            seq_len = input_ids.shape[1]
            cls_attention = np.ones(seq_len) / seq_len
            full_attention_matrix = np.ones((seq_len, seq_len)) / seq_len

        # pooler output으로 logits 계산
        pooled = getattr(encoder_outputs, "pooler_output", None)
        if pooled is None:
            pooled = encoder_outputs.last_hidden_state[:, 0, :]

        if bundle["model_type"] == "hybrid":
            vader_tensor = torch.tensor(
                [[vs["pos"], vs["neg"], vs["neu"], vs["compound"]]],
                dtype=torch.float32,
            ).to(device)
            combined = torch.cat([pooled, vader_tensor], dim=1)
            combined = model.dropout(combined)
            combined = model.hidden(combined)
            combined = model.relu(combined)
            logits = model.out(combined)
        elif bundle["model_type"] == "mlp":
            # TransformerMLPClassifier: dropout -> hidden -> relu -> out
            pooled = model.dropout(pooled)
            pooled = model.hidden(pooled)
            pooled = model.relu(pooled)
            logits = model.out(pooled)
        else:
            # TransformerCLSClassifier: dropout -> classifier
            pooled = model.dropout(pooled)
            logits = model.classifier(pooled)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # 토큰 복원 + attention 매핑
    token_ids = input_ids[0].cpu().tolist()
    tokens_raw = tokenizer.convert_ids_to_tokens(token_ids)
    mask = attention_mask[0].cpu().tolist()

    # special token 제외, subword 병합
    merged_tokens = []
    merged_attention = []
    for i, (tok, m) in enumerate(zip(tokens_raw, mask)):
        if m == 0:
            continue
        # 특수 토큰 건너뛰기 ([CLS], [SEP], <s>, </s>)
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
            continue
        att_val = float(cls_attention[i])
        # subword 병합 (##prefix 또는 G prefix for RoBERTa)
        if tok.startswith("##"):
            if merged_tokens:
                merged_tokens[-1] += tok[2:]
                merged_attention[-1] = max(merged_attention[-1], att_val)
            continue
        # RoBERTa: subword는 G로 시작하지 않음
        if tok.startswith("Ġ"):
            tok = tok[1:]
        merged_tokens.append(tok)
        merged_attention.append(att_val)

    # attention 정규화 (0~1)
    if merged_attention:
        max_att = max(merged_attention)
        min_att = min(merged_attention)
        rng = max_att - min_att if max_att > min_att else 1.0
        norm_attention = [(a - min_att) / rng for a in merged_attention]
    else:
        norm_attention = []

    token_attention = [
        {"token": t, "attention": round(a, 4), "raw": round(r, 6)}
        for t, a, r in zip(merged_tokens, norm_attention, merged_attention)
    ]

    # ---- 토큰 상호작용 행렬 구성 (subword 병합 기준) ----
    # 원본 토큰 인덱스 -> 병합된 토큰 인덱스 매핑
    raw_to_merged = []  # raw index -> merged index
    merged_idx = -1
    for i, (tok, m) in enumerate(zip(tokens_raw, mask)):
        if m == 0 or tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
            raw_to_merged.append(-1)
            continue
        if tok.startswith("##"):
            # subword는 이전 병합 토큰에 귀속
            raw_to_merged.append(merged_idx)
            continue
        if tok.startswith("Ġ"):
            pass  # RoBERTa 단어 시작 표시, 새 토큰
        merged_idx += 1
        raw_to_merged.append(merged_idx)

    n_merged = len(merged_tokens)
    if n_merged > 0 and n_merged <= 40:
        # 병합 기준으로 attention 행렬 재구성 (max pooling)
        merged_matrix = np.zeros((n_merged, n_merged))
        for i in range(len(raw_to_merged)):
            for j in range(len(raw_to_merged)):
                mi, mj = raw_to_merged[i], raw_to_merged[j]
                if mi >= 0 and mj >= 0 and mi < n_merged and mj < n_merged:
                    merged_matrix[mi][mj] = max(merged_matrix[mi][mj], float(full_attention_matrix[i][j]))
        # 행 단위 정규화 (각 토큰의 attention 분포를 0~1로)
        for i in range(n_merged):
            row_max = merged_matrix[i].max()
            row_min = merged_matrix[i].min()
            if row_max > row_min:
                merged_matrix[i] = (merged_matrix[i] - row_min) / (row_max - row_min)
        interaction_matrix = [[round(float(merged_matrix[i][j]), 4) for j in range(n_merged)] for i in range(n_merged)]
    else:
        # 토큰이 40개 초과면 히트맵 생략 (가독성)
        interaction_matrix = None

    label_names = ["hatespeech", "offensive", "normal"]
    pred_idx = int(np.argmax(probs))

    result = {
        "label": label_names[pred_idx],
        "label_idx": pred_idx,
        "probabilities": {name: round(float(probs[i]), 4) for i, name in enumerate(label_names)},
        "vader_scores": {k: round(v, 4) for k, v in vader_scores.items()},
        "token_attention": token_attention,
        "interaction_matrix": interaction_matrix,
        "interaction_tokens": merged_tokens if interaction_matrix else None,
    }

    # ---- LIME 설명 (선택적) ----
    if run_lime:
        try:
            from lime.lime_text import LimeTextExplainer
            lime_explainer = LimeTextExplainer(class_names=label_names)

            def _lime_predict_fn(texts):
                """LIME용 배치 예측 함수"""
                all_probs = []
                for t in texts:
                    enc = tokenizer([t], truncation=True, padding=True, max_length=128, return_tensors="pt")
                    ids = enc["input_ids"].to(device)
                    msk = enc["attention_mask"].to(device)
                    with torch.no_grad():
                        if bundle["model_type"] == "hybrid":
                            v = analyzer.polarity_scores(t)
                            vt = torch.tensor([[v["pos"], v["neg"], v["neu"], v["compound"]]], dtype=torch.float32).to(device)
                            lo = model(ids, msk, vt)
                        else:
                            lo = model(ids, msk)
                        p = torch.softmax(lo, dim=-1).cpu().numpy()[0]
                    all_probs.append(p)
                return np.array(all_probs)

            explanation = lime_explainer.explain_instance(
                text, _lime_predict_fn,
                labels=[0, 1, 2], num_features=8, num_samples=300,
            )
            # 예측 라벨에 대한 설명
            available = list(explanation.local_exp.keys())
            lime_label = pred_idx if pred_idx in available else (available[0] if available else 0)
            lime_weights = explanation.as_list(label=lime_label)
            result["lime"] = [
                {"token": tok, "weight": round(w, 4)}
                for tok, w in lime_weights
            ]
        except Exception as e:
            result["lime_error"] = str(e)

    return result

# 정적 파일 마운트 -- 이미지 서빙용
app.mount("/static/xai", StaticFiles(directory=str(OUTPUTS / "xai")), name="xai_static")
app.mount("/static/eda", StaticFiles(directory=str(OUTPUTS / "reports" / "eda")), name="eda_static")
app.mount("/static/runs", StaticFiles(directory=str(OUTPUTS / "runs")), name="runs_static")

# ---------------------------------------------------------------------------
# 데이터 로딩 헬퍼
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------------------------

@app.get("/api/learning_curves")
def api_learning_curves():
    """runs/*/seed_*/history.csv 데이터를 JSON으로 반환"""
    result = {}
    runs_dir = OUTPUTS / "runs"
    if not runs_dir.exists():
        return JSONResponse({})
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        result[model_name] = {}
        for seed_dir in sorted(model_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            history_path = seed_dir / "history.csv"
            if history_path.exists():
                rows = _read_csv(history_path)
                for r in rows:
                    for k, v in r.items():
                        try:
                            r[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                result[model_name][seed_dir.name] = rows
    return JSONResponse(result)


@app.get("/api/tuning_history")
def api_tuning_history():
    """tuning/transformer_tuning_log.csv 데이터를 JSON으로 반환"""
    rows = _read_csv(OUTPUTS / "tuning" / "transformer_tuning_log.csv")
    for r in rows:
        try:
            r["val_macro_f1"] = float(r["val_macro_f1"])
        except (ValueError, TypeError, KeyError):
            pass
    return JSONResponse(rows)


@app.post("/api/predict")
async def api_predict(request: Request):
    """텍스트를 받아 선택된 모델들로 추론 결과 반환"""
    body = await request.json()
    text = body.get("text", "").strip()
    models = body.get("models") or _available_playground_models()[:3]
    run_lime = body.get("run_lime", False)

    if not text:
        return JSONResponse({"error": "텍스트를 입력해주세요"}, status_code=400)

    results = {}
    device_name = str(_get_device())

    for model_key in models:
        try:
            bundle = _load_playground_model(model_key)
            if bundle is None:
                results[model_key] = {"error": f"체크포인트 없음: {model_key}"}
                continue
            results[model_key] = _predict_single(bundle, text, run_lime=run_lime)
        except Exception as e:
            results[model_key] = {"error": str(e)}

    return JSONResponse({"text": text, "device": device_name, "results": results})


@app.get("/api/predict/status")
def api_predict_status():
    """Playground 상태: 로드된 모델, 디바이스"""
    available_models = _available_playground_models()
    return JSONResponse({
        "device": str(_get_device()),
        "loaded_models": list(_playground_models.keys()),
        "available_models": available_models,
        "registry": str(BEST_MODELS_PATH),
    })


# ---------------------------------------------------------------------------
# Data Explorer API -- hatexplain_vader.csv 기반 페이지네이션 검색
# ---------------------------------------------------------------------------
_data_explorer_cache: list[dict] | None = None


def _load_explorer_data() -> list[dict]:
    """데이터 탐색기용 데��터를 한 번만 로드하여 캐싱한다.
    CSV가 있으면 CSV를 쓰고, 없으면 dataset.json에서 직접 구성한다."""
    global _data_explorer_cache
    if _data_explorer_cache is not None:
        return _data_explorer_cache

    # 1) CSV 후보 시도
    csv_candidates = [
        BASE / "data" / "hatexplain_vader.csv",
        BASE / "data" / "hatexplain_processed.csv",
        BASE / "data" / "dataset.csv",
    ]
    for p in csv_candidates:
        if p.exists():
            _data_explorer_cache = _read_csv(p)
            return _data_explorer_cache

    # 2) dataset.json에서 직접 구성 (majority vote 포함)
    json_path = BASE / "data" / "dataset.json"
    if json_path.exists():
        raw = _read_json(json_path)
        rows = []
        for post_id, entry in raw.items():
            tokens = entry.get("post_tokens", [])
            text = " ".join(tokens)
            # majority vote로 라벨 결정
            labels = [a.get("label", "") for a in entry.get("annotators", [])]
            from collections import Counter
            label_counts = Counter(labels)
            majority_label, majority_count = label_counts.most_common(1)[0] if label_counts else ("unknown", 0)
            if majority_count < 2:
                continue  # 합의 안 된 샘플�� 제외
            # 타겟 수집
            targets = set()
            for a in entry.get("annotators", []):
                for t in a.get("target", []):
                    if t != "None":
                        targets.add(t)
            rows.append({
                "post_id": post_id,
                "text": text,
                "label": majority_label,
                "target": ", ".join(sorted(targets)) if targets else "None",
                "word_count": str(len(tokens)),
            })
        _data_explorer_cache = rows
        return _data_explorer_cache

    _data_explorer_cache = []
    return _data_explorer_cache


@app.get("/api/data_explorer")
def api_data_explorer(q: str = "", label: str = "", page: int = 1, limit: int = 20):
    """데이터 탐색기 -- 텍스트 검색 + 라벨 필터 + 페이지네이션"""
    rows = _load_explorer_data()
    filtered = rows

    # 라벨 필터
    if label and label.lower() != "all":
        filtered = [r for r in filtered if r.get("label", "").lower() == label.lower()]

    # 텍스트 검색 (대소문자 무시)
    if q:
        q_lower = q.lower()
        filtered = [r for r in filtered if q_lower in r.get("text", "").lower()]

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    page_rows = filtered[start:end]

    # 각 행에서 필요한 필드만 추출
    results = []
    for r in page_rows:
        text = r.get("text", r.get("post_tokens", ""))
        word_count = r.get("word_count", len(text.split()) if text else 0)
        results.append({
            "text": text[:300],  # 미리보기용 300자 제한
            "label": r.get("label", r.get("final_label", "unknown")),
            "vader_compound": r.get("vader_compound", r.get("compound", "")),
            "target": r.get("target", ""),
            "word_count": int(word_count) if word_count else 0,
        })

    return JSONResponse({
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total + limit - 1) // limit),
        "results": results,
    })


# ---------------------------------------------------------------------------
# 메인 대시보드 HTML
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    # 데이터 로딩
    benchmark = _read_csv(OUTPUTS / "reports" / "benchmark_summary.csv")
    significance = _read_csv(OUTPUTS / "reports" / "significance_tests.csv")
    freeze = _read_csv(OUTPUTS / "reports" / "freeze_study.csv")
    eda = _read_json(OUTPUTS / "reports" / "eda" / "eda_summary.json")
    xai_summary = _read_json(OUTPUTS / "xai" / "xai_summary.json")
    overlap = _read_csv(OUTPUTS / "xai" / "overlap_at_5.csv")
    case_summary = _read_csv(OUTPUTS / "xai" / "case_summary.csv")
    rationale_overlap = _read_csv(OUTPUTS / "xai" / "rationale_overlap.csv")
    tuning_best = _read_json(OUTPUTS / "tuning" / "transformer_tuning_best.json")
    tuning_log = _read_csv(OUTPUTS / "tuning" / "transformer_tuning_log.csv")

    # 벤치마크 데이터 정렬 (F1 내림차순)
    for row in benchmark:
        try:
            row["_f1"] = float(row.get("macro_f1_mean", 0))
        except (ValueError, TypeError):
            row["_f1"] = 0
    benchmark.sort(key=lambda x: x["_f1"], reverse=True)

    # 케이스 이미지 경로 수집
    cases_dir = OUTPUTS / "xai" / "cases"
    case_images = sorted(cases_dir.glob("case_*.png")) if cases_dir.exists() else []

    # freeze 데이터 처리
    frozen_rows = [r for r in freeze if "Frozen" in r.get("model", "")]
    finetuned_rows = [r for r in freeze if "Fine-tuned" in r.get("model", "") or "Fine" in r.get("model", "")]
    frozen_f1_vals = []
    finetuned_f1_vals = []
    for r in frozen_rows:
        try:
            frozen_f1_vals.append(float(r["macro_f1"]))
        except (ValueError, KeyError):
            pass
    for r in finetuned_rows:
        try:
            finetuned_f1_vals.append(float(r["macro_f1"]))
        except (ValueError, KeyError):
            pass
    frozen_mean = sum(frozen_f1_vals) / len(frozen_f1_vals) if frozen_f1_vals else 0
    finetuned_mean = sum(finetuned_f1_vals) / len(finetuned_f1_vals) if finetuned_f1_vals else 0

    # XAI summary 안전 접근
    xai_baseline_f1 = xai_summary.get("baseline_macro_f1", 0)
    xai_improved_f1 = xai_summary.get("improved_macro_f1", 0)
    xai_baseline_overlap = xai_summary.get("baseline_overlap_mean", 0)
    xai_improved_overlap = xai_summary.get("improved_overlap_mean", 0)
    xai_baseline_ge60 = xai_summary.get("baseline_overlap_ge_60", 0)
    xai_improved_ge60 = xai_summary.get("improved_overlap_ge_60", 0)
    xai_fixed = xai_summary.get("fixed_error_count", 0)

    # Human Rationale 비교 메트릭
    xai_baseline_rat_shap = xai_summary.get("baseline_rationale_shap_mean")
    xai_improved_rat_shap = xai_summary.get("improved_rationale_shap_mean")
    xai_baseline_rat_lime = xai_summary.get("baseline_rationale_lime_mean")
    xai_improved_rat_lime = xai_summary.get("improved_rationale_lime_mean")
    xai_baseline_rat_ge50 = xai_summary.get("baseline_rationale_ge_50", 0)
    xai_improved_rat_ge50 = xai_summary.get("improved_rationale_ge_50", 0)
    xai_rat_sample_count = xai_summary.get("rationale_sample_count", 0)

    # JSON 직렬화 헬퍼
    def js(obj):
        return json.dumps(obj, ensure_ascii=False)

    # 벤치마크 JS 데이터
    bm_slim = []
    for row in benchmark:
        bm_slim.append({
            "model": row.get("model", ""),
            "macro_f1_mean": row.get("macro_f1_mean", "0"),
            "macro_f1_std": row.get("macro_f1_std", "0"),
            "accuracy_mean": row.get("accuracy_mean", "0"),
            "auroc_mean": row.get("auroc_mean", "0"),
            "per_class_f1.hatespeech_mean": row.get("per_class_f1.hatespeech_mean", "0"),
            "per_class_f1.offensive_mean": row.get("per_class_f1.offensive_mean", "0"),
            "per_class_f1.normal_mean": row.get("per_class_f1.normal_mean", "0"),
        })

    sig_slim = []
    for row in significance:
        sig_slim.append({
            "model_a": row.get("model_a", ""),
            "model_b": row.get("model_b", ""),
            "p_value": row.get("p_value", "1"),
            "significant": row.get("significant", "False"),
        })

    tuning_slim = []
    for row in tuning_log:
        tuning_slim.append({
            "model": row.get("model", ""),
            "parameter": row.get("parameter", ""),
            "candidate": row.get("candidate", ""),
            "val_macro_f1": row.get("val_macro_f1", "0"),
        })

    overlap_slim = []
    for row in overlap:
        overlap_slim.append({
            "model": row.get("model", ""),
            "sample_id": row.get("sample_id", ""),
            "overlap_at_5": row.get("overlap_at_5", "0"),
        })

    rationale_slim = []
    for row in rationale_overlap:
        rationale_slim.append({
            "model": row.get("model", ""),
            "xai_method": row.get("xai_method", ""),
            "post_id": row.get("post_id", ""),
            "overlap": row.get("overlap", "0"),
            "matched": row.get("matched", ""),
        })

    # ------------------------------------------------------------------
    # 벤치마크 테이블 HTML 생성
    # ------------------------------------------------------------------
    benchmark_table_rows = ""
    for row in benchmark:
        model = row.get("model", "")
        f1_d = row.get("macro_f1_display", "")
        acc_d = row.get("accuracy_display", "")
        auroc_d = row.get("auroc_display", "")
        try:
            hate_f1 = f"{float(row.get('per_class_f1.hatespeech_mean', 0)):.4f}"
        except (ValueError, TypeError):
            hate_f1 = "N/A"
        try:
            off_f1 = f"{float(row.get('per_class_f1.offensive_mean', 0)):.4f}"
        except (ValueError, TypeError):
            off_f1 = "N/A"
        try:
            nor_f1 = f"{float(row.get('per_class_f1.normal_mean', 0)):.4f}"
        except (ValueError, TypeError):
            nor_f1 = "N/A"
        style = "style='color:var(--accent-green);font-weight:700'" if model == "RoBERTa+VADER" else ""
        benchmark_table_rows += f"""<tr>
            <td {style}>{model}</td>
            <td {style}>{f1_d}</td>
            <td>{acc_d}</td>
            <td>{auroc_d}</td>
            <td>{hate_f1}</td>
            <td>{off_f1}</td>
            <td>{nor_f1}</td>
          </tr>"""

    # ------------------------------------------------------------------
    # 통계 검정 테이블 HTML 생성
    # ------------------------------------------------------------------
    sig_table_rows = ""
    for row in significance:
        sig = row.get("significant", "False") == "True"
        badge_cls = "badge-green" if sig else "badge-gray"
        badge_text = "Yes" if sig else "No"
        try:
            pval = float(row.get("p_value", 1))
            pval_str = f"{pval:.6f}"
        except (ValueError, TypeError):
            pval_str = row.get("p_value", "")
        try:
            cd = float(row.get("cohens_d", 0))
            cd_str = f"{cd:.4f}"
        except (ValueError, TypeError):
            cd_str = row.get("cohens_d", "")
        sig_table_rows += f"""<tr>
            <td>{row.get('model_a','')}</td>
            <td>{row.get('model_b','')}</td>
            <td>{row.get('mean_diff','')}</td>
            <td>{row.get('t_statistic','')}</td>
            <td>{pval_str}</td>
            <td>{cd_str}</td>
            <td><span class="badge {badge_cls}">{badge_text}</span></td>
          </tr>"""

    # ------------------------------------------------------------------
    # 튜닝 최적 파라미터 테이블
    # ------------------------------------------------------------------
    tuning_best_rows = ""
    for model_name, params in tuning_best.items():
        tuning_best_rows += f"""<tr>
          <td>{model_name}</td>
          <td>{params.get('learning_rate','')}</td>
          <td>{params.get('dropout','')}</td>
          <td>{params.get('batch_size','')}</td>
          <td>{params.get('epochs','')}</td>
        </tr>"""

    # ------------------------------------------------------------------
    # Freeze 테이블
    # ------------------------------------------------------------------
    freeze_table_rows = ""
    for row in freeze:
        try:
            f1_val = f"{float(row.get('macro_f1', 0)):.4f}"
        except (ValueError, TypeError):
            f1_val = row.get("macro_f1", "")
        try:
            acc_val = f"{float(row.get('accuracy', 0)):.4f}"
        except (ValueError, TypeError):
            acc_val = row.get("accuracy", "")
        freeze_table_rows += f"""<tr>
            <td>{row.get('model','')}</td>
            <td>{row.get('seed','')}</td>
            <td>{f1_val}</td>
            <td>{acc_val}</td>
          </tr>"""

    # ------------------------------------------------------------------
    # 어휘 중첩 테이블
    # ------------------------------------------------------------------
    vocab_table_rows = ""
    for v in eda.get("vocabulary_overlap", []):
        jac = v.get("jaccard_similarity", 0)
        interp_ko = "매우 높은 중첩" if jac > 0.7 else ("높은 중첩" if jac > 0.6 else "중간 중첩")
        interp_en = "Very high overlap" if jac > 0.7 else ("High overlap" if jac > 0.6 else "Moderate overlap")
        color = "var(--accent-red)" if jac > 0.7 else ("var(--accent-orange)" if jac > 0.6 else "var(--text-secondary)")
        vocab_table_rows += f"""<tr>
            <td>{v.get('class_a','')}</td>
            <td>{v.get('class_b','')}</td>
            <td style="color:{color};font-weight:700">{jac:.4f}</td>
            <td><span class="ko">{interp_ko}</span><span class="en">{interp_en}</span></td>
          </tr>"""

    # ------------------------------------------------------------------
    # 텍스트 길이 테이블
    # ------------------------------------------------------------------
    textlen_table_rows = ""
    for ts in eda.get("text_length_stats", []):
        textlen_table_rows += f"""<tr>
          <td>{ts.get('class','')}</td>
          <td>{ts.get('n_samples','')}</td>
          <td>{ts.get('word_mean','')}</td>
          <td>{ts.get('word_median','')}</td>
          <td>{ts.get('token_mean','')}</td>
          <td>{ts.get('token_max','')}</td>
        </tr>"""

    # ------------------------------------------------------------------
    # 클래스 분포 테이블
    # ------------------------------------------------------------------
    class_dist_rows = ""
    for cd in eda.get("class_distribution", []):
        cls_color = {"hatespeech": "var(--accent-red)", "offensive": "var(--accent-orange)", "normal": "var(--accent-green)"}.get(cd.get("class", ""), "")
        class_dist_rows += f"""<tr>
          <td style="color:{cls_color};font-weight:600">{cd.get('class','')}</td>
          <td>{cd.get('count',0):,}</td>
          <td>{cd.get('ratio_pct',0)}%</td>
        </tr>"""

    # ------------------------------------------------------------------
    # N-gram 테이블 (bigram)
    # ------------------------------------------------------------------
    ngram_html = ""
    for label_name in ["hatespeech", "offensive", "normal"]:
        bg_list = eda.get("ngram_top_bigrams", {}).get(label_name, [])[:8]
        cls_color = {"hatespeech": "#ff6b7a", "offensive": "#ffb347", "normal": "#00e5b0"}.get(label_name, "")
        ngram_html += f'<div><h4 style="color:{cls_color};margin-bottom:6px">{label_name}</h4><table class="data-table"><thead><tr><th>#</th><th>Bigram</th><th>Count</th></tr></thead><tbody>'
        for i, bg in enumerate(bg_list, 1):
            ngram_html += f'<tr><td>{i}</td><td style="font-family:monospace">{bg.get("ngram","")}</td><td>{bg.get("count",0):,}</td></tr>'
        ngram_html += '</tbody></table></div>'

    # ------------------------------------------------------------------
    # Rationale 분포 테이블
    # ------------------------------------------------------------------
    rationale_dist_rows = ""
    for rd in eda.get("rationale_distribution", []):
        cls_color = {"hatespeech": "var(--accent-red)", "offensive": "var(--accent-orange)", "normal": "var(--accent-green)"}.get(rd.get("class", ""), "")
        rationale_dist_rows += f"""<tr>
          <td style="color:{cls_color};font-weight:600">{rd.get('class','')}</td>
          <td>{rd.get('samples_with_rationale',0):,}</td>
          <td>{rd.get('mean_rationale_tokens','')}</td>
          <td>{rd.get('median_rationale_tokens','')}</td>
          <td>{rd.get('max_rationale_tokens','')}</td>
        </tr>"""

    # ------------------------------------------------------------------
    # VADER KS test 테이블
    # ------------------------------------------------------------------
    vader_ks_rows = ""
    for ks in eda.get("vader_separability", []):
        ks_stat = ks.get("ks_statistic", 0)
        p_val = ks.get("p_value", 1)
        sig = "Yes" if p_val < 0.05 else "No"
        sig_color = "var(--accent-green)" if p_val < 0.05 else "var(--accent-red)"
        vader_ks_rows += f"""<tr>
          <td>{ks.get('class_a','')}</td>
          <td>{ks.get('class_b','')}</td>
          <td>{ks_stat:.4f}</td>
          <td>{p_val:.2e}</td>
          <td style="color:{sig_color};font-weight:600">{sig}</td>
        </tr>"""

    # ------------------------------------------------------------------
    # 케이스 요약 테이블
    # ------------------------------------------------------------------
    case_table_rows = ""
    for cs in case_summary:
        case_table_rows += f"""<tr>
            <td>{cs.get('sample_id','')}</td>
            <td>{cs.get('category','')}</td>
            <td style="font-size:0.8rem">{cs.get('baseline_top_tokens','')}</td>
            <td style="font-size:0.8rem">{cs.get('improved_top_tokens','')}</td>
            <td>{cs.get('baseline_overlap_at_5','')}</td>
            <td>{cs.get('improved_overlap_at_5','')}</td>
          </tr>"""

    # ------------------------------------------------------------------
    # 케이스 이미지 갤러리
    # ------------------------------------------------------------------
    case_gallery_html = ""
    for img_path in case_images:
        case_num = img_path.stem.replace("case_", "")
        case_gallery_html += f"""<div class="gallery-item">
        <img src="/static/xai/cases/{img_path.name}" alt="Case {case_num}" loading="lazy">
        <div class="caption">Case {case_num}</div>
      </div>"""

    # ------------------------------------------------------------------
    # 완전한 HTML 조립
    # ------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HateSpeachStudy Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
/* ================================================================
   기본 리셋 및 다크 테마
   ================================================================ */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg-darkest: #0a0b10;
  --bg-dark: #12141f;
  --bg-card: #1a1d2e;
  --accent-blue: #7c8aff;
  --accent-green: #00e5b0;
  --accent-red: #ff6b7a;
  --accent-orange: #ffb347;
  --accent-purple: #b57aff;
  --text-primary: #e8eaf6;
  --text-secondary: #9ca3c4;
  --border-subtle: rgba(124, 138, 255, 0.15);
  --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
}}

body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg-darkest);
  color: var(--text-primary);
  line-height: 1.6;
  min-height: 100vh;
}}

body.light-mode {{
  --bg-darkest: #f0f2f5;
  --bg-dark: #ffffff;
  --bg-card: #ffffff;
  --text-primary: #1a1a2e;
  --text-secondary: #555;
  --border-subtle: rgba(0,0,0,0.1);
  --shadow-card: 0 2px 12px rgba(0,0,0,0.08);
}}

body .en {{ display: none; }}
body .ko {{ display: inline; }}
body.en-mode .en {{ display: inline; }}
body.en-mode .ko {{ display: none; }}
body .en-block {{ display: none; }}
body .ko-block {{ display: block; }}
body.en-mode .en-block {{ display: block; }}
body.en-mode .ko-block {{ display: none; }}

.top-bar {{
  background: var(--bg-dark);
  border-bottom: 1px solid var(--border-subtle);
  padding: 12px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 100;
}}
.top-bar h1 {{ font-size: 1.25rem; font-weight: 700; }}
.top-bar h1 .brand {{ color: var(--accent-blue); }}
.top-bar-actions {{ display: flex; gap: 8px; }}
.top-bar-actions button {{
  background: var(--bg-card);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  padding: 6px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: background 0.2s;
}}
.top-bar-actions button:hover {{ background: var(--accent-blue); color: #fff; }}

.container {{ max-width: 1440px; margin: 0 auto; padding: 24px 32px; }}

.tab-nav {{
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 24px;
  background: var(--bg-dark);
  padding: 8px;
  border-radius: 12px;
  border: 1px solid var(--border-subtle);
}}
.tab-btn {{
  background: transparent;
  color: var(--text-secondary);
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
  white-space: nowrap;
}}
.tab-btn:hover {{ background: rgba(124,138,255,0.1); color: var(--text-primary); }}
.tab-btn.active {{ background: var(--accent-blue); color: #fff; font-weight: 600; }}

.tab-content {{ display: none; animation: fadeIn 0.3s ease; }}
.tab-content.active {{ display: block; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}

.card {{
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-card);
}}
.card h2 {{ font-size: 1.3rem; margin-bottom: 8px; color: var(--accent-blue); }}
.card h3 {{ font-size: 1.1rem; margin-bottom: 6px; color: var(--accent-purple); }}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
.grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
@media (max-width: 1024px) {{
  .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
}}

.kpi-card {{
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  box-shadow: var(--shadow-card);
}}
.kpi-card .kpi-value {{ font-size: 2rem; font-weight: 700; margin: 8px 0; }}
.kpi-card .kpi-label {{ font-size: 0.85rem; color: var(--text-secondary); }}

.data-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.9rem; }}
.data-table th {{
  background: rgba(124,138,255,0.1);
  padding: 10px 14px;
  text-align: left;
  border-bottom: 2px solid var(--border-subtle);
  font-weight: 600;
  color: var(--accent-blue);
}}
.data-table td {{ padding: 8px 14px; border-bottom: 1px solid var(--border-subtle); }}
.data-table tr:hover td {{ background: rgba(124,138,255,0.05); }}

.insight {{
  border-radius: 10px;
  padding: 16px 20px;
  margin: 16px 0;
  font-size: 0.95rem;
  line-height: 1.7;
  border-left: 4px solid;
}}
.insight-green {{ background: rgba(0,229,176,0.08); border-color: var(--accent-green); }}
.insight-orange {{ background: rgba(255,179,71,0.08); border-color: var(--accent-orange); }}
.insight-red {{ background: rgba(255,107,122,0.08); border-color: var(--accent-red); }}
.insight-blue {{ background: rgba(124,138,255,0.08); border-color: var(--accent-blue); }}
.insight strong {{ color: var(--accent-blue); }}

.desc {{ color: var(--text-secondary); margin-bottom: 16px; font-size: 0.95rem; line-height: 1.7; }}

.chart-container {{ position: relative; width: 100%; margin: 12px 0; }}
.chart-container canvas {{ max-height: 500px; }}

.pipeline {{
  display: flex;
  align-items: center;
  gap: 0;
  flex-wrap: wrap;
  justify-content: center;
  margin: 20px 0;
}}
.pipeline-step {{
  background: var(--bg-dark);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 14px 20px;
  text-align: center;
  min-width: 130px;
  font-size: 0.85rem;
}}
.pipeline-step .step-title {{ font-weight: 700; color: var(--accent-blue); margin-bottom: 4px; }}
.pipeline-step .step-desc {{ color: var(--text-secondary); font-size: 0.8rem; }}
.pipeline-arrow {{ font-size: 1.5rem; color: var(--accent-blue); padding: 0 8px; }}

.arch-diagram {{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 20px;
}}
.arch-block {{
  background: var(--bg-dark);
  border: 2px solid var(--accent-blue);
  border-radius: 10px;
  padding: 12px 24px;
  text-align: center;
  min-width: 200px;
  font-weight: 600;
  font-size: 0.9rem;
}}
.arch-block.highlight {{ border-color: var(--accent-green); background: rgba(0,229,176,0.08); }}
.arch-block.vader {{ border-color: var(--accent-orange); background: rgba(255,179,71,0.08); }}
.arch-block.output {{ border-color: var(--accent-purple); background: rgba(181,122,255,0.08); }}
.arch-arrow {{ color: var(--accent-blue); font-size: 1.3rem; }}
.arch-side {{ display: flex; gap: 16px; align-items: flex-end; }}
.arch-col {{ display: flex; flex-direction: column; align-items: center; gap: 8px; }}

.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; }}
.gallery-item {{ text-align: center; }}
.gallery-item img {{ max-width: 100%; border-radius: 8px; border: 1px solid var(--border-subtle); }}
.gallery-item .caption {{ margin-top: 8px; color: var(--text-secondary); font-size: 0.85rem; }}

select {{
  background: var(--bg-card);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 0.9rem;
  cursor: pointer;
  margin-bottom: 12px;
}}

.badge {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 600;
}}
.badge-green {{ background: rgba(0,229,176,0.2); color: var(--accent-green); }}
.badge-gray {{ background: rgba(156,163,196,0.2); color: var(--text-secondary); }}
.badge-red {{ background: rgba(255,107,122,0.2); color: var(--accent-red); }}

/* 이미지 라이트박스 */
.lightbox-overlay {{ position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.85);z-index:9999;display:none;align-items:center;justify-content:center;cursor:zoom-out }}
.lightbox-overlay.active {{ display:flex }}
.lightbox-overlay img {{ max-width:92vw;max-height:90vh;border-radius:8px;box-shadow:0 0 40px rgba(0,0,0,0.6) }}
.lightbox-close {{ position:fixed;top:20px;right:24px;color:white;font-size:2rem;cursor:pointer;z-index:10000;background:none;border:none }}
.gallery-item img, .gallery img {{ cursor:zoom-in }}

/* 데이터 탐색기 카드 */
.explorer-card {{
  background: var(--bg-dark);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.explorer-card .explorer-text {{
  font-size: 0.9rem;
  color: var(--text-primary);
  line-height: 1.6;
  word-break: break-word;
}}
.explorer-card .explorer-meta {{
  display: flex;
  gap: 12px;
  font-size: 0.8rem;
  color: var(--text-secondary);
  flex-wrap: wrap;
}}
.explorer-search-bar {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 16px;
  align-items: center;
}}
.explorer-search-bar input {{
  flex: 1;
  min-width: 200px;
  background: var(--bg-dark);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 0.9rem;
}}
.explorer-search-bar select {{
  min-width: 140px;
}}
.explorer-search-bar button {{
  background: var(--accent-blue);
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
}}
.explorer-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 14px;
}}
.explorer-pagination {{
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
}}
.explorer-pagination button {{
  background: var(--bg-card);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
}}
.explorer-pagination button:disabled {{
  opacity: 0.4;
  cursor: not-allowed;
}}

/* 비교 탭 */
.comparison-selectors {{
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 20px;
  align-items: center;
}}
.comparison-selectors label {{
  font-weight: 600;
  color: var(--text-secondary);
  font-size: 0.9rem;
}}
.comparison-side {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}}
@media (max-width: 768px) {{
  .comparison-side {{ grid-template-columns: 1fr; }}
}}
.delta-row {{
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-subtle);
  font-size: 0.9rem;
}}
.delta-positive {{ color: var(--accent-green); font-weight: 700; }}
.delta-negative {{ color: var(--accent-red); font-weight: 700; }}
.delta-neutral {{ color: var(--text-secondary); }}

/* 리포트 탭 */
.report-container {{
  background: var(--bg-dark);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 28px;
  font-size: 0.95rem;
  line-height: 1.8;
}}
.report-container h2 {{ color: var(--accent-blue); margin: 20px 0 10px; font-size: 1.2rem; }}
.report-container h3 {{ color: var(--accent-purple); margin: 16px 0 8px; font-size: 1.05rem; }}
.report-container table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
.report-container th {{ background: rgba(124,138,255,0.1); padding: 8px 12px; text-align: left; border-bottom: 2px solid var(--border-subtle); }}
.report-container td {{ padding: 6px 12px; border-bottom: 1px solid var(--border-subtle); }}
.report-actions {{
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
}}
.report-actions button {{
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  font-weight: 600;
  cursor: pointer;
  font-size: 0.9rem;
}}

/* 인쇄 전용 스타일 */
@media print {{
  .top-bar, .tab-nav, .top-bar-actions, #tab-playground {{ display:none !important; }}
  .tab-content {{ display:block !important; page-break-inside:avoid; }}
  body {{ background:white !important; color:black !important; }}
  .card {{ box-shadow:none !important; border:1px solid #ddd !important; }}
  .lightbox-overlay {{ display:none !important; }}
  .report-actions {{ display:none !important; }}
}}

/* Pipeline Deep-Dive 스타일 */
.pipeline-stage {{ border-left:4px solid; padding:0; margin-bottom:24px; border-radius:12px; background:var(--bg-dark); overflow:hidden }}
.pipeline-stage-header {{ display:flex; align-items:center; gap:12px; padding:20px 24px 12px }}
.pipeline-stage-num {{ display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:50%;font-weight:800;font-size:1rem;color:#fff;flex-shrink:0 }}
.pipeline-stage-body {{ padding:0 24px 20px }}
.pipeline-section {{ margin-bottom:14px }}
.pipeline-section-title {{ font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; color:var(--text-secondary) }}
.pipeline-next {{ background:rgba(124,138,255,0.06); border-top:1px dashed var(--border-subtle); padding:14px 24px; border-radius:0 0 11px 11px; font-size:0.9rem }}
.pipeline-next strong {{ color:var(--accent-blue) }}
.pipeline-io {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }}
@media (max-width:768px) {{ .pipeline-io {{ grid-template-columns:1fr }} }}
.pipeline-stage.color-blue {{ border-left-color: var(--accent-blue) }}
.pipeline-stage.color-green {{ border-left-color: var(--accent-green) }}
.pipeline-stage.color-purple {{ border-left-color: var(--accent-purple) }}
.pipeline-stage.color-orange {{ border-left-color: var(--accent-orange) }}

/* References 스타일 */
.ref-item {{ background:var(--bg-dark); border:1px solid var(--border-subtle); border-radius:10px; padding:16px 20px; margin-bottom:12px }}
.ref-item .ref-citation {{ font-size:0.92rem; color:var(--text-primary); margin-bottom:6px; font-weight:500 }}
.ref-item .ref-role {{ font-size:0.85rem; color:var(--text-secondary); padding-left:12px; border-left:3px solid var(--accent-blue) }}
.ref-item .ref-diff {{ font-size:0.85rem; color:var(--accent-orange); margin-top:6px; padding-left:12px; border-left:3px solid var(--accent-orange) }}

/* Error Analysis 스타일 */
.error-pattern-card {{ background:var(--bg-dark); border:1px solid var(--border-subtle); border-radius:10px; padding:16px 20px; margin-bottom:12px }}
.error-pattern-arrow {{ display:inline-flex; align-items:center; gap:8px; font-weight:700; font-size:0.95rem; margin-bottom:8px }}
.error-pattern-arrow .from-label {{ color:var(--accent-red, #ff6b6b) }}
.error-pattern-arrow .to-label {{ color:var(--accent-blue) }}
.error-pattern-desc {{ font-size:0.9rem; color:var(--text-secondary) }}
.ablation-diagram {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin:16px 0 }}
@media (max-width:768px) {{ .ablation-diagram {{ grid-template-columns:1fr }} }}
.ablation-col {{ background:var(--bg-dark); border:1px solid var(--border-subtle); border-radius:10px; padding:20px; text-align:center }}
.ablation-col h4 {{ margin:0 0 8px; font-size:0.95rem }}
.ablation-col .ablation-f1 {{ font-size:1.6rem; font-weight:800; color:var(--accent-blue) }}
.ablation-col .ablation-arch {{ font-size:0.8rem; color:var(--text-secondary); margin-top:4px }}
.ablation-pval {{ display:flex; align-items:center; justify-content:center; gap:8px; font-size:0.85rem; color:var(--text-secondary); margin:8px 0 }}
.repro-table {{ width:100%; border-collapse:collapse }}
.repro-table th, .repro-table td {{ padding:10px 14px; text-align:left; border-bottom:1px solid var(--border-subtle) }}
.repro-table th {{ font-size:0.8rem; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-secondary) }}

/* E2E Pipeline Analysis */
.e2e-flow {{ display:flex; flex-direction:column; gap:0; position:relative }}
.e2e-node {{ background:var(--bg-dark); border:2px solid var(--border-subtle); border-radius:12px; padding:20px 24px; position:relative }}
.e2e-node.active {{ border-color:var(--accent-green) }}
.e2e-connector {{ display:flex; align-items:center; justify-content:center; padding:4px 0 }}
.e2e-connector .arrow {{ color:var(--accent-blue); font-size:1.6rem }}
.e2e-connector .transform {{ font-size:0.78rem; color:var(--text-secondary); background:var(--bg-card); padding:2px 12px; border-radius:12px; border:1px dashed var(--border-subtle); margin:0 12px }}
.e2e-node-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:10px }}
.e2e-node-title {{ font-weight:700; font-size:1rem }}
.e2e-badge {{ display:inline-block; padding:3px 10px; border-radius:6px; font-size:0.75rem; font-weight:600 }}
.e2e-metrics {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); gap:8px; margin-top:10px }}
.e2e-metric {{ background:var(--bg-card); padding:10px; border-radius:8px; text-align:center; border:1px solid var(--border-subtle) }}
.e2e-metric .val {{ font-size:1.2rem; font-weight:700 }}
.e2e-metric .lbl {{ font-size:0.75rem; color:var(--text-secondary); margin-top:2px }}
.e2e-decision {{ background:rgba(124,138,255,0.06); border-left:3px solid var(--accent-blue); padding:10px 14px; margin-top:10px; border-radius:0 8px 8px 0; font-size:0.85rem }}
.e2e-decision strong {{ color:var(--accent-blue) }}
.e2e-bottleneck {{ background:rgba(255,107,122,0.08); border-left:3px solid var(--accent-red); padding:10px 14px; margin-top:8px; border-radius:0 8px 8px 0; font-size:0.85rem }}
.e2e-bottleneck strong {{ color:var(--accent-red) }}
.e2e-sankey {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:0; text-align:center; margin:16px 0 }}
.e2e-sankey-col {{ padding:12px }}
.e2e-sankey-box {{ background:var(--bg-dark); border:1px solid var(--border-subtle); border-radius:8px; padding:10px; margin-bottom:6px; font-size:0.85rem }}
.e2e-time-bar {{ height:24px; border-radius:4px; display:flex; align-items:center; padding:0 8px; font-size:0.75rem; font-weight:600; color:#fff; min-width:40px }}
.e2e-section {{ margin-bottom:28px }}
</style>
</head>
<body>

<!-- 이미지 라이트박스 오버레이 -->
<div class="lightbox-overlay" id="lightbox" onclick="closeLightbox()">
  <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
  <img id="lightbox-img" src="" alt="Enlarged">
</div>

<!-- ================================================================
     상단바
     ================================================================ -->
<div class="top-bar">
  <h1><span class="brand">HateSpeachStudy</span>
    <span class="ko"> 종합 대시보드</span>
    <span class="en"> Comprehensive Dashboard</span>
  </h1>
  <div class="top-bar-actions">
    <button onclick="window.print()" title="Export Report as PDF"><span class="ko">Export</span><span class="en">Export</span></button>
    <button onclick="toggleLang()" id="langBtn">EN</button>
    <button onclick="toggleTheme()" id="themeBtn">Light</button>
  </div>
</div>

<div class="container">

<!-- ================================================================
     탭 내비게이션
     ================================================================ -->
<div class="tab-nav">
  <button class="tab-btn active" data-tab="overview">
    <span class="ko">Overview</span><span class="en">Overview</span>
  </button>
  <button class="tab-btn" data-tab="pipeline">
    <span class="ko">Pipeline Deep-Dive</span><span class="en">Pipeline Deep-Dive</span>
  </button>
  <button class="tab-btn" data-tab="e2e">
    <span class="ko">E2E Pipeline</span><span class="en">E2E Pipeline</span>
  </button>
  <button class="tab-btn" data-tab="benchmark">
    <span class="ko">Benchmark</span><span class="en">Benchmark</span>
  </button>
  <button class="tab-btn" data-tab="statistical">
    <span class="ko">Statistical Tests</span><span class="en">Statistical Tests</span>
  </button>
  <button class="tab-btn" data-tab="tuning">
    <span class="ko">Tuning</span><span class="en">Tuning</span>
  </button>
  <button class="tab-btn" data-tab="learning">
    <span class="ko">Learning Curves</span><span class="en">Learning Curves</span>
  </button>
  <button class="tab-btn" data-tab="freeze">
    <span class="ko">Freeze Study</span><span class="en">Freeze Study</span>
  </button>
  <button class="tab-btn" data-tab="eda">
    <span class="ko">EDA</span><span class="en">EDA</span>
  </button>
  <button class="tab-btn" data-tab="xai">
    <span class="ko">XAI Analysis</span><span class="en">XAI Analysis</span>
  </button>
  <button class="tab-btn" data-tab="xai_cases">
    <span class="ko">XAI Cases</span><span class="en">XAI Cases</span>
  </button>
  <button class="tab-btn" data-tab="errors">
    <span class="ko">Error Analysis</span><span class="en">Error Analysis</span>
  </button>
  <button class="tab-btn" data-tab="architecture">
    <span class="ko">Model Architecture</span><span class="en">Model Architecture</span>
  </button>
  <button class="tab-btn" data-tab="explorer">
    <span class="ko">Data Explorer</span><span class="en">Data Explorer</span>
  </button>
  <button class="tab-btn" data-tab="comparison">
    <span class="ko">Comparison</span><span class="en">Comparison</span>
  </button>
  <button class="tab-btn" data-tab="report">
    <span class="ko">Report</span><span class="en">Report</span>
  </button>
  <button class="tab-btn" data-tab="references">
    <span class="ko">References</span><span class="en">References</span>
  </button>
  <button class="tab-btn" data-tab="playground">
    <span class="ko">Playground</span><span class="en">Playground</span>
  </button>
</div>

<!-- ================================================================
     TAB 1: Overview
     ================================================================ -->
<div id="tab-overview" class="tab-content active">
  <div class="card">
    <h2><span class="ko">Executive Summary</span><span class="en">Executive Summary</span></h2>
    <div class="ko-block desc">
      <p>이 대시보드는 HateXplain 데이터셋 기반 혐오표현 탐지 연구의 전체 실험 결과를 종합적으로 보여준다.
      6개의 모델(TF-IDF+LR, TF-IDF+SVM, BERT-base, BERT+MLP, BERT+VADER, RoBERTa+VADER)을
      "가설 -> 실험 -> XAI 사후 검증"이라는 과학적 검증 프레임워크에 따라 평가하였다.</p>
      <p>핵심 가설: VADER 감성 점수를 보조 특성으로 결합하면 혐오표현 탐지 성능이 향상될 것이다.
      이 가설은 Cheng(2022)의 선행 연구에서 감성 분석이 혐오표현 분류에 유의미한 기여를 한다는 발견에 기반한다.</p>
    </div>
    <div class="en-block desc">
      <p>This dashboard presents the comprehensive experimental results of hate speech detection research based on the HateXplain dataset.
      Six models (TF-IDF+LR, TF-IDF+SVM, BERT-base, BERT+MLP, BERT+VADER, RoBERTa+VADER) were evaluated
      following a scientific verification framework of "Hypothesis -> Experiment -> XAI Post-hoc Verification".</p>
      <p>Core hypothesis: Combining VADER sentiment scores as auxiliary features will improve hate speech detection performance.
      This hypothesis is grounded in Cheng (2022)'s finding that sentiment analysis contributes meaningfully to hate speech classification.</p>
    </div>
  </div>

  <div class="grid-4">
    <div class="kpi-card">
      <div class="kpi-label"><span class="ko">최고 Macro F1</span><span class="en">Best Macro F1</span></div>
      <div class="kpi-value" style="color:var(--accent-green)">0.6863</div>
      <div class="kpi-label">RoBERTa+VADER</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span class="ko">모델 수</span><span class="en">Models Tested</span></div>
      <div class="kpi-value" style="color:var(--accent-blue)">6</div>
      <div class="kpi-label"><span class="ko">+ Freeze Study 1</span><span class="en">+ 1 Freeze Study</span></div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span class="ko">Fine-tuning 향상</span><span class="en">Fine-tuning Gain</span></div>
      <div class="kpi-value" style="color:var(--accent-orange)">+109%</div>
      <div class="kpi-label">0.324 &rarr; 0.679</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span class="ko">통계적 유의성</span><span class="en">Statistical Significance</span></div>
      <div class="kpi-value" style="color:var(--accent-purple)">p=0.037</div>
      <div class="kpi-label">RoBERTa vs BERT-base</div>
    </div>
  </div>

  <div class="card" style="margin-top:20px">
    <h3><span class="ko">과학적 검증 파이프라인</span><span class="en">Scientific Verification Pipeline</span></h3>
    <div class="ko-block desc">이 연구의 전체 흐름을 나타내는 파이프라인이다. 각 단계에서 엄밀한 실험 설계와 통계 검정을 수행한다.</div>
    <div class="en-block desc">The full pipeline of this study. Each stage involves rigorous experimental design and statistical testing.</div>
    <div class="pipeline">
      <div class="pipeline-step">
        <div class="step-title"><span class="ko">가설 수립</span><span class="en">Hypothesis</span></div>
        <div class="step-desc">VADER + Transformer</div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">
        <div class="step-title">EDA</div>
        <div class="step-desc"><span class="ko">19,192 텍스트 분석</span><span class="en">19,192 text analysis</span></div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">
        <div class="step-title"><span class="ko">베이스라인</span><span class="en">Baseline</span></div>
        <div class="step-desc">TF-IDF + LR/SVM</div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">
        <div class="step-title"><span class="ko">딥러닝</span><span class="en">Deep Learning</span></div>
        <div class="step-desc">BERT / RoBERTa</div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">
        <div class="step-title"><span class="ko">하이브리드</span><span class="en">Hybrid</span></div>
        <div class="step-desc">+VADER Sentiment</div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">
        <div class="step-title"><span class="ko">튜닝</span><span class="en">Tuning</span></div>
        <div class="step-desc">LR / Dropout</div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">
        <div class="step-title"><span class="ko">통계 검정</span><span class="en">Stat Test</span></div>
        <div class="step-desc">Paired t-test</div>
      </div>
      <div class="pipeline-arrow">&rarr;</div>
      <div class="pipeline-step">
        <div class="step-title">XAI</div>
        <div class="step-desc">SHAP + LIME</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">주요 발견 사항</span><span class="en">Key Findings</span></h3>
    <div class="insight insight-green">
      <span class="ko"><strong>RoBERTa+VADER가 최고 성능 달성:</strong> Macro F1 0.6863으로 BERT-base(0.6744) 대비 +1.19%p 향상.
      이 차이는 통계적으로 유의미하다 (p=0.0366, Cohen's d=-2.93). RoBERTa의 강화된 사전학습과
      VADER 감성 정보의 결합이 시너지를 이룸을 시사한다.</span>
      <span class="en"><strong>RoBERTa+VADER achieved best performance:</strong> Macro F1 of 0.6863, a +1.19%p improvement over BERT-base (0.6744).
      This difference is statistically significant (p=0.0366, Cohen's d=-2.93). This suggests synergy between
      RoBERTa's enhanced pre-training and VADER sentiment information.</span>
    </div>
    <div class="insight insight-orange">
      <span class="ko"><strong>VADER 단독 효과는 제한적:</strong> BERT+VADER(0.6794) vs BERT-base(0.6744)는 p=0.1384로 유의미하지 않다.
      VADER 감성 점수가 같은 인코더 기반에서는 결정적 차이를 만들지 못했다. 인코더 품질이 더 핵심 변수임을 보여준다.</span>
      <span class="en"><strong>VADER alone has limited effect:</strong> BERT+VADER (0.6794) vs BERT-base (0.6744) yields p=0.1384 (not significant).
      VADER sentiment scores did not create a decisive difference on the same encoder base. Encoder quality is the more critical variable.</span>
    </div>
    <div class="insight insight-red">
      <span class="ko"><strong>hate/offensive 분류 경계가 모호:</strong> 어휘 Jaccard 유사도 0.7094로 두 클래스가 상당히 중첩된다.
      hate 클래스 F1은 최고 모델에서도 0.775로 양호하나, offensive(0.547)가 여전히 도전적이다.</span>
      <span class="en"><strong>hate/offensive classification boundary is blurry:</strong> Vocabulary Jaccard similarity of 0.7094 indicates substantial overlap.
      Hate F1 is decent at 0.775 for the best model, but offensive (0.547) remains challenging.</span>
    </div>
  </div>
</div>

<!-- ================================================================
     TAB: Pipeline Deep-Dive
     ================================================================ -->
<div id="tab-pipeline" class="tab-content">
  <div class="card">
    <h2><span class="ko">Pipeline Deep-Dive: 8단계 실험 흐름</span><span class="en">Pipeline Deep-Dive: 8-Stage Experiment Flow</span></h2>
    <div class="ko-block desc">각 단계가 왜 필요했는지, 무엇을 발견했는지, 그리고 다음 단계로 어떻게 연결되는지를 상세히 서술한다. 이것은 과학적 검증 프레임워크의 전체 흐름이다.</div>
    <div class="en-block desc">Each stage explains why it was needed, what was discovered, and how it connects to the next. This is the full flow of the scientific verification framework.</div>
  </div>

  <!-- Stage 1 -->
  <div class="card pipeline-stage color-blue" style="border-left-color:var(--accent-blue)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-blue)">1</span>
      <h3 style="margin:0"><span class="ko">Stage 1: 가설 수립 (Hypothesis)</span><span class="en">Stage 1: Hypothesis Formation</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">Cheng(2022, Virginia Tech)의 연구에서 감성 분석이 혐오표현 분류에 유의미한 기여를 한다는 발견에 기반. 선행 연구의 사전 가설을 체계적으로 검증하기 위한 출발점.</div>
        <div class="en-block">Based on Cheng (2022, Virginia Tech) finding that sentiment analysis contributes meaningfully to hate speech classification. Starting point for systematically verifying prior research hypotheses.</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> <span class="ko">선행 연구 문헌 (Cheng 2022, Mathew et al. 2021 HateXplain)</span><span class="en">Prior research literature (Cheng 2022, Mathew et al. 2021 HateXplain)</span></div>
          <div><strong>Output:</strong> <span class="ko">"VADER 감성 점수를 보조 특성으로 결합하면 혐오표현 탐지 성능이 향상될 것이다"</span><span class="en">"Combining VADER sentiment scores as auxiliary features will improve hate speech detection performance"</span></div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">Cheng(2022)와의 차별점 -- 우리는 VADER를 XAI 산출물이 아닌 선행 연구 기반 사전 가설로 도입하며, XAI는 사후 검증 도구로만 사용한다.</div>
        <div class="en-block">Differentiation from Cheng(2022) -- we introduce VADER as a prior research-based hypothesis, not as an XAI diagnostic result. XAI is used solely as a post-hoc verification tool.</div>
      </div>
    </div>
    <div class="pipeline-next"><strong>&#x2192; <span class="ko">다음 단계:</span><span class="en">Next:</span></strong> <span class="ko">가설 검증 전, 데이터 특성을 파악해야 한다 &#x2192; EDA</span><span class="en">Before hypothesis testing, data characteristics must be understood &#x2192; EDA</span></div>
  </div>

  <!-- Stage 2 -->
  <div class="card pipeline-stage color-green" style="border-left-color:var(--accent-green)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-green)">2</span>
      <h3 style="margin:0"><span class="ko">Stage 2: EDA (탐색적 데이터 분석)</span><span class="en">Stage 2: EDA (Exploratory Data Analysis)</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">모델 설계 전 데이터의 본질적 특성과 난이도를 파악. 클래스 간 구별 가능성을 사전에 진단.</div>
        <div class="en-block">Understand intrinsic data characteristics and difficulty before model design. Pre-diagnose inter-class separability.</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> <span class="ko">HateXplain 원본 19,192 텍스트 &#x2192; majority vote 후 13,433 샘플</span><span class="en">HateXplain raw 19,192 texts &#x2192; 13,433 samples after majority vote</span></div>
          <div><strong>Output:</strong> <span class="ko">클래스 분포, VADER 감성 패턴, 어휘 중첩도, 텍스트 길이 통계</span><span class="en">Class distribution, VADER sentiment patterns, vocabulary overlap, text length statistics</span></div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">hate/offensive 어휘 Jaccard 유사도 <strong>0.7094</strong> -- 두 클래스가 71% 어휘를 공유. VADER compound: hate=-0.358 > offensive=-0.283 > normal=-0.181로 감성 gradient 존재 확인.</div>
        <div class="en-block">hate/offensive vocabulary Jaccard similarity <strong>0.7094</strong> -- 71% shared vocabulary. VADER compound: hate=-0.358 > offensive=-0.283 > normal=-0.181, confirming sentiment gradient.</div>
      </div>
    </div>
    <div class="pipeline-next"><strong>&#x2192; <span class="ko">다음 단계:</span><span class="en">Next:</span></strong> <span class="ko">높은 어휘 중첩 &#x2192; 단순 어휘 기반 모델의 한계 예측 &#x2192; 베이스라인 수립</span><span class="en">High vocabulary overlap &#x2192; predicted limits of simple lexical models &#x2192; establish baseline</span></div>
  </div>

  <!-- Stage 3 -->
  <div class="card pipeline-stage color-purple" style="border-left-color:var(--accent-purple)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-purple)">3</span>
      <h3 style="margin:0"><span class="ko">Stage 3: 베이스라인 (TF-IDF + ML)</span><span class="en">Stage 3: Baseline (TF-IDF + ML)</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">딥러닝 투입 전 전통 ML로 기준선 확보. 어휘 기반 접근의 실제 한계를 정량화.</div>
        <div class="en-block">Establish baseline with traditional ML before deep learning. Quantify actual limits of lexical approaches.</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> <span class="ko">TF-IDF 벡터화된 텍스트 (max_features=30000)</span><span class="en">TF-IDF vectorized text (max_features=30000)</span></div>
          <div><strong>Output:</strong> TF-IDF+LR F1=0.6370, TF-IDF+SVM F1=0.6393</div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">EDA에서 예측한 대로 어휘 중첩(0.71)이 성능 천장을 만듦. offensive F1이 특히 낮음 &#x2192; 문맥 이해가 필요.</div>
        <div class="en-block">As predicted by EDA, vocabulary overlap (0.71) creates a performance ceiling. Offensive F1 is particularly low &#x2192; contextual understanding needed.</div>
      </div>
    </div>
    <div class="pipeline-next"><strong>&#x2192; <span class="ko">다음 단계:</span><span class="en">Next:</span></strong> <span class="ko">어휘 한계 확인 &#x2192; 문맥적 표현 학습이 가능한 Transformer 도입</span><span class="en">Lexical limits confirmed &#x2192; introduce Transformer for contextual representation learning</span></div>
  </div>

  <!-- Stage 4 -->
  <div class="card pipeline-stage color-orange" style="border-left-color:var(--accent-orange)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-orange)">4</span>
      <h3 style="margin:0"><span class="ko">Stage 4: 딥러닝 (Transformer)</span><span class="en">Stage 4: Deep Learning (Transformer)</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">BERT의 양방향 문맥 표현으로 어휘 중첩 문제 극복 시도. RoBERTa는 더 많은 데이터, 더 긴 학습, 동적 마스킹으로 사전학습이 강화된 모델.</div>
        <div class="en-block">Attempt to overcome vocabulary overlap via BERT's bidirectional contextual representations. RoBERTa has enhanced pretraining with more data, longer training, and dynamic masking.</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> <span class="ko">토크나이즈된 텍스트 (max_len=128), [CLS] 토큰 표현</span><span class="en">Tokenized text (max_len=128), [CLS] token representation</span></div>
          <div><strong>Output:</strong> BERT-base F1=0.6744 (+5.5%p vs SVM)</div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">Transformer 도입으로 +5.5%p 향상. 하지만 BERT 단독으로는 감성 뉘앙스를 완전히 포착하지 못할 수 있음.</div>
        <div class="en-block">Transformer introduction yields +5.5%p improvement. However, BERT alone may not fully capture sentiment nuances.</div>
      </div>
    </div>
    <div class="pipeline-next"><strong>&#x2192; <span class="ko">다음 단계:</span><span class="en">Next:</span></strong> <span class="ko">가설 검증을 위해 VADER 감성을 보조 특성으로 결합 &#x2192; 하이브리드</span><span class="en">Combine VADER sentiment as auxiliary feature for hypothesis testing &#x2192; Hybrid</span></div>
  </div>

  <!-- Stage 5 -->
  <div class="card pipeline-stage color-blue" style="border-left-color:var(--accent-blue)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-blue)">5</span>
      <h3 style="margin:0"><span class="ko">Stage 5: 하이브리드 (Transformer + VADER)</span><span class="en">Stage 5: Hybrid (Transformer + VADER)</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">768차원 [CLS] 벡터에 VADER 4차원(compound, pos, neg, neu)을 concat하여 감성 정보를 직접 주입. 이것이 핵심 가설의 구현.</div>
        <div class="en-block">Concatenate VADER 4 dimensions (compound, pos, neg, neu) to 768-dim [CLS] vector for direct sentiment injection. This is the core hypothesis implementation.</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> [CLS](768d) + VADER(4d) = 772d &#x2192; MLP(256d) &#x2192; 3-class</div>
          <div><strong>Output:</strong> BERT+VADER F1=0.6794, RoBERTa+VADER F1=0.6863</div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">RoBERTa+VADER가 최고 성능. 하지만 BERT+VADER vs BERT-base 차이(+0.5%p)가 통계적으로 유의하지 않음 &#x2192; 인코더 품질이 더 중요한 변수.</div>
        <div class="en-block">RoBERTa+VADER achieves best performance. But BERT+VADER vs BERT-base difference (+0.5%p) is not statistically significant &#x2192; encoder quality is the more important variable.</div>
      </div>
    </div>
    <div class="pipeline-next"><strong>&#x2192; <span class="ko">다음 단계:</span><span class="en">Next:</span></strong> <span class="ko">VADER 효과와 MLP 구조 효과를 분리해야 함 &#x2192; Ablation</span><span class="en">Must separate VADER effect from MLP structure effect &#x2192; Ablation</span></div>
  </div>

  <!-- Stage 6 -->
  <div class="card pipeline-stage color-green" style="border-left-color:var(--accent-green)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-green)">6</span>
      <h3 style="margin:0"><span class="ko">Stage 6: Ablation Study</span><span class="en">Stage 6: Ablation Study</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">성능 향상이 VADER 때문인지, MLP 레이어 추가 때문인지 분리. 과학적 검증의 핵심 통제 실험.</div>
        <div class="en-block">Separate whether improvement is from VADER or MLP layer addition. Core controlled experiment of scientific verification.</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> <span class="ko">BERT+MLP (768d&#x2192;256d, VADER 없음) vs BERT+VADER (772d&#x2192;256d, VADER 있음)</span><span class="en">BERT+MLP (768d&#x2192;256d, no VADER) vs BERT+VADER (772d&#x2192;256d, with VADER)</span></div>
          <div><strong>Output:</strong> BERT+MLP F1=0.6810, BERT+VADER F1=0.6794, p=0.567</div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">두 모델 성능 거의 동일(p=0.567) &#x2192; 같은 인코더에서 VADER 추가 효과 미미. 성능 차이의 주요 원인은 인코더 사전학습 품질(BERT vs RoBERTa).</div>
        <div class="en-block">Nearly identical performance (p=0.567) &#x2192; VADER effect minimal on same encoder. Main performance driver is encoder pretraining quality (BERT vs RoBERTa).</div>
      </div>
    </div>
    <div class="pipeline-next"><strong>&#x2192; <span class="ko">다음 단계:</span><span class="en">Next:</span></strong> <span class="ko">인코더가 핵심이라면, 인코더 동결 시 성능이 얼마나 떨어지는지 확인 &#x2192; Freeze Study</span><span class="en">If encoder is key, how much does freezing it hurt? &#x2192; Freeze Study</span></div>
  </div>

  <!-- Stage 7 -->
  <div class="card pipeline-stage color-purple" style="border-left-color:var(--accent-purple)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-purple)">7</span>
      <h3 style="margin:0"><span class="ko">Stage 7: Freeze Study</span><span class="en">Stage 7: Freeze Study</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">인코더 미세조정의 중요성을 정량화. "VADER+MLP만으로 충분한가?"라는 질문에 답변.</div>
        <div class="en-block">Quantify the importance of encoder fine-tuning. Answer: "Is VADER+MLP alone sufficient?"</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> <span class="ko">BERT+VADER (encoder frozen) vs BERT+VADER (encoder fine-tuned)</span><span class="en">BERT+VADER (encoder frozen) vs BERT+VADER (encoder fine-tuned)</span></div>
          <div><strong>Output:</strong> Frozen F1=0.324, Fine-tuned F1=0.679 (+109%)</div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">동결 시 random 수준(0.33)에 근접 &#x2192; MLP+VADER만으로는 분류 불가능. 인코더의 도메인 특화 미세조정이 성능의 결정적 요인.</div>
        <div class="en-block">Frozen performance near random level (0.33) &#x2192; MLP+VADER alone cannot classify. Encoder domain-specific fine-tuning is the decisive performance factor.</div>
      </div>
    </div>
    <div class="pipeline-next"><strong>&#x2192; <span class="ko">다음 단계:</span><span class="en">Next:</span></strong> <span class="ko">모든 실험 완료 &#x2192; 모델의 예측 근거를 사후 검증 &#x2192; XAI</span><span class="en">All experiments complete &#x2192; post-hoc verification of model reasoning &#x2192; XAI</span></div>
  </div>

  <!-- Stage 8 -->
  <div class="card pipeline-stage color-orange" style="border-left-color:var(--accent-orange)">
    <div class="pipeline-stage-header">
      <span class="pipeline-stage-num" style="background:var(--accent-orange)">8</span>
      <h3 style="margin:0"><span class="ko">Stage 8: XAI 사후 검증</span><span class="en">Stage 8: XAI Post-hoc Verification</span></h3>
    </div>
    <div class="pipeline-stage-body">
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">목적 (Why)</span><span class="en">Purpose (Why)</span></div>
        <div class="ko-block">모델이 "올바른 이유로" 올바른 예측을 하는지 검증. 이것은 피드백 루프가 아니라 사후 검증 도구이다. SHAP과 LIME 두 기법의 교차 검증으로 설명의 신뢰성 확보.</div>
        <div class="en-block">Verify whether models predict correctly "for the right reasons." This is NOT a feedback loop but a post-hoc verification tool. Cross-validation between SHAP and LIME ensures explanation reliability.</div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">입력/출력 (Input / Output)</span><span class="en">Input / Output</span></div>
        <div class="pipeline-io">
          <div><strong>Input:</strong> <span class="ko">BERT-base(베이스라인) + RoBERTa+VADER(개선 모델) x 24 샘플</span><span class="en">BERT-base (baseline) + RoBERTa+VADER (improved) x 24 samples</span></div>
          <div><strong>Output:</strong> Overlap@5: Baseline 71.7%, Improved 67.5%. <span class="ko">오분류 수정 12건</span><span class="en">12 misclassification corrections</span></div>
        </div>
      </div>
      <div class="pipeline-section">
        <div class="pipeline-section-title"><span class="ko">핵심 결과 (Key Result)</span><span class="en">Key Result</span></div>
        <div class="ko-block">개선 모델의 Overlap이 약간 낮지만, 60% 이상 샘플 수는 더 많음(20 vs 18). 모델이 더 다양한 특성에 주목하면서도 일관성 유지.</div>
        <div class="en-block">Improved model's Overlap is slightly lower, but more samples exceed 60% (20 vs 18). Model attends to more diverse features while maintaining consistency.</div>
      </div>
    </div>
    <div class="pipeline-next"><strong><span class="ko">(파이프라인 종료) 결론 -- 가설 부분 채택. 인코더 품질이 핵심, VADER는 보조.</span><span class="en">(Pipeline complete) Conclusion -- Hypothesis partially adopted. Encoder quality is key, VADER is supplementary.</span></strong></div>
  </div>
</div>

<!-- ================================================================
     TAB: E2E Pipeline Analysis
     ================================================================ -->
<div id="tab-e2e" class="tab-content">
  <div class="card">
    <h2><span class="ko">End-to-End Pipeline Analysis</span><span class="en">End-to-End Pipeline Analysis</span></h2>
    <div class="ko-block desc">
      원본 데이터가 최종 예측 결과가 되기까지의 기술적 데이터 흐름을 추적한다.
      각 단계에서의 데이터 변환, 차원 변화, 처리 시간, 핵심 설계 결정을 정량적으로 분석한다.
      Pipeline Deep-Dive가 "왜"에 초점을 맞춘다면, 이 탭은 "어떻게"에 초점을 맞춘다.
    </div>
    <div class="en-block desc">
      Tracks the technical data flow from raw input to final prediction.
      Quantitatively analyzes data transformations, dimensionality changes, processing time, and key design decisions at each stage.
      While Pipeline Deep-Dive focuses on "why", this tab focuses on "how".
    </div>
  </div>

  <!-- ---- 데이터 흐름 전체 요약 ---- -->
  <div class="card">
    <h3><span class="ko">데이터 볼륨 추적</span><span class="en">Data Volume Tracking</span></h3>
    <div class="ko-block desc">원본 데이터에서 최종 평가까지 샘플 수와 차원이 어떻게 변하는지 추적한다.</div>
    <div class="en-block desc">Tracks how sample counts and dimensions change from raw data to final evaluation.</div>

    <div class="e2e-flow">
      <!-- Node 1: Raw Data -->
      <div class="e2e-node">
        <div class="e2e-node-header">
          <div class="e2e-node-title" style="color:var(--accent-blue)"><span class="ko">1. 원본 데이터 수집</span><span class="en">1. Raw Data Ingestion</span></div>
          <span class="e2e-badge" style="background:rgba(124,138,255,0.2);color:var(--accent-blue)">HateXplain</span>
        </div>
        <div style="font-size:0.9rem;color:var(--text-secondary)">
          <span class="ko">Mathew et al.(2021)의 HateXplain 데이터셋. JSON 형식, annotator 3명의 다수결 투표 포함.</span>
          <span class="en">HateXplain dataset by Mathew et al.(2021). JSON format with majority vote from 3 annotators.</span>
        </div>
        <div class="e2e-metrics">
          <div class="e2e-metric"><div class="val" style="color:var(--accent-blue)">19,192</div><div class="lbl"><span class="ko">전체 텍스트</span><span class="en">Total texts</span></div></div>
          <div class="e2e-metric"><div class="val">3</div><div class="lbl"><span class="ko">클래스</span><span class="en">Classes</span></div></div>
          <div class="e2e-metric"><div class="val">3</div><div class="lbl"><span class="ko">Annotators</span><span class="en">Annotators</span></div></div>
          <div class="e2e-metric"><div class="val">JSON</div><div class="lbl"><span class="ko">포맷</span><span class="en">Format</span></div></div>
        </div>
      </div>

      <div class="e2e-connector"><span class="arrow">&darr;</span><span class="transform"><span class="ko">Majority Vote 필터링</span><span class="en">Majority Vote Filtering</span></span></div>

      <!-- Node 2: Preprocessing -->
      <div class="e2e-node">
        <div class="e2e-node-header">
          <div class="e2e-node-title" style="color:var(--accent-green)"><span class="ko">2. 전처리 + VADER 증강</span><span class="en">2. Preprocessing + VADER Augmentation</span></div>
          <span class="e2e-badge" style="background:rgba(0,229,176,0.2);color:var(--accent-green)">~30%</span>
        </div>
        <div style="font-size:0.9rem;color:var(--text-secondary)">
          <span class="ko">Majority vote로 라벨 합의가 안 되는 샘플 제거. VADER로 4차원 감성 점수 생성. CSV 변환.</span>
          <span class="en">Remove samples without majority vote consensus. Generate 4-dim VADER sentiment scores. Convert to CSV.</span>
        </div>
        <div class="e2e-metrics">
          <div class="e2e-metric"><div class="val" style="color:var(--accent-green)">13,433</div><div class="lbl"><span class="ko">유효 샘플</span><span class="en">Valid samples</span></div></div>
          <div class="e2e-metric"><div class="val" style="color:var(--accent-red)">-5,759</div><div class="lbl"><span class="ko">제거됨 (30%)</span><span class="en">Removed (30%)</span></div></div>
          <div class="e2e-metric"><div class="val">+4d</div><div class="lbl">VADER (comp/pos/neg/neu)</div></div>
          <div class="e2e-metric"><div class="val">~5s</div><div class="lbl"><span class="ko">처리 시간</span><span class="en">Processing time</span></div></div>
        </div>
        <div class="e2e-decision"><strong><span class="ko">설계 결정:</span><span class="en">Design decision:</span></strong> <span class="ko">majority vote를 사용하여 annotator 간 불일치 해소. 2/3 이상 합의된 라벨만 사용하여 라벨 품질 확보.</span><span class="en">Majority vote resolves annotator disagreement. Only labels with 2/3+ consensus are used for label quality.</span></div>
      </div>

      <div class="e2e-connector"><span class="arrow">&darr;</span><span class="transform"><span class="ko">Train/Val/Test 분할</span><span class="en">Train/Val/Test Split</span></span></div>

      <!-- Node 3: Split -->
      <div class="e2e-node">
        <div class="e2e-node-header">
          <div class="e2e-node-title" style="color:var(--accent-purple)"><span class="ko">3. 데이터 분할</span><span class="en">3. Data Split</span></div>
          <span class="e2e-badge" style="background:rgba(181,122,255,0.2);color:var(--accent-purple)">Stratified</span>
        </div>
        <div style="font-size:0.9rem;color:var(--text-secondary)">
          <span class="ko">클래스 비율을 유지하는 계층화 분할. 정식 v2는 15개 시드로 반복하여 분산을 추정한다.</span>
          <span class="en">Stratified split maintaining class proportions. The formal v2 run estimates variance across 15 seeds.</span>
        </div>
        <div class="e2e-metrics">
          <div class="e2e-metric"><div class="val">8,060</div><div class="lbl">Train (60%)</div></div>
          <div class="e2e-metric"><div class="val">2,687</div><div class="lbl">Val (20%)</div></div>
          <div class="e2e-metric"><div class="val">2,686</div><div class="lbl">Test (20%)</div></div>
          <div class="e2e-metric"><div class="val">15</div><div class="lbl"><span class="ko">시드</span><span class="en">Seeds</span></div></div>
        </div>
        <div class="e2e-decision"><strong><span class="ko">설계 결정:</span><span class="en">Design decision:</span></strong> <span class="ko">정식 v2는 15 seed paired design으로 조건 차이, 신뢰구간, 효과크기를 함께 보고한다.</span><span class="en">The formal v2 run uses a 15-seed paired design to report condition differences, confidence intervals, and effect sizes together.</span></div>
      </div>

      <div class="e2e-connector"><span class="arrow">&darr;</span><span class="transform"><span class="ko">모델별 분기</span><span class="en">Model-specific branching</span></span></div>

      <!-- Node 4: Feature Engineering -->
      <div class="e2e-node">
        <div class="e2e-node-header">
          <div class="e2e-node-title" style="color:var(--accent-orange)"><span class="ko">4. 특성 추출 (2가지 경로)</span><span class="en">4. Feature Extraction (2 paths)</span></div>
        </div>
        <div class="grid-2" style="margin-top:10px">
          <div style="background:var(--bg-card);padding:14px;border-radius:8px;border:1px solid var(--border-subtle)">
            <div style="font-weight:700;color:var(--accent-orange);margin-bottom:6px">Path A: TF-IDF</div>
            <div style="font-size:0.85rem;color:var(--text-secondary)">
              <span class="ko">텍스트 &rarr; TF-IDF 벡터 (max 30,000 features). 단어 빈도 기반, 문맥 정보 없음.</span>
              <span class="en">Text &rarr; TF-IDF vector (max 30,000 features). Frequency-based, no context.</span>
            </div>
            <div class="e2e-metrics" style="margin-top:8px">
              <div class="e2e-metric"><div class="val">30K</div><div class="lbl"><span class="ko">차원</span><span class="en">Dimensions</span></div></div>
              <div class="e2e-metric"><div class="val">Sparse</div><div class="lbl"><span class="ko">희소 행렬</span><span class="en">Sparse matrix</span></div></div>
            </div>
          </div>
          <div style="background:var(--bg-card);padding:14px;border-radius:8px;border:1px solid var(--border-subtle)">
            <div style="font-weight:700;color:var(--accent-blue);margin-bottom:6px">Path B: Transformer Tokenization</div>
            <div style="font-size:0.85rem;color:var(--text-secondary)">
              <span class="ko">텍스트 &rarr; WordPiece/BPE 토큰 &rarr; input_ids + attention_mask. 서브워드 단위, 문맥 이해 가능.</span>
              <span class="en">Text &rarr; WordPiece/BPE tokens &rarr; input_ids + attention_mask. Subword-level, context-aware.</span>
            </div>
            <div class="e2e-metrics" style="margin-top:8px">
              <div class="e2e-metric"><div class="val">128</div><div class="lbl">max_len (tokens)</div></div>
              <div class="e2e-metric"><div class="val">30,522</div><div class="lbl"><span class="ko">어휘 크기 (BERT)</span><span class="en">Vocab size (BERT)</span></div></div>
            </div>
          </div>
        </div>
        <div class="e2e-decision"><strong><span class="ko">설계 결정:</span><span class="en">Design decision:</span></strong> <span class="ko">max_len=128은 EDA 결과(평균 토큰 수 ~20-30) 기반. 99%+ 텍스트가 잘리지 않으면서 메모리 효율적.</span><span class="en">max_len=128 based on EDA results (avg ~20-30 tokens). 99%+ texts untruncated while memory-efficient.</span></div>
      </div>

      <div class="e2e-connector"><span class="arrow">&darr;</span><span class="transform"><span class="ko">모델 학습</span><span class="en">Model Training</span></span></div>

      <!-- Node 5: Model Training -->
      <div class="e2e-node" style="border-color:var(--accent-green)">
        <div class="e2e-node-header">
          <div class="e2e-node-title" style="color:var(--accent-green)"><span class="ko">5. 모델 학습 (8조건 x 15 시드)</span><span class="en">5. Model Training (8 conditions x 15 seeds)</span></div>
          <span class="e2e-badge" style="background:rgba(0,229,176,0.2);color:var(--accent-green)"><span class="ko">핵심 단계</span><span class="en">Core Stage</span></span>
        </div>
        <div style="font-size:0.9rem;color:var(--text-secondary);margin-bottom:10px">
          <span class="ko">각 모델의 내부 차원 변환 경로와 학습 설정을 비교한다.</span>
          <span class="en">Compares internal dimension transformation paths and training configurations across models.</span>
        </div>

        <div style="overflow-x:auto">
          <table class="data-table" style="font-size:0.82rem">
            <thead>
              <tr>
                <th><span class="ko">모델</span><span class="en">Model</span></th>
                <th><span class="ko">입력 차원</span><span class="en">Input Dim</span></th>
                <th><span class="ko">인코더</span><span class="en">Encoder</span></th>
                <th><span class="ko">분류 헤드</span><span class="en">Classifier Head</span></th>
                <th><span class="ko">출력</span><span class="en">Output</span></th>
                <th><span class="ko">파라미터 수</span><span class="en">Parameters</span></th>
                <th><span class="ko">학습 시간/시드</span><span class="en">Time/Seed</span></th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>TF-IDF+LR</td><td>30,000 (sparse)</td><td><span class="ko">없음</span><span class="en">None</span></td><td>Linear</td><td>3</td><td>~90K</td><td>~5s</td>
              </tr>
              <tr>
                <td>TF-IDF+SVM</td><td>30,000 (sparse)</td><td><span class="ko">없음</span><span class="en">None</span></td><td>SVM (RBF)</td><td>3</td><td>~90K</td><td>~10s</td>
              </tr>
              <tr>
                <td>BERT-base</td><td>128 tokens</td><td>BERT 12L (768d)</td><td>Linear(768&rarr;3)</td><td>3</td><td>~109M</td><td>~8min</td>
              </tr>
              <tr>
                <td>BERT+MLP</td><td>128 tokens</td><td>BERT 12L (768d)</td><td>MLP(768&rarr;256&rarr;3)</td><td>3</td><td>~109.5M</td><td>~8min</td>
              </tr>
              <tr>
                <td>BERT+VADER</td><td>128 tokens + 4d</td><td>BERT 12L (768d)</td><td>MLP(772&rarr;256&rarr;3)</td><td>3</td><td>~109.5M</td><td>~9min</td>
              </tr>
              <tr style="color:var(--accent-green);font-weight:600">
                <td>RoBERTa+VADER</td><td>128 tokens + 4d</td><td>RoBERTa 12L (768d)</td><td>MLP(772&rarr;256&rarr;3)</td><td>3</td><td>~125M</td><td>~10min</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="e2e-decision"><strong><span class="ko">설계 결정:</span><span class="en">Design decision:</span></strong> <span class="ko">batch_size=64는 M3 Max 64GB에서 MPS 메모리 30-48GB를 소비. gradient accumulation 없이 안정적으로 학습 가능한 최대 크기.</span><span class="en">batch_size=64 consumes 30-48GB MPS memory on M3 Max 64GB. Maximum stable size without gradient accumulation.</span></div>
        <div class="e2e-bottleneck"><strong><span class="ko">병목:</span><span class="en">Bottleneck:</span></strong> <span class="ko">Transformer 학습은 시드당 시간이 커서 정식 v2 120회 실행 전 A_B/D_B seed 42 smoke gate를 먼저 통과해야 한다.</span><span class="en">Transformer training is expensive, so the formal v2 120-run benchmark must pass the A_B/D_B seed 42 smoke gate first.</span></div>
      </div>

      <div class="e2e-connector"><span class="arrow">&darr;</span><span class="transform"><span class="ko">추론 (Forward Pass)</span><span class="en">Inference (Forward Pass)</span></span></div>

      <!-- Node 6: Inference -->
      <div class="e2e-node">
        <div class="e2e-node-header">
          <div class="e2e-node-title" style="color:var(--accent-purple)"><span class="ko">6. 추론 차원 변환 상세</span><span class="en">6. Inference Dimension Transformation Detail</span></div>
        </div>
        <div class="ko-block desc" style="margin-bottom:12px">RoBERTa+VADER(최고 모델)의 단일 텍스트 추론 과정을 차원 단위로 추적한다.</div>
        <div class="en-block desc" style="margin-bottom:12px">Traces single-text inference through RoBERTa+VADER (best model) dimension by dimension.</div>

        <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:center;font-size:0.85rem">
          <div style="background:var(--bg-card);padding:10px 16px;border-radius:8px;border:1px solid var(--border-subtle);text-align:center">
            <div style="font-weight:700;color:var(--text-primary)">Text</div>
            <div style="color:var(--text-secondary);font-size:0.78rem">"I hate them all"</div>
          </div>
          <div style="color:var(--accent-blue);font-size:1.2rem">&rarr;</div>
          <div style="background:var(--bg-card);padding:10px 16px;border-radius:8px;border:1px solid var(--border-subtle);text-align:center">
            <div style="font-weight:700;color:var(--accent-blue)">Tokenizer</div>
            <div style="color:var(--text-secondary);font-size:0.78rem">[1, 128] int64</div>
          </div>
          <div style="color:var(--accent-blue);font-size:1.2rem">&rarr;</div>
          <div style="background:var(--bg-card);padding:10px 16px;border-radius:8px;border:2px solid var(--accent-green);text-align:center">
            <div style="font-weight:700;color:var(--accent-green)">RoBERTa</div>
            <div style="color:var(--text-secondary);font-size:0.78rem">[1, 768] float32</div>
          </div>
          <div style="color:var(--accent-blue);font-size:1.2rem">&rarr;</div>
          <div style="background:var(--bg-card);padding:10px 16px;border-radius:8px;border:2px solid var(--accent-orange);text-align:center">
            <div style="font-weight:700;color:var(--accent-orange)">+ VADER</div>
            <div style="color:var(--text-secondary);font-size:0.78rem">[1, 772] float32</div>
          </div>
          <div style="color:var(--accent-blue);font-size:1.2rem">&rarr;</div>
          <div style="background:var(--bg-card);padding:10px 16px;border-radius:8px;border:1px solid var(--border-subtle);text-align:center">
            <div style="font-weight:700;color:var(--accent-purple)">MLP</div>
            <div style="color:var(--text-secondary);font-size:0.78rem">[1, 256] &rarr; [1, 3]</div>
          </div>
          <div style="color:var(--accent-blue);font-size:1.2rem">&rarr;</div>
          <div style="background:var(--bg-card);padding:10px 16px;border-radius:8px;border:2px solid var(--accent-green);text-align:center">
            <div style="font-weight:700;color:var(--accent-green)">Softmax</div>
            <div style="color:var(--text-secondary);font-size:0.78rem">[0.12, 0.23, 0.65]</div>
          </div>
        </div>

        <div class="e2e-decision" style="margin-top:14px"><strong><span class="ko">핵심 관찰:</span><span class="en">Key observation:</span></strong> <span class="ko">768차원 [CLS] 벡터에 단 4차원(0.5%)을 추가. 이 미세한 감성 신호가 RoBERTa 인코더와 결합했을 때만 통계적으로 유의미한 차이를 만듦.</span><span class="en">Just 4 dims (0.5%) added to 768-dim [CLS] vector. This subtle sentiment signal creates statistically significant difference only when combined with RoBERTa encoder.</span></div>
      </div>

      <div class="e2e-connector"><span class="arrow">&darr;</span><span class="transform"><span class="ko">평가 + XAI</span><span class="en">Evaluation + XAI</span></span></div>

      <!-- Node 7: Evaluation -->
      <div class="e2e-node" style="border-color:var(--accent-green)">
        <div class="e2e-node-header">
          <div class="e2e-node-title" style="color:var(--accent-green)"><span class="ko">7. 평가 메트릭 산출</span><span class="en">7. Evaluation Metrics</span></div>
        </div>
        <div class="e2e-metrics">
          <div class="e2e-metric"><div class="val" style="color:var(--accent-green)">0.6863</div><div class="lbl">Macro F1 (Best)</div></div>
          <div class="e2e-metric"><div class="val">0.7258</div><div class="lbl">Accuracy</div></div>
          <div class="e2e-metric"><div class="val">0.8400</div><div class="lbl">AUROC</div></div>
          <div class="e2e-metric"><div class="val">p=0.037</div><div class="lbl"><span class="ko">통계적 유의성</span><span class="en">Statistical Sig.</span></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ---- 처리 시간 분석 ---- -->
  <div class="card">
    <h3><span class="ko">처리 시간 분석 (Pipeline Profiling)</span><span class="en">Processing Time Analysis (Pipeline Profiling)</span></h3>
    <div class="ko-block desc">각 파이프라인 단계별 예상 실행 시간이다. M3 Max 64GB, MPS 가속 기준.</div>
    <div class="en-block desc">Estimated execution time per pipeline stage. Based on M3 Max 64GB with MPS acceleration.</div>

    <div style="display:flex;flex-direction:column;gap:10px;margin-top:14px">
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)"><span class="ko">데이터 전처리</span><span class="en">Data preprocessing</span></span>
        <div class="e2e-time-bar" style="width:2%;background:var(--accent-blue)">5s</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)">VADER</span>
        <div class="e2e-time-bar" style="width:3%;background:var(--accent-orange)">10s</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)">EDA</span>
        <div class="e2e-time-bar" style="width:5%;background:var(--accent-purple)">30s</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)">TF-IDF + ML</span>
        <div class="e2e-time-bar" style="width:3%;background:var(--accent-blue)">15s</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)"><span class="ko">하이퍼파라미터 튜닝</span><span class="en">Hyperparameter Tuning</span></span>
        <div class="e2e-time-bar" style="width:55%;background:var(--accent-red)">~2h</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)"><span class="ko">벤치마크 (8조건x15시드)</span><span class="en">Benchmark (8x15)</span></span>
        <div class="e2e-time-bar" style="width:65%;background:var(--accent-green)">server batch</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)">Freeze Study</span>
        <div class="e2e-time-bar" style="width:20%;background:var(--accent-purple)">~45min</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)">XAI (SHAP+LIME)</span>
        <div class="e2e-time-bar" style="width:15%;background:var(--accent-orange)">~30min</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:160px;text-align:right;font-size:0.85rem;color:var(--text-secondary)"><span class="ko">대시보드 생성</span><span class="en">Dashboard</span></span>
        <div class="e2e-time-bar" style="width:1%;background:var(--accent-blue);min-width:40px">2s</div>
      </div>
    </div>

    <div class="insight insight-red" style="margin-top:16px">
      <span class="ko"><strong>총 소요 시간: 약 4-5시간.</strong> 튜닝(2h)과 벤치마크(2.5h)가 전체의 90% 이상을 차지한다. TF-IDF 모델은 수 초 만에 학습 가능하나, Transformer 모델의 반복 학습이 병목이다. GPU 자원이 제한된 환경에서는 튜닝 범위를 줄이거나 시드 수를 조정해야 한다.</span>
      <span class="en"><strong>Total time: ~4-5 hours.</strong> Tuning (2h) and benchmark (2.5h) account for 90%+. TF-IDF trains in seconds, but repeated Transformer training is the bottleneck. In GPU-limited environments, reduce tuning range or seed count.</span>
    </div>
  </div>

  <!-- ---- 데이터 손실/변환 추적 ---- -->
  <div class="card">
    <h3><span class="ko">데이터 손실/변환 추적 (Data Funnel)</span><span class="en">Data Loss/Transformation Tracking (Data Funnel)</span></h3>
    <div class="ko-block desc">원본에서 최종 평가까지 샘플이 어떻게 줄어들고 변환되는지 퍼널 형태로 시각화한다.</div>
    <div class="en-block desc">Visualizes how samples are reduced and transformed from source to final evaluation as a funnel.</div>

    <div style="display:flex;flex-direction:column;align-items:center;gap:0;margin:20px 0">
      <div style="width:90%;background:rgba(124,138,255,0.15);padding:14px;border-radius:10px 10px 0 0;text-align:center;border:1px solid var(--accent-blue)">
        <div style="font-weight:700;font-size:1.1rem;color:var(--accent-blue)">19,192</div>
        <div style="font-size:0.8rem;color:var(--text-secondary)"><span class="ko">HateXplain 원본</span><span class="en">HateXplain Raw</span></div>
      </div>
      <div style="width:75%;background:rgba(255,107,122,0.1);padding:6px;text-align:center;font-size:0.78rem;color:var(--accent-red)">
        <span class="ko">&darr; -5,759 (majority vote 미달)</span><span class="en">&darr; -5,759 (no majority vote)</span>
      </div>
      <div style="width:75%;background:rgba(0,229,176,0.15);padding:14px;text-align:center;border:1px solid var(--accent-green)">
        <div style="font-weight:700;font-size:1.1rem;color:var(--accent-green)">13,433</div>
        <div style="font-size:0.8rem;color:var(--text-secondary)"><span class="ko">유효 샘플 + VADER 증강</span><span class="en">Valid samples + VADER augmented</span></div>
      </div>
      <div style="width:60%;background:rgba(181,122,255,0.1);padding:6px;text-align:center;font-size:0.78rem;color:var(--accent-purple)">
        <span class="ko">&darr; Stratified Split (60/20/20)</span><span class="en">&darr; Stratified Split (60/20/20)</span>
      </div>
      <div style="width:60%;display:grid;grid-template-columns:3fr 1fr 1fr;gap:4px">
        <div style="background:rgba(124,138,255,0.15);padding:12px;text-align:center;border-radius:0 0 0 10px;border:1px solid var(--accent-blue)">
          <div style="font-weight:700;color:var(--accent-blue)">8,060</div>
          <div style="font-size:0.78rem;color:var(--text-secondary)">Train</div>
        </div>
        <div style="background:rgba(255,179,71,0.15);padding:12px;text-align:center;border:1px solid var(--accent-orange)">
          <div style="font-weight:700;color:var(--accent-orange)">2,687</div>
          <div style="font-size:0.78rem;color:var(--text-secondary)">Val</div>
        </div>
        <div style="background:rgba(0,229,176,0.15);padding:12px;text-align:center;border-radius:0 0 10px 0;border:1px solid var(--accent-green)">
          <div style="font-weight:700;color:var(--accent-green)">2,686</div>
          <div style="font-size:0.78rem;color:var(--text-secondary)">Test</div>
        </div>
      </div>
    </div>

    <div class="insight insight-orange">
      <span class="ko"><strong>30% 데이터 손실의 의미:</strong> majority vote로 5,759개 샘플이 제거된다. 이 샘플들은 annotator 간 의견이 갈린 "경계 사례"로, 제거하면 라벨 품질은 높아지지만 모델이 모호한 사례를 학습할 기회를 잃는다. 이는 트레이드오프이다.</span>
      <span class="en"><strong>Meaning of 30% data loss:</strong> 5,759 samples removed by majority vote are "borderline cases" where annotators disagreed. Removal improves label quality but loses opportunity to learn ambiguous cases. This is a trade-off.</span>
    </div>
  </div>

  <!-- ---- 기술 스택 매핑 ---- -->
  <div class="card">
    <h3><span class="ko">기술 스택 매핑</span><span class="en">Technology Stack Mapping</span></h3>
    <div class="ko-block desc">파이프라인 각 단계에서 사용되는 라이브러리와 하드웨어 자원을 매핑한다.</div>
    <div class="en-block desc">Maps libraries and hardware resources used at each pipeline stage.</div>

    <div style="overflow-x:auto">
      <table class="data-table" style="font-size:0.85rem">
        <thead>
          <tr>
            <th><span class="ko">단계</span><span class="en">Stage</span></th>
            <th><span class="ko">주요 라이브러리</span><span class="en">Key Libraries</span></th>
            <th><span class="ko">연산 장치</span><span class="en">Compute</span></th>
            <th><span class="ko">메모리 사용</span><span class="en">Memory Usage</span></th>
            <th><span class="ko">핵심 파라미터</span><span class="en">Key Parameters</span></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="ko">데이터 전처리</span><span class="en">Data Preprocessing</span></td>
            <td>pandas, json</td><td>CPU</td><td>&lt;1GB</td><td>majority_vote threshold=2/3</td>
          </tr>
          <tr>
            <td>VADER</td>
            <td>vaderSentiment</td><td>CPU</td><td>&lt;1GB</td><td>compound, pos, neg, neu</td>
          </tr>
          <tr>
            <td>EDA</td>
            <td>matplotlib, seaborn, scikit-learn</td><td>CPU</td><td>~2GB</td><td>top_n_words=500, Jaccard</td>
          </tr>
          <tr>
            <td>TF-IDF + ML</td>
            <td>scikit-learn</td><td>CPU</td><td>~3GB</td><td>max_features=30000, C=1.0</td>
          </tr>
          <tr>
            <td>Transformer</td>
            <td>transformers, PyTorch</td><td style="color:var(--accent-green);font-weight:600">MPS (M3 Max)</td><td style="color:var(--accent-orange)">30-48GB</td><td>lr=2e-5, batch=64, epochs=5</td>
          </tr>
          <tr>
            <td>XAI - SHAP</td>
            <td>shap</td><td style="color:var(--accent-red);font-weight:600">CPU only</td><td>~8GB</td><td><span class="ko">MPS 비호환 -- CPU 강제</span><span class="en">MPS incompatible -- forced CPU</span></td>
          </tr>
          <tr>
            <td>XAI - LIME</td>
            <td>lime</td><td>CPU + MPS</td><td>~4GB</td><td>num_samples=300, num_features=8</td>
          </tr>
          <tr>
            <td><span class="ko">대시보드</span><span class="en">Dashboard</span></td>
            <td>FastAPI, Chart.js, uvicorn</td><td>CPU</td><td>&lt;500MB</td><td>port=8501</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="e2e-bottleneck" style="margin-top:14px"><strong><span class="ko">SHAP의 CPU 제약:</span><span class="en">SHAP CPU constraint:</span></strong> <span class="ko">SHAP 라이브러리는 MPS를 지원하지 않아 CPU에서 실행해야 한다. 이로 인해 XAI 분석 시간이 GPU 가속 대비 3-5배 느려진다. 이것이 XAI 샘플 수를 24개로 제한한 이유이다.</span><span class="en">SHAP library doesn't support MPS, requiring CPU execution. This makes XAI analysis 3-5x slower than GPU. This is why XAI sample count is limited to 24.</span></div>
  </div>

  <!-- ---- 파이프라인 재현 명령어 ---- -->
  <div class="card">
    <h3><span class="ko">파이프라인 실행 명령어</span><span class="en">Pipeline Execution Commands</span></h3>
    <div class="ko-block desc">전체 파이프라인을 재현하기 위한 명령어 체계이다.</div>
    <div class="en-block desc">Command system for reproducing the full pipeline.</div>

    <div style="background:var(--bg-dark);padding:20px;border-radius:10px;font-family:'Fira Code',monospace;font-size:0.85rem;overflow-x:auto;border:1px solid var(--border-subtle)">
      <div style="color:var(--text-secondary);margin-bottom:8px"># <span class="ko">전체 파이프라인 한 번에 실행</span><span class="en">Run full pipeline at once</span></div>
      <div style="color:var(--accent-green)">./run.sh full</div>
      <br>
      <div style="color:var(--text-secondary);margin-bottom:8px"># <span class="ko">단계별 실행 (순서 중요)</span><span class="en">Step-by-step (order matters)</span></div>
      <div style="color:var(--accent-blue)">./run.sh data      </div><span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">1. 데이터 전처리 + VADER</span><span class="en">1. Data preprocessing + VADER</span></span><br>
      <div style="color:var(--accent-blue)">./run.sh eda       </div><span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">2. 탐색적 데이터 분석</span><span class="en">2. Exploratory data analysis</span></span><br>
      <div style="color:var(--accent-blue)">./run.sh tune      </div><span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">3. 하이퍼파라미터 튜닝 (~2시간)</span><span class="en">3. Hyperparameter tuning (~2h)</span></span><br>
      <div style="color:var(--accent-blue)">./run.sh e2e benchmark --run-id v2_15seed --execute --resume</div><span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">4. 8조건 x 15시드 벤치마크</span><span class="en">4. 8 conditions x 15 seeds benchmark</span></span><br>
      <div style="color:var(--accent-blue)">./run.sh freeze    </div><span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">5. Freeze Study (~45분)</span><span class="en">5. Freeze Study (~45min)</span></span><br>
      <div style="color:var(--accent-blue)">./run.sh xai       </div><span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">6. SHAP + LIME 분석 (~30분)</span><span class="en">6. SHAP + LIME analysis (~30min)</span></span><br>
      <div style="color:var(--accent-blue)">./run.sh dashboard </div><span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">7. 대시보드 생성</span><span class="en">7. Dashboard generation</span></span><br>
      <br>
      <div style="color:var(--text-secondary);margin-bottom:8px"># <span class="ko">대시보드 서버 실행</span><span class="en">Launch dashboard server</span></div>
      <div style="color:var(--accent-green)">python3 dashboard_app.py</div>
      <span style="color:var(--text-secondary);font-size:0.78rem"># <span class="ko">http://localhost:8501 접속</span><span class="en">Access http://localhost:8501</span></span>
    </div>

    <div class="insight insight-blue" style="margin-top:14px">
      <span class="ko"><strong>의존성 체계:</strong> data &rarr; (eda, vader) &rarr; tune &rarr; benchmark &rarr; freeze &rarr; xai &rarr; dashboard. 각 단계의 출력이 다음 단계의 입력이 되므로 순서를 지켜야 한다. <code>./run.sh full</code>이 자동으로 순서를 보장한다.</span>
      <span class="en"><strong>Dependency chain:</strong> data &rarr; (eda, vader) &rarr; tune &rarr; benchmark &rarr; freeze &rarr; xai &rarr; dashboard. Each stage's output feeds the next, so order matters. <code>./run.sh full</code> guarantees correct ordering.</span>
    </div>
  </div>
</div>

<!-- ================================================================
     TAB 2: Benchmark
     ================================================================ -->
<div id="tab-benchmark" class="tab-content">
  <div class="card">
    <h2><span class="ko">모델 벤치마크 결과</span><span class="en">Model Benchmark Results</span></h2>
    <div class="ko-block desc">
      8개 ablation 조건의 성능을 15개 seed에 걸쳐 평가한 결과이다.
      Macro F1은 클래스 불균형 상황에서 각 클래스를 동등하게 반영하는 지표로, 이 연구의 주요 평가 메트릭이다.
      표준편차가 작을수록 모델의 안정성이 높음을 의미한다.
    </div>
    <div class="en-block desc">
      Performance of 8 ablation conditions evaluated across 15 seeds.
      Macro F1 equally weighs each class under class imbalance, serving as the primary evaluation metric.
      Smaller standard deviation indicates higher model stability.
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3><span class="ko">Macro F1 비교 (오차 막대 포함)</span><span class="en">Macro F1 Comparison (with error bars)</span></h3>
      <div class="ko-block desc">각 모델의 평균 Macro F1과 표준편차를 수평 바 차트로 나타낸다. 오차 막대가 짧을수록 시드 간 일관성이 높다.</div>
      <div class="en-block desc">Horizontal bar chart showing mean Macro F1 with standard deviation error bars. Shorter bars indicate higher cross-seed consistency.</div>
      <div class="chart-container"><canvas id="chartBenchmarkF1"></canvas></div>
    </div>
    <div class="card">
      <h3><span class="ko">레이더 차트 - 상위 4개 모델</span><span class="en">Radar Chart - Top 4 Models</span></h3>
      <div class="ko-block desc">상위 4개 Transformer 모델을 6개 지표(Macro F1, Accuracy, AUROC, Hate F1, Offensive F1, Normal F1)에서 비교한다.</div>
      <div class="en-block desc">Compares top 4 Transformer models across 6 metrics. Larger area indicates better overall performance.</div>
      <div class="chart-container"><canvas id="chartRadar"></canvas></div>
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">클래스별 F1 Score</span><span class="en">Per-Class F1 Score</span></h3>
    <div class="ko-block desc">
      각 모델이 3개 클래스(hatespeech, offensive, normal)를 각각 얼마나 잘 분류하는지 보여준다.
      혐오표현 탐지에서 가장 중요한 것은 hate 클래스의 높은 F1이지만, offensive 클래스 분류가
      모든 모델에서 공통적으로 어려운 과제이다.
    </div>
    <div class="en-block desc">
      Shows how well each model classifies each of the 3 classes. While high hate F1 is most important,
      offensive class classification remains a shared challenge across all models.
    </div>
    <div class="chart-container"><canvas id="chartPerClassF1"></canvas></div>
  </div>

  <div class="insight insight-green">
    <span class="ko"><strong>RoBERTa+VADER가 hate F1에서 0.775로 최고 성적:</strong> BERT-base(0.766)보다 약 1%p 높다. RoBERTa의 사전학습이 혐오 맥락 파악에 유리하며, VADER가 부정적 감성 신호를 보강한다.</span>
    <span class="en"><strong>RoBERTa+VADER leads with 0.775 hate F1:</strong> About 1%p higher than BERT-base (0.766). RoBERTa's pre-training is advantageous for hate context understanding, and VADER reinforces negative sentiment signals.</span>
  </div>
  <div class="insight insight-red">
    <span class="ko"><strong>Offensive F1은 모든 모델에서 0.53~0.55:</strong> 이는 hate/offensive 간 어휘 Jaccard 유사도가 0.7094에 달하기 때문이다. 두 클래스의 언어적 특성이 매우 유사하여 구별이 본질적으로 어렵다.</span>
    <span class="en"><strong>Offensive F1 is 0.53-0.55 across all models:</strong> Due to hate/offensive vocabulary Jaccard similarity of 0.7094. Linguistic characteristics are inherently similar.</span>
  </div>

  <div class="card">
    <h3><span class="ko">상세 성능 테이블</span><span class="en">Detailed Performance Table</span></h3>
    <div class="ko-block desc">모든 조건의 주요 평가 지표를 표로 정리하였다. 각 값은 15회 seed 평균 +/- 표준편차이다.</div>
    <div class="en-block desc">All condition evaluation metrics in tabular form. Each value is the mean +/- standard deviation across 15 seeds.</div>
    <div style="overflow-x:auto">
      <table class="data-table">
        <thead>
          <tr>
            <th><span class="ko">모델</span><span class="en">Model</span></th>
            <th>Macro F1</th><th>Accuracy</th><th>AUROC</th>
            <th>Hate F1</th><th>Offensive F1</th><th>Normal F1</th>
          </tr>
        </thead>
        <tbody>{benchmark_table_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- ================================================================
     TAB 3: Statistical Tests
     ================================================================ -->
<div id="tab-statistical" class="tab-content">
  <div class="card">
    <h2><span class="ko">통계적 유의성 검정</span><span class="en">Statistical Significance Tests</span></h2>
    <div class="ko-block desc">
      모든 모델 쌍에 대해 Paired t-test를 수행하여 성능 차이가 우연에 의한 것인지 검정하였다.
      유의수준 alpha=0.05를 기준으로, p-value가 이보다 작으면 통계적으로 유의미하다고 판단한다.
      Cohen's d는 효과 크기를 나타내며, |d| > 0.8이면 큰 효과로 해석한다.
    </div>
    <div class="en-block desc">
      Paired t-tests for all model pairs. At alpha=0.05, lower p-values indicate significant differences.
      Cohen's d measures effect size; |d| > 0.8 is a large effect.
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">유의성 검정 결과 테이블</span><span class="en">Significance Test Results Table</span></h3>
    <div class="ko-block desc">녹색은 통계적으로 유의미한 차이(p &lt; 0.05), 회색은 유의미하지 않은 차이를 나타낸다.</div>
    <div class="en-block desc">Green indicates significant (p &lt; 0.05), gray indicates non-significant.</div>
    <div style="overflow-x:auto">
      <table class="data-table">
        <thead>
          <tr>
            <th><span class="ko">모델 A</span><span class="en">Model A</span></th>
            <th><span class="ko">모델 B</span><span class="en">Model B</span></th>
            <th><span class="ko">평균 차이</span><span class="en">Mean Diff</span></th>
            <th>t-statistic</th><th>p-value</th><th>Cohen's d</th>
            <th><span class="ko">유의성</span><span class="en">Significant</span></th>
          </tr>
        </thead>
        <tbody>{sig_table_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">P-value 히트맵</span><span class="en">P-value Heatmap</span></h3>
    <div class="ko-block desc">모델 쌍 간 p-value를 색상으로 표현한다. 진한 녹색일수록 유의미한 차이(낮은 p-value)이다.</div>
    <div class="en-block desc">P-values visualized as colors. Darker green = more significant difference.</div>
    <div class="chart-container"><canvas id="chartPvalueHeatmap" width="660" height="420"></canvas></div>
  </div>

  <div class="insight insight-green">
    <span class="ko"><strong>paired test와 Holm 보정:</strong> 정식 v2 report는 p-value만이 아니라 mean difference, 95% CI, Cohen's dz를 함께 보고한다.</span>
    <span class="en"><strong>Paired tests with Holm correction:</strong> The formal v2 report includes mean difference, 95% CI, and Cohen's dz alongside p-values.</span>
  </div>
  <div class="insight insight-orange">
    <span class="ko"><strong>BERT+VADER vs BERT-base: p=0.1384</strong> -- 유의미하지 않다. 같은 BERT 인코더에 VADER를 추가하는 것만으로는 유의미한 향상 불가. 인코더 자체의 표현력이 더 결정적이다.</span>
    <span class="en"><strong>BERT+VADER vs BERT-base: p=0.1384</strong> -- Not significant. Adding VADER to the same BERT encoder alone is insufficient. Encoder quality is the decisive factor.</span>
  </div>
  <div class="insight insight-orange">
    <span class="ko"><strong>BERT+MLP vs BERT+VADER: p=0.5666</strong> -- 유의미하지 않다. Ablation 통제로서 두 아키텍처의 성능이 거의 동일함을 확인.</span>
    <span class="en"><strong>BERT+MLP vs BERT+VADER: p=0.5666</strong> -- Not significant. Confirms near-identical performance as ablation control.</span>
  </div>
</div>

<!-- ================================================================
     TAB 4: Tuning
     ================================================================ -->
<div id="tab-tuning" class="tab-content">
  <div class="card">
    <h2><span class="ko">하이퍼파라미터 튜닝</span><span class="en">Hyperparameter Tuning</span></h2>
    <div class="ko-block desc">
      3개 Transformer 모델에 대해 학습률과 드롭아웃을 체계적으로 탐색하였다.
      그리드 서치로 학습률 [1e-5, 2e-5, 3e-5], 드롭아웃 [0.1, 0.2, 0.3]을 시드 42로 검증하였다.
    </div>
    <div class="en-block desc">
      Systematic grid search over learning rates [1e-5, 2e-5, 3e-5] and dropouts [0.1, 0.2, 0.3] with seed 42.
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">최적 하이퍼파라미터</span><span class="en">Optimal Hyperparameters</span></h3>
    <div class="ko-block desc">모든 모델에서 dropout=0.1이 최적이었다.</div>
    <div class="en-block desc">Dropout=0.1 was optimal across all models.</div>
    <table class="data-table">
      <thead>
        <tr>
          <th><span class="ko">모델</span><span class="en">Model</span></th>
          <th>Learning Rate</th><th>Dropout</th><th>Batch Size</th><th>Epochs</th>
        </tr>
      </thead>
      <tbody>{tuning_best_rows}</tbody>
    </table>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3><span class="ko">학습률 탐색 궤적</span><span class="en">Learning Rate Search Trajectory</span></h3>
      <div class="ko-block desc">각 모델에서 학습률 후보별 검증 Macro F1. 최적 학습률은 모델마다 다를 수 있다.</div>
      <div class="en-block desc">Validation Macro F1 for each learning rate candidate per model.</div>
      <div class="chart-container"><canvas id="chartTuningLR"></canvas></div>
    </div>
    <div class="card">
      <h3><span class="ko">드롭아웃 탐색 궤적</span><span class="en">Dropout Search Trajectory</span></h3>
      <div class="ko-block desc">드롭아웃 비율에 따른 검증 성능 변화이다.</div>
      <div class="en-block desc">Validation performance changes with dropout rate.</div>
      <div class="chart-container"><canvas id="chartTuningDropout"></canvas></div>
    </div>
  </div>

  <div class="insight insight-blue">
    <span class="ko"><strong>RoBERTa+VADER는 lr=2e-5에서 F1=0.6833으로 최고:</strong> lr=1e-5(0.6716)과 lr=3e-5(0.6717) 대비 약 1.2%p 차이. 학습률 선택이 성능에 상당한 영향을 미친다.</span>
    <span class="en"><strong>RoBERTa+VADER peaks at lr=2e-5 with F1=0.6833:</strong> ~1.2%p gap vs lr=1e-5 (0.6716) and lr=3e-5 (0.6717).</span>
  </div>
  <div class="insight insight-green">
    <span class="ko"><strong>Dropout=0.1이 전 모델 최적:</strong> 5 에폭의 짧은 학습에서는 과적합 위험이 낮아 강한 정규화가 불필요하다.</span>
    <span class="en"><strong>Dropout=0.1 optimal for all:</strong> Short 5-epoch training has low overfitting risk, making strong regularization unnecessary.</span>
  </div>
</div>

<!-- ================================================================
     TAB 5: Learning Curves
     ================================================================ -->
<div id="tab-learning" class="tab-content">
  <div class="card">
    <h2><span class="ko">학습 곡선</span><span class="en">Learning Curves</span></h2>
    <div class="ko-block desc">
      각 모델의 에폭별 학습 손실(train loss)과 검증 Macro F1의 변화를 추적한다.
      학습 손실이 지속적으로 감소하면서 검증 F1이 정체되거나 하락하면 과적합의 신호이다.
      15개 seed의 곡선을 겹쳐서 모델의 학습 안정성을 확인할 수 있다.
    </div>
    <div class="en-block desc">
      Tracks per-epoch training loss and validation Macro F1. Overlapping curves from 15 seeds reveal training stability.
    </div>
    <label for="lcModelSelect">
      <span class="ko">모델 선택: </span><span class="en">Select Model: </span>
    </label>
    <select id="lcModelSelect" onchange="updateLearningCurveChart()">
      <option value="bert_base">BERT-base</option>
      <option value="bert_mlp">BERT+MLP</option>
      <option value="bert_vader">BERT+VADER</option>
      <option value="roberta_vader" selected>RoBERTa+VADER</option>
    </select>
  </div>
  <div class="card">
    <div class="chart-container"><canvas id="chartLearningCurve"></canvas></div>
  </div>
  <div class="insight insight-blue">
    <span class="ko"><strong>Transformer 모델들은 대부분 2~3 에폭에서 최고 F1 도달:</strong> 이후 검증 F1이 정체되거나 미세하게 하락한다. Early stopping patience=2가 적절한 설정이다.</span>
    <span class="en"><strong>Most Transformers peak at epoch 2-3:</strong> Validation F1 then plateaus or slightly declines. Early stopping patience=2 is appropriate.</span>
  </div>
  <div class="insight insight-orange">
    <span class="ko"><strong>학습 손실은 지속 하락, 검증 성능은 정체:</strong> 전형적 과적합 신호이나, 5 에폭이라는 짧은 학습 기간 덕분에 심각한 과적합은 방지되었다.</span>
    <span class="en"><strong>Train loss keeps dropping while val performance plateaus:</strong> Classic overfitting signal, but the short 5-epoch training prevented severe overfitting.</span>
  </div>
</div>

<!-- ================================================================
     TAB 6: Freeze Study
     ================================================================ -->
<div id="tab-freeze" class="tab-content">
  <div class="card">
    <h2><span class="ko">Freeze Study: 인코더 동결 vs 미세조정</span><span class="en">Freeze Study: Frozen vs Fine-tuned Encoder</span></h2>
    <div class="ko-block desc">
      BERT+VADER 모델의 인코더 가중치를 동결했을 때와 미세조정했을 때의 성능 차이를 비교하는 절제 실험(ablation study)이다.
      동결 시 인코더는 범용 언어 표현만 사용하고, 미세조정 시 혐오표현 도메인에 특화된다.
    </div>
    <div class="en-block desc">
      Ablation study comparing frozen vs fine-tuned BERT+VADER encoder. Frozen uses general representations;
      fine-tuned specializes for hate speech.
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3><span class="ko">동결 vs 미세조정 바 차트</span><span class="en">Frozen vs Fine-tuned Bar Chart</span></h3>
      <div class="chart-container"><canvas id="chartFreeze"></canvas></div>
    </div>
    <div class="card">
      <h3><span class="ko">시드별 상세 결과</span><span class="en">Per-Seed Details</span></h3>
      <div class="ko-block desc">각 시드에서의 Macro F1과 Accuracy를 비교한다.</div>
      <div class="en-block desc">Per-seed Macro F1 and Accuracy comparison.</div>
      <table class="data-table">
        <thead>
          <tr>
            <th><span class="ko">모델</span><span class="en">Model</span></th>
            <th>Seed</th><th>Macro F1</th><th>Accuracy</th>
          </tr>
        </thead>
        <tbody>{freeze_table_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="insight insight-green">
    <span class="ko"><strong>미세조정 시 Macro F1이 약 109% 향상:</strong> 동결 평균 F1={frozen_mean:.4f}, 미세조정 평균 F1={finetuned_mean:.4f}.
    사전학습 인코더가 범용 표현만으로는 혐오표현의 미묘한 뉘앙스를 포착하기 어려우며, 도메인 특화 미세조정이 필수적이다.</span>
    <span class="en"><strong>Fine-tuning improves Macro F1 by ~109%:</strong> Frozen mean F1={frozen_mean:.4f}, fine-tuned mean F1={finetuned_mean:.4f}.
    Pre-trained encoders need domain-specific fine-tuning to capture subtle hate speech nuances.</span>
  </div>
  <div class="insight insight-red">
    <span class="ko"><strong>동결 인코더의 F1이 ~0.32 수준:</strong> random 수준(0.33)에 가까운 성능으로, MLP 헤드와 VADER 입력만으로는 분류가 거의 불가능하다. Transformer 인코더의 문맥적 표현 학습이 핵심 성능 요인이다.</span>
    <span class="en"><strong>Frozen encoder F1 at ~0.32:</strong> Near random-level (0.33). MLP head and VADER alone are nearly incapable of classification, proving contextual representation learning is the critical factor.</span>
  </div>
</div>

<!-- ================================================================
     TAB 7: EDA
     ================================================================ -->
<div id="tab-eda" class="tab-content">
  <div class="card">
    <h2><span class="ko">탐색적 데이터 분석 (EDA)</span><span class="en">Exploratory Data Analysis (EDA)</span></h2>
    <div class="ko-block desc">
      HateXplain 데이터셋(총 {eda.get('total_samples', 19192):,}개 텍스트)의 특성을 9가지 관점에서 분석하였다.
      데이터의 분포, 감성 패턴, 어휘 중첩도, N-gram 패턴, 타겟 커뮤니티, Human Rationale 분포 등을 파악하여
      모델 설계와 결과 해석에 필요한 기초 지식을 제공한다.
    </div>
    <div class="en-block desc">
      9-aspect analysis of the HateXplain dataset ({eda.get('total_samples', 19192):,} texts).
      Covers distribution, sentiment, vocabulary overlap, N-grams, target communities, and human rationale patterns.
    </div>
  </div>

  <!-- 1. 클래스 분포 -->
  <div class="grid-2">
    <div class="card">
      <h3><span class="ko">1. 클래스 분포</span><span class="en">1. Class Distribution</span></h3>
      <div class="ko-block desc">데이터셋의 라벨 분포와 불균형 비율. 불균형 비율이 1.42:1로 경미한 수준이나, balanced class weight를 적용하여 대응한다.</div>
      <div class="en-block desc">Label distribution and imbalance ratio. 1.42:1 ratio is mild but addressed with balanced class weights.</div>
      <table class="data-table">
        <thead><tr><th><span class="ko">클래스</span><span class="en">Class</span></th><th><span class="ko">샘플 수</span><span class="en">Count</span></th><th><span class="ko">비율</span><span class="en">Ratio</span></th></tr></thead>
        <tbody>{class_dist_rows}</tbody>
      </table>
      <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">
        <span class="ko">불균형 비율: <strong>{eda.get('imbalance_ratio', 'N/A')}:1</strong></span>
        <span class="en">Imbalance ratio: <strong>{eda.get('imbalance_ratio', 'N/A')}:1</strong></span>
      </div>
    </div>
    <div class="card">
      <img src="/static/eda/class_distribution.png" alt="Class distribution" loading="lazy" style="width:100%;border-radius:8px;cursor:pointer" onclick="openLightbox(this.src)">
    </div>
  </div>
  <div class="insight insight-blue">
    <span class="ko"><strong>불균형 비율 1.42:1:</strong> offensive(8,336)가 가장 많고 normal(5,862)이 가장 적다. imbalance_threshold=0.40 기준 경미한 불균형으로 판정되어 balanced class weight + label smoothing(0.1)을 자동 적용한다.</span>
    <span class="en"><strong>Imbalance ratio 1.42:1:</strong> Offensive is most common (8,336), normal is least (5,862). Detected as mild imbalance; balanced class weights + label smoothing (0.1) applied automatically.</span>
  </div>

  <!-- 2. VADER 감성 분포 -->
  <div class="card">
    <h3><span class="ko">2. VADER 감성 분포 (클래스별)</span><span class="en">2. VADER Sentiment Distribution (by class)</span></h3>
    <div class="ko-block desc">VADER의 compound, negative, positive 평균값을 클래스별로 비교한다. 혐오표현이 더 부정적인 감성을 띨 것이라는 가설의 근거이다.</div>
    <div class="en-block desc">VADER compound, negative, and positive mean scores by class, providing evidence for the sentiment hypothesis.</div>
    <div class="chart-container"><canvas id="chartVader"></canvas></div>
  </div>
  <div class="insight insight-green">
    <span class="ko"><strong>VADER compound 평균: hate=-0.358, offensive=-0.283, normal=-0.181</strong> -- 혐오표현일수록 부정적 감성이 강하다. 이 경향성이 VADER를 보조 특성으로 활용하는 근거이다. 다만, 세 클래스 모두 부정적 영역에 있어 VADER만으로는 분류가 어렵다.</span>
    <span class="en"><strong>VADER compound: hate=-0.358, offensive=-0.283, normal=-0.181</strong> -- Stronger negative sentiment for hateful content. This gradient supports VADER as auxiliary feature, though all classes are negative.</span>
  </div>

  <!-- 3. VADER 분리도 분석 -->
  <div class="grid-2">
    <div class="card">
      <h3><span class="ko">3. VADER 분리도 (KDE + KS 검정)</span><span class="en">3. VADER Separability (KDE + KS Test)</span></h3>
      <div class="ko-block desc">VADER compound 점수만으로 클래스를 분리할 수 있는지 KDE 분포 겹침과 KS 검정으로 확인한다.</div>
      <div class="en-block desc">Can classes be separated by VADER compound alone? KDE overlap and KS test results.</div>
      <table class="data-table">
        <thead><tr><th><span class="ko">클래스 A</span><span class="en">Class A</span></th><th><span class="ko">클래스 B</span><span class="en">Class B</span></th><th>KS Statistic</th><th>p-value</th><th><span class="ko">유의</span><span class="en">Sig.</span></th></tr></thead>
        <tbody>{vader_ks_rows}</tbody>
      </table>
    </div>
    <div class="card">
      <img src="/static/eda/vader_separability.png" alt="VADER separability" loading="lazy" style="width:100%;border-radius:8px;cursor:pointer" onclick="openLightbox(this.src)">
    </div>
  </div>
  <div class="insight insight-red">
    <span class="ko"><strong>KS 통계량이 모두 0.10~0.16:</strong> 통계적으로 유의하지만(p&lt;0.05), 실제 분리도는 매우 낮다. KDE 분포가 대부분 겹치며, VADER 단독으로는 3클래스 분류가 불가능하다. Transformer의 문맥 이해력이 반드시 필요한 이유.</span>
    <span class="en"><strong>KS statistics 0.10-0.16:</strong> Statistically significant (p&lt;0.05) but practically low separability. KDE distributions heavily overlap; VADER alone cannot classify 3 classes. Confirms need for Transformer contextual understanding.</span>
  </div>

  <!-- 4. 타겟 커뮤니티 + 어휘 중첩 -->
  <div class="grid-2">
    <div class="card">
      <h3><span class="ko">4. 타겟 커뮤니티 분포</span><span class="en">4. Target Community Distribution</span></h3>
      <div class="ko-block desc">혐오표현이 어떤 커뮤니티를 대상으로 하는지 보여준다. {eda.get('unique_targets', 25)}개 고유 타겟 중 상위 타겟을 분석한다.</div>
      <div class="en-block desc">Shows which communities are targeted. {eda.get('unique_targets', 25)} unique targets identified.</div>
      <div class="chart-container"><canvas id="chartTargets"></canvas></div>
    </div>
    <div class="card">
      <h3><span class="ko">5. 어휘 중첩도 (Jaccard)</span><span class="en">5. Vocabulary Overlap (Jaccard)</span></h3>
      <div class="ko-block desc">클래스 쌍의 상위 500 단어 간 Jaccard 유사도. 높을수록 분류가 어렵다.</div>
      <div class="en-block desc">Jaccard similarity between top 500 words. Higher = harder to classify.</div>
      <table class="data-table">
        <thead>
          <tr>
            <th><span class="ko">클래스 A</span><span class="en">Class A</span></th>
            <th><span class="ko">클래스 B</span><span class="en">Class B</span></th>
            <th>Jaccard</th>
            <th><span class="ko">해석</span><span class="en">Interpretation</span></th>
          </tr>
        </thead>
        <tbody>{vocab_table_rows}</tbody>
      </table>
    </div>
  </div>
  <div class="insight insight-red">
    <span class="ko"><strong>hate/offensive 간 Jaccard=0.7094:</strong> 상위 500 단어 중 71%가 공유된다. 어휘 기반 분류(TF-IDF)의 근본적 한계를 설명하며, 문맥적 Transformer 모델이 필요한 이유이다.</span>
    <span class="en"><strong>hate/offensive Jaccard=0.7094:</strong> 71% shared among top 500 words. Explains fundamental TF-IDF limitations and the need for contextual Transformers.</span>
  </div>

  <!-- 6. N-gram 빈도 분석 -->
  <div class="card">
    <h3><span class="ko">6. N-gram 빈도 분석 (Bigram)</span><span class="en">6. N-gram Frequency Analysis (Bigram)</span></h3>
    <div class="ko-block desc">클래스별 빈출 bigram을 비교한다. 공통 bigram이 많을수록 표면 패턴 기반 분류의 한계를 보여준다.</div>
    <div class="en-block desc">Top bigrams per class. Shared bigrams across classes reveal surface-level pattern limitations.</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">{ngram_html}</div>
  </div>
  <div class="grid-2">
    <div class="card">
      <img src="/static/eda/ngram_analysis.png" alt="N-gram analysis" loading="lazy" style="width:100%;border-radius:8px;cursor:pointer" onclick="openLightbox(this.src)">
    </div>
    <div class="card">
      <h3><span class="ko">Bigram 겹침 분석</span><span class="en">Bigram Overlap Analysis</span></h3>
      <div class="ko-block desc">3개 클래스 모두에서 공유되는 bigram은 분류에 도움이 되지 않는 노이즈이다.</div>
      <div class="en-block desc">Bigrams shared across all 3 classes are noise that doesn't help classification.</div>
      <div style="margin:10px 0">
        <div style="font-size:13px;font-weight:600;color:var(--accent-red);margin-bottom:4px"><span class="ko">3클래스 공유 bigram ({len(eda.get('ngram_shared_all_bigrams', []))}개)</span><span class="en">Shared by all 3 classes ({len(eda.get('ngram_shared_all_bigrams', []))})</span></div>
        <div style="display:flex;flex-wrap:wrap;gap:4px">{''.join(f'<span style="background:rgba(255,255,255,0.06);padding:2px 8px;border-radius:4px;font-size:12px;font-family:monospace">{bg}</span>' for bg in eda.get('ngram_shared_all_bigrams', []))}</div>
      </div>
      <div style="margin:10px 0">
        <div style="font-size:13px;font-weight:600;color:var(--accent-blue);margin-bottom:4px"><span class="ko">hate 고유 bigram ({len(eda.get('ngram_hate_unique_bigrams', []))}개)</span><span class="en">Hate-unique bigrams ({len(eda.get('ngram_hate_unique_bigrams', []))})</span></div>
        <div style="display:flex;flex-wrap:wrap;gap:4px">{''.join(f'<span style="background:rgba(255,107,122,0.15);padding:2px 8px;border-radius:4px;font-size:12px;font-family:monospace;color:#ff6b7a">{bg}</span>' for bg in eda.get('ngram_hate_unique_bigrams', []))}</div>
      </div>
    </div>
  </div>

  <!-- 7. 워드클라우드 -->
  <div class="card">
    <h3><span class="ko">7. TF-IDF 워드클라우드</span><span class="en">7. TF-IDF Word Clouds</span></h3>
    <div class="ko-block desc">TF-IDF 가중치 기반 워드클라우드. 단순 빈도가 아닌, 해당 클래스에서 "특징적인" 단어가 강조된다.</div>
    <div class="en-block desc">TF-IDF weighted word clouds. Highlights class-distinctive words, not just frequency.</div>
    <img src="/static/eda/wordcloud.png" alt="Word clouds" loading="lazy" style="width:100%;border-radius:8px;cursor:pointer" onclick="openLightbox(this.src)">
  </div>

  <!-- 8. 텍스트 길이 통계 -->
  <div class="card">
    <h3><span class="ko">8. 텍스트 길이 통계</span><span class="en">8. Text Length Statistics</span></h3>
    <div class="ko-block desc">클래스별 단어 수 및 토큰 수 통계. max_len=128 토큰으로 거의 모든 텍스트가 잘리지 않는다 (초과율: {eda.get('exceed_max_len_total_pct', '0.01')}%).</div>
    <div class="en-block desc">Per-class word and token count statistics. With max_len=128, almost no truncation (exceed rate: {eda.get('exceed_max_len_total_pct', '0.01')}%).</div>
    <table class="data-table">
      <thead>
        <tr>
          <th><span class="ko">클래스</span><span class="en">Class</span></th>
          <th><span class="ko">샘플 수</span><span class="en">Samples</span></th>
          <th><span class="ko">단어 평균</span><span class="en">Word Mean</span></th>
          <th><span class="ko">단어 중앙값</span><span class="en">Word Median</span></th>
          <th><span class="ko">토큰 평균</span><span class="en">Token Mean</span></th>
          <th><span class="ko">토큰 최대</span><span class="en">Token Max</span></th>
        </tr>
      </thead>
      <tbody>{textlen_table_rows}</tbody>
    </table>
  </div>

  <!-- 9. Human Rationale 분포 -->
  <div class="grid-2">
    <div class="card">
      <h3><span class="ko">9. Human Rationale 분포</span><span class="en">9. Human Rationale Distribution</span></h3>
      <div class="ko-block desc">HateXplain 주석자들이 표시한 근거(rationale)의 분포. normal 클래스는 rationale이 없다 (혐오/공격이 아니므로 근거를 표시할 필요 없음).</div>
      <div class="en-block desc">Distribution of annotator-marked rationales. Normal class has no rationale (no hate/offense to justify).</div>
      <table class="data-table">
        <thead><tr>
          <th><span class="ko">클래스</span><span class="en">Class</span></th>
          <th><span class="ko">Rationale 보유</span><span class="en">Has Rationale</span></th>
          <th><span class="ko">평균 토큰 수</span><span class="en">Mean Tokens</span></th>
          <th><span class="ko">중앙값</span><span class="en">Median</span></th>
          <th><span class="ko">최대</span><span class="en">Max</span></th>
        </tr></thead>
        <tbody>{rationale_dist_rows}</tbody>
      </table>
      <div style="margin-top:8px;font-size:12px;color:var(--text-muted)">
        <span class="ko">전체: rationale 보유 <strong>{eda.get('rationale_total_with', 0):,}</strong> / 미보유 <strong>{eda.get('rationale_total_without', 0):,}</strong></span>
        <span class="en">Total: with rationale <strong>{eda.get('rationale_total_with', 0):,}</strong> / without <strong>{eda.get('rationale_total_without', 0):,}</strong></span>
      </div>
    </div>
    <div class="card">
      <img src="/static/eda/rationale_distribution.png" alt="Rationale distribution" loading="lazy" style="width:100%;border-radius:8px;cursor:pointer" onclick="openLightbox(this.src)">
    </div>
  </div>
  <div class="insight insight-blue">
    <span class="ko"><strong>hatespeech({eda.get('rationale_distribution', [{}])[0].get('samples_with_rationale', 0):,})와 offensive({eda.get('rationale_distribution', [{}, {}])[1].get('samples_with_rationale', 0):,})에만 rationale 존재.</strong> XAI의 "설명 타당성" 평가(Model Top-5 vs Human Rationale)는 이 데이터를 활용한다. 평균 rationale 길이가 짧아(3~4 토큰) 모델의 Top-5와 비교하기에 적절한 스케일이다.</span>
    <span class="en"><strong>Rationale exists only for hatespeech and offensive.</strong> XAI "explanation validity" evaluation (Model Top-5 vs Human Rationale) leverages this data. Average rationale length (3-4 tokens) matches Top-5 comparison scale well.</span>
  </div>

  <!-- 시각화 갤러리 -->
  <div class="card">
    <h3><span class="ko">EDA 시각화 갤러리</span><span class="en">EDA Visualization Gallery</span></h3>
    <div class="gallery">
      <div class="gallery-item">
        <img src="/static/eda/vader_by_class.png" alt="VADER by class" loading="lazy">
        <div class="caption"><span class="ko">VADER 감성 점수 분포</span><span class="en">VADER Sentiment Score Distribution</span></div>
      </div>
      <div class="gallery-item">
        <img src="/static/eda/target_distribution.png" alt="Target distribution" loading="lazy">
        <div class="caption"><span class="ko">타겟 커뮤니티 분포</span><span class="en">Target Community Distribution</span></div>
      </div>
      <div class="gallery-item">
        <img src="/static/eda/text_length_distribution.png" alt="Text length" loading="lazy">
        <div class="caption"><span class="ko">텍스트 길이 분포</span><span class="en">Text Length Distribution</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================
     TAB 8: XAI Analysis
     ================================================================ -->
<div id="tab-xai" class="tab-content">
  <div class="card">
    <h2><span class="ko">XAI 분석: 설명가능성 평가</span><span class="en">XAI Analysis: Explainability Evaluation</span></h2>
    <div class="ko-block desc">
      SHAP과 LIME을 활용하여 모델의 예측 근거를 분석한다.
      Overlap@5는 SHAP과 LIME이 선정한 상위 5개 중요 토큰의 일치도를 측정한다.
      높을수록 두 XAI 기법이 동의하며 설명이 일관적임을 의미한다.
      BERT-base(베이스라인)와 RoBERTa+VADER(개선 모델) 간 설명가능성을 비교한다.
    </div>
    <div class="en-block desc">
      Using SHAP and LIME to analyze model prediction rationale.
      Overlap@5 measures agreement between top-5 tokens. Higher = more consistent explanations.
      Compares BERT-base (baseline) vs RoBERTa+VADER (improved).
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>Overlap@5</h3>
      <div class="ko-block desc">각 샘플별 SHAP-LIME 상위 5개 토큰 일치도이다.</div>
      <div class="en-block desc">Per-sample SHAP-LIME top-5 token agreement.</div>
      <div class="chart-container"><canvas id="chartOverlap"></canvas></div>
    </div>
    <div class="card">
      <h3><span class="ko">XAI 요약 지표</span><span class="en">XAI Summary Metrics</span></h3>
      <div class="ko-block desc">두 모델의 XAI 지표를 비교한다.</div>
      <div class="en-block desc">Compares XAI metrics between two models.</div>
      <table class="data-table">
        <thead>
          <tr><th><span class="ko">지표</span><span class="en">Metric</span></th><th>BERT-base</th><th>RoBERTa+VADER</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Macro F1</td>
            <td>{xai_baseline_f1:.4f}</td>
            <td style="color:var(--accent-green);font-weight:700">{xai_improved_f1:.4f}</td>
          </tr>
          <tr>
            <td>Overlap@5 Mean</td>
            <td>{xai_baseline_overlap:.4f}</td>
            <td>{xai_improved_overlap:.4f}</td>
          </tr>
          <tr>
            <td><span class="ko">Overlap >= 60% 샘플 수</span><span class="en">Overlap >= 60% Samples</span></td>
            <td>{xai_baseline_ge60}</td>
            <td>{xai_improved_ge60}</td>
          </tr>
          <tr>
            <td><span class="ko">오분류 수정 수</span><span class="en">Fixed Errors</span></td>
            <td colspan="2" style="text-align:center;color:var(--accent-green);font-weight:700">{xai_fixed}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="insight insight-orange">
    <span class="ko"><strong>Baseline Overlap@5 평균 71.7%, Improved 67.5%:</strong> RoBERTa+VADER의 overlap이 약간 낮다. 모델이 더 다양한 특성에 주목하기 때문일 수 있다. 성능 향상과 설명 일관성 사이의 트레이드오프가 존재한다.</span>
    <span class="en"><strong>Baseline Overlap@5 mean 71.7%, Improved 67.5%:</strong> RoBERTa+VADER overlap is slightly lower, possibly attending to more diverse features. A performance-explanation consistency trade-off exists.</span>
  </div>

  <!-- Human Rationale 비교 (설명 타당성 평가) -->
  <div class="card">
    <h3><span class="ko">Human Rationale Alignment (설명 타당성)</span><span class="en">Human Rationale Alignment (Explanation Validity)</span></h3>
    <div class="ko-block desc">
      모델이 중요하다고 판단한 Top-5 토큰이 HateXplain 인간 주석자(annotator)의 판단 근거(rationale)와 얼마나 일치하는지를 측정한다.
      Overlap@5(안정성: 두 XAI 기법 간 일치도)와 별개로, 이 지표는 "모델의 설명이 인간의 판단과 얼마나 정합적인지"를 평가하는 <strong>설명 타당성</strong> 지표이다.
    </div>
    <div class="en-block desc">
      Measures how well the model's Top-5 important tokens align with human annotators' rationale tokens from HateXplain.
      Independent from Overlap@5 (stability), this metric evaluates <strong>explanation validity</strong> — whether the model looks at the same evidence as humans.
    </div>
    <div class="grid-2">
      <div>
        <table class="data-table">
          <thead>
            <tr><th><span class="ko">지표</span><span class="en">Metric</span></th><th>BERT-base</th><th>RoBERTa+VADER</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>SHAP Top-5 vs Human (mean)</td>
              <td>{xai_baseline_rat_shap if xai_baseline_rat_shap is not None else 'N/A'}</td>
              <td>{xai_improved_rat_shap if xai_improved_rat_shap is not None else 'N/A'}</td>
            </tr>
            <tr>
              <td>LIME Top-5 vs Human (mean)</td>
              <td>{xai_baseline_rat_lime if xai_baseline_rat_lime is not None else 'N/A'}</td>
              <td>{xai_improved_rat_lime if xai_improved_rat_lime is not None else 'N/A'}</td>
            </tr>
            <tr>
              <td><span class="ko">SHAP overlap >= 50% 샘플</span><span class="en">SHAP overlap >= 50% samples</span></td>
              <td>{xai_baseline_rat_ge50}</td>
              <td>{xai_improved_rat_ge50}</td>
            </tr>
            <tr>
              <td><span class="ko">Rationale 보유 분석 샘플</span><span class="en">Samples with Rationale</span></td>
              <td colspan="2" style="text-align:center">{xai_rat_sample_count}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div>
        <div class="chart-container"><canvas id="chartRationale"></canvas></div>
      </div>
    </div>
  </div>

  <div class="insight insight-green" id="rationale-insight" style="display:{'block' if xai_baseline_rat_shap is not None else 'none'}">
    <span class="ko"><strong>설명 타당성 평가:</strong> 모델 Top-5 토큰과 인간 rationale의 overlap을 측정하여, 모델이 욕설 등 표면 토큰에 편중되어 있는지 vs 인간과 같은 맥락적 근거를 포착하는지를 검증한다. overlap >= 0.5이면 인간 판단과 일정 수준 정렬된 설명으로 해석한다.</span>
    <span class="en"><strong>Explanation validity:</strong> Measures whether the model captures the same contextual cues as human annotators, or relies on surface-level slur tokens. Overlap >= 0.5 indicates reasonable alignment with human judgment.</span>
  </div>

  <div class="card">
    <h3><span class="ko">혼동 행렬 비교</span><span class="en">Confusion Matrix Comparison</span></h3>
    <div class="ko-block desc">BERT-base와 RoBERTa+VADER의 혼동 행렬. 대각선 값이 높을수록 정확한 분류이다.</div>
    <div class="en-block desc">Confusion matrices for both models. Higher diagonal values = more accurate.</div>
    <div class="gallery">
      <div class="gallery-item">
        <img src="/static/xai/bert_base/confusion_matrix.png" alt="BERT-base CM" loading="lazy">
        <div class="caption">BERT-base</div>
      </div>
      <div class="gallery-item">
        <img src="/static/xai/roberta_vader/confusion_matrix.png" alt="RoBERTa+VADER CM" loading="lazy">
        <div class="caption">RoBERTa+VADER</div>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================
     TAB 9: XAI Cases
     ================================================================ -->
<div id="tab-xai_cases" class="tab-content">
  <div class="card">
    <h2><span class="ko">XAI 케이스 갤러리</span><span class="en">XAI Case Gallery</span></h2>
    <div class="ko-block desc">
      개별 샘플에 대한 SHAP/LIME 분석 결과이다.
      각 케이스에서 모델이 어떤 토큰에 주목했는지, 베이스라인과 개선 모델 간 차이를 확인할 수 있다.
    </div>
    <div class="en-block desc">
      SHAP/LIME analysis results for individual samples, showing token focus differences between models.
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">케이스 요약 테이블</span><span class="en">Case Summary Table</span></h3>
    <div style="overflow-x:auto">
      <table class="data-table">
        <thead>
          <tr>
            <th>ID</th>
            <th><span class="ko">카테고리</span><span class="en">Category</span></th>
            <th><span class="ko">Baseline 토큰</span><span class="en">Baseline Tokens</span></th>
            <th><span class="ko">Improved 토큰</span><span class="en">Improved Tokens</span></th>
            <th>Base O@5</th><th>Imp O@5</th>
          </tr>
        </thead>
        <tbody>{case_table_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">케이스 시각화</span><span class="en">Case Visualizations</span></h3>
    <div class="ko-block desc">각 케이스의 SHAP/LIME 속성 시각화이다. 빨간색은 해당 클래스 기여 토큰, 파란색은 반대 방향 기여 토큰이다.</div>
    <div class="en-block desc">SHAP/LIME attribution visualizations. Red = contributing tokens, Blue = opposing tokens.</div>
    <div class="gallery">{case_gallery_html}</div>
  </div>
</div>

<!-- ================================================================
     TAB: Error Analysis
     ================================================================ -->
<div id="tab-errors" class="tab-content">
  <!-- Overview -->
  <div class="card">
    <h2><span class="ko">Error Analysis: 오분류 패턴 분석</span><span class="en">Error Analysis: Misclassification Pattern Analysis</span></h2>
    <div class="ko-block desc">모델의 오분류 패턴을 분석하여 한계점과 개선 방향을 파악한다. 과학적 검증 프레임워크의 일환으로, 모델이 실패하는 지점을 투명하게 서술한다.</div>
    <div class="en-block desc">Analyze misclassification patterns to identify limitations and improvement directions. As part of the scientific verification framework, model failure points are described transparently.</div>
  </div>

  <!-- Error Pattern Summary -->
  <div class="card">
    <h3><span class="ko">주요 오분류 패턴</span><span class="en">Major Error Patterns</span></h3>
    <div class="error-pattern-card">
      <div class="error-pattern-arrow"><span class="from-label">hate</span> &#x2192; <span class="to-label">offensive</span></div>
      <div class="error-pattern-desc">
        <span class="ko">어휘 Jaccard 0.71 -- 동일 비속어를 공유하는 두 클래스. 맥락적 차이(직접 공격 vs 일반적 표현)를 잡기 어려움.</span>
        <span class="en">Vocabulary Jaccard 0.71 -- two classes share the same profanity. Contextual difference (direct attack vs general expression) is hard to capture.</span>
      </div>
    </div>
    <div class="error-pattern-card">
      <div class="error-pattern-arrow"><span class="from-label">offensive</span> &#x2192; <span class="to-label">normal</span></div>
      <div class="error-pattern-desc">
        <span class="ko">비꼬기, 슬랭, 은유적 표현이 offensive로 보이지만 실제 normal.</span>
        <span class="en">Sarcasm, slang, and metaphorical expressions appear offensive but are actually normal.</span>
      </div>
    </div>
    <div class="error-pattern-card">
      <div class="error-pattern-arrow"><span class="from-label">normal</span> &#x2192; <span class="to-label">hate</span></div>
      <div class="error-pattern-desc">
        <span class="ko">인용, 보도, 교육 목적의 혐오 단어 포함 텍스트를 hate로 과잉 분류.</span>
        <span class="en">Texts containing hate words for quotation, reporting, or educational purposes are over-classified as hate.</span>
      </div>
    </div>
    <div class="error-pattern-card">
      <div class="error-pattern-arrow"><span class="from-label">hate</span> &#x2192; <span class="to-label">normal</span></div>
      <div class="error-pattern-desc">
        <span class="ko">간접적, 완곡한 혐오 표현을 모델이 감지하지 못함.</span>
        <span class="en">Model fails to detect indirect, euphemistic hate speech.</span>
      </div>
    </div>
  </div>

  <!-- VADER Blind Spots -->
  <div class="card">
    <h3><span class="ko">VADER Blind Spots</span><span class="en">VADER Blind Spots</span></h3>
    <div class="error-pattern-card">
      <div class="error-pattern-arrow"><strong><span class="ko">반어 (Sarcasm)</span><span class="en">Sarcasm</span></strong></div>
      <div class="error-pattern-desc">
        <span class="ko">VADER compound > 0인데 실제 hate인 경우. 긍정적 표현으로 포장된 혐오를 탐지하지 못한다.</span>
        <span class="en">VADER compound > 0 but actually hate. Cannot detect hate wrapped in positive expressions.</span>
      </div>
    </div>
    <div class="error-pattern-card">
      <div class="error-pattern-arrow"><strong><span class="ko">중립 혐오 (Neutral Hate)</span><span class="en">Neutral Hate</span></strong></div>
      <div class="error-pattern-desc">
        <span class="ko">VADER compound &#x2248; 0인데 실제 hate인 경우 ("those people don't belong here"). 감성적으로 중립이지만 의미적으로 혐오.</span>
        <span class="en">VADER compound &#x2248; 0 but actually hate ("those people don't belong here"). Sentimentally neutral but semantically hateful.</span>
      </div>
    </div>
    <div style="margin-top:16px; padding:14px 18px; background:rgba(124,138,255,0.06); border-radius:8px; border-left:3px solid var(--accent-blue)">
      <span class="ko">VADER는 lexicon 기반이므로 반어/완곡어법을 해석하지 못한다. 이것이 VADER 단독 효과가 제한적(p=0.138)인 이유이다.</span>
      <span class="en">VADER is lexicon-based and cannot interpret sarcasm/euphemism. This is why VADER's standalone effect is limited (p=0.138).</span>
    </div>
  </div>

  <!-- Ablation Insight -->
  <div class="card">
    <h3><span class="ko">Ablation Insight: VADER vs MLP 효과 분리</span><span class="en">Ablation Insight: Separating VADER vs MLP Effects</span></h3>
    <div class="ablation-diagram">
      <div class="ablation-col">
        <h4>BERT-base</h4>
        <div class="ablation-f1">0.6744</div>
        <div class="ablation-arch">768 &#x2192; 3</div>
      </div>
      <div class="ablation-col">
        <h4>BERT+MLP</h4>
        <div class="ablation-f1">0.6810</div>
        <div class="ablation-arch">768 &#x2192; 256 &#x2192; 3</div>
      </div>
      <div class="ablation-col">
        <h4>BERT+VADER</h4>
        <div class="ablation-f1">0.6794</div>
        <div class="ablation-arch">772 &#x2192; 256 &#x2192; 3</div>
      </div>
    </div>
    <div class="ablation-pval">
      <span>BERT+MLP vs BERT+VADER: <strong>p = 0.567</strong> (<span class="ko">유의하지 않음</span><span class="en">not significant</span>)</span>
    </div>
    <div style="margin-top:12px; padding:14px 18px; background:rgba(124,138,255,0.06); border-radius:8px; border-left:3px solid var(--accent-orange)">
      <span class="ko">VADER 4차원이 768차원 대비 0.5%에 불과. 정보량 비율로 보면 VADER 기여가 작은 것은 당연. 그러나 RoBERTa처럼 인코더 품질이 높아지면 미세한 감성 보완이 유의미해진다.</span>
      <span class="en">VADER's 4 dimensions are only 0.5% of 768 dimensions. In terms of information ratio, small VADER contribution is expected. However, with higher encoder quality like RoBERTa, subtle sentiment supplementation becomes meaningful.</span>
    </div>
  </div>

  <!-- Limitations & Future Work -->
  <div class="card">
    <h3><span class="ko">한계점 및 향후 과제</span><span class="en">Limitations &amp; Future Work</span></h3>
    <div style="display:grid; gap:10px">
      <div class="error-pattern-card">
        <span class="ko"><strong>15 seed 설계:</strong> 정식 v2는 n=15 paired design으로 검정력과 효과크기 해석을 보강한다. 단, smoke run은 통계 결론으로 해석하지 않는다.</span>
        <span class="en"><strong>15-seed design:</strong> The formal v2 run uses n=15 paired design to strengthen power and effect-size interpretation. Smoke runs are not statistical conclusions.</span>
      </div>
      <div class="error-pattern-card">
        <span class="ko"><strong>영어 전용 데이터셋:</strong> HateXplain은 영어 전용. 한국어/다국어 혐오표현 탐지로의 확장 필요.</span>
        <span class="en"><strong>English-only dataset:</strong> HateXplain is English-only. Extension to Korean/multilingual hate speech detection needed.</span>
      </div>
      <div class="error-pattern-card">
        <span class="ko"><strong>VADER lexicon 한계:</strong> 일반 목적 감성 사전. 혐오표현 도메인 특화 감성 사전 구축 가능성.</span>
        <span class="en"><strong>VADER lexicon limitations:</strong> General-purpose sentiment lexicon. Domain-specific sentiment dictionary for hate speech is possible.</span>
      </div>
      <div class="error-pattern-card">
        <span class="ko"><strong>offensive 클래스 F1 0.53-0.55:</strong> 구조적 한계. hate/offensive 경계의 모호함은 데이터셋 자체의 문제.</span>
        <span class="en"><strong>Offensive class F1 0.53-0.55:</strong> Structural limitation. The ambiguous hate/offensive boundary is a dataset-level issue.</span>
      </div>
      <div class="error-pattern-card">
        <span class="ko"><strong>Annotator 간 불일치:</strong> majority vote는 소수 의견을 무시. 3명 중 2명만 동의한 샘플의 정보 손실.</span>
        <span class="en"><strong>Inter-annotator disagreement:</strong> Majority vote ignores minority opinions. Information loss from samples where only 2 of 3 annotators agree.</span>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================
     TAB 10: Model Architecture
     ================================================================ -->
<div id="tab-architecture" class="tab-content">
  <div class="card">
    <h2><span class="ko">모델 아키텍처</span><span class="en">Model Architecture</span></h2>
    <div class="ko-block desc">
      3가지 Transformer 기반 아키텍처의 구조를 시각적으로 설명한다.
      각 아키텍처의 입력 처리, 인코더, 분류 헤드 구성이 다르며, 이 차이가 성능에 영향을 미친다.
    </div>
    <div class="en-block desc">
      Visual explanation of 3 Transformer-based architectures. Each differs in design, affecting performance.
    </div>
  </div>

  <div class="grid-3">
    <div class="card">
      <h3>TransformerCLS</h3>
      <div class="ko-block desc">BERT-base 모델. [CLS] 토큰 임베딩을 직접 분류에 사용하는 가장 기본적인 구조.</div>
      <div class="en-block desc">BERT-base model. Most basic architecture using [CLS] directly for classification.</div>
      <div class="arch-diagram">
        <div class="arch-block">Input Text</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">BERT Tokenizer<br><span style="font-size:0.8rem;color:var(--text-secondary)">max_len=128</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block highlight">BERT Encoder<br><span style="font-size:0.8rem;color:var(--text-secondary)">12 layers, 768 dim</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">[CLS] Embedding<br><span style="font-size:0.8rem;color:var(--text-secondary)">768 dim</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">Dropout (0.1)</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block output">Linear (768 &rarr; 3)<br><span style="font-size:0.8rem;color:var(--text-secondary)">hate / offensive / normal</span></div>
      </div>
    </div>

    <div class="card">
      <h3>TransformerMLP</h3>
      <div class="ko-block desc">BERT+MLP. Ablation 통제 모델로, VADER 없이 동일 MLP 구조를 사용하여 VADER 효과를 분리한다.</div>
      <div class="en-block desc">BERT+MLP. Ablation control using identical MLP without VADER to isolate its effect.</div>
      <div class="arch-diagram">
        <div class="arch-block">Input Text</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">BERT Tokenizer<br><span style="font-size:0.8rem;color:var(--text-secondary)">max_len=128</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block highlight">BERT Encoder<br><span style="font-size:0.8rem;color:var(--text-secondary)">12 layers, 768 dim</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">[CLS] Embedding<br><span style="font-size:0.8rem;color:var(--text-secondary)">768 dim</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">MLP Hidden<br><span style="font-size:0.8rem;color:var(--text-secondary)">768 &rarr; 256, ReLU</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">Dropout (0.1)</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block output">Linear (256 &rarr; 3)<br><span style="font-size:0.8rem;color:var(--text-secondary)">hate / offensive / normal</span></div>
      </div>
    </div>

    <div class="card">
      <h3>HybridSentiment</h3>
      <div class="ko-block desc">BERT+VADER / RoBERTa+VADER. [CLS]와 VADER 4차원을 연결(concat)하여 MLP에 입력. 핵심 가설 아키텍처.</div>
      <div class="en-block desc">BERT+VADER / RoBERTa+VADER. Concatenates [CLS] with 4-dim VADER for MLP. Core hypothesis architecture.</div>
      <div class="arch-diagram">
        <div class="arch-side">
          <div class="arch-col">
            <div class="arch-block">Input Text</div>
            <div class="arch-arrow">&darr;</div>
            <div class="arch-block highlight">Transformer Encoder<br><span style="font-size:0.8rem;color:var(--text-secondary)">BERT or RoBERTa</span></div>
            <div class="arch-arrow">&darr;</div>
            <div class="arch-block">[CLS]<br><span style="font-size:0.8rem;color:var(--text-secondary)">768 dim</span></div>
          </div>
          <div class="arch-col">
            <div class="arch-block vader">VADER<br><span style="font-size:0.8rem;color:var(--text-secondary)">Sentiment Analyzer</span></div>
            <div class="arch-arrow">&darr;</div>
            <div class="arch-block vader">4-dim<br><span style="font-size:0.8rem;color:var(--text-secondary)">comp/pos/neg/neu</span></div>
          </div>
        </div>
        <div class="arch-arrow">&darr; Concatenate</div>
        <div class="arch-block">Combined<br><span style="font-size:0.8rem;color:var(--text-secondary)">772 dim (768+4)</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">MLP Hidden<br><span style="font-size:0.8rem;color:var(--text-secondary)">772 &rarr; 256, ReLU</span></div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block">Dropout (0.1)</div>
        <div class="arch-arrow">&darr;</div>
        <div class="arch-block output">Linear (256 &rarr; 3)<br><span style="font-size:0.8rem;color:var(--text-secondary)">hate / offensive / normal</span></div>
      </div>
    </div>
  </div>

  <div class="insight insight-blue">
    <span class="ko"><strong>HybridSentiment의 핵심 설계:</strong> 768차원 [CLS] 벡터에 단 4차원 VADER 감성을 추가한다 (비율 0.5%).
    RoBERTa와 결합 시 통계적으로 유의미한 성능 향상을 달성. "올바른 인코더 + 적절한 보조 정보"의 시너지를 보여준다.</span>
    <span class="en"><strong>Core HybridSentiment design:</strong> Just 4 VADER dims added to 768-dim [CLS] (0.5% ratio).
    Achieves significant improvement with RoBERTa, demonstrating "right encoder + right auxiliary" synergy.</span>
  </div>
  <div class="insight insight-orange">
    <span class="ko"><strong>TransformerMLP은 ablation 통제:</strong> BERT+MLP(F1=0.6810) vs BERT+VADER(F1=0.6794), p=0.567로 유의미하지 않다.
    같은 인코더 기반에서 VADER의 추가 효과가 미미하며, 인코더 사전학습 품질이 더 중요한 변수이다.</span>
    <span class="en"><strong>TransformerMLP as ablation control:</strong> BERT+MLP (F1=0.6810) vs BERT+VADER (F1=0.6794), p=0.567.
    Minimal VADER effect on same encoder, confirming pre-training quality as the more important variable.</span>
  </div>
</div>

<!-- ================================================================
     TAB 12: Data Explorer
     ================================================================ -->
<div id="tab-explorer" class="tab-content">
  <div class="card">
    <h2><span class="ko">Data Explorer -- 데이터셋 탐색</span><span class="en">Data Explorer -- Dataset Browser</span></h2>
    <div class="ko-block desc">
      HateXplain 데이터셋의 개별 샘플을 검색하고 탐색할 수 있다.
      텍스트 키워드 검색과 라벨 필터링을 지원한다.
    </div>
    <div class="en-block desc">
      Browse and search individual samples from the HateXplain dataset.
      Supports text keyword search and label filtering.
    </div>

    <div class="explorer-search-bar">
      <input type="text" id="explorer-query" placeholder="Search text..." onkeydown="if(event.key==='Enter') explorerSearch(1)">
      <select id="explorer-label">
        <option value="all"><span class="ko">All</span></option>
        <option value="hatespeech">hatespeech</option>
        <option value="offensive">offensive</option>
        <option value="normal">normal</option>
      </select>
      <button onclick="explorerSearch(1)"><span class="ko">검색</span><span class="en">Search</span></button>
    </div>

    <div id="explorer-status" style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:10px"></div>
    <div id="explorer-results" class="explorer-grid"></div>
    <div id="explorer-pagination" class="explorer-pagination"></div>
  </div>
</div>

<!-- ================================================================
     TAB 13: Model Comparison
     ================================================================ -->
<div id="tab-comparison" class="tab-content">
  <div class="card">
    <h2><span class="ko">Model Comparison -- 모델 비교 분석</span><span class="en">Model Comparison -- Side-by-Side Analysis</span></h2>
    <div class="ko-block desc">
      두 모델을 선택하여 성능 지표를 나란히 비교할 수 있다.
      레이더 차트와 상세 지표 카드를 통해 강점과 약점을 파악한다.
    </div>
    <div class="en-block desc">
      Select two models for side-by-side performance comparison.
      Radar chart and metric cards reveal strengths and weaknesses.
    </div>

    <div class="comparison-selectors">
      <label><span class="ko">Model A:</span><span class="en">Model A:</span></label>
      <select id="cmp-model-a" onchange="updateComparison()"></select>
      <label><span class="ko">Model B:</span><span class="en">Model B:</span></label>
      <select id="cmp-model-b" onchange="updateComparison()"></select>
    </div>
  </div>

  <div class="comparison-side">
    <div class="card">
      <h3 id="cmp-title-a" style="color:var(--accent-blue)">Model A</h3>
      <div id="cmp-metrics-a"></div>
    </div>
    <div class="card">
      <h3 id="cmp-title-b" style="color:var(--accent-green)">Model B</h3>
      <div id="cmp-metrics-b"></div>
    </div>
  </div>

  <div class="card">
    <h3><span class="ko">레이더 차트 비교</span><span class="en">Radar Chart Comparison</span></h3>
    <div class="chart-container"><canvas id="chartComparisonRadar"></canvas></div>
  </div>

  <div class="card">
    <h3><span class="ko">Delta 분석</span><span class="en">Delta Analysis</span></h3>
    <div class="ko-block desc">Model A 대비 Model B의 성능 차이를 보여준다. 양수는 B가 우세, 음수는 A가 우세이다.</div>
    <div class="en-block desc">Performance delta of Model B relative to Model A. Positive = B is better, Negative = A is better.</div>
    <div id="cmp-delta"></div>
  </div>
</div>

<!-- ================================================================
     TAB 14: Report
     ================================================================ -->
<div id="tab-report" class="tab-content">
  <div class="card">
    <h2><span class="ko">Report -- 자동 생성 보고서</span><span class="en">Report -- Auto-Generated Summary</span></h2>
    <div class="ko-block desc">실험 결과를 요약한 보고서가 자동으로 생성된다. 복사하거나 인쇄하여 활용할 수 있다.</div>
    <div class="en-block desc">Auto-generated experiment summary report. Copy or print for further use.</div>

    <div class="report-actions">
      <button onclick="copyReport()" style="background:var(--accent-blue);color:#fff">
        <span class="ko">클립보드에 복사</span><span class="en">Copy to Clipboard</span>
      </button>
      <button onclick="window.print()" style="background:var(--accent-green);color:#fff">
        <span class="ko">인쇄</span><span class="en">Print</span>
      </button>
    </div>
  </div>

  <div class="card">
    <div class="report-container" id="report-content"></div>
  </div>
</div>

<!-- ================================================================
     TAB: References
     ================================================================ -->
<div id="tab-references" class="tab-content">
  <!-- Key References -->
  <div class="card">
    <h2><span class="ko">주요 참고문헌</span><span class="en">Key References</span></h2>

    <div class="ref-item">
      <div class="ref-citation">Mathew, B., Saha, P., Yimam, S. M., et al. (2021). "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection." <em>AAAI 2021</em>.</div>
      <div class="ref-role"><span class="ko">데이터셋 및 XAI 평가 프레임워크 제공</span><span class="en">Provides the dataset and XAI evaluation framework</span></div>
    </div>

    <div class="ref-item">
      <div class="ref-citation">Cheng, L. (2022). "Towards Explainable and Adaptive Sentiment-enhanced Hate Speech Detection." <em>Virginia Tech, PhD Dissertation</em>.</div>
      <div class="ref-role"><span class="ko">감성 분석 기반 혐오표현 탐지의 선구적 연구</span><span class="en">Pioneering research on sentiment-based hate speech detection</span></div>
      <div class="ref-diff"><span class="ko">차별점: Cheng은 감성 feature fusion의 성능 효과를 보고한다. 본 연구는 선행연구 기반 사전 가설 &#x2192; 통제 ablation &#x2192; XAI 사후 검증 구조로 확장한다.</span><span class="en">Differentiation: Cheng reports the performance effect of sentiment feature fusion. Our study extends it with a prior hypothesis &#x2192; controlled ablation &#x2192; post-hoc XAI verification structure.</span></div>
    </div>

    <div class="ref-item">
      <div class="ref-citation">Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." <em>NAACL 2019</em>.</div>
      <div class="ref-role"><span class="ko">양방향 문맥 표현 학습</span><span class="en">Bidirectional contextual representation learning</span></div>
    </div>

    <div class="ref-item">
      <div class="ref-citation">Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." <em>arXiv:1907.11692</em>.</div>
      <div class="ref-role"><span class="ko">강화된 사전학습 전략</span><span class="en">Enhanced pretraining strategy</span></div>
    </div>

    <div class="ref-item">
      <div class="ref-citation">Hutto, C. J. &amp; Gilbert, E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." <em>ICWSM 2014</em>.</div>
      <div class="ref-role"><span class="ko">규칙 기반 감성 분석</span><span class="en">Rule-based sentiment analysis</span></div>
    </div>

    <div class="ref-item">
      <div class="ref-citation">Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." <em>KDD 2016</em>.</div>
      <div class="ref-role"><span class="ko">LIME 설명가능성 기법</span><span class="en">LIME explainability technique</span></div>
    </div>

    <div class="ref-item">
      <div class="ref-citation">Lundberg, S. &amp; Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." <em>NeurIPS 2017</em>.</div>
      <div class="ref-role"><span class="ko">SHAP 설명가능성 기법</span><span class="en">SHAP explainability technique</span></div>
    </div>
  </div>

  <!-- Reproducibility -->
  <div class="card">
    <h3><span class="ko">재현성 (Reproducibility)</span><span class="en">Reproducibility</span></h3>
    <table class="repro-table">
      <tr><th><span class="ko">항목</span><span class="en">Item</span></th><th><span class="ko">값</span><span class="en">Value</span></th></tr>
      <tr><td>Environment</td><td>Python 3.13, PyTorch (MPS), Apple M3 Max 64GB</td></tr>
      <tr><td>Seeds</td><td>42, 52, 62 (3-fold)</td></tr>
      <tr><td>Max sequence length</td><td>128 tokens</td></tr>
      <tr><td>Training epochs</td><td>5 (early stopping patience=2)</td></tr>
    </table>
    <div style="margin-top:14px; padding:14px 18px; background:rgba(124,138,255,0.06); border-radius:8px; border-left:3px solid var(--accent-green)">
      <span class="ko"><strong>재현 방법:</strong> <code>./run.sh full</code> 또는 단계별 <code>./run.sh data &amp;&amp; ./run.sh tune &amp;&amp; ./run.sh benchmark ...</code></span>
      <span class="en"><strong>How to reproduce:</strong> <code>./run.sh full</code> or step-by-step <code>./run.sh data &amp;&amp; ./run.sh tune &amp;&amp; ./run.sh benchmark ...</code></span>
    </div>
  </div>

  <!-- Estimated Training Time -->
  <div class="card">
    <h3><span class="ko">예상 학습 시간</span><span class="en">Estimated Training Time</span></h3>
    <table class="repro-table">
      <tr><th><span class="ko">모델/단계</span><span class="en">Model/Stage</span></th><th><span class="ko">소요 시간</span><span class="en">Duration</span></th></tr>
      <tr><td>TF-IDF+LR</td><td>~5<span class="ko">초</span><span class="en">s</span></td></tr>
      <tr><td>TF-IDF+SVM</td><td>~10<span class="ko">초</span><span class="en">s</span></td></tr>
      <tr><td>BERT-base</td><td>~8<span class="ko">분/시드</span><span class="en">min/seed</span> (MPS)</td></tr>
      <tr><td>BERT+MLP</td><td>~8<span class="ko">분/시드</span><span class="en">min/seed</span></td></tr>
      <tr><td>BERT+VADER</td><td>~9<span class="ko">분/시드</span><span class="en">min/seed</span></td></tr>
      <tr><td>RoBERTa+VADER</td><td>~10<span class="ko">분/시드</span><span class="en">min/seed</span></td></tr>
      <tr><td>Tuning (grid search)</td><td>~2<span class="ko">시간</span><span class="en">h</span></td></tr>
      <tr><td>XAI (SHAP+LIME)</td><td>~30<span class="ko">분</span><span class="en">min</span></td></tr>
      <tr style="font-weight:700; border-top:2px solid var(--border-subtle)"><td>Total</td><td>~4-5<span class="ko">시간</span><span class="en">h</span></td></tr>
    </table>
  </div>
</div>

<!-- ================================================================
     TAB 11: Playground
     ================================================================ -->
<div id="tab-playground" class="tab-content">
  <div class="card">
    <h2><span class="ko">Playground -- 모델 테스트</span><span class="en">Playground -- Model Testing</span></h2>
    <div class="ko-block desc">
      텍스트를 입력하면 학습된 모델들이 실시간으로 분류 결과를 보여줍니다.
      각 모델의 예측 확률(hatespeech / offensive / normal)과 VADER 감성 점수를 함께 확인할 수 있습니다.
      모델은 최초 요청 시 로드되므로 첫 번째 예측은 몇 초 걸릴 수 있습니다.
    </div>
    <div class="en-block desc">
      Enter text and trained models will classify it in real-time.
      See prediction probabilities (hatespeech / offensive / normal) and VADER sentiment scores.
      Models are lazy-loaded on first request, so the first prediction may take a few seconds.
    </div>

    <!-- Device Info -->
    <div id="pg-device" style="margin-bottom:16px;font-size:13px;color:var(--text-muted)">
      <span class="ko">디바이스 감지 중...</span><span class="en">Detecting device...</span>
    </div>

    <!-- Input Area -->
    <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">
      <textarea id="pg-input" rows="3" placeholder="Enter text to classify... (e.g., 'I hate all of them')"
        style="flex:1;min-width:300px;background:var(--bg-dark);color:var(--text-primary);border:1px solid var(--border-color);border-radius:8px;padding:12px;font-size:14px;font-family:inherit;resize:vertical"></textarea>
    </div>

    <!-- Model Selection -->
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;align-items:center">
      <label style="font-size:13px;color:var(--text-muted);margin-right:4px">
        <span class="ko">모델 선택:</span><span class="en">Select models:</span>
      </label>
      <label class="pg-check"><input type="checkbox" value="A_B" checked> A_B</label>
      <label class="pg-check"><input type="checkbox" value="D_B" checked> D_B</label>
      <label class="pg-check"><input type="checkbox" value="D_R" checked> D_R</label>
      <label class="pg-check"><input type="checkbox" value="D_B+Target"> D_B+Target</label>
      <button id="pg-btn" onclick="pgPredict()"
        style="margin-left:auto;padding:10px 28px;background:var(--accent-blue);color:white;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:14px">
        <span class="ko">분석하기</span><span class="en">Analyze</span>
      </button>
      <button id="pg-lime-btn" onclick="pgPredict(true)"
        style="padding:10px 28px;background:var(--accent-green, #22c55e);color:white;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:14px"
        title="Attention + LIME 분석 (모델당 5-10초 추가 소요)">
        <span class="ko">XAI 분석</span><span class="en">XAI Analysis</span>
      </button>
    </div>

    <!-- Loading -->
    <div id="pg-loading" style="display:none;text-align:center;padding:24px;color:var(--text-muted)">
      <div style="font-size:24px;animation:spin 1s linear infinite;display:inline-block">&#x21bb;</div>
      <div style="margin-top:8px"><span class="ko">모델 추론 중... (첫 요청은 모델 로딩으로 10-15초 소요)</span><span class="en">Running inference... (first request takes 10-15s to load models)</span></div>
    </div>

    <!-- Results -->
    <div id="pg-results"></div>

    <!-- VADER Detail -->
    <div id="pg-vader" style="display:none" class="card" style="margin-top:16px">
      <h3>VADER Sentiment Scores</h3>
      <div id="pg-vader-content"></div>
    </div>
  </div>

  <!-- Example Texts -->
  <div class="card">
    <h2><span class="ko">예시 텍스트</span><span class="en">Example Texts</span></h2>
    <div class="ko-block desc">클릭하면 자동으로 입력됩니다.</div>
    <div class="en-block desc">Click to auto-fill the input.</div>
    <div style="display:flex;flex-direction:column;gap:8px">
      <button class="pg-example" onclick="pgFill(this.dataset.text)" data-text="I love spending time with my friends and family"
        style="text-align:left;padding:10px 14px;background:var(--bg-dark);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);cursor:pointer;font-size:13px">
        <span style="color:var(--accent-green);font-weight:600">[Normal]</span> I love spending time with my friends and family
      </button>
      <button class="pg-example" onclick="pgFill(this.dataset.text)" data-text="shut up you stupid idiot nobody cares about your opinion"
        style="text-align:left;padding:10px 14px;background:var(--bg-dark);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);cursor:pointer;font-size:13px">
        <span style="color:var(--accent-orange);font-weight:600">[Offensive]</span> shut up you stupid idiot nobody cares about your opinion
      </button>
      <button class="pg-example" onclick="pgFill(this.dataset.text)" data-text="all jews must be eliminated from this country they are destroying everything"
        style="text-align:left;padding:10px 14px;background:var(--bg-dark);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);cursor:pointer;font-size:13px">
        <span style="color:var(--accent-red);font-weight:600">[Hatespeech]</span> all jews must be eliminated from this country they are destroying everything
      </button>
      <button class="pg-example" onclick="pgFill(this.dataset.text)" data-text="these immigrants are ruining our country go back where you came from"
        style="text-align:left;padding:10px 14px;background:var(--bg-dark);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);cursor:pointer;font-size:13px">
        <span style="color:var(--accent-red);font-weight:600">[Hatespeech]</span> these immigrants are ruining our country go back where you came from
      </button>
      <button class="pg-example" onclick="pgFill(this.dataset.text)" data-text="lmao this dude is so dumb he can barely spell his own name"
        style="text-align:left;padding:10px 14px;background:var(--bg-dark);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);cursor:pointer;font-size:13px">
        <span style="color:var(--accent-orange);font-weight:600">[Offensive]</span> lmao this dude is so dumb he can barely spell his own name
      </button>
    </div>
  </div>

  <!-- Interpretation Guide -->
  <div class="card">
    <h2><span class="ko">해석 가이드</span><span class="en">Interpretation Guide</span></h2>
    <div class="insight-box insight-blue">
      <span class="ko">
        <strong>확률 분포 읽는 법:</strong> 각 모델은 3개 클래스에 대한 확률을 출력합니다 (합계 = 100%).
        가장 높은 확률의 클래스가 예측 결과입니다. 확률이 고르게 분포되면(예: 35/33/32) 모델이 확신하지 못하는 것이고,
        한쪽으로 쏠리면(예: 85/10/5) 확신이 높은 것입니다.
      </span>
      <span class="en">
        <strong>Reading probabilities:</strong> Each model outputs probabilities for 3 classes (sum = 100%).
        Highest probability is the prediction. Even distribution (e.g., 35/33/32) means low confidence,
        skewed distribution (e.g., 85/10/5) means high confidence.
      </span>
    </div>
    <div class="insight-box insight-orange">
      <span class="ko">
        <strong>모델 간 불일치:</strong> 모델들이 다른 결과를 내면, 그 텍스트가 hate/offensive 경계에 있다는 뜻입니다.
        이런 애매한 사례가 바로 XAI 분석이 필요한 지점입니다.
      </span>
      <span class="en">
        <strong>Model disagreement:</strong> When models disagree, the text sits on the hate/offensive boundary.
        These ambiguous cases are exactly where XAI analysis is most valuable.
      </span>
    </div>
    <div class="insight-box insight-green">
      <span class="ko">
        <strong>VADER 점수:</strong> compound &lt; -0.5 이면 매우 부정적, &gt; 0.5 이면 매우 긍정적입니다.
        Hybrid 모델(BERT+VADER, RoBERTa+VADER)은 이 점수를 추가 입력으로 사용합니다.
      </span>
      <span class="en">
        <strong>VADER scores:</strong> compound &lt; -0.5 is very negative, &gt; 0.5 is very positive.
        Hybrid models (BERT+VADER, RoBERTa+VADER) use these as additional input features.
      </span>
    </div>
  </div>
</div>

</div><!-- /.container -->

<!-- ================================================================
     JavaScript
     ================================================================ -->
<script>
// ---- 데이터 주입 ----
const benchmarkData = {js(bm_slim)};
const significanceData = {js(sig_slim)};
const tuningLog = {js(tuning_slim)};
const edaData = {js(eda)};
const overlapData = {js(overlap_slim)};
const rationaleData = {js(rationale_slim)};
let learningCurveData = null;

// ---- 테마 / 언어 토글 ----
function toggleTheme() {{
  document.body.classList.toggle('light-mode');
  document.getElementById('themeBtn').textContent =
    document.body.classList.contains('light-mode') ? 'Dark' : 'Light';
}}

function toggleLang() {{
  document.body.classList.toggle('en-mode');
  document.getElementById('langBtn').textContent =
    document.body.classList.contains('en-mode') ? 'KO' : 'EN';
}}

// ---- 탭 전환 ----
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', function() {{
    const tabId = this.getAttribute('data-tab');
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');
    this.classList.add('active');
    if (tabId === 'learning' && !learningCurveData) {{ fetchLearningCurves(); }}
    if (tabId === 'report') {{ generateReport(); }}
    if (tabId === 'comparison' && cmpChart === null) {{ initComparison(); }}
  }});
}});

// ---- Chart.js 기본 설정 ----
Chart.defaults.color = '#9ca3c4';
Chart.defaults.borderColor = 'rgba(124,138,255,0.15)';
Chart.defaults.font.family = "'Inter', sans-serif";

const COLORS = ['#7c8aff', '#00e5b0', '#ff6b7a', '#ffb347', '#b57aff', '#4ecdc4'];

// ---- 벤치마크 수평 바 차트 ----
(function() {{
  const models = benchmarkData.map(r => r.model);
  const f1s = benchmarkData.map(r => parseFloat(r.macro_f1_mean));
  const stds = benchmarkData.map(r => parseFloat(r.macro_f1_std));
  const bgColors = models.map((_, i) => COLORS[i % COLORS.length]);

  new Chart(document.getElementById('chartBenchmarkF1'), {{
    type: 'bar',
    data: {{
      labels: models,
      datasets: [{{ label: 'Macro F1', data: f1s,
        backgroundColor: bgColors.map(c => c + '99'),
        borderColor: bgColors, borderWidth: 2 }}]
    }},
    options: {{
      indexAxis: 'y', responsive: true,
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{
          label: (ctx) => 'F1: ' + f1s[ctx.dataIndex].toFixed(4) + ' +/- ' + stds[ctx.dataIndex].toFixed(4)
        }} }}
      }},
      scales: {{ x: {{ min: 0.60, max: 0.72, title: {{ display: true, text: 'Macro F1' }} }} }}
    }},
    plugins: [{{
      id: 'errorBars',
      afterDatasetsDraw(chart) {{
        const meta = chart.getDatasetMeta(0);
        const ctx2 = chart.ctx;
        meta.data.forEach((bar, i) => {{
          const xMin = chart.scales.x.getPixelForValue(f1s[i] - stds[i]);
          const xMax = chart.scales.x.getPixelForValue(f1s[i] + stds[i]);
          const y = bar.y;
          ctx2.save(); ctx2.strokeStyle = bgColors[i]; ctx2.lineWidth = 2;
          ctx2.beginPath();
          ctx2.moveTo(xMin, y); ctx2.lineTo(xMax, y);
          ctx2.moveTo(xMin, y-6); ctx2.lineTo(xMin, y+6);
          ctx2.moveTo(xMax, y-6); ctx2.lineTo(xMax, y+6);
          ctx2.stroke(); ctx2.restore();
        }});
      }}
    }}]
  }});
}})();

// ---- 레이더 차트 ----
(function() {{
  const top4 = benchmarkData.slice(0, 4);
  const labels = ['Macro F1', 'Accuracy', 'AUROC', 'Hate F1', 'Offensive F1', 'Normal F1'];
  const datasets = top4.map((r, i) => ({{
    label: r.model,
    data: [parseFloat(r.macro_f1_mean), parseFloat(r.accuracy_mean), parseFloat(r.auroc_mean),
           parseFloat(r['per_class_f1.hatespeech_mean']), parseFloat(r['per_class_f1.offensive_mean']),
           parseFloat(r['per_class_f1.normal_mean'])],
    borderColor: COLORS[i], backgroundColor: COLORS[i] + '22',
    borderWidth: 2, pointRadius: 4, pointBackgroundColor: COLORS[i]
  }}));
  new Chart(document.getElementById('chartRadar'), {{
    type: 'radar', data: {{ labels, datasets }},
    options: {{ responsive: true,
      scales: {{ r: {{ min: 0.45, max: 0.90, ticks: {{ stepSize: 0.05 }}, grid: {{ color: 'rgba(124,138,255,0.1)' }} }} }},
      plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 16 }} }} }}
    }}
  }});
}})();

// ---- 클래스별 F1 ----
(function() {{
  const models = benchmarkData.map(r => r.model);
  const classes = ['hatespeech', 'offensive', 'normal'];
  const cColors = ['#ff6b7a', '#ffb347', '#00e5b0'];
  const datasets = classes.map((cls, ci) => ({{
    label: cls,
    data: benchmarkData.map(r => parseFloat(r['per_class_f1.' + cls + '_mean'])),
    backgroundColor: cColors[ci] + '99', borderColor: cColors[ci], borderWidth: 2
  }}));
  new Chart(document.getElementById('chartPerClassF1'), {{
    type: 'bar', data: {{ labels: models, datasets }},
    options: {{ responsive: true, plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ min: 0.4, max: 0.85, title: {{ display: true, text: 'F1 Score' }} }} }}
    }}
  }});
}})();

// ---- P-value 히트맵 (Canvas 직접 그리기) ----
(function() {{
  const allModels = [...new Set(significanceData.flatMap(r => [r.model_a, r.model_b]))];
  const n = allModels.length;
  const matrix = Array.from({{length: n}}, () => Array(n).fill(null));
  significanceData.forEach(r => {{
    const i = allModels.indexOf(r.model_a), j = allModels.indexOf(r.model_b);
    const p = parseFloat(r.p_value);
    if (i >= 0 && j >= 0) {{ matrix[i][j] = p; matrix[j][i] = p; }}
  }});
  const canvas = document.getElementById('chartPvalueHeatmap');
  const ctx = canvas.getContext('2d');
  const cellW = 85, cellH = 50, offsetX = 130, offsetY = 35;
  canvas.width = offsetX + n * cellW + 10;
  canvas.height = offsetY + n * cellH + 10;
  ctx.font = '11px Inter, sans-serif';
  allModels.forEach((m, i) => {{
    ctx.fillStyle = '#9ca3c4'; ctx.textAlign = 'right';
    ctx.fillText(m, offsetX - 8, offsetY + i * cellH + cellH/2 + 4);
    ctx.save(); ctx.translate(offsetX + i * cellW + cellW/2, offsetY - 6);
    ctx.rotate(-0.4); ctx.textAlign = 'right';
    ctx.fillText(m, 0, 0); ctx.restore();
  }});
  for (let i = 0; i < n; i++) {{
    for (let j = 0; j < n; j++) {{
      const v = i === j ? null : matrix[i][j];
      const x = offsetX + j * cellW, y = offsetY + i * cellH;
      if (v === null) {{
        ctx.fillStyle = '#1a1d2e'; ctx.fillRect(x, y, cellW-2, cellH-2);
        if (i === j) {{ ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.fillText('-', x+cellW/2, y+cellH/2+4); }}
      }} else {{
        const sig = v < 0.05;
        ctx.fillStyle = sig ? 'rgba(0,229,176,' + (0.15 + 0.6*Math.max(0, 1-v/0.05)) + ')' : 'rgba(156,163,196,0.15)';
        ctx.fillRect(x, y, cellW-2, cellH-2);
        ctx.fillStyle = sig ? '#00e5b0' : '#9ca3c4';
        ctx.textAlign = 'center';
        ctx.fillText(v.toFixed(4), x+cellW/2, y+cellH/2+4);
      }}
    }}
  }}
}})();

// ---- 튜닝 차트 ----
(function() {{
  const models = [...new Set(tuningLog.map(r => r.model))];
  const mColors = {{ 'BERT-base': '#7c8aff', 'BERT+VADER': '#ffb347', 'RoBERTa+VADER': '#00e5b0' }};

  // LR
  const lrData = tuningLog.filter(r => r.parameter === 'learning_rate');
  const lrLabels = [...new Set(lrData.map(r => r.candidate))];
  const lrDatasets = models.map(m => {{
    const rows = lrData.filter(r => r.model === m).sort((a,b) => parseFloat(a.candidate) - parseFloat(b.candidate));
    return {{ label: m, data: rows.map(r => parseFloat(r.val_macro_f1)),
      borderColor: mColors[m] || '#fff', backgroundColor: (mColors[m] || '#fff') + '33',
      borderWidth: 2, tension: 0.3, pointRadius: 5, fill: false }};
  }});
  new Chart(document.getElementById('chartTuningLR'), {{
    type: 'line', data: {{ labels: lrLabels, datasets: lrDatasets }},
    options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }},
      scales: {{ x: {{ title: {{ display: true, text: 'Learning Rate' }} }},
               y: {{ min: 0.655, max: 0.69, title: {{ display: true, text: 'Val Macro F1' }} }} }}
    }}
  }});

  // Dropout
  const doData = tuningLog.filter(r => r.parameter === 'dropout');
  const doLabels = [...new Set(doData.map(r => r.candidate))];
  const doDatasets = models.map(m => {{
    const rows = doData.filter(r => r.model === m).sort((a,b) => parseFloat(a.candidate) - parseFloat(b.candidate));
    return {{ label: m, data: rows.map(r => parseFloat(r.val_macro_f1)),
      borderColor: mColors[m] || '#fff', backgroundColor: (mColors[m] || '#fff') + '33',
      borderWidth: 2, tension: 0.3, pointRadius: 5, fill: false }};
  }});
  new Chart(document.getElementById('chartTuningDropout'), {{
    type: 'line', data: {{ labels: doLabels, datasets: doDatasets }},
    options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }},
      scales: {{ x: {{ title: {{ display: true, text: 'Dropout' }} }},
               y: {{ min: 0.655, max: 0.69, title: {{ display: true, text: 'Val Macro F1' }} }} }}
    }}
  }});
}})();

// ---- 학습 곡선 (비동기) ----
let lcChart = null;
async function fetchLearningCurves() {{
  const res = await fetch('/api/learning_curves');
  learningCurveData = await res.json();
  updateLearningCurveChart();
}}

function updateLearningCurveChart() {{
  if (!learningCurveData) return;
  const model = document.getElementById('lcModelSelect').value;
  const modelData = learningCurveData[model];
  if (!modelData) return;
  if (lcChart) lcChart.destroy();

  const seeds = Object.keys(modelData).sort();
  const seedColors = ['#7c8aff', '#00e5b0', '#ff6b7a'];
  const datasets = [];
  seeds.forEach((seed, si) => {{
    const rows = modelData[seed];
    datasets.push({{
      label: seed + ' train_loss', data: rows.map(r => r.train_loss),
      borderColor: seedColors[si], borderWidth: 2, borderDash: [6,3],
      yAxisID: 'y', pointRadius: 3, tension: 0.3
    }});
    datasets.push({{
      label: seed + ' val_f1', data: rows.map(r => r.val_macro_f1),
      borderColor: seedColors[si], borderWidth: 2,
      yAxisID: 'y1', pointRadius: 4, tension: 0.3
    }});
  }});
  const maxEp = Math.max(...seeds.map(s => modelData[s].length));
  const labels = Array.from({{length: maxEp}}, (_, i) => i + 1);

  lcChart = new Chart(document.getElementById('chartLearningCurve'), {{
    type: 'line', data: {{ labels, datasets }},
    options: {{
      responsive: true, interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 12 }} }} }},
      scales: {{
        y: {{ type: 'linear', position: 'left', title: {{ display: true, text: 'Train Loss' }}, min: 0.2, max: 1.2 }},
        y1: {{ type: 'linear', position: 'right', title: {{ display: true, text: 'Val Macro F1' }}, min: 0.55, max: 0.72, grid: {{ drawOnChartArea: false }} }}
      }}
    }}
  }});
}}

// ---- Freeze 차트 ----
(function() {{
  const frozenMean = {round(frozen_mean, 6)};
  const finetunedMean = {round(finetuned_mean, 6)};
  const diff = ((finetunedMean - frozenMean) / frozenMean * 100).toFixed(1);
  new Chart(document.getElementById('chartFreeze'), {{
    type: 'bar',
    data: {{
      labels: ['Frozen Encoder', 'Fine-tuned Encoder'],
      datasets: [{{ label: 'Macro F1', data: [frozenMean, finetunedMean],
        backgroundColor: ['#ff6b7a99', '#00e5b099'], borderColor: ['#ff6b7a', '#00e5b0'], borderWidth: 2 }}]
    }},
    options: {{ responsive: true,
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ afterLabel: () => '+' + diff + '% improvement' }} }}
      }},
      scales: {{ y: {{ min: 0, max: 0.8, title: {{ display: true, text: 'Macro F1' }} }} }}
    }},
    plugins: [{{
      id: 'diffAnnotation',
      afterDraw(chart) {{
        const meta = chart.getDatasetMeta(0);
        const ctx2 = chart.ctx;
        const bar0 = meta.data[0], bar1 = meta.data[1];
        ctx2.save(); ctx2.font = 'bold 16px Inter, sans-serif'; ctx2.fillStyle = '#00e5b0'; ctx2.textAlign = 'center';
        ctx2.fillText('+' + diff + '%', (bar0.x + bar1.x) / 2, Math.min(bar0.y, bar1.y) - 12);
        ctx2.restore();
      }}
    }}]
  }});
}})();

// ---- VADER 차트 ----
(function() {{
  const vaderStats = (edaData.vader_by_class_stats || []).filter(v => v['class'] !== 'ALL');
  const classes = vaderStats.map(v => v['class']);
  const metrics = ['compound_mean', 'neg_mean', 'pos_mean'];
  const mLabels = ['Compound', 'Negative', 'Positive'];
  const mColors = ['#7c8aff', '#ff6b7a', '#00e5b0'];
  const datasets = metrics.map((m, mi) => ({{
    label: mLabels[mi], data: vaderStats.map(v => v[m]),
    backgroundColor: mColors[mi] + '99', borderColor: mColors[mi], borderWidth: 2
  }}));
  new Chart(document.getElementById('chartVader'), {{
    type: 'bar', data: {{ labels: classes, datasets }},
    options: {{ responsive: true, plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'VADER Score' }} }} }}
    }}
  }});
}})();

// ---- 타겟 커뮤니티 차트 ----
(function() {{
  const targets = (edaData.top_targets || []).slice(0, 13);
  new Chart(document.getElementById('chartTargets'), {{
    type: 'bar',
    data: {{
      labels: targets.map(t => t.target),
      datasets: [{{ label: 'Count', data: targets.map(t => t.count),
        backgroundColor: targets.map((_, i) => COLORS[i % COLORS.length] + '99'),
        borderColor: targets.map((_, i) => COLORS[i % COLORS.length]), borderWidth: 1 }}]
    }},
    options: {{ indexAxis: 'y', responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ x: {{ title: {{ display: true, text: 'Count' }} }} }}
    }}
  }});
}})();

// ---- Overlap@5 차트 ----
(function() {{
  const sampleIds = [...new Set(overlapData.map(r => r.sample_id))];
  const baselineMap = {{}};
  const improvedMap = {{}};
  overlapData.forEach(r => {{
    if (r.model === 'BERT-base') baselineMap[r.sample_id] = parseFloat(r.overlap_at_5);
    else improvedMap[r.sample_id] = parseFloat(r.overlap_at_5);
  }});
  new Chart(document.getElementById('chartOverlap'), {{
    type: 'bar',
    data: {{
      labels: sampleIds.map(id => '#' + id),
      datasets: [
        {{ label: 'BERT-base', data: sampleIds.map(id => baselineMap[id] || 0),
          backgroundColor: '#7c8aff99', borderColor: '#7c8aff', borderWidth: 1 }},
        {{ label: 'RoBERTa+VADER', data: sampleIds.map(id => improvedMap[id] || 0),
          backgroundColor: '#00e5b099', borderColor: '#00e5b0', borderWidth: 1 }}
      ]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ min: 0, max: 1.0, title: {{ display: true, text: 'Overlap@5' }} }} }}
    }}
  }});
}})();

// ---- Human Rationale 차트 ----
(function() {{
  // SHAP 기준 Baseline vs Improved 비교 막대 차트
  const shapData = rationaleData.filter(r => r.xai_method === 'SHAP');
  const baseShap = shapData.filter(r => r.model === 'BERT-base').map(r => parseFloat(r.overlap));
  const impShap = shapData.filter(r => r.model !== 'BERT-base').map(r => parseFloat(r.overlap));
  const baseMean = baseShap.length ? baseShap.reduce((a,b)=>a+b,0)/baseShap.length : 0;
  const impMean = impShap.length ? impShap.reduce((a,b)=>a+b,0)/impShap.length : 0;

  const limeData = rationaleData.filter(r => r.xai_method === 'LIME');
  const baseLime = limeData.filter(r => r.model === 'BERT-base').map(r => parseFloat(r.overlap));
  const impLime = limeData.filter(r => r.model !== 'BERT-base').map(r => parseFloat(r.overlap));
  const baseLimeMean = baseLime.length ? baseLime.reduce((a,b)=>a+b,0)/baseLime.length : 0;
  const impLimeMean = impLime.length ? impLime.reduce((a,b)=>a+b,0)/impLime.length : 0;

  const canvas = document.getElementById('chartRationale');
  if (canvas && (baseShap.length > 0 || impShap.length > 0)) {{
    new Chart(canvas, {{
      type: 'bar',
      data: {{
        labels: ['SHAP Top-5', 'LIME Top-5'],
        datasets: [
          {{ label: 'BERT-base', data: [baseMean, baseLimeMean],
            backgroundColor: '#7c8aff99', borderColor: '#7c8aff', borderWidth: 2 }},
          {{ label: 'RoBERTa+VADER', data: [impMean, impLimeMean],
            backgroundColor: '#00e5b099', borderColor: '#00e5b0', borderWidth: 2 }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ position: 'top' }},
          title: {{ display: true, text: 'Model Top-5 vs Human Rationale (Mean Overlap)' }}
        }},
        scales: {{
          y: {{ min: 0, max: 1.0, title: {{ display: true, text: 'Overlap' }},
            ticks: {{ callback: v => (v*100).toFixed(0) + '%' }} }}
        }}
      }}
    }});
  }}
}})();

// ===== Playground Functions =====
// 디바이스 상태 확인
fetch('/api/predict/status').then(r=>r.json()).then(d=>{{
  const el=document.getElementById('pg-device');
  if(el) el.innerHTML=`<span style="color:var(--accent-green)">Device: ${{d.device}}</span> | <span class="ko">사용 가능: </span><span class="en">Available: </span>${{d.available_models.join(', ')}}`;
}}).catch(()=>{{}});

function pgFill(text) {{
  document.getElementById('pg-input').value = text;
}}

async function pgPredict(withLime=false) {{
  const text = document.getElementById('pg-input').value.trim();
  if (!text) return;

  const models = [...document.querySelectorAll('.pg-check input:checked')].map(c=>c.value);
  if (models.length === 0) return;

  const btn = document.getElementById('pg-btn');
  const limeBtn = document.getElementById('pg-lime-btn');
  const loading = document.getElementById('pg-loading');
  const resultsDiv = document.getElementById('pg-results');

  btn.disabled = true;
  if(limeBtn) limeBtn.disabled = true;
  loading.style.display = 'block';
  if(withLime) loading.querySelector('div:last-child').innerHTML = '<span class="ko">LIME 분석 중... (모델당 5-10초 소요)</span><span class="en">Running LIME... (5-10s per model)</span>';
  else loading.querySelector('div:last-child').innerHTML = '<span class="ko">모델 추론 중... (첫 요청은 모델 로딩으로 10-15초 소요)</span><span class="en">Running inference... (first request takes 10-15s to load models)</span>';
  resultsDiv.innerHTML = '';

  try {{
    const resp = await fetch('/api/predict', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ text, models, run_lime: withLime }})
    }});
    const data = await resp.json();

    if (data.error) {{
      resultsDiv.innerHTML = `<div class="insight-box insight-red">${{data.error}}</div>`;
      return;
    }}

    const labelColors = {{ hatespeech: '#ff6b7a', offensive: '#ffb347', normal: '#00e5b0' }};
    const labelBg = {{ hatespeech: 'rgba(255,107,122,0.1)', offensive: 'rgba(255,179,71,0.1)', normal: 'rgba(0,229,176,0.1)' }};

    let html = `<div style="font-size:12px;color:var(--text-muted);margin-bottom:14px">Device: ${{data.device}}</div>`;

    // ---- Attention Heatmap Section (all models) ----
    html += `<div class="card" style="margin-bottom:16px">`;
    html += `<h3 style="color:var(--accent-blue);margin-bottom:4px"><span class="ko">Attention Heatmap -- 모델이 주목하는 토큰</span><span class="en">Attention Heatmap -- Tokens the Model Focuses On</span></h3>`;
    html += `<div style="font-size:12px;color:var(--text-muted);margin-bottom:14px"><span class="ko">색이 진할수록 [CLS] 토큰이 해당 단어에 더 많이 주목합니다 (마지막 레이어, 모든 head 평균).</span><span class="en">Darker = more attention from [CLS] token (last layer, all heads averaged).</span></div>`;

    for (const [modelName, result] of Object.entries(data.results)) {{
      if (result.error || !result.token_attention) continue;
      const attn = result.token_attention;
      const color = labelColors[result.label] || '#7c8aff';

      html += `<div style="margin-bottom:14px">`;
      html += `<div style="font-size:13px;font-weight:600;margin-bottom:6px;color:${{color}}">${{modelName}} <span style="font-weight:400;color:var(--text-muted)">→ ${{result.label}}</span></div>`;
      html += `<div style="display:flex;flex-wrap:wrap;gap:3px;line-height:2">`;
      for (const tok of attn) {{
        const opacity = Math.max(0.08, tok.attention);
        const bg = `rgba(${{result.label==='hatespeech'?'255,107,122':result.label==='offensive'?'255,179,71':'0,229,176'}},${{opacity.toFixed(2)}})`;
        const textColor = tok.attention > 0.6 ? 'white' : 'var(--text-primary)';
        const border = tok.attention > 0.8 ? `2px solid ${{color}}` : '1px solid transparent';
        html += `<span title="attention: ${{tok.raw.toFixed(5)}}" style="display:inline-block;padding:2px 6px;border-radius:4px;font-size:13px;background:${{bg}};color:${{textColor}};border:${{border}};cursor:default">${{tok.token}}</span>`;
      }}
      html += `</div></div>`;
    }}
    html += `</div>`;

    // ---- Token Interaction Heatmap (전체 attention 행렬) ----
    const hasInteraction = Object.values(data.results).some(r => r.interaction_matrix && r.interaction_tokens);
    if (hasInteraction) {{
      html += `<div class="card" style="margin-bottom:16px">`;
      html += `<h3 style="color:var(--accent-orange);margin-bottom:4px"><span class="ko">토큰 상호작용 행렬 -- 토큰이 서로를 얼마나 참조하는가</span><span class="en">Token Interaction Matrix -- How tokens attend to each other</span></h3>`;
      html += `<div style="font-size:12px;color:var(--text-muted);margin-bottom:14px"><span class="ko">행(row) = 해당 토큰이, 열(col) = 참조하는 대상. 색이 진할수록 강한 참조 (마지막 레이어, head 평균, 행 정규화).</span><span class="en">Row = source token, Col = attended token. Darker = stronger attention (last layer, head avg, row-normalized).</span></div>`;

      for (const [modelName, result] of Object.entries(data.results)) {{
        if (result.error || !result.interaction_matrix || !result.interaction_tokens) continue;
        const matrix = result.interaction_matrix;
        const tokens = result.interaction_tokens;
        const n = tokens.length;
        const color = labelColors[result.label] || '#7c8aff';
        const canvasId = `interaction-${{modelName.replace(/[^a-zA-Z0-9]/g, '_')}}`;

        html += `<div style="margin-bottom:20px">`;
        html += `<div style="font-size:13px;font-weight:600;margin-bottom:8px;color:${{color}}">${{modelName}} <span style="font-weight:400;color:var(--text-muted)">→ ${{result.label}} (${{n}} tokens)</span></div>`;
        html += `<div style="overflow-x:auto"><canvas id="${{canvasId}}" style="display:block;margin:0 auto"></canvas></div>`;
        html += `</div>`;
      }}
      html += `</div>`;
    }}

    // ---- LIME Section (if requested) ----
    const hasLime = Object.values(data.results).some(r => r.lime && r.lime.length > 0);
    if (hasLime) {{
      html += `<div class="card" style="margin-bottom:16px">`;
      html += `<h3 style="color:var(--accent-purple);margin-bottom:4px"><span class="ko">LIME 설명 -- 예측에 기여한 단어</span><span class="en">LIME Explanation -- Words Contributing to Prediction</span></h3>`;
      html += `<div style="font-size:12px;color:var(--text-muted);margin-bottom:14px"><span class="ko">양수(초록)=해당 라벨 방향, 음수(빨강)=반대 방향. 입력을 교란하여 로컬 선형 모델로 기여도를 추정합니다.</span><span class="en">Positive (green) = towards predicted label, Negative (red) = against. Estimated by perturbing input and fitting local linear model.</span></div>`;

      for (const [modelName, result] of Object.entries(data.results)) {{
        if (result.error || !result.lime) continue;
        const color = labelColors[result.label] || '#7c8aff';
        const maxAbsWeight = Math.max(...result.lime.map(l => Math.abs(l.weight)), 0.001);

        html += `<div style="margin-bottom:16px">`;
        html += `<div style="font-size:13px;font-weight:600;margin-bottom:8px;color:${{color}}">${{modelName}} <span style="font-weight:400;color:var(--text-muted)">→ ${{result.label}}</span></div>`;

        for (const item of result.lime) {{
          const pct = Math.abs(item.weight) / maxAbsWeight * 100;
          const isPositive = item.weight > 0;
          const barColor = isPositive ? 'var(--accent-green)' : 'var(--accent-red)';
          html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:12px">`;
          html += `<span style="width:100px;text-align:right;color:var(--text-primary);font-weight:500">${{item.token}}</span>`;
          html += `<div style="flex:1;height:14px;background:var(--bg-dark);border-radius:3px;overflow:hidden;position:relative">`;
          if (isPositive) {{
            html += `<div style="position:absolute;left:50%;width:${{pct/2}}%;height:100%;background:${{barColor}};border-radius:0 3px 3px 0"></div>`;
          }} else {{
            html += `<div style="position:absolute;right:50%;width:${{pct/2}}%;height:100%;background:${{barColor}};border-radius:3px 0 0 3px"></div>`;
          }}
          html += `<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--text-muted)"></div>`;
          html += `</div>`;
          html += `<span style="width:60px;color:${{isPositive?'var(--accent-green)':'var(--accent-red)'}};font-size:11px">${{item.weight > 0 ? '+' : ''}}${{item.weight.toFixed(4)}}</span>`;
          html += `</div>`;
        }}
        html += `</div>`;
      }}
      html += `</div>`;
    }}

    // ---- Prediction Cards ----
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px">';
    let vaderHtml = '';

    for (const [modelName, result] of Object.entries(data.results)) {{
      if (result.error) {{
        html += `<div class="card" style="border-color:var(--accent-red)"><h3>${{modelName}}</h3><p style="color:var(--accent-red)">${{result.error}}</p></div>`;
        continue;
      }}

      const probs = result.probabilities;
      const label = result.label;
      const color = labelColors[label] || '#7c8aff';
      const maxProb = Math.max(probs.hatespeech, probs.offensive, probs.normal);

      html += `<div class="card" style="border-left:3px solid ${{color}}">`;
      html += `<h3 style="margin-bottom:8px">${{modelName}}</h3>`;
      html += `<div style="font-size:22px;font-weight:700;color:${{color}};margin-bottom:10px">${{label.toUpperCase()}}</div>`;

      for (const cls of ['hatespeech', 'offensive', 'normal']) {{
        const p = probs[cls]; const pct = (p * 100).toFixed(1);
        const isMax = (p === maxProb);
        html += `<div style="margin-bottom:5px"><div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:1px;${{isMax?'font-weight:700;color:var(--text-primary)':'color:var(--text-muted)'}}"><span>${{cls}}</span><span>${{pct}}%</span></div>`;
        html += `<div style="background:var(--bg-dark);border-radius:3px;height:6px;overflow:hidden"><div style="width:${{pct}}%;height:100%;background:${{labelColors[cls]}};border-radius:3px;transition:width 0.5s"></div></div></div>`;
      }}

      const confidence = maxProb > 0.7 ? 'High' : maxProb > 0.45 ? 'Medium' : 'Low';
      const confColor = maxProb > 0.7 ? 'var(--accent-green)' : maxProb > 0.45 ? 'var(--accent-orange)' : 'var(--accent-red)';
      html += `<div style="margin-top:8px;font-size:11px;color:${{confColor}}">Confidence: ${{confidence}} (${{(maxProb*100).toFixed(1)}}%)</div>`;
      html += `</div>`;

      if (result.vader_scores && !vaderHtml) {{
        const vs = result.vader_scores;
        vaderHtml = `<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px">`;
        for (const [k, v] of Object.entries(vs)) {{
          const vc = k === 'compound' ? (v < -0.3 ? 'var(--accent-red)' : v > 0.3 ? 'var(--accent-green)' : 'var(--accent-orange)') : 'var(--text-muted)';
          vaderHtml += `<div style="text-align:center;background:var(--bg-dark);padding:10px;border-radius:8px"><div style="font-size:18px;font-weight:700;color:${{vc}}">${{v.toFixed(4)}}</div><div style="font-size:10px;color:var(--text-muted)">${{k}}</div></div>`;
        }}
        vaderHtml += `</div>`;
      }}
    }}
    html += '</div>';

    // Model agreement
    const predictions = Object.entries(data.results).filter(([_,r])=>!r.error).map(([m,r])=>({{model:m,label:r.label}}));
    const uniqueLabels = [...new Set(predictions.map(p=>p.label))];
    if (predictions.length > 1) {{
      if (uniqueLabels.length === 1) {{
        html += `<div class="insight-box insight-green" style="margin-top:14px"><strong><span class="ko">모델 합의:</span><span class="en">Consensus:</span></strong> <span class="ko">모든 모델이 <strong>${{uniqueLabels[0]}}</strong>으로 일치합니다.</span><span class="en">All models agree: <strong>${{uniqueLabels[0]}}</strong>.</span></div>`;
      }} else {{
        const dis = predictions.map(p=>`${{p.model}}=${{p.label}}`).join(', ');
        html += `<div class="insight-box insight-orange" style="margin-top:14px"><strong><span class="ko">모델 불일치:</span><span class="en">Disagreement:</span></strong> ${{dis}}</div>`;
      }}
    }}

    // VADER
    if (vaderHtml) {{
      html += `<div class="card" style="margin-top:14px"><h3>VADER Sentiment</h3>`;
      html += `<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px"><span class="ko">compound: -1(매우 부정) ~ +1(매우 긍정)</span><span class="en">compound: -1 (very negative) ~ +1 (very positive)</span></div>`;
      html += vaderHtml + `</div>`;
    }}

    resultsDiv.innerHTML = html;

    // ---- Canvas 히트맵 렌더링 (토큰 상호작용 행렬) ----
    for (const [modelName, result] of Object.entries(data.results)) {{
      if (!result.interaction_matrix || !result.interaction_tokens) continue;
      const matrix = result.interaction_matrix;
      const tokens = result.interaction_tokens;
      const n = tokens.length;
      const canvasId = `interaction-${{modelName.replace(/[^a-zA-Z0-9]/g, '_')}}`;
      const canvas = document.getElementById(canvasId);
      if (!canvas) continue;

      const cellSize = Math.max(28, Math.min(48, 600 / n));
      const labelSpace = 80;  // 토큰 라벨 공간
      const topLabelSpace = 70;  // 상단 회전 라벨 공간
      const w = labelSpace + n * cellSize;
      const h = topLabelSpace + n * cellSize;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      // 히트맵 셀 그리기
      for (let i = 0; i < n; i++) {{
        for (let j = 0; j < n; j++) {{
          const val = matrix[i][j];
          // 색상: 낮은 값 = 어두운 배경, 높은 값 = 주황~빨강
          const r = Math.round(255 * val);
          const g = Math.round(140 * val);
          const b = Math.round(50 * (1 - val));
          ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
          ctx.fillRect(labelSpace + j * cellSize, topLabelSpace + i * cellSize, cellSize - 1, cellSize - 1);

          // 셀 값 텍스트 (셀이 충분히 클 때만)
          if (cellSize >= 32) {{
            ctx.fillStyle = val > 0.5 ? '#fff' : '#aaa';
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(val.toFixed(2), labelSpace + j * cellSize + cellSize / 2, topLabelSpace + i * cellSize + cellSize / 2);
          }}
        }}
      }}

      // 왼쪽 라벨 (row: source 토큰)
      ctx.fillStyle = '#e0e0e0';
      ctx.font = '11px monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < n; i++) {{
        const label = tokens[i].length > 10 ? tokens[i].slice(0, 9) + '..' : tokens[i];
        ctx.fillText(label, labelSpace - 6, topLabelSpace + i * cellSize + cellSize / 2);
      }}

      // 상단 라벨 (col: target 토큰, 회전)
      ctx.save();
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#e0e0e0';
      for (let j = 0; j < n; j++) {{
        const x = labelSpace + j * cellSize + cellSize / 2;
        const y = topLabelSpace - 6;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 3);
        const label = tokens[j].length > 10 ? tokens[j].slice(0, 9) + '..' : tokens[j];
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }}
      ctx.restore();
    }}

  }} catch(e) {{
    resultsDiv.innerHTML = `<div class="insight-box insight-red">Error: ${{e.message}}</div>`;
  }} finally {{
    btn.disabled = false;
    if(limeBtn) limeBtn.disabled = false;
    loading.style.display = 'none';
  }}
}}

// ===== 이미지 라이트박스 =====
function openLightbox(src) {{
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('active');
  document.body.style.overflow = 'hidden';
}}
function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('active');
  document.body.style.overflow = '';
}}
document.addEventListener('keydown', e => {{ if(e.key==='Escape') closeLightbox(); }});
// 모든 갤러리 이미지에 라이트박스 연결
document.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('.gallery-item img, .gallery img').forEach(img => {{
    img.addEventListener('click', () => openLightbox(img.src));
  }});
}});

// ===== Data Explorer =====
let explorerCurrentPage = 1;
async function explorerSearch(page) {{
  explorerCurrentPage = page || 1;
  const q = document.getElementById('explorer-query').value;
  const label = document.getElementById('explorer-label').value;
  const statusEl = document.getElementById('explorer-status');
  const resultsEl = document.getElementById('explorer-results');
  const paginationEl = document.getElementById('explorer-pagination');

  statusEl.textContent = document.body.classList.contains('en-mode') ? 'Searching...' : '검색 중...';
  resultsEl.innerHTML = '';
  paginationEl.innerHTML = '';

  try {{
    const resp = await fetch(`/api/data_explorer?q=${{encodeURIComponent(q)}}&label=${{encodeURIComponent(label)}}&page=${{explorerCurrentPage}}&limit=20`);
    const data = await resp.json();

    const totalMsg = document.body.classList.contains('en-mode')
      ? `${{data.total}} results found (Page ${{data.page}}/${{data.total_pages}})`
      : `${{data.total}}건 검색됨 (${{data.page}}/${{data.total_pages}} 페이지)`;
    statusEl.textContent = totalMsg;

    if (data.results.length === 0) {{
      resultsEl.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--text-secondary)"><span class="ko">검색 결과가 없습니다. 데이터 파일이 data/ 디렉토리에 있는지 확인하세요.</span><span class="en">No results found. Make sure data files are in the data/ directory.</span></div>`;
      return;
    }}

    const labelColors = {{ hatespeech: 'var(--accent-red)', offensive: 'var(--accent-orange)', normal: 'var(--accent-green)' }};
    const labelBg = {{ hatespeech: 'rgba(255,107,122,0.15)', offensive: 'rgba(255,179,71,0.15)', normal: 'rgba(0,229,176,0.15)' }};

    let html = '';
    for (const r of data.results) {{
      const color = labelColors[r.label] || 'var(--text-secondary)';
      const bg = labelBg[r.label] || 'rgba(156,163,196,0.1)';
      const compound = r.vader_compound !== '' ? parseFloat(r.vader_compound).toFixed(4) : 'N/A';
      html += `<div class="explorer-card">
        <div class="explorer-text">${{r.text || '(empty)'}}</div>
        <div class="explorer-meta">
          <span style="background:${{bg}};color:${{color}};padding:2px 10px;border-radius:12px;font-weight:600">${{r.label}}</span>
          <span>VADER: ${{compound}}</span>
          <span><span class="ko">단어 수:</span><span class="en">Words:</span> ${{r.word_count}}</span>
        </div>
      </div>`;
    }}
    resultsEl.innerHTML = html;

    // 페이지네이션 버튼
    let pagHtml = '';
    pagHtml += `<button onclick="explorerSearch(${{data.page - 1}})" ${{data.page <= 1 ? 'disabled' : ''}}>&laquo; <span class="ko">이전</span><span class="en">Prev</span></button>`;
    pagHtml += `<span style="color:var(--text-secondary);font-size:0.9rem">${{data.page}} / ${{data.total_pages}}</span>`;
    pagHtml += `<button onclick="explorerSearch(${{data.page + 1}})" ${{data.page >= data.total_pages ? 'disabled' : ''}}><span class="ko">다음</span><span class="en">Next</span> &raquo;</button>`;
    paginationEl.innerHTML = pagHtml;

  }} catch(e) {{
    statusEl.textContent = 'Error: ' + e.message;
  }}
}}

// ===== Model Comparison =====
let cmpChart = null;
function initComparison() {{
  const selA = document.getElementById('cmp-model-a');
  const selB = document.getElementById('cmp-model-b');
  if (!selA || !selB || !benchmarkData.length) return;

  // 드롭다운 채우기
  benchmarkData.forEach((r, i) => {{
    selA.innerHTML += `<option value="${{i}}" ${{i===0?'selected':''}}>${{r.model}}</option>`;
    selB.innerHTML += `<option value="${{i}}" ${{i===1?'selected':''}}>${{r.model}}</option>`;
  }});
  updateComparison();
}}

function updateComparison() {{
  const idxA = parseInt(document.getElementById('cmp-model-a').value);
  const idxB = parseInt(document.getElementById('cmp-model-b').value);
  const a = benchmarkData[idxA];
  const b = benchmarkData[idxB];
  if (!a || !b) return;

  document.getElementById('cmp-title-a').textContent = a.model;
  document.getElementById('cmp-title-b').textContent = b.model;

  const metricKeys = [
    ['macro_f1_mean', 'Macro F1'],
    ['accuracy_mean', 'Accuracy'],
    ['auroc_mean', 'AUROC'],
    ['per_class_f1.hatespeech_mean', 'Hatespeech F1'],
    ['per_class_f1.offensive_mean', 'Offensive F1'],
    ['per_class_f1.normal_mean', 'Normal F1']
  ];

  // 지표 카드 렌더링
  function renderMetrics(model, targetId) {{
    let html = '';
    for (const [key, label] of metricKeys) {{
      const val = parseFloat(model[key]);
      html += `<div class="delta-row"><span>${{label}}</span><span style="font-weight:700">${{isNaN(val) ? 'N/A' : val.toFixed(4)}}</span></div>`;
    }}
    document.getElementById(targetId).innerHTML = html;
  }}
  renderMetrics(a, 'cmp-metrics-a');
  renderMetrics(b, 'cmp-metrics-b');

  // Delta 분석
  let deltaHtml = '';
  for (const [key, label] of metricKeys) {{
    const va = parseFloat(a[key]);
    const vb = parseFloat(b[key]);
    const diff = vb - va;
    const cls = diff > 0.001 ? 'delta-positive' : (diff < -0.001 ? 'delta-negative' : 'delta-neutral');
    const sign = diff > 0 ? '+' : '';
    const winner = diff > 0.001 ? b.model : (diff < -0.001 ? a.model : '-');
    deltaHtml += `<div class="delta-row">
      <span>${{label}}</span>
      <span class="${{cls}}">${{sign}}${{diff.toFixed(4)}}</span>
      <span style="color:var(--text-secondary);font-size:0.8rem">${{winner}}</span>
    </div>`;
  }}
  document.getElementById('cmp-delta').innerHTML = deltaHtml;

  // 레이더 차트
  const radarLabels = metricKeys.map(m => m[1]);
  const dataA = metricKeys.map(([k]) => parseFloat(a[k]));
  const dataB = metricKeys.map(([k]) => parseFloat(b[k]));

  if (cmpChart) cmpChart.destroy();
  cmpChart = new Chart(document.getElementById('chartComparisonRadar'), {{
    type: 'radar',
    data: {{
      labels: radarLabels,
      datasets: [
        {{ label: a.model, data: dataA, borderColor: '#7c8aff', backgroundColor: '#7c8aff22', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#7c8aff' }},
        {{ label: b.model, data: dataB, borderColor: '#00e5b0', backgroundColor: '#00e5b022', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#00e5b0' }}
      ]
    }},
    options: {{
      responsive: true,
      scales: {{ r: {{ min: 0.40, max: 0.90, ticks: {{ stepSize: 0.05 }}, grid: {{ color: 'rgba(124,138,255,0.1)' }} }} }},
      plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 16 }} }} }}
    }}
  }});
}}

// 비교 탭 초기화
document.addEventListener('DOMContentLoaded', () => {{ initComparison(); }});

// ===== Report 자동 생성 =====
function generateReport() {{
  const isEn = document.body.classList.contains('en-mode');
  const container = document.getElementById('report-content');
  if (!container || !benchmarkData.length) return;

  const best = benchmarkData[0]; // 이미 F1 기준 정렬됨
  const worst = benchmarkData[benchmarkData.length - 1];

  let html = '';

  // Executive Summary
  html += `<h2>${{isEn ? 'Executive Summary' : '요약 (Executive Summary)'}}</h2>`;
  html += `<p>${{isEn
    ? `This report summarizes the HateXplain hate speech detection experiment. ${{benchmarkData.length}} conditions were evaluated using the v2 15-seed paired design. The best performing condition is <strong>${{best.model}}</strong> with Macro F1 of <strong>${{parseFloat(best.macro_f1_mean).toFixed(4)}}</strong>.`
    : `본 보고서는 HateXplain 혐오표현 탐지 실험 결과를 요약한다. ${{benchmarkData.length}}개 조건을 v2 15-seed paired design으로 평가하였다. 최고 성능 조건은 <strong>${{best.model}}</strong>이며 Macro F1은 <strong>${{parseFloat(best.macro_f1_mean).toFixed(4)}}</strong>이다.`
  }}</p>`;

  // Key Metrics Table
  html += `<h2>${{isEn ? 'Key Metrics' : '주요 지표'}}</h2>`;
  html += `<table><thead><tr><th>Model</th><th>Macro F1</th><th>Accuracy</th><th>AUROC</th></tr></thead><tbody>`;
  for (const r of benchmarkData) {{
    const isBest = r === best;
    const style = isBest ? ' style="color:var(--accent-green);font-weight:700"' : '';
    html += `<tr${{style}}><td>${{r.model}}</td><td>${{parseFloat(r.macro_f1_mean).toFixed(4)}}</td><td>${{parseFloat(r.accuracy_mean).toFixed(4)}}</td><td>${{parseFloat(r.auroc_mean).toFixed(4)}}</td></tr>`;
  }}
  html += `</tbody></table>`;

  // Statistical Findings
  html += `<h2>${{isEn ? 'Statistical Findings' : '통계적 발견'}}</h2>`;
  const sigPairs = significanceData.filter(r => r.significant === 'True' || r.significant === true);
  if (sigPairs.length > 0) {{
    html += `<p>${{isEn
      ? `${{sigPairs.length}} model pairs showed statistically significant differences (p < 0.05):`
      : `${{sigPairs.length}}개 모델 쌍에서 통계적으로 유의미한 차이가 확인되었다 (p < 0.05):`
    }}</p><ul>`;
    for (const s of sigPairs.slice(0, 8)) {{
      html += `<li>${{s.model_a}} vs ${{s.model_b}} (p=${{parseFloat(s.p_value).toFixed(6)}})</li>`;
    }}
    html += `</ul>`;
  }} else {{
    html += `<p>${{isEn ? 'No statistically significant differences found among model pairs.' : '모델 쌍 간 통계적으로 유의미한 차이가 발견되지 않았다.'}}</p>`;
  }}

  // XAI Findings
  html += `<h2>${{isEn ? 'XAI Findings' : 'XAI 분석 결과'}}</h2>`;
  html += `<p>${{isEn
    ? 'SHAP and LIME were used as post-hoc verification tools to analyze model prediction rationale. Overlap@5 measures agreement between SHAP and LIME top-5 important tokens.'
    : 'SHAP과 LIME을 사후 검증 도구로 활용하여 모델의 예측 근거를 분석하였다. Overlap@5는 SHAP과 LIME 상위 5개 토큰의 일치도를 측정한다.'
  }}</p>`;

  // Limitations
  html += `<h2>${{isEn ? 'Limitations' : '한계점'}}</h2>`;
  html += `<ul>`;
  html += `<li>${{isEn ? 'English-only dataset (HateXplain) -- results may not generalize to other languages' : '영어 전용 데이터셋(HateXplain) -- 다른 언어로의 일반화에 한계가 있다'}}</li>`;
  html += `<li>${{isEn ? 'Smoke runs validate execution only; final statistical claims require the full 15-seed run' : 'smoke run은 실행 검증용이며, 최종 통계 주장은 full 15-seed 결과가 필요하다'}}</li>`;
  html += `<li>${{isEn ? 'VADER sentiment is lexicon-based and has known limitations with sarcasm and context' : 'VADER 감성 분석은 사전 기반으로, 풍자나 맥락 파악에 한계가 있다'}}</li>`;
  html += `<li>${{isEn ? 'Hate/offensive boundary is inherently subjective and annotator-dependent' : '혐오/공격적 경계는 본질적으로 주관적이며 주석자에 의존한다'}}</li>`;
  html += `</ul>`;

  // 생성 시간 기록
  html += `<hr style="margin:20px 0;border-color:var(--border-subtle)">`;
  html += `<p style="font-size:0.8rem;color:var(--text-secondary)">${{isEn ? 'Report generated at' : '보고서 생성 시각'}}: ${{new Date().toLocaleString()}}</p>`;

  container.innerHTML = html;
}}

function copyReport() {{
  const container = document.getElementById('report-content');
  if (!container) return;
  const text = container.innerText;
  navigator.clipboard.writeText(text).then(() => {{
    const isEn = document.body.classList.contains('en-mode');
    alert(isEn ? 'Report copied to clipboard!' : '보고서가 클립보드에 복사되었습니다!');
  }}).catch(err => {{
    alert('Copy failed: ' + err);
  }});
}}

// 리포트 탭 접근 시 생성
document.addEventListener('DOMContentLoaded', () => {{ generateReport(); }});
</script>
<style>
/* Playground-specific styles */
@keyframes spin {{ from {{transform:rotate(0deg)}} to {{transform:rotate(360deg)}} }}
.pg-check {{ display:inline-flex;align-items:center;gap:4px;font-size:13px;color:var(--text-muted);cursor:pointer;padding:4px 10px;background:var(--bg-dark);border-radius:6px;border:1px solid var(--border-color) }}
.pg-check input {{ accent-color:var(--accent-blue) }}
.pg-example:hover {{ border-color:var(--accent-blue) !important; }}
</style>
</body>
</html>"""

    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
