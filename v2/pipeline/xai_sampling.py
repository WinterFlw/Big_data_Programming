"""Sample selection utilities for v2 XAI stages.

XAI 비교의 핵심 전제는 "모든 (condition, seed) checkpoint가 정확히 같은 sample에 대해
attribution을 만든다"는 것이다. seed 의존적인 sample 선택은 explanation-stability
지표(top-k Jaccard, rank correlation 등) 자체를 망가뜨린다. 따라서 본 모듈은
seed=0의 numpy RandomState로 stratified sample을 뽑고, 그 결과를 manifest와
sample_size만으로 결정되는 sample_id 리스트로 돌려준다.

본 모듈은 runtime 데이터 적재 함수에 의존하지만, runtime 모듈 자체는 부르는 측에서
import해서 넘긴다 (sys.path 조작이 pipeline.xai.py에 집중되도록).
"""

from __future__ import annotations

from typing import Any

import numpy as np


SAMPLE_COLUMNS = [
    "sample_id",
    "label",
    "label_name",
    "case_type",
    "source",
    "target",
    "rationale_count",
    "text",
]


def _frame_for_xai(runtime_core: Any):
    """v2 데이터 split 중 test split DataFrame을 반환.

    runtime_core.load_splits()는 train/val/test DataFrame 딕셔너리를 돌려준다.
    XAI 분석은 test split을 표준으로 한다 — train/val은 모델이 본 적이 있어서
    attribution 비교가 데이터 누출과 섞일 수 있기 때문.
    """
    splits = runtime_core.load_splits()
    if "test" not in splits:
        raise KeyError("load_splits() did not return a 'test' DataFrame")
    return splits["test"]


def _rationale_count(value: Any) -> int:
    """rationale_mask 컬럼 값에서 1의 개수를 안전하게 센다."""
    if value is None:
        return 0
    try:
        return int(sum(int(v) for v in value if int(v) == 1))
    except (TypeError, ValueError):
        return 0


def _row_to_sample(row: Any, case_type: str) -> dict[str, Any]:
    """test DataFrame의 한 행을 sample dict로 평탄화."""
    targets = row.get("targets") if hasattr(row, "get") else None
    if targets is None:
        targets = []
    # multi-label target은 join해서 한 셀에 평탄화. case 분석 단계에서 split 가능.
    target_text = ",".join(str(item) for item in targets if item and str(item) != "None")
    return {
        "sample_id": str(row["post_id"]),
        "label": int(row["label"]),
        "label_name": str(row.get("label_name", "")),
        "case_type": case_type,
        "source": str(row.get("source", "")),
        "target": target_text,
        "rationale_count": _rationale_count(row.get("rationale_mask")),
        "text": str(row["text"]),
    }


def _stratified_indices(
    labels: list[int],
    sample_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """라벨 분포를 보존하면서 sample_size개의 인덱스를 뽑는다.

    클래스별 비율을 그대로 유지하기 위해 비례 할당을 한다. rounding 차이는
    마지막 클래스에 흡수시켜 합을 sample_size로 맞춘다.
    """
    if sample_size >= len(labels):
        # 요청량이 데이터보다 크면 전체 인덱스를 그대로 돌려준다.
        return list(range(len(labels)))

    label_array = np.asarray(labels)
    unique_labels = sorted(set(int(label) for label in labels))
    proportions = {
        label: int(round(sample_size * (label_array == label).sum() / len(labels)))
        for label in unique_labels
    }
    # rounding 보정.
    diff = sample_size - sum(proportions.values())
    if diff != 0 and unique_labels:
        proportions[unique_labels[-1]] += diff

    selected: list[int] = []
    for label in unique_labels:
        target_count = max(0, proportions[label])
        candidates = np.where(label_array == label)[0]
        if target_count >= candidates.size:
            selected.extend(candidates.tolist())
            continue
        chosen = rng.choice(candidates, size=target_count, replace=False)
        selected.extend(int(value) for value in chosen)
    selected.sort()
    return selected


def select_primary_samples(
    runtime_core: Any,
    manifest: dict[str, Any],
    sample_size: int,
) -> list[dict[str, Any]]:
    """Primary XAI용 stratified 200 sample. seed 무관 고정 sample_id.

    Primary는 A_B vs D_B를 15 seed 모두에서 같은 sample로 비교하는 분석이라,
    sample 선택이 seed에 의존하면 안 된다. numpy seed=0 고정으로 manifest와
    sample_size만이 입력이 되는 결정적 선택을 보장한다.
    """
    frame = _frame_for_xai(runtime_core)
    rng = np.random.default_rng(seed=0)
    indices = _stratified_indices(frame["label"].tolist(), sample_size, rng)
    return [_row_to_sample(frame.iloc[index], "primary") for index in indices]


def select_deep_samples(
    runtime_core: Any,
    manifest: dict[str, Any],
    sample_size: int,
) -> list[dict[str, Any]]:
    """Deep XAI용 median seed × 500 stratified sample. 정성 case 분석.

    Deep은 median seed checkpoint 1개로 더 큰 sample을 본다. sample 선택은
    Primary와 같이 seed=0 결정적이며, sample_size만 다르다.
    """
    frame = _frame_for_xai(runtime_core)
    rng = np.random.default_rng(seed=1)  # Primary와 다른 stream을 쓰기 위해 1로 분리.
    indices = _stratified_indices(frame["label"].tolist(), sample_size, rng)
    return [_row_to_sample(frame.iloc[index], "deep") for index in indices]


def select_ablation_samples(
    runtime_core: Any,
    manifest: dict[str, Any],
    sample_size: int,
) -> list[dict[str, Any]]:
    """Ablation 매트릭스용 50 sample. 8조건 × median seed로 directional check."""
    frame = _frame_for_xai(runtime_core)
    rng = np.random.default_rng(seed=2)
    indices = _stratified_indices(frame["label"].tolist(), sample_size, rng)
    return [_row_to_sample(frame.iloc[index], "ablation") for index in indices]
