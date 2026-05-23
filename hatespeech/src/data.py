"""
데이터 모듈 (src/data.py)
==========================
HateXplain 원본 데이터 다운로드 -> 라벨링/정제 -> train/val/test 분할,
그리고 PyTorch Dataset 클래스를 담아요. ucam의 `src/data.py`에 대응해요.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import ExperimentConfig, get_config, save_config
from src.path import (
    RAW_DATASET_PATH,
    RAW_SPLIT_PATH,
    SPLITS_PICKLE_PATH,
    DATA_PROFILE_PATH,
    DATA_DIR,
    RAW_PATH,
    REPORT_DIR,
)
from src.utils import (
    LABEL_NAMES,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    VADER_COLUMNS,
    dataframe_to_markdown,
    ensure_dir,
    save_dataframe,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_text,
    set_seed,
    slugify,
    plot_split_distribution,
)




# ── 데이터 다운로드 ─────────────────────────────
# HateXplain 데이터셋이 없으면 GitHub에서 자동으로 받아와요.
# 이미 있으면 건너뛰니까 걱정 마세요!
def ensure_raw_hatexplain(force_download: bool = False) -> None:
    """HateXplain 원본 데이터가 없으면 GitHub에서 다운로드해요."""
    base_url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url_name, destination in [
        ("dataset.json", RAW_DATASET_PATH),           # 본문 + 어노테이션 전체
        ("post_id_divisions.json", RAW_SPLIT_PATH),   # 원본 train/val/test ID
    ]:
        if force_download or not destination.exists():
            print(f"Downloading {url_name}...")
            urlretrieve(base_url + url_name, destination)


# ── 다수결 투표 로직 ────────────────────────────
# HateXplain은 각 게시물에 3명의 어노테이터가 라벨을 달아요.
# 의견이 갈릴 수 있으니까, 2명 이상 같은 의견이면 그걸 정답으로 삼아요.
# 민주주의처럼 다수결이에요! 3명 다 다른 의견이면 제외한답니다.
def _majority_label(sample: dict[str, Any]) -> int | None:
    """
    3명 어노테이터의 다수결 투표로 라벨을 결정해요.
    2표 이상 일치하지 않으면 None을 반환해서 분석에서 제외해요.
    (HateXplain에서 약 919건이 이렇게 제외돼요)
    """
    # 각 어노테이터의 라벨을 모아서...
    annotator_labels = [annotator["label"] for annotator in sample["annotators"]]
    # Counter로 투표 결과를 세요!
    label_counts = Counter(annotator_labels)
    majority_label, majority_count = label_counts.most_common(1)[0]
    if majority_count < 2:
        return None  # 의견이 갈려서 결정 불가 (undecided)
    return LABEL2ID.get(majority_label)  # 문자열 라벨 → 숫자 ID로 변환


# ── 대상 커뮤니티 수집 ──────────────────────────
# 혐오표현이 누구를 향한 것인지 수집해요 (예: African, Islam, Jewish 등)
def _collect_targets(sample: dict[str, Any]) -> list[str]:
    """어노테이터들이 표시한 대상 커뮤니티를 모아서 중복 제거 후 반환해요."""
    targets: list[str] = []
    for annotator in sample["annotators"]:
        if "target" in annotator:
            targets.extend(annotator["target"])
    return sorted(set(targets)) if targets else ["None"]


def _extract_source(post_id: str) -> str:
    """HateXplain post_id 접미사에서 source(gab/twitter)를 분리해요."""
    if "_" not in post_id:
        return "unknown"
    return post_id.rsplit("_", 1)[-1].lower()


def _annotator_agreement(sample: dict[str, Any]) -> float:
    """3인 annotator 다수결 동의 비율. 학습 입력이 아니라 EDA/보고용입니다."""
    labels = [annotator["label"] for annotator in sample["annotators"]]
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


def _majority_rationale_mask(sample: dict[str, Any]) -> list[int]:
    """3인 rationale mask를 토큰 단위 majority vote로 통합합니다."""
    tokens = sample.get("post_tokens", [])
    rationales = sample.get("rationales") or []
    if not tokens:
        return []
    if not rationales:
        return [0] * len(tokens)

    threshold = max(1, (len(rationales) + 1) // 2)
    counts = [0] * len(tokens)
    for rationale in rationales:
        for index, value in enumerate(rationale[: len(tokens)]):
            counts[index] += int(value)
    return [1 if count >= threshold else 0 for count in counts]


def _target_vector(targets: list[str], target_labels: list[str]) -> list[int]:
    """target multi-label supervision용 0/1 벡터를 만듭니다."""
    target_set = {target for target in targets if target and target != "None"}
    return [1 if label in target_set else 0 for label in target_labels]


def _has_v2_columns(frame: pd.DataFrame) -> bool:
    required_columns = {
        "post_tokens",
        "source",
        "agreement",
        "rationale_mask",
        "target_multilabel",
    }
    return required_columns.issubset(set(frame.columns))


# ╔══════════════════════════════════════════════════════════╗
# ║  데이터 전처리 — 원본 JSON을 깔끔한 DataFrame으로!       ║
# ╚══════════════════════════════════════════════════════════╝
def load_processed_dataframe(force_download: bool = False) -> pd.DataFrame:
    """HateXplain 원본 JSON을 다운로드하고 전처리해서 DataFrame으로 만들어요."""
    ensure_raw_hatexplain(force_download=force_download)

    # Step 1: 원본 JSON 파일을 읽어요
    with open(RAW_DATASET_PATH, "r", encoding="utf-8") as handle:
        raw_dataset = json.load(handle)

    # Step 2: 각 게시물을 하나씩 처리해요
    records = []
    for post_id, sample in raw_dataset.items():
        # 다수결 투표로 라벨을 정해요
        label = _majority_label(sample)
        if label is None:
            continue  # 투표 결과가 애매하면 건너뛰어요
        # 토큰들을 합쳐서 원본 텍스트를 복원하고, @멘션은 익명화해요
        post_tokens = [
            "<user>" if re.match(r"@\S+", token) else token
            for token in sample["post_tokens"]
        ]
        text = " ".join(post_tokens)
        text = re.sub(r"@\S+", "<user>", text)  # 개인정보 보호!
        targets = _collect_targets(sample)
        records.append(
            {
                "post_id": post_id,
                "source": _extract_source(post_id),
                "post_tokens": post_tokens,
                "text": text,
                "label": label,              # 숫자 라벨 (0, 1, 2)
                "label_name": ID2LABEL[label],  # 사람이 읽을 수 있는 라벨명
                "targets": targets,  # 대상 커뮤니티
                "agreement": _annotator_agreement(sample),
                "rationale_mask": _majority_rationale_mask(sample),
                "annotator_labels": [annotator["label"] for annotator in sample["annotators"]],
            }
        )

    # Step 3: DataFrame으로 변환하고 중복 텍스트 제거!
    frame = pd.DataFrame(records)
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    target_labels = sorted(
        {
            target
            for targets in frame["targets"]
            for target in targets
            if target and target != "None"
        }
    )
    frame["target_multilabel"] = frame["targets"].apply(lambda targets: _target_vector(targets, target_labels))
    frame.attrs["target_labels"] = target_labels
    return frame


# ╔══════════════════════════════════════════════════════════╗
# ║  데이터 준비 (prepare_data) — 실험의 첫 번째 단계!       ║
# ╚══════════════════════════════════════════════════════════╝
# 전체 데이터를 train/val/test로 나누고, 분포 통계도 저장해요.
# stratified split을 써서 각 분할에 라벨 비율이 골고루 들어가게 해요!
def prepare_data(config: ExperimentConfig | None = None, force_refresh: bool = False, force_download: bool = False) -> dict[str, pd.DataFrame]:
    """70/10/20 stratified split을 만들고 데이터셋 진단 결과를 저장해요."""
    config = config or get_config()
    save_config(config)

    # 이미 분할된 데이터가 있으면 바로 불러와요 (시간 절약!)
    if SPLITS_PICKLE_PATH.exists() and not force_refresh:
        cached_splits = pd.read_pickle(SPLITS_PICKLE_PATH)
        if all(_has_v2_columns(frame) for frame in cached_splits.values()):
            if not config.target_labels:
                target_labels = sorted(
                    {
                        target
                        for frame in cached_splits.values()
                        for targets in frame.get("targets", [])
                        for target in targets
                        if target and target != "None"
                    }
                )
                config.target_labels = target_labels
                save_config(config)
            return cached_splits
        print("[data] cached splits are v1 schema; rebuilding v2.1 data contract", flush=True)

    frame = load_processed_dataframe(force_download=force_download)
    target_labels = list(frame.attrs.get("target_labels", []))
    config.target_labels = target_labels
    save_config(config)

    # Step 1: 먼저 test 20%를 떼어놓고...
    train_val_df, test_df = train_test_split(
        frame,
        test_size=config.split_test,
        random_state=config.tuning_seed,
        stratify=frame["label"],  # 라벨 비율을 유지하면서 나눠요!
    )
    # Step 2: 나머지에서 val을 분리해요 (상대적 비율로 계산)
    val_relative_size = config.split_val / (config.split_train + config.split_val)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=config.tuning_seed,
        stratify=train_val_df["label"],
    )

    # Step 3: 깔끔하게 딕셔너리로 정리!
    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    # Step 4: 데이터셋 프로필(통계 요약)을 만들어요
    profile = {
        "total_samples": int(len(frame)),
        "splits": {name: int(len(df)) for name, df in splits.items()},
        "label_distribution": {
            name: {
                LABEL_NAMES[index]: int(count)
                for index, count in df["label"].value_counts().reindex(range(NUM_LABELS), fill_value=0).items()
            }
            for name, df in splits.items()
        },
        "split_ratio": {
            "train": config.split_train,
            "val": config.split_val,
            "test": config.split_test,
        },
        "target_labels": target_labels,
        "v2_columns": ["post_tokens", "source", "agreement", "rationale_mask", "target_multilabel"],
    }

    # Step 5: 결과물을 저장해요 — pickle, JSON, 시각화까지!
    pd.to_pickle(splits, SPLITS_PICKLE_PATH)
    save_json(profile, DATA_PROFILE_PATH)
    plot_split_distribution(splits, REPORT_DIR / "split_distribution.png")

    # Step 6: 보고서용 CSV와 마크다운 표도 만들어요
    profile_rows = []
    for split_name, distribution in profile["label_distribution"].items():
        row = {"split": split_name, "samples": profile["splits"][split_name]}
        for label_name, count in distribution.items():
            row[label_name] = count
        profile_rows.append(row)
    save_dataframe(pd.DataFrame(profile_rows), REPORT_DIR / "split_distribution.csv")
    save_text(
        "# Data Profile\n\n"
        + dataframe_to_markdown(pd.DataFrame(profile_rows)),
        REPORT_DIR / "data_profile.md",
    )
    return splits


# ── 분할 데이터 로드 ────────────────────────────
def load_splits() -> dict[str, pd.DataFrame]:
    """전처리된 train/val/test 분할을 불러와요. 없으면 자동으로 만들어줘요!"""
    if not SPLITS_PICKLE_PATH.exists():
        return prepare_data()
    splits = pd.read_pickle(SPLITS_PICKLE_PATH)
    if not all(_has_v2_columns(frame) for frame in splits.values()):
        return prepare_data(force_refresh=True)
    return splits



# ╔══════════════════════════════════════════════════════════╗
# ║  PyTorch Dataset 클래스들 — 데이터를 모델에 먹여줘요!     ║
# ╚══════════════════════════════════════════════════════════╝

# ── 트랜스포머 전용 데이터셋 ────────────────────
# BERT/RoBERTa에 넣을 수 있도록 텍스트를 토큰 ID로 변환해요.
# DataLoader가 이 클래스에서 배치 단위로 데이터를 꺼내갑니다!
class TransformerTextDataset(Dataset):
    """트랜스포머 전용 데이터셋. 텍스트 -> input_ids + attention_mask + labels"""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer,
        max_len: int,
        post_tokens: list[list[str]] | None = None,
        rationale_masks: list[list[int]] | None = None,
        target_vectors: list[list[int]] | None = None,
    ) -> None:
        self.encodings = {"input_ids": [], "attention_mask": []}
        self.rationale_masks: list[torch.Tensor] | None = [] if rationale_masks is not None else None

        for index, text in enumerate(texts):
            tokens = post_tokens[index] if post_tokens is not None else None
            if tokens:
                encoding = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_len,
                )
                word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else [None] * len(encoding["input_ids"])
                source_mask = rationale_masks[index] if rationale_masks is not None else []
                subword_mask = [
                    float(source_mask[word_id]) if word_id is not None and word_id < len(source_mask) else 0.0
                    for word_id in word_ids
                ]
            else:
                encoding = tokenizer(text, truncation=True, max_length=max_len)
                subword_mask = [0.0] * len(encoding["input_ids"])

            self.encodings["input_ids"].append(encoding["input_ids"])
            self.encodings["attention_mask"].append(encoding["attention_mask"])
            if self.rationale_masks is not None:
                self.rationale_masks.append(torch.tensor(subword_mask, dtype=torch.float32))

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.target_vectors = (
            torch.tensor(target_vectors, dtype=torch.float32)
            if target_vectors is not None and len(target_vectors) > 0 and len(target_vectors[0]) > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.encodings["input_ids"][index],       # 토큰 ID들
            "attention_mask": self.encodings["attention_mask"][index],  # 실제 토큰=1, 패딩=0
            "labels": self.labels[index],                           # 정답 라벨
        }
        if self.rationale_masks is not None:
            item["rationale_mask"] = self.rationale_masks[index]
        if self.target_vectors is not None:
            item["target_multilabel"] = self.target_vectors[index]
        return item


# ── 하이브리드 데이터셋 (트랜스포머 + VADER) ────
# 기본 데이터셋에 VADER 감성 점수 4개를 추가로 포함해요.
# 이 추가 피처가 모델의 판단력을 높여주는 비밀 무기예요!
class HybridTextDataset(TransformerTextDataset):
    """트랜스포머 + VADER 하이브리드용 데이터셋. VADER 4차원 감성 피처를 함께 전달해요."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        vader_features: np.ndarray,  # shape: (N, 4) — pos, neg, neu, compound
        tokenizer,
        max_len: int,
        post_tokens: list[list[str]] | None = None,
        rationale_masks: list[list[int]] | None = None,
        target_vectors: list[list[int]] | None = None,
    ) -> None:
        super().__init__(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_len=max_len,
            post_tokens=post_tokens,
            rationale_masks=rationale_masks,
            target_vectors=target_vectors,
        )
        self.vader = torch.tensor(vader_features, dtype=torch.float32)  # VADER 피처도 텐서로!

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        item["vader"] = self.vader[index]  # 여기가 기본 데이터셋과 다른 점!
        return item

