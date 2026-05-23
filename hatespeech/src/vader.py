"""
VADER 피처 추출 모듈 (src/vader.py)
====================================
규칙 기반 감성 분석기(VADER)로 각 텍스트의 감성 점수 4개(pos/neg/neu/compound)를
추출해요. ucam의 `src/roberta.py`(피처 추출기)에 대응하는 모듈이에요.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.data import load_splits
from src.path import VADER_PICKLE_PATH, SPLITS_PICKLE_PATH, REPORT_DIR
from src.utils import VADER_COLUMNS, save_dataframe, _artifact_current


def extract_vader_features(
    splits: dict[str, pd.DataFrame] | None = None,
    force_refresh: bool = False,
) -> dict[str, np.ndarray]:
    """모든 분할(train/val/test)에 대해 VADER 감성 점수를 계산해요."""
    splits = splits or load_splits()
    # 이미 계산해둔 게 있으면 재사용해요 (VADER는 결정적이라 결과가 항상 같아요)
    if VADER_PICKLE_PATH.exists() and not force_refresh:
        if _artifact_current(VADER_PICKLE_PATH, [SPLITS_PICKLE_PATH]):
            cached_features = pd.read_pickle(VADER_PICKLE_PATH)
            if all(split_name in cached_features and len(cached_features[split_name]) == len(frame) for split_name, frame in splits.items()):
                return cached_features
        print("[vader] cached features are stale for current data splits; rebuilding", flush=True)

    analyzer = SentimentIntensityAnalyzer()
    features: dict[str, np.ndarray] = {}

    for split_name, frame in splits.items():
        rows = []
        for text in frame["text"].tolist():
            # 각 텍스트의 감성 점수 4개를 추출해요
            scores = analyzer.polarity_scores(text)
            rows.append([scores[column] for column in VADER_COLUMNS])
        features[split_name] = np.asarray(rows, dtype=np.float32)

        # 분석 결과를 CSV로도 저장해요 (보고서에 활용 가능!)
        save_dataframe(
            pd.DataFrame(features[split_name], columns=VADER_COLUMNS),
            REPORT_DIR / f"vader_{split_name}.csv",
        )

    pd.to_pickle(features, VADER_PICKLE_PATH)
    return features

