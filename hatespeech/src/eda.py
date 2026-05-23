"""
# ╔══════════════════════════════════════════════════════════╗
# ║  experiment_eda.py — 탐색적 데이터 분석 (EDA) 모듈       ║
# ╚══════════════════════════════════════════════════════════╝
#
# 안녕하세요! 이 파일은 HateXplain 데이터셋을 깊이 들여다보는 EDA 모듈이에요.
#
# 모델을 학습하기 전에 데이터의 특성을 충분히 이해하는 것이 정말 중요하거든요!
# 마치 여행을 떠나기 전에 지도를 꼼꼼히 살펴보는 것처럼,
# EDA를 통해 데이터의 지형을 먼저 파악하면 실험이 훨씬 수월해져요 :)
#
# 이 모듈이 만들어주는 분석 결과물:
#   1. 텍스트 길이 분포 — 클래스별로 텍스트 길이가 어떻게 다른지 살펴봐요
#   2. VADER 감성 점수 분포 — 혐오/공격/일반 텍스트의 감성 차이를 확인해요
#   3. 타겟 커뮤니티 분석 — 어떤 집단이 주로 혐오의 대상이 되는지 알아봐요
#   4. 어휘 겹침 분석 — 클래스 간 단어가 얼마나 겹치는지 Jaccard로 측정해요
#   5. 종합 요약 보고서 — 모든 통계를 JSON과 마크다운으로 정리해요
#
# 모든 결과물은 outputs/reports/eda/ 폴더에 저장된답니다!
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  임포트(import) — 분석에 필요한 도구들을 불러와요          ║
# ╚══════════════════════════════════════════════════════════╝

from __future__ import annotations

import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── 우리 프로젝트의 유틸리티 & 핵심 모듈 ──────────
from src.utils import (
    LABEL_NAMES,
    REPORT_DIR,
    VADER_COLUMNS,
    ensure_dir,
    save_dataframe,
    save_json,
    save_text,
)
from src.config import ExperimentConfig, get_config
from src.vader import extract_vader_features
from src.data import prepare_data
from src.path import RAW_DATASET_PATH

# ╔══════════════════════════════════════════════════════════╗
# ║  EDA 출력 디렉토리 — 분석 결과가 모이는 보금자리예요       ║
# ╚══════════════════════════════════════════════════════════╝
EDA_DIR = REPORT_DIR / "eda"


# ╔══════════════════════════════════════════════════════════╗
# ║  1. 텍스트 길이 분포 분석                                  ║
# ║     — 클래스별로 텍스트 길이가 어떻게 다를까요?             ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_text_length(
    all_df: pd.DataFrame,
    output_dir: Path,
    max_len: int = 128,
) -> dict[str, Any]:
    """
    텍스트 길이 분포를 분석하고 시각화해요.

    두 가지 관점에서 길이를 측정합니다:
      - 단어 수 (word count): 공백으로 나눈 토큰 개수
      - BERT 토큰 수 (token count): BERT 토크나이저가 만드는 서브워드 토큰 개수

    max_len=128을 넘는 텍스트가 얼마나 되는지도 확인해봐요.
    만약 너무 많으면 max_len을 늘려야 할 수도 있거든요!
    """
    # ── 단어 수 계산 ──────────────────────────────
    # 간단하게 공백 기준으로 분리해서 세요. 직관적이죠?
    all_df = all_df.copy()
    all_df["word_count"] = all_df["text"].str.split().str.len()

    # ── BERT 토큰 수 계산 ─────────────────────────
    # BERT 토크나이저는 단어를 서브워드로 쪼개요.
    # 예: "unbelievable" → ["un", "##believ", "##able"] (3개 토큰)
    # 그래서 단어 수보다 토큰 수가 더 많을 수 있어요!
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 배치로 토크나이즈하면 훨씬 빨라요 (하나씩 하면 느려요!)
    encoded = tokenizer(
        all_df["text"].tolist(),
        truncation=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    all_df["token_count"] = [len(ids) for ids in encoded["input_ids"]]

    # ── 클래스별 통계 계산 ─────────────────────────
    # 각 클래스의 평균, 중앙값, 최대 길이 등을 정리해요
    stats_rows = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        subset = all_df[all_df["label"] == label_idx]
        n_exceed = int((subset["token_count"] > max_len).sum())
        pct_exceed = round(n_exceed / max(len(subset), 1) * 100, 2)

        stats_rows.append({
            "class": label_name,
            "n_samples": int(len(subset)),
            "word_mean": round(float(subset["word_count"].mean()), 2),
            "word_median": float(subset["word_count"].median()),
            "word_max": int(subset["word_count"].max()),
            "token_mean": round(float(subset["token_count"].mean()), 2),
            "token_median": float(subset["token_count"].median()),
            "token_max": int(subset["token_count"].max()),
            "exceed_max_len_count": n_exceed,
            "exceed_max_len_pct": pct_exceed,
        })

    # 전체 데이터에 대한 통계도 추가해요
    n_total_exceed = int((all_df["token_count"] > max_len).sum())
    pct_total_exceed = round(n_total_exceed / max(len(all_df), 1) * 100, 2)
    stats_rows.append({
        "class": "ALL",
        "n_samples": int(len(all_df)),
        "word_mean": round(float(all_df["word_count"].mean()), 2),
        "word_median": float(all_df["word_count"].median()),
        "word_max": int(all_df["word_count"].max()),
        "token_mean": round(float(all_df["token_count"].mean()), 2),
        "token_median": float(all_df["token_count"].median()),
        "token_max": int(all_df["token_count"].max()),
        "exceed_max_len_count": n_total_exceed,
        "exceed_max_len_pct": pct_total_exceed,
    })

    stats_df = pd.DataFrame(stats_rows)
    save_dataframe(stats_df, output_dir / "text_length_stats.csv")

    # ── 시각화: 단어 수 & 토큰 수 히스토그램 ──────
    # 위아래 2행으로 배치해서 한 눈에 비교할 수 있게 해요!
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 위쪽: 단어 수 분포
    for label_idx, label_name in enumerate(LABEL_NAMES):
        subset = all_df[all_df["label"] == label_idx]["word_count"]
        axes[0].hist(subset, bins=50, alpha=0.6, label=label_name)
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Word Count Distribution by Class")
    axes[0].legend()

    # 아래쪽: BERT 토큰 수 분포
    for label_idx, label_name in enumerate(LABEL_NAMES):
        subset = all_df[all_df["label"] == label_idx]["token_count"]
        axes[1].hist(subset, bins=50, alpha=0.6, label=label_name)
    # max_len 기준선을 빨간 점선으로 표시해요 — 이 선을 넘으면 잘려나가요!
    axes[1].axvline(x=max_len, color="red", linestyle="--", linewidth=1.5, label=f"max_len={max_len}")
    axes[1].set_xlabel("BERT Token Count")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("BERT Token Count Distribution by Class")
    axes[1].legend()

    fig.suptitle("Text Length Distribution (텍스트 길이 분포)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "text_length_distribution.png", dpi=160)
    plt.close(fig)

    print(f"  [text length] max_len={max_len} 초과 비율: {pct_total_exceed}% ({n_total_exceed}/{len(all_df)})")
    return {
        "text_length_stats": stats_rows,
        "exceed_max_len_total_pct": pct_total_exceed,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  2. VADER 감성 점수 분포 분석 (클래스별)                   ║
# ║     — 혐오 vs 공격적 vs 일반 텍스트의 감성이 다를까요?     ║
# ║     — 이 차이가 클수록 VADER가 도움이 된다는 증거!         ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_vader_by_class(
    all_df: pd.DataFrame,
    vader_features: dict[str, np.ndarray],
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Any]:
    """
    클래스별 VADER 점수 분포를 박스플롯으로 시각화해요.

    이 분석이 왜 중요한지 설명할게요!
    만약 hatespeech와 offensive의 VADER 점수 분포가 확연히 다르다면,
    VADER 피처를 추가했을 때 모델이 이 둘을 더 잘 구분할 수 있을 거예요.
    즉, VADER 활용의 이론적 근거가 되는 핵심 분석이랍니다!
    """
    # ── 전체 데이터에 VADER 점수를 붙여요 ─────────
    # splits별로 나뉜 VADER 피처를 다시 합치는 작업이에요
    vader_rows = []
    for split_name in ["train", "val", "test"]:
        split_df = splits[split_name]
        split_vader = vader_features[split_name]
        for i in range(len(split_df)):
            row = {"label": int(split_df.iloc[i]["label"])}
            for j, col in enumerate(VADER_COLUMNS):
                row[col] = float(split_vader[i, j])
            vader_rows.append(row)

    vader_df = pd.DataFrame(vader_rows)
    vader_df["class"] = vader_df["label"].map(lambda x: LABEL_NAMES[x])

    # ── 클래스별 기술 통계 계산 ────────────────────
    # 평균, 표준편차, 중앙값 등을 깔끔하게 정리해요
    stats_rows = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        subset = vader_df[vader_df["label"] == label_idx]
        row = {"class": label_name, "n_samples": int(len(subset))}
        for col in VADER_COLUMNS:
            row[f"{col}_mean"] = round(float(subset[col].mean()), 4)
            row[f"{col}_std"] = round(float(subset[col].std()), 4)
            row[f"{col}_median"] = round(float(subset[col].median()), 4)
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    save_dataframe(stats_df, output_dir / "vader_by_class_stats.csv")

    # ── 시각화: VADER 4차원 × 클래스별 박스플롯 ───
    # 2×2 그리드로 pos, neg, neu, compound 한 번에 보여줘요!
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, col in enumerate(VADER_COLUMNS):
        sns.boxplot(
            data=vader_df,
            x="class",
            y=col,
            order=LABEL_NAMES,
            ax=axes[idx],
            palette="Set2",
        )
        axes[idx].set_title(f"VADER {col} by Class")
        axes[idx].set_xlabel("Class")
        axes[idx].set_ylabel(col)

    fig.suptitle("VADER Score Distribution by Class (클래스별 VADER 점수 분포)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "vader_by_class.png", dpi=160)
    plt.close(fig)

    # ── 핵심 발견 요약 ────────────────────────────
    # hate vs offensive의 compound 차이를 계산해서 보고서에 넣을 수 있게 해요
    hate_compound = vader_df[vader_df["label"] == 0]["compound"].mean()
    offensive_compound = vader_df[vader_df["label"] == 1]["compound"].mean()
    normal_compound = vader_df[vader_df["label"] == 2]["compound"].mean()

    print(f"  [VADER] compound 평균 — hate: {hate_compound:.4f}, "
          f"offensive: {offensive_compound:.4f}, normal: {normal_compound:.4f}")

    return {
        "vader_by_class_stats": stats_rows,
        "compound_mean_hate": round(float(hate_compound), 4),
        "compound_mean_offensive": round(float(offensive_compound), 4),
        "compound_mean_normal": round(float(normal_compound), 4),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  3. 타겟 커뮤니티 분석                                     ║
# ║     — 어떤 집단이 주로 혐오의 대상이 되고 있을까요?        ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_targets(
    all_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    타겟 커뮤니티의 빈도와 라벨과의 교차 관계를 분석해요.

    HateXplain 데이터셋에는 각 텍스트가 어떤 집단을 대상으로 하는지
    어노테이터가 표시해 두었어요 (예: African, Islam, Women 등).
    이 정보를 분석하면 혐오표현의 패턴을 더 깊이 이해할 수 있답니다!
    """
    # ── targets 컬럼이 리스트이므로 풀어서(explode) 세어야 해요 ──
    # 하나의 텍스트가 여러 커뮤니티를 대상으로 할 수 있거든요!
    target_rows = []
    for _, row in all_df.iterrows():
        targets = row.get("targets", ["None"])
        if not isinstance(targets, list):
            targets = ["None"]
        for t in targets:
            target_rows.append({
                "target": t,
                "label": int(row["label"]),
                "label_name": LABEL_NAMES[int(row["label"])],
            })

    target_df = pd.DataFrame(target_rows)

    # ── 상위 N개 타겟 커뮤니티 추출 ──────────────
    target_counts = target_df["target"].value_counts()
    top_targets = target_counts.head(top_n).index.tolist()

    # ── 교차표(crosstab) 만들기 ───────────────────
    # 행: 타겟 커뮤니티, 열: 라벨 → 어떤 커뮤니티가 어떤 라벨과 연관되는지!
    crosstab = pd.crosstab(
        target_df["target"],
        target_df["label_name"],
        margins=True,
    )
    # 상위 타겟 + 전체(All)만 남겨요
    crosstab_top = crosstab.loc[
        crosstab.index.isin(top_targets + ["All"])
    ].sort_values("All", ascending=False)

    save_dataframe(crosstab_top.reset_index(), output_dir / "target_crosstab.csv")

    # ── 시각화 1: 상위 타겟 커뮤니티 빈도 막대그래프 ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # 위쪽: 단순 빈도 막대그래프
    top_counts = target_counts.head(top_n)
    axes[0].barh(
        range(len(top_counts)),
        top_counts.values,
        color="#4c72b0",
        edgecolor="black",
    )
    axes[0].set_yticks(range(len(top_counts)))
    axes[0].set_yticklabels(top_counts.index)
    axes[0].invert_yaxis()  # 가장 많은 게 위로 오게!
    axes[0].set_xlabel("Frequency")
    axes[0].set_title(f"Top-{top_n} Target Communities")

    # 아래쪽: 타겟 × 라벨 히트맵
    # "All" 행과 열은 빼고 히트맵을 그려요
    heatmap_data = crosstab_top.drop(index="All", errors="ignore")
    for col in ["All"]:
        if col in heatmap_data.columns:
            heatmap_data = heatmap_data.drop(columns=col)
    # 라벨 순서를 맞춰요
    heatmap_cols = [c for c in LABEL_NAMES if c in heatmap_data.columns]
    if heatmap_cols:
        sns.heatmap(
            heatmap_data[heatmap_cols],
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=axes[1],
        )
    axes[1].set_title("Target Community × Label (Heatmap)")
    axes[1].set_ylabel("Target Community")
    axes[1].set_xlabel("Label")

    fig.suptitle("Target Community Analysis (타겟 커뮤니티 분석)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)

    top_target_list = [
        {"target": t, "count": int(target_counts[t])}
        for t in top_targets
    ]
    print(f"  [targets] 상위 3개: {', '.join(top_targets[:3])}")

    return {
        "top_targets": top_target_list,
        "unique_targets": int(target_counts.shape[0]),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  4. 클래스 혼동 분석 — 어휘 겹침 (Jaccard Similarity)     ║
# ║     — hate와 offensive의 단어가 얼마나 겹칠까요?          ║
# ║     — 겹침이 클수록 분류가 어렵다는 뜻이에요!              ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_vocabulary_overlap(
    all_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = 500,
) -> dict[str, Any]:
    """
    클래스별 상위 단어를 뽑아서 Jaccard 유사도를 계산해요.

    Jaccard 유사도 = |A ∩ B| / |A ∪ B|
    1에 가까울수록 두 클래스의 어휘가 비슷하다는 뜻이에요.

    hate와 offensive의 Jaccard가 높으면?
    → 단어만 봐서는 구분하기 어렵다는 의미!
    → 그래서 VADER 같은 추가 피처가 필요하다는 근거가 되죠!
    """
    # ── 클래스별 상위 단어 추출 ────────────────────
    # 간단한 공백 분리 + 소문자 변환으로 토큰화해요
    class_vocabs: dict[str, set[str]] = {}
    for label_idx, label_name in enumerate(LABEL_NAMES):
        subset = all_df[all_df["label"] == label_idx]
        word_counter: Counter = Counter()
        for text in subset["text"].tolist():
            # 소문자로 통일하고, 알파벳+숫자만 남겨요
            tokens = text.lower().split()
            word_counter.update(tokens)
        # 빈도 상위 top_k개만 남겨요
        top_words = {word for word, _ in word_counter.most_common(top_k)}
        class_vocabs[label_name] = top_words

    # ── Jaccard 유사도 계산 ────────────────────────
    # 모든 클래스 쌍에 대해 계산해요
    overlap_rows = []
    for class_a, class_b in combinations(LABEL_NAMES, 2):
        vocab_a = class_vocabs[class_a]
        vocab_b = class_vocabs[class_b]
        intersection = len(vocab_a & vocab_b)
        union = len(vocab_a | vocab_b)
        jaccard = round(intersection / max(union, 1), 4)
        overlap_rows.append({
            "class_a": class_a,
            "class_b": class_b,
            f"top_{top_k}_intersection": intersection,
            f"top_{top_k}_union": union,
            "jaccard_similarity": jaccard,
        })

    overlap_df = pd.DataFrame(overlap_rows)
    save_dataframe(overlap_df, output_dir / "vocabulary_overlap.csv")

    # hate-offensive 겹침이 특히 중요해요!
    hate_off = [r for r in overlap_rows if set([r["class_a"], r["class_b"]]) == {"hatespeech", "offensive"}]
    if hate_off:
        print(f"  [vocabulary] hate↔offensive Jaccard (top-{top_k}): {hate_off[0]['jaccard_similarity']}")

    return {
        "vocabulary_overlap": overlap_rows,
        "top_k": top_k,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  5. 클래스 분포 분석                                        ║
# ║     — 데이터셋의 클래스 불균형을 시각적으로 확인해요         ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_class_distribution(
    all_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    클래스별 샘플 수와 비율을 분석하고, 파이 차트 + 바 차트로 시각화해요.
    불균형이 심하면 학습 전략(balanced weight 등)을 조정해야 하니까요.
    """
    class_counts = all_df["label"].value_counts().sort_index()
    total = len(all_df)

    dist_rows = []
    for label_idx, label_name in enumerate(LABEL_NAMES):
        count = int(class_counts.get(label_idx, 0))
        ratio = round(count / total * 100, 2)
        dist_rows.append({
            "class": label_name,
            "count": count,
            "ratio_pct": ratio,
        })

    dist_df = pd.DataFrame(dist_rows)
    save_dataframe(dist_df, output_dir / "class_distribution.csv")

    # 시각화: 바 차트 + 비율 표시
    colors = ["#ff6b7a", "#ffb347", "#00e5b0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 바 차트
    ax = axes[0]
    bars = ax.bar(
        [r["class"] for r in dist_rows],
        [r["count"] for r in dist_rows],
        color=colors,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, row in zip(bars, dist_rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{row['count']:,}\n({row['ratio_pct']}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sample Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 파이 차트
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        [r["count"] for r in dist_rows],
        labels=[r["class"] for r in dist_rows],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Class Ratio", fontsize=14, fontweight="bold")

    # 불균형 비율 계산
    max_count = max(r["count"] for r in dist_rows)
    min_count = min(r["count"] for r in dist_rows)
    imbalance_ratio = round(max_count / max(min_count, 1), 2)

    fig.suptitle(
        f"HateXplain Dataset: {total:,} samples (imbalance ratio: {imbalance_ratio}:1)",
        fontsize=12, color="gray", y=0.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [class dist] total={total}, imbalance ratio={imbalance_ratio}:1")

    return {
        "class_distribution": dist_rows,
        "imbalance_ratio": imbalance_ratio,
        "total_samples": total,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  6. N-gram 빈도 분석                                       ║
# ║     — 클래스별 자주 등장하는 bigram/trigram을 비교해요      ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_ngrams(
    all_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = 15,
) -> dict[str, Any]:
    """
    클래스별로 가장 빈번한 bigram과 trigram을 추출해요.
    hate에서만 자주 나타나는 n-gram은 혐오 표현의 패턴을 보여주고,
    모든 클래스에서 공통인 n-gram은 분류를 어렵게 만드는 노이즈를 의미해요.
    """
    import re

    def _tokenize(text: str) -> list[str]:
        """소문자 변환 + 알파벳/숫자만 남기는 간단한 토크나이저"""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _get_ngrams(tokens: list[str], n: int) -> list[str]:
        """토큰 리스트에서 n-gram 추출"""
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    ngram_data = {}
    for n, name in [(2, "bigram"), (3, "trigram")]:
        class_ngrams = {}
        for label_idx, label_name in enumerate(LABEL_NAMES):
            subset = all_df[all_df["label"] == label_idx]
            ngram_counter: Counter = Counter()
            for text in subset["text"].tolist():
                tokens = _tokenize(text)
                ngram_counter.update(_get_ngrams(tokens, n))
            class_ngrams[label_name] = ngram_counter.most_common(top_k)

        ngram_data[name] = class_ngrams

        # CSV 저장
        rows = []
        for label_name, ngrams in class_ngrams.items():
            for rank, (ngram, count) in enumerate(ngrams, 1):
                rows.append({
                    "class": label_name,
                    "rank": rank,
                    name: ngram,
                    "count": count,
                })
        save_dataframe(pd.DataFrame(rows), output_dir / f"top_{name}s.csv")

    # 시각화: 클래스별 bigram Top-10 가로 바 차트
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = {"hatespeech": "#ff6b7a", "offensive": "#ffb347", "normal": "#00e5b0"}

    for ax, label_name in zip(axes, LABEL_NAMES):
        bigrams = ngram_data["bigram"][label_name][:10]
        if not bigrams:
            continue
        names = [b[0] for b in bigrams][::-1]
        counts = [b[1] for b in bigrams][::-1]
        ax.barh(names, counts, color=colors[label_name], edgecolor="white")
        ax.set_title(f"{label_name} Top-10 Bigrams", fontsize=13, fontweight="bold")
        ax.set_xlabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(counts):
            ax.text(v + max(counts) * 0.01, i, str(v), va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "ngram_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 클래스 간 고유 bigram 분석 (hate에만 있는 bigram)
    hate_set = {b[0] for b in ngram_data["bigram"]["hatespeech"]}
    off_set = {b[0] for b in ngram_data["bigram"]["offensive"]}
    norm_set = {b[0] for b in ngram_data["bigram"]["normal"]}
    hate_unique = hate_set - off_set - norm_set
    shared_all = hate_set & off_set & norm_set

    print(f"  [ngram] hate-unique bigrams: {len(hate_unique)}, shared by all: {len(shared_all)}")

    return {
        "ngram_top_bigrams": {
            label: [{"ngram": ng, "count": c} for ng, c in grams[:10]]
            for label, grams in ngram_data["bigram"].items()
        },
        "ngram_top_trigrams": {
            label: [{"ngram": ng, "count": c} for ng, c in grams[:10]]
            for label, grams in ngram_data["trigram"].items()
        },
        "ngram_hate_unique_bigrams": sorted(list(hate_unique)),
        "ngram_shared_all_bigrams": sorted(list(shared_all)),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  7. 워드클라우드 — 클래스별 핵심 단어를 한눈에 보여줘요     ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_wordcloud(
    all_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    클래스별 TF-IDF 기반 워드클라우드를 생성해요.
    단순 빈도가 아니라 TF-IDF를 쓰면 해당 클래스에서 "특징적인" 단어가 강조돼요.
    모든 클래스에서 공통인 불용어('the', 'is' 등)는 자연스럽게 약해지거든요.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  [wordcloud] wordcloud 패키지 없음 -- pip install wordcloud")
        return {"wordcloud_generated": False}

    from sklearn.feature_extraction.text import TfidfVectorizer

    colors_map = {
        "hatespeech": "Reds",
        "offensive": "Oranges",
        "normal": "Greens",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    wc_stats = {}

    for ax, (label_idx, label_name) in zip(axes, enumerate(LABEL_NAMES)):
        subset = all_df[all_df["label"] == label_idx]["text"].tolist()

        # TF-IDF로 해당 클래스 텍스트의 중요 단어 추출
        tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 1))
        tfidf_matrix = tfidf.fit_transform(subset)
        mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
        feature_names = tfidf.get_feature_names_out()

        # TF-IDF 점수를 단어별 가중치로 변환
        word_weights = {word: float(score) for word, score in zip(feature_names, mean_tfidf)}
        top_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:20]
        wc_stats[label_name] = [{"word": w, "tfidf": round(s, 4)} for w, s in top_words]

        wc = WordCloud(
            width=600, height=400,
            background_color="black",
            colormap=colors_map[label_name],
            max_words=100,
            prefer_horizontal=0.7,
        )
        wc.generate_from_frequencies(word_weights)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{label_name}", fontsize=14, fontweight="bold",
                     color={"hatespeech": "#ff6b7a", "offensive": "#ffb347", "normal": "#00e5b0"}[label_name])
        ax.axis("off")

    plt.suptitle("TF-IDF Word Clouds by Class", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "wordcloud.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [wordcloud] 생성 완료")

    return {
        "wordcloud_generated": True,
        "wordcloud_top_words": wc_stats,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  8. Human Rationale 분포 분석                              ║
# ║     — 어떤 클래스에 인간 주석 근거가 있는지 파악해요        ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_rationale_distribution(
    all_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    HateXplain dataset.json의 인간 rationale 분포를 분석해요.
    어떤 클래스에 rationale이 집중되어 있는지,
    rationale 길이(주석된 토큰 수)는 어떤 분포인지 확인해요.
    """
    if not RAW_DATASET_PATH.exists():
        print("  [rationale] dataset.json 없음 -- 건너뜀")
        return {"rationale_analysis": False}

    import math

    with open(RAW_DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # post_id별 majority vote rationale 구성
    rationale_stats = {"hatespeech": [], "offensive": [], "normal": []}
    total_with_rationale = 0
    total_without_rationale = 0

    for post_id, entry in raw_data.items():
        annotators = entry.get("annotators", [])
        rationales = entry.get("rationales", [])

        # majority vote로 라벨 결정
        labels = [a.get("label", "") for a in annotators]
        label_counter = Counter(labels)
        majority_label = label_counter.most_common(1)[0][0]

        if not rationales:
            total_without_rationale += 1
            continue

        # majority vote rationale
        n_annotators = len(rationales)
        threshold = math.ceil(n_annotators / 2)
        tokens = entry.get("post_tokens", [])
        if not tokens:
            total_without_rationale += 1
            continue

        rationale_sum = [0] * len(tokens)
        for r in rationales:
            for i, val in enumerate(r):
                if i < len(rationale_sum):
                    rationale_sum[i] += val

        rationale_tokens = [tokens[i] for i in range(len(tokens)) if rationale_sum[i] >= threshold]

        if rationale_tokens:
            total_with_rationale += 1
            if majority_label in rationale_stats:
                rationale_stats[majority_label].append(len(rationale_tokens))
        else:
            total_without_rationale += 1

    # 통계 계산
    dist_rows = []
    for label_name in LABEL_NAMES:
        lengths = rationale_stats[label_name]
        if lengths:
            dist_rows.append({
                "class": label_name,
                "samples_with_rationale": len(lengths),
                "mean_rationale_tokens": round(np.mean(lengths), 2),
                "median_rationale_tokens": round(float(np.median(lengths)), 1),
                "max_rationale_tokens": int(np.max(lengths)),
            })
        else:
            dist_rows.append({
                "class": label_name,
                "samples_with_rationale": 0,
                "mean_rationale_tokens": 0,
                "median_rationale_tokens": 0,
                "max_rationale_tokens": 0,
            })

    save_dataframe(pd.DataFrame(dist_rows), output_dir / "rationale_distribution.csv")

    # 시각화: rationale 유무 + 길이 분포
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#ff6b7a", "#ffb347", "#00e5b0"]

    # 왼쪽: 클래스별 rationale 보유 샘플 수
    ax = axes[0]
    counts = [r["samples_with_rationale"] for r in dist_rows]
    bars = ax.bar([r["class"] for r in dist_rows], counts, color=colors, edgecolor="white")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{c:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Samples with Human Rationale", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 오른쪽: rationale 토큰 수 분포 (박스플롯)
    ax2 = axes[1]
    box_data = [rationale_stats[ln] for ln in LABEL_NAMES if rationale_stats[ln]]
    box_labels = [ln for ln in LABEL_NAMES if rationale_stats[ln]]
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_title("Rationale Length Distribution (tokens)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Number of rationale tokens")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "rationale_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [rationale] with={total_with_rationale}, without={total_without_rationale}")

    return {
        "rationale_analysis": True,
        "rationale_distribution": dist_rows,
        "rationale_total_with": total_with_rationale,
        "rationale_total_without": total_without_rationale,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  9. 클래스 간 혼동 분석 — VADER 분리도                      ║
# ║     — VADER 점수만으로 클래스 분리가 가능한지 확인해요      ║
# ╚══════════════════════════════════════════════════════════╝
def _analyze_vader_separability(
    all_df: pd.DataFrame,
    vader_features: dict,
    splits: dict,
    output_dir: Path,
) -> dict[str, Any]:
    """
    VADER compound 점수의 클래스별 분포를 겹쳐 그려서,
    감성 점수만으로 혐오/공격/일반을 분리할 수 있는지 시각적으로 확인해요.
    KDE (Kernel Density Estimation) 오버레이로 분포 겹침을 보여줘요.
    """
    # 전체 데이터에 VADER compound 추가
    all_vader = pd.concat([
        splits["train"], splits["val"], splits["test"]
    ], ignore_index=True)

    # VADER features를 합쳐요
    vader_all = np.concatenate([
        vader_features["train"], vader_features["val"], vader_features["test"]
    ], axis=0)

    all_vader = all_vader.copy()
    all_vader["vader_compound"] = vader_all[:, 3]  # compound는 4번째 열

    colors = {"hatespeech": "#ff6b7a", "offensive": "#ffb347", "normal": "#00e5b0"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KDE 분포 겹침
    ax = axes[0]
    for label_idx, label_name in enumerate(LABEL_NAMES):
        subset = all_vader[all_vader["label"] == label_idx]["vader_compound"]
        subset.plot.kde(ax=ax, label=label_name, color=colors[label_name], linewidth=2)
    ax.set_title("VADER Compound Distribution (KDE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("VADER Compound Score")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 누적 분포 (CDF)
    ax2 = axes[1]
    for label_idx, label_name in enumerate(LABEL_NAMES):
        subset = all_vader[all_vader["label"] == label_idx]["vader_compound"].sort_values()
        cdf = np.arange(1, len(subset) + 1) / len(subset)
        ax2.plot(subset.values, cdf, label=label_name, color=colors[label_name], linewidth=2)
    ax2.set_title("VADER Compound CDF", fontsize=13, fontweight="bold")
    ax2.set_xlabel("VADER Compound Score")
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "vader_separability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # KS 통계량 계산 (hate vs offensive, hate vs normal)
    from scipy import stats as sp_stats
    ks_results = []
    for ca, cb in combinations(LABEL_NAMES, 2):
        ca_vals = all_vader[all_vader["label"] == LABEL_NAMES.index(ca)]["vader_compound"]
        cb_vals = all_vader[all_vader["label"] == LABEL_NAMES.index(cb)]["vader_compound"]
        ks_stat, ks_pval = sp_stats.ks_2samp(ca_vals, cb_vals)
        ks_results.append({
            "class_a": ca,
            "class_b": cb,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": float(f"{ks_pval:.2e}"),
        })
    save_dataframe(pd.DataFrame(ks_results), output_dir / "vader_ks_test.csv")

    print("  [vader sep] KS tests: " + ", ".join(
        f"{r['class_a']}↔{r['class_b']}={r['ks_statistic']}" for r in ks_results
    ))

    return {
        "vader_separability": ks_results,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  10. 종합 요약 보고서 생성                                  ║
# ║      — 모든 분석 결과를 JSON과 마크다운으로 정리해요         ║
# ╚══════════════════════════════════════════════════════════╝
def _generate_summary(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    EDA 결과를 종합 JSON과 마크다운 보고서로 정리해서 저장해요.
    이 파일들이 있으면 보고서 작성할 때 아주 편하답니다!
    """
    # ── JSON 요약 저장 ─────────────────────────────
    save_json(results, output_dir / "eda_summary.json")

    # ── 마크다운 보고서 생성 ───────────────────────
    md_lines = [
        "# EDA Report (탐색적 데이터 분석 보고서)",
        "",
        "자동 생성된 EDA 보고서입니다.",
        "",
        "---",
        "",
        "## 1. 텍스트 길이 분포 (Text Length Distribution)",
        "",
    ]

    # 텍스트 길이 통계
    if "text_length_stats" in results:
        md_lines.append("| Class | N | Word Mean | Word Median | Token Mean | Token Median | Exceed max_len (%) |")
        md_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in results["text_length_stats"]:
            md_lines.append(
                f"| {row['class']} | {row['n_samples']} | {row['word_mean']} | "
                f"{row['word_median']} | {row['token_mean']} | {row['token_median']} | "
                f"{row['exceed_max_len_pct']}% |"
            )
        md_lines.append("")
        md_lines.append("![Text Length Distribution](text_length_distribution.png)")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 2. VADER 감성 점수 분포 (VADER Score Distribution by Class)")
    md_lines.append("")

    if "vader_by_class_stats" in results:
        md_lines.append("| Class | N | compound_mean | compound_std | neg_mean | neg_std | pos_mean | pos_std |")
        md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in results["vader_by_class_stats"]:
            md_lines.append(
                f"| {row['class']} | {row['n_samples']} | "
                f"{row['compound_mean']} | {row['compound_std']} | "
                f"{row['neg_mean']} | {row['neg_std']} | "
                f"{row['pos_mean']} | {row['pos_std']} |"
            )
        md_lines.append("")
        md_lines.append(f"> **핵심 발견**: hate compound={results.get('compound_mean_hate', 'N/A')}, "
                        f"offensive compound={results.get('compound_mean_offensive', 'N/A')}, "
                        f"normal compound={results.get('compound_mean_normal', 'N/A')}")
        md_lines.append("")
        md_lines.append("![VADER by Class](vader_by_class.png)")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 3. 타겟 커뮤니티 분석 (Target Community Analysis)")
    md_lines.append("")

    if "top_targets" in results:
        md_lines.append(f"고유 타겟 커뮤니티 수: **{results.get('unique_targets', 'N/A')}**")
        md_lines.append("")
        md_lines.append("| Rank | Target | Count |")
        md_lines.append("| --- | --- | --- |")
        for i, t in enumerate(results["top_targets"], 1):
            md_lines.append(f"| {i} | {t['target']} | {t['count']} |")
        md_lines.append("")
        md_lines.append("![Target Distribution](target_distribution.png)")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 4. 클래스 혼동 분석 — 어휘 겹침 (Vocabulary Overlap)")
    md_lines.append("")

    if "vocabulary_overlap" in results:
        top_k = results.get("top_k", 500)
        md_lines.append(f"상위 {top_k}개 단어 기준 Jaccard Similarity:")
        md_lines.append("")
        md_lines.append("| Class A | Class B | Jaccard |")
        md_lines.append("| --- | --- | --- |")
        for row in results["vocabulary_overlap"]:
            md_lines.append(f"| {row['class_a']} | {row['class_b']} | {row['jaccard_similarity']} |")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("*이 보고서는 `experiment_eda.py`에 의해 자동 생성되었습니다.*")

    save_text("\n".join(md_lines), output_dir / "eda_report.md")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║   run_eda() — EDA 전체를 한 번에 실행하는 메인 함수예요!              ║
# ║                                                                      ║
# ║   데이터를 불러오고, 위에서 정의한 분석 함수들을 차례로 호출해요.      ║
# ║   결과물은 모두 outputs/reports/eda/ 아래에 저장된답니다!             ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def run_eda(
    config: ExperimentConfig | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    탐색적 데이터 분석(EDA) 전체를 실행하고 결과를 반환해요.

    Parameters
    ----------
    config : ExperimentConfig | None
        실험 설정. None이면 기본 설정을 불러와요.
    force_refresh : bool
        True면 기존 EDA 결과를 무시하고 처음부터 다시 만들어요.

    Returns
    -------
    dict[str, Any]
        모든 EDA 분석 결과를 담은 딕셔너리.
        eda_summary.json에도 동일한 내용이 저장돼요!
    """
    config = config or get_config()
    output_dir = ensure_dir(EDA_DIR)

    # 이미 EDA를 돌린 적이 있고, force가 아니면 캐시를 재사용해요
    summary_path = output_dir / "eda_summary.json"
    if summary_path.exists() and not force_refresh:
        print("[EDA] 기존 결과를 불러올게요 (다시 하려면 --force를 써주세요)")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    start_time = time.time()
    print("\n" + "=" * 60)
    print("  EDA (탐색적 데이터 분석) 시작!")
    print("=" * 60)

    # ── 데이터 로딩 ────────────────────────────────
    # prepare_data()가 아직 안 됐으면 자동으로 해줘요. 걱정 마세요!
    print("\n[1/5] 데이터 로딩 중...")
    splits = prepare_data(config=config)
    all_df = pd.concat(
        [splits["train"], splits["val"], splits["test"]],
        ignore_index=True,
    )
    print(f"  전체 데이터: {len(all_df)} 샘플 "
          f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")

    # ── VADER 피처 로딩 ────────────────────────────
    print("\n[2/5] VADER 감성 피처 로딩 중...")
    vader_features = extract_vader_features(splits=splits)

    # ── 분석 실행 ──────────────────────────────────
    results: dict[str, Any] = {}

    # 1. 클래스 분포 (신규)
    print("\n[1/9] 클래스 분포 분석 중...")
    class_dist_results = _analyze_class_distribution(all_df, output_dir)
    results.update(class_dist_results)

    # 2. 텍스트 길이 분포
    print("\n[2/9] 텍스트 길이 분포 분석 중...")
    text_length_results = _analyze_text_length(all_df, output_dir, max_len=config.max_len)
    results.update(text_length_results)

    # 3. VADER 점수 분포
    print("\n[3/9] VADER 감성 점수 분포 분석 중...")
    vader_results = _analyze_vader_by_class(all_df, vader_features, splits, output_dir)
    results.update(vader_results)

    # 4. VADER 분리도 분석 (신규)
    print("\n[4/9] VADER 분리도 분석 중...")
    vader_sep_results = _analyze_vader_separability(all_df, vader_features, splits, output_dir)
    results.update(vader_sep_results)

    # 5. 타겟 커뮤니티 분석
    print("\n[5/9] 타겟 커뮤니티 분석 중...")
    target_results = _analyze_targets(all_df, output_dir)
    results.update(target_results)

    # 6. 어휘 겹침 분석
    print("\n[6/9] 어휘 겹침 분석 중...")
    vocab_results = _analyze_vocabulary_overlap(all_df, output_dir)
    results.update(vocab_results)

    # 7. N-gram 빈도 분석 (신규)
    print("\n[7/9] N-gram 빈도 분석 중...")
    ngram_results = _analyze_ngrams(all_df, output_dir)
    results.update(ngram_results)

    # 8. 워드클라우드 (신규)
    print("\n[8/9] 워드클라우드 생성 중...")
    wc_results = _analyze_wordcloud(all_df, output_dir)
    results.update(wc_results)

    # 9. Human Rationale 분포 (신규)
    print("\n[9/9] Human Rationale 분포 분석 중...")
    rationale_results = _analyze_rationale_distribution(all_df, output_dir)
    results.update(rationale_results)

    # ── 종합 요약 저장 ─────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    results["elapsed_seconds"] = elapsed
    _generate_summary(results, output_dir)

    print("\n" + "=" * 60)
    print(f"  EDA 완료! ({elapsed}초 소요)")
    print(f"  결과물 저장 위치: {output_dir}")
    print("=" * 60 + "\n")

    return results
