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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── 우리 프로젝트의 유틸리티 & 핵심 모듈 ──────────
from utils import (
    LABEL_NAMES,
    NUM_LABELS,
    REPORT_DIR,
    VADER_COLUMNS,
    ensure_dir,
    save_dataframe,
    save_json,
    save_text,
)
from experiment_core import (
    ExperimentConfig,
    get_config,
    load_splits,
    extract_vader_features,
    prepare_data,
)

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
# ║  5. 종합 요약 보고서 생성                                  ║
# ║     — 모든 분석 결과를 JSON과 마크다운으로 정리해요         ║
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
        md_lines.append(f"![Text Length Distribution](text_length_distribution.png)")
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
        md_lines.append(f"![VADER by Class](vader_by_class.png)")
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
        md_lines.append(f"![Target Distribution](target_distribution.png)")
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

    # 1. 텍스트 길이 분포
    print("\n[3/5] 텍스트 길이 분포 분석 중...")
    text_length_results = _analyze_text_length(all_df, output_dir, max_len=config.max_len)
    results.update(text_length_results)

    # 2. VADER 점수 분포
    print("\n[4/5] VADER 감성 점수 분포 분석 중...")
    vader_results = _analyze_vader_by_class(all_df, vader_features, splits, output_dir)
    results.update(vader_results)

    # 3. 타겟 커뮤니티 분석
    print("\n[5/5] 타겟 커뮤니티 & 어휘 겹침 분석 중...")
    target_results = _analyze_targets(all_df, output_dir)
    results.update(target_results)

    # 4. 어휘 겹침 분석
    vocab_results = _analyze_vocabulary_overlap(all_df, output_dir)
    results.update(vocab_results)

    # ── 종합 요약 저장 ─────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    results["elapsed_seconds"] = elapsed
    _generate_summary(results, output_dir)

    print("\n" + "=" * 60)
    print(f"  EDA 완료! ({elapsed}초 소요)")
    print(f"  결과물 저장 위치: {output_dir}")
    print("=" * 60 + "\n")

    return results
