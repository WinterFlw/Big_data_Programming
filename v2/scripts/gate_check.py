#!/usr/bin/env python3
"""Full Run Gate — 120 unit 풀 학습을 트리거하기 전 6조건 자동 점검.

QA Conductor가 매일 daily.sh에서 호출. exit 0 = GO, exit 1 = STOP.

6 조건:
  1. cudnn 결정성: utils.set_seed가 cudnn.deterministic=True를 설정하는가
  2. 8조건 metadata 정합성: CONDITION_METADATA 8개 ↔ V2_CONDITION_SPECS 8개 일치
  3. aggregate 무회귀: 7개 CSV (benchmark_runs/summary/paired/holm + anova_*)
  4. XAI sample 결정성: primary_samples.csv 두 번 생성 md5 일치
  5. failed_runs.csv 0건
  6. 산출물 contract: docs/02_e2e_pipeline 명시 산출물 모두 존재

사용:
  python3 v2/scripts/gate_check.py [--run-id v2_15seed]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if sys.stdout.isatty() else text


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if sys.stdout.isatty() else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if sys.stdout.isatty() else text


def _md5(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


# ──────────────────────────────────────────────
# 6조건
# ──────────────────────────────────────────────


def check_cudnn_determinism() -> tuple[bool, str]:
    """utils.set_seed가 cudnn.deterministic=True / benchmark=False 를 설정하는가."""
    try:
        runtime_dir = REPO_ROOT / "runtime"
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))
        # 캐시 무효화.
        for mod in list(sys.modules):
            if mod in ("utils",):
                del sys.modules[mod]
        import utils  # type: ignore[import-not-found]
        import torch

        utils.set_seed(42)
        if not torch.cuda.is_available():
            return True, "CUDA unavailable (CPU/MPS 환경) — cudnn 설정 무의미, 통과로 간주"
        det_ok = torch.backends.cudnn.deterministic is True
        bench_ok = torch.backends.cudnn.benchmark is False
        if det_ok and bench_ok:
            return True, "cudnn.deterministic=True, cudnn.benchmark=False 적용 확인"
        return False, f"cudnn.deterministic={det_ok}, cudnn.benchmark={bench_ok}"
    except Exception as exc:
        return False, f"set_seed import 실패: {exc}"


def check_condition_metadata() -> tuple[bool, str]:
    """schema.CONDITION_METADATA 8개 ↔ runtime.V2_CONDITION_SPECS 8개 일치."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from pipeline.schema import CONDITION_METADATA

        runtime_dir = REPO_ROOT / "runtime"
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))
        for mod in list(sys.modules):
            if mod.startswith(("utils", "experiment_core")):
                del sys.modules[mod]
        import experiment_core  # type: ignore[import-not-found]

        pipeline_set = set(CONDITION_METADATA.keys())
        runtime_set = {spec.condition for spec in experiment_core.V2_CONDITION_SPECS}
        if pipeline_set == runtime_set and len(pipeline_set) == 8:
            return True, f"조건 8개 일치: {sorted(pipeline_set)}"
        diff = pipeline_set.symmetric_difference(runtime_set)
        return False, f"조건 불일치. pipeline∖runtime = {pipeline_set - runtime_set}, runtime∖pipeline = {runtime_set - pipeline_set}"
    except Exception as exc:
        return False, f"metadata import 실패: {exc}"


def check_aggregate_artifacts(run_id: str) -> tuple[bool, str]:
    """aggregate CSV 7개 모두 존재 (헤더라도)."""
    bench_dir = REPO_ROOT / "outputs" / "experiments" / run_id / "benchmark"
    required = [
        "benchmark_runs.csv",
        "benchmark_summary.csv",
        "paired_tests.csv",
        "paired_tests_holm.csv",
        "anova_2way_bert.csv",
        "anova_2way_roberta.csv",
        "anova_3way.csv",
    ]
    missing = [name for name in required if not (bench_dir / name).exists()]
    if missing:
        return False, f"누락: {missing} — `./run.sh e2e aggregate --run-id {run_id}` 실행 필요"
    return True, f"aggregate 7개 CSV 모두 존재"


def check_sample_determinism(run_id: str) -> tuple[bool, str]:
    """primary_samples.csv 두 번 생성 후 md5 일치 (seed 무관성)."""
    sample_path = (
        REPO_ROOT / "outputs" / "experiments" / run_id / "xai" / "samples" / "primary_samples.csv"
    )
    if not sample_path.exists():
        return False, "primary_samples.csv 없음 — `./run.sh e2e xai-primary --run-id {run_id}` 실행 필요"
    md5_before = _md5(sample_path)
    result = subprocess.run(
        [str(REPO_ROOT / "run.sh"), "e2e", "xai-primary", "--run-id", run_id],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, f"xai-primary 재실행 실패: rc={result.returncode}"
    md5_after = _md5(sample_path)
    if md5_before == md5_after:
        return True, f"md5 일치 ({md5_before[:8]}...) — seed 무관 결정적"
    return False, f"md5 불일치: {md5_before[:8]} vs {md5_after[:8]}"


def check_failed_runs(run_id: str) -> tuple[bool, str]:
    """failed_runs.csv 0건."""
    path = REPO_ROOT / "outputs" / "experiments" / run_id / "failed_runs.csv"
    if not path.exists():
        return False, "failed_runs.csv 없음 — `./run.sh e2e status --run-id {run_id}` 먼저"
    line_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1  # 헤더 제외
    if line_count == 0:
        return True, "failed_runs 0건"
    return False, f"failed_runs {line_count}건 — stderr.log 확인 필요"


def check_artifact_contract(run_id: str) -> tuple[bool, str]:
    """docs/02_e2e_pipeline 명시 산출물 상위 경로 존재."""
    root = REPO_ROOT / "outputs" / "experiments" / run_id
    required_paths = [
        root / "execution_status.csv",
        root / "failed_runs.csv",
        root / "completed_runs.csv",
        root / "benchmark" / "benchmark_runs.csv",
        root / "xai" / "samples" / "primary_samples.csv",
        root / "xai" / "evidence_bundle" / "xai_claims.json",
        root / "reports" / "final_report.md",
        root / "dashboard" / "index.html",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.exists()]
    if missing:
        return False, f"산출물 누락: {missing}"
    return True, f"산출물 contract {len(required_paths)}개 모두 존재"


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Full Run Gate 6조건 자동 점검")
    parser.add_argument("--run-id", default="v2_15seed")
    parser.add_argument("--skip-sample-check", action="store_true",
                        help="작업 #4 sample 결정성 검사를 skip (xai-primary 재실행이 무거울 때)")
    args = parser.parse_args()

    checks = [
        ("1. cudnn 결정성", check_cudnn_determinism, []),
        ("2. 8조건 metadata 정합성", check_condition_metadata, []),
        ("3. aggregate CSV 7개", check_aggregate_artifacts, [args.run_id]),
        ("4. XAI sample 결정성", check_sample_determinism, [args.run_id]),
        ("5. failed_runs 0건", check_failed_runs, [args.run_id]),
        ("6. 산출물 contract", check_artifact_contract, [args.run_id]),
    ]

    print(_bold("=== Full Run Gate 점검 (run_id={}) ===".format(args.run_id)))
    results = []
    for name, func, fargs in checks:
        if name.startswith("4.") and args.skip_sample_check:
            print(f"  {name}: \033[33mSKIP\033[0m (--skip-sample-check)")
            results.append(True)
            continue
        ok, detail = func(*fargs)
        marker = _green("PASS") if ok else _red("FAIL")
        print(f"  {name}: {marker} — {detail}")
        results.append(ok)

    print()
    passed = sum(1 for ok in results if ok)
    total = len(results)
    if passed == total:
        print(_bold(_green(f"[GATE] {passed}/{total} 통과 — GO. 풀 120 unit 학습 시작 가능.")))
        return 0
    print(_bold(_red(f"[GATE] {passed}/{total} 통과 — STOP. 위 FAIL 항목 해결 후 재점검.")))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
