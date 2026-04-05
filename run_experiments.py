# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                    ║
# ║   run_experiments.py  --  파이프라인의 관제탑 (Control Tower)        ║
# ║                                                                    ║
# ║   이 파일은 우리 혐오표현 탐지 프로젝트의 "관제탑" 역할을 해요!       ║
# ║   터미널에서 ./run.sh 명령을 치면 가장 먼저 이 파일이 실행된답니다.   ║
# ║                                                                    ║
# ║   비행기가 이륙하려면 관제탑의 허가가 필요하듯이,                     ║
# ║   데이터 전처리 → 피처 추출 → 튜닝 → 벤치마크 → XAI 분석까지       ║
# ║   모든 단계가 이 관제탑을 통해 순서대로 출발해요. :)                  ║
# ║                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
CLI 엔트리포인트 -- run.sh에서 호출되는 메인 실행기.

안녕하세요, 이 파일에 오신 걸 환영해요!
이 스크립트는 HateXplain 혐오표현 탐지 파이프라인의 "관제탑"이에요.

터미널에서 하나의 명령어로 복잡한 실험 과정을 간편하게 실행할 수 있도록
argparse를 이용해 CLI(Command Line Interface)를 만들어 두었답니다.

사용법 (하나씩 따라 해 보세요!):
  ./run.sh data          # 1단계: 데이터 전처리 (모든 것의 시작이에요!)
  ./run.sh vader         # 2단계: VADER 감성 피처 추출 (감성 점수를 뽑아요)
  ./run.sh tune          # 3단계: 하이퍼파라미터 탐색 (최적의 설정을 찾아요)
  ./run.sh benchmark     # 4단계: seed 3회 반복 벤치마크 (공정한 성능 비교!)
  ./run.sh freeze-study  # 5단계: encoder 동결 비교 (fine-tune vs freeze 실험)
  ./run.sh xai           # 6단계: SHAP + LIME 분석 (모델이 왜 그런 판단을 했는지!)
  ./run.sh dashboard     # 7단계: HTML 대시보드 생성 (예쁜 리포트 완성!)
  ./run.sh all           # 전체 파이프라인 순차 실행 (한 방에 다 돌리기!)

Tip: 각 단계는 독립적으로도, 순차적으로도 실행할 수 있어요.
     처음이라면 'all'로 한번에 돌려보는 것도 좋은 방법이에요!
"""

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  임포트 (Import) 섹션                                               ║
# ║                                                                     ║
# ║  요리를 시작하기 전에 재료를 준비하는 것처럼,                          ║
# ║  코드를 실행하기 전에 필요한 모듈들을 불러와야 해요!                    ║
# ║  각 모듈이 어떤 역할을 하는지 하나씩 살펴볼게요.                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Python 3.10+ 스타일의 타입 힌트를 이전 버전에서도 쓸 수 있게 해줘요.
# 예를 들어 list[str] | None 같은 표현이 가능해진답니다!
from __future__ import annotations

# argparse: 터미널에서 명령어를 파싱(분석)해주는 파이썬 표준 라이브러리예요.
# 우리가 ./run.sh data 라고 치면, "data"를 알아서 읽어주는 고마운 친구!
import argparse

# pandas: 데이터 분석의 국민 라이브러리! DataFrame으로 표 형태 데이터를 다뤄요.
# 여기서는 최종 리포트에서 XAI 요약을 DataFrame으로 변환할 때 쓰여요.
import pandas as pd

# experiment_dashboard: 실험 결과를 예쁜 HTML 대시보드로 만들어주는 모듈이에요.
# 실험이 끝날 때마다 대시보드를 업데이트해서 결과를 한눈에 볼 수 있게 해줘요!
from experiment_dashboard import DASHBOARD_HTML_PATH, run_dashboard

# experiment_core: 우리 파이프라인의 핵심 엔진이 들어있는 모듈이에요!
# 데이터 준비부터 벤치마크까지, 무거운 연산은 대부분 이 친구가 담당해요.
from experiment_core import (
    BENCHMARK_SUMMARY_PATH,   # 벤치마크 결과가 저장되는 경로
    FREEZE_STUDY_PATH,        # 동결 실험 결과가 저장되는 경로
    OUTPUT_DIR,               # 전체 출력 디렉토리 (outputs/)
    REPORT_DIR,               # 리포트가 저장되는 디렉토리 (reports/)
    TUNING_LOG_PATH,          # 하이퍼파라미터 튜닝 로그 경로
    VADER_PICKLE_PATH,        # VADER 감성 피처가 저장되는 pickle 경로
    XAI_DIR,                  # XAI 분석 결과가 저장되는 디렉토리
    describe_status,          # 파이프라인 각 단계의 상태를 확인하는 함수
    extract_vader_features,   # VADER 감성 점수를 추출하는 함수
    get_config,               # 실험 설정(config)을 불러오는 함수
    prepare_data,             # 원본 데이터를 train/val/test로 나누는 함수
    run_benchmark,            # 모델 벤치마크를 실행하는 함수
    run_freeze_study,         # encoder 동결 비교 실험을 실행하는 함수
    run_hyperparameter_tuning,  # 최적 하이퍼파라미터를 탐색하는 함수
    save_config,              # 현재 설정을 파일로 저장하는 함수
)

# experiment_xai: SHAP과 LIME을 활용한 설명 가능한 AI(XAI) 분석 모듈이에요.
# "이 문장이 왜 혐오표현으로 분류됐을까?"에 대한 답을 찾아줘요!
from experiment_xai import run_xai

# utils: 여러 곳에서 공통으로 쓰이는 유틸리티 함수 모음이에요.
# 파일 저장, 디렉토리 삭제, 마크다운 변환 등 자잘하지만 꼭 필요한 기능들!
from utils import CHECKPOINT_DIR, dataframe_to_markdown, remove_tree, save_text


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  _print_status() -- 파이프라인 건강 체크 함수                        ║
# ║                                                                     ║
# ║  병원에서 건강검진 받듯이, 파이프라인의 각 단계가                      ║
# ║  제대로 완료되었는지 한눈에 확인할 수 있어요!                          ║
# ║                                                                     ║
# ║  "ready"가 뜨면 해당 단계가 이미 완료된 거고,                         ║
# ║  "missing"이 뜨면 아직 실행하지 않은 단계예요.                        ║
# ║  걱정 마세요, missing이 있어도 괜찮아요! 차근차근 하면 돼요 :)         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _print_status() -> None:
    # describe_status()가 딕셔너리를 돌려줘요: {"data": True, "vader": False, ...}
    # True면 해당 산출물이 존재하는 것이고, False면 아직 없다는 뜻이에요.
    status = describe_status()

    # 터미널에 깔끔하게 상태표를 출력해 줄게요!
    print("\nPipeline status")
    print("================")
    for key, value in status.items():
        # 각 단계의 이름을 20자 너비로 맞추고, 상태를 ready/missing으로 표시해요.
        # 포맷 문자열 f"{key:20s}"는 문자열을 20칸으로 정렬해주는 파이썬 기능이에요!
        print(f"{key:20s}: {'ready' if value else 'missing'}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  _clean_outputs() -- 산출물 초기화 함수                              ║
# ║                                                                     ║
# ║  !! 주의 !! 이 함수는 outputs/와 checkpoints/ 폴더를                 ║
# ║  통째로 삭제하고 다시 만들어요!                                       ║
# ║                                                                     ║
# ║  실험을 처음부터 깨끗하게 다시 시작하고 싶을 때 사용하세요.             ║
# ║  이미 오래 걸린 실험 결과가 있다면, 삭제하기 전에 백업해 두는 게       ║
# ║  좋겠죠? 한번 지우면 되돌릴 수 없으니까요! (꼭 기억해 주세요~)        ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _clean_outputs() -> None:
    # OUTPUT_DIR(outputs/)과 CHECKPOINT_DIR(checkpoints/)을 순회하며 삭제해요.
    for directory in [OUTPUT_DIR, CHECKPOINT_DIR]:
        remove_tree(directory)  # 디렉토리와 그 안의 모든 파일을 재귀적으로 삭제!

    # 삭제한 뒤, 빈 디렉토리를 다시 만들어줘요.
    # parents=True: 상위 디렉토리가 없으면 같이 생성
    # exist_ok=True: 이미 존재해도 에러 없이 넘어감
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("Removed outputs/ and checkpoints/ artifacts.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                                                                     ║
# ║               main() -- 관제탑의 메인 컨트롤 함수                    ║
# ║                                                                     ║
# ║   여기가 바로 모든 명령이 분기되는 핵심 함수예요!                      ║
# ║   터미널에서 입력한 명령어(data, vader, tune, ...)를 읽고,            ║
# ║   해당하는 파이프라인 단계를 실행해요.                                ║
# ║                                                                     ║
# ║   마치 식당 주방에서 주문서를 읽고 요리를 시작하는 것처럼,             ║
# ║   main()은 사용자의 "주문"을 받아 적절한 "요리"를 시작합니다!         ║
# ║                                                                     ║
# ║   반환값: 0이면 성공, 1이면 실패 (유닉스 관례를 따라요)               ║
# ╚══════════════════════════════════════════════════════════════════════╝
def main(argv: list[str] | None = None) -> int:
    # ── argparse 설정: 어떤 명령어를 받을지 정의하는 부분이에요 ──
    # argparse는 터미널 입력을 자동으로 파싱하고, --help도 만들어줘요! 정말 편하죠?
    parser = argparse.ArgumentParser(description="Run hate-speech experiments aligned with the report.")

    # command: 사용자가 어떤 단계를 실행할지 선택하는 위치 인자(positional argument)예요.
    # choices에 나열된 것만 허용되고, 그 외의 값을 넣으면 argparse가 에러를 띄워줘요.
    parser.add_argument(
        "command",
        choices=["data", "vader", "tune", "benchmark", "freeze-study", "xai", "dashboard", "all", "status", "clean"],
        help="Pipeline stage to run.",
    )

    # --force: 이미 캐시된 결과물이 있어도 강제로 다시 만들고 싶을 때 쓰는 옵션이에요.
    # "이전 결과 무시하고 처음부터 다시 해줘!"라는 의미예요.
    parser.add_argument("--force", action="store_true", help="Rebuild cached artifacts for this stage.")

    # --with-tuning: 'all' 명령어를 쓸 때, 하이퍼파라미터 튜닝까지 포함할지 결정해요.
    # 튜닝은 시간이 오래 걸릴 수 있어서, 기본값은 꺼져 있답니다. 필요할 때만 켜세요!
    parser.add_argument(
        "--with-tuning",
        action="store_true",
        help="When running 'all', execute tuning before the benchmark.",
    )

    # 인자를 파싱해요! argv가 None이면 sys.argv를 자동으로 읽어요.
    args = parser.parse_args(argv)

    # ── 설정(config) 로딩: 실험에 필요한 설정값들을 불러와요 ──
    # config에는 모델 이름, 학습률, 배치 크기 등 실험의 "레시피"가 들어있어요!
    config = get_config()
    # 현재 설정을 파일로 저장해 두면, 나중에 "이 실험 때 어떤 설정이었지?" 확인할 수 있어요.
    save_config(config)

    # ╔═══════════════════════════════════════════════════════╗
    # ║  여기서부터 각 명령어에 따른 분기(branching)가 시작돼요!  ║
    # ║  if문을 하나씩 읽어보면 파이프라인 흐름이 보일 거예요 :)  ║
    # ╚═══════════════════════════════════════════════════════╝

    # ── [data] 데이터 전처리 단계 ──
    # HateXplain 원본 데이터를 다운로드하고, train/val/test로 나눠요.
    # 모든 실험의 첫걸음! 데이터가 없으면 아무것도 할 수 없으니까요.
    if args.command == "data":
        prepare_data(config=config, force_refresh=args.force, force_download=False)
        dashboard_path = run_dashboard()  # 대시보드도 업데이트해 줘요!
        print(f"Saved prepared splits under {OUTPUT_DIR}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0  # 0은 "성공적으로 끝났어요!"라는 뜻이에요

    # ── [vader] VADER 감성 피처 추출 단계 ──
    # VADER(Valence Aware Dictionary and sEntiment Reasoner)는
    # 사전 기반 감성 분석 도구예요. 각 문장에 긍정/부정/중립 점수를 매겨줘요.
    # 이 점수를 추가 피처로 활용하면 모델 성능이 올라갈 수 있어요!
    if args.command == "vader":
        splits = prepare_data(config=config)  # 먼저 데이터가 준비되어 있는지 확인!
        extract_vader_features(splits=splits, force_refresh=args.force)
        dashboard_path = run_dashboard()
        print(f"Saved VADER features to {VADER_PICKLE_PATH}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    # ── [tune] 하이퍼파라미터 튜닝 단계 ──
    # 학습률, 드롭아웃 비율 등 최적의 조합을 자동으로 탐색해요.
    # 요리로 치면 "소금을 얼마나 넣어야 가장 맛있을까?"를 찾는 과정이에요!
    # 시간이 좀 걸릴 수 있지만, 좋은 하이퍼파라미터는 성능에 큰 영향을 줘요.
    if args.command == "tune":
        tuning_summary = run_hyperparameter_tuning(config=config)
        dashboard_path = run_dashboard()
        print("Tuning complete.")
        # 각 모델별로 찾아낸 최적의 파라미터를 출력해 줘요.
        for model_name, params in tuning_summary.items():
            print(f"  {model_name}: {params}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    # ── [benchmark] 벤치마크 단계 ──
    # 여러 모델을 공정하게 비교 평가하는 단계예요!
    # seed를 3번 바꿔가며 반복 실행해서, 운이 좋아서 잘 나온 건 아닌지 확인해요.
    # Macro F1, Precision, Recall, Accuracy를 한눈에 비교할 수 있어요.
    if args.command == "benchmark":
        summary = run_benchmark(config=config)
        dashboard_path = run_dashboard()
        print("Benchmark complete.")
        # 결과를 깔끔한 표 형태로 터미널에 출력해요.
        # to_string(index=False)는 왼쪽의 인덱스 번호를 숨겨서 더 깔끔하게 보여줘요!
        print(summary[["model", "macro_f1_display", "macro_precision_display", "macro_recall_display", "accuracy_display"]].to_string(index=False))
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    # ── [freeze-study] Encoder 동결 비교 실험 단계 ──
    # BERT 같은 사전학습 모델의 encoder 부분을 얼려놓고(freeze) 학습하면
    # 성능이 어떻게 달라지는지 비교하는 실험이에요.
    # "Fine-tuning이 정말 필요한가?" 라는 연구 질문에 답을 줘요!
    if args.command == "freeze-study":
        summary = run_freeze_study(config=config)
        dashboard_path = run_dashboard()
        print("Freeze study complete.")
        print(summary[["model", "macro_f1_display", "macro_precision_display", "macro_recall_display", "accuracy_display"]].to_string(index=False))
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    # ── [xai] 설명 가능한 AI (Explainable AI) 분석 단계 ──
    # SHAP과 LIME을 사용해서 모델의 판단 근거를 시각적으로 보여줘요!
    # "이 문장에서 어떤 단어 때문에 혐오표현이라고 판단했을까?"
    # 블랙박스 모델을 투명하게 만들어주는 아주 중요한 단계예요!
    if args.command == "xai":
        summary = run_xai(config=config)
        dashboard_path = run_dashboard()
        print("XAI complete.")
        # XAI 분석 요약 정보를 하나씩 출력해요.
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"Dashboard updated: {dashboard_path}")
        return 0

    # ── [dashboard] 대시보드 생성 단계 ──
    # 지금까지 만들어진 모든 결과를 예쁜 HTML 페이지로 모아줘요!
    # 브라우저에서 열면 차트와 표가 한눈에 보인답니다. 발표 자료로도 딱이에요!
    if args.command == "dashboard":
        dashboard_path = run_dashboard()
        print(f"Dashboard generated: {dashboard_path}")
        return 0

    # ── [all] 전체 파이프라인 순차 실행 ──
    # 위의 모든 단계를 순서대로 한 번에 실행해요!
    # 처음 프로젝트를 세팅하거나, 전체 재현(reproduction)이 필요할 때 유용해요.
    # 커피 한 잔 마시면서 기다리면, 모든 결과가 자동으로 만들어져요!
    if args.command == "all":
        # 1단계: 데이터 준비 (모든 것의 기초!)
        prepare_data(config=config, force_refresh=args.force, force_download=False)

        # 2단계: VADER 감성 피처 추출
        extract_vader_features(force_refresh=args.force)

        # 3단계 (선택): 하이퍼파라미터 튜닝 -- --with-tuning 플래그가 있을 때만!
        # 튜닝은 시간이 오래 걸려서, 기본적으로는 건너뛰어요.
        if args.with_tuning:
            run_hyperparameter_tuning(config=config)

        # 4단계: 벤치마크 -- 모든 모델의 성능을 공정하게 비교!
        benchmark_summary = run_benchmark(config=config)

        # 5단계: Encoder 동결 비교 실험
        freeze_summary = run_freeze_study(config=config)

        # 6단계: XAI 분석 -- 모델의 판단 근거를 설명!
        xai_summary = run_xai(config=config)

        # ── 최종 리포트 생성 ──
        # 모든 실험 결과를 하나의 마크다운 파일로 정리해요.
        # 이 파일 하나만 읽으면 실험 전체를 파악할 수 있답니다!
        final_report = (
            "# Final Experiment Bundle\n\n"
            "## Benchmark Summary\n\n"
            + dataframe_to_markdown(
                benchmark_summary[
                    [
                        "model",
                        "macro_f1_display",
                        "macro_precision_display",
                        "macro_recall_display",
                        "accuracy_display",
                    ]
                ]
            )
            + "\n\n## Freeze Study\n\n"
            + dataframe_to_markdown(
                freeze_summary[
                    [
                        "model",
                        "macro_f1_display",
                        "macro_precision_display",
                        "macro_recall_display",
                        "accuracy_display",
                    ]
                ]
            )
            + "\n\n## XAI Summary\n\n"
            + dataframe_to_markdown(
                pd.DataFrame([xai_summary])
            )
        )
        # 리포트를 파일로 저장! reports/final_bundle.md에 생성돼요.
        save_text(final_report, REPORT_DIR / "final_bundle.md")

        # 마지막으로 대시보드도 업데이트!
        dashboard_path = run_dashboard()

        # 전체 파이프라인 완료! 축하해요!
        print("Full pipeline complete.")
        print(f"Benchmark summary: {BENCHMARK_SUMMARY_PATH}")
        print(f"Freeze study: {FREEZE_STUDY_PATH}")
        # 튜닝 로그는 --with-tuning을 사용했을 때만 존재해요.
        print(f"Tuning log: {TUNING_LOG_PATH if TUNING_LOG_PATH.exists() else 'not generated'}")
        print(f"XAI outputs: {XAI_DIR}")
        print(f"Dashboard: {dashboard_path}")
        return 0

    # ── [status] 파이프라인 상태 확인 ──
    # 현재 어떤 단계까지 완료됐는지 간단히 체크해요.
    # 실험 중간에 "지금 어디까지 했더라?" 할 때 유용해요!
    if args.command == "status":
        _print_status()
        return 0

    # ── [clean] 산출물 초기화 ──
    # 모든 실험 결과를 삭제하고 깨끗한 상태로 되돌려요.
    # 신중하게 사용해 주세요! (백업 먼저 하는 거, 잊지 않으셨죠?)
    if args.command == "clean":
        _clean_outputs()
        return 0

    # 여기까지 왔다면 알 수 없는 명령어가 입력된 거예요.
    # (사실 argparse의 choices가 먼저 잡아주기 때문에, 여기까지 올 일은 거의 없어요!)
    return 1


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  프로그램 시작점 (Entry Point)                                       ║
# ║                                                                     ║
# ║  파이썬 파일을 직접 실행하면 __name__이 "__main__"이 돼요.            ║
# ║  이 if문 덕분에 "import용"과 "실행용"을 구분할 수 있답니다.           ║
# ║                                                                     ║
# ║  raise SystemExit(main())은 main()의 반환값(0 또는 1)을              ║
# ║  프로세스 종료 코드로 전달해요. 0이면 성공, 그 외는 실패!              ║
# ║                                                                     ║
# ║  여기까지 읽어주셔서 감사해요!                                        ║
# ║  코드를 읽는 것만으로도 대단한 거예요. 화이팅! :)                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    raise SystemExit(main())
