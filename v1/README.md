# v1 Legacy Baseline Archive

이 폴더는 과거 1차 파이프라인을 보관하는 기록용 공간입니다. 현재 새 실험과 팀 분업의 기준은 repo root가 아니라 `../v2/`입니다.

## 포함된 것

| 경로 | 내용 |
|---|---|
| `experiment_*.py`, `run_experiments.py`, `run.sh` | 1차 baseline 파이프라인 코드 |
| `data/` | 당시 사용한 HateXplain 전처리 데이터 |
| `outputs/` | 기존 실험 리포트, 대시보드, 정적 결과 |
| `docs/` | 발표 자료, 과거 v2.1 설계 초안, 참고 문서 |
| `checkpoints/` | 로컬 모델 체크포인트 보관 위치 |
| `빅데프 참고문헌pdf/` | 로컬 참고문헌 PDF 보관 위치 |

`checkpoints/`, 일부 pickle 산출물, 참고문헌 PDF는 대용량 또는 저작권 이슈 때문에 Git 추적 대상이 아닙니다.

## 기존 대시보드 실행

```bash
cd v1
pip install -r requirements.txt
python3 dashboard_app.py
```

기본 포트는 기존 코드 기준 `8501`입니다.

## 주의

- v1은 재현 기록과 baseline 확인용입니다.
- 새 코드 구현, 15 seed 실행, 통계 검증, XAI 보강은 `../v2/`에서 진행합니다.
- v1 문서 중 `v2`, `v2.1`이라는 이름이 들어간 파일은 과거 루트에서 작성된 초안입니다. 현재 canonical 문서는 `../v2/docs/`에 있습니다.
