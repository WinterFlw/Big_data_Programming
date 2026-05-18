# 역할별 Word 업무지시서

이 폴더는 팀원 5명에게 바로 나눠줄 수 있는 Word 자료를 담는다. 역할은 코드 파일 단위가 아니라 연구 산출물 단위로 나눈다.

```text
코드 리뷰/파이프라인 검증
-> 학습 실행/실험 관리
-> 결과 분석/통계 해석
-> XAI 설명/evidence bundle
-> 발표자료/최종 보고서 제작
```

## 파일 목록

| 파일 | 담당 역할 | 핵심 질문 |
|---|---|---|
| `01_code_review_pipeline_validation.docx` | 코드 리뷰 / 파이프라인 검증 | v2가 full run 전에 깨지지 않는가? |
| `02_training_execution_experiment_management.docx` | 학습 실행 / 실험 관리 | 서버에서 15 seed 실행과 실패 복구를 관리할 수 있는가? |
| `03_result_analysis_statistics.docx` | 결과 분석 / 통계 해석 | 성능 차이가 통계적으로 의미 있는가? |
| `04_xai_explanation_evidence_bundle.docx` | XAI 설명 / evidence bundle | 왜 그런 결과가 나왔는지 설명 근거를 남겼는가? |
| `05_presentation_report_final_integration.docx` | 발표자료 / 최종 보고서 제작 | 결과를 제출 가능한 발표/보고서로 묶었는가? |

## 분석 수준 한 줄 기준

메인 분석은 15 seed `mean ± std`, 핵심 비교 `A_B vs D_B paired t-test`, 평균 차이, 95% CI, effect size, XAI 대표 사례다. Holm-Bonferroni 보정은 여러 조건 비교를 한꺼번에 보여줄 때 과대해석을 줄이는 보조 안전장치로만 쓴다.

자세한 설명과 예시는 `03_result_analysis_statistics.docx`에 넣어두었다.
