# HateSpeech XAI Pipeline

## 디렉토리 구조

```
hatespeech/
├── main.py              
├── requirements.txt
├── run.sh               
├── data/
│   ├── raw/             
│   └── processed/       
├── src/
│   ├── config.yaml      
│   ├── config.py        
│   ├── path.py          
│   ├── utils.py         
│   ├── data.py          
│   ├── vader.py         
│   ├── eda.py           
│   ├── xai.py           
│   ├── dashboard.py     
│   └── dashboard_app.py 
└── model/
    ├── models.py        
    └── train.py         
```

## 실행 방법

프로젝트 루트(`hatespeech/`)에서 실행 (`from src...`/`from model...` 임포트 기준).

```bash
python main.py status        # 파이프라인 상태 확인
python main.py data          # 1. 데이터 전처리 → data/processed/
python main.py vader         # 2. VADER 피처 추출
python main.py eda           # 3. 탐색적 데이터 분석
python main.py tune          # 4. 하이퍼파라미터 튜닝
python main.py benchmark     # 5. 8조건 벤치마크 (BERT/RoBERTa 학습)
python main.py freeze-study  # 6. encoder 동결 비교
python main.py xai           # 7. SHAP/LIME 설명
python main.py dashboard     # 8. HTML 대시보드
python main.py all           # 전체 순차 실행
```

