# C. elegans 신경망 시냅스 예측 (C. elegans ANN)

머신러닝을 사용하여 선충류(C. elegans)의 신경 쌍 사이에 시냅스(신경 연결)가 존재할지 **이진 분류로 예측**하는 프로젝트입니다.

## 📋 프로젝트 개요

**목표**: C. elegans의 신경 연결(시냅스) 존재 여부를 머신러닝으로 예측

**데이터**: 
- C. elegans 커넥톰(신경 연결 지도)
- 302개의 뉴런, ~7,000개의 시냅스
- 출처: [altundag/celegans-connectome](https://github.com/altundag/celegans-connectome)

**접근 방식**:
1. 신경 쌍의 공간적, 구조적 특징 계산
2. 기준선 모델부터 시작 (Logistic Regression, Random Forest)
3. 신경망 및 XGBoost 등 고급 모델로 개선
4. 앙상블 모델로 최종 최적화

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 다운로드

```bash
# C. elegans 커넥톰 데이터 다운로드 및 변환
python scripts/download_data.py
```

이 명령은 다음을 수행합니다:
- GitHub에서 CSV 형식 데이터 다운로드
- JSON 형식으로 변환
- `data/raw/connectome.json` 저장

### 3. 데이터 전처리 및 모델 훈련

```bash
# (곧 구현됨)
python scripts/train_model.py
```

## 📁 프로젝트 구조

```
C.-elegans_ANN/
├── README.md                    # 프로젝트 설명
├── requirements.txt             # Python 의존성
│
├── data/
│   ├── raw/                     # 원본 커넥톰 데이터
│   │   └── connectome.json
│   └── processed/               # 전처리된 훈련 데이터
│
├── src/                         # 소스 코드
│   ├── data/
│   │   ├── loader.py            # 데이터 로딩
│   │   └── preprocessor.py      # 특징 엔지니어링
│   ├── models/                  # 머신러닝 모델
│   ├── training/                # 훈련 유틸리티
│   ├── evaluation/              # 평가 및 시각화
│   └── utils/
│       └── config.py            # 설정 관리
│
├── scripts/
│   ├── download_data.py         # 데이터 다운로드
│   ├── train_model.py           # 모델 훈련 (예정)
│   └── evaluate_model.py        # 모델 평가 (예정)
│
├── models/                      # 저장된 모델
├── notebooks/                   # Jupyter 노트북
├── tests/                       # 단위 테스트
├── config/
│   └── default.yaml             # 기본 설정
└── reports/                     # 결과 보고서
```

## 📊 데이터 설명

### 입력 특징 (Features)

각 신경 쌍(source neuron, target neuron)에 대해 다음 특징들을 계산합니다:

- **euclidean_distance**: 3D 공간에서 두 뉴런 사이의 거리
- **spatial_proximity_score**: 거리 기반 근접성 점수 (0~1)
- **source_type_*, target_type_***: 뉴런 타입의 원-핫 인코딩
  - sensory (감각 뉴런)
  - motor (운동 뉴런)
  - interneuron (중간 뉴런)
- **same_type**: 두 뉴런이 같은 타입인지 여부

### 출력 레이블 (Label)

- **0**: 시냅스 없음 (뉴런 쌍이 연결되지 않음)
- **1**: 시냅스 있음 (뉴런 쌍이 연결됨)

### 데이터 분할

- **Train**: 70% (4,900개 시냅스)
- **Validation**: 15% (1,050개)
- **Test**: 15% (1,050개)

**주의**: 클래스 불균형 (양성:음성 ≈ 1:10) 처리됨

## 🧠 모델 아키텍처

### Week 2: MVP 모델 (기준선)

1. **Logistic Regression**
   - 목표: 75-80% Accuracy
   - 빠른 훈련, 해석 가능

2. **Random Forest**
   - 목표: 82-87% Accuracy
   - 특징 중요도 제공

3. **Simple Neural Network**
   - 구조: Input → Dense(128) → Dropout(0.3) → Dense(64) → Dense(32) → Output(sigmoid)
   - 목표: 85-90% Accuracy

### Week 3-4: 개선된 모델

4. **XGBoost**
   - 목표: 90-93% Accuracy

5. **Advanced Neural Network**
   - Batch Normalization 추가
   - 더 깊은 레이어 (256 → 128 → 64 → 32)

6. **Ensemble Model**
   - 여러 모델의 예측을 결합
   - 목표: 94-96% Accuracy

## 📈 개발 로드맵 (4주)

- **Week 1**: ✓ 기초 구조 설정 (현재)
  - 프로젝트 디렉토리 생성
  - 데이터 로더, 전처리기 구현
  - 데이터 다운로드 스크립트 작성

- **Week 2**: MVP 모델 개발
  - 전처리 파이프라인 완성
  - 기준선 모델 (Logistic Regression, Random Forest) 구현
  - 간단한 신경망 훈련
  - 첫 평가 및 비교

- **Week 3**: 모델 개선
  - 고급 특징 추가
  - XGBoost 모델 구현
  - 하이퍼파라미터 튜닝
  - 성능 비교

- **Week 4**: 최적화 및 완성
  - 최고 성능 모델 최적화
  - 앙상블 모델 구현
  - 최종 평가 및 시각화
  - 문서화

## 🛠️ 사용 기술

- **데이터 처리**: NumPy, Pandas, SciPy
- **머신러닝**: scikit-learn, TensorFlow/Keras, XGBoost
- **시각화**: Matplotlib, Seaborn, Plotly
- **개발 도구**: Jupyter, pytest

## 📚 주요 참고 자료

- [WormAtlas - C. elegans Connectome](http://www.wormneural.org/)
- [altundag/celegans-connectome - GitHub](https://github.com/altundag/celegans-connectome)
- C. elegans의 신경계 연구 논문

## ✅ 다음 단계

1. **데이터 다운로드**: `python scripts/download_data.py` 실행
2. **데이터 탐색**: `notebooks/01_data_exploration.ipynb` 작성
3. **전처리 파이프라인**: 완전히 테스트
4. **첫 번째 모델**: Logistic Regression 훈련

---

**시작일**: 2026년 10월 1일  
**상태**: 📍 Week 1 - 기초 구조 설정 중