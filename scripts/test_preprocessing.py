"""
데이터 전처리 파이프라인 테스트

로드한 커넥톤 데이터를 처리하여 머신러닝 모델에 사용할 수 있는 형태로 변환합니다.
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_connectome, extract_neuron_pairs, generate_neuron_pair_data
from src.data.preprocessor import DataPreprocessor
from src.utils.config import get_config
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """전처리 파이프라인 전체 실행"""

    config = get_config()
    connectome_file = config.get('data.connectome_file')
    processed_dir = config.get('data.processed_dir')

    print("\n" + "="*70)
    print("🔨 전처리 파이프라인 테스트")
    print("="*70)

    # 1단계: 데이터 로드
    print("\n[1단계] 데이터 로드")
    print("-" * 70)
    connectome = load_connectome(connectome_file)
    pairs, labels = extract_neuron_pairs(connectome)
    pair_data, labels = generate_neuron_pair_data(connectome, pairs, labels)

    print(f"✓ 로드 완료:")
    print(f"  - 신경 쌍: {len(pairs)}개")
    print(f"  - 양성 샘플: {sum(labels)}개")
    print(f"  - 음성 샘플: {len(labels) - sum(labels)}개")

    # 2단계: 특징 엔지니어링
    print("\n[2단계] 특징 엔지니어링")
    print("-" * 70)
    preprocessor = DataPreprocessor(normalize=True, normalization_method='standard')
    X, y = preprocessor.build_feature_matrix(pair_data, labels)

    print(f"✓ 특징 행렬 생성 완료:")
    print(f"  - 행(샘플): {X.shape[0]}")
    print(f"  - 열(특징): {X.shape[1]}")
    print(f"\n특징 목록:")
    feature_names = X.columns.tolist()
    for i, fname in enumerate(feature_names[:10], 1):
        print(f"  {i:2d}. {fname}")
    if len(feature_names) > 10:
        print(f"  ... 외 {len(feature_names) - 10}개")

    print(f"\n특징 통계:")
    print(X.describe().to_string())

    # 3단계: 데이터 분할
    print("\n[3단계] Train/Val/Test 분할")
    print("-" * 70)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(
        X, y,
        test_size=config.get('preprocessing.test_size'),
        val_size=config.get('preprocessing.val_size')
    )

    print(f"✓ 데이터 분할 완료:")
    print(f"  - Train: {X_train.shape[0]} 샘플, 양성: {y_train.sum()}, 음성: {len(y_train) - y_train.sum()}")
    print(f"  - Val:   {X_val.shape[0]} 샘플, 양성: {y_val.sum()}, 음성: {len(y_val) - y_val.sum()}")
    print(f"  - Test:  {X_test.shape[0]} 샘플, 양성: {y_test.sum()}, 음성: {len(y_test) - y_test.sum()}")

    # 4단계: 정규화 확인
    print("\n[4단계] 정규화 확인")
    print("-" * 70)
    print(f"정규화 방법: {config.get('preprocessing.normalization_method')}")
    print(f"\n정규화 후 (numpy 배열) 통계:")
    print(f"  평균: {X_train.mean():.4f}")
    print(f"  표준편차: {X_train.std():.4f}")
    print(f"  최소값: {X_train.min():.4f}")
    print(f"  최대값: {X_train.max():.4f}")

    # 5단계: 클래스 불균형 확인
    print("\n[5단계] 클래스 불균형 분석")
    print("-" * 70)
    train_pos = y_train.sum()
    train_neg = len(y_train) - y_train.sum()
    val_pos = y_val.sum()
    val_neg = len(y_val) - y_val.sum()
    test_pos = y_test.sum()
    test_neg = len(y_test) - y_test.sum()

    print(f"Train 세트: {train_pos}개 양성, {train_neg}개 음성 → 비율 1:{train_neg/train_pos if train_pos > 0 else 0:.1f}")
    print(f"Val 세트:   {val_pos}개 양성, {val_neg}개 음성 → 비율 1:{val_neg/val_pos if val_pos > 0 else 0:.1f}")
    print(f"Test 세트:  {test_pos}개 양성, {test_neg}개 음성 → 비율 1:{test_neg/test_pos if test_pos > 0 else 0:.1f}")

    # 6단계: 데이터 저장
    print("\n[6단계] 전처리 데이터 저장")
    print("-" * 70)
    preprocessor.save_processed_data(
        processed_dir,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=feature_names
    )

    print(f"✓ 저장 완료:")
    print(f"  - {Path(processed_dir) / 'train_data.npz'}")
    print(f"  - {Path(processed_dir) / 'val_data.npz'}")
    print(f"  - {Path(processed_dir) / 'test_data.npz'}")
    print(f"  - {Path(processed_dir) / 'feature_names.npy'}")

    # 7단계: 요약
    print("\n" + "="*70)
    print("✅ 전처리 파이프라인 완료!")
    print("="*70)
    print(f"\n다음 단계:")
    print(f"  1. 기준선 모델 훈련: Logistic Regression")
    print(f"  2. Random Forest 모델 훈련")
    print(f"  3. 신경망 모델 훈련")
    print(f"  4. 모든 모델 비교 및 평가")

    return 0


if __name__ == "__main__":
    sys.exit(main())
