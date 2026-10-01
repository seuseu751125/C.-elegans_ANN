"""
데이터 전처리 및 특징 엔지니어링

커넥톰 데이터를 머신러닝 모델에 사용할 수 있는 형태로 변환합니다.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    C. elegans 커넥톰 데이터를 전처리하는 클래스

    기능:
    1. 신경 쌍에 대한 특징 계산
    2. 특징 정규화
    3. 데이터 불균형 처리
    4. Train/Val/Test 분할
    """

    def __init__(
        self,
        normalize: bool = True,
        normalization_method: str = 'standard',
        random_state: int = 42
    ):
        """
        전처리기 초기화

        Args:
            normalize: 특징 정규화 여부
            normalization_method: 'standard' (StandardScaler) 또는 'minmax' (MinMaxScaler)
            random_state: 난수 생성 시드
        """
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None

    def compute_spatial_features(
        self,
        source_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        source_type: str,
        target_type: str
    ) -> Dict[str, float]:
        """
        신경 쌍에 대한 공간적 특징 계산

        특징들:
        - euclidean_distance: 3D 유클리디안 거리
        - spatial_proximity_score: 거리 기반 근접성 점수 (0~1)
        - source_type_encoded: 출발 뉴런 타입의 원-핫 인코딩
        - target_type_encoded: 도착 뉴런 타입의 원-핫 인코딩

        Args:
            source_pos: 출발 뉴런의 3D 좌표 (x, y, z)
            target_pos: 도착 뉴런의 3D 좌표
            source_type: 출발 뉴런의 타입
            target_type: 도착 뉴런의 타입

        Returns:
            특징 딕셔너리
        """
        features = {}

        # 유클리디안 거리 계산
        if source_pos and target_pos:
            source_arr = np.array(source_pos)
            target_arr = np.array(target_pos)

            euclidean_dist = np.linalg.norm(source_arr - target_arr)
            features['euclidean_distance'] = euclidean_dist

            # 근접성 점수: 거리가 작을수록 높은 값 (시그모이드 함수 사용)
            # 거리 200을 기준으로 스케일링 (C. elegans의 전형적인 거리)
            proximity_score = 1.0 / (1.0 + np.exp((euclidean_dist - 50) / 50))
            features['spatial_proximity_score'] = proximity_score
        else:
            features['euclidean_distance'] = np.nan
            features['spatial_proximity_score'] = np.nan

        # 신경 타입 인코딩
        neuron_types = ['sensory', 'motor', 'interneuron', 'polymodal', 'unknown']
        source_type_encoded = source_type if source_type in neuron_types else 'unknown'
        target_type_encoded = target_type if target_type in neuron_types else 'unknown'

        features['source_type'] = source_type_encoded
        features['target_type'] = target_type_encoded

        # 같은 타입인지 다른 타입인지
        features['same_type'] = 1.0 if source_type == target_type else 0.0

        return features

    def build_feature_matrix(
        self,
        pair_data: List[Dict],
        labels: List[int]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        신경 쌍 데이터에서 특징 행렬 생성

        Args:
            pair_data: 신경 쌍 메타데이터 리스트
            labels: 시냅스 존재 여부 (0/1)

        Returns:
            (특징 데이터프레임, 레이블 배열)
        """
        features_list = []

        for pair in pair_data:
            source_pos = pair.get('source_position')
            target_pos = pair.get('target_position')
            source_type = pair.get('source_type', 'unknown')
            target_type = pair.get('target_type', 'unknown')

            # 특징 계산
            features = self.compute_spatial_features(
                source_pos,
                target_pos,
                source_type,
                target_type
            )

            # 신경 쌍 ID 추가
            features['source_neuron_id'] = pair['source_neuron_id']
            features['target_neuron_id'] = pair['target_neuron_id']

            features_list.append(features)

        # DataFrame으로 변환
        df = pd.DataFrame(features_list)

        # 신경 타입 원-핫 인코딩
        source_type_dummies = pd.get_dummies(
            df['source_type'],
            prefix='source_type'
        )
        target_type_dummies = pd.get_dummies(
            df['target_type'],
            prefix='target_type'
        )

        # 원본 특징에서 타입 열 제거
        df = df.drop(['source_type', 'target_type'], axis=1)

        # 원-핫 인코딩된 특징 추가
        df = pd.concat([df, source_type_dummies, target_type_dummies], axis=1)

        # NaN 값 처리 (평균값으로 대체)
        df = df.fillna(df.mean(numeric_only=True))

        labels_array = np.array(labels)

        logger.info(f"특징 행렬 생성 완료: {df.shape[0]}개 샘플, {df.shape[1]}개 특징")

        return df, labels_array

    def normalize_features(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        특징을 정규화합니다.

        훈련 데이터의 통계를 사용하여 모든 데이터를 정규화합니다
        (데이터 누수 방지).

        Args:
            X_train: 훈련 특징 데이터프레임
            X_val: 검증 특징 데이터프레임
            X_test: 테스트 특징 데이터프레임

        Returns:
            정규화된 데이터 (numpy 배열)
        """
        if not self.normalize:
            return X_train.values, (X_val.values if X_val is not None else None), \
                   (X_test.values if X_test is not None else None)

        # 스케일러 선택
        if self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:  # 기본값: standard
            self.scaler = StandardScaler()

        # 훈련 데이터로 스케일러 학습
        X_train_normalized = self.scaler.fit_transform(X_train)

        # 검증/테스트 데이터 정규화
        X_val_normalized = self.scaler.transform(X_val) if X_val is not None else None
        X_test_normalized = self.scaler.transform(X_test) if X_test is not None else None

        logger.info(f"특징 정규화 완료 ({self.normalization_method})")

        return X_train_normalized, X_val_normalized, X_test_normalized

    def split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],  # (X_train, y_train)
        Tuple[np.ndarray, np.ndarray],  # (X_val, y_val)
        Tuple[np.ndarray, np.ndarray]   # (X_test, y_test)
    ]:
        """
        데이터를 Train/Val/Test로 분할합니다.

        분할 비율:
        - Train: (1 - test_size - val_size) * 100 %
        - Val: val_size * 100 %
        - Test: test_size * 100 %

        Args:
            X: 특징 데이터
            y: 레이블
            test_size: 테스트 세트 비율
            val_size: 검증 세트 비율

        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        # 1단계: Test 세트 분리
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # 클래스 분포 유지
        )

        # 2단계: Train/Val 분리
        # val_size를 train_val 데이터 기준으로 조정
        adjusted_val_size = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=adjusted_val_size,
            random_state=self.random_state,
            stratify=y_train_val
        )

        # 특징 정규화
        X_train_norm, X_val_norm, X_test_norm = self.normalize_features(
            X_train, X_val, X_test
        )

        logger.info(f"데이터 분할 완료:")
        logger.info(f"  Train: {X_train_norm.shape[0]} 샘플 (양성: {y_train.sum()})")
        logger.info(f"  Val: {X_val_norm.shape[0]} 샘플 (양성: {y_val.sum()})")
        logger.info(f"  Test: {X_test_norm.shape[0]} 샘플 (양성: {y_test.sum()})")

        return (X_train_norm, y_train), (X_val_norm, y_val), (X_test_norm, y_test)

    def handle_class_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'random_oversample'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        클래스 불균형 처리

        시냅스 존재(1)와 비존재(0)의 비율이 1:10 정도로 불균형합니다.
        이를 개선하는 방법:
        - random_oversample: 소수 클래스 오버샘플링
        - random_undersample: 다수 클래스 언더샘플링

        Args:
            X: 특징 데이터
            y: 레이블
            method: 처리 방법

        Returns:
            (균형잡힌 X, 균형잡힌 y)
        """
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"원본 클래스 분포: {dict(zip(unique, counts))}")

        # 양성과 음성 클래스 분리
        idx_positive = np.where(y == 1)[0]
        idx_negative = np.where(y == 0)[0]

        X_positive = X[idx_positive]
        X_negative = X[idx_negative]

        if method == 'random_oversample':
            # 소수 클래스(양성) 오버샘플링
            n_samples = len(idx_negative)
            indices = np.random.choice(len(idx_positive), size=n_samples, replace=True)
            X_positive_resampled = X_positive[indices]
            y_positive_resampled = np.ones(n_samples)

            X_balanced = np.vstack([X_negative, X_positive_resampled])
            y_balanced = np.hstack([np.zeros(len(idx_negative)), y_positive_resampled])

        elif method == 'random_undersample':
            # 다수 클래스(음성) 언더샘플링
            n_samples = len(idx_positive)
            indices = np.random.choice(len(idx_negative), size=n_samples, replace=False)
            X_negative_resampled = X_negative[indices]
            y_negative_resampled = np.zeros(n_samples)

            X_balanced = np.vstack([X_positive, X_negative_resampled])
            y_balanced = np.hstack([np.ones(len(idx_positive)), y_negative_resampled])

        else:
            raise ValueError(f"지원하지 않는 방법: {method}")

        unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
        logger.info(f"처리 후 클래스 분포: {dict(zip(unique_balanced, counts_balanced))}")

        return X_balanced, y_balanced

    def save_processed_data(
        self,
        save_dir: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        전처리된 데이터를 파일로 저장

        Args:
            save_dir: 저장 디렉토리
            X_train, y_train: 훈련 데이터
            X_val, y_val: 검증 데이터
            X_test, y_test: 테스트 데이터
            feature_names: 특징 이름 리스트
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 데이터 저장
        np.savez(save_path / 'train_data.npz', X=X_train, y=y_train)
        np.savez(save_path / 'val_data.npz', X=X_val, y=y_val)
        np.savez(save_path / 'test_data.npz', X=X_test, y=y_test)

        # 특징 이름 저장
        if feature_names:
            np.save(save_path / 'feature_names.npy', feature_names)

        logger.info(f"전처리된 데이터 저장 완료: {save_path}")

    @staticmethod
    def load_processed_data(
        save_dir: str
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """
        저장된 전처리 데이터 로드

        Args:
            save_dir: 데이터 저장 디렉토리

        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        load_path = Path(save_dir)

        train_data = np.load(load_path / 'train_data.npz')
        val_data = np.load(load_path / 'val_data.npz')
        test_data = np.load(load_path / 'test_data.npz')

        return (
            (train_data['X'], train_data['y']),
            (val_data['X'], val_data['y']),
            (test_data['X'], test_data['y'])
        )
