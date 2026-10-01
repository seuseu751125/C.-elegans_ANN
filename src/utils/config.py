"""Configuration management for the project"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    프로젝트 전체 설정을 관리하는 클래스.

    YAML 파일에서 설정을 읽거나, 기본 설정을 제공합니다.
    사용법:
        config = Config()
        print(config.data_dir)  # 데이터 디렉토리 경로
        print(config.get('model.hidden_dim'))  # 중첩 설정 접근
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        설정 초기화

        Args:
            config_file: YAML 설정 파일 경로. None이면 기본 설정 사용
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dict = self._get_default_config()

        if config_file:
            user_config = self._load_yaml(config_file)
            self.config_dict.update(user_config)

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정값 반환"""
        return {
            # 데이터 경로
            'data': {
                'raw_dir': str(self.project_root / 'data' / 'raw'),
                'processed_dir': str(self.project_root / 'data' / 'processed'),
                'connectome_file': str(self.project_root / 'data' / 'raw' / 'connectome.json'),
            },

            # 모델 저장 경로
            'model': {
                'save_dir': str(self.project_root / 'models'),
                'checkpoint_dir': str(self.project_root / 'models' / 'checkpoints'),
            },

            # 보고서 경로
            'report': {
                'output_dir': str(self.project_root / 'reports'),
            },

            # 데이터 전처리 설정
            'preprocessing': {
                'test_size': 0.15,
                'val_size': 0.15,
                'random_state': 42,
                'normalize': True,
                'normalization_method': 'standard',  # 'standard' 또는 'minmax'
                'handle_imbalance': True,
                'imbalance_method': 'random_oversample',  # 또는 'random_undersample'
            },

            # 훈련 설정
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'early_stopping_patience': 10,
                'validation_split': 0.2,
                'random_seed': 42,
            },

            # 신경망 모델 설정
            'model_nn': {
                'input_dim': None,  # 특징 개수 (자동 계산됨)
                'hidden_dims': [128, 64, 32],
                'dropout_rate': 0.3,
                'activation': 'relu',
                'optimizer': 'adam',
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', 'AUC'],
            },

            # XGBoost 모델 설정
            'model_xgb': {
                'n_estimators': 200,
                'max_depth': 7,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 3,
                'random_state': 42,
            },

            # 평가 설정
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'save_plots': True,
                'plot_dir': str(self.project_root / 'reports' / 'figures'),
            },

            # 로깅 설정
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            }
        }

    def _load_yaml(self, config_file: str) -> Dict[str, Any]:
        """YAML 파일에서 설정 로드"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값 조회

        Args:
            key: 설정 키 (예: 'data.raw_dir', 'model_nn.hidden_dims')
            default: 키를 찾을 수 없을 때 반환할 기본값

        Returns:
            설정값
        """
        keys = key.split('.')
        value = self.config_dict

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def __getattr__(self, name: str) -> Any:
        """설정값에 속성으로 접근 (예: config.data_dir)"""
        if name.startswith('_'):
            return super().__getattribute__(name)

        # snake_case를 config dict의 키로 변환 시도
        if name in self.config_dict:
            return self.config_dict[name]

        raise AttributeError(f"설정에 '{name}' 속성이 없습니다")

    def to_dict(self) -> Dict[str, Any]:
        """모든 설정을 딕셔너리로 반환"""
        return self.config_dict.copy()

    def __repr__(self) -> str:
        """설정 객체의 문자열 표현"""
        return f"Config({self.project_root})"


# 전역 설정 객체 (편의용)
_global_config = None


def get_config(config_file: Optional[str] = None) -> Config:
    """전역 설정 객체 반환 (싱글톤)"""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_file)
    return _global_config


# 주요 경로를 쉽게 접근하기 위한 헬퍼 함수들
def get_data_dir() -> str:
    """데이터 디렉토리 경로"""
    return get_config().get('data.raw_dir')


def get_model_dir() -> str:
    """모델 저장 디렉토리"""
    return get_config().get('model.save_dir')


def get_processed_data_dir() -> str:
    """전처리된 데이터 디렉토리"""
    return get_config().get('data.processed_dir')
