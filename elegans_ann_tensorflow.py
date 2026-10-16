"""
C. elegans 인공신경망(ANN) - TensorFlow 버전
C. elegans 생물학적 데이터를 분석하기 위한 완전한 신경망 구현입니다.
지원: 유전자 발현, 연결체(connectome) 분석, 행동 예측, 단백질 상호작용.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics, callbacks
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.train_test_split import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime


# ============================================================================
# 설정 관리
# ============================================================================
class Config:
    """YAML 또는 기본값에서 설정을 로드하고 관리합니다."""
    
    DEFAULT_CONFIG = {
        'model': {
            'input_size': 100,
            'hidden_sizes': [512, 256, 128],
            'output_size': 1,
            'activation': 'relu',  # relu, tanh, sigmoid
            'dropout': 0.3,
            'batch_norm': True,
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',  # adam, sgd, rmsprop
            'loss_fn': 'mse',  # mse, mae, cross_entropy
            'early_stopping_patience': 10,
            'weight_decay': 1e-5,
        },
        'data': {
            'test_split': 0.2,
            'validation_split': 0.1,
            'normalize': True,
            'random_seed': 42,
        },
        'paths': {
            'data_dir': 'data/',
            'results_dir': 'results/',
            'config_file': 'config/config.yaml',
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._deep_update(self.config, user_config)
    
    @staticmethod
    def _deep_update(d: Dict, u: Dict) -> None:
        """사전 d를 재귀적으로 u로 업데이트합니다."""
        for k, v in u.items():
            if isinstance(v, dict):
                if k not in d:
                    d[k] = {}
                d[k].update(v)
            else:
                d[k] = v
    
    def __getitem__(self, key):
        return self.config[key]
    
    def get(self, key, default=None):
        return self.config.get(key, default)


# ============================================================================
# 데이터셋 생성 (합성 C. elegans 데이터)
# ============================================================================
class CElegansDataset:
    """
    C. elegans 생물학적 데이터를 위한 합성 데이터셋.
    생성 항목:
    - 유전자 발현 패턴
    - 연결체(connectome) 연결 정보
    - 행동 특성
    - 단백질 상호작용
    """
    
    def __init__(self, n_samples: int = 1000, input_size: int = 100, 
                 output_size: int = 1, task: str = 'regression', seed: int = 42):
        """
        Args:
            n_samples: 샘플 수
            input_size: 입력 특성 차원 (유전자/뉴런)
            output_size: 출력 차원 (예측 대상)
            task: 'regression' 또는 'classification'
            seed: 랜덤 시드
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.n_samples = n_samples
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        
        self.X, self.y = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """합성 C. elegans 생물학적 데이터를 생성합니다."""
        # 유전자 발현 패턴 (로그정규 분포)
        X = np.random.lognormal(mean=0, sigma=1.5, size=(self.n_samples, self.input_size))
        X = X / (X.max(axis=0) + 1e-8)  # [0, 1]로 정규화
        
        # 비선형 관계로 타깃 변수 생성
        weights = np.random.randn(self.input_size)
        y = np.dot(X, weights) + np.random.randn(self.input_size) @ np.random.randn(self.input_size, self.output_size)
        
        if self.task == 'classification':
            y = (y > y.median(axis=0)).astype(np.float32)
        else:
            y = y.astype(np.float32)
        
        return X.astype(np.float32), y
    
    def get_train_test_split(self, test_size: float = 0.2, val_size: float = 0.1):
        """학습/검증/테스트 세트로 데이터를 분할합니다."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size), random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================================
# 신경망 모델 빌더
# ============================================================================
class CElegansANNBuilder:
    """
    TensorFlow/Keras를 이용한 C. elegans ANN 빌더.
    드롭아웃과 배치 정규화를 적용할 수 있는 구성 가능 아키텍처.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self) -> models.Sequential:
        """순차 신경망 모델을 구성합니다."""
        model_cfg = self.config['model']
        input_size = model_cfg['input_size']
        hidden_sizes = model_cfg['hidden_sizes']
        output_size = model_cfg['output_size']
        activation = model_cfg['activation'].lower()
        dropout = model_cfg.get('dropout', 0.3)
        use_batch_norm = model_cfg.get('batch_norm', True)
        
        model = models.Sequential()
        
        # 입력 및 첫 번째 은닉층
        model.add(layers.Input(shape=(input_size,)))
        
        # 은닉층들
        for hidden_size in hidden_sizes:
            model.add(layers.Dense(hidden_size, activation=activation))
            if use_batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # 출력층
        model.add(layers.Dense(output_size))
        
        return model
    
    def get_model(self) -> models.Sequential:
        """구성된 모델을 반환합니다."""
        return self.model


# ============================================================================
# 학습 엔진
# ============================================================================
class Trainer:
    """신경망을 학습하고 평가합니다."""
    
    def __init__(self, model: models.Sequential, config: Config, 
                 verbose: int = 1):
        self.model = model
        self.config = config
        self.verbose = verbose
        
        self.optimizer = self._get_optimizer(config['training'])
        self.loss_fn = self._get_loss_fn(config['training'])
        self.scaler = StandardScaler()
        
        self.history = None
    
    def _get_optimizer(self, train_cfg: Dict):
        """설정에 따라 옵티마이저를 반환합니다."""
        optimizer_name = train_cfg.get('optimizer', 'adam').lower()
        lr = train_cfg['learning_rate']
        weight_decay = train_cfg.get('weight_decay', 0)
        
        if optimizer_name == 'adam':
            return optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optimizers.SGD(learning_rate=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            return optimizers.RMSprop(learning_rate=lr, weight_decay=weight_decay)
        else:
            return optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
    
    def _get_loss_fn(self, train_cfg: Dict):
        """설정에 따라 손실 함수를 반환합니다."""
        loss_name = train_cfg.get('loss_fn', 'mse').lower()
        
        if loss_name == 'mae':
            return losses.MeanAbsoluteError()
        elif loss_name == 'cross_entropy':
            return losses.BinaryCrossentropy()
        else:
            return losses.MeanSquaredError()
    
    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """데이터를 정규화합니다."""
        if self.config['data']['normalize']:
            X = self.scaler.fit_transform(X)
        return X.astype(np.float32)
    
    def compile_model(self):
        """모델을 컴파일합니다."""
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['mae', 'mse']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
        """모델을 학습합니다."""
        
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        
        # 얼리 스톱핑 콜백
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=self.verbose
        )
        
        # 학습률 감소 콜백
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=self.verbose
        )
        
        # 모델 학습
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=self.verbose
        )
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """테스트 세트에서 평가합니다."""
        
        # 모델 예측 얻기
        predictions = self.model.predict(X_test, verbose=0)
        
        # 지표 계산
        test_loss = float(self.model.evaluate(X_test, y_test, verbose=0)[0])
        test_mse = mean_squared_error(y_test, predictions)
        test_mae = mean_absolute_error(y_test, predictions)
        test_r2 = r2_score(y_test, predictions)
        
        results = {
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'predictions': predictions,
            'targets': y_test
        }
        
        return results


# ============================================================================
# 시각화 및 결과
# ============================================================================
class ResultsVisualizer:
    """학습 이력과 예측을 시각화합니다."""
    
    @staticmethod
    def plot_training_history(history: keras.callbacks.History, 
                             save_path: Optional[str] = None):
        """학습 및 검증 손실을 플롯합니다."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(history.history['loss'], label='학습 손실', linewidth=2)
        ax.plot(history.history['val_loss'], label='검증 손실', linewidth=2)
        
        ax.set_xlabel('에포크', fontsize=12)
        ax.set_ylabel('손실', fontsize=12)
        ax.set_title('C. elegans ANN 학습 이력', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_metrics_history(history: keras.callbacks.History,
                            save_path: Optional[str] = None):
        """MAE 및 MSE 이력을 플롯합니다."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # MAE
        ax1.plot(history.history['mae'], label='학습 MAE', linewidth=2)
        ax1.plot(history.history['val_mae'], label='검증 MAE', linewidth=2)
        ax1.set_xlabel('에포크', fontsize=12)
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_title('평균 절대 오차(MAE)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # MSE
        ax2.plot(history.history['mse'], label='학습 MSE', linewidth=2)
        ax2.plot(history.history['val_mse'], label='검증 MSE', linewidth=2)
        ax2.set_xlabel('에포크', fontsize=12)
        ax2.set_ylabel('MSE', fontsize=12)
        ax2.set_title('평균 제곱 오차(MSE)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_predictions(targets: np.ndarray, predictions: np.ndarray, 
                        save_path: Optional[str] = None):
        """타깃과 예측을 비교하여 플롯합니다."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 산점도
        ax1.scatter(targets, predictions, alpha=0.5, s=20)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='완벽한 예측')
        ax1.set_xlabel('타깃 값', fontsize=12)
        ax1.set_ylabel('예측 값', fontsize=12)
        ax1.set_title('예측 vs 타깃', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 잔차 분포
        residuals = targets.flatten() - predictions.flatten()
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('잔차', fontsize=12)
        ax2.set_ylabel('빈도', fontsize=12)
        ax2.set_title('잔차 분포', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_model_architecture(model: models.Sequential, save_path: Optional[str] = None):
        """모델 아키텍처를 플롯합니다."""
        try:
            fig = plt.figure(figsize=(10, 8))
            keras.utils.plot_model(
                model, 
                to_file=save_path if save_path else None,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=100
            )
            if save_path:
                plt.close()
        except Exception as e:
            print(f"모델 아키텍처를 플롯할 수 없습니다: {e}")


# ============================================================================
# 메인 파이프라인
# ============================================================================
def main():
    """TensorFlow를 사용한 전체 C. elegans ANN 파이프라인."""
    
    print("=" * 70)
    print("C. elegans 인공신경망(ANN) - TensorFlow 버전")
    print("=" * 70)
    
    # 설정
    config = Config()
    print(f"\n사용 중인 TensorFlow 버전: {tf.__version__}")
    print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # 결과 디렉토리 생성
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # 합성 C. elegans 데이터 생성
    print("\n[1] 합성 C. elegans 데이터셋 생성 중...")
    dataset = CElegansDataset(
        n_samples=1000,
        input_size=config['model']['input_size'],
        output_size=config['model']['output_size'],
        task='regression',
        seed=config['data']['random_seed']
    )
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.get_train_test_split(
        test_size=config['data']['test_split'],
        val_size=config['data']['validation_split']
    )
    
    print(f"  학습 세트: {X_train.shape}")
    print(f"  검증 세트: {X_val.shape}")
    print(f"  테스트 세트: {X_test.shape}")
    
    # 모델 빌더 및 트레이너 초기화
    print("\n[2] 신경망 모델 구성 중...")
    builder = CElegansANNBuilder(config)
    model = builder.get_model()
    
    print(f"  모델 요약:")
    model.summary()
    
    trainer = Trainer(model, config, verbose=1)
    
    # 모델 컴파일
    print("\n[3] 모델 컴파일 중...")
    trainer.compile_model()
    print("  모델이 성공적으로 컴파일되었습니다")
    
    # 데이터 준비
    print("\n[4] 데이터 준비 (정규화)...")
    X_train_norm = trainer.prepare_data(X_train)
    X_val_norm = trainer.prepare_data(X_val)
    X_test_norm = trainer.prepare_data(X_test)
    
    y_train = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
    y_val = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
    y_test = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test
    
    print("  데이터가 정규화되고 형태가 조정되었습니다")
    
    # 모델 학습
    print("\n[5] 모델 학습 중...")
    history = trainer.fit(X_train_norm, y_train, X_val_norm, y_val)
    
    # 테스트 세트에서 평가
    print("\n[6] 테스트 세트에서 평가 중...")
    test_results = trainer.evaluate(X_test_norm, y_test)
    print(f"  테스트 손실: {test_results['test_loss']:.6f}")
    print(f"  테스트 MSE: {test_results['test_mse']:.6f}")
    print(f"  테스트 MAE: {test_results['test_mae']:.6f}")
    print(f"  테스트 R²: {test_results['test_r2']:.6f}")
    
    # 결과 저장
    print("\n[7] 결과 저장 중...")
    
    # 모델 저장
    model_path = results_dir / 'elegans_ann_model.keras'
    model.save(model_path)
    print(f"  모델이 저장됨: {model_path}")
    
    # 모델 가중치 저장
    weights_path = results_dir / 'elegans_ann_weights.h5'
    model.save_weights(weights_path)
    print(f"  모델 가중치가 저장됨: {weights_path}")
    
    # 학습 이력 저장
    history_path = results_dir / 'training_history.json'
    history_data = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'mse': [float(x) for x in history.history['mse']],
        'val_mse': [float(x) for x in history.history['val_mse']],
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  학습 이력이 저장됨: {history_path}")
    
    # 테스트 결과 저장
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'test_loss': float(test_results['test_loss']),
            'test_mse': float(test_results['test_mse']),
            'test_mae': float(test_results['test_mae']),
            'test_r2': float(test_results['test_r2']),
        }, f, indent=2)
    print(f"  결과가 저장됨: {results_path}")
    
    # 설정 저장
    config_path = results_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config.config, f, indent=2)
    print(f"  설정이 저장됨: {config_path}")
    
    # 시각화 생성
    print("\n[8] 시각화 생성 중...")
    
    viz = ResultsVisualizer()
    
    loss_plot = results_dir / 'training_history.png'
    viz.plot_training_history(history, save_path=loss_plot)
    print(f"  학습 이력 플롯 저장됨: {loss_plot}")
    
    metrics_plot = results_dir / 'metrics_history.png'
    viz.plot_metrics_history(history, save_path=metrics_plot)
    print(f"  지표 이력 플롯 저장됨: {metrics_plot}")
    
    pred_plot = results_dir / 'predictions.png'
    viz.plot_predictions(y_test, test_results['predictions'], 
                        save_path=pred_plot)
    print(f"  예측 플롯 저장됨: {pred_plot}")
    
    # 요약 리포트
    print("\n" + "=" * 70)
    print("학습 완료 - 요약 리포트")
    print("=" * 70)
    print(f"타임스탬프: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 에포크 수: {len(history.history['loss'])}")
    print(f"최저 검증 손실: {min(history.history['val_loss']):.6f}")
    print(f"\n테스트 세트 성능:")
    print(f"  손실: {test_results['test_loss']:.6f}")
    print(f"  MSE:  {test_results['test_mse']:.6f}")
    print(f"  MAE:  {test_results['test_mae']:.6f}")
    print(f"  R²:   {test_results['test_r2']:.6f}")
    print(f"\n모델 구성:")
    print(f"  입력 크기: {config['model']['input_size']}")
    print(f"  은닉층: {config['model']['hidden_sizes']}")
    print(f"  출력 크기: {config['model']['output_size']}")
    print(f"  활성화 함수: {config['model']['activation']}")
    print(f"  드롭아웃: {config['model']['dropout']}")
    print(f"\n결과가 저장된 위치: {results_dir.absolute()}")
    print("=" * 70)


if __name__ == '__main__':
    main()