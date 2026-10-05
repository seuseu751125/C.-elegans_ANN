"""
C. elegans Artificial Neural Network (ANN) - TensorFlow Version
A complete neural network implementation for analyzing C. elegans biological data.
Supports: gene expression, connectome analysis, behavioral prediction, protein interactions.
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
# Configuration Management
# ============================================================================
class Config:
    """Load and manage configuration from YAML or defaults."""
    
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
        """Recursively update dictionary d with u."""
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
# Dataset Generation (Synthetic C. elegans Data)
# ============================================================================
class CElegansDataset:
    """
    Synthetic dataset for C. elegans biological data.
    Generates data for:
    - Gene expression patterns
    - Connectome connectivity
    - Behavioral features
    - Protein interactions
    """
    
    def __init__(self, n_samples: int = 1000, input_size: int = 100, 
                 output_size: int = 1, task: str = 'regression', seed: int = 42):
        """
        Args:
            n_samples: Number of samples
            input_size: Input feature dimension (genes/neurons)
            output_size: Output dimension (prediction target)
            task: 'regression' or 'classification'
            seed: Random seed
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.n_samples = n_samples
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        
        self.X, self.y = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic C. elegans biological data."""
        # Gene expression patterns (lognormal distribution)
        X = np.random.lognormal(mean=0, sigma=1.5, size=(self.n_samples, self.input_size))
        X = X / (X.max(axis=0) + 1e-8)  # Normalize to [0, 1]
        
        # Create target variable with nonlinear relationship
        weights = np.random.randn(self.input_size)
        y = np.dot(X, weights) + np.random.randn(self.input_size) @ np.random.randn(self.input_size, self.output_size)
        
        if self.task == 'classification':
            y = (y > y.median(axis=0)).astype(np.float32)
        else:
            y = y.astype(np.float32)
        
        return X.astype(np.float32), y
    
    def get_train_test_split(self, test_size: float = 0.2, val_size: float = 0.1):
        """Split data into train, validation, and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size), random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================================
# Neural Network Model Builder
# ============================================================================
class CElegansANNBuilder:
    """
    Builder for C. elegans ANN using TensorFlow/Keras.
    Configurable architecture with dropout and batch normalization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self) -> models.Sequential:
        """Build sequential neural network model."""
        model_cfg = self.config['model']
        input_size = model_cfg['input_size']
        hidden_sizes = model_cfg['hidden_sizes']
        output_size = model_cfg['output_size']
        activation = model_cfg['activation'].lower()
        dropout = model_cfg.get('dropout', 0.3)
        use_batch_norm = model_cfg.get('batch_norm', True)
        
        model = models.Sequential()
        
        # Input and first hidden layer
        model.add(layers.Input(shape=(input_size,)))
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            model.add(layers.Dense(hidden_size, activation=activation))
            if use_batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # Output layer
        model.add(layers.Dense(output_size))
        
        return model
    
    def get_model(self) -> models.Sequential:
        """Return the built model."""
        return self.model


# ============================================================================
# Training Engine
# ============================================================================
class Trainer:
    """Train and evaluate the neural network."""
    
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
        """Get optimizer based on configuration."""
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
        """Get loss function based on configuration."""
        loss_name = train_cfg.get('loss_fn', 'mse').lower()
        
        if loss_name == 'mae':
            return losses.MeanAbsoluteError()
        elif loss_name == 'cross_entropy':
            return losses.BinaryCrossentropy()
        else:
            return losses.MeanSquaredError()
    
    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize data."""
        if self.config['data']['normalize']:
            X = self.scaler.fit_transform(X)
        return X.astype(np.float32)
    
    def compile_model(self):
        """Compile the model."""
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['mae', 'mse']
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
        """Train the model."""
        
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        
        # Early stopping callback
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=self.verbose
        )
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=self.verbose
        )
        
        # Train model
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
        """Evaluate on test set."""
        
        # Get model predictions
        predictions = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
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
# Visualization & Results
# ============================================================================
class ResultsVisualizer:
    """Visualize training history and predictions."""
    
    @staticmethod
    def plot_training_history(history: keras.callbacks.History, 
                             save_path: Optional[str] = None):
        """Plot training and validation loss."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('C. elegans ANN Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_metrics_history(history: keras.callbacks.History,
                            save_path: Optional[str] = None):
        """Plot MAE and MSE history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # MAE
        ax1.plot(history.history['mae'], label='Train MAE', linewidth=2)
        ax1.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # MSE
        ax2.plot(history.history['mse'], label='Train MSE', linewidth=2)
        ax2.plot(history.history['val_mse'], label='Validation MSE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MSE', fontsize=12)
        ax2.set_title('Mean Squared Error', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_predictions(targets: np.ndarray, predictions: np.ndarray, 
                        save_path: Optional[str] = None):
        """Plot predictions vs targets."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1.scatter(targets, predictions, alpha=0.5, s=20)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Target Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title('Predictions vs Targets', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = targets.flatten() - predictions.flatten()
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_model_architecture(model: models.Sequential, save_path: Optional[str] = None):
        """Plot model architecture."""
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
            print(f"Could not plot model architecture: {e}")


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Complete C. elegans ANN pipeline using TensorFlow."""
    
    print("=" * 70)
    print("C. elegans Artificial Neural Network (ANN) - TensorFlow Version")
    print("=" * 70)
    
    # Configuration
    config = Config()
    print(f"\nUsing TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create results directory
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Generate synthetic C. elegans data
    print("\n[1] Generating synthetic C. elegans dataset...")
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
    
    print(f"  Train set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Initialize model builder and trainer
    print("\n[2] Building neural network model...")
    builder = CElegansANNBuilder(config)
    model = builder.get_model()
    
    print(f"  Model Summary:")
    model.summary()
    
    trainer = Trainer(model, config, verbose=1)
    
    # Compile model
    print("\n[3] Compiling model...")
    trainer.compile_model()
    print("  Model compiled successfully")
    
    # Prepare data
    print("\n[4] Preparing data (normalization)...")
    X_train_norm = trainer.prepare_data(X_train)
    X_val_norm = trainer.prepare_data(X_val)
    X_test_norm = trainer.prepare_data(X_test)
    
    y_train = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
    y_val = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
    y_test = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test
    
    print("  Data normalized and reshaped")
    
    # Train model
    print("\n[5] Training model...")
    history = trainer.fit(X_train_norm, y_train, X_val_norm, y_val)
    
    # Evaluate on test set
    print("\n[6] Evaluating on test set...")
    test_results = trainer.evaluate(X_test_norm, y_test)
    print(f"  Test Loss: {test_results['test_loss']:.6f}")
    print(f"  Test MSE: {test_results['test_mse']:.6f}")
    print(f"  Test MAE: {test_results['test_mae']:.6f}")
    print(f"  Test R²: {test_results['test_r2']:.6f}")
    
    # Save results
    print("\n[7] Saving results...")
    
    # Save model
    model_path = results_dir / 'elegans_ann_model.keras'
    model.save(model_path)
    print(f"  Model saved to: {model_path}")
    
    # Save model weights
    weights_path = results_dir / 'elegans_ann_weights.h5'
    model.save_weights(weights_path)
    print(f"  Model weights saved to: {weights_path}")
    
    # Save training history
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
    print(f"  History saved to: {history_path}")
    
    # Save test results
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'test_loss': float(test_results['test_loss']),
            'test_mse': float(test_results['test_mse']),
            'test_mae': float(test_results['test_mae']),
            'test_r2': float(test_results['test_r2']),
        }, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Save configuration
    config_path = results_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config.config, f, indent=2)
    print(f"  Configuration saved to: {config_path}")
    
    # Generate visualizations
    print("\n[8] Generating visualizations...")
    
    viz = ResultsVisualizer()
    
    loss_plot = results_dir / 'training_history.png'
    viz.plot_training_history(history, save_path=loss_plot)
    print(f"  Training history plot saved to: {loss_plot}")
    
    metrics_plot = results_dir / 'metrics_history.png'
    viz.plot_metrics_history(history, save_path=metrics_plot)
    print(f"  Metrics history plot saved to: {metrics_plot}")
    
    pred_plot = results_dir / 'predictions.png'
    viz.plot_predictions(y_test, test_results['predictions'], 
                        save_path=pred_plot)
    print(f"  Predictions plot saved to: {pred_plot}")
    
    # Summary Report
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY REPORT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Epochs: {len(history.history['loss'])}")
    print(f"Best Validation Loss: {min(history.history['val_loss']):.6f}")
    print(f"\nTest Set Performance:")
    print(f"  Loss: {test_results['test_loss']:.6f}")
    print(f"  MSE:  {test_results['test_mse']:.6f}")
    print(f"  MAE:  {test_results['test_mae']:.6f}")
    print(f"  R²:   {test_results['test_r2']:.6f}")
    print(f"\nModel Configuration:")
    print(f"  Input Size: {config['model']['input_size']}")
    print(f"  Hidden Layers: {config['model']['hidden_sizes']}")
    print(f"  Output Size: {config['model']['output_size']}")
    print(f"  Activation: {config['model']['activation']}")
    print(f"  Dropout: {config['model']['dropout']}")
    print(f"\nResults saved in: {results_dir.absolute()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
