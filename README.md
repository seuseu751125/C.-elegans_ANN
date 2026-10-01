# C. elegans Artificial Neural Network (ANN)

A machine learning project to build and train artificial neural networks for analyzing *Caenorhabditis elegans* (C. elegans) biological data.

## Project Overview

This project implements neural networks using PyTorch to model and predict various aspects of C. elegans biology, including:
- Gene expression patterns
- Connectome analysis
- Behavioral prediction
- Protein interactions

## Project Structure

```
C.-elegans_ANN/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── datasets.py
├── tests/
│   ├── __init__.py
│   └── test_model.py
├── .gitignore
├── LICENSE
├── results/
│   └── .gitkeep
└── notebooks/
    └── .gitkeep
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/seuseu751125/C.-elegans_ANN.git
cd C.-elegans_ANN
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training a model

```bash
python src/train.py --config config/config.yaml
```

### Evaluating a model

```bash
python src/evaluate.py --model results/models/final_model.pth
```

### Testing

```bash
pytest -q
```

## Model Architecture

### Baseline ANN
- Input size: configurable
- Hidden layers: configurable, default `[512, 256, 128]`
- Output size: configurable
- Activation: ReLU / Tanh / Sigmoid
- Optimizer: Adam
- Loss: CrossEntropyLoss or MSELoss

## Requirements

Key dependencies:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyYAML

## License

This project is licensed under the MIT License.

## Author

- seuseu751125
