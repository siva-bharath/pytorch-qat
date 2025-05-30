# Quantization-Aware Training (QAT) for MobileNetV2

This repository implements Quantization-Aware Training (QAT) for MobileNetV2 on the Flowers102 dataset using PyTorch and Weights & Biases for hyperparameter optimization.

## Features

- Quantization-Aware Training implementation
- Hyperparameter optimization using Weights & Biases
- Early stopping with Hyperband algorithm
- Support for multiple optimizers (SGD, AdamW, Adam)
- Learning rate scheduling (Cosine, Step)
- Data augmentation strategies
- Model size and performance tracking

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- Weights & Biases account

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pytorch-qat.git
cd pytorch-qat
```

2. Create and activate virtual environment:
```bash
python -m venv qat_env
source qat_env/bin/activate  # On Windows: qat_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Login to Weights & Biases:
```bash
wandb login
```

## Usage

1. Prepare your data:
   - Download the Flowers102 dataset
   - Place it in the `data/flowers102` directory

2. Run the training:
```bash
python QatWbHpo.py
```

## Configuration

The hyperparameter search space is defined in `sweep_config.yaml`. You can modify:
- Optimizer settings
- Learning rates
- Batch sizes
- Early stopping parameters
- And more

## Project Structure

```
pytorch-qat/
├── QatWbHpo.py           # Main training script
├── sweep_config.yaml     # W&B sweep configuration
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── data/                # Data directory
    └── flowers102/      # Flowers102 dataset
```

## Results

The training results can be viewed in the Weights & Biases dashboard, including:
- Model accuracy
- Training/validation metrics
- Model size reduction
- Quantization effects

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 