# GPT Model

This project includes a custom implementation of the decoder stack of GPT model using PyTorch. The model is designed for sequence generation tasks and leverages causal self-attention, layer normalization, multi-layer perceptrons (MLP), and custom configurations to achieve its objectives. The implementation allows for significant customization, including adjustments to block size, vocabulary size, layer counts, head counts, embedding dimensions, and dropout rates.

## Features
- Implementation of causal self-attention, MLP, and transformer blocks.
- Support for generating text sequences given a prefix.
- Efficient weight initialization and parameter counting utilities.
- Compatibility check for using fast attention mechanisms with PyTorch versions >= 2.0.

## Requirements
- Python 3.6 or later

## Installation
To set up your environment to run the code, clone the repository and install the requirements (PyTorch):

```bash
pip install -r requirements.txt
```

## Usage
To start the training, you can use the sample training script that uses the tinyshakespeare text file to train the GPT model. 

Use the following for training on single GPU/CPU:
```
python train.py
```

If you have multiple GPUs on a single cluster/node, use following for training using Distributed Data-Parallel (DDP):
```
torchrun --standalone --nproc_per_node=gpu train.py
```

To generate text sequences, run the following script to get an 'output.txt' file:
```python
python generate.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
