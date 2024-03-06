# GPT Model

This project includes a custom implementation of the decoder stack of GPT model using PyTorch. The model is designed for sequence generation tasks and leverages causal self-attention, layer normalization, multi-layer perceptrons (MLP), and custom configurations to achieve its objectives. The implementation allows for significant customization, including adjustments to block size, vocabulary size, layer counts, head counts, embedding dimensions, and dropout rates.

## Features
- Implementation of causal self-attention, MLP, and transformer blocks.
- Support for generating text sequences given a prefix.
- Efficient weight initialization and parameter counting utilities.
- Compatibility check for using fast attention mechanisms with PyTorch versions >= 2.0.

## Requirements
- Python 3.6 or later
- PyTorch 1.8.0 or later (For optimal performance, PyTorch 2.0 or later is recommended for utilizing fast attention mechanisms.)

## Installation
To set up your environment to run the code, you will need Python and PyTorch installed. It is recommended to use a virtual environment:

```bash
python3 -m venv gpt-venv
source gpt-venv/bin/activate
pip install torch torchvision
```

## Usage
To use the GPT model in your project, first import the necessary modules and initialize the model with your desired configuration:

```python
from model import GPT, GPTParameter

config = GPTParameter(
    block_size=128,  # Example block size
    vocab_size=50257,  # Vocabulary size
    num_layer=12,  # Number of transformer layers
    num_head=12,  # Number of attention heads
    num_embd=768,  # Embedding dimension
    dropout=0.1,  # Dropout rate
    bias=True  # Use bias in layers
)

model = GPT(config)
```

For generating text sequences, use the `generate` method with a provided token prefix:

```python
prefix_tokens = torch.tensor([[token_ids]], dtype=torch.long)  # Token IDs as a tensor
generated_sequence = model.generate(prefix_tokens, max_len=50)  # Generate sequence with max length of 50 tokens
```


## License
This project is licensed under the MIT License - see the LICENSE file for details.