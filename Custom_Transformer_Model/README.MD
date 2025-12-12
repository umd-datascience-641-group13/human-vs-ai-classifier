# Custom Transformer Model

This folder contains the implementation of a lightweight Transformer architecture used to classify text as either Human, 0, or AI generated, 1.

## Files

### `transformer_model.py`
A PyTorch implementation of an encoder-only Transformer model, inspired by "Attention Is All You Need".  
Key features include:
- Learned token embeddings
- Sinusoidal positional encoding
- Multi-head self-attention encoder layers
- Feed-forward projection layers
- Classification head with 2 outputs
- Cross-entropy training objective

This model is significantly smaller than RoBERTa and trains much more quickly.

- Automatically runs the preprocessing script 
- Splits into train/val/test sets
- Trains for N epochs (default = 10)
- Saves:
  - `model.pt` (PyTorch weights)
  - `config.json`
  - `vocab.json`

## Output Labels
0 = Human 
1 = AI

