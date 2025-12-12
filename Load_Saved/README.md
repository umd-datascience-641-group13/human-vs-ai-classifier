# Load Saved Models (Local Training)

This folder contains scripts for loading **your own locally trained models**.

## Files

### `Load_Custom_Transformer.py`
Loads:
- `model.pt`
- `config.json`
- `vocab.json`

Reconstructs the custom transformer model and runs:
- Classification on validation data
- Confusion matrix
- Precision / Recall / F1-score
- Optional ROC/AUC plot

### `Load_Roberta.py`
Loads the fine-tuned RoBERTa model from:
- saved_roberta
Runs evaluation identical to the custom loader.

## When To Use These Scripts

Use these when:
- You trained models yourself  


