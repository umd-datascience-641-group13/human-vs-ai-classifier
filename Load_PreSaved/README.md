# Load Pre-Saved Models (From Google Drive)

This folder contains scripts for loading **pretrained** (10-epoch) models hosted on Google Drive.

These scripts are far more convenient, especially for RoBERTa, than training the models locally.

## Files

### `Load_Presaved_Custom_Transformer.py`
- Downloads ZIP from Google Drive
- Extracts:
  - `model.pt`
  - `config.json`
  - `vocab.json`
- Rebuilds the custom model
- Runs full evaluation:
  - Accuracy
  - Precision / Recall / F1
  - Confusion Matrix
  - ROC Curve + AUC

### `Load_Presaved_Roberta_Model.py`
Similar process, but loads the pretrained RoBERTa folder with:
- config.json
- tokenizer.json
- model.safetensors
- vocab.json

