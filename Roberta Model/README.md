# RoBERTa Fine-Tuned Model

This folder contains the training script for fine-tuning **HuggingFace RoBERTa**
to classify text as Human or AI.

## Files

### `RoBERTa_model.py`
Trains `roberta-base` using:

- HuggingFace `AutoTokenizer`
- `AutoModelForSequenceClassification`
- Train/val/test DataLoaders with padding + truncation
- AdamW optimization
- F1-based checkpoint saving

A full 10-epoch run takes ~7 hours on an RTX 4070.

The script saves the best epoch during epoch runs.

## Metrics Produced
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix

## Output Labels
`0` = Human

`1` = AI
