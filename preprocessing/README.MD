# Preprocessing Pipeline

This folder contains the preprocessing script responsible for cleaning and structuring the dataset used by both models.

## Files

### `ai_human_preprocess.py`
This script:
1. Downloads the dataset from Google Drive (via `gdown`)  
2. Loads `AI_Human.csv`, originally from [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text_blank)
3. Cleans the text:
   - Unicode normalization (NFC)
   - Remove outer quotes
   - Strip whitespace
   - Collapse repeated whitespace
   - Preserve capitalization (important for AI/Human differences)
4. Performs a **stratified split**:
   - 70% train
   - 15% validation
   - 15% test
5. Saves the cleaned dataset for use by training scripts

## Notes
- This script runs automatically when training either model.
- If download fails, place the CSV manually in this directory.
