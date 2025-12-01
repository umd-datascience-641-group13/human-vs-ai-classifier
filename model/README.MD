# Model

This directory contains the training and evaluation code for the **Human vs AI Text Classifier**.

---

## ⚠️ Important: Preprocessing Required First
The transformer_model.py will preprocess the data using the preprocessing script. If an error occurs, please follow the manual method outlined below, 
or point the script to the correct ai_human_preprocess.py location.

### Manual Method
Before running any model scripts, you must generate the cleaned dataset.

1. Navigate to the preprocessing directory:

```
human-vs-ai-classifier/preprocessing
```

2. Run the preprocessing script:

```bash
python ai_human_preprocess.py
```

3. After the script completes, ensure that the generated file:

- `ai_human_clean.csv`

exists or is moved into the current `/model` directory.

This file is required for training and evaluation.

