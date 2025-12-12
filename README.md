# Human vs AI Classifier
A Transformer-based system that classifies text as either human-authored or AI-generated.

This project compares:
- HuggingFace RoBERTa (fine-tuned)
- A custom PyTorch Transformer model (built from scratch)

Both models are evaluated on the Kaggle Human vs AI dataset using ROC, AUC, Precision, Recall, and F1.

## Python
- Python = 3.10

## Verify Results
To simply verify results, start with the Load_PreSaved folder. Running `Load_Presaved_Custom_Transformer.py` is the least resource intensive script.

## Repo Contents
Included in this repo, as shown in the folder structure, are the following items:
- Custom Transformer Model
  - Contains `transformer_model.py`
    - A custom torch nn model that uses an encoder only (BERT) style transformer with the classic "Attention Is All You Need" Positional Encoding. It's a very light-weight model compared to the HuggingFace RoBERTa model.
- Roberta Model
  - Contains `RoBERTa_model.py`
    - This is the HuggingFace RoBERTa pretrained model that has been fine tuned on the Human Vs. AI dataset. It is a very large model that requires a GPU to run. For 10 epochs, it takes around 7 hours on an RTX 4070.
- Preprocessing
  - Contains `ai_human_preprocess.py`
    - This is the data preprocessing script that will automatically run with the above two python scripts. It performs the following tasks:
      - Downloads the data from Google Drive using the Gdown library
      - Drops missing rows from the dataset
      - Strips leading and trailing whitespace
      - Removes outer quotations from input text
      - Normalizes the text to Unicode
      - Removes multiple repeated white space (speeds up training)
      - It does NOT lowercase text, keeps human and AI differences in writing
      - Adds an additional column that indicates if a row has been selected for training, testing, or validation (70%, 15%, 15%)
- Load Saved
  - `Load_Custom_Transformer.py`
    - This code will load the trained data, if you train the model yourself.
  - `Load_Roberta.py`
    - Also loads the trained data when you train the model yourself. (Takes a long time, even on a powerful GPU)
- Load_Pre_Saved
  - `Load_Presaved_Custom_Transformer.py`
    - This will pull a saved pretrained model, 10 epochs, from Google Drive and run validation tests
    - Produces ROC and AUC results along with Precision, Recall, and F1 Scores
  - Load_Presaved_Roberta_Model.py
    - This will pull a saved pretrained RoBERTa model, 10 epochs, and run validation tests
    - Produces ROC and AUC results along with Precision, Recall, and F1 Scores


Output classification labels:
- `Human = 0`
- `AI = 1`


## Running the Model
Both the custom transformer model and the RoBERTa model should automatically run the ai_human_preprocess.py and generate the necessary cleaned dataframe.
The custom transformer, `transformer_model.py`,  runs much quicker than RoBERTa during training and is located in the Custom_Transformer_Model folder.

## Install Dependencies
`pip install -r requirements.txt`

GPU users may install a CUDA-enabled PyTorch version from:
https://pytorch.org/get-started/locally/

## Dataset
The Dataset should automatically be downloaded, but in case of an error: 

  - Download the dataset from: [Kaggle Dataset Source](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
  
  After downloading, extract the files into:
  - `./preprocessing`
  
  Required file(s):
  - `AI_Human.csv`
  
## Performance Metrics
- RoBERTa AUC ≈ 0.999
- Custom Transformer AUC ≈ 0.93
