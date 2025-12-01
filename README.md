# Human vs AI Classifier
A Transformer-based approach to text classification as either human authored or AI generated.

- `Human = 0`
- `AI = 1`


## Dataset
Download the dataset from:

[Kaggle Dataset Source](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

After downloading, extract the files into:
- `./preprocessing`


Required file(s):
- `AI_Human.csv`

## Running the Model
Both the custom transformer model and the RoBERTa model should automatically run the ai_human_preprocess.py and generate the necessary cleaned dataframe.
The custom transformer runs much quicker than RoBERTa during training. Start with - `transformer_model.py` located in the model folder.
