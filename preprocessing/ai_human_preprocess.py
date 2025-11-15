# Data Imports, Python == 3.10
import pandas as pd
import re
import unicodedata
import os

# Getting the data
raw_df = pd.read_csv("preprocessing/AI_Human.csv")
print(f"Original Shape: {raw_df.shape}")

# Dropping the empty text row and clearing raw_df from memory
# Was only one empty row with no text. Check eda.
if (raw_df.text.str.lower() == ' ').sum() == 1:
    missing_value = raw_df[(raw_df.text == ' ')].index[0]
    df = raw_df.drop(missing_value, axis = 0).reset_index(drop=True)
else:
    df = raw_df.copy()

# Clearing raw_df from memory and continuing with df.    
del(raw_df)
print(f"Shape after removing null text: {df.shape}")

def clean_text_series(text_series: pd.Series) -> pd.Series:
    """
    Preprocessing for AI vs Human text classification.
    - keeps case (no lowercasing)
    - strips leading/trailing whitespace
    - removes matching outer quotes "..." or '...'
    - normalizes unicode
    - collapses repeated whitespace to a single space
    """
    def _clean(text):
        """
        Helper function for the clean_text_series function.
        """
        # checking for nan
        if pd.isna(text):
            return ""

        text = str(text)

        # Normalizing unicode (handles weird encodings)
        text = unicodedata.normalize("NFC", text)

        # Removing leading/trailing whitespace
        text = text.strip()

        # Stripping quotes around the entire text
        # Not internal Quotes
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ['"', "'"]:
            text = text[1:-1].strip()

        # Collapses multiple whitespace into a single space
        text = re.sub(r"\s+", " ", text)

        return text

    return text_series.apply(_clean)


# Clean the text
df["text_clean"] = clean_text_series(df["text"])

# Optionally drop rows that become empty after cleaning (should be rare now)
df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)

# Removing the dirty text column
df_export = df.drop(["text"], axis = 1)
# Clearing df from memory, now df_export
del(df)

# Final Shape
print(f"Final Output Shape: {df_export.shape}")

# Renaming Columns
df_export.columns = ["ai", "text"]

# Creating the output directory if needed
os.makedirs(f"./model", exist_ok=True)
# Exporting to csv with headers
df_export.to_csv("model/ai_human_clean.csv", header=True, encoding='utf-8', index=False)
# Removing df_export from memory
del(df_export)
print("✓ File saved as 'au_human_clean.csv' in ./Model")

