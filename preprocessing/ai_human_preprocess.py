# Data Imports, Python == 3.10
import pandas as pd
import re
import unicodedata
import os, sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import gdown



def ensure_raw_from_drive(RAW_DIR, GDRIVE_FOLDER_URL):
    """
    Download Data from from Google Drive
    if not already present.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / "AI_Human.csv"
    print(f"Raw: {RAW_DIR}")
    print(f"Path: {path}")
    if not (path.exists()):
        print(f"Downloading CSVs from Google Drive folder to {RAW_DIR}")
        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=str(RAW_DIR),
            quiet=False,
            use_cookies=False)

    if not path.exists():
        raise FileNotFoundError(f"Expected {path}")
    

    return path


def main():
    script_dir = Path(__file__).resolve().parent

    print(f"Script_dir{script_dir}")

    GDRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1mJZr1eYYwLhswzXwV9StwnOwmMvLX15e")


    # Path to AI_Human.csv in the same folder
    csv_path = script_dir / "AI_Human.csv"
    
    print("----------------------------------------------------------------------")
    ensure_raw_from_drive(script_dir, GDRIVE_FOLDER_URL)

    # Getting the data
    raw_df = pd.read_csv(csv_path)
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
    df_export.columns = ["label", "text"]
    # indices
    train_idx, temp_idx = train_test_split(
        df_export.index,
        test_size=0.3,                 # 70% train, 30% temp
        stratify=df_export["label"],
        random_state=42)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,                 # 15% val, 15% test
        stratify=df_export.loc[temp_idx, "label"],
        random_state=42)

    df_export.loc[train_idx, "split"] = "train"
    df_export.loc[val_idx, "split"] = "val"
    df_export.loc[test_idx, "split"] = "test"

    
    # Creating the output directory if needed
    output_path = Path(__file__).resolve().parent.parent / "Cleaned Data" / "ai_human_clean.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    df_export.to_csv(output_path, header=True, encoding="utf-8", index=False)
    print(f"✔ File saved as '{output_path.name}' in {output_path.parent}")
    return df_export

if __name__ == "__main__":
    main()