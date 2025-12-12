import json
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import gdown
import zipfile

model_dir = Path(__file__).resolve().parent
os.chdir(model_dir)

def download_and_extract(url,zip_name,target_dir):
    target_dir = Path(target_dir)

    print(f"Downloading ZIP from: {url}")
    
    gdown.cached_download(
        url=url,
        path=zip_name,
        quiet=False,
        fuzzy=True)
    
    zip_path = Path(zip_name)

    target_dir.mkdir(parents=True, exist_ok=True)
    #extract_dir = Path(".")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir)

    print(f"Files Downloaded and Extracted")
    return target_dir


GDRIVE_FOLDER_URL = (
    "https://drive.google.com/file/d/1Md6G89Qh1hTG5w_uiz7Y-mFKs3O9UgNK/view?usp=sharing")

file = "Collab_Custom_Presaved.zip"
file_dir = "Collab_Custom_Presaved"
download_and_extract(GDRIVE_FOLDER_URL, file, model_dir)

# Load config
with open(f"{model_dir}/{file_dir}/config.json", "r") as f:
    config = json.load(f)

# Load vocab
with open(f"{model_dir}/{file_dir}/vocab.json", "r") as f:
    vocab = json.load(f)

sys.path.append(str(model_dir/ file_dir))

from transformer_model import AIHumanTransformerClassifier

model = AIHumanTransformerClassifier(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    num_layers=config["num_layers"],
    dim_ff=config["dim_ff"],
    max_len=config["max_len"],
    pad_idx=config["pad_idx"],
    num_classes=2)

#model_dir = os.path.join(os.getcwd(),"saved_custom_transformer")
state_dict = torch.load(os.path.join(file_dir, "model.pt"), map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model_dir)
path = f"{file_dir}/ai_human_clean.csv"
df = pd.read_csv(path)
df_val = df[df["split"] == "val"]
df_val = df_val.drop("split", axis=1)
texts = df_val["text"].tolist()
labels = df_val["label"].astype(int).tolist()   # 0 = Human, 1 = AI
print(f"Texts: {len(texts)}, Labels: {len(labels)}")

import torch
import numpy as np

pad_idx = config["pad_idx"]
max_len = config["max_len"]  # 128

def encode_batch(batch_texts, vocab, max_len, pad_idx):
    """
    Turn a list of texts into a tensor [batch, max_len] of token ids.
    Simple whitespace tokenizer + lowercasing to match training.
    """
    unk_idx = vocab.get("<unk>", 1)  # fallback if <unk> not in vocab

    encoded = []
    for text in batch_texts:
        tokens = str(text).lower().split()
        ids = [vocab.get(tok, unk_idx) for tok in tokens]
        ids = ids[:max_len]
        ids += [pad_idx] * (max_len - len(ids))
        encoded.append(ids)

    return torch.tensor(encoded, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_labels = []

batch_size = 64
max_len = config["max_len"]  # 128

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_labels = labels[i:i + batch_size]

    input_ids = encode_batch(batch_texts, vocab, max_len, pad_idx).to(device)
    batch_labels_t = torch.tensor(batch_labels, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=-1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch_labels_t.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix)

accuracy = accuracy_score(all_labels, all_preds)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    average="binary",
    pos_label=1)  # AI = 1

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification report:")
print(classification_report(all_labels,all_preds,
    target_names=["Human (0)", "AI (1)"]))

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))

fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_preds)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"RoBERTa (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0,1], [0,1], 'k--', label="Random Baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()