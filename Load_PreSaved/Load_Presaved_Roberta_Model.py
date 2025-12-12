from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import json, os, sys
import torch
import gdown
import matplotlib.pyplot as plt
from pathlib import Path
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

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir)

    print(f"Files Downloaded and Extracted")
    return target_dir


GDRIVE_FOLDER_URL = (
    "https://drive.google.com/file/d/1PbLeU7oo3jAGVX7u3b2PoZpcVnpRG3cc/view?usp=sharing")

file = "Collab_Roberta_Presaved.zip"
file_dir = "Collab_Roberta_Presaved"
download_and_extract(GDRIVE_FOLDER_URL, file, model_dir)

sys.path.append(str(model_dir/ file_dir))

path = f"{file_dir}/saved_roberta"
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)

path = f"{file_dir}/ai_human_clean.csv"
df = pd.read_csv(path)
df_val = df[df["split"] == "val"]
df_val = df_val.drop("split", axis=1)
texts = df_val["text"].tolist()
labels = df_val["label"].astype(int).tolist()   # 0 = Human, 1 = AI
print(f"Texts: {len(texts)}, Labels: {len(labels)}")


import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_labels = []

batch_size = 64
max_len = 128  # match your training

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_labels = labels[i:i + batch_size]

    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len)

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    batch_labels_t = torch.tensor(batch_labels).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
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
    pos_label=1) # AI = 1

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=["Human (0)", "AI (1)"]))

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))

fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_preds)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"RoBERTa (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0,1], [0,1], 'k--', label="Random Baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()