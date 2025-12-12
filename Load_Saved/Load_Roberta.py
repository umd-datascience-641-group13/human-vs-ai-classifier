from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score , accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent

model_dir = root_dir / "saved_roberta"

tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)

def classify_text(text, tokenizer, model, max_length=128):
    model.eval()

    # tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length)

    # inference
    with torch.no_grad():
        outputs = model(**inputs)

    # logits/probabilities
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0]

    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item()

    return pred_class, confidence, probs

path = root_dir / "Cleaned Data/ai_human_clean.csv"
df = pd.read_csv(path)
df_val = df[df["split"] == "val"]
df_val = df_val.drop("split", axis=1)
texts = df_val["text"].tolist()
labels = df_val["label"].astype(int).tolist()   # 0 = Human, 1 = AI
print(f"Texts: {len(texts)}, Labels: {len(labels)}")

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

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"RoBERTa (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0,1], [0,1], 'k--', label="Random Baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()