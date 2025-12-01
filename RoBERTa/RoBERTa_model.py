import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, math, sys, re, timeit, csv, subprocess, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification


def get_project_root():
    """
    If running as a .py file, use __file__.
    If running in Jupyter, detect if current working directory
    ends with 'model' and go one level up.
    """

    # Running as a .py file → reliable
    if "__file__" in globals():
        print("True")
        return Path(__file__).resolve().parent.parent

    # Running in Jupyter → check folder name
    cwd = Path.cwd().resolve()

    # If you're inside .../model, go up a level
    if cwd.name.lower() == "model":
        return cwd.parent
    
    # Otherwise just use the cwd
    return cwd



def run_preprocessing():
    from preprocessing.ai_human_preprocess import main as preprocess_main
    # Run preprocessing
    # If this fails, comment out and run preprocessing manually.
    df = preprocess_main()
    print() # Just for console prints
    df.info()
    return df



project_root = get_project_root()
os.chdir(project_root)
print("Project root set to:", project_root)
sys.path.append(str(project_root)) # Making sure dir is good




def split_data(df):

    # Creating the splits for train, test, val
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.ai, test_size=0.05, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
    print(f"Train Samples: {len(y_train)}")
    print(f"Test Samples: {len(y_test)}")
    print(f"Val Samples: {len(y_val)}\n")
    return X_train, y_train, X_test, y_test, X_val, y_val

def testing_model_small_data(X_train, y_train, X_test, y_test, X_val, y_val, n_rows=100):
    X_train = X_train.iloc[:n_rows]
    y_train = y_train.iloc[:n_rows]
    X_test = X_test.iloc[:n_rows]
    y_test = y_test.iloc[:n_rows]
    X_val = X_val.iloc[:n_rows]
    y_val = y_val.iloc[:n_rows]

    return X_train, y_train, X_test, y_test, X_val, y_val



class AIHumanRobertaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        texts: iterable of raw text
        labels: iterable of 0/1 integers
        tokenizer: HuggingFace tokenizer (e.g. roberta-base)
        """
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # squeeze(0) because tokenizer returns shape [1, max_len]
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return item


def make_roberta_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    tokenizer,
    max_len=128,
    batch_size=64,
    num_workers=0
):
    train_ds = AIHumanRobertaDataset(X_train, y_train, tokenizer, max_len)
    val_ds   = AIHumanRobertaDataset(X_val,   y_val,   tokenizer, max_len)
    test_ds  = AIHumanRobertaDataset(X_test,  y_test,  tokenizer, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=(num_workers > 0), drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader




class AIHumanTransformerClassifier(nn.Module):
    """
    Small encoder-only transformer for AI vs Human text classification.

    Architecturally inspired by the encoder stack in
    'Attention Is All You Need' (Vaswani et al., 2017) and
    encoder-only models like BERT/Roberta, but much smaller and trained from scratch.
    """
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_ff=256,
        num_classes=2,
        max_len=128,
        dropout=0.1,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # 1) Token embedding: [B, L] -> [B, L, d_model]
        self.token_embedding = nn.Embedding(
            padding_idx=pad_idx,
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        # 2) Positional encoding (your sinusoidal implementation)
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )

        # 3) Encoder stack (multi-head self-attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",      # closer to BERT/Roberta
            batch_first=True        # input: [B, L, d_model]
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len] of token IDs
        returns: logits [batch_size, num_classes]
        """
        # mask: True where padding
        pad_mask = (input_ids == self.pad_idx)          # [B, L]

        # token + positional embeddings
        x = self.token_embedding(input_ids)             # [B, L, d_model]
        x = self.pos_encoding(x)                        # [B, L, d_model]

        # encoder expects src_key_padding_mask: True = ignore
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, L, d_model]

        # mean-pool over non-pad tokens
        mask = (~pad_mask).unsqueeze(-1)                # [B, L, 1]
        x_masked = x * mask
        summed = x_masked.sum(dim=1)                    # [B, d_model]
        lengths = mask.sum(dim=1).clamp(min=1)          # [B, 1]
        pooled = summed / lengths                       # [B, d_model]

        logits = self.classifier(pooled)                # [B, num_classes]
        return logits




def main(num_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    df = run_preprocessing()
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(df)
    #X_train, y_train, X_test, y_test, X_val, y_val = testing_model_small_data(X_train, y_train, X_test, y_test, X_val, y_val, n_rows = nrows)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    train_loader, val_loader, test_loader = make_roberta_loaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        tokenizer=tokenizer,
        max_len=128,
        batch_size=64,
        num_workers=0
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  

    
    from tqdm.auto import tqdm

    num_epochs = num_epochs  

    for epoch in range(num_epochs):
        model.train()
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in epoch_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_bar.set_postfix(loss=loss.item())

        

        # simple val loop
        model.eval()
        correct = total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

                val_loss_sum += loss.item() * labels.size(0)

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss_sum / total
        val_acc = correct / total
        print(f"Epoch {epoch}: val loss = {avg_val_loss:.4f}, val acc = {val_acc:.4f}")
    
    model.save_pretrained("saved_roberta")
    tokenizer.save_pretrained("saved_roberta")


if __name__ == "__main__":
    main() 







#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#model = RobertaForSequenceClassification.from_pretrained('roberta-base')