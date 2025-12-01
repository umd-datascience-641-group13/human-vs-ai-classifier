import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, math, sys, re, timeit, csv, subprocess, datetime, random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split


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

project_root = get_project_root()
os.chdir(project_root)
print("Project root set to:", project_root)
sys.path.append(str(project_root)) # Making sure dir is good


def run_preprocessing():
    from preprocessing.ai_human_preprocess import main as preprocess_main
    # Run preprocessing
    # If this fails, comment out and run preprocessing manually.
    df = preprocess_main()
    print() # Just for console prints
    df.info()
    return df


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

class AIHumanDataset(Dataset):
    """
        Stores all text samples and labels, and
        converts each text into token IDs on-the-fly when indexed.

        - texts: list of raw text strings
        - labels: list of integer labels (0=human, 1=AI)
        - vocab: a global token→id dictionary for the entire dataset
        - max_len: fixed sequence length for tokenized output (truncates text after, pads with 0's if short)
    """
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, idx):
        """
        Pytorch under the hood call.
        
        Encodes/Tokenizes text using the master vocab and
        returns a tensor of vocab inx's. e.x.[5,235,15,12,05,...]
        """
        ids = encode_text(self.texts[idx], self.vocab, self.max_len)
        input_ids = torch.tensor(ids, dtype=torch.long)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label

    def __len__(self):
        """
        Returns train, test, val sizes
        """
        return len(self.labels)

def encode_text(text, vocab, max_len):
    """
    Coverts text to a list of integers from the vocab dictionary
    """
    tokens = basic_tokenize(text)

    # convert tokens to integers using vocab dict or
    # "unknown" if tokens not in train dict.
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

    # truncates to the max_len allowed
    ids = ids[:max_len]

    # pads up until the max len if necessary
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids)) # Creates a list of zeroes [0,0,0] and appends
        
        
    # returns the indexed vocab 
    return ids

def basic_tokenize(text):
    """
    Tokenize into:
    words: \w+
    or: |
    punctuation: [^\w\s] (all but words or whitespaces)
    """

    return re.findall(r"\w+|[^\w\s]", text)
    


def build_vocab(X_train):
    """
    Creates a dictionary containing every token in the training split.
    Tokens appear in the order they are first seen.
    Only X_train data is used to build the vocabulary.
    """
    vocab = {"<pad>": 0, "<unk>": 1}
    index = 2
    for text in X_train:
        for word in basic_tokenize(text):
            if word not in vocab:
                vocab[word] = index
                index+=1
    return vocab




def make_loaders(X_train, y_train, X_test, y_test, X_val, y_val, 
                 vocab, max_len=128, batch_size=512, num_workers=0):

    train_ds = AIHumanDataset(X_train, y_train, vocab, max_len)
    val_ds = AIHumanDataset(X_val, y_val,   vocab, max_len)
    test_ds = AIHumanDataset(X_test, y_test, vocab, max_len)

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

    return train_loader, test_loader, val_loader


# "sinusoidal" positional-encoding from Vaswani et al.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)



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

    batch_size = 64
    max_len=128
    val_accy_ = []
    training_loss = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    df = run_preprocessing()
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(df)
    #X_train, y_train, X_test, y_test, X_val, y_val = testing_model_small_data(X_train, y_train, X_test, y_test, X_val, y_val, n_rows = 1000)
    n_rows = len(y_train)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    vocab = build_vocab(X_train)
    print(f"Vocab size: {len(vocab)}")
    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, 
                                                         X_test, y_test, 
                                                         X_val, y_val,
                                                         vocab, 
                                                         max_len=max_len, 
                                                         batch_size=batch_size, 
                                                         num_workers=0)
    ds = train_loader.dataset
    h=0
    for i in range(len(ds)):
        try:
            _ = ds[i]
            h+=1
        except Exception as e:
            print("First failing sample index:", i)
            print("Error:", e)
            break
    
    print(f"Done Loading {h} Documents")

    vocab_size = len(vocab)
    model = AIHumanTransformerClassifier(
        vocab_size=vocab_size,
        d_model=max_len,
        n_heads=4,
        num_layers=2,
        dim_ff=256,
        num_classes=2,
        max_len=max_len,
        dropout=0.1,
        pad_idx=vocab["<pad>"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    from tqdm.auto import tqdm
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for input_ids, labels in epoch_bar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            training_loss.append(loss)
            loss.backward()
            optimizer.step()
            epoch_bar.set_postfix(loss=loss.item())
            
        print("Done Training")
        # simple val loop
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accy_.append(round(correct / total,4))

        print(f"Epoch {epoch+1}: val acc = {correct / total:.4f}")




    if True:
        # append_row.py
        import csv, os
        from pathlib import Path
        
        logging_path = "history/run_data/"
        os.makedirs(logging_path, exist_ok=True)
        
        csv_path = "history/accuracy_log.csv"
        def append_row(values, path=csv_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(values)
        
        current_datetime = datetime.now()
        safe_filename_timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
        output_zip_path = f"./run_data/{safe_filename_timestamp}.zip"
        
        avg_val_accy = np.array(val_accy_).mean()
        avg_train_loss = np.mean([l.detach().cpu().item() for l in training_loss])
        append_row([
            safe_filename_timestamp , batch_size, max_len, n_rows, vocab_size, num_epochs, avg_train_loss, avg_val_accy])
        print(f"Run results printed to {csv_path}")


        # Zipping the graphs and saving in ./logs
        from zipfile import ZipFile

        # List of files to include in the zip archive
        
        #files_to_zip = ["Avg_lr_wd.png", "val_loss_and_slope.png", "val_metrics.png", "lr_wd_schedules.png", "plt.png"]

        # Create dummy files for demonstration if they don't exist
        # Safety check for the code to always work



        # for filename in files_to_zip:
        #     if not os.path.exists(filename):
        #         with open(filename, "w") as f:
        #             f.write(f"This is content for {filename}\n")

        # Create the zip file and add the specified files
        # with ZipFile(output_zip_path, 'w') as zipf:
        #     for file_path in files_to_zip:
        #         # Add the file to the zip archive
        #         zipf.write(file_path, os.path.basename(file_path))
        # print(f"Zip file '{output_zip_path}' created successfully with files: {files_to_zip}")





if __name__ == "__main__":
    main(num_epochs=3) 






