import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import clean_and_split, build_vocab, encode, MAX_SEQ_LEN

PATH="/home/abdelali/news-transformer-project/data/train.csv"

class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        token_ids = encode(text, self.vocab) 
        
        attention_mask = [1 if tid != 0 else 0 for tid in token_ids]  

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_ag_news(csv_path: str=PATH):
    df = pd.read_csv(csv_path, header=None, names=["label", "title", "description"])
    df["text"] = df["title"] + " " + df["description"]
    df["label"] = df["label"] - 1  # 1..4 -> 0..3
    return df["text"].tolist(), df["label"].tolist()


