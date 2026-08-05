import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tokenizer import build_vocab, MAX_SEQ_LEN
from dataset import AGNewsDataset, load_ag_news
from model import NewsTransformer

TRAIN_CSV = "/home/abdelali/news-transformer-project/data/train.csv"
MODEL_PATH = "best_model.pt"

D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.1
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def main():
    # 1. Recreate the SAME split + vocab used during training
    #    (must match train.py exactly, so the model sees the same val set)
    texts, labels = load_ag_news(TRAIN_CSV)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    vocab = build_vocab(train_texts)

    val_dataset = AGNewsDataset(val_texts, val_labels, vocab)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NewsTransformer(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # dropout off, evaluation mode


    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["label"].to(DEVICE)

            logits, _ = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())


    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # 5. Confusion matrix: plot and save
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - AG News Transformer")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
