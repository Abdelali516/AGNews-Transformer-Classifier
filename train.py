import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from tokenizer import build_vocab, MAX_SEQ_LEN
from dataset import AGNewsDataset, load_ag_news
from model import NewsTransformer, make_padding_mask


# ---- Config ----
TRAIN_CSV = "/home/abdelali/news-transformer-project/data/train.csv"
BATCH_SIZE = 64
D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, optimizer, criterion, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # no_grad during evaluation: skip building the backprop graph, saves memory/time
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += input_ids.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    print(f"Using device: {DEVICE}")

    # 1. Load and split data
    texts, labels = load_ag_news(TRAIN_CSV)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # 2. Build vocab from training data only
    vocab = build_vocab(train_texts)
    print(f"Vocab size: {len(vocab)}")

    # 3. Datasets + DataLoaders
    train_dataset = AGNewsDataset(train_texts, train_labels, vocab)
    val_dataset = AGNewsDataset(val_texts, val_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Build the model
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")

    # 5. Loss function + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training loop across epochs
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, DEVICE, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, DEVICE, train=False
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # save the model whenever validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
