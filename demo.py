"""
Demo: classify a news headline with the trained NewsTransformer.

Type a headline, get back its predicted category and confidence.
"""

import torch
from sklearn.model_selection import train_test_split

from tokenizer import build_vocab, encode, MAX_SEQ_LEN
from dataset import load_ag_news
from model import NewsTransformer, make_padding_mask

MODEL_PATH = "best_model.pt"

D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2
NUM_CLASSES = 4
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_model_and_vocab():
    # load_ag_news() with no argument uses the same PATH already defined
    # in dataset.py -- no second copy of the path to keep in sync
    texts, labels = load_ag_news()
    train_texts, _, _, _ = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    vocab = build_vocab(train_texts)

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
    model.eval()

    return model, vocab


def classify(text: str, model, vocab):
    token_ids = encode(text, vocab)
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)
    attention_mask = make_padding_mask(input_ids, pad_id=0)

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)[0]

    pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], probs[pred_idx].item()


def main():
    print("Loading model...")
    model, vocab = load_model_and_vocab()

    print("\nType a news headline and press enter (empty line to quit).\n")
    while True:
        text = input("> ").strip()
        if not text:
            break

        category, confidence = classify(text, model, vocab)
        print(f"  -> {category}  ({confidence * 100:.1f}% confidence)\n")


if __name__ == "__main__":
    main()