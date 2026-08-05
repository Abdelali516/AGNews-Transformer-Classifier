import re
from collections import Counter
import pandas as pd

VOCAB_SIZE=15000
MAX_SEQ_LEN=60
PAD_TOKENS="<PAD>"
UNK_TOKENS="<UNK>"

def clean_and_split(text):
    text=text.lower()
    text=re.sub(r"[^a-z0-9\s]", " ", text)
    tokens=text.split()

    return tokens

def build_vocab(texts, vocab_size: int = VOCAB_SIZE):
    counter=Counter()

    for text in texts:
        tokens=clean_and_split(text)
        counter.update(tokens)

    most_common=counter.most_common(vocab_size-2)

    vocab={PAD_TOKENS:0,UNK_TOKENS:1}
    for word,_ in most_common:
        vocab[word]=len(vocab)

    return vocab

def encode(text, vocab,max_len: int=MAX_SEQ_LEN):
    tokens=clean_and_split(text)
    ids=[]
    for tok in tokens:
        token_ids=vocab.get(tok, vocab[UNK_TOKENS])
        ids.append(token_ids)

    if len(ids)<max_len:
        ids=ids + [vocab[PAD_TOKENS]] * (max_len - len(ids))
    else:
        ids=ids[:max_len]

    return ids

