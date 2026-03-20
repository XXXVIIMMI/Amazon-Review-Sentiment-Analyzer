import re, csv, json, os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD, UNK         = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1
LABEL2IDX        = {"positive": 0, "negative": 1, "neutral": 2}
IDX2LABEL        = {v: k for k, v in LABEL2IDX.items()}


def tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower()).split()


def build_vocab(path: str, max_vocab: int = 50_000, min_freq: int = 2) -> Dict[str, int]:
    counter: Counter = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counter.update(tokenize(row["text"]))
    vocab = {PAD: PAD_IDX, UNK: UNK_IDX}
    for word, freq in counter.most_common(max_vocab - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def encode(text: str, vocab: Dict[str, int], max_len: int = 256) -> List[int]:
    return [vocab.get(t, UNK_IDX) for t in tokenize(text)[:max_len]]


class SentimentDataset(Dataset):
    def __init__(self, path: str, vocab: Dict[str, int], max_len: int = 256):
        self.samples: List[Tuple[List[int], int]] = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids   = encode(row["text"], vocab, max_len)
                label = LABEL2IDX.get(row["label"].strip().lower(), -1)
                if ids and label != -1:
                    self.samples.append((ids, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return {"input_ids": torch.tensor(ids,   dtype=torch.long),
                "label":     torch.tensor(label, dtype=torch.long)}


def collate_fn(batch):
    seqs    = [b["input_ids"] for b in batch]
    labels  = torch.stack([b["label"] for b in batch])
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return {"input_ids": padded, "lengths": lengths, "labels": labels}
