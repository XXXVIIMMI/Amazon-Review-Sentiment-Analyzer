import os, re, csv, random, argparse
from typing import List, Tuple

NEUTRAL_PHRASES = [
    "okay", "ok ", "average", "decent", "fine ", "alright", "not bad",
    "mediocre", "so so", "so-so", "fair ", "acceptable", "nothing special",
    "nothing exceptional", "works as expected", "does the job", "does what it",
    "as described", "as advertised", "mixed feelings", "could be better",
    "could be worse", "three star", "3 star", "3/5",
]


def clean(title: str, body: str) -> str:
    text = f"{title}. {body}" if title.strip() else body
    text = text.lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>",      " ", text)
    text = re.sub(r"[^a-z0-9\s'!?.,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:512]


def is_neutral(text: str) -> bool:
    return any(p in text for p in NEUTRAL_PHRASES)


def load_amazon_dataset(
    total_samples: int   = 60_000,
    val_ratio:     float = 0.10,
    test_ratio:    float = 0.10,
    neutral_ratio: float = 0.15,
    seed:          int   = 42,
    output_dir:    str   = "data/processed",
    verbose:       bool  = True,
):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run:  pip install datasets huggingface-hub")

    random.seed(seed)
    if verbose:
        print("Downloading amazon_polarity (~650 MB, cached after first run)\n")

    ds        = load_dataset("amazon_polarity")
    train_raw = ds["train"]
    test_raw  = ds["test"]

    n_neu = int(total_samples * neutral_ratio)
    n_bin = total_samples - n_neu
    n_pos = n_bin // 2
    n_neg = n_bin - n_pos

    pos_s: List[Tuple[str, str]] = []
    neg_s: List[Tuple[str, str]] = []
    neu_s: List[Tuple[str, str]] = []

    idxs = list(range(len(train_raw)))
    random.shuffle(idxs)

    for idx in idxs:
        row  = train_raw[idx]
        text = clean(row.get("title", ""), row.get("content", ""))
        lbl  = row["label"]
        if not text: continue
        if   lbl == 1 and len(pos_s) < n_pos: pos_s.append((text, "positive"))
        elif lbl == 0 and len(neg_s) < n_neg: neg_s.append((text, "negative"))
        elif len(neu_s) < n_neu and is_neutral(text): neu_s.append((text, "neutral"))
        if len(pos_s) >= n_pos and len(neg_s) >= n_neg and len(neu_s) >= n_neu:
            break

    for row in test_raw:
        if len(neu_s) >= n_neu: break
        text = clean(row.get("title", ""), row.get("content", ""))
        if text and is_neutral(text): neu_s.append((text, "neutral"))

    all_s = pos_s + neg_s + neu_s
    random.shuffle(all_s)

    if verbose:
        from collections import Counter
        for lbl, cnt in Counter(s[1] for s in all_s).items():
            print(f"  {lbl}: {cnt:,}  ({cnt/len(all_s)*100:.1f}%)")
        print()

    n = len(all_s)
    nv, nt = int(n * val_ratio), int(n * test_ratio)
    splits = {"train": all_s[:n-nv-nt], "val": all_s[n-nv-nt:n-nt], "test": all_s[n-nt:]}

    os.makedirs(output_dir, exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(output_dir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["text", "label"]); w.writerows(rows)
        if verbose: print(f"  {name:6s} -> {path}  ({len(rows):,} rows)")

    if verbose: print("\nReady!  ->  python -m src.train\n")
    return splits


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples",    type=int,   default=60_000)
    p.add_argument("--output_dir", default="data/processed")
    p.add_argument("--val_ratio",  type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    a = p.parse_args()
    load_amazon_dataset(a.samples, a.val_ratio, a.test_ratio, output_dir=a.output_dir, seed=a.seed)
