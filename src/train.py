import os, time, json, argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.models.sentiment_model import build_model
from src.utils.dataset import (
    SentimentDataset, build_vocab, save_vocab, load_vocab, collate_fn, IDX2LABEL
)


def _macro_f1(y_true, y_pred, n=3):
    tp=[0]*n; fp=[0]*n; fn=[0]*n
    for t, p in zip(y_true, y_pred):
        if t == p: tp[t] += 1
        else: fp[p] += 1; fn[t] += 1
    f1s = []
    for i in range(n):
        pr = tp[i] / (tp[i] + fp[i] + 1e-8)
        rc = tp[i] / (tp[i] + fn[i] + 1e-8)
        f1s.append(2 * pr * rc / (pr + rc + 1e-8))
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return acc, sum(f1s) / n, f1s


def train_epoch(model, loader, opt, crit, device, clip=1.0):
    model.train()
    ls = co = tot = 0
    for i, batch in enumerate(loader):
        x, L, y = batch["input_ids"].to(device), batch["lengths"], batch["labels"].to(device)
        opt.zero_grad(set_to_none=True)
        logits, _ = model(x, L)
        loss = crit(logits, y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        bs = x.size(0); ls += loss.item() * bs; co += (logits.argmax(-1) == y).sum().item(); tot += bs
        if (i + 1) % 200 == 0:
            print(f"    step {i+1:>5}  loss={ls/tot:.4f}  acc={co/tot:.4f}", flush=True)
    return ls / tot, co / tot


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    ls = 0; preds = []; labels = []
    for batch in loader:
        x, L, y = batch["input_ids"].to(device), batch["lengths"], batch["labels"].to(device)
        logits, _ = model(x, L)
        ls += crit(logits, y).item() * x.size(0)
        preds  += logits.argmax(-1).cpu().tolist()
        labels += y.cpu().tolist()
    acc, mf1, f1s = _macro_f1(labels, preds)
    return {"loss": ls / len(loader.dataset), "accuracy": acc, "macro_f1": mf1,
            "f1_pos": f1s[0], "f1_neg": f1s[1], "f1_neu": f1s[2]}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*56}\n  Amazon Sentiment — LSTM  |  Device: {device}\n{'='*56}")

    train_path = os.path.join(args.data_dir, "train.csv")
    val_path   = os.path.join(args.data_dir, "val.csv")
    vocab_path = os.path.join(args.data_dir, "vocab.json")

    if os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path); print(f"[vocab] Loaded  {len(vocab):,} tokens")
    else:
        print("[vocab] Building ...")
        vocab = build_vocab(train_path, args.vocab_size, min_freq=2)
        save_vocab(vocab, vocab_path); print(f"[vocab] Built   {len(vocab):,} tokens")

    train_ds = SentimentDataset(train_path, vocab, args.max_len)
    val_ds   = SentimentDataset(val_path,   vocab, args.max_len)
    print(f"\n[data] Train {len(train_ds):,}  |  Val {len(val_ds):,}")
    dist = Counter(s[1] for s in train_ds.samples)
    for i, nm in IDX2LABEL.items(): print(f"  {nm}: {dist[i]:,}")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=2, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = build_model(vocab_size=len(vocab), embed_dim=args.embed_dim,
                        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                        dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] AmazonLSTMSentiment  hidden={args.hidden_dim}  layers={args.num_layers}  params={n_params:,}")

    opt    = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    warmup = LinearLR(opt, 0.1, total_iters=2)
    cosine = CosineAnnealingLR(opt, T_max=max(args.epochs - 2, 1))
    sched  = SequentialLR(opt, [warmup, cosine], milestones=[2])
    label_dist = Counter(s[1] for s in train_ds.samples)
    total_count = sum(label_dist.values())
    class_weights = torch.tensor(
        [total_count / (3 * label_dist[i]) for i in range(3)], dtype=torch.float
    ).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_f1 = patience = 0; history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        tl, ta = train_epoch(model, train_loader, opt, crit, device)
        vm = evaluate(model, val_loader, crit, device)
        sched.step()

        row = {"epoch": epoch, "train_loss": round(tl, 4), "train_acc": round(ta, 4),
               **{k: round(v, 4) for k, v in vm.items()}}
        history.append(row)

        print(f"  train  loss={tl:.4f}  acc={ta:.4f}")
        print(f"  val    loss={vm['loss']:.4f}  acc={vm['accuracy']:.4f}  macro_f1={vm['macro_f1']:.4f}")
        print(f"  f1     pos={vm['f1_pos']:.3f}  neg={vm['f1_neg']:.3f}  neu={vm['f1_neu']:.3f}")
        print(f"  {time.time()-t0:.0f}s")

        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]; patience = 0
            ckpt = os.path.join(args.checkpoint_dir, "best_lstm.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "vocab_path": vocab_path, "val_f1": best_f1,
                        "val_acc": vm["accuracy"], "args": vars(args)}, ckpt)
            print(f"  New best  ->  {ckpt}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\nEarly stopping (patience={args.patience})")
                break

    hist_path = os.path.join(args.checkpoint_dir, "history_lstm.json")
    with open(hist_path, "w") as f: json.dump(history, f, indent=2)
    print(f"\n{'='*56}\n  Done!  Best val macro-F1: {best_f1:.4f}\n  History -> {hist_path}\n{'='*56}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="data/processed/")
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--embed_dim",      type=int,   default=128)
    p.add_argument("--hidden_dim",     type=int,   default=256)
    p.add_argument("--num_layers",     type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.5)
    p.add_argument("--vocab_size",     type=int,   default=50_000)
    p.add_argument("--max_len",        type=int,   default=256)
    p.add_argument("--patience",       type=int,   default=4)
    train(p.parse_args())
