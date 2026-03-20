import argparse, json, os, torch
import torch.nn.functional as F
from src.models.sentiment_model import AmazonLSTMSentiment
from src.utils.dataset import encode, load_vocab

IDX2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
EMOJI     = {0: "😊", 1: "😞", 2: "😐"}
STARS     = {0: "★★★★★", 1: "★☆☆☆☆", 2: "★★★☆☆"}


def load_checkpoint(checkpoint_dir="checkpoints/"):
    path = os.path.join(checkpoint_dir, "best_lstm.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}\n  -> Run: python -m src.train")
    ckpt  = torch.load(path, map_location="cpu")
    a     = ckpt.get("args", {})
    vpath = ckpt.get("vocab_path", "data/processed/vocab.json")
    vocab = load_vocab(vpath) if os.path.exists(vpath) else ckpt["vocab"]
    model = AmazonLSTMSentiment(
        vocab_size=len(vocab), embed_dim=a.get("embed_dim", 128),
        hidden_dim=a.get("hidden_dim", 256), num_classes=3,
        num_layers=a.get("num_layers", 2), dropout=0.0)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, vocab, ckpt


@torch.no_grad()
def predict(text, model, vocab, max_len=256):
    ids = encode(text, vocab, max_len)
    if not ids: return None
    x       = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)])
    logits, attn = model(x, lengths)
    probs = F.softmax(logits, -1)[0].tolist()
    pred  = int(logits.argmax(-1).item())
    toks  = text.lower().split()[:max_len]
    top5  = sorted(zip(toks, attn[0, :len(toks)].tolist()), key=lambda kv: kv[1], reverse=True)[:5]
    return {"sentiment": IDX2LABEL[pred], "emoji": EMOJI[pred], "stars": STARS[pred],
            "confidence": round(max(probs), 4),
            "positive": round(probs[0], 4), "negative": round(probs[1], 4), "neutral": round(probs[2], 4),
            "top_tokens": top5}


def run(args):
    model, vocab, ckpt = load_checkpoint(args.checkpoint_dir)
    print(f"\n  LSTM loaded  val_f1={ckpt.get('val_f1','?')}  vocab={len(vocab):,}  dataset=Amazon Polarity\n")
    texts = [args.text] if args.text else []
    if args.file:
        with open(args.file) as f: texts += [l.strip() for l in f if l.strip()]
    results = []
    for text in texts:
        r = predict(text, model, vocab)
        if not r: continue
        print(f'  {r["stars"]}  "{text[:80]}{"..." if len(text) > 80 else ""}"')
        print(f'  -> {r["emoji"]}  {r["sentiment"].upper()}   ({r["confidence"]*100:.1f}% confidence)')
        print(f'     pos={r["positive"]:.3f}   neg={r["negative"]:.3f}   neu={r["neutral"]:.3f}')
        print(f'     Top tokens: {", ".join(t for t, _ in r["top_tokens"])}\n')
        results.append({**r, "text": text, "top_tokens": r["top_tokens"]})
    if args.output and results:
        with open(args.output, "w") as f: json.dump(results, f, indent=2)
        print(f"  Saved -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--text",           default=None)
    p.add_argument("--file",           default=None)
    p.add_argument("--output",         default=None)
    run(p.parse_args())
