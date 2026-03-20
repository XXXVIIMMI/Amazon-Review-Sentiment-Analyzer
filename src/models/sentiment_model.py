import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor, pad_mask: torch.Tensor = None):
        scores = self.v(torch.tanh(self.W(lstm_out))).squeeze(-1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)
        return context, weights


class AmazonLSTMSentiment(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = 128,
        hidden_dim:    int   = 256,
        num_classes:   int   = 3,
        num_layers:    int   = 2,
        dropout:       float = 0.5,
        bidirectional: bool  = True,
        pad_idx:       int   = 0,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.bidirectional = bidirectional
        self.D             = 2 if bidirectional else 1

        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = bidirectional,
        )

        self.attention = BahdanauAttention(hidden_dim * self.D)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.D, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        pad_mask = (x == 0)
        emb      = self.embed_drop(self.embedding(x))

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _   = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            pad_mask       = pad_mask[:, :lstm_out.size(1)]
        else:
            lstm_out, _ = self.lstm(emb)

        context, attn_w = self.attention(lstm_out, pad_mask)
        logits          = self.classifier(context)
        return logits, attn_w

    @torch.no_grad()
    def predict(self, x: torch.Tensor, lengths: torch.Tensor = None):
        self.eval()
        logits, attn = self.forward(x, lengths)
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(-1), probs, attn


def build_model(vocab_size: int, **kwargs) -> AmazonLSTMSentiment:
    defaults = dict(
        embed_dim=128, hidden_dim=256, num_classes=3,
        num_layers=2,  dropout=0.5,    bidirectional=True,
    )
    defaults.update(kwargs)
    return AmazonLSTMSentiment(vocab_size=vocab_size, **defaults)
