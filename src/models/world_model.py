"""
The world model: learns P(S_{t+1} | S_t, ..., S_{t-T+1}) via a next-state
regression head, with infiltration-probability and MITRE-stage classification
heads riding on the same learned latent dynamics (techsoln.md sec 5).

Both encoders (LSTM default, Transformer optional) expose the same
forward() contract, so rollout.py / explain.py / the demo app never need to
know which one is active.
"""
import math

import torch
import torch.nn as nn


class WorldModelLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 num_stages: int = 7, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.next_state_head = nn.Linear(hidden_dim, input_dim)
        self.infiltration_head = nn.Linear(hidden_dim, 1)
        self.stage_head = nn.Linear(hidden_dim, num_stages)

    def forward(self, x: torch.Tensor) -> dict:
        outputs, _ = self.encoder(x)                                  # (B, T, H)
        attn_weights = torch.softmax(self.attn_score(outputs).squeeze(-1), dim=1)  # (B, T)
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=1)            # (B, H)
        return {
            "next_state": self.next_state_head(context),
            "infiltration_logit": self.infiltration_head(context).squeeze(-1),
            "stage_logits": self.stage_head(context),
            "attention_weights": attn_weights,
        }


class _PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class WorldModelTransformer(nn.Module):
    """Config-swappable alternative encoder — same forward() contract as
    WorldModelLSTM. Multi-head self-attention (averaged across heads) plays
    the same explainability role the additive-attention layer plays above."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 num_stages: int = 7, dropout: float = 0.2, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = _PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.next_state_head = nn.Linear(hidden_dim, input_dim)
        self.infiltration_head = nn.Linear(hidden_dim, 1)
        self.stage_head = nn.Linear(hidden_dim, num_stages)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.pos_encoding(self.input_proj(x))
        outputs = self.encoder(h)
        attn_weights = torch.softmax(self.attn_score(outputs).squeeze(-1), dim=1)
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=1)
        return {
            "next_state": self.next_state_head(context),
            "infiltration_logit": self.infiltration_head(context).squeeze(-1),
            "stage_logits": self.stage_head(context),
            "attention_weights": attn_weights,
        }


def build_model(model_cfg: dict, input_dim: int) -> nn.Module:
    encoder = model_cfg.get("encoder", "lstm")
    kwargs = dict(input_dim=input_dim, hidden_dim=model_cfg.get("hidden_dim", 64),
                  num_layers=model_cfg.get("num_layers", 2),
                  num_stages=model_cfg.get("num_stages", 7),
                  dropout=model_cfg.get("dropout", 0.2))
    if encoder == "transformer":
        return WorldModelTransformer(**kwargs)
    if encoder == "lstm":
        return WorldModelLSTM(**kwargs)
    raise ValueError(f"Unknown model.encoder '{encoder}' — expected 'lstm' or 'transformer'")


def world_model_loss(pred: dict, y_next: torch.Tensor, y_inf: torch.Tensor,
                      y_stage: torch.Tensor, weights=(1.0, 1.0, 1.0), pos_weight=None):
    """
    pos_weight: scalar tensor upweighting the rare positive (infiltration)
    class in the BCE term, typically n_negative/n_positive on the training
    set. Without it, on a dataset where positives are ~1% of sequences, the
    model can drive BCE near zero by predicting negative for everything and
    never learns to detect the rare positives (this is what pos_weight
    corrects for).
    """
    mse = nn.functional.mse_loss(pred["next_state"], y_next)
    bce = nn.functional.binary_cross_entropy_with_logits(
        pred["infiltration_logit"], y_inf, pos_weight=pos_weight
    )
    ce = nn.functional.cross_entropy(pred["stage_logits"], y_stage)
    total = weights[0] * mse + weights[1] * bce + weights[2] * ce
    return total, {"mse": mse.item(), "bce": bce.item(), "ce": ce.item()}
