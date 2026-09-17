"""
Static neural baseline for ablation: single-window MLP with no temporal modeling.

This baseline takes only the current window's features (no history) and predicts:
- next_state: same shape as temporal models for fair comparison
- infiltration_logit: binary classification logit
- stage_logit: multi-class stage prediction

Demonstrates the importance of temporal context by comparing to the world model.
"""
import torch
import torch.nn as nn


class StaticMLP(nn.Module):
    """
    Single-window MLP: takes features from the current window only, no history.
    Mirrors the world model's output structure for direct comparison in benchmarks.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, num_stages: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        # Input projection (no temporal dimension)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Hidden layers
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.fc_hidden = nn.Sequential(*layers)

        # Output heads (same structure as world model for fair comparison)
        self.next_state_head = nn.Linear(hidden_dim, input_dim)
        self.infiltration_head = nn.Linear(hidden_dim, 1)
        self.stage_head = nn.Linear(hidden_dim, num_stages)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) - but only use x[:, -1, :]

        Returns:
            dict with keys: next_state, infiltration_logit, stage_logit
        """
        # Take only the last window (no temporal modeling)
        x_latest = x[:, -1, :]  # (batch_size, input_dim)

        # Forward pass
        h = self.fc_in(x_latest)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.fc_hidden(h)

        # Output heads
        return {
            "next_state": self.next_state_head(h),
            "infiltration_logit": self.infiltration_head(h).squeeze(-1),
            "stage_logit": self.stage_head(h),
        }


def build_static_model(model_cfg: dict, input_dim: int) -> nn.Module:
    """Factory function for StaticMLP, matching world_model.build_model signature."""
    kwargs = dict(
        input_dim=input_dim,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
        num_stages=model_cfg.get("num_stages", 7),
    )
    return StaticMLP(**kwargs)
