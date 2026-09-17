"""
Rollout correctness tests: verify K-step autoregressive forecasting.

Tests cover:
- Trajectory shape and bounds (infiltration ∈ [0, 1])
- Stage validity (0-6 MITRE stages)
- MC-dropout uncertainty growth
- Non-trivial predictions (not constant)
"""
import numpy as np
import pytest
import torch

from src.rollout import rollout, rollout_with_uncertainty, _load_artifacts
from src.mitre_mapping import STAGE_NAMES


@pytest.fixture
def model_and_data():
    """Load trained model and test data once per test session."""
    model, scaler, cfg = _load_artifacts()
    data = np.load("data/processed/test_split.npz")
    X_test, y_inf_test = data["X_test"], data["y_inf_test"]

    # Pick a benign sequence for deterministic testing
    benign_idx = np.where(y_inf_test == 0)[0][0]
    x_sample = X_test[benign_idx:benign_idx+1]  # Shape: (1, T, F)
    x_tensor = torch.from_numpy(x_sample).float()

    return model, cfg, x_tensor, scaler


def test_rollout_produces_trajectory(model_and_data):
    """Test that rollout produces a valid K-step trajectory."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]

    result = rollout(model, x_tensor, K=K)

    # Check trajectory lengths
    assert len(result["infiltration_prob_trajectory"]) == K
    assert len(result["predicted_stage_trajectory"]) == K
    assert len(result["predicted_stage_names"]) == K
    assert len(result["imagined_states"]) == K
    assert len(result["attention_trace"]) == K


def test_infiltration_prob_bounds(model_and_data):
    """Test that infiltration probabilities stay in [0, 1]."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]

    result = rollout(model, x_tensor, K=K)
    probs = result["infiltration_prob_trajectory"]

    # All probabilities should be valid
    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability {p} out of bounds [0, 1]"


def test_stage_validity(model_and_data):
    """Test that stage predictions are valid (0-6 MITRE stages)."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]
    num_stages = cfg["model"]["num_stages"]

    result = rollout(model, x_tensor, K=K)
    stages = result["predicted_stage_trajectory"]

    # All stages should be valid indices
    for stage in stages:
        assert 0 <= stage < num_stages, f"Stage {stage} out of range [0, {num_stages})"
        assert stage < len(STAGE_NAMES), "Stage name missing"


def test_state_dimension_consistency(model_and_data):
    """Test that predicted states match input feature dimension."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]

    # Get input dimension from x_tensor
    input_dim = x_tensor.shape[-1]

    result = rollout(model, x_tensor, K=K)
    imagined_states = result["imagined_states"]

    # Each imagined state should match input dimension
    for state in imagined_states:
        assert len(state) == input_dim, \
            f"State dimension {len(state)} != input_dim {input_dim}"


def test_attention_weights_valid(model_and_data):
    """Test that attention weights are valid (sum to 1, all positive)."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]
    T = cfg["data"]["history_length"]

    result = rollout(model, x_tensor, K=K)
    attn_traces = result["attention_trace"]

    for attn in attn_traces:
        # Should have T attention weights (one per history window)
        assert len(attn) == T, f"Expected {T} attention weights, got {len(attn)}"

        # All weights should be non-negative and sum to 1
        attn_arr = np.array(attn)
        assert np.all(attn_arr >= 0), "Attention weights should be non-negative"
        assert np.isclose(np.sum(attn_arr), 1.0), \
            f"Attention weights should sum to 1, got {np.sum(attn_arr)}"


def test_nontrivial_predictions(model_and_data):
    """Test that rollout produces non-trivial predictions (not constant)."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]

    result = rollout(model, x_tensor, K=K)
    probs = result["infiltration_prob_trajectory"]

    # At least one prediction should be different from the first
    first_prob = probs[0]
    assert not all(p == first_prob for p in probs), \
        "Predictions should vary across steps, not constant"


def test_mc_dropout_produces_samples(model_and_data):
    """Test that MC-dropout produces multiple samples with variation."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]
    n_samples = 10

    result = rollout_with_uncertainty(model, x_tensor, K=K, n_samples=n_samples)

    # Should have prob_mean and prob_std
    assert "prob_mean" in result
    assert "prob_std" in result
    assert len(result["prob_mean"]) == K
    assert len(result["prob_std"]) == K


def test_uncertainty_is_nonnegative(model_and_data):
    """Test that MC-dropout uncertainty (std) is non-negative."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]

    result = rollout_with_uncertainty(model, x_tensor, K=K, n_samples=20)
    stds = result["prob_std"]

    # All standard deviations should be non-negative
    for std in stds:
        assert std >= 0.0, f"Uncertainty std {std} should be non-negative"


def test_uncertainty_mean_in_bounds(model_and_data):
    """Test that MC-dropout mean estimates stay in [0, 1]."""
    model, cfg, x_tensor, _ = model_and_data
    K = cfg["data"]["horizon"]

    result = rollout_with_uncertainty(model, x_tensor, K=K, n_samples=20)
    means = result["prob_mean"]

    # All means should be valid probabilities
    for mean in means:
        assert 0.0 <= mean <= 1.0, f"Mean {mean} out of bounds [0, 1]"


def test_multiple_sequences_independent(model_and_data):
    """Test that different input sequences produce different rollouts."""
    model, cfg, data_tensor, _ = model_and_data
    # Get multiple sequences
    K = cfg["data"]["horizon"]

    # We have at least 2 test sequences; use first two
    data = np.load("data/processed/test_split.npz")
    X_test = data["X_test"]

    if len(X_test) >= 2:
        x1 = torch.from_numpy(X_test[0:1]).float()
        x2 = torch.from_numpy(X_test[1:2]).float()

        result1 = rollout(model, x1, K=K)
        result2 = rollout(model, x2, K=K)

        probs1 = result1["infiltration_prob_trajectory"]
        probs2 = result2["infiltration_prob_trajectory"]

        # At least some probabilities should differ
        assert probs1 != probs2, \
            "Different input sequences should produce different trajectories"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
