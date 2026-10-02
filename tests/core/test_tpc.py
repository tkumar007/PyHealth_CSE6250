import torch
import pytest
from pyhealth.models.tpc import TPC


class MockDataset:
    """Mock PyHealth dataset to validate BaseModel compatibility."""
    def __init__(self):
        self.tokens = {"static": ["token1", "token2", "token3"]}
        self.input_schema = {
            "static": "tensor",
            "vitals": "tensor",
            "vitals_decay": "tensor",
        }
        self.output_schema = {
            "los": "tensor",
        }

    def get_all_tokens(self, key):
        return self.tokens.get(key, [])


@pytest.mark.parametrize("batch_size, seq_len, num_vitals, num_static", [
    (2, 24, 12, 2),  # Standard 24-hour window
    (4, 48, 12, 2),  # 48-hour window, larger batch
])
def test_tpc_sequence_forward_and_loss(batch_size, seq_len, num_vitals, num_static):
    """Tests the sequence-level hourly remaining LoS prediction and MSLE loss."""
    mock_dataset = MockDataset()
    
    # 1. Inputs
    vitals = torch.randn(batch_size, seq_len, num_vitals)
    vitals_time_delta = torch.rand(batch_size, seq_len, num_vitals) * 5.0
    static = torch.randn(batch_size, num_static)
    # Remaining LoS targets in days (> 0)
    mock_labels = torch.rand(batch_size, seq_len, 1) * 10.0 + 0.1
    
    # 2. Instantiate TPC model with default MSLE loss & sequence outputs
    model = TPC(
        dataset=mock_dataset,
        continuous_dim=num_vitals,
        static_dim=num_static,
        n_layers=3,
        temp_dim=8,
        point_dim=16,
        kernel_size=3,
        loss_fn="msle",
        sequence_output=True,
    )
    
    # 3. Forward pass
    output = model(
        static=static,
        vitals=vitals,
        vitals_time_delta=vitals_time_delta,
        y_true=mock_labels,
    )
    
    # 4. Assertions
    assert "loss" in output, "Model must return a loss"
    assert "y_prob" in output, "Model must return y_prob"
    # Sequence output shape should match (batch_size, seq_len, 1)
    assert output["y_prob"].shape == (batch_size, seq_len, 1), "Hourly sequence output shape mismatch"
    assert not torch.isnan(output["loss"]), "MSLE loss evaluated to NaN"
    assert (output["y_prob"] > 0).all(), "Predictions must be strictly positive for LoS"


def test_tpc_causality():
    """Verifies strict causal convolution: future observations cannot leak into past predictions."""
    batch_size = 1
    seq_len = 24
    num_vitals = 12
    
    model = TPC(
        dataset=None,
        continuous_dim=num_vitals,
        static_dim=0,
        n_layers=3,
        kernel_size=3,
        sequence_output=True,
    )
    model.eval()
    
    vitals1 = torch.randn(batch_size, seq_len, num_vitals)
    vitals2 = vitals1.clone()
    
    # Alter future timesteps starting at t = 16
    perturb_step = 16
    vitals2[:, perturb_step:, :] += 50.0
    
    with torch.no_grad():
        out1 = model(vitals=vitals1)["y_prob"]
        out2 = model(vitals=vitals2)["y_prob"]
        
    # Past predictions (t < perturb_step) must be identical
    past_diff = (out1[:, :perturb_step, :] - out2[:, :perturb_step, :]).abs().max().item()
    future_diff = (out1[:, perturb_step:, :] - out2[:, perturb_step:, :]).abs().max().item()
    
    assert past_diff == 0.0, f"Future information leaked into past! Diff: {past_diff}"
    assert future_diff > 0.0, "Perturbation in future should affect future predictions"


def test_tpc_pooled_summary():
    """Tests optional pooled single-patient summary prediction."""
    batch_size = 3
    seq_len = 24
    num_vitals = 12
    
    model = TPC(
        dataset=None,
        continuous_dim=num_vitals,
        static_dim=0,
        sequence_output=False,
    )
    
    vitals = torch.randn(batch_size, seq_len, num_vitals)
    y_true = torch.rand(batch_size, 1) * 5.0 + 0.1
    
    output = model(vitals=vitals, y_true=y_true)
    assert output["y_prob"].shape == (batch_size, 1), "Pooled shape must be (batch_size, 1)"
    assert not torch.isnan(output["loss"]), "Loss evaluated to NaN"


def test_tpc_multitask():
    """Tests multitask learning with both LoS and in-hospital mortality."""
    batch_size = 2
    seq_len = 24
    num_vitals = 12
    
    model = TPC(
        dataset=None,
        continuous_dim=num_vitals,
        multitask=True,
        alpha=0.5,
    )
    
    vitals = torch.randn(batch_size, seq_len, num_vitals)
    y_true = torch.rand(batch_size, seq_len, 1) * 5.0 + 0.1
    mortality = torch.randint(0, 2, (batch_size, 1)).float()
    
    output = model(
        vitals=vitals,
        y_true=y_true,
        mortality=mortality,
    )
    
    assert "loss" in output
    assert "y_prob" in output
    assert "mortality_prob" in output
    assert output["mortality_prob"].shape == (batch_size, seq_len, 1)
    assert ((output["mortality_prob"] >= 0.0) & (output["mortality_prob"] <= 1.0)).all()