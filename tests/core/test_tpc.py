import sys
from pathlib import Path
from datetime import datetime, timedelta
import torch
import pytest

# Ensure repository root is on sys.path regardless of execution directory
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pyhealth.models.tpc import TPC
from pyhealth.datasets import collate_fn_dict_with_padding
from examples.mimic4_los_tpc import generate_tpc_tensors, LengthOfStayTask


class MockDataset:
    """Mock PyHealth dataset to validate BaseModel compatibility."""
    def __init__(self):
        self.tokens = {"static": ["token1", "token2", "token3"]}
        self.input_schema = {
            "static": "tensor",
            "vitals": "tensor",
            "vitals_decay": "tensor",
            "mask": "tensor",
        }
        self.output_schema = {
            "los": "tensor",
            "mortality": "tensor",
        }

    def get_all_tokens(self, key):
        return self.tokens.get(key, [])


class MockPatient:
    """Minimal mock patient for unit testing LengthOfStayTask logic."""
    def __init__(self, patient_id="p0", anchor_age=50, anchor_year=2020, stay_hours=24, has_lab=True):
        self.patient_id = patient_id
        in_time = datetime(2020, 5, 1, 10, 0, 0)
        out_time = in_time + timedelta(hours=stay_hours)
        
        self.events = {
            "patients": [
                type("Event", (), {
                    "attr_dict": {
                        "anchor_age": str(anchor_age),
                        "anchor_year": str(anchor_year),
                        "gender": "F",
                    }
                })()
            ],
            "admissions": [
                type("Event", (), {
                    "attr_dict": {
                        "hadm_id": "h0",
                        "hospital_expire_flag": "0",
                    }
                })()
            ],
            "icustays": [
                type("Event", (), {
                    "timestamp": in_time,
                    "attr_dict": {
                        "stay_id": "s0",
                        "hadm_id": "h0",
                        "outtime": out_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                })()
            ],
            "labevents": [
                type("Event", (), {
                    "timestamp": in_time + timedelta(hours=2),
                    "attr_dict": {
                        "itemid": "50902",
                        "valuenum": "105.0",
                    }
                })()
            ] if has_lab else [],
        }

    def get_events(self, event_type):
        return self.events.get(event_type, [])


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
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
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
    )
    
    # 3. Forward pass
    output = model(
        static=static,
        vitals=vitals,
        vitals_time_delta=vitals_time_delta,
        mask=mask,
        y_true=mock_labels,
    )
    
    # 4. Assertions
    assert "loss" in output, "Model must return a loss"
    assert "y_prob" in output, "Model must return y_prob"
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
    
    assert past_diff < 1e-6, f"Future information leaked into past! Diff: {past_diff}"
    assert future_diff > 0.0, "Perturbation in future should affect future predictions"


def test_tpc_padding_invariance():
    """Verifies that zero-padding steps (where mask = 0) do not contaminate loss calculation."""
    num_vitals = 12
    model = TPC(dataset=None, continuous_dim=num_vitals)
    model.eval()
    
    # Patient 1: real stay of 10 hours
    T_valid = 10
    vitals_valid = torch.randn(1, T_valid, num_vitals)
    y_true_valid = torch.rand(1, T_valid, 1) + 0.5
    mask_valid = torch.ones(1, T_valid, dtype=torch.bool)
    
    loss_unpadded = model(vitals=vitals_valid, y_true=y_true_valid, mask=mask_valid)["loss"].item()
    
    # Patient 1 padded with 10 zero-padding hours (total T = 20)
    T_padded = 20
    vitals_padded = torch.zeros(1, T_padded, num_vitals)
    vitals_padded[:, :T_valid, :] = vitals_valid
    
    y_true_padded = torch.zeros(1, T_padded, 1)
    y_true_padded[:, :T_valid, :] = y_true_valid
    
    mask_padded = torch.zeros(1, T_padded, dtype=torch.bool)
    mask_padded[:, :T_valid] = True
    
    # Force identical predictions on the valid segment by passing unpadded features
    with torch.no_grad():
        out_padded = model(vitals=vitals_padded, y_true=y_true_padded, mask=mask_padded)
        loss_padded = model._compute_loss(out_padded["y_prob"], y_true_padded, mask=mask_padded).item()
        
    assert not torch.isnan(torch.tensor(loss_padded))
    assert loss_padded > 0.0
    assert abs(loss_padded - loss_unpadded) < 1e-5, f"Padded loss ({loss_padded}) does not match unpadded ({loss_unpadded})"


def test_generate_tpc_tensors():
    """Tests forward-filling and exponential decay indicator 0.75^j calculation."""
    seq_len = 4
    num_features = 2
    raw = torch.tensor([
        [10.0, 0.0],
        [0.0,  5.0],
        [0.0,  0.0],
        [20.0, 0.0],
    ])
    mask = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    
    vitals_ff, decay = generate_tpc_tensors(raw, mask)
    
    # Feature 0: observed at t=0 (10.0), missing at t=1,2 (forward-filled 10.0), observed at t=3 (20.0)
    assert vitals_ff[0, 0].item() == 10.0
    assert vitals_ff[1, 0].item() == 10.0
    assert vitals_ff[2, 0].item() == 10.0
    assert vitals_ff[3, 0].item() == 20.0
    
    # Decay for feature 0: t=0 -> 0.75^0=1.0, t=1 -> 0.75^1=0.75, t=2 -> 0.75^2=0.5625, t=3 -> 0.75^0=1.0
    assert abs(decay[0, 0].item() - 1.0) < 1e-5
    assert abs(decay[1, 0].item() - 0.75) < 1e-5
    assert abs(decay[2, 0].item() - 0.5625) < 1e-5
    assert abs(decay[3, 0].item() - 1.0) < 1e-5
    
    # Feature 1: never observed before t=1, so t=0 decay is 0.0
    assert decay[0, 1].item() == 0.0
    assert decay[1, 1].item() == 1.0


def test_task_filters_and_target():
    """Tests LengthOfStayTask filters (age > 18, stay >= 5h, has observation) and target format."""
    task = LengthOfStayTask(lab_item_ids=["50902"])
    
    # Adult with valid stay and observation
    p_valid = MockPatient(anchor_age=50, stay_hours=10, has_lab=True)
    samples = task(p_valid)
    assert len(samples) == 1
    assert samples[0]["los"].shape == (10, 1)
    assert samples[0]["mask"].shape == (10,)
    assert samples[0]["vitals"].shape == (10, 1)
    # Target remaining LoS should be strictly descending
    assert samples[0]["los"][0].item() > samples[0]["los"][-1].item()
    
    # Child (< 18) should be skipped
    p_child = MockPatient(anchor_age=16, stay_hours=10, has_lab=True)
    assert len(task(p_child)) == 0
    
    # Short stay (< 5h) should be skipped
    p_short = MockPatient(anchor_age=50, stay_hours=3, has_lab=True)
    assert len(task(p_short)) == 0
    
    # Stay with no observations should be skipped
    p_nolab = MockPatient(anchor_age=50, stay_hours=10, has_lab=False)
    assert len(task(p_nolab)) == 0


def test_tpc_collate_and_training_padding():
    """Tests PyHealth collation padding of variable-length stays and training stability."""
    num_vitals = 12
    T1, T2 = 8, 16
    
    # Patient 1: 8 hours
    s1 = {
        "vitals": torch.randn(T1, num_vitals),
        "vitals_decay": torch.rand(T1, num_vitals),
        "static": torch.tensor([0.5, 1.0]),
        "mask": torch.ones(T1, dtype=torch.bool),
        "los": torch.rand(T1, 1) + 0.1,
    }
    # Patient 2: 16 hours
    s2 = {
        "vitals": torch.randn(T2, num_vitals),
        "vitals_decay": torch.rand(T2, num_vitals),
        "static": torch.tensor([0.6, 0.0]),
        "mask": torch.ones(T2, dtype=torch.bool),
        "los": torch.rand(T2, 1) + 0.1,
    }
    
    batch = collate_fn_dict_with_padding([s1, s2])
    
    # 1. Verify collation shapes and padding masks
    assert batch["mask"].shape == (2, T2)
    assert batch["mask"].sum(dim=1).tolist() == [T1, T2]
    assert (batch["mask"][0, T1:] == False).all()
    assert (batch["los"][0, T1:] == 0.0).all()
    assert (batch["vitals"][0, T1:] == 0.0).all()
    
    # 2. Verify model execution in train() mode with optimizer backward step
    model = TPC(dataset=None, continuous_dim=num_vitals, static_dim=2)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    out = model(**batch)
    assert "loss" in out
    assert not torch.isnan(out["loss"])
    out["loss"].backward()
    optimizer.step()


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


def test_tpc_no_decay_ablation():
    """Tests TPC ablation variant without decay channels (use_decay=False)."""
    batch_size = 2
    seq_len = 16
    num_vitals = 12
    
    model = TPC(dataset=None, continuous_dim=num_vitals, use_decay=False)
    vitals = torch.randn(batch_size, seq_len, num_vitals)
    y_true = torch.rand(batch_size, seq_len, 1) + 0.1
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Layer 1 input should have 1 channel per feature (instead of 2)
    assert model.blocks[0].C_n == 1
    # Pointwise input dimension should exclude decay indicators
    assert model.blocks[0].P_n == num_vitals * 1
    
    out = model(vitals=vitals, y_true=y_true, mask=mask)
    assert out["y_prob"].shape == (batch_size, seq_len, 1)
    assert not torch.isnan(out["loss"])