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
from examples.mimic4_los_tpc import (
    DEFAULT_LAB_ITEMS,
    BucketBatchSampler,
    LengthOfStayTask,
    generate_tpc_tensors,
    load_chartevents_cache,
    select_qualifying_labs,
)


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


def test_static_features_deterministic_and_one_hot():
    """Tests that static feature extraction is deterministic, known categories land in assigned slots, and blocks sum to 1."""
    patient = MockPatient(patient_id="p1", anchor_age=65, anchor_year=2020, stay_hours=24)
    patient.events["admissions"][0].attr_dict.update({
        "admission_location": "EMERGENCY ROOM",
        "insurance": "Medicare",
        "race": "WHITE",
    })
    patient.events["icustays"][0].attr_dict["first_careunit"] = "MICU"
    
    task = LengthOfStayTask()
    
    # 1. Deterministic across two task calls
    res1 = task(patient)
    res2 = task(patient)
    assert len(res1) == 1 and len(res2) == 1
    static1 = res1[0]["static"]
    static2 = res2[0]["static"]
    assert torch.equal(static1, static2)
    assert static1.shape[0] == task.static_dim
    
    # 2. Verify known category lands in its exact assigned slot and unknown slot stays 0.0
    known_cats = {
        "first_careunit": "MICU",
        "admission_location": "EMERGENCY ROOM",
        "insurance": "Medicare",
        "race": "WHITE",
    }
    offset = 11  # continuous & indicator features come first
    for cat_name, vocab in task.cat_vocabs.items():
        block_len = len(vocab) + 1  # includes unknown slot
        block = static1[offset : offset + block_len]
        assert block.sum().item() == pytest.approx(1.0)
        assert ((block == 0.0) | (block == 1.0)).all()
        val = known_cats[cat_name]
        assert static1[offset + vocab.index(val)].item() == 1.0
        assert static1[offset + len(vocab)].item() == 0.0  # unknown slot stays 0
        offset += block_len
    assert offset == task.static_dim

    # 3. Test unknown category handling sets the unknown slot and leaves known slots 0.0
    patient_unk = MockPatient(patient_id="p_unk", anchor_age=65, anchor_year=2020, stay_hours=24)
    patient_unk.events["admissions"][0].attr_dict.update({
        "admission_location": "UNSEEN_LOCATION",
        "insurance": "UNSEEN_INSURANCE",
        "race": "UNSEEN_RACE",
    })
    patient_unk.events["icustays"][0].attr_dict["first_careunit"] = "UNSEEN_CAREUNIT"
    res_unk = task(patient_unk)
    static_unk = res_unk[0]["static"]
    
    offset = 11
    for cat_name, vocab in task.cat_vocabs.items():
        block_len = len(vocab) + 1
        block = static_unk[offset : offset + block_len]
        assert block.sum().item() == pytest.approx(1.0)
        assert block[-1].item() == 1.0  # unknown slot set
        assert (block[:-1] == 0.0).all()  # all known slots stay 0
        offset += block_len


def test_baseline_values_outside_window_ignored(tmp_path):
    """Tests that baseline chartevents outside [-24h, +5h] of admission are ignored."""
    import gzip
    
    in_time = datetime(2020, 5, 1, 10, 0, 0)
    
    mock_dataset = type("MockDS", (), {
        "iter_patients": lambda self: [
            type("Patient", (), {
                "get_events": lambda self, ev_type: [
                    type("Stay", (), {
                        "timestamp": in_time,
                        "attr_dict": {"stay_id": "stay_100"}
                    })()
                ] if ev_type == "icustays" else []
            })()
        ]
    })()
    
    # 1. -26h before admission -> outside window, ignored
    # 2. -2h before admission -> inside window, weight = 75.0
    # 3. +1h after admission -> inside window and closer to intime, weight = 76.0
    # 4. +6h after admission -> outside window, ignored
    # 5. height at +6h -> outside window, ignored completely
    icu_dir = tmp_path / "icu"
    icu_dir.mkdir(parents=True)
    chart_file = icu_dir / "chartevents.csv.gz"
    
    csv_content = (
        "stay_id,charttime,itemid,valuenum\n"
        "stay_100,2020-04-30 08:00:00,226512,70.0\n"
        "stay_100,2020-05-01 08:00:00,226512,75.0\n"
        "stay_100,2020-05-01 11:00:00,226512,76.0\n"
        "stay_100,2020-05-01 16:00:00,226512,85.0\n"
        "stay_100,2020-05-01 16:00:00,226730,175.0\n"
    )
    with gzip.open(chart_file, "wt") as f:
        f.write(csv_content)
        
    _, baseline_static_by_stay = load_chartevents_cache(str(tmp_path), mock_dataset)
    
    assert "stay_100" in baseline_static_by_stay
    base = baseline_static_by_stay["stay_100"]
    # Weight at +1h should be selected (closest to admission intime within [-24h, +5h])
    assert "weight" in base
    assert base["weight"] == 76.0
    # Height was only recorded at +6h (outside [-24h, +5h]), so it must be ignored
    assert "height" not in base


def test_select_qualifying_labs_respects_stay_window():
    """Tests that select_qualifying_labs counts labs strictly within the ICU stay window."""
    in_time = datetime(2020, 5, 1, 10, 0, 0)
    p = MockPatient(
        patient_id="p_labs", anchor_age=40, anchor_year=2020, stay_hours=48, has_lab=False
    )

    def ev(item, hours):
        return type(
            "Event",
            (),
            {
                "timestamp": in_time + timedelta(hours=hours),
                "attr_dict": {"itemid": item, "valuenum": "1.0"},
            },
        )()

    labs = []
    for item in ["A1", "A2", "A3", "A4", "A5"]:  # 4 in-window obs each -> qualify
        labs += [ev(item, h) for h in (1, 5, 10, 20)]
    labs += [ev("X", 10)]                         # 1 in-window obs
    labs += [ev("X", -h) for h in range(1, 6)]    # 5 before admission
    labs += [ev("X", 49 + h) for h in range(5)]   # 5 after discharge
    p.events["labevents"] = labs

    ds = type("MockDS", (), {"iter_patients": lambda self: [p]})()
    selected, met = select_qualifying_labs(
        ds,
        train_pids={"p_labs"},
        min_prevalence=0.25,
        min_avg_obs=3.0,
        max_labs=45,
    )
    assert met is True
    assert set(selected) == {"A1", "A2", "A3", "A4", "A5"}
    assert "X" not in selected  # would qualify (11 obs) if the stay window were ignored


def test_select_qualifying_labs_fallback():
    """Tests that select_qualifying_labs triggers fallback when criteria are not met."""
    in_time = datetime(2020, 5, 1, 10, 0, 0)
    p = MockPatient(
        patient_id="p_fallback", anchor_age=40, anchor_year=2020, stay_hours=48, has_lab=False
    )
    p.events["labevents"] = [
        type(
            "Event",
            (),
            {
                "timestamp": in_time + timedelta(hours=10),
                "attr_dict": {"itemid": "50902", "valuenum": "101"},
            },
        )()
    ]

    mock_dataset = type("MockDS", (), {"iter_patients": lambda self: [p]})()

    # 1 observation is below the min_avg_obs=3.0 threshold, triggering fallback
    selected, threshold_met = select_qualifying_labs(
        mock_dataset,
        train_pids={"p_fallback"},
        min_prevalence=0.25,
        min_avg_obs=3.0,
        max_labs=45,
    )
    assert threshold_met is False
    assert selected == DEFAULT_LAB_ITEMS


def test_bucket_batch_sampler():
    """Tests that BucketBatchSampler yields each index once per epoch and replays identically after reset()."""
    lengths = [12, 14, 15, 20, 22, 25, 48, 50, 52, 100, 105, 110]
    batch_size = 3
    sampler = BucketBatchSampler(lengths, batch_size=batch_size, shuffle=True, seed=123)
    
    # 1. Epoch 0: All indices yielded exactly once
    batches_ep0 = list(sampler)
    all_indices_ep0 = [idx for batch in batches_ep0 for idx in batch]
    assert sorted(all_indices_ep0) == list(range(len(lengths)))
    assert len(batches_ep0) == 4
    for b in batches_ep0:
        assert len(b) == batch_size
        
    # 2. Epoch 1: Yields all indices, but shuffled batch order across epochs
    batches_ep1 = list(sampler)
    all_indices_ep1 = [idx for batch in batches_ep1 for idx in batch]
    assert sorted(all_indices_ep1) == list(range(len(lengths)))
    assert batches_ep0 != batches_ep1
    
    # 3. Reset: Replays epoch 0 sequence identically
    sampler.reset()
    batches_ep0_replay = list(sampler)
    assert batches_ep0_replay == batches_ep0