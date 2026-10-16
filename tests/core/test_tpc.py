"""Author: Tamjit Kumar
NetID: tkumar70
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Unit tests for TPC model architecture, forward-filling & decay indicators,
             eICU length of stay task logic, and unconfounded batch sampling.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure repository root is on sys.path regardless of execution directory
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pyhealth.datasets import collate_fn_dict_with_padding, eICUDataset
from pyhealth.models.tpc import TPC
from pyhealth.tasks import LengthOfStayTPCeICU
from pyhealth.tasks.length_of_stay_tpc_eicu import (
    DEFAULT_LAB_NAMES,
    BucketBatchSampler,
    MapDataset,
    compute_empirical_training_stats,
    generate_tpc_tensors,
    get_valid_eicu_stay_ids,
    load_eicu_vitals_cache,
    select_qualifying_labs,
)

LengthOfStayTask = LengthOfStayTPCeICU


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

    def __getitem__(self, idx):
        return {
            "vitals": torch.randn(24, 16),
            "static": torch.randn(44),
            "los": torch.ones(24, 1),
            "mortality": torch.tensor([0.0]),
        }


class MockPatient:
    """Minimal mock patient for unit testing eICU LengthOfStayTask logic."""

    def __init__(
        self,
        patient_id: str = "p0",
        age: float = 50.0,
        stay_hours: int = 24,
        has_lab: bool = True,
        unittype: str = "Med-Surg ICU",
        unitadmitsource: str = "Emergency Department",
        unitstaytype: str = "admit",
        ethnicity: str = "Caucasian",
    ):
        self.patient_id = patient_id
        discharge_offset = float(stay_hours * 60)  # minutes

        self.events = {
            "patient": [
                type(
                    "Event",
                    (),
                    {
                        "patientunitstayid": "s0",
                        "unitdischargeoffset": discharge_offset,
                        "gender": "Female",
                        "age": str(age),
                        "ethnicity": ethnicity,
                        "admissionheight": 165.0,
                        "admissionweight": 65.0,
                        "unittype": unittype,
                        "unitadmittime24": "14:00:00",
                        "unitadmitsource": unitadmitsource,
                        "unitvisitnumber": 1,
                        "unitstaytype": unitstaytype,
                        "hospitaldischargestatus": "Alive",
                        "unitdischargestatus": "Alive",
                    },
                )()
            ],
            "lab": [
                type(
                    "Event",
                    (),
                    {
                        "patientunitstayid": "s0",
                        "labresultoffset": 120.0,  # 2 hours into stay
                        "labname": "potassium",
                        "labresult": 4.2,
                    },
                )()
            ]
            if has_lab
            else [],
        }

    def get_events(self, event_type: str):
        return self.events.get(event_type, [])


class TestTPCModel(unittest.TestCase):
    """Unit tests for the TPC model architecture and forward pass."""

    def test_tpc_sequence_forward_and_loss(self):
        """Tests sequence-level hourly LoS prediction and MSLE loss."""
        for batch_size, seq_len, num_vitals, num_static in [
            (2, 24, 12, 2),
            (4, 48, 12, 2),
        ]:
            mock_dataset: Any = MockDataset()
            vitals = torch.randn(batch_size, seq_len, num_vitals)
            vitals_time_delta = (
                torch.rand(batch_size, seq_len, num_vitals) * 5.0
            )
            static = torch.randn(batch_size, num_static)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            mock_labels = torch.rand(batch_size, seq_len, 1) * 10.0 + 0.1

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

            output = model(
                static=static,
                vitals=vitals,
                vitals_time_delta=vitals_time_delta,
                mask=mask,
                y_true=mock_labels,
            )

            self.assertIn("loss", output)
            self.assertIn("y_prob", output)
            self.assertEqual(
                output["y_prob"].shape, (batch_size, seq_len, 1)
            )
            self.assertFalse(torch.isnan(output["loss"]))
            self.assertTrue((output["y_prob"] > 0).all())

            # Test transposed labels (B, 1, T)
            transposed_labels = mock_labels.transpose(1, 2)
            out_trans = model(
                vitals=vitals,
                static=static,
                y_true=transposed_labels,
                mask=mask,
            )
            self.assertIn("loss", out_trans)
            self.assertFalse(torch.isnan(out_trans["loss"]))

    def test_tpc_causality(self):
        """Verifies strict causal convolution without future lookahead."""
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

        past_diff = (
            (out1[:, :perturb_step, :] - out2[:, :perturb_step, :])
            .abs()
            .max()
            .item()
        )
        future_diff = (
            (out1[:, perturb_step:, :] - out2[:, perturb_step:, :])
            .abs()
            .max()
            .item()
        )

        self.assertLess(past_diff, 1e-6)
        self.assertGreater(future_diff, 0.0)

    def test_tpc_padding_invariance(self):
        """Verifies that zero-padding steps do not contaminate loss."""
        num_vitals = 12
        model = TPC(dataset=None, continuous_dim=num_vitals)
        model.eval()

        # Patient 1: real stay of 10 hours
        T_valid = 10
        vitals_valid = torch.randn(1, T_valid, num_vitals)
        y_true_valid = torch.rand(1, T_valid, 1) + 0.5
        mask_valid = torch.ones(1, T_valid, dtype=torch.bool)

        loss_unpadded = model(
            vitals=vitals_valid, y_true=y_true_valid, mask=mask_valid
        )["loss"].item()

        # Patient 1 padded with 10 zero-padding hours (total T = 20)
        T_padded = 20
        vitals_padded = torch.zeros(1, T_padded, num_vitals)
        vitals_padded[:, :T_valid, :] = vitals_valid

        y_true_padded = torch.zeros(1, T_padded, 1)
        y_true_padded[:, :T_valid, :] = y_true_valid

        mask_padded = torch.zeros(1, T_padded, dtype=torch.bool)
        mask_padded[:, :T_valid] = True

        with torch.no_grad():
            out_padded = model(
                vitals=vitals_padded, y_true=y_true_padded, mask=mask_padded
            )
            loss_padded = model._compute_loss(
                out_padded["y_prob"], y_true_padded, mask=mask_padded
            ).item()

        self.assertFalse(torch.isnan(torch.tensor(loss_padded)))
        self.assertGreater(loss_padded, 0.0)
        self.assertAlmostEqual(loss_padded, loss_unpadded, places=5)

    def test_generate_tpc_tensors(self):
        """Tests forward-filling and exponential decay indicator calculation."""
        raw = torch.tensor(
            [
                [10.0, 0.0],
                [0.0, 5.0],
                [0.0, 0.0],
                [20.0, 0.0],
            ]
        )
        mask = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

        vitals_ff, decay = generate_tpc_tensors(raw, mask)

        self.assertEqual(vitals_ff[0, 0].item(), 10.0)
        self.assertEqual(vitals_ff[1, 0].item(), 10.0)
        self.assertEqual(vitals_ff[2, 0].item(), 10.0)
        self.assertEqual(vitals_ff[3, 0].item(), 20.0)

        self.assertAlmostEqual(decay[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(decay[1, 0].item(), 0.75, places=5)
        self.assertAlmostEqual(decay[2, 0].item(), 0.5625, places=5)
        self.assertAlmostEqual(decay[3, 0].item(), 1.0, places=5)

        self.assertEqual(decay[0, 1].item(), 0.0)
        self.assertEqual(decay[1, 1].item(), 1.0)

    def test_tpc_collate_and_training_padding(self):
        """Tests PyHealth collation padding and model training stability."""
        num_vitals = 12
        T1, T2 = 8, 16

        s1 = {
            "vitals": torch.randn(T1, num_vitals),
            "vitals_decay": torch.rand(T1, num_vitals),
            "static": torch.tensor([0.5, 1.0]),
            "mask": torch.ones(T1, dtype=torch.bool),
            "los": torch.rand(T1, 1) + 0.1,
        }
        s2 = {
            "vitals": torch.randn(T2, num_vitals),
            "vitals_decay": torch.rand(T2, num_vitals),
            "static": torch.tensor([0.6, 0.0]),
            "mask": torch.ones(T2, dtype=torch.bool),
            "los": torch.rand(T2, 1) + 0.1,
        }

        batch = collate_fn_dict_with_padding([s1, s2])

        self.assertEqual(batch["mask"].shape, (2, T2))
        self.assertEqual(batch["mask"].sum(dim=1).tolist(), [T1, T2])
        self.assertTrue((batch["mask"][0, T1:] == False).all())
        self.assertTrue((batch["los"][0, T1:] == 0.0).all())
        self.assertTrue((batch["vitals"][0, T1:] == 0.0).all())

        model = TPC(dataset=None, continuous_dim=num_vitals, static_dim=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertFalse(torch.isnan(out["loss"]))
        out["loss"].backward()
        optimizer.step()

    def test_tpc_full_gradient_flow(self):
        """Verifies that gradients flow to all model parameters without NaN."""
        batch_size = 2
        seq_len = 16
        num_vitals = 6
        static_dim = 4

        model = TPC(
            dataset=None,
            continuous_dim=num_vitals,
            static_dim=static_dim,
            n_layers=2,
            multitask=True,
        )
        model.train()

        vitals = torch.randn(batch_size, seq_len, num_vitals)
        vitals_decay = torch.rand(batch_size, seq_len, num_vitals)
        static = torch.randn(batch_size, static_dim)
        y_true = torch.rand(batch_size, seq_len, 1) + 0.1
        mortality = torch.randint(0, 2, (batch_size, 1)).float()

        out = model(
            vitals=vitals,
            vitals_decay=vitals_decay,
            static=static,
            y_true=y_true,
            mortality=mortality,
        )
        self.assertIn("loss", out)
        out["loss"].backward()

        for name, param in model.named_parameters():
            if name != "_dummy_param" and param.requires_grad:
                self.assertIsNotNone(
                    param.grad, f"Parameter '{name}' received no gradient."
                )
                assert param.grad is not None
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"Parameter '{name}' received NaN gradient.",
                )

    def test_tpc_multitask(self):
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

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("mortality_prob", output)
        self.assertEqual(output["mortality_prob"].shape, (batch_size, seq_len, 1))
        self.assertTrue(
            (
                (output["mortality_prob"] >= 0.0)
                & (output["mortality_prob"] <= 1.0)
            ).all()
        )

        # Test with 3D mask (B, T, 1)
        mask_3d = torch.ones(batch_size, seq_len, 1, dtype=torch.bool)
        output_3d = model(
            vitals=vitals,
            y_true=y_true,
            mortality=mortality,
            mask=mask_3d,
        )
        self.assertIn("loss", output_3d)
        self.assertFalse(torch.isnan(output_3d["loss"]))

        # Test with sequence mortality (B, T) and scalar y_true (B,)
        mort_seq = torch.randint(0, 2, (batch_size, seq_len)).float()
        y_scalar = torch.tensor([2.5, 3.5])
        out_seq = model(
            vitals=vitals,
            y_true=y_scalar,
            mortality=mort_seq,
        )
        self.assertIn("loss", out_seq)
        self.assertFalse(torch.isnan(out_seq["loss"]))
        self.assertEqual(out_seq["y_prob"].shape, (batch_size, seq_len, 1))

    def test_tpc_no_decay_ablation(self):
        """Tests TPC ablation variant without decay channels (use_decay=False)."""
        batch_size = 2
        seq_len = 16
        num_vitals = 12

        model = TPC(dataset=None, continuous_dim=num_vitals, use_decay=False)
        vitals = torch.randn(batch_size, seq_len, num_vitals)
        y_true = torch.rand(batch_size, seq_len, 1) + 0.1
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        self.assertEqual(model.blocks[0].C_n, 1)
        self.assertEqual(model.blocks[0].P_n, num_vitals * 1)

        out = model(vitals=vitals, y_true=y_true, mask=mask)
        self.assertEqual(out["y_prob"].shape, (batch_size, seq_len, 1))
        self.assertFalse(torch.isnan(out["loss"]))

    def test_tpc_save_and_load_state(self):
        """Verifies that model state can be saved and loaded accurately."""
        model = TPC(dataset=None, continuous_dim=12, static_dim=5)
        model.eval()
        vitals = torch.randn(2, 24, 12)
        static = torch.randn(2, 5)

        with torch.no_grad():
            out1 = model(vitals=vitals, static=static)["y_prob"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            weight_path = Path(tmp_dir) / "tpc_weights.pt"
            torch.save(model.state_dict(), weight_path)

            loaded_model = TPC(dataset=None, continuous_dim=12, static_dim=5)
            loaded_model.load_state_dict(
                torch.load(weight_path, weights_only=True)
            )
            loaded_model.eval()

            with torch.no_grad():
                out2 = loaded_model(vitals=vitals, static=static)["y_prob"]

        torch.testing.assert_close(out1, out2)

    def test_tpc_dataset_dimension_inference(self):
        """Verifies input dimensions are inferred from dataset and logit is returned."""
        ds: Any = MockDataset()
        model = TPC(dataset=ds)
        self.assertEqual(model.continuous_dim, 16)
        self.assertEqual(model.static_dim, 44)

        out = model(vitals=torch.randn(2, 24, 16), static=torch.randn(2, 44))
        self.assertIn("y_prob", out)
        self.assertIn("logit", out)
        self.assertEqual(out["logit"].shape, (2, 24, 1))

    def test_tpc_static_1d_broadcast(self):
        """Verifies 1D static tensor is safely broadcast across mini-batch."""
        model = TPC(dataset=None, continuous_dim=12, static_dim=5)
        vitals = torch.randn(3, 20, 12)
        static_1d = torch.randn(5)
        out = model(vitals=vitals, static=static_1d)
        self.assertEqual(out["y_prob"].shape, (3, 20, 1))

    def test_tpc_input_validation_raises_type_error(self):
        """Verifies that non-Tensor vitals input raises TypeError."""
        model = TPC(continuous_dim=12, static_dim=0)
        with self.assertRaises(TypeError):
            model(vitals="invalid_type")


class TestTPCTask(unittest.TestCase):
    """Unit tests for the LengthOfStayTPCeICU task pipeline and batching."""

    def test_task_filters_and_target(self):
        """Tests task filters (age > 18, stay >= 5h, has obs) and target format."""
        task = LengthOfStayTask(lab_names=["potassium"])

        # Adult with valid stay and observation
        p_valid = MockPatient(age=50, stay_hours=10, has_lab=True)
        samples = task(p_valid)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["los"].shape, (10, 1))
        self.assertEqual(samples[0]["mask"].shape, (10,))
        self.assertEqual(samples[0]["vitals"].shape, (10, 1))
        # Target remaining LoS should be strictly descending
        self.assertGreater(
            samples[0]["los"][0].item(), samples[0]["los"][-1].item()
        )

        # Child (< 18) should be skipped
        p_child = MockPatient(age=16, stay_hours=10, has_lab=True)
        self.assertEqual(len(task(p_child)), 0)

        # Short stay (< 5h) should be skipped
        p_short = MockPatient(age=50, stay_hours=3, has_lab=True)
        self.assertEqual(len(task(p_short)), 0)

        # Stay with no observations should be skipped
        p_nolab = MockPatient(age=50, stay_hours=10, has_lab=False)
        self.assertEqual(len(task(p_nolab)), 0)

        # Missing or empty age string should be skipped
        p_empty_age = MockPatient(age=50, stay_hours=10, has_lab=True)
        p_empty_age.events["patient"][0].age = ""
        self.assertEqual(len(task(p_empty_age)), 0)

        p_null_age = MockPatient(age=50, stay_hours=10, has_lab=True)
        p_null_age.events["patient"][0].age = None
        self.assertEqual(len(task(p_null_age)), 0)

    def test_static_features_deterministic_and_one_hot(self):
        """Tests that static feature extraction is deterministic and valid one-hot."""
        patient = MockPatient(
            patient_id="p1",
            age=65.0,
            stay_hours=24,
            unittype="MICU",
            unitadmitsource="Emergency Department",
            unitstaytype="admit",
            ethnicity="Caucasian",
        )

        task = LengthOfStayTask()

        # 1. Deterministic across two task calls
        res1 = task(patient)
        res2 = task(patient)
        self.assertEqual(len(res1), 1)
        self.assertEqual(len(res2), 1)
        static1 = res1[0]["static"]
        static2 = res2[0]["static"]
        self.assertTrue(torch.equal(static1, static2))
        self.assertEqual(static1.shape[0], task.static_dim)

        # 2. Known category lands in exact assigned slot
        known_cats = {
            "unittype": "MICU",
            "unitadmitsource": "Emergency Department",
            "unitstaytype": "admit",
            "ethnicity": "Caucasian",
        }
        offset = 11  # continuous & indicator features come first
        for cat_name, vocab in task.cat_vocabs.items():
            block_len = len(vocab) + 1  # includes unknown slot
            block = static1[offset : offset + block_len]
            self.assertAlmostEqual(block.sum().item(), 1.0, places=5)
            self.assertTrue(((block == 0.0) | (block == 1.0)).all())
            val = known_cats[cat_name]
            self.assertEqual(static1[offset + vocab.index(val)].item(), 1.0)
            self.assertEqual(static1[offset + len(vocab)].item(), 0.0)
            offset += block_len
        self.assertEqual(offset, task.static_dim)

        # 3. Unknown category sets unknown slot and leaves known slots 0.0
        patient_unk = MockPatient(
            patient_id="p_unk",
            age=65.0,
            stay_hours=24,
            unittype="UNSEEN_UNIT",
            unitadmitsource="UNSEEN_SOURCE",
            unitstaytype="UNSEEN_STAYTYPE",
            ethnicity="UNSEEN_ETHNICITY",
        )
        res_unk = task(patient_unk)
        static_unk = res_unk[0]["static"]

        offset = 11
        for cat_name, vocab in task.cat_vocabs.items():
            block_len = len(vocab) + 1
            block = static_unk[offset : offset + block_len]
            self.assertAlmostEqual(block.sum().item(), 1.0, places=5)
            self.assertEqual(block[-1].item(), 1.0)
            self.assertTrue((block[:-1] == 0.0).all())
            offset += block_len

    def test_baseline_values_outside_window_ignored(self):
        """Tests that baseline measurements outside [-24h, +5h] are ignored."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_content = (
                "patientunitstayid,physicalexamoffset,physicalexampath,"
                "physicalexamtext\n"
                "stay_100,-1560,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/2,2\n"
                "stay_100,-120,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/3,3\n"
                "stay_100,60,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/4,4\n"
                "stay_100,360,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/1,1\n"
            )
            pe_file = tmp_path / "physicalExam.csv"
            pe_file.write_text(csv_content)

            _, baseline_static_by_stay = load_eicu_vitals_cache(str(tmp_path))

            self.assertIn("stay_100", baseline_static_by_stay)
            base = baseline_static_by_stay["stay_100"]
            self.assertIn("eyes", base)
            self.assertEqual(base["eyes"], 4.0)

    def test_select_qualifying_labs_respects_stay_window(self):
        """Tests that select_qualifying_labs counts labs within ICU window."""
        stay_minutes = 48 * 60  # 48 hours = 2880 minutes
        p = MockPatient(
            patient_id="p_labs", age=40.0, stay_hours=48, has_lab=False
        )

        def ev(name, minutes):
            return type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": minutes,
                    "labname": name,
                    "labresult": 1.0,
                },
            )()

        labs = []
        for item in ["A1", "A2", "A3", "A4", "A5"]:
            labs += [ev(item, m) for m in (60, 300, 600, 1200)]
        labs += [ev("X", 600)]  # 1 in-window obs
        labs += [ev("X", -m) for m in (60, 120, 180, 240, 300)]
        labs += [ev("X", stay_minutes + m) for m in (60, 120, 180, 240, 300)]
        p.events["lab"] = labs

        ds: Any = type("MockDS", (), {"iter_patients": lambda self: [p]})()
        selected, met = select_qualifying_labs(
            ds,
            train_pids={"p_labs"},
            min_prevalence=0.25,
            min_avg_obs=3.0,
            max_labs=45,
        )
        self.assertTrue(met)
        self.assertEqual(set(selected), {"A1", "A2", "A3", "A4", "A5"})
        self.assertNotIn("X", selected)

    def test_select_qualifying_labs_fallback(self):
        """Tests that select_qualifying_labs triggers fallback when needed."""
        p = MockPatient(
            patient_id="p_fallback", age=40.0, stay_hours=48, has_lab=False
        )
        p.events["lab"] = [
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 600.0,
                    "labname": "potassium",
                    "labresult": 4.0,
                },
            )()
        ]

        mock_dataset: Any = type(
            "MockDS", (), {"iter_patients": lambda self: [p]}
        )()

        selected, threshold_met = select_qualifying_labs(
            mock_dataset,
            train_pids={"p_fallback"},
            min_prevalence=0.25,
            min_avg_obs=3.0,
            max_labs=45,
        )
        self.assertFalse(threshold_met)
        self.assertEqual(selected, DEFAULT_LAB_NAMES)

    def test_compute_empirical_training_stats_non_finite_resilience(self):
        """Tests that compute_empirical_training_stats handles non-finite values."""
        p = MockPatient(
            patient_id="p_finite", age=45.0, stay_hours=24, has_lab=False
        )
        # Add lab events with valid, NaN, and Inf values
        p.events["lab"] = [
            type("Event", (), {
                "patientunitstayid": "s0",
                "labresultoffset": 120.0,
                "labname": "glucose",
                "labresult": 100.0,
            })(),
            type("Event", (), {
                "patientunitstayid": "s0",
                "labresultoffset": 180.0,
                "labname": "glucose",
                "labresult": float("nan"),
            })(),
            type("Event", (), {
                "patientunitstayid": "s0",
                "labresultoffset": 240.0,
                "labname": "glucose",
                "labresult": float("inf"),
            })(),
            type("Event", (), {
                "patientunitstayid": "s0",
                "labresultoffset": 300.0,
                "labname": "glucose",
                "labresult": 110.0,
            })(),
        ]
        mock_ds: Any = type(
            "MockDS", (), {"iter_patients": lambda self: [p]}
        )()

        vitals = {
            "s0": [
                (1, "heart_rate", 80.0),
                (2, "heart_rate", float("nan")),
                (3, "heart_rate", float("inf")),
                (4, "heart_rate", 90.0),
            ]
        }

        ts_stats, _, static_stats, _ = compute_empirical_training_stats(
            mock_ds,
            train_pids={"p_finite"},
            lab_names=["glucose"],
            vitals_by_stay=vitals,
        )

        for k, (mean, std) in ts_stats.items():
            self.assertFalse(np.isnan(mean), f"NaN mean in {k}")
            self.assertFalse(np.isinf(mean), f"Inf mean in {k}")
            self.assertFalse(np.isnan(std), f"NaN std in {k}")
            self.assertFalse(np.isinf(std), f"Inf std in {k}")

        for k, (mean, std, med) in static_stats.items():
            self.assertFalse(np.isnan(mean), f"NaN static mean in {k}")
            self.assertFalse(np.isnan(std), f"NaN static std in {k}")
            self.assertFalse(np.isnan(med), f"NaN static med in {k}")

    def test_bucket_batch_sampler(self):
        """Tests BucketBatchSampler indexing, shuffling, and reset replay."""
        lengths = [12, 14, 15, 20, 22, 25, 48, 50, 52, 100, 105, 110]
        batch_size = 3
        sampler = BucketBatchSampler(
            lengths, batch_size=batch_size, shuffle=True, seed=123
        )

        # 1. Epoch 0: All indices yielded exactly once
        batches_ep0 = list(sampler)
        all_indices_ep0 = [idx for batch in batches_ep0 for idx in batch]
        self.assertEqual(sorted(all_indices_ep0), list(range(len(lengths))))
        self.assertEqual(len(batches_ep0), 4)
        for b in batches_ep0:
            self.assertEqual(len(b), batch_size)

        # 2. Epoch 1: Yields all indices with shuffled batch order
        batches_ep1 = list(sampler)
        all_indices_ep1 = [idx for batch in batches_ep1 for idx in batch]
        self.assertEqual(sorted(all_indices_ep1), list(range(len(lengths))))
        self.assertNotEqual(batches_ep0, batches_ep1)

        # 3. Reset: Replays epoch 0 sequence identically
        sampler.reset()
        batches_ep0_replay = list(sampler)
        self.assertEqual(batches_ep0_replay, batches_ep0)

        # 4. Error handling and empty sequence checks
        with self.assertRaises(ValueError):
            BucketBatchSampler([10, 20], batch_size=0)
        empty_sampler = BucketBatchSampler([], batch_size=4)
        self.assertEqual(len(empty_sampler), 0)
        self.assertEqual(list(empty_sampler), [])

    def test_map_dataset_lazy_indexing(self):
        """Tests that MapDataset supports both eager lists and lazy index mapping."""
        mock_data = [{"id": f"s_{i}"} for i in range(10)]

        # Eager mode
        ds_eager = MapDataset(mock_data)
        self.assertEqual(len(ds_eager), 10)
        self.assertEqual(ds_eager[0]["id"], "s_0")
        self.assertEqual(ds_eager[9]["id"], "s_9")

        # Lazy subset indexing mode
        subset_indices = [2, 5, 8]
        ds_lazy = MapDataset(mock_data, indices=subset_indices)
        self.assertEqual(len(ds_lazy), 3)
        self.assertEqual(ds_lazy[0]["id"], "s_2")
        self.assertEqual(ds_lazy[1]["id"], "s_5")
        self.assertEqual(ds_lazy[2]["id"], "s_8")

    def test_load_eicu_vitals_cache_chunked_and_filtered(self):
        """Tests that load_eicu_vitals_cache chunks and filters stays."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            p_prefix = (
                "notes/Progress Notes/Physical Exam/Physical Exam/Vital Signs/"
            )
            csv_content = (
                "patientunitstayid,physicalexamoffset,physicalexampath,"
                "physicalexamtext\n"
                f"stay_1,60,{p_prefix}HR/HR Current,80\n"
                f"stay_2,120,{p_prefix}HR/HR Current,90\n"
                f"stay_1,180,{p_prefix}HR/HR Current,85\n"
                f"stay_3,240,{p_prefix}HR/HR Current,95\n"
            )
            pe_file = tmp_path / "physicalExam.csv"
            pe_file.write_text(csv_content)

            # Test chunksize=2 with filter on stay_1 only
            vitals, _ = load_eicu_vitals_cache(
                str(tmp_path), chunksize=2, valid_stay_ids={"stay_1"}
            )

            self.assertIn("stay_1", vitals)
            self.assertNotIn("stay_2", vitals)
            self.assertNotIn("stay_3", vitals)
            self.assertEqual(len(vitals["stay_1"]), 2)
            # Ordered chronologically: hour 1 (60m) then hour 3 (180m)
            self.assertEqual(vitals["stay_1"][0], (1, "heart_rate", 80.0))
            self.assertEqual(vitals["stay_1"][1], (3, "heart_rate", 85.0))

    def test_load_eicu_vitals_cache_periodic_and_aperiodic_tables(self):
        """Tests that periodic and aperiodic tables are parsed and merged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            p_content = (
                "PatientUnitStayID,ObservationOffset,heartRate,"
                "systemicSystolic,temperature\n"
                "stay_1,30,72,120,36.8\n"
                "stay_2,60,80,110,37.0\n"
            )
            (tmp_path / "vitalPeriodic.csv").write_text(p_content)

            ap_content = (
                "patientunitstayid,observationoffset,noninvasivesystolic,"
                "temperature\n"
                "stay_1,90,125,98.6\n"
                "stay_1,15,118,97.7\n"
            )
            (tmp_path / "vitalAperiodic.csv").write_text(ap_content)

            vitals, _ = load_eicu_vitals_cache(
                str(tmp_path), chunksize=1, valid_stay_ids={"stay_1"}
            )
            self.assertIn("stay_1", vitals)
            self.assertNotIn("stay_2", vitals)

            records = vitals["stay_1"]
            # All stay_1 records: offset 15, offset 30, offset 90
            self.assertTrue(len(records) >= 4)
            # Chronological check: hour of offset 15 is 0, hour of offset 90 is 1
            self.assertEqual(records[0][0], 0)
            self.assertEqual(records[-1][0], 1)

            # Check that 98.6 deg F was converted to approx 37.0 deg C
            temp_records = [r for r in records if r[1] == "temperature"]
            self.assertTrue(len(temp_records) >= 2)
            converted_temps = [r[2] for r in temp_records]
            # Should have ~36.5 (from 97.7 F), 36.8 (C), and ~37.0 (from 98.6 F)
            self.assertTrue(any(abs(t - 37.0) < 0.1 for t in converted_temps))
            self.assertTrue(any(abs(t - 36.5) < 0.1 for t in converted_temps))

    def test_get_valid_eicu_stay_ids(self):
        """Tests get_valid_eicu_stay_ids filtering on adult and >=5h criteria."""
        class MockPatientObj:
            def __init__(self, sid, offset, age):
                self.patientunitstayid = sid
                self.unitdischargeoffset = offset
                self.age = age

        p_adult = type(
            "P", (),
            {"get_events": lambda self, event_type: [
                MockPatientObj("stay_valid", 360, "55"),
                MockPatientObj("stay_short", 120, "55"),
                MockPatientObj("stay_child", 400, "16"),
            ]}
        )()
        mock_ds = type("DS", (), {"iter_patients": lambda self: [p_adult]})()
        valid_ids = get_valid_eicu_stay_ids(mock_ds)
        self.assertEqual(valid_ids, {"stay_valid"})

    def test_length_of_stay_tpc_eicu_fn(self):
        """Tests length_of_stay_tpc_eicu_fn functional task wrapper."""
        from pyhealth.tasks import length_of_stay_tpc_eicu_fn
        p_empty = type(
            "P", (), {"get_events": lambda self, event_type: []}
        )()
        samples = length_of_stay_tpc_eicu_fn(p_empty)
        self.assertEqual(samples, [])

    def test_task_cache_fingerprint_decoupled(self):
        """Tests that task keeps large dicts out of vars(task) for json.dumps."""
        import json
        large_vitals = {
            f"stay_{i}": [(0, "heart_rate", 80.0)] for i in range(50)
        }
        task = LengthOfStayTask(vitals_by_stay=large_vitals)

        # 1. vars(task) does not contain vitals_by_stay directly
        v = vars(task)
        self.assertNotIn("vitals_by_stay", v)
        self.assertNotIn("baseline_static_by_stay", v)
        self.assertIn("vitals_hash", v)

        # 2. json.dumps is small and compact (< 2 KB)
        dumped = json.dumps(v, default=str)
        self.assertLess(len(dumped), 2048)

        # 3. Transparent property access returns the full dictionary
        self.assertEqual(len(task.vitals_by_stay), 50)
        self.assertIn("stay_0", task.vitals_by_stay)


class TestEICUDemoTPCIntegration(unittest.TestCase):
    """Integration tests for TPC task and model pipeline using local eICU demo data."""

    def setUp(self):
        """Set up demo dataset path for integration tests."""
        test_dir = Path(__file__).resolve().parents[2]
        self.demo_dataset_path = str(
            test_dir / "test-resources" / "core" / "eicudemo"
        )

    def test_eicu_demo_dataset_stats(self):
        """Tests eICUDataset instantiation and stats() on demo data."""
        if not os.path.exists(self.demo_dataset_path):
            self.skipTest(f"Demo data not found at {self.demo_dataset_path}")

        dataset = eICUDataset(
            root=self.demo_dataset_path, tables=["patient", "lab"]
        )
        dataset.stats()
        self.assertGreater(len(dataset.unique_patient_ids), 0)

    def test_eicu_demo_tpc_task_and_model_pipeline(self):
        """Tests LengthOfStayTPCeICU set_task and TPC model forward on demo data."""
        if not os.path.exists(self.demo_dataset_path):
            self.skipTest(f"Demo data not found at {self.demo_dataset_path}")

        dataset = eICUDataset(
            root=self.demo_dataset_path, tables=["patient", "lab"]
        )
        valid_ids = get_valid_eicu_stay_ids(dataset)
        self.assertGreater(len(valid_ids), 0)

        vitals, base = load_eicu_vitals_cache(
            self.demo_dataset_path, valid_stay_ids=valid_ids
        )
        self.assertGreater(len(vitals), 0)
        self.assertGreater(len(base), 0)

        task = LengthOfStayTPCeICU(
            vitals_by_stay=vitals,
            baseline_static_by_stay=base,
        )

        self.assertEqual(
            task.task_name, "eICU Remaining Length of Stay Prediction (TPC)"
        )
        for k in ["vitals", "vitals_decay", "static", "mask"]:
            self.assertIn(k, task.input_schema)
        for k in ["los", "mortality"]:
            self.assertIn(k, task.output_schema)

        sample_dataset = dataset.set_task(task)
        self.assertIsNotNone(sample_dataset)
        self.assertGreater(len(sample_dataset), 0)

        sample0 = sample_dataset[0]
        required_keys = [
            "visit_id",
            "patient_id",
            "vitals",
            "vitals_decay",
            "static",
            "mask",
            "los",
            "mortality",
        ]
        for key in required_keys:
            self.assertIn(key, sample0)

        self.assertEqual(sample0["vitals"].shape[-1], task.continuous_dim)
        self.assertEqual(sample0["vitals_decay"].shape, sample0["vitals"].shape)
        self.assertEqual(sample0["static"].shape[0], task.static_dim)
        self.assertEqual(sample0["mask"].shape[0], sample0["vitals"].shape[0])
        self.assertEqual(sample0["los"].shape[0], sample0["vitals"].shape[0])
        self.assertEqual(sample0["los"].shape[1], 1)
        self.assertTrue((sample0["los"] >= 1.0 / 48.0).all())

        # Test TPC model instantiation from real SampleDataset
        model = TPC(dataset=sample_dataset, multitask=True)
        self.assertEqual(model.continuous_dim, task.continuous_dim)
        self.assertEqual(model.static_dim, task.static_dim)

        # Test batch forward propagation and loss computation
        batch = collate_fn_dict_with_padding(
            [sample_dataset[0], sample_dataset[1]]
        )
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("logit", out)
        self.assertIn("mortality_prob", out)
        self.assertIn("mortality_logit", out)
        self.assertFalse(torch.isnan(out["loss"]))
        self.assertTrue((out["y_prob"] > 0).all())


if __name__ == "__main__":
    unittest.main()