"""Author: Tamjit Kumar
NetID: tkumar70
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Temporal Pointwise Convolution (TPC) Length of Stay Prediction Pipeline
             and reproducible ablation study for eICU.

Primary Dataset:
    eICU Collaborative Research Database (v2.0)
    https://eicu-crd.mit.edu/

Implementation Notes & Methodological Alignment:
1. Multi-Modal Clinical Feature Pipeline (eICU):
   - Dynamic Time-Series Features:
     • Core Bedside Vital Signs (Heart Rate, Systolic BP, Diastolic BP,
       Respiratory Rate, SpO2, Temperature). Extracted from physicalExam
       and periodic vital records.
     • Laboratory Blood Tests dynamically selected on training split
       matching official preprocessing pipeline (>= 25% stay prevalence
       and > 3 obs/stay within the stay window).
       Source: EmmaRocheteau/TPC-LoS-prediction (eICU_preprocessing/timeseries.sql).
   - Static (Flat) Features matching Section 4.1, Table 16, and flat_features.sql:
     • Continuous: Age, Admission Hour, Height, Weight, GCS Eyes, GCS Motor,
       GCS Verbal, Gender.
     • Missingness Indicators: Binary flags for unmeasured baseline height,
       weight, and GCS in admission window [-24h, +5h].
     • Categorical (One-Hot Encoded with Unknown Slot): Unit Type, Admission Source,
       Unit Stay Type, Ethnicity. Vocabularies and empirical medians/means are
       built strictly from the training split.
2. Data Leakage Prevention & Empirical Normalization:
   - Patient cohort partitioned into Train (70%), Val (15%), Test (15%) before
     feature selection or statistics computation.
   - Lab selection and normalization parameters computed strictly on train split
     with 1st/99th percentile clipping.
   - Baseline measurements restricted to [-24h, +5h] relative to ICU admission;
     lookahead leakage prevented for predictions at t >= 4h (enforced via
     warmup evaluation mask and min_time=4 on both LoS and mortality losses).
3. Hourly Discretization & Decay Indicators:
   - Hourly aggregation from ICU admission out to 14 days (336 hours).
   - Exponential decay: x''_t = 0.75^j, where j is elapsed hours since last obs.
4. Target Alignment:
   - Remaining LoS in days anchored to end of each hour t with a 30-minute floor.
5. Multitask Learning:
   - Joint hourly LoS regression (MSLE) + in-hospital mortality prediction
     (BCEWithLogits).
6. Length-Bucketed Batch Sampling & Unconfounded Ablation:
   - Groups stays of similar length to minimize padding overhead.
   - Resets epoch counters and shares identical random seeds across configurations,
     ensuring identical random seed and batch order across configs.
7. Known Limitations & Caching:
   - BatchNorm over padded timesteps: 1D Batch Normalization in TPC blocks computes
     statistics across all steps including padding. Length-bucketed batch sampling
     greatly reduces padding overhead but does not eliminate it in uneven batches.
   - Cache key invalidation: PyHealth BaseDataset.set_task hashes vars(task) via uuid5,
     so changes to stats or dictionaries automatically generate a new cache directory.
     Users can also call dataset.clean_tmpdir() to clear.
"""

from __future__ import annotations

import argparse
import os
import random
from itertools import chain
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import collate_fn_dict_with_padding, eICUDataset
from pyhealth.models.tpc import TPC
from pyhealth.tasks import LengthOfStayTPCeICU
from pyhealth.tasks.length_of_stay_tpc_eicu import (
    CORE_VITALS_MAP,
    LOS_FLOOR_DAYS,
    BucketBatchSampler,
    MapDataset,
    compute_empirical_training_stats,
    get_valid_eicu_stay_ids,
    load_eicu_vitals_cache,
    select_qualifying_labs,
)


class AblationConfig(TypedDict):
    name: str
    loss_fn: str
    use_decay: bool


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_sampler: BucketBatchSampler,
    continuous_dim: int,
    static_dim: int,
    epochs: int = 2,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> None:
    """Runs the architectural ablation study replicating the paper's methodology.

    Ablations Evaluated (Table 3, Rocheteau et al.):
    1. TPC (Proposed): MSLE Loss, Exponential Decay, Multi-Modal Inputs
    2. TPC (No Decay): Ablates missing data decay channels (use_decay=False)
    3. TPC (MSE Loss): Ablates logarithmic MSLE loss in favor of standard MSE loss

    Experimental Control:
    - Same random seed and batch order across configs.
    - Sampler reset ensures identical mini-batch sequences across all configs.
    - Configurable training epochs (default 2 for rapid validation).

    Args:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for test set.
        train_sampler: BucketBatchSampler used on train_loader.
        continuous_dim: Number of dynamic time-series features.
        static_dim: Number of static baseline features.
        epochs: Number of training epochs per ablation config. Default: 2.
        lr: Learning rate for Adam optimizer. Default: 1e-3.
        device: Device to train on ('cpu' or 'cuda'). Default: 'cpu'.
        seed: Random seed for initialization. Default: 42.
    """
    dev = torch.device(device)
    print("\n" + "=" * 85)
    print(
        "HELD-OUT TEST ABLATION STUDY: MULTI-MODAL TPC PIPELINE VERIFICATION (eICU)"
    )
    print(
        f"Features: {continuous_dim} Continuous Time-Series + "
        f"{static_dim} Static Features | Device: {dev}"
    )
    print("=" * 85)

    configs: list[AblationConfig] = [
        {"name": "TPC (Proposed)", "loss_fn": "msle", "use_decay": True},
        {"name": "TPC (No Decay)", "loss_fn": "msle", "use_decay": False},
        {"name": "TPC (MSE Loss)", "loss_fn": "mse", "use_decay": True},
    ]

    def evaluate_loader(
        model: TPC, loader: DataLoader
    ) -> tuple[float, float, float, float, float]:
        model.eval()
        sum_msle = 0.0
        sum_mse = 0.0
        sum_mae = 0.0
        sum_mape = 0.0
        sum_mort_bce = 0.0
        total_valid_hours = 0
        total_mort_batches = 0

        with torch.no_grad():
            for raw_batch in loader:
                batch = {
                    k: v.to(dev) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_batch.items()
                }
                out = model(**batch)
                y_pred = out["y_prob"]
                y_true = batch["los"]
                raw_mask = batch.get("mask")
                if raw_mask is not None:
                    mask = raw_mask.bool()
                else:
                    mask = (y_true > 0).squeeze(-1).bool()
                if mask.dim() == 3:
                    mask = mask.squeeze(-1)

                yp_2d = y_pred.squeeze(-1)
                yt_2d = y_true.squeeze(-1)

                # Exclude warmup hours t < 4 (consistent with min_time=4 / 5h)
                T = yp_2d.shape[1]
                if T > 4:
                    eval_mask = mask.clone()
                    eval_mask[:, :4] = False
                else:
                    eval_mask = mask
                yp = yp_2d[eval_mask]
                yt = yt_2d[eval_mask]

                n_hrs = yp.numel()
                if n_hrs > 0:
                    total_valid_hours += n_hrs
                    yp_clamped = torch.clamp(yp, min=LOS_FLOOR_DAYS)
                    yt_clamped = torch.clamp(yt, min=LOS_FLOOR_DAYS)
                    sum_msle += torch.sum(
                        (torch.log(yp_clamped) - torch.log(yt_clamped)) ** 2
                    ).item()
                    sum_mse += torch.sum((yp - yt) ** 2).item()
                    sum_mae += torch.sum(torch.abs(yp - yt)).item()
                    sum_mape += torch.sum(
                        torch.abs(yp - yt) / yt_clamped * 100.0
                    ).item()

                if (
                    model.multitask
                    and "mortality_prob" in out
                    and "mortality" in batch
                ):
                    raw_mort = (
                        batch["mortality"]
                        .float()
                        .view(-1, 1, 1)
                        .expand_as(out["mortality_prob"])
                    )
                    m_eval = eval_mask.unsqueeze(-1)
                    if m_eval.sum() > 0:
                        mort_loss = torch.nn.functional.binary_cross_entropy(
                            out["mortality_prob"][m_eval],
                            raw_mort[m_eval],
                        ).item()
                        sum_mort_bce += mort_loss
                        total_mort_batches += 1
        avg_mort = sum_mort_bce / max(total_mort_batches, 1)
        return (
            sum_msle / max(total_valid_hours, 1),
            sum_mse / max(total_valid_hours, 1),
            sum_mae / max(total_valid_hours, 1),
            sum_mape / max(total_valid_hours, 1),
            avg_mort,
        )

    results = []

    for cfg in configs:
        print(f"\nTraining configuration: {cfg['name']}...")

        # Reset sampler epoch so each configuration sees identical batch order
        train_sampler.reset()

        # Reset seeds for reproducible parameter initialization
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TPC(
            dataset=None,
            continuous_dim=continuous_dim,
            static_dim=static_dim,
            n_layers=3,
            temp_dim=8,
            point_dim=16,
            kernel_size=3,
            hidden_dim=64,
            loss_fn=cfg["loss_fn"],
            min_time=4,
            floor_days=LOS_FLOOR_DAYS,
            use_decay=cfg["use_decay"],
            multitask=True,
        )
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for raw_batch in train_loader:
                batch = {
                    k: v.to(dev) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_batch.items()
                }
                optimizer.zero_grad()
                out = model(**batch)
                out["loss"].backward()
                optimizer.step()

        val_msle, _, _, _, _ = evaluate_loader(model, val_loader)
        test_msle, test_mse, test_mae, test_mape, test_mort = evaluate_loader(
            model, test_loader
        )

        results.append(
            {
                "Configuration": cfg["name"],
                "Val MSLE": val_msle,
                "Test MSLE": test_msle,
                "Test MSE": test_mse,
                "Test MAD": test_mae,
                "Test MAPE": test_mape,
                "Test Mort BCE": test_mort,
            }
        )

    print("\n" + "=" * 110)
    print(
        f"HELD-OUT TEST ABLATION STUDY RESULTS "
        f"(eICU, {epochs} epochs – pipeline check only):"
    )
    print(
        f"{'Configuration':<24} | {'Val MSLE':<9} | {'Test MSLE':<9} | "
        f"{'Test MSE':<9} | {'Test MAD':<9} | {'Test MAPE':<10} | "
        f"{'Test Mort BCE':<13}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r['Configuration']:<24} | {r['Val MSLE']:<9.4f} | "
            f"{r['Test MSLE']:<9.4f} | {r['Test MSE']:<9.4f} | "
            f"{r['Test MAD']:<9.4f} | {r['Test MAPE']:<9.2f}% | "
            f"{r['Test Mort BCE']:<13.4f}"
        )
    print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eICU Length of Stay Prediction with TPC"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("EICU_ROOT", "./test-resources/core/eicudemo"),
        help=(
            "Path to eICU directory containing CSV files "
            "(default: ./test-resources/core/eicudemo)"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250000,
        help=(
            "Chunk size (number of rows) for reading large CSVs "
            "(default: 250000)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs per ablation config (default: 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and evaluation (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and training (default: 42)",
    )
    args = parser.parse_args()

    eicu_root = args.data_dir
    if not os.path.exists(eicu_root):
        raise FileNotFoundError(
            f"eICU data directory not found: '{eicu_root}'. "
            "Please specify a valid path containing eICU CSV files using "
            "--data_dir, or ensure './test-resources/core/eicudemo' exists."
        )

    print("1. Initializing eICUDataset...")
    dataset = eICUDataset(
        root=eicu_root,
        tables=["patient", "lab"],
    )

    # Partition patients strictly BEFORE task processing to prevent any data leakage
    print("\n2. Partitioning patient cohort (70% Train, 15% Val, 15% Test)...")
    all_pids = sorted(dataset.unique_patient_ids)
    rng = random.Random(args.seed)
    perm_pids = list(all_pids)
    rng.shuffle(perm_pids)
    n_p = len(perm_pids)
    n_train = int(n_p * 0.70)
    n_val = int(n_p * 0.15)
    train_pids = set(perm_pids[:n_train])
    val_pids = set(perm_pids[n_train : n_train + n_val])
    test_pids = set(perm_pids[n_train + n_val :])
    print(
        f"  Train patients: {len(train_pids)}, Val patients: {len(val_pids)}, "
        f"Test patients: {len(test_pids)}"
    )

    print(
        "\n3. Loading bedside vital signs & admission baseline items "
        "from physicalExam..."
    )
    valid_stay_ids = get_valid_eicu_stay_ids(dataset)
    print(f"  Identified {len(valid_stay_ids)} adult ICU stays (>= 5h).")
    vitals_by_stay, baseline_static_by_stay = load_eicu_vitals_cache(
        eicu_root,
        dataset,
        chunksize=args.chunksize,
        valid_stay_ids=valid_stay_ids,
    )
    print(
        f"  Loaded vitals for {len(vitals_by_stay)} stays, "
        f"baseline items for {len(baseline_static_by_stay)} stays."
    )

    print(
        "\n4. Dynamically selecting laboratory tests "
        "(Paper inclusion rule on training stays)..."
    )
    qualifying_labs, thresh_met = select_qualifying_labs(
        dataset, train_pids, min_prevalence=0.25, min_avg_obs=3.0, max_labs=45
    )
    if thresh_met:
        print(
            f"  Selected {len(qualifying_labs)} qualifying lab tests "
            "meeting >=25% stay prevalence & >3 obs/stay."
        )
    else:
        print(
            f"  Notice: Small cohort fallback used; selected "
            f"{len(qualifying_labs)} default clinical labs."
        )

    print(
        "\n5. Computing empirical training statistics & vocabularies "
        "(strictly on train split)..."
    )
    emp_stats, feat_counts, static_stats, cat_vocabs = (
        compute_empirical_training_stats(
            dataset,
            train_pids,
            qualifying_labs,
            vitals_by_stay,
            baseline_static_by_stay,
        )
    )
    total_ts_features = len(CORE_VITALS_MAP) + len(qualifying_labs)
    print(
        f"  Computed normalization parameters across {total_ts_features} "
        "time-series features."
    )
    print(f"  Core Vital Signs ({len(CORE_VITALS_MAP)}):")
    for vid, vname in CORE_VITALS_MAP.items():
        mean, std = emp_stats.get(vid, (0.0, 1.0))
        cnt = feat_counts.get(vid, 0)
        print(
            f"    {vname} ({vid}): {cnt:5d} measurements, "
            f"mean={mean:7.2f}, std={std:6.2f}"
        )

    print(
        "\n6. Applying ICU Length of Stay Task "
        "(Multi-Modal Time Series + Static Features)..."
    )
    task = LengthOfStayTPCeICU(
        lab_names=qualifying_labs,
        norm_stats=emp_stats,
        vitals_by_stay=vitals_by_stay,
        baseline_static_by_stay=baseline_static_by_stay,
        static_stats=static_stats,
        cat_vocabs=cat_vocabs,
    )
    print(f"  Derived static feature dimension: {task.static_dim}")
    sample_dataset = dataset.set_task(task)
    print(f"Total valid ICU stays created: {len(sample_dataset)}")

    if len(sample_dataset) > 0:
        sample0 = sample_dataset[0]
        print("\nSample 0 summary:")
        print(f"  Patient ID: {sample0['patient_id']}")
        print(f"  Visit/Stay ID: {sample0['visit_id']}")
        print(f"  Vitals shape (T x F): {sample0['vitals'].shape}")
        print(f"  Vitals decay shape (T x F): {sample0['vitals_decay'].shape}")
        print(f"  Static features shape: {sample0['static'].shape}")
        print(f"  Mask shape: {sample0['mask'].shape}")
        print(f"  Hourly remaining LoS shape: {sample0['los'].shape}")
        print(f"  Remaining LoS at hour 0: {sample0['los'][0].item():.2f} days")
        print(
            f"  Remaining LoS at final hour: {sample0['los'][-1].item():.2f} days"
        )

        mort_count = sum(s["mortality"].item() for s in sample_dataset)
        mort_rate = mort_count / len(sample_dataset) * 100.0
        print(
            f"  Cohort In-Hospital Mortality Prevalence: "
            f"{mort_count:.0f}/{len(sample_dataset)} ({mort_rate:.1f}%)"
        )

        p2idx = sample_dataset.patient_to_index
        train_indices = list(
            chain(*[p2idx[pid] for pid in train_pids if pid in p2idx])
        )
        val_indices = list(
            chain(*[p2idx[pid] for pid in val_pids if pid in p2idx])
        )
        test_indices = list(
            chain(*[p2idx[pid] for pid in test_pids if pid in p2idx])
        )
        print(
            f"  Train stays: {len(train_indices)}, "
            f"Val stays: {len(val_indices)}, "
            f"Test stays: {len(test_indices)}"
        )

        # Sequence lengths for bucket batching without materializing
        # all tensors into RAM
        train_lengths = [
            sample_dataset[i]["vitals"].shape[0] for i in train_indices
        ]
        train_sampler = BucketBatchSampler(
            train_lengths,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        train_loader = DataLoader(
            MapDataset(sample_dataset, train_indices),
            batch_sampler=train_sampler,
            collate_fn=collate_fn_dict_with_padding,
        )
        val_loader = DataLoader(
            MapDataset(sample_dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_dict_with_padding,
        )
        test_loader = DataLoader(
            MapDataset(sample_dataset, test_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_dict_with_padding,
        )

        run_ablation_study(
            train_loader,
            val_loader,
            test_loader,
            train_sampler=train_sampler,
            continuous_dim=task.continuous_dim,
            static_dim=task.static_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )

        print(
            "\nEnd-to-end eICU TPC pipeline & unconfounded ablation "
            "successfully completed!"
        )
    else:
        print("No valid ICU samples generated.")
