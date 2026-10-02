"""ICU Remaining Length of Stay Prediction with Temporal Pointwise Convolution (TPC).

Paper Reference:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit'
    (Emma Rocheteau, Pietro Liò, & Stephanie Hyland, ACM CHIL 2021).

Note on Scope & Dataset:
    The original paper evaluated TPC on 101 time-series features (from both chartevents and
    labevents) and 12 static features using the full MIMIC-IV database (50,042 patients).
    For this PyHealth benchmark reproduction on the publicly distributable MIMIC-IV demo dataset,
    we demonstrate the complete end-to-end pipeline (data parsing -> cohort filtering ->
    hourly discretization -> forward-filling + decay calculation -> patient-level splitting ->
    causal TPC model training -> ablation analysis) using 12 representative laboratory tests
    and core demographic features.
"""

import os
from datetime import datetime, timedelta
from itertools import chain
from typing import Any
import numpy as np
import torch
from pyhealth.datasets import MIMIC4Dataset, get_dataloader
from pyhealth.tasks import BaseTask
from pyhealth.models.tpc import TPC

# 12 representative laboratory tests for ICU patients
TOP_LAB_ITEMS = [
    "50902",  # Chloride (Blood)
    "50868",  # Anion Gap (Blood)
    "50882",  # Bicarbonate (Blood)
    "51279",  # Red Blood Cells (Blood)
    "51249",  # MCHC (Blood)
    "51222",  # Hemoglobin (Blood)
    "51265",  # Platelet Count (Blood)
    "51221",  # Hematocrit (Blood)
    "51277",  # RDW (Blood)
    "51250",  # MCV (Blood)
    "51248",  # MCH (Blood)
    "51301",  # White Blood Cells (Blood)
]

# Minimum floor for remaining LoS (1/48 days = 30 minutes) to prevent log singularities
LOS_FLOOR_DAYS = 1.0 / 48.0


def compute_empirical_stats(
    dataset: MIMIC4Dataset,
    lab_item_ids: list[str],
    patient_ids: set[str] | list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Computes empirical mean and std for each lab item with 1st/99th percentile outlier clipping.
    
    Computed strictly on the training patient split to avoid data leakage.
    Ensures that unobserved values (filled with 0 after z-score standardization)
    accurately represent the empirical population mean (0.0).
    """
    vals_by_item: dict[str, list[float]] = {item: [] for item in lab_item_ids}
    allowed_pids = set(patient_ids) if patient_ids is not None else None
    
    for patient in dataset.iter_patients():
        if allowed_pids is not None and str(patient.patient_id) not in allowed_pids:
            continue
        for lab in patient.get_events("labevents"):
            item_id = str(lab.attr_dict.get("itemid"))
            val = lab.attr_dict.get("valuenum")
            if item_id in vals_by_item and val is not None:
                try:
                    vals_by_item[item_id].append(float(val))
                except (ValueError, TypeError):
                    pass
                    
    stats = {}
    for item_id, vals in vals_by_item.items():
        if len(vals) > 0:
            arr = np.array(vals)
            p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
            clipped = np.clip(arr, p1, p99)
            mean = float(np.mean(clipped))
            std = float(np.std(clipped))
            stats[item_id] = (mean, std if std > 0 else 1.0)
        else:
            stats[item_id] = (0.0, 1.0)
            
    return stats


def generate_tpc_tensors(
    raw_grid: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates forward-filled time series and decay indicator tensors.
    
    Implements Section 4.1:
    - Forward-fills across gaps in time series.
    - Computes decay indicators: x''_t = 0.75^j, where j is hours since last recording.
    
    Args:
        raw_grid: Standardized tensor (seq_len, num_features).
        mask: Binary tensor (seq_len, num_features), 1 if measured at step t, 0 otherwise.
        
    Returns:
        vitals_ff: Forward-filled tensor (seq_len, num_features).
        vitals_decay: Decay indicators tensor 0.75^j (seq_len, num_features).
    """
    seq_len, num_features = raw_grid.shape
    vitals_ff = raw_grid.clone()
    vitals_decay = torch.zeros(seq_len, num_features, dtype=torch.float32)
    time_since_last = torch.full((num_features,), float("inf"))
    
    for t in range(seq_len):
        measured = (mask[t] == 1.0)
        time_since_last += 1.0
        time_since_last[measured] = 0.0
        
        # Forward-fill previous observations
        if t > 0:
            unmeasured = ~measured
            vitals_ff[t, unmeasured] = vitals_ff[t - 1, unmeasured]
            
        # Decay indicator: 0.75^j (0.0 if never observed)
        finite = ~torch.isinf(time_since_last)
        vitals_decay[t, finite] = torch.pow(0.75, time_since_last[finite])
        
    return vitals_ff, vitals_decay


class LengthOfStayTask(BaseTask):
    """ICU Remaining Length of Stay (LoS) Prediction Task.
    
    Reproduces the task definition from Rocheteau et al. (ACM CHIL 2021).
    
    Key Cohort & Task Criteria (Section 4 & 5.1.1):
    - Adult ICU patients (> 18 years old at admission).
    - Minimum ICU length of stay of 5 hours.
    - At least one recorded clinical observation during the stay.
    - Capped at first 14 days (336 hours).
    - Target: Remaining LoS in days anchored at the end of each hour t:
      y_t = max((out_time - (in_time + t + 1 hours)) / 24.0, 1/48 days).
    """
    task_name = "ICU Remaining Length of Stay Prediction"
    
    input_schema = {
        "vitals": "tensor",
        "vitals_decay": "tensor",
        "static": "tensor",
        "mask": "tensor",
    }
    
    output_schema = {
        "los": "tensor",
        "mortality": "tensor",
    }

    def __init__(
        self,
        lab_item_ids: list[str] | None = None,
        norm_stats: dict[str, tuple[float, float]] | None = None,
        max_stay_hours: int = 336,
    ):
        super(LengthOfStayTask, self).__init__()
        self.lab_item_ids = lab_item_ids or TOP_LAB_ITEMS
        self.feature_map = {item_id: idx for idx, item_id in enumerate(self.lab_item_ids)}
        self.num_features = len(self.lab_item_ids)
        self.norm_stats = norm_stats or {}
        self.max_stay_hours = max_stay_hours

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = []
        
        patients_info = patient.get_events(event_type="patients")
        if not patients_info:
            return []
            
        anchor_age = float(patients_info[0].attr_dict.get("anchor_age", 0))
        anchor_year = float(patients_info[0].attr_dict.get("anchor_year", 0))
        gender_str = str(patients_info[0].attr_dict.get("gender", "M")).upper()
        gender_val = 1.0 if gender_str == "M" else 0.0
        
        icustays = patient.get_events(event_type="icustays")
        admissions = patient.get_events(event_type="admissions")
        
        # Pre-filter labevents for this patient once
        raw_labs = patient.get_events(event_type="labevents")
        patient_labs = [
            lab for lab in raw_labs
            if str(lab.attr_dict.get("itemid")) in self.feature_map and lab.timestamp is not None
        ]
        
        # Build map of admission mortality flags
        hadm_expire_map = {}
        for adm in admissions:
            hadm_id = adm.attr_dict.get("hadm_id")
            if hadm_id:
                flag = str(adm.attr_dict.get("hospital_expire_flag", "0"))
                hadm_expire_map[str(hadm_id)] = (flag == "1")

        for icustay in icustays:
            in_time = icustay.timestamp
            out_time_raw = icustay.attr_dict.get("outtime")
            if in_time is None or out_time_raw is None:
                continue
                
            if isinstance(out_time_raw, datetime):
                out_time = out_time_raw
            else:
                try:
                    out_time = datetime.strptime(str(out_time_raw), "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    continue
                    
            # 1. Compute true age at admission
            true_age = anchor_age + (in_time.year - anchor_year) if anchor_year > 0 else anchor_age
            if true_age <= 18:
                continue
                
            total_stay_hours = int((out_time - in_time).total_seconds() // 3600)
            if total_stay_hours < 5:
                continue
                
            T = min(total_stay_hours, self.max_stay_hours)
            
            # 2. Bin observations into hourly grid with empirical standardization
            raw_grid = torch.zeros(T, self.num_features, dtype=torch.float32)
            mask = torch.zeros(T, self.num_features, dtype=torch.float32)
            
            for lab in patient_labs:
                if in_time <= lab.timestamp < out_time:
                    hour = int((lab.timestamp - in_time).total_seconds() // 3600)
                    if 0 <= hour < T:
                        item_id = str(lab.attr_dict.get("itemid"))
                        val = lab.attr_dict.get("valuenum")
                        if val is not None:
                            try:
                                f_idx = self.feature_map[item_id]
                                mean, std = self.norm_stats.get(item_id, (0.0, 1.0))
                                norm_val = (float(val) - mean) / (std if std > 0 else 1.0)
                                raw_grid[hour, f_idx] = norm_val
                                mask[hour, f_idx] = 1.0
                            except (ValueError, TypeError):
                                pass
                                
            # Require at least one recorded observation during the stay (Section 4.1)
            if mask.sum() == 0:
                continue
                
            # 3. Apply forward filling and compute decay indicators: 0.75^j
            vitals_ff, vitals_decay = generate_tpc_tensors(raw_grid, mask)
            
            # 4. Hourly Remaining LoS Target (anchored at end of hour t with floor matching prediction)
            los_targets = torch.tensor(
                [
                    max((out_time - (in_time + timedelta(hours=t + 1))).total_seconds() / 86400.0, LOS_FLOOR_DAYS)
                    for t in range(T)
                ],
                dtype=torch.float32,
            ).unsqueeze(1)  # Shape (T, 1)
            
            # 5. In-Hospital Mortality Label
            hadm_id = str(icustay.attr_dict.get("hadm_id", ""))
            mort_val = 1.0 if hadm_expire_map.get(hadm_id, False) else 0.0
            
            static_tensor = torch.tensor([true_age / 100.0, gender_val], dtype=torch.float32)
            stay_mask = torch.ones(T, dtype=torch.bool)
            
            sample = {
                "visit_id": str(icustay.attr_dict.get("stay_id", hadm_id)),
                "patient_id": str(patient.patient_id),
                "vitals": vitals_ff,
                "vitals_decay": vitals_decay,
                "static": static_tensor,
                "mask": stay_mask,
                "los": los_targets,
                "mortality": torch.tensor([mort_val], dtype=torch.float32),
            }
            samples.append(sample)
            
        return samples


def run_ablation_study(
    train_loader,
    val_loader,
    test_loader,
    continuous_dim: int = 12,
    static_dim: int = 2,
):
    """Runs the required ablation experiment comparing TPC configurations (Table 3 in paper).
    
    Ablations evaluated:
    1. Full TPC (MSLE Loss + Decay Indicators)
    2. TPC with MSE Loss
    3. TPC without Decay Indicators (architectural removal of decay channel)
    """
    print("\n" + "=" * 80)
    print("RUNNING ABLATION STUDY: EVALUATING LOSS FUNCTIONS & DECAY MECHANISMS")
    print("=" * 80)
    
    configs = [
        {"name": "Full TPC (MSLE Loss + Decay)", "loss_fn": "msle", "use_decay": True},
        {"name": "TPC Ablation: MSE Loss", "loss_fn": "mse", "use_decay": True},
        {"name": "TPC Ablation: No Decay", "loss_fn": "msle", "use_decay": False},
    ]
    
    def evaluate_loader(model, loader):
        model.eval()
        total_valid_hours = 0
        sum_msle = 0.0
        sum_mse = 0.0
        sum_mae = 0.0
        with torch.no_grad():
            for batch in loader:
                out = model(**batch)
                y_pred = out["y_prob"]
                y_true = batch["los"]
                mask = batch["mask"].bool()
                
                T = y_pred.shape[1]
                if T > 5:
                    m = mask.clone()
                    m[:, :5] = False  # Evaluate on hours t >= 5 as per Section 5.1.1
                    yp = y_pred[m]
                    yt = y_true[m]
                else:
                    yp = y_pred[mask]
                    yt = y_true[mask]
                    
                n_hrs = yp.numel()
                if n_hrs > 0:
                    total_valid_hours += n_hrs
                    sum_msle += torch.sum((torch.log(yp) - torch.log(yt)) ** 2).item()
                    sum_mse += torch.sum((yp - yt) ** 2).item()
                    sum_mae += torch.sum(torch.abs(yp - yt)).item()
                    
        return (
            sum_msle / max(total_valid_hours, 1),
            sum_mse / max(total_valid_hours, 1),
            sum_mae / max(total_valid_hours, 1),
        )

    results = []
    
    for cfg in configs:
        print(f"\nTraining configuration: {cfg['name']}...")
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
            floor_days=LOS_FLOOR_DAYS,
            sequence_output=True,
            use_decay=cfg["use_decay"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 2 epochs
        model.train()
        for epoch in range(2):
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(**batch)
                out["loss"].backward()
                optimizer.step()
                
        # Evaluate on validation loader and held-out test loader
        val_msle, val_mse, val_mae = evaluate_loader(model, val_loader)
        test_msle, test_mse, test_mae = evaluate_loader(model, test_loader)
        
        results.append({
            "Configuration": cfg["name"],
            "Val MSLE": val_msle,
            "Test MSLE": test_msle,
            "Test MSE": test_mse,
            "Test MAE": test_mae,
        })
        
    print("\n" + "=" * 85)
    print("HELD-OUT TEST ABLATION STUDY RESULTS (Corresponds to Table 3 in Rocheteau et al.):")
    print(f"{'Configuration':<32} | {'Val MSLE':<10} | {'Test MSLE':<10} | {'Test MSE':<10} | {'Test MAE':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['Configuration']:<32} | {r['Val MSLE']:<10.4f} | {r['Test MSLE']:<10.4f} | {r['Test MSE']:<10.4f} | {r['Test MAE']:<10.4f}")
    print("=" * 85)


if __name__ == "__main__":
    mimic4_root = "./data/mimiciv/demo" 
    
    if not os.path.exists(mimic4_root):
        print(f"Please download the MIMIC-IV demo data and extract it to {mimic4_root}")
    else:
        print("1. Initializing MIMIC4Dataset with labevents...")
        # Note: MIMIC4Dataset automatically loads default tables: patients, admissions, icustays
        dataset = MIMIC4Dataset(
            ehr_root=mimic4_root,
            ehr_tables=["labevents"],
        ) 
        
        # Partition patients strictly BEFORE task processing to prevent any data leakage
        print("\n2. Partitioning patient cohort (70% Train, 15% Val, 15% Test)...")
        all_pids = sorted([str(pid) for pid in dataset.unique_patient_ids])
        rng = np.random.default_rng(42)
        perm_pids = rng.permutation(all_pids)
        n_p = len(perm_pids)
        n_train = int(n_p * 0.70)
        n_val = int(n_p * 0.15)
        train_pids = set(perm_pids[:n_train])
        val_pids = set(perm_pids[n_train : n_train + n_val])
        test_pids = set(perm_pids[n_train + n_val :])
        print(f"  Train patients: {len(train_pids)}, Val patients: {len(val_pids)}, Test patients: {len(test_pids)}")
        
        print("\n3. Computing empirical training statistics (strictly on train split, 1st/99th percentile clipping)...")
        emp_stats = compute_empirical_stats(dataset, TOP_LAB_ITEMS, patient_ids=train_pids)
        for k in TOP_LAB_ITEMS[:3]:
            print(f"  Item {k}: mean={emp_stats[k][0]:.2f}, std={emp_stats[k][1]:.2f}")
            
        print("\n4. Applying ICU Length of Stay Task (Hourly Remaining LoS)...")
        sample_dataset = dataset.set_task(LengthOfStayTask(norm_stats=emp_stats))
        print(f"Total valid ICU stays created: {len(sample_dataset)}")
        
        if len(sample_dataset) > 0:
            sample0 = sample_dataset[0]
            print("\nSample 0 summary:")
            print(f"  Patient ID: {sample0['patient_id']}")
            print(f"  Visit/Stay ID: {sample0['visit_id']}")
            print(f"  Vitals shape: {sample0['vitals'].shape}")
            print(f"  Vitals decay shape: {sample0['vitals_decay'].shape}")
            print(f"  Static features shape: {sample0['static'].shape}")
            print(f"  Mask shape: {sample0['mask'].shape}")
            print(f"  Hourly remaining LoS shape: {sample0['los'].shape}")
            print(f"  Remaining LoS at hour 0: {sample0['los'][0].item():.2f} days")
            print(f"  Remaining LoS at final hour: {sample0['los'][-1].item():.2f} days")
            
            # Check mortality prevalence across the cohort
            mort_count = sum(s["mortality"].item() for s in sample_dataset)
            mort_rate = mort_count / len(sample_dataset) * 100.0
            print(f"  Cohort In-Hospital Mortality Prevalence: {mort_count:.0f}/{len(sample_dataset)} ({mort_rate:.1f}%)")
            assert mort_count > 0, "Mortality extraction yielded 0 positive cases!"
            
            # Build patient subsets using patient_to_index to avoid data leakage
            p2idx = sample_dataset.patient_to_index
            train_indices = list(chain(*[p2idx[pid] for pid in train_pids if pid in p2idx]))
            val_indices = list(chain(*[p2idx[pid] for pid in val_pids if pid in p2idx]))
            test_indices = list(chain(*[p2idx[pid] for pid in test_pids if pid in p2idx]))
            
            # Sort train stays by sequence length to group similar lengths into batches and minimize padding
            train_indices.sort(key=lambda i: sample_dataset[i]["vitals"].shape[0])
            
            train_ds = sample_dataset.subset(train_indices)
            val_ds = sample_dataset.subset(val_indices)
            test_ds = sample_dataset.subset(test_indices)
            print(f"  Train stays: {len(train_ds)}, Val stays: {len(val_ds)}, Test stays: {len(test_ds)}")
            
            train_loader = get_dataloader(train_ds, batch_size=4, shuffle=False)
            val_loader = get_dataloader(val_ds, batch_size=4, shuffle=False)
            test_loader = get_dataloader(test_ds, batch_size=4, shuffle=False)
            
            # Run the ablation experiment with held-out test evaluation
            run_ablation_study(train_loader, val_loader, test_loader, continuous_dim=12, static_dim=2)
            
            print("\nEnd-to-end TPC pipeline & ablation successfully completed!")
        else:
            print("No valid ICU samples generated.")