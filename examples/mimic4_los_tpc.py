"""Temporal Pointwise Convolution (TPC) Length of Stay Prediction Pipeline.

Reproduces the computational pipeline and experimental protocol of:
    "Temporal Pointwise Convolutional Networks for Length of Stay Prediction
     in the Intensive Care Unit"
    Emma Rocheteau, Pietro Liò, Stephanie Hyland (ACM CHIL 2021)
    https://doi.org/10.1145/3450439.3451860

Architecture & Pipeline Alignment with Rocheteau et al.:
1. Multi-Modal Clinical Feature Pipeline:
   - Dynamic Time-Series Features:
     • 7 Core Bedside Vital Signs from chartevents (Heart Rate, Systolic BP,
       Diastolic BP, Mean BP, Respiration, SpO2, Temperature).
     • Laboratory Blood Tests dynamically selected from training cohort matching
       the paper's exact inclusion threshold (>= 25% stay prevalence and > 3 obs/stay).
   - 12 Static (Flat) Features matching Section 4.2 & Table 6:
     • Age, Gender, Admission Hour, Height, Weight, GCS Eyes, GCS Motor,
       GCS Verbal, Care Unit, Admission Location, Insurance, Race.
2. Data Leakage Prevention & Empirical Normalization:
   - Cohort split into Train (70%), Val (15%), Test (15%) before feature extraction.
   - Empirical mean/std computed strictly on training split with 1st/99th percentile clipping.
   - Standardized missing values filled with 0.0 represent the true population training mean.
3. Hourly Discretization & Decay Indicators:
   - Hourly aggregation from ICU admission out to 14 days (336 hours).
   - Exponential decay: x''_t = 0.75^j, where j is elapsed hours since last recording.
4. Target Alignment:
   - Remaining LoS in days anchored to the end of each hour t with a 30-minute floor.
5. Multitask Learning:
   - Joint hourly LoS regression (MSLE) + in-hospital mortality prediction (BCEWithLogits).
6. Length-Bucketed Batch Sampling:
   - Groups stays of similar length to minimize padding, shuffling batch order every epoch.
"""

from collections.abc import Iterator
from datetime import datetime, timedelta
from itertools import chain
import os
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from pyhealth.datasets import MIMIC4Dataset, collate_fn_dict_with_padding, get_dataloader
from pyhealth.tasks import BaseTask
from pyhealth.models.tpc import TPC

# 7 Core Bedside Vital Signs from chartevents (Rocheteau et al. CHIL 2021)
CORE_VITALS_MAP = {
    "220045": "Heart Rate",
    "220179": "NIBP Systolic",
    "220180": "NIBP Diastolic",
    "220181": "NIBP Mean",
    "220210": "Respiratory Rate",
    "220277": "SpO2",
    "223761": "Temperature (F)",
}

# Baseline clinical assessments extracted within admission window (Section 4.2 / Table 6)
BASELINE_STATIC_ITEMS = {
    226512: "weight",
    226730: "height",
    220739: "eyes",
    223901: "motor",
    223900: "verbal",
}

# Fallback core lab tests if dynamic selection is not used (for standalone testing)
DEFAULT_LAB_ITEMS = [
    "50868",  # Anion Gap
    "50882",  # Bicarbonate
    "50902",  # Chloride
    "50912",  # Creatinine
    "50931",  # Glucose
    "50960",  # Magnesium
    "50971",  # Potassium
    "50983",  # Sodium
    "51006",  # Urea Nitrogen (BUN)
    "51221",  # Hematocrit
    "51222",  # Hemoglobin
    "51265",  # Platelet Count
]

# Minimum floor for remaining LoS (1/48 days = 30 minutes) to prevent log singularities
LOS_FLOOR_DAYS = 1.0 / 48.0


def select_qualifying_labs(
    dataset: MIMIC4Dataset,
    train_pids: set[str],
    min_prevalence: float = 0.25,
    min_avg_obs: float = 3.0,
    max_labs: int = 45,
) -> list[str]:
    """Dynamically selects laboratory tests conforming to the paper's inclusion rule.
    
    Rocheteau et al. (ACM CHIL 2021, Section 4 & timeseries.sql):
    - Evaluates ICU stays in the training set to prevent data leakage.
    - Laboratory variables must be present in >= 25% of patient stays with > 3 observations per stay.
    - Selects up to `max_labs` (the paper selected 45 lab features in MIMIC-IV).
    """
    total_train_stays = 0
    obs_per_lab: dict[str, int] = {}
    stays_per_lab: dict[str, int] = {}

    for patient in dataset.iter_patients():
        if str(patient.patient_id) not in train_pids:
            continue
        icustays = patient.get_events("icustays")
        labs = patient.get_events("labevents")

        for _ in icustays:
            total_train_stays += 1
            stay_labs = set()
            for lab in labs:
                item_raw = lab.attr_dict.get("itemid")
                val = lab.attr_dict.get("valuenum")
                if item_raw is not None and val is not None:
                    item_id = str(item_raw).split(".")[0]
                    stay_labs.add(item_id)
                    obs_per_lab[item_id] = obs_per_lab.get(item_id, 0) + 1
            for item_id in stay_labs:
                stays_per_lab[item_id] = stays_per_lab.get(item_id, 0) + 1

    qualifying = []
    for item_id, count in stays_per_lab.items():
        prev = count / max(total_train_stays, 1)
        avg_obs = obs_per_lab[item_id] / count
        if prev >= min_prevalence and avg_obs > min_avg_obs:
            qualifying.append((prev, avg_obs, item_id))

    # Sort descending by stay prevalence, then average observations
    qualifying.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [item_id for _, _, item_id in qualifying[:max_labs]]

    if len(selected) < 5:
        # Fallback to defaults if training set is too small
        return DEFAULT_LAB_ITEMS
    return selected


def load_chartevents_cache(
    ehr_root: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, float]]]:
    """Loads bedside vital signs and baseline clinical measurements from chartevents.csv.gz.
    
    Extracts:
    - 7 core continuous ICU vitals across the stay.
    - 5 baseline static features: Admission Weight, Height, GCS Eyes, GCS Motor, GCS Verbal.
    """
    chart_path = os.path.join(ehr_root, "icu", "chartevents.csv.gz")
    if not os.path.exists(chart_path):
        print(f"Notice: {chart_path} not found. Proceeding with labevents only.")
        return {}, {}

    df_chart = pd.read_csv(
        chart_path,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
    )
    df_chart = df_chart[df_chart["valuenum"].notnull()]
    df_chart["charttime"] = pd.to_datetime(df_chart["charttime"])

    # Extract continuous vitals
    vitals_by_stay: dict[str, list[dict[str, Any]]] = {}
    vitals_ids = set(int(k) for k in CORE_VITALS_MAP.keys())
    df_v = df_chart[df_chart["itemid"].isin(vitals_ids)]
    for stay_id, grp in df_v.groupby("stay_id"):
        vitals_by_stay[str(stay_id)] = grp[["charttime", "itemid", "valuenum"]].to_dict("records")

    # Extract baseline static features
    baseline_static_by_stay: dict[str, dict[str, float]] = {}
    df_s = df_chart[df_chart["itemid"].isin(BASELINE_STATIC_ITEMS.keys())]
    for stay_id, grp in df_s.groupby("stay_id"):
        d: dict[str, float] = {}
        for _, row in grp.iterrows():
            feat_name = BASELINE_STATIC_ITEMS[int(row["itemid"])]
            if feat_name not in d:
                d[feat_name] = float(row["valuenum"])
        baseline_static_by_stay[str(stay_id)] = d

    return vitals_by_stay, baseline_static_by_stay


def compute_empirical_stats(
    dataset: MIMIC4Dataset,
    train_pids: set[str] | list[str] | None,
    lab_item_ids: list[str],
    vitals_by_stay: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Computes empirical mean and std for each feature with 1st/99th percentile outlier clipping.
    
    Computed strictly on the training patient split to avoid data leakage.
    Ensures that unobserved values (filled with 0 after z-score standardization)
    accurately represent the empirical population mean (0.0).
    """
    allowed_pids = set(train_pids) if train_pids is not None else None
    vals_by_item: dict[str, list[float]] = {item: [] for item in lab_item_ids}
    
    # Collect lab values from training split
    for patient in dataset.iter_patients():
        if allowed_pids is not None and str(patient.patient_id) not in allowed_pids:
            continue
        for lab in patient.get_events("labevents"):
            item_raw = lab.attr_dict.get("itemid")
            val = lab.attr_dict.get("valuenum")
            if item_raw is not None and val is not None:
                item_id = str(item_raw).split(".")[0]
                if item_id in vals_by_item:
                    try:
                        vals_by_item[item_id].append(float(val))
                    except (ValueError, TypeError):
                        pass

    # Collect vitals values from training stays
    if vitals_by_stay:
        train_stay_ids: set[str] = set()
        for patient in dataset.iter_patients():
            if allowed_pids is not None and str(patient.patient_id) not in allowed_pids:
                continue
            for stay in patient.get_events("icustays"):
                sid = str(stay.attr_dict.get("stay_id", ""))
                if sid:
                    train_stay_ids.add(sid)
                    
        for vid in CORE_VITALS_MAP.keys():
            if vid not in vals_by_item:
                vals_by_item[vid] = []
            int_vid = int(vid)
            for sid in train_stay_ids:
                for rec in vitals_by_stay.get(sid, []):
                    if rec["itemid"] == int_vid:
                        vals_by_item[vid].append(float(rec["valuenum"]))

    stats: dict[str, tuple[float, float]] = {}
    counts: dict[str, int] = {}
    all_keys = list(lab_item_ids) + ([k for k in CORE_VITALS_MAP.keys()] if vitals_by_stay else [])
    
    for item_id in all_keys:
        vals = vals_by_item.get(item_id, [])
        counts[item_id] = len(vals)
        if len(vals) > 0:
            arr = np.array(vals)
            p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
            clipped = np.clip(arr, p1, p99)
            mean = float(np.mean(clipped))
            std = float(np.std(clipped))
            stats[item_id] = (mean, std if std > 1e-4 else 1.0)
        else:
            stats[item_id] = (0.0, 1.0)
            
    return stats, counts


def generate_tpc_tensors(
    raw_grid: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates forward-filled time series and decay indicator tensors.
    
    Conforms to Rocheteau et al. (ACM CHIL 2021) Section 4.1:
    - Forward-fills across missing hourly intervals.
    - Computes decay indicators: x''_t = 0.75^j, where j is hours since last recording.
    - If never observed, value is 0.0 (population mean) and decay indicator is 0.0.
    
    Args:
        raw_grid: Standardized tensor of shape (seq_len, num_features).
        mask: Binary tensor of shape (seq_len, num_features), 1 if measured at step t.
        
    Returns:
        vitals_ff: Forward-filled tensor of shape (seq_len, num_features).
        vitals_decay: Decay indicators tensor of shape (seq_len, num_features).
    """
    seq_len, num_features = raw_grid.shape
    vitals_ff = raw_grid.clone()
    vitals_decay = torch.zeros_like(raw_grid)
    
    # Vector of hours elapsed since last measurement
    time_since_last = torch.full((num_features,), float("inf"), dtype=torch.float32)
    
    for t in range(seq_len):
        measured = (mask[t] == 1.0)
        time_since_last[measured] = 0.0
        time_since_last[~measured] += 1.0
        
        # Forward fill unmeasured values from previous hour
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
    - Multi-Modal Features:
      • Dynamic Time-Series: Bedside vitals + dynamic laboratory tests.
      • 12 Static (Flat) Features: Age, Gender, Admission Hour, Height, Weight,
        GCS Eyes, GCS Motor, GCS Verbal, Care Unit, Admission Location, Insurance, Race.
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
        vitals_by_stay: dict[str, list[dict[str, Any]]] | None = None,
        baseline_static_by_stay: dict[str, dict[str, float]] | None = None,
        max_stay_hours: int = 336,
    ):
        super(LengthOfStayTask, self).__init__()
        self.lab_item_ids = lab_item_ids if lab_item_ids is not None else DEFAULT_LAB_ITEMS
        self.vitals_by_stay = vitals_by_stay or {}
        self.baseline_static_by_stay = baseline_static_by_stay or {}
        self.norm_stats = norm_stats or {}
        self.max_stay_hours = max_stay_hours

        # Construct unified feature mapping
        # 1. Vital signs (if chartevents cache provided)
        self.vital_keys = list(CORE_VITALS_MAP.keys()) if self.vitals_by_stay else []
        # 2. Dynamic lab tests
        self.all_feature_ids = self.vital_keys + self.lab_item_ids
        self.feature_map = {item_id: idx for idx, item_id in enumerate(self.all_feature_ids)}
        self.num_features = len(self.all_feature_ids)

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
            if str(lab.attr_dict.get("itemid", "")).split(".")[0] in self.feature_map and lab.timestamp is not None
        ]
        
        # Build map of admission mortality flags and demographics
        hadm_expire_map: dict[str, bool] = {}
        adm_meta_map: dict[str, dict[str, str]] = {}
        for adm in admissions:
            hadm_id = adm.attr_dict.get("hadm_id")
            if hadm_id:
                hid_str = str(hadm_id)
                flag = str(adm.attr_dict.get("hospital_expire_flag", "0"))
                hadm_expire_map[hid_str] = (flag == "1")
                adm_meta_map[hid_str] = {
                    "admission_location": str(adm.attr_dict.get("admission_location", "")),
                    "insurance": str(adm.attr_dict.get("insurance", "")),
                    "race": str(adm.attr_dict.get("race", "")),
                }

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
                    
            stay_id_str = str(icustay.attr_dict.get("stay_id", ""))
            hadm_id = str(icustay.attr_dict.get("hadm_id", ""))
            
            # True age at admission (Section 4.2: intime.year - anchor_year + anchor_age)
            true_age = float(in_time.year - anchor_year + anchor_age)
            if true_age <= 18:
                continue
                
            stay_seconds = (out_time - in_time).total_seconds()
            stay_hours = int(stay_seconds // 3600)
            
            # Minimum ICU length of stay filter (>= 5 hours, Section 4.1)
            if stay_hours < 5:
                continue
                
            # Cap at first 14 days (336 hours, Section 5.1.1)
            T = min(stay_hours, self.max_stay_hours)
            if T <= 0:
                continue
                
            # Initialize 2D grid for time-series features
            raw_grid = torch.zeros((T, self.num_features), dtype=torch.float32)
            mask = torch.zeros((T, self.num_features), dtype=torch.float32)
            
            # 1. Fill bedside vital signs from chartevents (if cached)
            if self.vitals_by_stay and stay_id_str in self.vitals_by_stay:
                for rec in self.vitals_by_stay[stay_id_str]:
                    v_time = rec["charttime"]
                    item_id = str(rec["itemid"])
                    if item_id not in self.feature_map:
                        continue
                    f_idx = self.feature_map[item_id]
                    hour = int((v_time - in_time).total_seconds() // 3600)
                    if 0 <= hour < T:
                        val = float(rec["valuenum"])
                        mean, std = self.norm_stats.get(item_id, (0.0, 1.0))
                        norm_val = (val - mean) / (std if std > 1e-4 else 1.0)
                        raw_grid[hour, f_idx] = norm_val
                        mask[hour, f_idx] = 1.0

            # 2. Fill laboratory tests from labevents
            for lab in patient_labs:
                lab_time = lab.timestamp
                item_raw = lab.attr_dict.get("itemid")
                item_id = str(item_raw).split(".")[0]
                if item_id in self.feature_map:
                    f_idx = self.feature_map[item_id]
                    hour = int((lab_time - in_time).total_seconds() // 3600)
                    if 0 <= hour < T:
                        val_raw = lab.attr_dict.get("valuenum")
                        if val_raw is not None:
                            try:
                                val = float(val_raw)
                                mean, std = self.norm_stats.get(item_id, (0.0, 1.0))
                                norm_val = (val - mean) / (std if std > 1e-4 else 1.0)
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
            mort_val = 1.0 if hadm_expire_map.get(hadm_id, False) else 0.0
            
            # 6. Extract 12 Static (Flat) Features matching Section 4.2 / Table 6
            base_s = self.baseline_static_by_stay.get(stay_id_str, {})
            adm_m = adm_meta_map.get(hadm_id, {})
            
            age_norm = (true_age - 60.0) / 20.0
            adm_hour_norm = (float(in_time.hour) - 12.0) / 12.0
            height_norm = (base_s.get("height", 170.0) - 170.0) / 15.0
            weight_norm = (base_s.get("weight", 80.0) - 80.0) / 20.0
            eyes_norm = (base_s.get("eyes", 4.0) - 2.5) / 1.5
            motor_norm = (base_s.get("motor", 6.0) - 3.5) / 2.5
            verbal_norm = (base_s.get("verbal", 5.0) - 3.0) / 2.0
            careunit_norm = (float(hash(icustay.attr_dict.get("first_careunit", "")) % 10) - 4.5) / 4.5
            adm_loc_norm = (float(hash(adm_m.get("admission_location", "")) % 10) - 4.5) / 4.5
            ins_norm = (float(hash(adm_m.get("insurance", "")) % 10) - 4.5) / 4.5
            race_norm = (float(hash(adm_m.get("race", "")) % 10) - 4.5) / 4.5

            static_tensor = torch.tensor([
                age_norm,
                gender_val,
                adm_hour_norm,
                height_norm,
                weight_norm,
                eyes_norm,
                motor_norm,
                verbal_norm,
                careunit_norm,
                adm_loc_norm,
                ins_norm,
                race_norm,
            ], dtype=torch.float32)
            
            stay_mask = torch.ones(T, dtype=torch.bool)
            
            sample = {
                "visit_id": stay_id_str if stay_id_str else hadm_id,
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


class MapDataset(torch.utils.data.Dataset):
    """In-memory map-style dataset enabling custom BatchSampler and length bucketing."""
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class BucketBatchSampler(Sampler[list[int]]):
    """Batches sequences of similar length together to minimize zero padding overhead.
    
    Partitions stays into length-sorted buckets and shuffles batch execution order
    every epoch. This eliminates the systematic shortest-to-longest training bias
    while retaining the memory and computation benefits of length bucketing.
    """
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.argsort(self.lengths)
        batches = [
            indices[i : i + self.batch_size].tolist()
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = rng.permutation(len(batches))
            batches = [batches[i] for i in perm]
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    continuous_dim: int,
    static_dim: int = 12,
):
    """Runs the architectural ablation study reproducing Table 3 from Rocheteau et al.
    
    Ablations Evaluated:
    1. TPC (Proposed: MSLE Loss, Exponential Decay, Multi-Modal Inputs)
    2. TPC without Decay (Ablates missing data decay channels: use_decay=False)
    3. TPC with MSE Loss (Ablates MSLE loss: loss_fn='mse')
    
    Evaluates:
    - Val MSLE: Used for early stopping / model selection
    - Test MSLE: Mean Squared Logarithmic Error (Primary paper metric)
    - Test MSE: Mean Squared Error in days^2
    - Test MAE: Mean Absolute Error in days (MAD in paper)
    """
    print("\n" + "=" * 85)
    print("DEMO ABLATION STUDY: MULTI-MODAL TPC PIPELINE VERIFICATION")
    print(f"Features: {continuous_dim} Continuous Time-Series + {static_dim} Static Features")
    print("=" * 85)
    
    configs = [
        {"name": "TPC (Proposed)", "loss_fn": "msle", "use_decay": True},
        {"name": "TPC (No Decay)", "loss_fn": "msle", "use_decay": False},
        {"name": "TPC (MSE Loss)", "loss_fn": "mse", "use_decay": True},
    ]

    def evaluate_loader(model: TPC, loader: DataLoader) -> tuple[float, float, float]:
        model.eval()
        sum_msle = 0.0
        sum_mse = 0.0
        sum_mae = 0.0
        total_valid_hours = 0
        
        with torch.no_grad():
            for batch in loader:
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
                
                # Exclude warmup hours t < 4 (consistent with min_time=4 / 5h observation)
                T = yp_2d.shape[1]
                if T > 4:
                    m = mask.clone()
                    m[:, :4] = False
                    yp = yp_2d[m]
                    yt = yt_2d[m]
                else:
                    yp = yp_2d[mask]
                    yt = yt_2d[mask]
                    
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
    
    for seed_idx, cfg in enumerate(configs):
        print(f"\nTraining configuration: {cfg['name']}...")
        torch.manual_seed(42 + seed_idx)
        np.random.seed(42 + seed_idx)
        
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 2 epochs
        model.train()
        for _ in range(2):
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(**batch)
                out["loss"].backward()
                optimizer.step()
                
        # Evaluate on validation loader and held-out test loader
        val_msle, _, _ = evaluate_loader(model, val_loader)
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
        print("1. Initializing MIMIC4Dataset...")
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
        
        print("\n3. Loading bedside vital signs & baseline static items from chartevents...")
        vitals_by_stay, baseline_static_by_stay = load_chartevents_cache(mimic4_root)
        print(f"  Loaded vitals for {len(vitals_by_stay)} stays, baseline static for {len(baseline_static_by_stay)} stays.")

        print("\n4. Dynamically selecting laboratory tests (Paper inclusion rule on training split)...")
        qualifying_labs = select_qualifying_labs(dataset, train_pids, min_prevalence=0.25, min_avg_obs=3.0, max_labs=45)
        print(f"  Selected {len(qualifying_labs)} qualifying lab tests meeting >=25% stay prevalence & >3 obs/stay.")

        print("\n5. Computing empirical training statistics (strictly on train split, 1st/99th percentile clipping)...")
        emp_stats, feat_counts = compute_empirical_stats(dataset, train_pids, qualifying_labs, vitals_by_stay)
        total_ts_features = len(CORE_VITALS_MAP) + len(qualifying_labs)
        print(f"  Computed normalization parameters across {total_ts_features} time-series features.")
        print(f"  Core Vital Signs ({len(CORE_VITALS_MAP)}):")
        for vid, vname in CORE_VITALS_MAP.items():
            mean, std = emp_stats.get(vid, (0.0, 1.0))
            cnt = feat_counts.get(vid, 0)
            print(f"    {vname} ({vid}): {cnt:5d} measurements, mean={mean:7.2f}, std={std:6.2f}")
            
        print("\n6. Applying ICU Length of Stay Task (Multi-Modal Time Series + 12 Static Features)...")
        task = LengthOfStayTask(
            lab_item_ids=qualifying_labs,
            norm_stats=emp_stats,
            vitals_by_stay=vitals_by_stay,
            baseline_static_by_stay=baseline_static_by_stay,
        )
        sample_dataset = dataset.set_task(task)
        print(f"Total valid ICU stays created: {len(sample_dataset)}")
        
        if len(sample_dataset) > 0:
            sample0 = sample_dataset[0]
            print("\nSample 0 summary:")
            print(f"  Patient ID: {sample0['patient_id']}")
            print(f"  Visit/Stay ID: {sample0['visit_id']}")
            print(f"  Vitals shape (T x F): {sample0['vitals'].shape}")
            print(f"  Vitals decay shape (T x F): {sample0['vitals_decay'].shape}")
            print(f"  Static features shape (12,): {sample0['static'].shape}")
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
            
            train_samples = [sample_dataset[i] for i in train_indices]
            val_samples = [sample_dataset[i] for i in val_indices]
            test_samples = [sample_dataset[i] for i in test_indices]
            print(f"  Train stays: {len(train_samples)}, Val stays: {len(val_samples)}, Test stays: {len(test_samples)}")
            
            # Length bucketing sampler shuffles batch order every epoch while keeping stays of similar length together
            train_lengths = [s["vitals"].shape[0] for s in train_samples]
            train_sampler = BucketBatchSampler(train_lengths, batch_size=4, shuffle=True, seed=42)
            train_loader = DataLoader(
                MapDataset(train_samples),
                batch_sampler=train_sampler,
                collate_fn=collate_fn_dict_with_padding,
            )
            val_loader = DataLoader(
                MapDataset(val_samples),
                batch_size=4,
                shuffle=False,
                collate_fn=collate_fn_dict_with_padding,
            )
            test_loader = DataLoader(
                MapDataset(test_samples),
                batch_size=4,
                shuffle=False,
                collate_fn=collate_fn_dict_with_padding,
            )
            
            # Run the ablation experiment with held-out test evaluation
            run_ablation_study(
                train_loader,
                val_loader,
                test_loader,
                continuous_dim=task.num_features,
                static_dim=12,
            )
            
            print("\nEnd-to-end TPC pipeline & ablation successfully completed!")
        else:
            print("No valid ICU samples generated.")