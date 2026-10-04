"""Temporal Pointwise Convolution (TPC) Length of Stay Prediction Pipeline.

Reproduces the computational pipeline and experimental protocol of:
    "Temporal Pointwise Convolutional Networks for Length of Stay Prediction
     in the Intensive Care Unit"
    Emma Rocheteau, Pietro Liò, Stephanie Hyland (ACM CHIL 2021)
    https://doi.org/10.1145/3450439.3451860

Implementation Notes & Methodological Alignment:
1. Multi-Modal Clinical Feature Pipeline:
   - Dynamic Time-Series Features:
     • 7 Core Bedside Vital Signs from chartevents (Heart Rate, Systolic BP,
       Diastolic BP, Mean BP, Respiration, SpO2, Temperature).
     • Laboratory Blood Tests dynamically selected on the training split
       matching the authors' official preprocessing pipeline (>= 25% stay
       prevalence and > 3 obs/stay within the stay window).
       Source: EmmaRocheteau/TPC-LoS-prediction (MIMIC_preprocessing/timeseries.sql).
   - Static (Flat) Features matching Section 4.2, Table 6, and flat_features.sql:
     • Continuous: Age, Admission Hour, Height, Weight, GCS Eyes, GCS Motor,
       GCS Verbal, Gender.
     • Missingness Indicators: Binary flags for unmeasured baseline height,
       weight, and GCS in the admission window [-24h, +5h].
     • Categorical (One-Hot Encoded with Unknown Slot): First Care Unit,
       Admission Location, Insurance, Race/Ethnicity. Vocabularies and
       empirical medians/means are built strictly from the training split.
2. Data Leakage Prevention & Empirical Normalization:
   - Patient cohort partitioned into Train (70%), Val (15%), Test (15%) before
     feature selection or statistics computation.
   - Lab selection and normalization parameters computed strictly on train split
     with 1st/99th percentile clipping.
   - Baseline measurements restricted to [-24h, +5h] relative to ICU admission;
     lookahead leakage is prevented for predictions at t >= 4h (enforced via
     warmup evaluation mask and min_time=4 on both LoS and mortality losses).
3. Hourly Discretization & Decay Indicators:
   - Hourly aggregation from ICU admission out to 14 days (336 hours).
   - Exponential decay: x''_t = 0.75^j, where j is elapsed hours since last recording.
4. Target Alignment:
   - Remaining LoS in days anchored to the end of each hour t with a 30-minute floor.
5. Multitask Learning:
   - Joint hourly LoS regression (MSLE) + in-hospital mortality prediction (BCEWithLogits).
6. Length-Bucketed Batch Sampling & Unconfounded Ablation:
   - Groups stays of similar length to minimize padding overhead.
   - Resets epoch counters and shares identical random seeds across configurations,
     ensuring the same random seed and batch order across configs (initial weights
     are identical only for configs with the same architecture).
7. Known Limitations & Caching:
   - BatchNorm over padded timesteps: 1D Batch Normalization in TPC blocks computes
     statistics across all steps including padding. Length-bucketed batch sampling
     greatly reduces padding overhead but does not completely eliminate it in uneven batches.
   - Cache key invalidation: PyHealth's BaseDataset.set_task hashes vars(task) via uuid5,
     so changes to norm_stats, static_stats, cat_vocabs, or vitals dicts automatically
     generate a new cache directory. Users can also call dataset.clean_tmpdir() to clear.
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
from pyhealth.datasets import MIMIC4Dataset, collate_fn_dict_with_padding
from pyhealth.tasks import BaseTask
from pyhealth.models.tpc import TPC

# 7 hand-selected core vitals from chartevents (Rocheteau et al. dynamically
# filtered chartevents by >=25% stay prevalence & >5 obs/stay, then hand-trimmed)
CORE_VITALS_MAP = {
    "220045": "Heart Rate",
    "220179": "NIBP Systolic",
    "220180": "NIBP Diastolic",
    "220181": "NIBP Mean",
    "220210": "Respiratory Rate",
    "220277": "SpO2",
    "223761": "Temperature (F)",
}

# Baseline clinical assessments extracted within admission window [-24h, +5h]
# adapted from flat_features.sql (true elapsed-time window)
BASELINE_STATIC_ITEMS = {
    226512: "weight",
    226730: "height",
    220739: "eyes",
    223901: "motor",
    223900: "verbal",
}

# Default clinical lab tests used if dataset is too small for statistical thresholding
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
) -> tuple[list[str], bool]:
    """Dynamically selects laboratory tests conforming to the paper's inclusion rule.
    
    Evaluates ICU stays strictly on the training set to prevent data leakage.
    Counts measurements strictly within each ICU stay window [in_time, min(out_time, in_time + 14d)).
    Cohort filters (adults > 18, stays >= 5 hours) match the task definition exactly.
    
    Args:
        dataset: MIMIC4Dataset instance.
        train_pids: Patient IDs belonging to the training split.
        min_prevalence: Minimum stay prevalence threshold (default: 0.25).
        min_avg_obs: Minimum average observations per stay (default: 3.0).
        max_labs: Cap on number of qualifying labs (defaults to 45, which
            corresponds to the number of qualifying lab tests produced on the paper's cohort).
            
    Returns:
        selected_labs: List of item IDs.
        threshold_met: True if selected via statistical thresholds, False if fallback was used.
    """
    total_train_stays = 0
    obs_per_lab: dict[str, int] = {}
    stays_per_lab: dict[str, int] = {}

    for patient in dataset.iter_patients():
        if str(patient.patient_id) not in train_pids:
            continue

        patients_info = patient.get_events("patients")
        if not patients_info:
            continue
        anchor_age = float(patients_info[0].attr_dict.get("anchor_age", 0))
        anchor_year = float(patients_info[0].attr_dict.get("anchor_year", 0))

        icustays = patient.get_events("icustays")
        labs = patient.get_events("labevents")

        for stay in icustays:
            in_time = stay.timestamp
            out_time_raw = stay.attr_dict.get("outtime")
            if in_time is None or out_time_raw is None:
                continue
            if isinstance(out_time_raw, datetime):
                out_time = out_time_raw
            else:
                try:
                    out_time = datetime.strptime(str(out_time_raw), "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    continue

            # Cohort filter: Adult ICU patients (> 18 years at admission)
            true_age = float(in_time.year - anchor_year + anchor_age)
            if true_age <= 18:
                continue

            # Cohort filter: Minimum ICU length of stay >= 5 hours
            stay_hours = int((out_time - in_time).total_seconds() // 3600)
            if stay_hours < 5:
                continue
                    
            stay_end = min(out_time, in_time + timedelta(days=14))
            total_train_stays += 1
            stay_labs: set[str] = set()
            
            for lab in labs:
                if lab.timestamp is None or not (in_time <= lab.timestamp < stay_end):
                    continue
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

    qualifying.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [item_id for _, _, item_id in qualifying[:max_labs]]

    if len(selected) < 5:
        return DEFAULT_LAB_ITEMS, False
    return selected, True


def load_chartevents_cache(
    ehr_root: str,
    dataset: MIMIC4Dataset,
) -> tuple[dict[str, list[tuple[int, str, float]]], dict[str, dict[str, float]]]:
    """Loads bedside vital signs and admission-window baseline measurements from chartevents.
    
    Adheres to Rocheteau et al. (adapted from flat_features.sql with true elapsed-time window):
    - Vital signs are sorted chronologically by charttime so "last value in the hour wins"
      strictly follows temporal order rather than file order.
    - Continuous vitals are returned as lightweight (hour, item_id, val) primitive tuples
      rather than nested dictionaries with Timestamp objects. This reduces serialization
      and RAM overhead and eliminates pickling bottlenecks across multiprocessing workers.
    - Preserves legitimate zero/negative clinical measurements in chartevents (filtering
      non-positive values specifically for physical dimensions like height and weight).
    - Baseline measurements are restricted strictly to [-24h, +5h] relative to ICU admission.
    - If multiple baseline measurements exist in the window, sorts by proximity to admission time.
    - itertuples() is used for efficient row extraction in this demo data adapter.
    """
    chart_path = os.path.join(ehr_root, "icu", "chartevents.csv.gz")
    if not os.path.exists(chart_path):
        return {}, {}

    df_chart = pd.read_csv(
        chart_path,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
    )
    df_chart = df_chart[df_chart["valuenum"].notnull()]
    df_chart["charttime"] = pd.to_datetime(df_chart["charttime"])

    # Build lookup of admission intime per stay_id from PyHealth dataset
    stay_intimes: dict[str, datetime] = {}
    for patient in dataset.iter_patients():
        for stay in patient.get_events("icustays"):
            sid = str(stay.attr_dict.get("stay_id", ""))
            if sid and stay.timestamp is not None:
                stay_intimes[sid] = stay.timestamp

    # 1. Continuous vital signs across the stay (sorted chronologically)
    vitals_by_stay: dict[str, list[tuple[int, str, float]]] = {}
    vitals_ids = set(int(k) for k in CORE_VITALS_MAP.keys())
    df_v = df_chart[df_chart["itemid"].isin(vitals_ids)].sort_values(by="charttime")
    for stay_id, grp in df_v.groupby("stay_id"):
        sid_str = str(stay_id)
        if sid_str not in stay_intimes:
            continue
        in_time = stay_intimes[sid_str]
        offsets = ((grp["charttime"] - in_time).dt.total_seconds() // 3600).astype(int)
        records = []
        for h, itemid, val in zip(offsets, grp["itemid"], grp["valuenum"]):
            if h >= 0:
                records.append((int(h), str(itemid), float(val)))
        if records:
            vitals_by_stay[sid_str] = records

    # 2. Baseline static measurements restricted strictly to [-24h, +5h] of admission
    baseline_static_by_stay: dict[str, dict[str, float]] = {}
    df_s = df_chart[df_chart["itemid"].isin(BASELINE_STATIC_ITEMS.keys())].copy()
    
    for stay_id, grp in df_s.groupby("stay_id"):
        sid_str = str(stay_id)
        if sid_str not in stay_intimes:
            continue
        in_time = stay_intimes[sid_str]
        
        # Calculate offset in hours relative to admission intime
        grp_offsets = (grp["charttime"] - in_time).dt.total_seconds() / 3600.0
        # Restrict strictly between -24h and +5h (flat_features.sql)
        windowed = grp[(grp_offsets >= -24.0) & (grp_offsets <= 5.0)].copy()
        if len(windowed) == 0:
            continue
            
        # Sort by distance to admission time to select closest baseline measurement
        windowed["abs_offset"] = ((windowed["charttime"] - in_time).dt.total_seconds()).abs()
        windowed.sort_values(by="abs_offset", inplace=True)
        
        d: dict[str, float] = {}
        for row in windowed.itertuples(index=False):
            feat_name = BASELINE_STATIC_ITEMS[int(row.itemid)]
            val = float(row.valuenum)
            # Filter non-positive values only for physical dimensions (height, weight)
            if feat_name in ("height", "weight") and val <= 0:
                continue
            if feat_name not in d:
                d[feat_name] = val
        baseline_static_by_stay[sid_str] = d

    return vitals_by_stay, baseline_static_by_stay


def compute_empirical_training_stats(
    dataset: MIMIC4Dataset,
    train_pids: set[str],
    lab_item_ids: list[str],
    vitals_by_stay: dict[str, list[tuple[int, str, float]]] | None = None,
    baseline_static_by_stay: dict[str, dict[str, float]] | None = None,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, int],
    dict[str, tuple[float, float, float]],
    dict[str, list[str]],
]:
    """Computes empirical training statistics and builds fixed categorical vocabularies.
    
    Strictly isolated to the training split and identical cohort filters (adults > 18, stays >= 5h):
    - Time-series: empirical mean and std with 1st/99th percentile clipping.
    - Continuous static: empirical mean, std, and median (for imputation) on training stays.
    - Categorical static: deterministic sorted vocabulary for one-hot encoding.
    """
    ts_vals: dict[str, list[float]] = {item: [] for item in lab_item_ids}
    if vitals_by_stay:
        for vid in CORE_VITALS_MAP.keys():
            if vid not in ts_vals:
                ts_vals[vid] = []

    continuous_static_vals: dict[str, list[float]] = {
        "age": [], "height": [], "weight": [], "eyes": [], "motor": [], "verbal": []
    }
    cat_sets: dict[str, set[str]] = {
        "first_careunit": set(),
        "admission_location": set(),
        "insurance": set(),
        "race": set(),
    }
    
    train_stay_ids: set[str] = set()

    for patient in dataset.iter_patients():
        if str(patient.patient_id) not in train_pids:
            continue
            
        patients_info = patient.get_events("patients")
        if not patients_info:
            continue
        p_info = patients_info[0]
        anchor_age = float(p_info.attr_dict.get("anchor_age", 0))
        anchor_year = float(p_info.attr_dict.get("anchor_year", 0))
        
        admissions = patient.get_events("admissions")
        adm_map = {str(a.attr_dict.get("hadm_id")): a for a in admissions}
        icustays = patient.get_events("icustays")

        # Collect lab values within ICU stays
        for stay in icustays:
            in_time = stay.timestamp
            out_time_raw = stay.attr_dict.get("outtime")
            if in_time is None or out_time_raw is None:
                continue
            if isinstance(out_time_raw, datetime):
                out_time = out_time_raw
            else:
                try:
                    out_time = datetime.strptime(str(out_time_raw), "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    continue

            # Cohort filter: Adult ICU patients (> 18 years at admission)
            true_age = float(in_time.year - anchor_year + anchor_age)
            if true_age <= 18:
                continue

            # Cohort filter: Minimum ICU length of stay >= 5 hours
            stay_hours = int((out_time - in_time).total_seconds() // 3600)
            if stay_hours < 5:
                continue

            stay_end = min(out_time, in_time + timedelta(days=14))
            
            sid = str(stay.attr_dict.get("stay_id", ""))
            hadm_id = str(stay.attr_dict.get("hadm_id", ""))
            train_stay_ids.add(sid)
            
            # Static continuous age
            continuous_static_vals["age"].append(true_age)
            
            # Static categoricals
            cat_sets["first_careunit"].add(str(stay.attr_dict.get("first_careunit", "")))
            if hadm_id in adm_map:
                adm = adm_map[hadm_id]
                cat_sets["admission_location"].add(str(adm.attr_dict.get("admission_location", "")))
                cat_sets["insurance"].add(str(adm.attr_dict.get("insurance", "")))
                cat_sets["race"].add(str(adm.attr_dict.get("race", "")))
                
            # Baseline measurements from admission window
            if baseline_static_by_stay and sid in baseline_static_by_stay:
                base = baseline_static_by_stay[sid]
                for k in ["height", "weight", "eyes", "motor", "verbal"]:
                    if k in base:
                        continuous_static_vals[k].append(base[k])

            # Labs within stay
            for lab in patient.get_events("labevents"):
                if lab.timestamp is None or not (in_time <= lab.timestamp < stay_end):
                    continue
                item_raw = lab.attr_dict.get("itemid")
                val = lab.attr_dict.get("valuenum")
                if item_raw is not None and val is not None:
                    item_id = str(item_raw).split(".")[0]
                    if item_id in ts_vals:
                        try:
                            ts_vals[item_id].append(float(val))
                        except (ValueError, TypeError):
                            pass

    # Collect vitals values from training stays
    if vitals_by_stay:
        for sid in train_stay_ids:
            for hour, item_id, val in vitals_by_stay.get(sid, []):
                if item_id in ts_vals:
                    ts_vals[item_id].append(val)

    # 1. Empirical time-series normalization stats (mean, std)
    ts_stats: dict[str, tuple[float, float]] = {}
    ts_counts: dict[str, int] = {}
    all_keys = list(lab_item_ids) + ([k for k in CORE_VITALS_MAP.keys()] if vitals_by_stay else [])
    
    for item_id in all_keys:
        vals = ts_vals.get(item_id, [])
        ts_counts[item_id] = len(vals)
        if len(vals) > 0:
            arr = np.array(vals)
            p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
            clipped = np.clip(arr, p1, p99)
            mean = float(np.mean(clipped))
            std = float(np.std(clipped))
            ts_stats[item_id] = (mean, std if std > 1e-4 else 1.0)
        else:
            ts_stats[item_id] = (0.0, 1.0)

    # 2. Empirical continuous static stats: (mean, std, median)
    static_stats: dict[str, tuple[float, float, float]] = {}
    default_priors = {
        "age": (60.0, 15.0, 60.0),
        "height": (170.0, 10.0, 170.0),
        "weight": (80.0, 20.0, 80.0),
        "eyes": (3.5, 0.8, 4.0),
        "motor": (5.5, 1.0, 6.0),
        "verbal": (4.5, 1.0, 5.0),
    }
    for k, prior in default_priors.items():
        vals = continuous_static_vals.get(k, [])
        if len(vals) > 0:
            arr = np.array(vals)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            median = float(np.median(arr))
            static_stats[k] = (mean, std if std > 1e-4 else prior[1], median)
        else:
            static_stats[k] = prior

    # 3. Deterministic sorted categorical vocabularies
    cat_vocabs = {k: sorted(list(v)) for k, v in cat_sets.items()}
    
    return ts_stats, ts_counts, static_stats, cat_vocabs


def generate_tpc_tensors(
    raw_grid: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates forward-filled time series and decay indicator tensors.
    
    Conforms to Rocheteau et al. (ACM CHIL 2021) Section 4.1:
    - Forward-fills across missing hourly intervals.
    - Computes decay indicators: x''_t = 0.75^j, where j is hours since last recording.
    - If never observed, value is 0.0 (population mean) and decay indicator is 0.0.
    """
    seq_len, num_features = raw_grid.shape
    vitals_ff = raw_grid.clone()
    vitals_decay = torch.zeros_like(raw_grid)
    time_since_last = torch.full((num_features,), float("inf"), dtype=torch.float32)
    
    for t in range(seq_len):
        measured = (mask[t] == 1.0)
        time_since_last[measured] = 0.0
        time_since_last[~measured] += 1.0
        
        if t > 0:
            unmeasured = ~measured
            vitals_ff[t, unmeasured] = vitals_ff[t - 1, unmeasured]
            
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

    PyHealth Caching & Multiprocessing Worker Serialization:
    - BaseDataset.set_task hashes vars(task) via uuid5 to determine cache directories,
      so any change to task parameters or vitals/stat dictionaries invalidates the cache.
    - vitals_by_stay is stored as compact (hour, item_id, val) primitive tuples rather
      than nested dicts with Timestamp objects, minimizing memory and eliminating
      multiprocessing worker pickling bottlenecks on large datasets.
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
        vitals_by_stay: dict[str, list[tuple[int, str, float]]] | None = None,
        baseline_static_by_stay: dict[str, dict[str, float]] | None = None,
        static_stats: dict[str, tuple[float, float, float]] | None = None,
        cat_vocabs: dict[str, list[str]] | None = None,
        max_stay_hours: int = 336,
    ):
        super(LengthOfStayTask, self).__init__()
        self.lab_item_ids = lab_item_ids if lab_item_ids is not None else DEFAULT_LAB_ITEMS
        self.vitals_by_stay = vitals_by_stay or {}
        self.baseline_static_by_stay = baseline_static_by_stay or {}
        self.norm_stats = norm_stats or {}
        self.max_stay_hours = max_stay_hours

        # Construct time-series feature mapping
        self.vital_keys = list(CORE_VITALS_MAP.keys()) if self.vitals_by_stay else []
        self.all_feature_ids = self.vital_keys + self.lab_item_ids
        self.feature_map = {item_id: idx for idx, item_id in enumerate(self.all_feature_ids)}
        self.continuous_dim = len(self.all_feature_ids)
        self.num_features = self.continuous_dim

        # Static continuous feature priors (mean, std, median)
        self.static_stats = static_stats or {
            "age": (60.0, 15.0, 60.0),
            "height": (170.0, 10.0, 170.0),
            "weight": (80.0, 20.0, 80.0),
            "eyes": (3.5, 0.8, 4.0),
            "motor": (5.5, 1.0, 6.0),
            "verbal": (4.5, 1.0, 5.0),
        }

        # Deterministic categorical vocabularies
        self.cat_vocabs = cat_vocabs or {
            "first_careunit": ["MICU", "SICU", "CCU", "CVICU", "TSICU"],
            "admission_location": ["EMERGENCY ROOM", "TRANSFER FROM HOSPITAL", "PHYSICIAN REFERRAL"],
            "insurance": ["Medicare", "Medicaid", "Other"],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO", "ASIAN"],
        }
        
        # 11 continuous/indicator features + one-hot categories (each with unknown slot)
        # Continuous (11): age, gender, adm_hour, height, missing_height, weight,
        #                  missing_weight, eyes, motor, verbal, missing_gcs
        self.static_dim = 11 + sum(len(vocab) + 1 for vocab in self.cat_vocabs.values())

    @staticmethod
    def _one_hot(val: str, vocab: list[str]) -> list[float]:
        vec = [0.0] * (len(vocab) + 1)
        if val in vocab:
            vec[vocab.index(val)] = 1.0
        else:
            vec[-1] = 1.0  # Unknown slot
        return vec

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
        
        raw_labs = patient.get_events(event_type="labevents")
        patient_labs = [
            lab for lab in raw_labs
            if str(lab.attr_dict.get("itemid", "")).split(".")[0] in self.feature_map and lab.timestamp is not None
        ]
        # Chronological sort: guarantees "last value in the hour wins" follows time, not file order
        patient_labs.sort(key=lambda lab: lab.timestamp)
        
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
            
            # True age at admission (Section 4.2)
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
                
            raw_grid = torch.zeros((T, self.num_features), dtype=torch.float32)
            mask = torch.zeros((T, self.num_features), dtype=torch.float32)
            
            # 1. Bedside vital signs from chartevents (if provided)
            # Pre-sorted chronologically: last value in the hour overwrites earlier ones
            if self.vitals_by_stay and stay_id_str in self.vitals_by_stay:
                for hour, item_id, val in self.vitals_by_stay[stay_id_str]:
                    if item_id not in self.feature_map:
                        continue
                    f_idx = self.feature_map[item_id]
                    if 0 <= hour < T:
                        mean, std = self.norm_stats.get(item_id, (0.0, 1.0))
                        raw_grid[hour, f_idx] = (val - mean) / (std if std > 1e-4 else 1.0)
                        mask[hour, f_idx] = 1.0

            # 2. Laboratory tests from labevents
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
                                raw_grid[hour, f_idx] = (val - mean) / (std if std > 1e-4 else 1.0)
                                mask[hour, f_idx] = 1.0
                            except (ValueError, TypeError):
                                pass
                                
            # Require at least one recorded observation during the stay (Section 4.1)
            if mask.sum() == 0:
                continue
                
            # 3. Forward filling & exponential decay indicators: 0.75^j
            vitals_ff, vitals_decay = generate_tpc_tensors(raw_grid, mask)
            
            # 4. Hourly Remaining LoS Target anchored at end of hour t
            los_targets = torch.tensor(
                [
                    max((out_time - (in_time + timedelta(hours=t + 1))).total_seconds() / 86400.0, LOS_FLOOR_DAYS)
                    for t in range(T)
                ],
                dtype=torch.float32,
            ).unsqueeze(1)
            
            # 5. In-Hospital Mortality Label
            mort_val = 1.0 if hadm_expire_map.get(hadm_id, False) else 0.0
            
            # 6. Static Features (Continuous standardized + Missingness flags + One-Hot Categoricals)
            base_s = self.baseline_static_by_stay.get(stay_id_str, {})
            adm_m = adm_meta_map.get(hadm_id, {})
            
            # Continuous standardized with train split parameters and median imputation
            age_m, age_s, _ = self.static_stats["age"]
            age_norm = (true_age - age_m) / age_s
            adm_hour_norm = (float(in_time.hour) - 12.0) / 12.0
            
            # Height
            h_m, h_s, h_med = self.static_stats["height"]
            if "height" in base_s:
                height_norm = (base_s["height"] - h_m) / h_s
                missing_height = 0.0
            else:
                height_norm = (h_med - h_m) / h_s
                missing_height = 1.0
                
            # Weight
            w_m, w_s, w_med = self.static_stats["weight"]
            if "weight" in base_s:
                weight_norm = (base_s["weight"] - w_m) / w_s
                missing_weight = 0.0
            else:
                weight_norm = (w_med - w_m) / w_s
                missing_weight = 1.0
                
            # GCS (Eyes, Motor, Verbal)
            e_m, e_s, e_med = self.static_stats["eyes"]
            m_m, m_s, m_med = self.static_stats["motor"]
            v_m, v_s, v_med = self.static_stats["verbal"]
            if "eyes" in base_s and "motor" in base_s and "verbal" in base_s:
                eyes_norm = (base_s["eyes"] - e_m) / e_s
                motor_norm = (base_s["motor"] - m_m) / m_s
                verbal_norm = (base_s["verbal"] - v_m) / v_s
                missing_gcs = 0.0
            else:
                eyes_norm = (e_med - e_m) / e_s
                motor_norm = (m_med - m_m) / m_s
                verbal_norm = (v_med - v_m) / v_s
                missing_gcs = 1.0

            # One-hot encoded categoricals with dedicated unknown slot
            careunit_oh = self._one_hot(str(icustay.attr_dict.get("first_careunit", "")), self.cat_vocabs["first_careunit"])
            adm_loc_oh = self._one_hot(adm_m.get("admission_location", ""), self.cat_vocabs["admission_location"])
            ins_oh = self._one_hot(adm_m.get("insurance", ""), self.cat_vocabs["insurance"])
            race_oh = self._one_hot(adm_m.get("race", ""), self.cat_vocabs["race"])

            static_list = [
                age_norm,
                gender_val,
                adm_hour_norm,
                height_norm,
                missing_height,
                weight_norm,
                missing_weight,
                eyes_norm,
                motor_norm,
                verbal_norm,
                missing_gcs,
            ] + careunit_oh + adm_loc_oh + ins_oh + race_oh

            static_tensor = torch.tensor(static_list, dtype=torch.float32)
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
    every epoch. Allows resetting epoch counter to ensure strictly unconfounded
    ablation comparisons across model configurations.
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

    def reset(self) -> None:
        """Resets the epoch counter so all ablation configurations see the exact same batch orders."""
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
    train_sampler: BucketBatchSampler,
    continuous_dim: int,
    static_dim: int,
    seed: int = 42,
):
    """Runs the architectural ablation study replicating the paper's ablation methodology.
    
    Ablations Evaluated:
    1. TPC (Proposed): MSLE Loss, Exponential Decay, Multi-Modal Inputs
    2. TPC (No Decay): Ablates missing data decay channels (use_decay=False)
    3. TPC (MSE Loss): Ablates logarithmic MSLE loss in favor of standard MSE loss
    
    Experimental Control:
    - Same random seed and batch order across configs (initial weights are identical only for configs with the same architecture).
    - Sampler reset ensures identical mini-batch sequences across all configs.
    - Fixed 2-epoch training is used here for rapid demonstration and automated testing.
    """
    print("\n" + "=" * 85)
    print("HELD-OUT TEST ABLATION STUDY: MULTI-MODAL TPC PIPELINE VERIFICATION")
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
    
    for cfg in configs:
        print(f"\nTraining configuration: {cfg['name']}...")
        
        # Reset sampler epoch so each configuration sees the exact same batch ordering
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        for _ in range(2):
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(**batch)
                out["loss"].backward()
                optimizer.step()
                
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
    print("HELD-OUT TEST ABLATION STUDY RESULTS (MIMIC-IV demo, 2 epochs – pipeline check only):")
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
        
        print("\n3. Loading bedside vital signs & admission baseline items from chartevents...")
        print("  (Note: Adapter for demo data; chartevents is large; this adapter reads the demo file directly)")
        vitals_by_stay, baseline_static_by_stay = load_chartevents_cache(mimic4_root, dataset)
        print(f"  Loaded vitals for {len(vitals_by_stay)} stays, baseline items for {len(baseline_static_by_stay)} stays.")

        print("\n4. Dynamically selecting laboratory tests (Paper inclusion rule on training stays)...")
        qualifying_labs, thresh_met = select_qualifying_labs(
            dataset, train_pids, min_prevalence=0.25, min_avg_obs=3.0, max_labs=45
        )
        if thresh_met:
            print(f"  Selected {len(qualifying_labs)} qualifying lab tests meeting >=25% stay prevalence & >3 obs/stay.")
        else:
            print(f"  Notice: Small cohort fallback used; selected {len(qualifying_labs)} default clinical labs.")

        print("\n5. Computing empirical training statistics & vocabularies (strictly on train split)...")
        emp_stats, feat_counts, static_stats, cat_vocabs = compute_empirical_training_stats(
            dataset, train_pids, qualifying_labs, vitals_by_stay, baseline_static_by_stay
        )
        total_ts_features = len(CORE_VITALS_MAP) + len(qualifying_labs)
        print(f"  Computed normalization parameters across {total_ts_features} time-series features.")
        print(f"  Core Vital Signs ({len(CORE_VITALS_MAP)}):")
        for vid, vname in CORE_VITALS_MAP.items():
            mean, std = emp_stats.get(vid, (0.0, 1.0))
            cnt = feat_counts.get(vid, 0)
            print(f"    {vname} ({vid}): {cnt:5d} measurements, mean={mean:7.2f}, std={std:6.2f}")
            
        print("\n6. Applying ICU Length of Stay Task (Multi-Modal Time Series + Static Features)...")
        task = LengthOfStayTask(
            lab_item_ids=qualifying_labs,
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
            print(f"  Remaining LoS at final hour: {sample0['los'][-1].item():.2f} days")
            
            mort_count = sum(s["mortality"].item() for s in sample_dataset)
            mort_rate = mort_count / len(sample_dataset) * 100.0
            print(f"  Cohort In-Hospital Mortality Prevalence: {mort_count:.0f}/{len(sample_dataset)} ({mort_rate:.1f}%)")
            assert mort_count > 0, "Mortality extraction yielded 0 positive cases!"
            
            p2idx = sample_dataset.patient_to_index
            train_indices = list(chain(*[p2idx[pid] for pid in train_pids if pid in p2idx]))
            val_indices = list(chain(*[p2idx[pid] for pid in val_pids if pid in p2idx]))
            test_indices = list(chain(*[p2idx[pid] for pid in test_pids if pid in p2idx]))
            
            train_samples = [sample_dataset[i] for i in train_indices]
            val_samples = [sample_dataset[i] for i in val_indices]
            test_samples = [sample_dataset[i] for i in test_indices]
            print(f"  Train stays: {len(train_samples)}, Val stays: {len(val_samples)}, Test stays: {len(test_samples)}")
            
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
            
            run_ablation_study(
                train_loader,
                val_loader,
                test_loader,
                train_sampler=train_sampler,
                continuous_dim=task.continuous_dim,
                static_dim=task.static_dim,
                seed=42,
            )
            
            print("\nEnd-to-end TPC pipeline & unconfounded ablation successfully completed!")
        else:
            print("No valid ICU samples generated.")