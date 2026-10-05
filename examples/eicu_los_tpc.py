"""Temporal Pointwise Convolution (TPC) Length of Stay Prediction Pipeline for eICU.

Reproduces the computational pipeline and experimental protocol of:
    "Temporal Pointwise Convolutional Networks for Length of Stay Prediction
     in the Intensive Care Unit"
    Emma Rocheteau, Pietro Liò, Stephanie Hyland (ACM CHIL 2021)
    https://doi.org/10.1145/3450439.3451860

Primary Dataset:
    eICU Collaborative Research Database (v2.0)
    https://eicu-crd.mit.edu/

Implementation Notes & Methodological Alignment:
1. Multi-Modal Clinical Feature Pipeline (eICU):
   - Dynamic Time-Series Features:
     • Core Bedside Vital Signs (Heart Rate, Systolic BP, Diastolic BP,
       Respiratory Rate, SpO2, Temperature). Extracted from physicalExam
       and periodic vital records.
     • Laboratory Blood Tests dynamically selected on the training split
       matching the authors' official preprocessing pipeline (>= 25% stay
       prevalence and > 3 obs/stay within the stay window).
       Source: EmmaRocheteau/TPC-LoS-prediction (eICU_preprocessing/timeseries.sql).
   - Static (Flat) Features matching Section 4.1, Table 16, and flat_features.sql:
     • Continuous: Age, Admission Hour, Height, Weight, GCS Eyes, GCS Motor,
       GCS Verbal, Gender.
     • Missingness Indicators: Binary flags for unmeasured baseline height,
       weight, and GCS in the admission window [-24h, +5h].
     • Categorical (One-Hot Encoded with Unknown Slot): Unit Type, Admission Source,
       Unit Stay Type, Ethnicity. Vocabularies and empirical medians/means are
       built strictly from the training split.
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
from itertools import chain
import os
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from pyhealth.datasets import eICUDataset, collate_fn_dict_with_padding
from pyhealth.tasks import BaseTask
from pyhealth.models.tpc import TPC

# 6 Core Bedside Vital Signs extracted from physicalExam / vitalPeriodic
CORE_VITALS_MAP = {
    "heart_rate": "Heart Rate",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "respiration_rate": "Respiratory Rate",
    "spo2": "SpO2 (Oxygen Saturation)",
    "temperature": "Temperature",
}

# Substring matches for bedside vital sign paths in eICU physicalExam table
PHYSICAL_EXAM_VITALS_MAP = {
    "HR/HR Current": "heart_rate",
    "BP (systolic)/BP (systolic) Current": "systolic_bp",
    "BP (diastolic)/BP (diastolic) Current": "diastolic_bp",
    "Resp Rate/Resp Current": "respiration_rate",
    "O2 Sat%/O2 Sat% Current": "spo2",
    "Temperature": "temperature",
}

# Substring matches for baseline GCS components in eICU physicalExam table
BASELINE_GCS_MAP = {
    "Eyes Score": "eyes",
    "Motor Score": "motor",
    "Verbal Score": "verbal",
}

# Default clinical lab tests used if dataset is too small for statistical thresholding
# Derived from ld_commonlabs in eICU_preprocessing/timeseries.sql
DEFAULT_LAB_NAMES = [
    "potassium",
    "sodium",
    "chloride",
    "creatinine",
    "BUN",
    "glucose",
    "bicarbonate",
    "Hct",
    "Hgb",
    "platelets x 1000",
    "WBC x 1000",
    "calcium",
]

# Minimum floor for remaining LoS (1/48 days = 30 minutes) to prevent log singularities
LOS_FLOOR_DAYS = 1.0 / 48.0


def select_qualifying_labs(
    dataset: eICUDataset,
    train_pids: set[str],
    min_prevalence: float = 0.25,
    min_avg_obs: float = 3.0,
    max_labs: int = 45,
) -> tuple[list[str], bool]:
    """Dynamically selects laboratory tests conforming to the paper's inclusion rule.
    
    Evaluates ICU stays strictly on the training set to prevent data leakage.
    Counts measurements strictly within each ICU stay window [0, min(unitdischargeoffset, 20160 minutes)).
    Cohort filters (adults > 18, stays >= 5 hours) match the task definition exactly.
    
    Args:
        dataset: eICUDataset instance.
        train_pids: Patient IDs belonging to the training split.
        min_prevalence: Minimum stay prevalence threshold (default: 0.25).
        min_avg_obs: Minimum average observations per stay (default: 3.0).
        max_labs: Cap on number of qualifying labs (defaults to 45, matching eICU ld_commonlabs).
            
    Returns:
        selected_labs: List of qualifying lab names.
        threshold_met: True if selected via statistical thresholds, False if fallback was used.
    """
    total_train_stays = 0
    obs_per_lab: dict[str, int] = {}
    stays_per_lab: dict[str, int] = {}

    for patient in dataset.iter_patients():
        if str(patient.patient_id) not in train_pids:
            continue

        patient_stays = patient.get_events("patient")
        patient_labs = patient.get_events("lab")

        for stay in patient_stays:
            raw_offset = getattr(stay, "unitdischargeoffset", None)
            if raw_offset is None:
                continue
            try:
                discharge_offset = float(raw_offset)
            except (ValueError, TypeError):
                continue

            # Cohort filter: Minimum ICU length of stay >= 5 hours (300 minutes)
            if discharge_offset < 300.0:
                continue

            # Cohort filter: Adult ICU patients (> 18 years at admission)
            age_str = str(getattr(stay, "age", "63")).replace(">", "").strip()
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                age = 63.0
            if age <= 18:
                continue

            sid = str(getattr(stay, "patientunitstayid", ""))
            stay_end = min(discharge_offset, 20160.0)  # Capped at 14 days
            total_train_stays += 1
            stay_labs: set[str] = set()

            for lab in patient_labs:
                if str(getattr(lab, "patientunitstayid", "")) != sid:
                    continue
                loff_raw = getattr(lab, "labresultoffset", None)
                if loff_raw is None:
                    continue
                try:
                    loff = float(loff_raw)
                except (ValueError, TypeError):
                    continue
                if not (0.0 <= loff < stay_end):
                    continue

                name_raw = getattr(lab, "labname", None)
                val_raw = getattr(lab, "labresult", None)
                if name_raw is not None and val_raw is not None:
                    lab_name = str(name_raw).strip()
                    stay_labs.add(lab_name)
                    obs_per_lab[lab_name] = obs_per_lab.get(lab_name, 0) + 1

            for lab_name in stay_labs:
                stays_per_lab[lab_name] = stays_per_lab.get(lab_name, 0) + 1

    qualifying = []
    for lab_name, count in stays_per_lab.items():
        prev = count / max(total_train_stays, 1)
        avg_obs = obs_per_lab[lab_name] / count
        if prev >= min_prevalence and avg_obs > min_avg_obs:
            qualifying.append((prev, avg_obs, lab_name))

    qualifying.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [lab_name for _, _, lab_name in qualifying[:max_labs]]

    if len(selected) < 5:
        return DEFAULT_LAB_NAMES, False
    return selected, True


def load_eicu_vitals_cache(
    eicu_root: str,
    dataset: eICUDataset | None = None,
) -> tuple[dict[str, list[tuple[int, str, float]]], dict[str, dict[str, float]]]:
    """Loads bedside vital signs and admission-window baseline measurements for eICU.
    
    Adheres to Rocheteau et al. (adapted from flat_features.sql with true elapsed-time window):
    - Vital signs are sorted chronologically by offset so "last value in the hour wins"
      strictly follows temporal order.
    - Continuous vitals are returned as lightweight (hour, vital_key, val) primitive tuples
      rather than nested dictionaries with Timestamp objects. This reduces serialization
      and RAM overhead and eliminates pickling bottlenecks across multiprocessing workers.
    - Preserves legitimate zero/negative measurements (filtering non-positive values
      specifically for physical dimensions like height and weight).
    - Baseline measurements are restricted strictly to [-24h, +5h] relative to ICU admission.
    - If multiple baseline measurements exist in the window, sorts by proximity to admission time.
    """
    vitals_by_stay: dict[str, list[tuple[int, str, float]]] = {}
    baseline_static_by_stay: dict[str, dict[str, float]] = {}

    # 1. Check for physicalExam.csv in eicu_root
    pe_csv = os.path.join(eicu_root, "physicalExam.csv")
    pe_gz = os.path.join(eicu_root, "physicalExam.csv.gz")
    pe_path = pe_csv if os.path.exists(pe_csv) else (pe_gz if os.path.exists(pe_gz) else None)

    if pe_path is not None:
        df_pe = pd.read_csv(
            pe_path,
            usecols=["patientunitstayid", "physicalexamoffset", "physicalexampath", "physicalexamtext"],
        )
        df_pe = df_pe[df_pe["physicalexamtext"].notnull()]
        df_pe["offset"] = pd.to_numeric(df_pe["physicalexamoffset"], errors="coerce")
        df_pe = df_pe[df_pe["offset"].notnull()]
        # 1a. Continuous bedside vitals (sorted chronologically by offset)
        df_vitals = df_pe[df_pe["offset"] >= 0].sort_values(by="offset")
        for row in df_vitals.itertuples(index=False):
            sid = str(row.patientunitstayid)
            path = str(row.physicalexampath)
            text = str(row.physicalexamtext)
            hour = int(float(row.offset) // 60)
            for sub, vkey in PHYSICAL_EXAM_VITALS_MAP.items():
                if sub in path:
                    try:
                        val = float(text)
                        if sid not in vitals_by_stay:
                            vitals_by_stay[sid] = []
                        vitals_by_stay[sid].append((hour, vkey, val))
                    except (ValueError, TypeError):
                        pass
                    break

        # 1b. Baseline measurements restricted strictly to [-24h, +5h] = [-1440m, +300m]
        # Sorted by absolute distance to admission intime (offset=0) so closest observation wins
        df_windowed = df_pe[(df_pe["offset"] >= -1440.0) & (df_pe["offset"] <= 300.0)].copy()
        df_windowed["abs_offset"] = df_windowed["offset"].abs()
        df_windowed.sort_values(by="abs_offset", inplace=True)

        for row in df_windowed.itertuples(index=False):
            sid = str(row.patientunitstayid)
            path = str(row.physicalexampath)
            text = str(row.physicalexamtext)
            for sub, gkey in BASELINE_GCS_MAP.items():
                if sub in path:
                    try:
                        val = float(text)
                        if sid not in baseline_static_by_stay:
                            baseline_static_by_stay[sid] = {}
                        if gkey not in baseline_static_by_stay[sid]:
                            baseline_static_by_stay[sid][gkey] = val
                    except (ValueError, TypeError):
                        pass
                    break

    # 2. Check for optional vitalPeriodic.csv / vitalPeriodic.csv.gz
    vp_csv = os.path.join(eicu_root, "vitalPeriodic.csv")
    vp_gz = os.path.join(eicu_root, "vitalPeriodic.csv.gz")
    vp_path = vp_csv if os.path.exists(vp_csv) else (vp_gz if os.path.exists(vp_gz) else None)

    if vp_path is not None:
        try:
            df_vp = pd.read_csv(vp_path)
            if "observationoffset" in df_vp.columns:
                df_vp.sort_values(by="observationoffset", inplace=True)
                col_map = {
                    "heartrate": "heart_rate",
                    "systemicsystolic": "systolic_bp",
                    "systemicdiastolic": "diastolic_bp",
                    "respiration": "respiration_rate",
                    "sao2": "spo2",
                    "temperature": "temperature",
                }
                for row in df_vp.itertuples(index=False):
                    sid = str(getattr(row, "patientunitstayid", ""))
                    off = float(getattr(row, "observationoffset", 0.0))
                    if off < 0:
                        continue
                    hour = int(off // 60)
                    if sid not in vitals_by_stay:
                        vitals_by_stay[sid] = []
                    for c_name, vkey in col_map.items():
                        if hasattr(row, c_name):
                            c_val = getattr(row, c_name)
                            if pd.notnull(c_val):
                                try:
                                    vitals_by_stay[sid].append((hour, vkey, float(c_val)))
                                except (ValueError, TypeError):
                                    pass
        except Exception:
            pass

    return vitals_by_stay, baseline_static_by_stay


def compute_empirical_training_stats(
    dataset: eICUDataset,
    train_pids: set[str],
    lab_names: list[str],
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
    ts_vals: dict[str, list[float]] = {name: [] for name in lab_names}
    if vitals_by_stay:
        for vid in CORE_VITALS_MAP.keys():
            if vid not in ts_vals:
                ts_vals[vid] = []

    continuous_static_vals: dict[str, list[float]] = {
        "age": [], "height": [], "weight": [], "eyes": [], "motor": [], "verbal": []
    }
    cat_sets: dict[str, set[str]] = {
        "unittype": set(),
        "unitadmitsource": set(),
        "unitstaytype": set(),
        "ethnicity": set(),
    }
    
    train_stay_ids: set[str] = set()

    for patient in dataset.iter_patients():
        if str(patient.patient_id) not in train_pids:
            continue

        patient_stays = patient.get_events("patient")
        patient_labs = patient.get_events("lab")

        for stay in patient_stays:
            raw_offset = getattr(stay, "unitdischargeoffset", None)
            if raw_offset is None:
                continue
            try:
                discharge_offset = float(raw_offset)
            except (ValueError, TypeError):
                continue

            if discharge_offset < 300.0:
                continue

            age_str = str(getattr(stay, "age", "63")).replace(">", "").strip()
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                age = 63.0
            if age <= 18:
                continue

            sid = str(getattr(stay, "patientunitstayid", ""))
            train_stay_ids.add(sid)
            stay_end = min(discharge_offset, 20160.0)

            # Continuous static: age
            continuous_static_vals["age"].append(age)

            # Static categoricals
            cat_sets["unittype"].add(str(getattr(stay, "unittype", "")))
            cat_sets["unitadmitsource"].add(str(getattr(stay, "unitadmitsource", "")))
            cat_sets["unitstaytype"].add(str(getattr(stay, "unitstaytype", "")))
            cat_sets["ethnicity"].add(str(getattr(stay, "ethnicity", "")))

            # Height and Weight
            raw_h = getattr(stay, "admissionheight", None)
            raw_w = getattr(stay, "admissionweight", None)
            if raw_h is not None:
                try:
                    h_val = float(raw_h)
                    if h_val > 0:
                        continuous_static_vals["height"].append(h_val)
                except (ValueError, TypeError):
                    pass
            if raw_w is not None:
                try:
                    w_val = float(raw_w)
                    if w_val > 0:
                        continuous_static_vals["weight"].append(w_val)
                except (ValueError, TypeError):
                    pass

            # Baseline measurements from admission window
            if baseline_static_by_stay and sid in baseline_static_by_stay:
                base = baseline_static_by_stay[sid]
                for k in ["height", "weight", "eyes", "motor", "verbal"]:
                    if k in base:
                        continuous_static_vals[k].append(base[k])

            # Labs within stay window
            for lab in patient_labs:
                if str(getattr(lab, "patientunitstayid", "")) != sid:
                    continue
                loff_raw = getattr(lab, "labresultoffset", None)
                if loff_raw is None:
                    continue
                try:
                    loff = float(loff_raw)
                except (ValueError, TypeError):
                    continue
                if not (0.0 <= loff < stay_end):
                    continue

                name_raw = getattr(lab, "labname", None)
                val_raw = getattr(lab, "labresult", None)
                if name_raw is not None and val_raw is not None:
                    lab_name = str(name_raw).strip()
                    if lab_name in ts_vals:
                        try:
                            ts_vals[lab_name].append(float(val_raw))
                        except (ValueError, TypeError):
                            pass

    # Collect vitals values from training stays
    if vitals_by_stay:
        for sid in train_stay_ids:
            for hour, vkey, val in vitals_by_stay.get(sid, []):
                if vkey in ts_vals:
                    ts_vals[vkey].append(val)

    # 1. Empirical time-series normalization stats (mean, std)
    ts_stats: dict[str, tuple[float, float]] = {}
    ts_counts: dict[str, int] = {}
    all_keys = list(lab_names) + ([k for k in CORE_VITALS_MAP.keys()] if vitals_by_stay else [])

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
        "age": (63.0, 15.0, 63.0),
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
    """ICU Remaining Length of Stay (LoS) Prediction Task for eICU.
    
    Reproduces the task definition from Rocheteau et al. (ACM CHIL 2021).
    
    Key Cohort & Task Criteria (Section 4 & 5.1.1):
    - Adult ICU patients (> 18 years old at admission).
    - Minimum ICU length of stay of 5 hours (unitdischargeoffset >= 300 minutes).
    - At least one recorded clinical observation during the stay.
    - Capped at first 14 days (336 hours = 20,160 minutes).
    - Target: Remaining LoS in days anchored at the end of each hour t:
      y_t = max((unitdischargeoffset - (t + 1) * 60) / 1440.0, 1/48 days).

    PyHealth Caching & Multiprocessing Worker Serialization:
    - BaseDataset.set_task hashes vars(task) via uuid5 to determine cache directories,
      so any change to task parameters or vitals/stat dictionaries invalidates the cache.
    - vitals_by_stay is stored as compact (hour, item_id, val) primitive tuples rather
      than nested dicts with Timestamp objects, minimizing memory and eliminating
      multiprocessing worker pickling bottlenecks on large datasets.
    """
    task_name = "eICU Remaining Length of Stay Prediction"

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
        lab_names: list[str] | None = None,
        norm_stats: dict[str, tuple[float, float]] | None = None,
        vitals_by_stay: dict[str, list[tuple[int, str, float]]] | None = None,
        baseline_static_by_stay: dict[str, dict[str, float]] | None = None,
        static_stats: dict[str, tuple[float, float, float]] | None = None,
        cat_vocabs: dict[str, list[str]] | None = None,
        max_stay_hours: int = 336,
    ):
        super(LengthOfStayTask, self).__init__()
        self.lab_names = lab_names if lab_names is not None else DEFAULT_LAB_NAMES
        self.vitals_by_stay = vitals_by_stay or {}
        self.baseline_static_by_stay = baseline_static_by_stay or {}
        self.norm_stats = norm_stats or {}
        self.max_stay_hours = max_stay_hours

        # Construct time-series feature mapping
        self.vital_keys = list(CORE_VITALS_MAP.keys()) if self.vitals_by_stay else []
        self.all_feature_ids = self.vital_keys + self.lab_names
        self.feature_map = {item_id: idx for idx, item_id in enumerate(self.all_feature_ids)}
        self.continuous_dim = len(self.all_feature_ids)
        self.num_features = self.continuous_dim

        # Static continuous feature priors (mean, std, median)
        self.static_stats = static_stats or {
            "age": (63.0, 15.0, 63.0),
            "height": (170.0, 10.0, 170.0),
            "weight": (80.0, 20.0, 80.0),
            "eyes": (3.5, 0.8, 4.0),
            "motor": (5.5, 1.0, 6.0),
            "verbal": (4.5, 1.0, 5.0),
        }

        # Deterministic categorical vocabularies
        self.cat_vocabs = cat_vocabs or {
            "unittype": ["Med-Surg ICU", "CTICU", "MICU", "SICU", "CCU-CTICU"],
            "unitadmitsource": ["Emergency Department", "Operating Room", "Floor", "Direct Admit"],
            "unitstaytype": ["admit", "readmit", "transfer"],
            "ethnicity": ["Caucasian", "African American", "Hispanic", "Asian"],
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

        patient_stays = patient.get_events(event_type="patient")
        if not patient_stays:
            return []

        patient_labs = patient.get_events(event_type="lab")
        # Pre-filter labs belonging to feature map and sort chronologically
        filtered_labs = [
            lab for lab in patient_labs
            if str(getattr(lab, "labname", "")).strip() in self.feature_map
            and getattr(lab, "labresultoffset", None) is not None
        ]
        filtered_labs.sort(key=lambda lab: float(getattr(lab, "labresultoffset", 0.0)))

        for stay in patient_stays:
            sid = str(getattr(stay, "patientunitstayid", ""))
            raw_offset = getattr(stay, "unitdischargeoffset", None)
            if raw_offset is None:
                continue
            try:
                discharge_offset = float(raw_offset)
            except (ValueError, TypeError):
                continue

            # Cohort filter: Minimum ICU length of stay >= 5 hours (300 minutes)
            if discharge_offset < 300.0:
                continue

            # Cohort filter: Adult ICU patients (> 18 years at admission)
            age_str = str(getattr(stay, "age", "63")).replace(">", "").strip()
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                age = 63.0
            if age <= 18:
                continue

            stay_hours = int(discharge_offset // 60)
            T = min(stay_hours, self.max_stay_hours)
            if T <= 0:
                continue

            raw_grid = torch.zeros((T, self.num_features), dtype=torch.float32)
            mask = torch.zeros((T, self.num_features), dtype=torch.float32)

            # 1. Bedside vital signs from cache (if provided)
            # Pre-sorted chronologically: last value in the hour overwrites earlier ones
            if self.vitals_by_stay and sid in self.vitals_by_stay:
                for hour, vkey, val in self.vitals_by_stay[sid]:
                    if vkey not in self.feature_map:
                        continue
                    f_idx = self.feature_map[vkey]
                    if 0 <= hour < T:
                        mean, std = self.norm_stats.get(vkey, (0.0, 1.0))
                        raw_grid[hour, f_idx] = (val - mean) / (std if std > 1e-4 else 1.0)
                        mask[hour, f_idx] = 1.0

            # 2. Laboratory tests from lab table
            for lab in filtered_labs:
                if str(getattr(lab, "patientunitstayid", "")) != sid:
                    continue
                loff = float(getattr(lab, "labresultoffset", 0.0))
                hour = int(loff // 60)
                if 0 <= hour < T:
                    lab_name = str(getattr(lab, "labname", "")).strip()
                    val_raw = getattr(lab, "labresult", None)
                    if val_raw is not None and lab_name in self.feature_map:
                        try:
                            val = float(val_raw)
                            f_idx = self.feature_map[lab_name]
                            mean, std = self.norm_stats.get(lab_name, (0.0, 1.0))
                            raw_grid[hour, f_idx] = (val - mean) / (std if std > 1e-4 else 1.0)
                            mask[hour, f_idx] = 1.0
                        except (ValueError, TypeError):
                            pass

            # Require at least one recorded observation during the stay (Section 4.1)
            if mask.sum() == 0:
                continue

            # 3. Forward filling & exponential decay indicators: 0.75^j
            vitals_ff, vitals_decay = generate_tpc_tensors(raw_grid, mask)

            # 4. Hourly Remaining LoS Target anchored at end of hour t:
            # y_t = max((discharge_offset - (t + 1) * 60) / 1440.0, 1/48 days)
            los_targets = torch.tensor(
                [
                    max((discharge_offset - (t + 1) * 60.0) / 1440.0, LOS_FLOOR_DAYS)
                    for t in range(T)
                ],
                dtype=torch.float32,
            ).unsqueeze(1)

            # 5. In-Hospital Mortality Label
            mort_str = str(getattr(stay, "hospitaldischargestatus", "")).lower()
            mort_val = 1.0 if mort_str == "expired" else 0.0

            # 6. Static Features (Continuous standardized + Missingness flags + One-Hot Categoricals)
            base_s = self.baseline_static_by_stay.get(sid, {})

            # Continuous standardized with train split parameters and median imputation
            age_m, age_s, _ = self.static_stats["age"]
            age_norm = (age - age_m) / age_s

            adm_time_str = str(getattr(stay, "unitadmittime24", "12:00:00"))
            try:
                adm_hour = float(adm_time_str.split(":")[0])
            except (ValueError, IndexError):
                adm_hour = 12.0
            adm_hour_norm = (adm_hour - 12.0) / 12.0

            # Height
            h_m, h_s, h_med = self.static_stats["height"]
            raw_h = getattr(stay, "admissionheight", None)
            if raw_h is not None and str(raw_h).strip() != "":
                try:
                    h_val = float(raw_h)
                    height_norm = (h_val - h_m) / h_s
                    missing_height = 0.0
                except (ValueError, TypeError):
                    height_norm = (h_med - h_m) / h_s
                    missing_height = 1.0
            elif "height" in base_s:
                height_norm = (base_s["height"] - h_m) / h_s
                missing_height = 0.0
            else:
                height_norm = (h_med - h_m) / h_s
                missing_height = 1.0

            # Weight
            w_m, w_s, w_med = self.static_stats["weight"]
            raw_w = getattr(stay, "admissionweight", None)
            if raw_w is not None and str(raw_w).strip() != "":
                try:
                    w_val = float(raw_w)
                    weight_norm = (w_val - w_m) / w_s
                    missing_weight = 0.0
                except (ValueError, TypeError):
                    weight_norm = (w_med - w_m) / w_s
                    missing_weight = 1.0
            elif "weight" in base_s:
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

            # Gender: 1.0 for Male, 0.0 for Female
            gender_str = str(getattr(stay, "gender", "Female")).lower()
            gender_val = 1.0 if gender_str in ("male", "m") else 0.0

            # One-hot encoded categoricals with dedicated unknown slot
            unittype_oh = self._one_hot(str(getattr(stay, "unittype", "")), self.cat_vocabs["unittype"])
            unitadmitsource_oh = self._one_hot(str(getattr(stay, "unitadmitsource", "")), self.cat_vocabs["unitadmitsource"])
            unitstaytype_oh = self._one_hot(str(getattr(stay, "unitstaytype", "")), self.cat_vocabs["unitstaytype"])
            ethnicity_oh = self._one_hot(str(getattr(stay, "ethnicity", "")), self.cat_vocabs["ethnicity"])

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
            ] + unittype_oh + unitadmitsource_oh + unitstaytype_oh + ethnicity_oh

            static_tensor = torch.tensor(static_list, dtype=torch.float32)
            stay_mask = torch.ones(T, dtype=torch.bool)

            sample = {
                "visit_id": sid,
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
    
    Ablations Evaluated (Table 3, Rocheteau et al.):
    1. TPC (Proposed): MSLE Loss, Exponential Decay, Multi-Modal Inputs
    2. TPC (No Decay): Ablates missing data decay channels (use_decay=False)
    3. TPC (MSE Loss): Ablates logarithmic MSLE loss in favor of standard MSE loss
    
    Experimental Control:
    - Same random seed and batch order across configs (initial weights are identical only for configs with the same architecture).
    - Sampler reset ensures identical mini-batch sequences across all configs.
    - Fixed 2-epoch training is used here for rapid demonstration and automated testing.
    """
    print("\n" + "=" * 85)
    print("HELD-OUT TEST ABLATION STUDY: MULTI-MODAL TPC PIPELINE VERIFICATION (eICU)")
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
    print("HELD-OUT TEST ABLATION STUDY RESULTS (eICU, 2 epochs – pipeline check only):")
    print(f"{'Configuration':<32} | {'Val MSLE':<10} | {'Test MSLE':<10} | {'Test MSE':<10} | {'Test MAE':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['Configuration']:<32} | {r['Val MSLE']:<10.4f} | {r['Test MSLE']:<10.4f} | {r['Test MSE']:<10.4f} | {r['Test MAE']:<10.4f}")
    print("=" * 85)


if __name__ == "__main__":
    eicu_root = "./test-resources/core/eicudemo"
    if not os.path.exists(eicu_root):
        eicu_root = "https://storage.googleapis.com/pyhealth/eicu-demo/"

    print("1. Initializing eICUDataset...")
    dataset = eICUDataset(
        root=eicu_root,
        tables=["lab", "physicalexam", "diagnosis"],
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

    print("\n3. Loading bedside vital signs & admission baseline items from physicalExam...")
    vitals_by_stay, baseline_static_by_stay = load_eicu_vitals_cache(eicu_root, dataset)
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
        print(f"  Remaining LoS at final hour: {sample0['los'][-1].item():.2f} days")

        mort_count = sum(s["mortality"].item() for s in sample_dataset)
        mort_rate = mort_count / len(sample_dataset) * 100.0
        print(f"  Cohort In-Hospital Mortality Prevalence: {mort_count:.0f}/{len(sample_dataset)} ({mort_rate:.1f}%)")

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

        print("\nEnd-to-end eICU TPC pipeline & unconfounded ablation successfully completed!")
    else:
        print("No valid ICU samples generated.")
