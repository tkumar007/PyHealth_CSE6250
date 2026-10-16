"""Author: Tamjit Kumar
NetID: tkumar70
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Temporal Pointwise Convolution (TPC) Length of Stay Prediction Task
             and multi-modal dynamic feature extraction pipeline for eICU.

Primary Dataset:
    eICU Collaborative Research Database (v2.0)
    https://eicu-crd.mit.edu/

Key Features:
1. Multi-Modal Dynamic Time Series:
   - Core Bedside Vital Signs: Heart Rate, Systolic BP, Diastolic BP,
     Respiratory Rate, SpO2, Temperature.
   - Laboratory Blood Tests dynamically selected on training split using the
     official inclusion rule (>= 25% stay prevalence and > 3 obs/stay).
   - Forward-filling and exponential decay indicators: x''_t = 0.75^j.
2. Multi-Modal Static Admission Features:
   - Continuous: Age, Admission Hour, Height, Weight, GCS (Eyes, Motor, Verbal)
     with binary missingness flags.
   - Categorical (One-Hot Encoded with Unknown Slot): Unit Type, Admission Source,
     Unit Stay Type, Ethnicity.
3. Target Alignment:
   - Hourly remaining LoS in days:
     max((unitdischargeoffset - 60*(t+1))/1440, 1/48).
   - Multitask in-hospital mortality label:
     hospitaldischargestatus == "Expired".
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyhealth.datasets import eICUDataset

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from pyhealth.tasks.base_task import BaseTask

# Global in-memory storage for large vitals caches decoupled from task __dict__
_VITALS_MEM_CACHE: dict[str, dict[str, list[tuple[int, str, float]]]] = {}
_BASELINE_MEM_CACHE: dict[str, dict[str, dict[str, float]]] = {}


def _compute_cache_fingerprint(data: Any) -> str:
    """Computes a deterministic lightweight SHA-256 fingerprint for caching."""
    if data is None:
        return "none"
    if isinstance(data, (str, Path)):
        p = str(data)
        if os.path.exists(p):
            stat = os.stat(p)
            token = f"{p}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        return hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
    if not isinstance(data, dict) or not data:
        return "empty"
    keys = sorted(data.keys())
    sample = [keys[0], keys[len(keys) // 2], keys[-1]] if len(keys) > 3 else keys
    sample_preview = [str(data[k])[:80] for k in sample]
    token = f"len={len(data)};sample_keys={sample};sample_val={sample_preview}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


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
    dataset: eICUDataset | Any,
    train_pids: set[str],
    min_prevalence: float = 0.25,
    min_avg_obs: float = 3.0,
    max_labs: int = 45,
) -> tuple[list[str], bool]:
    """Dynamically selects laboratory tests conforming to the paper's inclusion rule.

    Evaluates ICU stays strictly on the training set to prevent data leakage.
    Counts measurements strictly within each ICU stay window
    [0, min(unitdischargeoffset, 20160 minutes)).
    Cohort filters (adults > 18, stays >= 5 hours) match the task definition.

    Args:
        dataset: eICUDataset instance.
        train_pids: Patient IDs belonging to the training split.
        min_prevalence: Minimum stay prevalence threshold (default: 0.25).
        min_avg_obs: Minimum average observations per stay (default: 3.0).
        max_labs: Cap on number of qualifying labs (defaults to 45).

    Returns:
        tuple[list[str], bool]: Selected lab names and boolean indicator
            signifying whether statistical thresholds were met.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import (
        ...     select_qualifying_labs,
        ... )
        >>> labs, ok = select_qualifying_labs(None, {"stay_1"})  # doctest: +SKIP
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
                # Count measurements strictly within stay window
                if 0.0 <= loff < stay_end:
                    lab_name_raw = getattr(lab, "labname", None)
                    if lab_name_raw is not None:
                        lab_name = str(lab_name_raw).strip()
                        obs_per_lab[lab_name] = obs_per_lab.get(lab_name, 0) + 1
                        stay_labs.add(lab_name)

            for lab_name in stay_labs:
                stays_per_lab[lab_name] = stays_per_lab.get(lab_name, 0) + 1

    if total_train_stays == 0:
        return DEFAULT_LAB_NAMES, False

    qualifying: list[tuple[float, float, str]] = []
    for lab_name, obs_count in obs_per_lab.items():
        prevalence = stays_per_lab.get(lab_name, 0) / float(total_train_stays)
        avg_obs = obs_count / float(total_train_stays)
        if prevalence >= min_prevalence and avg_obs > min_avg_obs:
            qualifying.append((prevalence, avg_obs, lab_name))

    qualifying.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [lab_name for _, _, lab_name in qualifying[:max_labs]]

    if len(selected) < 5:
        return DEFAULT_LAB_NAMES, False
    return selected, True


def get_valid_eicu_stay_ids(
    dataset: Any,
    min_stay_hours: float = 5.0,
    min_age: float = 18.0,
) -> set[str]:
    """Extracts stay IDs from an eICU dataset matching adult & minimum LoS criteria.

    Args:
        dataset: eICUDataset or dataset with iter_patients().
        min_stay_hours: Minimum ICU discharge offset in hours (default: 5.0).
        min_age: Minimum patient age in years (default: 18.0).

    Returns:
        set[str]: Set of valid patientunitstayid strings.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import get_valid_eicu_stay_ids
        >>> valid_ids = get_valid_eicu_stay_ids(None)  # doctest: +SKIP
    """
    valid_ids: set[str] = set()
    min_offset = min_stay_hours * 60.0
    for patient in dataset.iter_patients():
        for stay in patient.get_events(event_type="patient"):
            sid = str(getattr(stay, "patientunitstayid", ""))
            raw_offset = getattr(stay, "unitdischargeoffset", None)
            if raw_offset is None:
                continue
            try:
                offset = float(raw_offset)
            except (ValueError, TypeError):
                continue
            if offset < min_offset:
                continue
            age_str = str(getattr(stay, "age", "63")).replace(">", "").strip()
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                age = 63.0
            if age <= min_age:
                continue
            valid_ids.add(sid)
    return valid_ids


def load_eicu_vitals_cache(
    eicu_root: str,
    dataset: eICUDataset | None = None,
    chunksize: int = 250_000,
    valid_stay_ids: set[str] | None = None,
    output_path: str | Path | None = None,
) -> tuple[dict[str, list[tuple[int, str, float]]], dict[str, dict[str, float]]]:
    """Loads bedside vital signs and admission-window baseline measurements for eICU.

    Adheres to Rocheteau et al. (adapted from flat_features.sql):
    - Vital signs are sorted chronologically by offset so "last value in hour wins"
      strictly follows temporal order.
    - Continuous vitals are returned as lightweight (hour, vital_key, val) tuples.
    - Preserves legitimate zero/negative measurements.
    - Baseline measurements are restricted strictly to [-24h, +5h] relative to ICU.
    - If multiple baseline measurements exist in window, sorts by proximity to intime.
    - Streams large CSV files in chunks (chunksize) to prevent out-of-memory errors
      on full multi-gigabyte eICU releases (e.g. vitalPeriodic.csv > 15 GB).

    Args:
        eicu_root: Path to eICU directory containing CSV files.
        dataset: Optional eICUDataset instance for stay validation.
        chunksize: Number of rows to process per chunk (default: 250,000).
        valid_stay_ids: Optional set of ICU stay IDs to filter during ingestion.
        output_path: Optional file path to persist parsed vitals cache on disk.

    Returns:
        tuple: (vitals_by_stay, baseline_static_by_stay) dictionaries mapping stay ID
            to list of vital sign tuples and baseline assessment dictionaries.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import load_eicu_vitals_cache
        >>> vitals, baseline = load_eicu_vitals_cache("./data")  # doctest: +SKIP
    """
    stay_filter: set[str] | None = None
    if valid_stay_ids is not None:
        stay_filter = set(valid_stay_ids)

    vitals_raw: dict[str, list[tuple[float, int, str, float]]] = {}
    baseline_best: dict[str, dict[str, tuple[float, float]]] = {}

    # 1. Check for physicalExam.csv in eicu_root
    pe_csv = os.path.join(eicu_root, "physicalExam.csv")
    pe_gz = os.path.join(eicu_root, "physicalExam.csv.gz")
    pe_path = pe_csv if os.path.exists(pe_csv) else (
        pe_gz if os.path.exists(pe_gz) else None
    )

    if pe_path is not None:
        pe_cols_set = {
            "patientunitstayid",
            "physicalexamoffset",
            "physicalexampath",
            "physicalexamtext",
        }
        # Chunked ingestion to handle multi-gigabyte tables without memory exhaustion
        for chunk in pd.read_csv(
            pe_path,
            usecols=lambda c: str(c) in pe_cols_set,
            chunksize=chunksize,
        ):
            chunk = chunk[chunk["physicalexamtext"].notnull()]
            chunk["offset"] = pd.to_numeric(
                chunk["physicalexamoffset"], errors="coerce"
            )
            chunk = chunk[chunk["offset"].notnull()]
            if stay_filter is not None:
                chunk = chunk[
                    chunk["patientunitstayid"].astype(str).isin(stay_filter)
                ]

            # 1a. Bedside vitals (offset >= 0)
            df_vitals = chunk[chunk["offset"] >= 0]
            for row in df_vitals.itertuples(index=False):
                sid = str(row.patientunitstayid)
                path = str(row.physicalexampath)
                text = str(row.physicalexamtext)
                off = float(row.offset)
                hour = int(off // 60)
                for sub, vkey in PHYSICAL_EXAM_VITALS_MAP.items():
                    if sub in path:
                        try:
                            val = float(text)
                            if sid not in vitals_raw:
                                vitals_raw[sid] = []
                            vitals_raw[sid].append((off, hour, vkey, val))
                        except (ValueError, TypeError):
                            pass
                        break

            # 1b. Baseline measurements in window [-24h, +5h] = [-1440m, +300m]
            # Track observation with minimum abs(offset) relative to ICU
            # admission (offset=0)
            df_windowed = chunk[
                (chunk["offset"] >= -1440.0) & (chunk["offset"] <= 300.0)
            ]
            for row in df_windowed.itertuples(index=False):
                sid = str(row.patientunitstayid)
                path = str(row.physicalexampath)
                text = str(row.physicalexamtext)
                abs_off = abs(float(row.offset))
                for sub, gkey in BASELINE_GCS_MAP.items():
                    if sub in path:
                        try:
                            val = float(text)
                            if sid not in baseline_best:
                                baseline_best[sid] = {}
                            if (
                                gkey not in baseline_best[sid]
                                or abs_off < baseline_best[sid][gkey][0]
                            ):
                                baseline_best[sid][gkey] = (abs_off, val)
                        except (ValueError, TypeError):
                            pass
                        break

    # 2. Check for optional vitalPeriodic.csv / vitalPeriodic.csv.gz
    vp_csv = os.path.join(eicu_root, "vitalPeriodic.csv")
    vp_gz = os.path.join(eicu_root, "vitalPeriodic.csv.gz")
    vp_path = vp_csv if os.path.exists(vp_csv) else (
        vp_gz if os.path.exists(vp_gz) else None
    )

    if vp_path is not None:
        try:
            col_map = {
                "heartrate": "heart_rate",
                "systemicsystolic": "systolic_bp",
                "systemicdiastolic": "diastolic_bp",
                "respiration": "respiration_rate",
                "sao2": "spo2",
                "temperature": "temperature",
            }
            # Inspect column headers without reading rows
            header_cols = [
                c.lower() for c in pd.read_csv(vp_path, nrows=0).columns
            ]
            available_cols = [
                c
                for c in ["patientunitstayid", "observationoffset"]
                + list(col_map.keys())
                if c in header_cols
            ]
            if (
                "patientunitstayid" in available_cols
                and "observationoffset" in available_cols
            ):
                available_set = set(available_cols)
                for chunk in pd.read_csv(
                    vp_path,
                    usecols=lambda c: str(c) in available_set,
                    chunksize=chunksize,
                ):
                    chunk.columns = chunk.columns.str.lower()
                    if stay_filter is not None:
                        chunk = chunk[
                            chunk["patientunitstayid"]
                            .astype(str)
                            .isin(stay_filter)
                        ]
                    chunk["observationoffset"] = pd.to_numeric(
                        chunk["observationoffset"], errors="coerce"
                    )
                    chunk = chunk[
                        chunk["observationoffset"].notnull()
                        & (chunk["observationoffset"] >= 0)
                    ]
                    for row in chunk.itertuples(index=False):
                        sid = str(getattr(row, "patientunitstayid", ""))
                        off = float(getattr(row, "observationoffset", 0.0))
                        hour = int(off // 60)
                        if sid not in vitals_raw:
                            vitals_raw[sid] = []
                        for c_name, vkey in col_map.items():
                            if hasattr(row, c_name):
                                c_val = getattr(row, c_name)
                                if pd.notnull(c_val):
                                    try:
                                        vitals_raw[sid].append(
                                            (off, hour, vkey, float(c_val))
                                        )
                                    except (ValueError, TypeError):
                                        pass
        except (OSError, ValueError):
            pass

    # Sort each stay's vitals chronologically by observation offset
    vitals_by_stay: dict[str, list[tuple[int, str, float]]] = {}
    for sid, records in vitals_raw.items():
        records.sort(key=lambda r: r[0])
        vitals_by_stay[sid] = [(r[1], r[2], r[3]) for r in records]

    baseline_static_by_stay: dict[str, dict[str, float]] = {
        sid: {gkey: best[1] for gkey, best in feats.items()}
        for sid, feats in baseline_best.items()
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            pickle.dump((vitals_by_stay, baseline_static_by_stay), f)

    return vitals_by_stay, baseline_static_by_stay


def _percentile(data: list[float], q: float) -> float:
    """Computes the q-th percentile (0-100) using linear interpolation."""
    s = sorted(data)
    n = len(s)
    if n == 1:
        return s[0]
    idx = (q / 100.0) * (n - 1)
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return s[low]
    weight = idx - low
    return s[low] * (1.0 - weight) + s[high] * weight


def compute_empirical_training_stats(
    dataset: eICUDataset | Any,
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
    """Computes empirical training statistics and builds fixed categorical vocabs.

    Strictly isolated to training split and cohort filters (adults > 18, stays >= 5h):
    - Time-series: empirical mean and std with 1st/99th percentile clipping.
    - Continuous static: empirical mean, std, and median (for imputation).
    - Categorical static: deterministic sorted vocabulary for one-hot encoding.

    Args:
        dataset: eICUDataset instance.
        train_pids: Set of patient IDs belonging to training split.
        lab_names: List of laboratory test names to include.
        vitals_by_stay: Optional bedside vitals cache dictionary.
        baseline_static_by_stay: Optional baseline measurements dictionary.

    Returns:
        tuple: (ts_stats, ts_counts, static_stats, cat_vocabs) where:
            ts_stats: Dict mapping time-series item ID to (mean, std).
            ts_counts: Dict mapping time-series item ID to observation count.
            static_stats: Dict mapping continuous static features to
                (mean, std, median).
            cat_vocabs: Dict mapping categorical static features to sorted
                category lists.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import (
        ...     compute_empirical_training_stats,
        ... )
        >>> stats, counts, s_stats, vocabs = compute_empirical_training_stats(
        ...     None, {"p1"}, ["potassium"]
        ... )  # doctest: +SKIP
    """
    ts_vals: dict[str, list[float]] = {name: [] for name in lab_names}
    if vitals_by_stay:
        for vid in CORE_VITALS_MAP:
            if vid not in ts_vals:
                ts_vals[vid] = []

    continuous_static_vals: dict[str, list[float]] = {
        "age": [],
        "height": [],
        "weight": [],
        "eyes": [],
        "motor": [],
        "verbal": [],
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
            cat_sets["unitadmitsource"].add(
                str(getattr(stay, "unitadmitsource", ""))
            )
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
    all_keys = list(lab_names) + (
        [k for k in CORE_VITALS_MAP] if vitals_by_stay else []
    )

    for item_id in all_keys:
        vals = ts_vals.get(item_id, [])
        ts_counts[item_id] = len(vals)
        if len(vals) > 0:
            arr = np.asarray(vals, dtype=np.float64)
            p1 = float(np.percentile(arr, 1.0))
            p99 = float(np.percentile(arr, 99.0))
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
            arr = np.asarray(vals, dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            median = float(np.median(arr))
            static_stats[k] = (mean, std if std > 1e-4 else prior[1], median)
        else:
            static_stats[k] = prior

    # 3. Deterministic sorted categorical vocabularies
    cat_vocabs = {k: sorted(v) for k, v in cat_sets.items()}

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

    Args:
        raw_grid: 2D Tensor of shape (seq_len, num_features) with observations.
        mask: 2D Tensor of shape (seq_len, num_features) with binary indicators (1=obs).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Forward-filled time series tensor and
            exponential decay indicator tensor, each of shape (seq_len, num_features).

    Examples:
        >>> import torch
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import generate_tpc_tensors
        >>> grid = torch.ones((24, 6))
        >>> mask = torch.ones((24, 6))
        >>> ff, decay = generate_tpc_tensors(grid, mask)
        >>> ff.shape
        torch.Size([24, 6])
    """
    seq_len, num_features = raw_grid.shape
    vitals_ff = raw_grid.clone()
    vitals_decay = torch.zeros_like(raw_grid)
    time_since_last = torch.full(
        (num_features,), float("inf"), dtype=torch.float32
    )

    for t in range(seq_len):
        measured = mask[t] == 1.0
        time_since_last[measured] = 0.0
        time_since_last[~measured] += 1.0

        if t > 0:
            unmeasured = ~measured
            vitals_ff[t, unmeasured] = vitals_ff[t - 1, unmeasured]

        finite = ~torch.isinf(time_since_last)
        vitals_decay[t, finite] = torch.pow(0.75, time_since_last[finite])

    return vitals_ff, vitals_decay


class LengthOfStayTPCeICU(BaseTask):
    """ICU Remaining Length of Stay (LoS) Prediction Task for eICU.

    Reproduces the multi-modal dynamic time series and static feature pipeline from:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit'
    (Emma Rocheteau, Pietro Liò, Stephanie Hyland, ACM CHIL 2021).

    Key Cohort & Task Criteria (Section 4 & 5.1.1):
    - Adult ICU patients (> 18 years old at admission).
    - Minimum ICU length of stay of 5 hours (unitdischargeoffset >= 300 minutes).
    - At least one recorded clinical observation during the stay.
    - Capped at first 14 days (336 hours = 20,160 minutes).
    - Target: Remaining LoS in days anchored at the end of each hour t:
      y_t = max((unitdischargeoffset - (t + 1) * 60) / 1440.0, 1/48 days).

    Args:
        lab_names: List of laboratory test names to include in dynamic time series.
        norm_stats: Dictionary mapping feature IDs to (mean, std) normalization tuples.
        vitals_by_stay: Pre-loaded bedside vital signs mapped by stay ID.
        baseline_static_by_stay: Baseline assessments in [-24h, +5h] window.
        static_stats: Continuous static priors (mean, std, median).
        cat_vocabs: Deterministic sorted category lists for one-hot encoding.
        max_stay_hours: Sequence truncation cutoff in hours (default: 336).

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import LengthOfStayTPCeICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["lab", "physicalexam"],
        ... )  # doctest: +SKIP
        >>> task = LengthOfStayTPCeICU()  # doctest: +SKIP
        >>> sample_dataset = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "eICU Remaining Length of Stay Prediction (TPC)"
    input_schema: dict[str, str | type] = {  # noqa: RUF012
        "vitals": "tensor",
        "vitals_decay": "tensor",
        "static": "tensor",
        "mask": "tensor",
    }
    output_schema: dict[str, str | type] = {  # noqa: RUF012
        "los": "tensor",
        "mortality": "tensor",
    }

    def __init__(
        self,
        lab_names: list[str] | None = None,
        norm_stats: dict[str, tuple[float, float]] | None = None,
        vitals_by_stay: dict[str, list[tuple[int, str, float]]]
        | str
        | Path
        | None = None,
        baseline_static_by_stay: dict[str, dict[str, float]]
        | str
        | Path
        | None = None,
        static_stats: dict[str, tuple[float, float, float]] | None = None,
        cat_vocabs: dict[str, list[str]] | None = None,
        max_stay_hours: int = 336,
    ):
        super().__init__()
        self.lab_names = (
            lab_names if lab_names is not None else DEFAULT_LAB_NAMES
        )
        self.norm_stats = norm_stats or {}
        self.max_stay_hours = max_stay_hours

        # Fingerprint and decouple large vitals caches outside __dict__ to prevent
        # BaseDataset.set_task from stringifying multi-gigabyte dicts into JSON.
        self.vitals_cache_path: str | None = None
        self.baseline_cache_path: str | None = None

        if isinstance(vitals_by_stay, (str, Path)):
            self.vitals_cache_path = str(vitals_by_stay)
            self.vitals_hash = _compute_cache_fingerprint(vitals_by_stay)
            self.vitals_count = -1
        else:
            v_dict = vitals_by_stay or {}
            self.vitals_hash = _compute_cache_fingerprint(v_dict)
            self.vitals_count = len(v_dict)
            if self.vitals_hash != "empty":
                _VITALS_MEM_CACHE[self.vitals_hash] = v_dict

        if isinstance(baseline_static_by_stay, (str, Path)):
            self.baseline_cache_path = str(baseline_static_by_stay)
            self.baseline_hash = _compute_cache_fingerprint(
                baseline_static_by_stay
            )
            self.baseline_count = -1
        else:
            b_dict = baseline_static_by_stay or {}
            self.baseline_hash = _compute_cache_fingerprint(b_dict)
            self.baseline_count = len(b_dict)
            if self.baseline_hash != "empty":
                _BASELINE_MEM_CACHE[self.baseline_hash] = b_dict

        # Construct time-series feature mapping
        self.vital_keys = (
            list(CORE_VITALS_MAP.keys()) if self.vitals_by_stay else []
        )
        self.all_feature_ids = self.vital_keys + self.lab_names
        self.feature_map = {
            item_id: idx for idx, item_id in enumerate(self.all_feature_ids)
        }
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
            "unitadmitsource": [
                "Emergency Department",
                "Operating Room",
                "Floor",
                "Direct Admit",
            ],
            "unitstaytype": ["admit", "readmit", "transfer"],
            "ethnicity": [
                "Caucasian",
                "African American",
                "Hispanic",
                "Asian",
            ],
        }

        # 11 continuous/indicator features + one-hot categories (each with unknown slot)
        self.static_dim = 11 + sum(
            len(vocab) + 1 for vocab in self.cat_vocabs.values()
        )

    @property
    def vitals_by_stay(self) -> dict[str, list[tuple[int, str, float]]]:
        """Bedside vitals cache mapping stay ID to vital sign tuples."""
        if self.vitals_hash in _VITALS_MEM_CACHE:
            return _VITALS_MEM_CACHE[self.vitals_hash]
        if self.vitals_cache_path and os.path.exists(self.vitals_cache_path):
            with open(self.vitals_cache_path, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    loaded = loaded[0]
                _VITALS_MEM_CACHE[self.vitals_hash] = loaded
                return loaded
        return {}

    @property
    def baseline_static_by_stay(self) -> dict[str, dict[str, float]]:
        """Baseline static admissions cache mapping stay ID to measurements."""
        if self.baseline_hash in _BASELINE_MEM_CACHE:
            return _BASELINE_MEM_CACHE[self.baseline_hash]
        if self.baseline_cache_path and os.path.exists(self.baseline_cache_path):
            with open(self.baseline_cache_path, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    loaded = loaded[1]
                _BASELINE_MEM_CACHE[self.baseline_hash] = loaded
                return loaded
        return {}

    def __getstate__(self) -> dict[str, Any]:
        """Prepares state for pickling across worker processes."""
        state = self.__dict__.copy()
        if (
            not state.get("vitals_cache_path")
            and self.vitals_hash in _VITALS_MEM_CACHE
        ):
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"pyhealth_vitals_{self.vitals_hash}.pkl",
            )
            if not os.path.exists(tmp_path):
                with open(tmp_path, "wb") as f:
                    pickle.dump(_VITALS_MEM_CACHE[self.vitals_hash], f)
            state["vitals_cache_path"] = tmp_path
        if (
            not state.get("baseline_cache_path")
            and self.baseline_hash in _BASELINE_MEM_CACHE
        ):
            tmp_path = os.path.join(
                tempfile.gettempdir(),
                f"pyhealth_baseline_{self.baseline_hash}.pkl",
            )
            if not os.path.exists(tmp_path):
                with open(tmp_path, "wb") as f:
                    pickle.dump(_BASELINE_MEM_CACHE[self.baseline_hash], f)
            state["baseline_cache_path"] = tmp_path
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restores state upon unpickling."""
        self.__dict__.update(state)

    @staticmethod
    def _one_hot(val: str, vocab: list[str]) -> list[float]:
        """Encodes a categorical string value into a one-hot vector with unknown slot.

        Args:
            val: Categorical value to encode.
            vocab: Deterministic sorted category list.

        Returns:
            list[float]: One-hot vector of length len(vocab) + 1.
        """
        vec = [0.0] * (len(vocab) + 1)
        if val in vocab:
            vec[vocab.index(val)] = 1.0
        else:
            vec[-1] = 1.0  # Unknown slot
        return vec

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Processes a patient record into TPC task samples.

        Extracts dynamic vital signs, laboratory tests, admission baseline
        measurements, and hourly remaining length of stay targets.

        Args:
            patient: PyHealth Patient object with patient and lab events.

        Returns:
            list[dict[str, Any]]: List of sample dictionaries containing 'vitals',
                'vitals_decay', 'static', 'mask', 'los', and 'mortality' tensors.
        """
        samples = []

        patient_stays = patient.get_events(event_type="patient")
        if not patient_stays:
            return []

        patient_labs = patient.get_events(event_type="lab")
        # Pre-filter labs belonging to feature map and sort chronologically
        filtered_labs = [
            lab
            for lab in patient_labs
            if str(getattr(lab, "labname", "")).strip() in self.feature_map
            and getattr(lab, "labresultoffset", None) is not None
        ]
        filtered_labs.sort(
            key=lambda lab: float(getattr(lab, "labresultoffset", 0.0))
        )

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
            # Pre-sorted chronologically: last value in hour overwrites earlier ones
            if self.vitals_by_stay and sid in self.vitals_by_stay:
                for hour, vkey, val in self.vitals_by_stay[sid]:
                    if vkey not in self.feature_map:
                        continue
                    f_idx = self.feature_map[vkey]
                    if 0 <= hour < T:
                        mean, std = self.norm_stats.get(vkey, (0.0, 1.0))
                        raw_grid[hour, f_idx] = (val - mean) / (
                            std if std > 1e-4 else 1.0
                        )
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
                            mean, std = self.norm_stats.get(
                                lab_name, (0.0, 1.0)
                            )
                            raw_grid[hour, f_idx] = (val - mean) / (
                                std if std > 1e-4 else 1.0
                            )
                            mask[hour, f_idx] = 1.0
                        except (ValueError, TypeError):
                            pass

            # Require at least one recorded observation during stay (Section 4.1)
            if mask.sum() == 0:
                continue

            # 3. Forward filling & exponential decay indicators: 0.75^j
            vitals_ff, vitals_decay = generate_tpc_tensors(raw_grid, mask)

            # 4. Hourly Remaining LoS Target anchored at end of hour t:
            los_targets = torch.tensor(
                [
                    max(
                        (discharge_offset - (t + 1) * 60.0) / 1440.0,
                        LOS_FLOOR_DAYS,
                    )
                    for t in range(T)
                ],
                dtype=torch.float32,
            ).unsqueeze(1)

            # 5. In-Hospital Mortality Label
            mort_str = str(getattr(stay, "hospitaldischargestatus", "")).lower()
            mort_val = 1.0 if mort_str == "expired" else 0.0

            # 6. Static Features
            base_s = self.baseline_static_by_stay.get(sid, {})

            # Continuous standardized with train split parameters
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
            unittype_oh = self._one_hot(
                str(getattr(stay, "unittype", "")), self.cat_vocabs["unittype"]
            )
            unitadmitsource_oh = self._one_hot(
                str(getattr(stay, "unitadmitsource", "")),
                self.cat_vocabs["unitadmitsource"],
            )
            unitstaytype_oh = self._one_hot(
                str(getattr(stay, "unitstaytype", "")),
                self.cat_vocabs["unitstaytype"],
            )
            ethnicity_oh = self._one_hot(
                str(getattr(stay, "ethnicity", "")),
                self.cat_vocabs["ethnicity"],
            )

            static_list = (
                [
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
                ]
                + unittype_oh
                + unitadmitsource_oh
                + unitstaytype_oh
                + ethnicity_oh
            )

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


# Alias for convenience matching PyHealth naming convention
LengthOfStayPredictionTPCeICU = LengthOfStayTPCeICU


class MapDataset(torch.utils.data.Dataset):
    """Map-style dataset enabling custom BatchSampler and length bucketing.

    Supports both in-memory sample lists and lazy subset indexing into an underlying
    streaming dataset (e.g., SampleDataset), preventing OOM on multi-gigabyte cohorts.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import MapDataset
        >>> ds = MapDataset([{"patient_id": "p1"}, {"patient_id": "p2"}])
        >>> len(ds)
        2
        >>> ds[0]["patient_id"]
        'p1'
        >>> ds_indexed = MapDataset(
        ...     [{"patient_id": "p1"}, {"patient_id": "p2"}], indices=[1]
        ... )
        >>> len(ds_indexed)
        1
        >>> ds_indexed[0]["patient_id"]
        'p2'
    """

    def __init__(
        self,
        samples: Any,
        indices: Sequence[int] | None = None,
    ):
        """Initializes MapDataset.

        Args:
            samples: List of sample dictionaries or an underlying Dataset.
            indices: Optional sequence of indices for lazy subset indexing.
        """
        self.samples = samples
        self.indices = list(indices) if indices is not None else None

    def __len__(self) -> int:
        """Returns total number of samples."""
        if self.indices is not None:
            return len(self.indices)
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """Retrieves a sample dictionary by index.

        Args:
            index: Index of sample.

        Returns:
            dict: Sample dictionary.
        """
        if self.indices is not None:
            return self.samples[self.indices[index]]
        return self.samples[index]


class BucketBatchSampler(Sampler[list[int]]):
    """Batches sequences of similar length together to minimize zero padding.

    Partitions stays into length-sorted buckets and shuffles batch execution order
    every epoch. Allows resetting epoch counter to ensure strictly unconfounded
    ablation comparisons across model configurations.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_eicu import BucketBatchSampler
        >>> sampler = BucketBatchSampler(
        ...     lengths=[10, 15, 20, 25], batch_size=2, shuffle=False
        ... )
        >>> list(sampler)
        [[0, 1], [2, 3]]
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initializes BucketBatchSampler.

        Args:
            lengths: Sequence lengths of all samples.
            batch_size: Mini-batch size.
            shuffle: Whether to shuffle batches each epoch. Default: True.
            seed: Random seed for deterministic batch permutation. Default: 42.
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def reset(self) -> None:
        """Resets epoch counter so ablation configurations see identical batches."""
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        """Yields mini-batch index lists grouped by sequence length."""
        indices = [int(idx) for idx in np.argsort(self.lengths)]
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = [int(p) for p in rng.permutation(len(batches))]
            batches = [batches[i] for i in perm]
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        """Returns total number of mini-batches."""
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
