"""Run an exploratory MIMIC-IV TPC remaining ICU length-of-stay experiment.

Example (MIMIC-IV Clinical Database Demo v2.2; run from the repository root)::

    python examples/mimic4_los_tpc.py --data-dir PATH_TO_DEMO \
        --cache-dir SHORT_LOCAL_CACHE_PATH --epochs 1

The automatic itemid selection is a pipeline check, not the paper's manually
curated 101-feature MIMIC-IV v0.4 reproduction. For a fixed feature set, pass
``--feature-map`` with JSON keys ``lab`` and ``chart``, each mapping itemid to
a unique feature name. Train/validation/test are split by subject_id before
feature selection, normalization, and model fitting. No test data is used for
model selection. The cache directory can contain restricted data: keep it
outside Git and in a location allowed by the dataset's data-use agreement.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import MIMIC4EHRDataset, collate_fn_dict_with_padding
from pyhealth.models import TPC
from pyhealth.tasks.length_of_stay_tpc import (
    BucketBatchSampler,
    FastReservoirSampler,
    MapDataset,
    compute_los_metrics,
)
from pyhealth.tasks.length_of_stay_tpc_mimic4 import (
    ChartEventSQLite,
    LengthOfStayTPCMIMIC4,
    build_chart_event_cache,
)


def _file(root: Path, table: str) -> Path:
    folder = "icu" if table in {"icustays", "chartevents", "d_items"} else "hosp"
    for suffix in (".csv.gz", ".csv"):
        path = root / folder / f"{table}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {folder}/{table}.csv[.gz] in {root}")


def _rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def _date(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value or "")
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def eligible_stays(root: Path) -> dict[str, dict[str, Any]]:
    """Adult ICU stays lasting at least five hours, keyed by stay_id."""
    people = {r["subject_id"]: r for r in _rows(_file(root, "patients"))}
    admissions = {r["hadm_id"] for r in _rows(_file(root, "admissions"))}
    stays = {}
    for row in _rows(_file(root, "icustays")):
        person = people.get(row["subject_id"])
        start, end = _date(row["intime"]), _date(row["outtime"])
        if person is None or start is None or end is None or end <= start:
            continue
        anchor_age = _float(person.get("anchor_age"))
        anchor_year = _float(person.get("anchor_year"))
        if anchor_age is None or anchor_year is None:
            continue
        age = anchor_age + start.year - int(anchor_year)
        if age <= 18 or (end - start).total_seconds() < 5 * 3600:
            continue
        if row["hadm_id"] not in admissions:
            continue
        stays[row["stay_id"]] = {
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "start": start,
            "end": end,
        }
    return stays


def split_patients(stays: dict[str, dict[str, Any]], seed: int):
    ids = sorted({row["subject_id"] for row in stays.values()})
    random.Random(seed).shuffle(ids)
    n = len(ids)
    train, val, test = (
        set(ids[: int(n * 0.70)]),
        set(ids[int(n * 0.70) : int(n * 0.85)]),
        set(ids[int(n * 0.85) :]),
    )
    if not train or not val or not test:
        raise ValueError("At least three eligible patients are needed")
    return train, val, test


def select_itemids(
    root: Path,
    train_stays: dict[str, dict[str, Any]],
    max_chart: int,
    max_lab: int,
    min_prevalence: float,
    min_avg_obs: float,
) -> tuple[dict[str, str], dict[str, str]]:
    """Select numerical items from training ICU stay windows only."""
    by_hadm: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for sid, stay in train_stays.items():
        by_hadm[stay["hadm_id"]].append((sid, stay))
    counts = {"chart": Counter(), "lab": Counter()}
    seen = {"chart": defaultdict(set), "lab": defaultdict(set)}
    for row in _rows(_file(root, "chartevents")):
        sid = row["stay_id"]
        stay = train_stays.get(sid)
        time = _date(row.get("charttime"))
        if (
            stay
            and time
            and stay["start"] <= time < stay["end"]
            and _float(row.get("valuenum")) is not None
        ):
            item = row["itemid"]
            counts["chart"][item] += 1
            seen["chart"][item].add(sid)
    for row in _rows(_file(root, "labevents")):
        if _float(row.get("valuenum")) is None:
            continue
        charttime, storetime = _date(row.get("charttime")), _date(row.get("storetime"))
        if charttime is None:
            continue
        available = max(charttime, storetime) if storetime else charttime
        for sid, stay in by_hadm.get(row.get("hadm_id", ""), []):
            if stay["start"] <= available < stay["end"]:
                item = row["itemid"]
                counts["lab"][item] += 1
                seen["lab"][item].add(sid)

    def choose(kind: str, limit: int) -> dict[str, str]:
        candidates = [
            item
            for item, count in counts[kind].items()
            if len(seen[kind][item]) / len(train_stays) >= min_prevalence
            and count / len(seen[kind][item]) >= min_avg_obs
        ]
        candidates.sort(
            key=lambda item: (-len(seen[kind][item]), -counts[kind][item], item)
        )
        return {item: f"{kind}_{item}" for item in candidates[:limit]}

    return choose("lab", max_lab), choose("chart", max_chart)


def fit_stats(dataset, train_ids: set[str], task: LengthOfStayTPCMIMIC4, seed: int):
    """Fit numeric statistics and category vocabularies on training patients."""
    samplers = {
        feature: FastReservoirSampler(capacity=100_000, seed=seed + i)
        for i, feature in enumerate(task.time_series_features)
    }
    static_keys = ("age", "height", "weight", "eyes", "motor", "verbal")
    static_vals: dict[str, list[float]] = {key: [] for key in static_keys}
    cat_vals: dict[str, set[str]] = {
        "unittype": set(),
        "unitadmitsource": set(),
        "ethnicity": set(),
    }
    for pid in sorted(train_ids):
        for stay in task.extract_patient_events(dataset.get_patient(pid)):
            for _, feature, value in stay.hourly_vitals:
                samplers[feature].add(value)
            for key in static_keys:
                value = stay.static_continuous.get(key)
                if value is not None and math.isfinite(value):
                    static_vals[key].append(value)
            for key, val_set in cat_vals.items():
                value = stay.static_categorical.get(key, "")
                if value:
                    val_set.add(value)
    norm = {}
    for feature, sampler in samplers.items():
        values = np.asarray(sampler.samples, dtype=float)
        if values.size == 0:
            raise ValueError(f"No training observations for selected feature {feature}")
        mean = float(values.mean())
        std = max(float(values.std()), 1e-4)
        p1, p99 = np.percentile(values, [1, 99])
        norm[feature] = (mean, std, float(p1), float(p99))
    static = {}
    for key, values in static_vals.items():
        array = np.asarray(values, dtype=float)
        if array.size:
            static[key] = (
                float(array.mean()),
                max(float(array.std()), 1e-4),
                float(np.median(array)),
            )
        else:
            static[key] = (0.0, 1.0, 0.0)
    return norm, static, {key: sorted(values) for key, values in cat_vals.items()}


def make_loader(samples, indices: list[int], batch_size: int, seed: int, shuffle: bool):
    lengths = [samples[i]["vitals"].shape[0] for i in indices]
    sampler = BucketBatchSampler(
        lengths, batch_size=batch_size, shuffle=shuffle, seed=seed
    )
    return DataLoader(
        MapDataset(samples, indices),
        batch_sampler=sampler,
        collate_fn=collate_fn_dict_with_padding,
    )


def evaluate(model: TPC, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    truth, pred, masks = [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }
            out = model(**batch)
            for i in range(out["y_prob"].shape[0]):
                truth.append(batch["los"][i].cpu())
                pred.append(out["y_prob"][i].cpu())
                masks.append(batch["mask"][i].cpu())
    if not truth:
        raise ValueError("Evaluation split contains no samples")
    return compute_los_metrics(truth, pred, mask=masks, min_eval_hours=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-map", type=Path, help="JSON mapping with lab and chart itemids"
    )
    parser.add_argument("--max-chart-features", type=int, default=12)
    parser.add_argument("--max-lab-features", type=int, default=6)
    parser.add_argument("--min-prevalence", type=float, default=0.10)
    parser.add_argument("--min-avg-obs", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00226)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--temp-dim", type=int, default=4)
    parser.add_argument("--point-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    root, cache = args.data_dir.resolve(), args.cache_dir.resolve()
    for table in ("patients", "admissions", "icustays", "labevents", "chartevents"):
        _file(root, table)
    _file(root, "d_labitems")
    _file(root, "d_items")
    cache.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    stays = eligible_stays(root)
    train, val, test = split_patients(stays, args.seed)
    train_stays = {
        sid: stay for sid, stay in stays.items() if stay["subject_id"] in train
    }
    print(
        f"Eligible ICU stays: {len(stays)} | patients train/val/test: {len(train)}/{len(val)}/{len(test)}"
    )
    if args.feature_map:
        maps = json.loads(args.feature_map.read_text(encoding="utf-8"))
        lab_map = {str(k): str(v) for k, v in maps["lab"].items()}
        chart_map = {str(k): str(v) for k, v in maps["chart"].items()}
    else:
        lab_map, chart_map = select_itemids(
            root,
            train_stays,
            args.max_chart_features,
            args.max_lab_features,
            args.min_prevalence,
            args.min_avg_obs,
        )
        print(
            "Exploratory training-only item selection; not the paper's fixed 101-feature set."
        )
    if not lab_map and not chart_map:
        raise ValueError("No numeric features selected; adjust feature thresholds")
    print(f"Selected {len(chart_map)} chart and {len(lab_map)} lab itemids")
    cache_tag = hashlib.sha256(
        json.dumps(
            [str(root), sorted(chart_map), sorted(stays)], sort_keys=True
        ).encode()
    ).hexdigest()[:12]
    chart_cache = cache / f"tpc_chart_{cache_tag}.sqlite"
    if chart_map and not chart_cache.exists():
        count = build_chart_event_cache(
            _file(root, "chartevents"),
            chart_cache,
            selected_itemids=list(chart_map),
            selected_stay_ids=set(stays),
        )
        print(f"Cached {count} selected chart observations")
    chart_by_stay = ChartEventSQLite(chart_cache) if chart_map else {}
    dataset = MIMIC4EHRDataset(
        root=str(root), tables=["labevents"], num_workers=1, cache_dir=str(cache)
    )
    names = dict.fromkeys([*lab_map.values(), *chart_map.values()])
    placeholder_norm = {name: (0.0, 1.0) for name in names}
    placeholder_static = {
        key: (0.0, 1.0, 0.0)
        for key in ("age", "height", "weight", "eyes", "motor", "verbal")
    }
    provisional = LengthOfStayTPCMIMIC4(
        lab_item_to_feature=lab_map,
        chart_item_to_feature=chart_map,
        norm_stats=placeholder_norm,
        static_stats=placeholder_static,
        chart_events_by_stay=chart_by_stay,
    )
    norm, static, vocabs = fit_stats(dataset, train, provisional, args.seed)
    task = LengthOfStayTPCMIMIC4(
        lab_item_to_feature=lab_map,
        chart_item_to_feature=chart_map,
        norm_stats=norm,
        static_stats=static,
        cat_vocabs=vocabs,
        chart_events_by_stay=chart_by_stay,
    )
    samples = dataset.set_task(task, num_workers=1)
    p2idx = samples.patient_to_index
    splits = [
        list(chain.from_iterable(p2idx.get(pid, []) for pid in sorted(pids)))
        for pids in (train, val, test)
    ]
    print(f"Samples train/val/test: {[len(x) for x in splits]}")
    if any(not part for part in splits):
        raise ValueError(
            "At least one split has no usable stay; use more data or adjust feature selection"
        )
    loaders = [
        make_loader(samples, idx, args.batch_size, args.seed, j == 0)
        for j, idx in enumerate(splits)
    ]
    device = torch.device(args.device)
    model = TPC(
        dataset=None,
        continuous_dim=task.continuous_dim,
        static_dim=task.static_dim,
        time_dim=task.time_dim,
        n_layers=args.n_layers,
        temp_dim=args.temp_dim,
        point_dim=args.point_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=4,
        dropout=0.45,
        temp_dropout=0.05,
        batchnorm="mybatchnorm",
        loss_fn="msle",
        min_time=4,
        seq_to_seq=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state, best_msle = None, float("inf")
    for epoch in range(args.epochs):
        model.train()
        total, n_batches = 0.0, 0
        for raw in loaders[0]:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
        val_metrics = evaluate(model, loaders[1], device)
        print(
            f"Epoch {epoch + 1}: train loss {total / max(n_batches, 1):.4f}; val MSLE {val_metrics['msle']:.4f}"
        )
        if val_metrics["msle"] < best_msle:
            best_msle = val_metrics["msle"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    print("Held-out test:", json.dumps(evaluate(model, loaders[2], device), indent=2))


if __name__ == "__main__":
    main()
