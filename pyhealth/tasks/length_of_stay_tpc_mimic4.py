"""MIMIC-IV ICU remaining length-of-stay adapter for the shared TPC task.

This adapter accepts PyHealth ``MIMIC4EHRDataset`` patient events for
``patients``, ``admissions``, ``icustays`` and ``labevents``. The current
MIMIC-IV dataset configuration does not expose ``chartevents``; chart events
must be supplied as a per-stay mapping produced by a separate, chunked reader.

Feature item IDs, training-only normalization statistics, and categorical
vocabularies are explicit inputs. This keeps MIMIC-IV feature selection and
statistics separate from the eICU defaults and prevents fitting on test data.
"""

from __future__ import annotations

import csv
import gzip
import math
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from pyhealth.tasks.length_of_stay_tpc import (
    LOS_FLOOR_DAYS,
    BaseLengthOfStayTPC,
    RawStayRecord,
)


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _events(patient: Any, event_type: str) -> list[Any]:
    try:
        return list(patient.get_events(event_type=event_type))
    except (KeyError, ValueError):
        return []


class ChartEventSQLite(Mapping[str, list[dict[str, Any]]]):
    """Disk-backed chart events, queried one ICU stay at a time.

    Construct with a cache created by :func:`build_chart_event_cache`.
    Opening a fresh read-only connection per lookup is safe with PyHealth's
    worker processes and avoids retaining the full chart table in RAM.

    Examples:
        >>> import tempfile, sqlite3
        >>> from pathlib import Path
        >>> from pyhealth.tasks.length_of_stay_tpc_mimic4 import ChartEventSQLite
        >>> tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        >>> with sqlite3.connect(tmp.name) as conn:
        ...     _ = conn.execute("CREATE TABLE chart_events (stay_id TEXT, charttime TEXT, itemid TEXT, valuenum REAL)")
        ...     _ = conn.execute("INSERT INTO chart_events VALUES ('s1', '2020-01-01 00:00:00', '220045', 80.0)")
        >>> cache = ChartEventSQLite(tmp.name)
        >>> len(cache)
        1
        >>> cache['s1'][0]['valuenum']
        80.0
        >>> Path(tmp.name).unlink()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)

    def __repr__(self) -> str:
        stat = self.path.stat()
        return (
            f"ChartEventSQLite(path={str(self.path)!r}, "
            f"size={stat.st_size}, modified_ns={stat.st_mtime_ns})"
        )

    def __getitem__(self, stay_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT charttime, itemid, valuenum FROM chart_events "
                "WHERE stay_id = ? ORDER BY charttime",
                (str(stay_id),),
            ).fetchall()
        if not rows:
            raise KeyError(stay_id)
        return [
            {"charttime": time, "itemid": itemid, "valuenum": value}
            for time, itemid, value in rows
        ]

    def __iter__(self):
        with closing(self._connect()) as connection:
            ids = connection.execute(
                "SELECT DISTINCT stay_id FROM chart_events"
            ).fetchall()
        return iter(row[0] for row in ids)

    def __len__(self) -> int:
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT COUNT(DISTINCT stay_id) FROM chart_events"
            ).fetchone()[0]


def build_chart_event_cache(
    chart_csv: str | Path,
    cache_path: str | Path,
    *,
    selected_itemids: Sequence[str],
    selected_stay_ids: set[str] | None = None,
    batch_size: int = 10_000,
) -> int:
    """Stream MIMIC-IV ``chartevents.csv[.gz]`` into a filtered SQLite cache.

    Build after deciding the itemid mapping and ICU cohort. The output can
    contain restricted patient data: keep it outside Git and only in an
    approved local location. Existing cache files are never overwritten.
    Returns the number of selected rows written. This is data ingestion, not
    the paper's feature-selection or train-only statistics step.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from pyhealth.tasks.length_of_stay_tpc_mimic4 import build_chart_event_cache
        >>> csv_f = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        >>> _ = csv_f.write("stay_id,charttime,itemid,valuenum\\ns1,2020-01-01 00:00:00,220045,80.0\\n")
        >>> csv_f.close()
        >>> tmp_dir = tempfile.mkdtemp()
        >>> out_path = Path(tmp_dir) / "cache.sqlite"
        >>> n = build_chart_event_cache(csv_f.name, out_path, selected_itemids=["220045"])
        >>> n
        1
        >>> Path(csv_f.name).unlink()
        >>> out_path.unlink()
        >>> Path(tmp_dir).rmdir()
    """
    source, target = Path(chart_csv), Path(cache_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    itemids = {str(value) for value in selected_itemids}
    if not itemids:
        raise ValueError("selected_itemids must not be empty")
    stay_ids = (
        {str(value) for value in selected_stay_ids}
        if selected_stay_ids is not None
        else None
    )
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    inserted = 0
    batch: list[tuple[str, str, str, float]] = []
    with closing(sqlite3.connect(target)) as connection:
        connection.execute(
            "CREATE TABLE chart_events ("
            "stay_id TEXT NOT NULL, charttime TEXT NOT NULL, "
            "itemid TEXT NOT NULL, valuenum REAL NOT NULL)"
        )
        with opener(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"stay_id", "charttime", "itemid", "valuenum"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"chartevents is missing columns: {required - set(reader.fieldnames or [])}"
                )
            for row in reader:
                sid = row["stay_id"]
                if row["itemid"] not in itemids or (
                    stay_ids is not None and sid not in stay_ids
                ):
                    continue
                if _timestamp(row["charttime"]) is None:
                    continue
                value = _number(row["valuenum"])
                if value is None:
                    continue
                batch.append((sid, row["charttime"], row["itemid"], value))
                if len(batch) >= batch_size:
                    connection.executemany(
                        "INSERT INTO chart_events VALUES (?, ?, ?, ?)", batch
                    )
                    inserted += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO chart_events VALUES (?, ?, ?, ?)", batch
                )
                inserted += len(batch)
        connection.execute(
            "CREATE INDEX chart_events_stay_time ON chart_events(stay_id, charttime)"
        )
        connection.commit()
    return inserted


class LengthOfStayTPCMIMIC4(BaseLengthOfStayTPC):
    """Convert MIMIC-IV ICU stays into the existing TPC tensor schema.

    ``lab_item_to_feature`` and ``chart_item_to_feature`` map MIMIC-IV itemid
    strings to stable feature names. A chart mapping is needed for the full
    paper-style feature set. ``chart_events_by_stay`` maps a stay_id to rows
    with ``charttime``, ``itemid`` and ``valuenum`` (objects or dictionaries).
    Only observations at or after ICU ``intime`` and before ``outtime`` enter
    the dynamic series. The caller must build these mappings and statistics
    using the training patients only.

    This adapter can read a filtered on-disk chart cache. It does not select
    the paper's 101 time-series features or fit normalization statistics.

    Examples:
        >>> from pyhealth.tasks.length_of_stay_tpc_mimic4 import LengthOfStayTPCMIMIC4
        >>> task = LengthOfStayTPCMIMIC4(
        ...     lab_item_to_feature={"50983": "sodium"},
        ...     chart_item_to_feature={"220045": "heart_rate"},
        ...     norm_stats={"sodium": (140.0, 5.0), "heart_rate": (75.0, 15.0)},
        ...     static_stats={
        ...         "age": (60.0, 15.0, 60.0), "height": (170.0, 10.0, 170.0), "weight": (80.0, 20.0, 80.0),
        ...         "eyes": (4.0, 1.0, 4.0), "motor": (6.0, 1.0, 6.0), "verbal": (5.0, 1.0, 5.0),
        ...     },
        ... )
        >>> task.continuous_dim
        2
    """

    task_name = "MIMIC-IV Remaining ICU Length of Stay Prediction (TPC)"

    def __init__(
        self,
        *,
        lab_item_to_feature: Mapping[str, str],
        chart_item_to_feature: Mapping[str, str],
        norm_stats: Mapping[str, tuple[float, ...]],
        chart_events_by_stay: Mapping[str, Sequence[Any]] | None = None,
        baseline_by_stay: Mapping[str, Mapping[str, float | None]] | None = None,
        static_stats: Mapping[str, tuple[float, float, float]],
        cat_vocabs: Mapping[str, list[str]] | None = None,
        max_stay_hours: int = 336,
        min_stay_hours: float = 5.0,
        min_age: float = 18.0,
        floor_days: float = LOS_FLOOR_DAYS,
        decay_rate: float = 0.75,
        include_mortality: bool = True,
        include_time_features: bool = True,
    ) -> None:
        self.lab_item_to_feature = {str(k): v for k, v in lab_item_to_feature.items()}
        self.chart_item_to_feature = {
            str(k): v for k, v in chart_item_to_feature.items()
        }
        self.chart_events_by_stay = (
            chart_events_by_stay if chart_events_by_stay is not None else {}
        )
        self.baseline_by_stay = baseline_by_stay or {}
        features = list(
            dict.fromkeys(
                [
                    *self.lab_item_to_feature.values(),
                    *self.chart_item_to_feature.values(),
                ]
            )
        )
        if not features:
            raise ValueError("At least one MIMIC-IV lab or chart feature is required")
        missing = set(features) - set(norm_stats)
        if missing:
            raise ValueError(f"Missing training-set norm_stats for: {sorted(missing)}")
        static_keys = {"age", "height", "weight", "eyes", "motor", "verbal"}
        missing_static = static_keys - set(static_stats)
        if missing_static:
            raise ValueError(
                f"Missing training-set static_stats for: {sorted(missing_static)}"
            )
        # Nonempty mapping prevents the base task from using eICU vocabularies.
        mimic_vocabs = (
            dict(cat_vocabs)
            if cat_vocabs is not None
            else {"unittype": [], "unitadmitsource": [], "ethnicity": []}
        )
        if not mimic_vocabs:
            raise ValueError("cat_vocabs must not be empty; use MIMIC-IV keys")
        super().__init__(
            time_series_features=features,
            norm_stats=norm_stats,
            static_stats=static_stats,
            cat_vocabs=mimic_vocabs,
            max_stay_hours=max_stay_hours,
            min_stay_hours=min_stay_hours,
            min_age=min_age,
            floor_days=floor_days,
            decay_rate=decay_rate,
            include_mortality=include_mortality,
            include_time_features=include_time_features,
        )

    @staticmethod
    def _field(event: Any, name: str, table: str | None = None) -> Any:
        """Read a PyHealth event attribute or a raw preprocessing row."""
        if isinstance(event, Mapping):
            if name in event:
                return event[name]
            if table is not None:
                return event.get(f"{table}/{name}")
            return None
        return getattr(event, name, None)

    def _append_observations(
        self,
        result: list[tuple[datetime, int, str, float]],
        events: Sequence[Any],
        item_to_feature: Mapping[str, str],
        table: str,
        start: datetime,
        end: datetime,
    ) -> None:
        for event in events:
            itemid = self._field(event, "itemid", table)
            feature = item_to_feature.get(str(itemid))
            if feature is None:
                continue
            charttime = _timestamp(
                self._field(event, "charttime", table)
                or self._field(event, "timestamp")
            )
            # A lab value cannot be used before it was stored/released.
            if table == "labevents":
                storetime = _timestamp(self._field(event, "storetime", table))
                if storetime is not None and charttime is not None:
                    charttime = max(charttime, storetime)
            value = _number(self._field(event, "valuenum", table))
            if charttime is None or value is None or not (start <= charttime < end):
                continue
            hour = int((charttime - start).total_seconds() // 3600)
            if hour < self.max_stay_hours:
                result.append((charttime, hour, feature, value))

    def extract_patient_events(self, patient: Any) -> list[RawStayRecord]:
        demographics = _events(patient, "patients")
        if not demographics:
            return []
        person = demographics[0]
        admissions = {
            str(self._field(event, "hadm_id", "admissions")): event
            for event in _events(patient, "admissions")
        }
        labs_by_hadm: dict[str, list[Any]] = {}
        for event in _events(patient, "labevents"):
            hadm_id = self._field(event, "hadm_id", "labevents")
            if hadm_id is not None:
                labs_by_hadm.setdefault(str(hadm_id), []).append(event)

        records: list[RawStayRecord] = []
        for stay in _events(patient, "icustays"):
            sid = self._field(stay, "stay_id", "icustays")
            hadm = self._field(stay, "hadm_id", "icustays")
            start = _timestamp(
                self._field(stay, "intime", "icustays")
                or self._field(stay, "timestamp")
            )
            end = _timestamp(self._field(stay, "outtime", "icustays"))
            if (
                sid is None
                or hadm is None
                or start is None
                or end is None
                or end <= start
            ):
                continue
            duration_minutes = (end - start).total_seconds() / 60.0
            if duration_minutes < self.min_stay_offset:
                continue
            anchor_age = _number(self._field(person, "anchor_age", "patients"))
            anchor_year = _number(self._field(person, "anchor_year", "patients"))
            age = (
                anchor_age + start.year - int(anchor_year)
                if anchor_age is not None and anchor_year is not None
                else None
            )
            if age is None or age <= self.min_age:
                continue
            sid_str, hadm_str = str(sid), str(hadm)
            admission = admissions.get(hadm_str)
            if admission is None:
                continue
            timed_observations: list[tuple[datetime, int, str, float]] = []
            self._append_observations(
                timed_observations,
                labs_by_hadm.get(hadm_str, []),
                self.lab_item_to_feature,
                "labevents",
                start,
                end,
            )
            self._append_observations(
                timed_observations,
                self.chart_events_by_stay.get(sid_str, []),
                self.chart_item_to_feature,
                "chartevents",
                start,
                end,
            )
            timed_observations.sort(key=lambda row: row[0])
            observations = [
                (hour, feature, value) for _, hour, feature, value in timed_observations
            ]
            baseline = self.baseline_by_stay.get(sid_str, {})
            gender = str(self._field(person, "gender", "patients") or "").upper()
            race = self._field(admission, "race", "admissions")
            expired = self._field(admission, "hospital_expire_flag", "admissions")
            records.append(
                RawStayRecord(
                    stay_id=sid_str,
                    patient_id=str(patient.patient_id),
                    stay_hours=int(duration_minutes // 60),
                    discharge_offset_minutes=duration_minutes,
                    admission_hour=float(start.hour),
                    hourly_vitals=observations,
                    static_continuous={
                        "age": age,
                        "gender": 1.0 if gender == "M" else 0.0,
                        "height": baseline.get("height"),
                        "weight": baseline.get("weight"),
                        "eyes": baseline.get("eyes"),
                        "motor": baseline.get("motor"),
                        "verbal": baseline.get("verbal"),
                    },
                    static_categorical={
                        "unittype": str(
                            self._field(stay, "first_careunit", "icustays") or ""
                        ),
                        "unitadmitsource": str(
                            self._field(admission, "admission_location", "admissions")
                            or ""
                        ),
                        "ethnicity": str(race or ""),
                    },
                    is_expired=str(expired).strip() == "1",
                )
            )
        return records
