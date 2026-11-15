import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/beijing.npz"
URL = "https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"


def configuration():
    return json.loads((ROOT / "models/config.json").read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def split_indices(times):
    times = pd.DatetimeIndex(times)
    boundaries = {
        "train": ("2013-03-01", "2016-01-01"),
        "validation": ("2016-01-01", "2016-07-01"),
        "test": ("2016-07-01", "2017-03-01"),
    }
    return {
        name: np.flatnonzero((times >= start) & (times < end))
        for name, (start, end) in boundaries.items()
    }


def fit_stats(values, times):
    train = values[split_indices(times)["train"]]
    if not len(train) or np.any(np.isfinite(train).sum(0) == 0):
        raise ValueError("Every station needs observed training values")
    return {
        "mean": np.nanmean(train, axis=0).tolist(),
        "std": np.maximum(np.nanstd(train, axis=0), 1).tolist(),
        "threshold": float(np.nanquantile(train, 0.9)),
    }


def calendar_features(times):
    times = pd.DatetimeIndex(times)
    hour = 2 * np.pi * times.hour.to_numpy() / 24
    month = 2 * np.pi * (times.month.to_numpy() - 1) / 12
    return np.stack(
        [np.sin(hour), np.cos(hour), np.sin(month), np.cos(month)], 1
    ).astype("float32")


def archive_csvs(blob, depth=0):
    if depth > 5:
        raise ValueError("Unexpected archive nesting")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in sorted(archive.namelist()):
            if name.lower().endswith(".zip"):
                yield from archive_csvs(archive.read(name), depth + 1)
            elif name.lower().endswith(".csv") and Path(name).name.startswith(
                "PRSA_Data_"
            ):
                yield archive.read(name)


def prepare():
    raw = ROOT / "data/raw/uci501.zip"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        raw.write_bytes(response.content)
    frames = []
    for content in archive_csvs(raw.read_bytes()):
        frame = pd.read_csv(
            io.BytesIO(content),
            usecols=["year", "month", "day", "hour", "station", "PM2.5"],
        )
        frame["timestamp"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
        frames.append(frame[["timestamp", "station", "PM2.5"]])
    if len(frames) != 12:
        raise ValueError("Expected 12 station files")
    table = pd.concat(frames, ignore_index=True)
    if len(table) != 420768 or table.duplicated(["timestamp", "station"]).any():
        raise ValueError("Unexpected station-hour keys")
    matrix = (
        table.pivot(index="timestamp", columns="station", values="PM2.5")
        .sort_index()
        .sort_index(axis=1)
    )
    expected = pd.date_range("2013-03-01", "2017-02-28 23:00", freq="h")
    if matrix.shape != (35064, 12) or not matrix.index.equals(expected):
        raise ValueError("Invalid hourly matrix")
    values = matrix.to_numpy(dtype="float32")
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Negative observed PM2.5")
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_FILE,
        values=values,
        original_observed=np.isfinite(values),
        timestamps=matrix.index.to_numpy(dtype="datetime64[h]").astype("int64"),
        stations=matrix.columns.to_numpy(dtype="U32"),
    )
    return load_data()


def load_data():
    if not DATA_FILE.exists():
        return prepare()
    with np.load(DATA_FILE) as data:
        values = data["values"].copy()
        times = pd.to_datetime(data["timestamps"], unit="h")
        stations = data["stations"].tolist()
        if not np.array_equal(data["original_observed"], np.isfinite(values)):
            raise ValueError("Original missing-value mask changed")
    config = configuration()
    if stations != config["data"]["stations"] or not times.equals(
        pd.date_range("2013-03-01", "2017-02-28 23:00", freq="h")
    ):
        raise ValueError("Time axis or station order changed")
    stats = fit_stats(values, times)
    for key in ["mean", "std", "threshold"]:
        if not np.allclose(
            stats[key], config["preprocessing"][key], rtol=1e-5, atol=1e-5
        ):
            raise ValueError("Train-only preprocessing differs from the saved model")
    return (values, times, stations, config["preprocessing"])


def make_inputs(values, operational, mean, std, cycle):
    available = operational.bool() & torch.isfinite(values)
    safe = torch.where(available, values, mean)
    return (torch.cat([(safe - mean) / std, available.float(), cycle], 1), available)


def score_mask(values, operational):
    return torch.isfinite(values) & ~operational.bool()


def outage_plan(n, k, seed, device="cuda"):
    if not 0 <= k <= 12:
        raise ValueError("Operational count must be between 0 and 12")
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.rand((n, 12), generator=generator, device=device).argsort(1).argsort(1)
        < k
    )


def causal_windows(values, history_hours=24):
    padding = torch.full(
        (history_hours, values.shape[1]),
        float("nan"),
        dtype=values.dtype,
        device=values.device,
    )
    return (
        torch.cat([padding, values])
        .unfold(0, history_hours + 1, 1)
        .permute(0, 2, 1)
        .contiguous()
    )


def masked_sequence(values, operational, mean, std):
    available = operational.bool() & torch.isfinite(values)
    safe = torch.where(available, values, mean)
    return torch.where(available, (safe - mean) / std, float("nan"))


def episode_masks(n, endpoints, offline_subsets, duration):
    operational = np.ones((n, 12), dtype=bool)
    for endpoint, subset in zip(endpoints, offline_subsets):
        start = int(endpoint) - duration + 1
        if start < 0 or endpoint >= n:
            raise ValueError("Failure interval lies outside the partition")
        operational[start : int(endpoint) + 1, list(subset)] = False
    return operational


if __name__ == "__main__":
    prepare()
