from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from .data import (
    ROOT,
    calendar_features,
    episode_masks,
    load_data,
    outage_plan,
    score_mask,
    split_indices,
)
from .models import ModelBundle, METHODS


def metrics(truth, prediction, mask=None, threshold=187.0):
    if mask is not None:
        truth, prediction = truth[mask], prediction[mask]
    if not len(truth):
        raise ValueError("No genuine target observations")
    error = prediction - truth
    high, event = truth >= 187.0, prediction >= threshold
    true_positive = int((high & event).sum())
    false_positive = int((~high & event).sum())
    false_negative = int((high & ~event).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "n": len(truth),
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "high_n": int(high.sum()),
        "high_mae": float(error[high].abs().mean()) if high.any() else None,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None,
        "f1": 2 * true_positive / denominator if denominator else 0.0,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "decision_threshold": float(threshold),
    }


@torch.inference_mode()
def predictions(bundle, values, operational, cycle):
    result = {
        name: bundle.predict(name, values, operational, cycle)
        for name in METHODS
        if name not in ["ensemble", "saits_calibrated"]
    }
    weight = bundle.config["ensemble"]["selected"]["weight_saits"]
    result["ensemble"] = weight * result["saits"] + (1 - weight) * result["extreme"]
    result["saits_calibrated"] = result["saits"]
    return result


@torch.inference_mode()
def hourly_evaluation(bundle, values, cycle):
    targets, estimates = [], defaultdict(list)
    for k in range(1, 12):
        for repeat in range(3):
            operational = outage_plan(len(values), k, 2026 + 20000 + k * 100 + repeat)
            targets.append(score_mask(values, operational))
            for name, prediction in predictions(
                bundle, values, operational, cycle
            ).items():
                estimates[name].append(prediction)
    truth = values.repeat(33, 1)
    target = torch.cat(targets)
    records = []
    for name, parts in estimates.items():
        prediction = torch.cat(parts)
        records.append(
            {
                "method": name,
                "scope": "overall",
                "k": 0,
                **metrics(truth, prediction, target, bundle.threshold(name)),
            }
        )
        for k in range(1, 12):
            start, end = (k - 1) * 3 * len(values), k * 3 * len(values)
            records.append(
                {
                    "method": name,
                    "scope": "by_k",
                    "k": k,
                    **metrics(
                        truth[start:end],
                        prediction[start:end],
                        target[start:end],
                        bundle.threshold(name),
                    ),
                }
            )
    return pd.DataFrame(records)


@torch.inference_mode()
def continuous_evaluation(bundle, values, cycle):
    config = bundle.config["final"]
    endpoints = np.arange(
        config["failure_first_endpoint"], len(values), config["failure_endpoint_stride"]
    )
    pooled = defaultdict(list)
    for k in config["failure_operational_k"]:
        for repeat in range(config["failure_repeats"]):
            generator = np.random.default_rng(config["seed"] + 8000 + k * 100 + repeat)
            subsets = [
                sorted(generator.choice(12, 12 - k, replace=False).tolist())
                for _ in endpoints
            ]
            previous_target, long_reference, current_reference = None, {}, {}
            for duration in config["failure_durations_hours"]:
                operational = torch.tensor(
                    episode_masks(len(values), endpoints, subsets, duration),
                    device="cuda",
                )
                target = score_mask(values, operational)
                estimates = predictions(bundle, values, operational, cycle)
                endpoint_target = target[endpoints]
                if previous_target is not None and not torch.equal(
                    previous_target, endpoint_target
                ):
                    raise ValueError(
                        "Duration comparisons have different endpoint labels"
                    )
                previous_target = endpoint_target
                for name, prediction in estimates.items():
                    endpoint_prediction = prediction[endpoints]
                    if duration == 1 and name in [
                        "ordinary",
                        "ridge",
                        "matched_current",
                    ]:
                        current_reference[name] = endpoint_prediction.clone()
                    if name in current_reference and not torch.equal(
                        current_reference[name], endpoint_prediction
                    ):
                        raise ValueError(
                            "Current-only predictions changed with identical current inputs"
                        )
                    if duration == 72:
                        long_reference[name] = endpoint_prediction.clone()
                    if duration == 168 and not torch.equal(
                        long_reference[name], endpoint_prediction
                    ):
                        raise ValueError(
                            "Identical 24h contexts produced different predictions"
                        )
                    for scope, truth, estimate in [
                        (
                            "endpoint",
                            values[endpoints][endpoint_target],
                            endpoint_prediction[endpoint_target],
                        ),
                        ("whole_interval", values[target], prediction[target]),
                    ]:
                        pooled[(scope, duration, k, name)].append((truth, estimate))
                        pooled[(scope, duration, 0, name)].append((truth, estimate))
    records = []
    for (scope, duration, k, name), parts in pooled.items():
        truth = torch.cat([part[0] for part in parts])
        estimate = torch.cat([part[1] for part in parts])
        records.append(
            {
                "scope": scope,
                "duration_hours": duration,
                "k": k,
                "method": name,
                **metrics(truth, estimate, threshold=bundle.threshold(name)),
            }
        )
    return pd.DataFrame(records)


@torch.inference_mode()
def select_stations(bundle, values, cycle, stations, partition="validation"):
    if partition != "validation":
        raise ValueError("Station selection requires validation labels")
    choices = {}
    for k in [3, 5, 8]:
        candidates = list(combinations(range(12), k))
        scores = []
        for start in range(0, len(candidates), 16):
            batch = candidates[start : start + 16]
            masks = torch.zeros(
                (len(batch), len(values), 12), dtype=torch.bool, device="cuda"
            )
            for index, subset in enumerate(batch):
                masks[index, :, list(subset)] = True
            labels = values.expand(len(batch), -1, -1).reshape(-1, 12)
            calendars = cycle.expand(len(batch), -1, -1).reshape(-1, 4)
            prediction = bundle.predict(
                "ordinary", labels, masks.reshape(-1, 12), calendars
            ).reshape(len(batch), len(values), 12)
            target = torch.isfinite(values).unsqueeze(0) & ~masks
            error = torch.where(target, (prediction - values).abs(), 0)
            losses = (error.sum((1, 2)) / target.sum((1, 2))).tolist()
            scores.extend(zip(losses, batch))
        loss, subset = min(scores)
        expected = bundle.config["station_selection"]["choices"][str(k)]
        if not bundle.regenerated and (
            list(subset) != expected["indices"]
            or not np.isclose(loss, expected["validation_mae"], rtol=1e-5)
        ):
            raise ValueError("Validation-selected stations did not reproduce")
        if bundle.regenerated:
            subset = tuple(expected["indices"])
            loss = next(value for value, candidate in scores if candidate == subset)
        choices[str(k)] = {
            "indices": list(subset),
            "validation_mae": loss,
            "stations": [stations[i] for i in subset],
        }
    return choices


@torch.inference_mode()
def select_ensemble(bundle, values, cycle, partition="validation"):
    if partition != "validation":
        raise ValueError("Ensemble selection requires validation labels")
    truths, saits_predictions, extreme_predictions = [], [], []
    settings = bundle.config["final"]
    for k in settings["validation_k"]:
        operational = outage_plan(len(values), k, 2026 + 100 + k)
        target = score_mask(values, operational)
        truths.append(values[target])
        saits_predictions.append(
            bundle.predict("saits", values, operational, cycle)[target]
        )
        extreme_predictions.append(
            bundle.predict("extreme", values, operational, cycle)[target]
        )
    truth, saits, extreme = map(
        torch.cat, [truths, saits_predictions, extreme_predictions]
    )
    limit = float((saits - truth).abs().mean()) * settings["max_mae_ratio_to_saits"]
    candidates = []
    for weight in settings["ensemble_weights_saits"]:
        prediction = weight * saits + (1 - weight) * extreme
        for threshold in settings["decision_thresholds"]:
            result = metrics(truth, prediction, threshold=threshold)
            if result["mae"] <= limit + 1e-7:
                candidates.append({"weight_saits": weight, **result})
    selected = min(
        candidates,
        key=lambda row: (
            -row["f1"],
            row["mae"],
            abs(row["decision_threshold"] - 187.0),
            -row["weight_saits"],
            row["decision_threshold"],
        ),
    )
    expected = bundle.config["ensemble"]["selected"]
    if not bundle.regenerated and (
        selected["weight_saits"] != expected["weight_saits"]
        or selected["decision_threshold"] != expected["decision_threshold"]
    ):
        raise ValueError("Validation-selected ensemble did not reproduce")
    return selected


@torch.inference_mode()
def station_evaluation(bundle, values, cycle, stations, choices):
    generator = np.random.default_rng(2026 + 9000)
    records = []
    for k in [3, 5, 8]:
        candidates = list(combinations(range(12), k))
        random_indices = generator.choice(len(candidates), 50, replace=False)
        subsets = [
            ("optimized", 0, choices[str(k)]["indices"]),
            ("fixed", 0, np.linspace(0, 11, k, dtype=int).tolist()),
        ]
        subsets += [
            ("random", repeat, candidates[index])
            for repeat, index in enumerate(random_indices)
        ]
        for selection, repeat, subset in subsets:
            operational = torch.zeros_like(values, dtype=torch.bool)
            operational[:, list(subset)] = True
            target = score_mask(values, operational)
            for name in ["ordinary", "operational_mean"]:
                prediction = bundle.predict(name, values, operational, cycle)
                full = torch.where(
                    operational & torch.isfinite(values), values, prediction
                )
                full_mask = torch.isfinite(values)
                full_mae = float(
                    torch.where(full_mask, (full - values).abs(), 0).sum()
                    / full_mask.sum()
                )
                records.append(
                    {
                        "k": k,
                        "selection": selection,
                        "repeat": repeat,
                        "method": name,
                        "stations": "|".join(stations[i] for i in subset),
                        "full_network_mae": full_mae,
                        **metrics(values, prediction, target),
                    }
                )
    return pd.DataFrame(records)


def compare(actual, reference, keys, regenerated=False):
    reference = reference[reference.method.isin(METHODS)]
    combined = actual.merge(
        reference, on=keys, suffixes=("_actual", "_reference"), validate="one_to_one"
    )
    if len(combined) != len(reference) or len(actual) != len(reference):
        raise ValueError("Evaluation support or grouping changed")
    for field in [
        "n",
        "mae",
        "rmse",
        "precision",
        "recall",
        "high_n",
        "high_mae",
        "tp",
        "fp",
        "fn",
        "f1",
        "full_network_mae",
    ]:
        if field in actual and field in reference:
            left, right = combined[field + "_actual"], combined[field + "_reference"]
            if regenerated and field in ["tp", "fp", "fn"]:
                continue
            relative = (
                0.05
                if regenerated
                and field in ["mae", "rmse", "high_mae", "full_network_mae"]
                else 1e-5
            )
            absolute = (
                0.02 if regenerated and field in ["precision", "recall", "f1"] else 1e-5
            )
            if not np.allclose(
                left, right, rtol=relative, atol=absolute, equal_nan=True
            ):
                raise ValueError(f"Published {field} did not reproduce")


@torch.inference_mode()
def audit(bundle, values, times):
    partitions = split_indices(times)
    validation_values = torch.tensor(values[partitions["validation"]], device="cuda")
    validation_cycle = torch.tensor(
        calendar_features(times[partitions["validation"]]), device="cuda"
    )
    select_ensemble(bundle, validation_values, validation_cycle)
    print("Validation-only ensemble selection verified", flush=True)
    if [len(partitions[name]) for name in ["train", "validation", "test"]] != [
        24864,
        4368,
        5832,
    ]:
        raise ValueError("Chronological split changed")
    if max(partitions["train"]) >= min(partitions["validation"]) or max(
        partitions["validation"]
    ) >= min(partitions["test"]):
        raise ValueError("Partitions overlap")
    sample = torch.tensor(values[partitions["test"][:64]], device="cuda")
    cycle = torch.tensor(
        calendar_features(times[partitions["test"][:64]]), device="cuda"
    )
    operational = outage_plan(len(sample), 5, 2026)
    changed = sample.clone()
    changed[~operational] = 999999
    before, after = (
        predictions(bundle, sample, operational, cycle),
        predictions(bundle, changed, operational, cycle),
    )
    for name in METHODS:
        if (
            not torch.equal(before[name], after[name])
            or not torch.isfinite(before[name]).all()
        ):
            raise ValueError(
                f"Hidden-label input leakage or nonfinite prediction: {name}"
            )


def main():
    values, times, stations, stats = load_data()
    bundle = ModelBundle()
    audit(bundle, values, times)
    partitions = split_indices(times)
    test_values = torch.tensor(values[partitions["test"]], device="cuda")
    test_cycle = torch.tensor(
        calendar_features(times[partitions["test"]]), device="cuda"
    )
    hourly = hourly_evaluation(bundle, test_values, test_cycle)
    compare(
        hourly,
        pd.read_csv(ROOT / "results/metrics.csv"),
        ["method", "scope", "k"],
        bool(bundle.regenerated),
    )
    print("Hourly outage metrics reproduced from saved weights", flush=True)
    continuous = continuous_evaluation(bundle, test_values, test_cycle)
    compare(
        continuous,
        pd.read_csv(ROOT / "results/continuous.csv"),
        ["method", "scope", "k", "duration_hours"],
        bool(bundle.regenerated),
    )
    print("Continuous outage metrics and paired endpoints verified", flush=True)
    validation_values = torch.tensor(values[partitions["validation"]], device="cuda")
    validation_cycle = torch.tensor(
        calendar_features(times[partitions["validation"]]), device="cuda"
    )
    choices = select_stations(bundle, validation_values, validation_cycle, stations)
    station_results = station_evaluation(
        bundle, test_values, test_cycle, stations, choices
    )
    compare(
        station_results,
        pd.read_csv(ROOT / "results/station_subsets.csv"),
        ["method", "k", "selection", "repeat"],
        bool(bundle.regenerated),
    )
    print(
        "Exhaustive validation station selection and held-out subset evaluation reproduced",
        flush=True,
    )
    print(
        hourly.query("scope == 'overall'")[
            ["method", "mae", "rmse", "precision", "recall", "f1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
