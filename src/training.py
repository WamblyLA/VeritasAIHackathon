import argparse
import importlib
from pathlib import Path

import torch

from .data import (
    ROOT,
    calendar_features,
    causal_windows,
    configuration,
    load_data,
    make_inputs,
    masked_sequence,
    outage_plan,
    score_mask,
    split_indices,
)
from .models import ContextMDAE, MaskedAutoencoder, configure, create_saits


def training_mask(shape):
    counts = torch.randint(1, 12, (*shape[:-1], 1), device="cuda")
    return torch.rand(shape, device="cuda").argsort(-1).argsort(-1) < counts


def validation_data(values, times, mean, std):
    indices = split_indices(times)["validation"]
    labels = torch.tensor(values[indices], device="cuda")
    cycle = torch.tensor(calendar_features(times[indices]), device="cuda")
    plans = []
    for k in [3, 5, 8]:
        operational = outage_plan(len(labels), k, 2026 + 100 + k)
        plans.append(
            (
                operational,
                score_mask(labels, operational),
                causal_windows(masked_sequence(labels, operational, mean, std)),
            )
        )
    return labels, cycle, plans


@torch.inference_mode()
def validation_metrics(model, name, labels, cycle, plans, mean, std, multiplier):
    model.eval()
    squared, absolute, count, weight_count = 0.0, 0.0, 0, 0.0
    for operational, target, windows in plans:
        if name == "saits":
            outputs = []
            for batch in windows.split(1024):
                result = model(
                    {
                        "X": torch.nan_to_num(batch),
                        "missing_mask": torch.isfinite(batch).float(),
                    },
                    calc_criterion=False,
                )
                outputs.append(result["imputation"][:, -1])
            normalized = torch.cat(outputs)
        elif name in ["temporal_mdae", "matched_current"]:
            normalized = model(windows)
        else:
            inputs, available = make_inputs(labels, operational, mean, std, cycle)
            normalized = model(inputs)
        clean = torch.nan_to_num((labels - mean) / std)
        weights = torch.where(labels >= 187.0, multiplier, 1.0) * target
        squared += ((normalized - clean).square() * weights).sum()
        weight_count += weights.sum()
        prediction = (normalized * std + mean).clamp_min(0)
        absolute += torch.where(target, (prediction - labels).abs(), 0).sum()
        count += target.sum()
    mae = float(absolute / count)
    return (mae if name == "saits" else float(squared / weight_count)), mae


def fit_ridge(
    values, cycle, validation, validation_cycle, plans, mean, std, settings, output
):
    gram = torch.zeros(12, 29, 29, device="cuda", dtype=torch.float64)
    target_products = torch.zeros(12, 29, device="cuda", dtype=torch.float64)
    for _ in range(settings["ridge_augmentations"]):
        operational = training_mask(values.shape)
        inputs, available = make_inputs(values, operational, mean, std, cycle)
        shared = (inputs[:, :12] * available).sum(1, keepdim=True) / available.sum(
            1, keepdim=True
        ).clamp_min(1)
        inputs = torch.cat(
            [inputs, torch.ones(len(inputs), 1, device="cuda")], 1
        ).double()
        targets = score_mask(values, operational)
        labels = (torch.nan_to_num((values - mean) / std) - shared).double()
        for station in range(12):
            design = inputs[targets[:, station]]
            gram[station] += design.T @ design
            target_products[station] += design.T @ labels[targets[:, station], station]
    penalty = torch.eye(29, device="cuda", dtype=torch.float64)
    penalty[-1, -1] = 0
    best = None
    for alpha in settings["ridge_alphas"]:
        weights = (
            torch.linalg.solve(gram + alpha * penalty, target_products.unsqueeze(-1))
            .squeeze(-1)
            .T.float()
        )
        total, count = 0.0, 0
        for operational, target, windows in plans:
            inputs, available = make_inputs(
                validation, operational, mean, std, validation_cycle
            )
            shared = (inputs[:, :12] * available).sum(1, keepdim=True) / available.sum(
                1, keepdim=True
            ).clamp_min(1)
            design = torch.cat([inputs, torch.ones(len(inputs), 1, device="cuda")], 1)
            prediction = ((design @ weights + shared) * std + mean).clamp_min(0)
            total += torch.where(target, (prediction - validation).abs(), 0).sum()
            count += target.sum()
        mae = float(total / count)
        if best is None or mae < best[0]:
            best = mae, alpha, weights.clone()
    return {
        "weights": best[2].cpu(),
        "alpha": best[1],
        "validation_mae": best[0],
        "fit_partition": "train",
        "selection_partition": "validation",
    }


def train(name, output=None):
    output = Path(output) if output is not None else ROOT / "models"
    destination = output / f"{name}.pt"
    if destination.exists():
        return torch.load(destination, map_location="cpu", weights_only=True)
    values, times, stations, stats = load_data()
    config = configuration()
    settings = (
        config["saits"]["config"]
        if name == "saits"
        else config["final"]
        if name in ["temporal_mdae", "matched_current"]
        else config["training"]
    )
    if name == "saits":
        importlib.import_module("pypots.imputation")
    configure(2026)
    mean, std = [torch.tensor(stats[key], device="cuda") for key in ["mean", "std"]]
    indices = split_indices(times)["train"]
    labels = torch.tensor(values[indices], device="cuda")
    cycle = torch.tensor(calendar_features(times[indices]), device="cuda")
    validation, validation_cycle, plans = validation_data(values, times, mean, std)
    output.mkdir(parents=True, exist_ok=True)
    if name == "ridge":
        checkpoint = fit_ridge(
            labels,
            cycle,
            validation,
            validation_cycle,
            plans,
            mean,
            std,
            settings,
            output,
        )
        checkpoint.update(
            training_seed=2026,
            model_name=name,
            stats=stats,
            checkpoint_partition="validation",
            training_settings=settings,
        )
        torch.save(checkpoint, destination)
        return checkpoint
    temporal = name in ["saits", "temporal_mdae", "matched_current"]
    windows = causal_windows((labels - mean) / std) if temporal else None
    if name == "saits":
        wrapper = create_saits()
        model = wrapper.model
        optimizer = wrapper.optimizer
    else:
        model = (
            ContextMDAE(name == "temporal_mdae") if temporal else MaskedAutoencoder()
        ).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
    multiplier = 3.0 if name == "extreme" else 1.0
    best, stale, checkpoint = float("inf"), 0, None
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(labels), device="cuda")
        for batch in order.split(settings["batch_size"]):
            if temporal:
                original = windows[batch]
                finite = torch.isfinite(original)
                available = finite & training_mask(original.shape)
                masked = torch.where(available, original, float("nan"))
                if name == "saits":
                    inputs = {
                        "X": torch.where(available, original, 0),
                        "missing_mask": available.float(),
                        "X_ori": torch.nan_to_num(original),
                        "indicating_mask": (finite & ~available).float(),
                    }
                    loss = model(inputs, calc_criterion=True)["loss"]
                else:
                    target = finite[:, -1] & ~available[:, -1]
                    prediction = model(masked)
                    loss = (
                        (prediction - torch.nan_to_num(original[:, -1])).square()
                        * target
                    ).sum() / target.sum().clamp_min(1)
            else:
                subset = labels[batch]
                operational = training_mask(subset.shape)
                inputs, available = make_inputs(
                    subset, operational, mean, std, cycle[batch]
                )
                target = score_mask(subset, operational)
                weights = torch.where(subset >= 187.0, multiplier, 1.0) * target
                prediction = model(inputs)
                loss = (
                    (prediction - torch.nan_to_num((subset - mean) / std)).square()
                    * weights
                ).sum() / weights.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        with torch.inference_mode():
            monitor, mae = validation_metrics(
                model, name, validation, validation_cycle, plans, mean, std, multiplier
            )
        tolerance = 1e-5 if name == "saits" else 1e-6
        if monitor < best - tolerance:
            best, stale = monitor, 0
            checkpoint = {
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "stats": stats,
                "best_epoch": epoch,
                "validation_loss": monitor,
                "validation_mae": mae,
                "fit_partition": "train",
                "checkpoint_partition": "validation",
            }
            if name in ["temporal_mdae", "matched_current"]:
                checkpoint["use_history"] = name == "temporal_mdae"
            if name == "saits":
                checkpoint["trial"] = config["saits"]["winner"]["trial"]
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name}: epoch={epoch}, validation_mae={mae:.4f}", flush=True)
        if stale >= settings["patience"]:
            break
    checkpoint.update(
        training_seed=2026,
        model_name=name,
        training_settings=settings,
    )
    torch.save(checkpoint, destination)
    print(
        f"Saved {name}: best_epoch={checkpoint['best_epoch']}, validation_mae={checkpoint['validation_mae']:.4f}",
        flush=True,
    )
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "ridge",
            "ordinary",
            "extreme",
            "saits",
            "matched_current",
            "temporal_mdae",
        ],
    )
    args = parser.parse_args()
    train(args.model)


if __name__ == "__main__":
    main()
