import random

import numpy as np
import torch
from torch import nn

from .data import ROOT, configuration, make_inputs, causal_windows, masked_sequence

CHECKPOINTS = [
    "ridge",
    "ordinary",
    "extreme",
    "matched_current",
    "temporal_mdae",
    "saits",
]

METHODS = [
    "station_mean",
    "operational_mean",
    "ridge",
    "ordinary",
    "extreme",
    "saits",
    "matched_current",
    "temporal_mdae",
    "ensemble",
    "saits_calibrated",
]
LABELS = {
    "station_mean": "Historical mean",
    "operational_mean": "Operational mean",
    "ridge": "Ridge",
    "ordinary": "MDAE",
    "extreme": "Extreme-aware MDAE",
    "saits": "SAITS",
    "matched_current": "Matched MDAE without history",
    "temporal_mdae": "MDAE with 24h history",
    "ensemble": "SAITS + Extreme-aware MDAE",
    "saits_calibrated": "SAITS at threshold 180",
}


def configure(seed=2026):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model training and inference")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 24),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(24, 64), nn.SiLU(), nn.Linear(64, 12))

    def forward(self, inputs):
        values, mask = inputs[:, :12], inputs[:, 12:24]
        shared = (values * mask).sum(1, keepdim=True) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        return self.decoder(self.encoder(inputs)) + shared


class ContextMDAE(nn.Module):
    def __init__(self, use_history=True):
        super().__init__()
        base = MaskedAutoencoder()
        self.encoder, self.decoder = base.encoder, base.decoder
        self.encoder[0] = nn.Linear(600, 128)
        self.use_history = use_history

    def forward(self, windows):
        available = torch.isfinite(windows)
        if not self.use_history:
            available = available.clone()
            available[:, :-1] = False
        values = torch.where(available, windows, 0.0)
        inputs = torch.cat([values.flatten(1), available.float().flatten(1)], 1)
        shared = (values[:, -1] * available[:, -1]).sum(1, keepdim=True) / available[
            :, -1
        ].sum(1, keepdim=True).clamp_min(1)
        return self.decoder(self.encoder(inputs)) + shared


def create_saits(trial=None):
    from pypots.imputation import SAITS
    from pypots.optim import Adam

    config = configuration()
    trial = trial or config["saits"]["winner"]["trial"]
    return SAITS(
        n_steps=25,
        n_features=12,
        n_layers=trial["n_layers"],
        d_model=trial["d_model"],
        n_heads=trial["n_heads"],
        d_k=trial["d_model"] // trial["n_heads"],
        d_v=trial["d_model"] // trial["n_heads"],
        d_ffn=trial["d_ffn"],
        dropout=trial["dropout"],
        attn_dropout=0.0,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
        batch_size=512,
        epochs=120,
        patience=15,
        optimizer=Adam(lr=trial["lr"], weight_decay=1e-5),
        device="cuda:0",
        num_workers=0,
        model_saving_strategy=None,
        verbose=False,
    )


def ensure_checkpoint(name):
    path = ROOT / "models" / f"{name}.pt"
    if not path.exists():
        from .training import train

        print(f"Training missing checkpoint: {name}", flush=True)
        train(name, path.parent)
    return path


class ModelBundle:
    def __init__(self):
        configure()
        for name in [
            "ridge",
            "ordinary",
            "extreme",
            "matched_current",
            "temporal_mdae",
            "saits",
        ]:
            ensure_checkpoint(name)
        self.regenerated = [
            name
            for name in CHECKPOINTS
            if torch.load(
                ROOT / f"models/{name}.pt", map_location="cpu", weights_only=True
            ).get("training_seed")
            == 2026
        ]
        self.config = configuration()
        self.mean = torch.tensor(self.config["preprocessing"]["mean"], device="cuda")
        self.std = torch.tensor(self.config["preprocessing"]["std"], device="cuda")
        self.networks = {}
        for name in ["ordinary", "extreme", "matched_current", "temporal_mdae"]:
            checkpoint = torch.load(
                ROOT / f"models/{name}.pt", map_location="cuda", weights_only=True
            )
            model = (
                ContextMDAE(checkpoint["use_history"])
                if "use_history" in checkpoint
                else MaskedAutoencoder()
            )
            model = model.cuda()
            model.load_state_dict(checkpoint["state_dict"])
            self.networks[name] = model.eval()
        self.ridge = torch.load(
            ROOT / "models/ridge.pt", map_location="cuda", weights_only=True
        )["weights"]
        checkpoint = torch.load(
            ROOT / "models/saits.pt", map_location="cuda", weights_only=True
        )
        self.saits = create_saits(checkpoint["trial"])
        self.saits.model.load_state_dict(checkpoint["state_dict"])
        self.saits.model.eval()

    @torch.inference_mode()
    def predict(self, method, values, operational, cycle):
        if method == "ensemble":
            weight = self.config["ensemble"]["selected"]["weight_saits"]
            return weight * self.predict("saits", values, operational, cycle) + (
                1 - weight
            ) * self.predict("extreme", values, operational, cycle)
        if method == "saits_calibrated":
            return self.predict("saits", values, operational, cycle)
        inputs, available = make_inputs(values, operational, self.mean, self.std, cycle)
        if method == "station_mean":
            return self.mean.expand_as(values)
        if method == "operational_mean":
            count = available.sum(1, keepdim=True)
            total = torch.where(available, values, 0).sum(1, keepdim=True)
            return (
                torch.where(count > 0, total / count.clamp_min(1), self.mean)
                .expand_as(values)
                .clamp_min(0)
            )
        if method == "ridge":
            augmented = torch.cat(
                [inputs, torch.ones((len(inputs), 1), device="cuda")], 1
            )
            shared = (inputs[:, :12] * available).sum(1, keepdim=True) / available.sum(
                1, keepdim=True
            ).clamp_min(1)
            return ((augmented @ self.ridge + shared) * self.std + self.mean).clamp_min(
                0
            )
        if method in ["ordinary", "extreme"]:
            return (self.networks[method](inputs) * self.std + self.mean).clamp_min(0)
        windows = causal_windows(
            masked_sequence(values, operational, self.mean, self.std)
        )
        return self.predict_windows(method, windows)

    @torch.inference_mode()
    def predict_windows(self, method, windows):
        predictions = []
        for batch in windows.split(1024):
            if method == "saits":
                result = self.saits.model(
                    {
                        "X": torch.nan_to_num(batch),
                        "missing_mask": torch.isfinite(batch).float(),
                    },
                    calc_criterion=False,
                )
                output = result["imputation"][:, -1]
            else:
                output = self.networks[method](batch)
            predictions.append((output * self.std + self.mean).clamp_min(0))
        return torch.cat(predictions)

    def threshold(self, method):
        if method == "ensemble":
            return self.config["ensemble"]["selected"]["decision_threshold"]
        if method == "saits_calibrated":
            return self.config["ensemble"]["calibrated_saits"]["decision_threshold"]
        return 187.0
