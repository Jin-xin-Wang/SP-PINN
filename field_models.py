from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn as nn


FIELD_NAMES = ("alpha", "n", "theta_s", "theta_r", "ks")
FIELD_MODE_CHOICES = ("constant", "heterogeneous")


class FieldMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, hidden_layers=3, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, coords_dim):
        return self.network(coords_dim)


class AlphaFieldNet(FieldMLP):
    pass


class NFieldNet(FieldMLP):
    pass


class ThetaSFieldNet(FieldMLP):
    pass


class ThetaRFieldNet(FieldMLP):
    pass


class KsFieldNet(FieldMLP):
    pass


FIELD_MODEL_CLASSES = {
    "alpha": AlphaFieldNet,
    "n": NFieldNet,
    "theta_s": ThetaSFieldNet,
    "theta_r": ThetaRFieldNet,
    "ks": KsFieldNet,
}


def build_field_model(field_name):
    if field_name not in FIELD_MODEL_CLASSES:
        raise ValueError(f"Unsupported field model name: {field_name}")
    return FIELD_MODEL_CLASSES[field_name]()


def restore_field_values(field_name, normalized_values, field_stats):
    stats = field_stats["fields"][field_name]
    restored = normalized_values * float(stats["std"]) + float(stats["mean"])
    if field_name == "ks":
        return torch.exp(torch.clamp(restored, min=-30.0, max=10.0))
    return restored


def load_field_stats(field_model_dir):
    field_model_dir = Path(field_model_dir)
    stats_path = field_model_dir / "field_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Field stats not found: {stats_path}")
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _normalize_mode(mode):
    mode = str(mode).strip().lower()
    if mode not in FIELD_MODE_CHOICES:
        raise ValueError(f"Unsupported field mode: {mode}")
    return mode


def _torch_load_state_dict(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_frozen_field_models(field_model_dir, field_names, device):
    field_model_dir = Path(field_model_dir)
    stats = deepcopy(load_field_stats(field_model_dir))
    models = {}
    generation = {}
    for field_name in field_names:
        if field_name not in FIELD_NAMES:
            raise ValueError(f"Unsupported frozen field name: {field_name}")
        cfg = deepcopy(stats["fields"][field_name])
        mode = _normalize_mode(cfg.get("mode", "heterogeneous"))
        cfg["mode"] = mode
        if mode == "constant":
            generation[field_name] = {
                "mode": mode,
                "constant_value": float(cfg["constant_value"]),
                "uses_frozen_model": False,
            }
            continue

        weight_file = cfg.get("weight_file")
        if not weight_file:
            raise ValueError(f"Missing weight_file for heterogeneous field {field_name}")
        weight_path = field_model_dir / weight_file
        if not weight_path.exists():
            raise FileNotFoundError(f"Field model weights not found: {weight_path}")
        model = build_field_model(field_name).to(device)
        model.load_state_dict(_torch_load_state_dict(weight_path, device))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        models[field_name] = model
        generation[field_name] = {
            "mode": mode,
            "constant_value": None,
            "uses_frozen_model": True,
        }
    return stats, models, generation


def evaluate_frozen_field(field_name, xyz_dim, frozen_models, frozen_stats, frozen_generation):
    generation = frozen_generation.get(field_name, {})
    if generation.get("mode") == "constant":
        return torch.full_like(xyz_dim[:, :1], float(generation["constant_value"]))
    model = frozen_models[field_name]
    return restore_field_values(field_name, model(xyz_dim), frozen_stats)
