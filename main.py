import argparse
import math
import platform
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from data_io import (
    PARAMETER_FIELD_VARIABLES,
    PARAMETER_POINT_VARIABLES,
    build_depth_filtered_parameter_grid_points,
    build_parameter_field_plane_reference,
    load_input_manifest,
    load_parameter_field_dataset,
    load_parameter_point_targets,
    load_plane_reference_data,
    load_probe_dataset,
    metrics_for_pair,
    phys_to_dim_coords_numpy,
    save_loss_history,
    save_probe_variable_comparison,
    save_parameter_field_comparison,
    save_parameter_point_comparison,
    save_probe_theta_comparison,
    save_state_plane_comparison,
)
from model import (
    InverseFieldModelBundle,
    MINMAX_PARAMETER_FIELDS,
    N_OUTPUT_MODE,
    PARAMETER_NORMALIZATION_EXPANSION_FRACTION,
    PARAMETER_LOSS_FIELDS,
    PARAMETER_OUTPUT_MODE,
    RichardsPINN,
    UNIFIED_HIDDEN_DIM,
    UNIFIED_HIDDEN_LAYERS,
)
from physical import (
    compute_parameter_point_losses,
    compute_pde_loss,
    compute_probe_variable_loss,
    compute_state,
    evaluate_parameter_fields,
    sample_training_points,
)
from utils import (
    choose_device,
    count_parameters,
    elapsed_minutes,
    ensure_dir,
    format_log_row,
    set_seed,
    write_json,
)

PROJECT_DIR = Path(__file__).resolve().parent
PORTABLE_PROJECT_DIR = PROJECT_DIR
REFERENCE_DATA_DIR = (PROJECT_DIR / "reference_data").resolve()
DEFAULT_REFERENCE_MANIFEST = REFERENCE_DATA_DIR / "input_manifest.json"
DEFAULT_SEGMENTS_DIR = REFERENCE_DATA_DIR / "segments"
DEFAULT_PARAMETER_FIELD_DIR = REFERENCE_DATA_DIR / "parameter_fields_roi"
DEFAULT_PARAMETER_POINTS = (
    REFERENCE_DATA_DIR / "parameter_points_regular_5x3" / "parameter_point_targets.npz"
)
DEFAULT_PARAMETER_POINT_SET = "regular_5x3"
DEFAULT_THETA_PROBE_GROUP = "probes_regular_5x3"
DEFAULT_H_PROBE_GROUP = "probes_regular_5x3"
THETA_NOISE_ALLOWED_STD = (0.0, 0.02)
THETA_NOISE_DEFAULT_STD = 0.02
THETA_NOISE_DEFAULT_SEED = 20260526
THETA_NOISE_MODE = "fixed_white_gaussian_train_target_only"
THETA_NOISE_CLIP_RANGE = (0.0, 1.0)
LR_DECAY_COUNT = 20
LR_DECAY_GAMMA = 0.9
SNAPSHOT_FIGURE_DPI = 120
SNAPSHOT_PARAMETER_GRID_SHAPE_XZ = (50, 30)
RAIN1_STATE_PLOT_TIMES_LOCAL_D = (1.0, 3.0, 7.0, 12.0, 20.0)
DOMAIN = {"x_min": 0.0, "x_max": 100.0, "z_min": -0.6, "z_max": 0.0, "t_end": 10.0}
STATE_ERROR_VARIABLES = ("h", "k_h", "se", "theta_h")
STATE_ERROR_TIMES_LOCAL_D = (1.0, 5.0, 10.0)
LOSS_COMPONENT_KEY_BY_FIELD = {
    "lnalpha": "param_alpha",
    "n": "param_n",
    "theta_s": "param_theta_s",
    "theta_r": "param_theta_r",
    "lnks": "param_lnks",
}


def state_plot_times_for_segment(segment):
    if str(segment["label"]) == "rain1_0_20d":
        return RAIN1_STATE_PLOT_TIMES_LOCAL_D
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train SP-PINN: shared64, 5x3 parameter/water-content observations, rain1_0_20d, no BC/IC."
    )
    parser.add_argument("--epochs", type=int, default=300006)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--n-residual", type=int, default=100000)
    parser.add_argument("--lambda-pde", type=float, default=10.0)
    parser.add_argument("--lambda-probe-theta", type=float, default=1.0)
    parser.add_argument("--lambda-param", type=float, default=0.1)
    parser.add_argument("--lambda-param-alpha", type=float, default=1.0)
    parser.add_argument("--lambda-param-n", type=float, default=1.0)
    parser.add_argument("--lambda-param-theta-s", type=float, default=1.0)
    parser.add_argument("--lambda-param-theta-r", type=float, default=1.0)
    parser.add_argument("--lambda-param-lnks", type=float, default=1.0)
    parser.add_argument("--probe-batch-size", type=int, default=20480)
    parser.add_argument("--plane-batch-size", type=int, default=4096)
    parser.add_argument("--probe-plot-seed", type=int, default=42)
    parser.add_argument(
        "--theta-noise-std", type=float, default=THETA_NOISE_DEFAULT_STD,
        help="Absolute Gaussian noise std for water-content training targets: 0 or 0.02 (default).",
    )
    parser.add_argument("--theta-noise-seed", type=int, default=THETA_NOISE_DEFAULT_SEED)
    parser.add_argument("--comparison-every", type=int, default=20000)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic torch kernels."
    )
    return parser


def _remap_portable_project_path(path):
    parts = path.parts
    lowered = [part.lower() for part in parts]
    for marker, root in (
        ("reference_data", REFERENCE_DATA_DIR),
        ("pinn_inverse", PORTABLE_PROJECT_DIR),
    ):
        if marker in lowered:
            index = lowered.index(marker)
            suffix = parts[index + 1 :]
            return (root / Path(*suffix)).resolve() if suffix else root.resolve()
    return None


def resolve_portable_path(path_text, base_dir=None):
    path = Path(path_text)
    if path.is_absolute():
        if path.exists():
            return path
        remapped = _remap_portable_project_path(path)
        return remapped if remapped is not None else path
    candidates = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    candidates.append(PORTABLE_PROJECT_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_project_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def build_loss_plot_keys(args, segment_labels, active_parameter_loss_fields):
    keys = ["total"]
    if float(args.lambda_pde) > 0.0 and int(args.n_residual) > 0:
        keys.append("pde")
        keys.extend((f"pde_{label}" for label in segment_labels))
    if float(args.lambda_probe_theta) > 0.0:
        keys.append("probe_theta")
        keys.extend((f"probe_theta_{label}" for label in segment_labels))
    if float(args.lambda_param) > 0.0 and active_parameter_loss_fields:
        keys.append("param")
        keys.extend(
            (
                LOSS_COMPONENT_KEY_BY_FIELD[field]
                for field in PARAMETER_LOSS_FIELDS
                if field in active_parameter_loss_fields
            )
        )
    return keys


def build_lr_scheduler(optimizer, epochs):
    step_size = max(1, int(math.ceil(float(epochs) / float(LR_DECAY_COUNT))))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=LR_DECAY_GAMMA)


def build_dimless(manifest, field_global_stats):
    domain = manifest.get("domain", {})
    local_domain = manifest.get("local_domain", {})
    x_min = float(domain.get("x_min", DOMAIN["x_min"]))
    x_max = float(domain.get("x_max", DOMAIN["x_max"]))
    z_min = float(domain.get("z_min", DOMAIN["z_min"]))
    z_max = float(domain.get("z_max", DOMAIN["z_max"]))
    local_t_end = float(local_domain.get("t_end_day", DOMAIN["t_end"]))
    x_mid = 0.5 * (x_min + x_max)
    sx = 0.5 * (x_max - x_min)
    dimless = {
        "x_min": x_min,
        "x_max": x_max,
        "x_mid": x_mid,
        "sx": sx,
        "z_min": z_min,
        "z_max": z_max,
        "Lx": x_max - x_min,
        "Lz": z_max - z_min,
        "H0": 1.0,
        "Theta0": float(field_global_stats["theta0"]["median"]),
        "K0": 0.15,
    }
    dimless["T"] = dimless["Theta0"] * dimless["H0"] / dimless["K0"]
    dimless["tau_end"] = local_t_end / dimless["T"]
    dimless["Cx"] = (
        dimless["K0"] * dimless["T"] * dimless["H0"] / (dimless["Theta0"] * dimless["sx"] ** 2)
    )
    dimless["Cz"] = dimless["K0"] * dimless["T"] / (dimless["Theta0"] * dimless["Lz"])
    return dimless


def build_output_dir(model_type, epochs, data_window, user_output_dir):
    if user_output_dir:
        return resolve_project_path(user_output_dir)
    return (
        PROJECT_DIR
        / "outputs"
        / f"formal_{model_type}_inverse_fieldnet_{data_window}_e{int(epochs)}"
    )


def emit_log(message, log_path):
    print(message, flush=True)
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def se_model_state_dicts(se_models):
    return {label: deepcopy(model.state_dict()) for label, model in se_models.items()}


def load_se_model_state_dicts(se_models, state_dicts):
    for label, state_dict in state_dicts.items():
        se_models[label].load_state_dict(state_dict)


def build_checkpoint_payload(
    args,
    se_state_dicts,
    field_model_state_dict,
    dimless,
    best_total_loss,
    field_global_stats,
    segments,
    parameter_model_metadata,
):
    return {
        "se_model_state_dicts": se_state_dicts,
        "field_model_state_dict": field_model_state_dict,
        "args": {
            **vars(args),
            "field_global_stats": field_global_stats,
            "se_hidden_dim": int(UNIFIED_HIDDEN_DIM),
            "se_hidden_layers": int(UNIFIED_HIDDEN_LAYERS),
            "parameter_model_variant": parameter_model_metadata["parameter_model_variant"],
            "parameter_hidden_dim": int(parameter_model_metadata["parameter_hidden_dim"]),
            "parameter_hidden_layers": int(parameter_model_metadata["parameter_hidden_layers"]),
            "parameter_head_count": int(parameter_model_metadata["parameter_head_count"]),
            "shared_parameter_network": bool(parameter_model_metadata["shared_parameter_network"]),
            "segments": segments,
        },
        "dimless": dimless,
        "best_total_loss": float(best_total_loss),
    }


def validate_theta_noise_std(value):
    value = float(value)
    for allowed in THETA_NOISE_ALLOWED_STD:
        if abs(value - allowed) <= 1.0e-12:
            return float(allowed)
    choices = ", ".join(f"{allowed:g}" for allowed in THETA_NOISE_ALLOWED_STD)
    raise ValueError(f"--theta-noise-std must be one of {{{choices}}}; got {value:g}.")


def build_probe_training_tensors(
    reference_probe,
    dimless,
    device,
    observed_variable="theta_h",
    theta_noise_std=THETA_NOISE_DEFAULT_STD,
    theta_noise_seed=THETA_NOISE_DEFAULT_SEED,
):
    coords_dim = phys_to_dim_coords_numpy(reference_probe["coords"], dimless)
    if observed_variable not in reference_probe:
        raise KeyError(f"Probe dataset does not contain observed variable {observed_variable!r}.")
    value_array = reference_probe[observed_variable].astype(np.float32).reshape(-1, 1)
    value_min = float(np.min(value_array))
    value_max = float(np.max(value_array))
    value_range = float(max(value_max - value_min, 1.0e-6))
    target_array = value_array
    noise_stats = None
    if observed_variable == "theta_h":
        theta_noise_std = validate_theta_noise_std(theta_noise_std)
        if theta_noise_std > 0.0:
            rng = np.random.default_rng(int(theta_noise_seed))
            noise = rng.normal(0.0, theta_noise_std, size=value_array.shape).astype(np.float32)
            target_array = np.clip(
                value_array + noise,
                THETA_NOISE_CLIP_RANGE[0],
                THETA_NOISE_CLIP_RANGE[1],
            ).astype(np.float32)
        else:
            noise = np.zeros_like(value_array, dtype=np.float32)
        noise_stats = {
            "theta_noise_std": float(theta_noise_std),
            "theta_noise_seed": None if theta_noise_seed is None else int(theta_noise_seed),
            "theta_noise_mode": THETA_NOISE_MODE,
            "theta_noise_clip_range": [float(THETA_NOISE_CLIP_RANGE[0]), float(THETA_NOISE_CLIP_RANGE[1])],
            "clean_min": value_min,
            "clean_max": value_max,
            "noisy_min": float(np.min(target_array)),
            "noisy_max": float(np.max(target_array)),
            "noise_mean": float(np.mean(noise)),
            "noise_std_empirical": float(np.std(noise)),
        }
    value_stats = {
        "min": value_min,
        "max": value_max,
        "range": value_range,
        "normalization": "probe_minmax",
        "observed_variable": observed_variable,
    }
    if noise_stats is not None:
        value_stats.update(noise_stats)
    payload = {
        "coords_dim_t": torch.tensor(coords_dim, dtype=torch.float32, device=device),
        "value_targets_norm_t": torch.tensor(
            (target_array - value_min) / value_range,
            dtype=torch.float32,
            device=device,
        ),
        "value_stats": value_stats,
        "observed_variable": observed_variable,
    }
    if observed_variable == "theta_h":
        payload["theta_targets_norm_t"] = payload["value_targets_norm_t"]
        payload["theta_stats"] = value_stats
    if observed_variable == "h":
        payload["h_targets_norm_t"] = payload["value_targets_norm_t"]
        payload["h_stats"] = value_stats
    return payload


def inverse_softplus_numpy(value):
    value = np.asarray(value, dtype=np.float64)
    safe = np.maximum(value, 1e-12)
    out = np.empty_like(safe, dtype=np.float64)
    large = safe > 20.0
    out[large] = safe[large]
    out[~large] = np.log(np.expm1(safe[~large]))
    return out.astype(np.float32)


def build_parameter_output_stats(parameter_points):
    stats = {}
    for field_name in MINMAX_PARAMETER_FIELDS:
        values = np.asarray(parameter_points[field_name], dtype=np.float64).reshape(-1)
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        observed_range = float(observed_max - observed_min)
        expanded_span = float(max(observed_range, 1e-06))
        min_value = float(observed_min - PARAMETER_NORMALIZATION_EXPANSION_FRACTION * expanded_span)
        max_value = float(observed_max + PARAMETER_NORMALIZATION_EXPANSION_FRACTION * expanded_span)
        value_range = float(max_value - min_value)
        stats[field_name] = {
            "min": min_value,
            "max": max_value,
            "range": value_range,
            "observed_min": observed_min,
            "observed_max": observed_max,
            "observed_range": observed_range,
            "median": float(np.median(values)),
            "source": "selected_parameter_points_expanded_25pct",
            "expansion_fraction": PARAMETER_NORMALIZATION_EXPANSION_FRACTION,
            "normalization_formula": "norm = 2 * (value - min) / (max - min) - 1",
            "inverse_formula": "value = (norm + 1) * (max - min) / 2 + min",
            "output_mode": "normalized_minus1_to_1",
            "loss_mode": "normalized_mse_direct",
        }
    return stats


def build_parameter_point_tensors(parameter_points, dimless, parameter_output_stats, device):
    coords_dim = phys_to_dim_coords_numpy(parameter_points["coords_phys"], dimless)
    targets = {}
    for field_name in PARAMETER_LOSS_FIELDS:
        if field_name in MINMAX_PARAMETER_FIELDS:
            stats = parameter_output_stats[field_name]
            values = np.asarray(parameter_points[field_name], dtype=np.float32)
            target_values = 2.0 * (values - float(stats["min"])) / float(stats["range"]) - 1.0
        elif field_name == "n":
            target_values = (
                np.asarray(parameter_points["raw_n"], dtype=np.float32)
                if "raw_n" in parameter_points
                else inverse_softplus_numpy(
                    np.asarray(parameter_points["n"], dtype=np.float64) - 1.0
                )
            )
        else:
            target_values = parameter_points[field_name]
        targets[field_name] = torch.tensor(target_values, dtype=torch.float32, device=device)
    stats = {
        field_name: dict(parameter_output_stats[field_name])
        for field_name in MINMAX_PARAMETER_FIELDS
    }
    return {
        "coords_dim_t": torch.tensor(coords_dim, dtype=torch.float32, device=device),
        "targets": targets,
        "stats": stats,
    }


def build_plane_coords(plane_name, coord1_1d, coord2_1d, fixed_value, time_d):
    c1, c2 = np.meshgrid(coord1_1d, coord2_1d, indexing="xy")
    tt = np.full_like(c1, float(time_d), dtype=np.float32)
    if plane_name != "xz":
        raise ValueError(f"Unsupported plane name: {plane_name}")
    coords = np.column_stack([c1.reshape(-1), c2.reshape(-1), tt.reshape(-1)])
    return coords.astype(np.float32)


def build_static_plane_coords(plane_name, coord1_1d, coord2_1d, fixed_value):
    c1, c2 = np.meshgrid(coord1_1d, coord2_1d, indexing="xy")
    if plane_name != "xz":
        raise ValueError(f"Unsupported plane name: {plane_name}")
    coords = np.column_stack([c1.reshape(-1), c2.reshape(-1)])
    return coords.astype(np.float32)


def predict_state_plane_fields(se_model, field_model, plane_reference, dimless, batch_size, device):
    predictions = {}
    se_model.eval()
    field_model.eval()
    for plane_name, plane_ref in plane_reference["planes"].items():
        plane_pred = {"fields": {name: [] for name in plane_ref["fields"].keys()}}
        for time_d in plane_reference["times_d"]:
            coords_phys = build_plane_coords(
                plane_name,
                plane_ref["coord1_1d"],
                plane_ref["coord2_1d"],
                plane_ref["fixed_value"],
                time_d,
            )
            coords_dim = phys_to_dim_coords_numpy(coords_phys, dimless)
            chunks = {name: [] for name in plane_ref["fields"].keys()}
            with torch.enable_grad():
                for start in range(0, coords_dim.shape[0], batch_size):
                    stop = start + batch_size
                    coords_t = torch.tensor(
                        coords_dim[start:stop], dtype=torch.float32, device=device
                    )
                    state = compute_state(
                        se_model, field_model, coords_t, dimless, require_spatial=True
                    )
                    for var_name in plane_ref["fields"].keys():
                        chunks[var_name].append(state[var_name].detach().cpu().numpy())
            for var_name in plane_ref["fields"].keys():
                plane_pred["fields"][var_name].append(
                    np.concatenate(chunks[var_name], axis=0)
                    .reshape(len(plane_ref["coord2_1d"]), len(plane_ref["coord1_1d"]))
                    .astype(np.float32)
                )
        for var_name in plane_ref["fields"].keys():
            plane_pred["fields"][var_name] = np.stack(plane_pred["fields"][var_name], axis=0)
        predictions[plane_name] = plane_pred
    return predictions


def predict_probe_variable(
    se_model, field_model, reference_probe, dimless, batch_size, device, observed_variable="theta_h"
):
    coords_dim = phys_to_dim_coords_numpy(reference_probe["coords"], dimless)
    if observed_variable not in ("theta_h", "h"):
        raise ValueError(f"Unsupported observed variable {observed_variable!r}.")
    if observed_variable not in reference_probe:
        raise KeyError(f"Probe dataset does not contain observed variable {observed_variable!r}.")
    value_parts = []
    se_model.eval()
    field_model.eval()
    with torch.no_grad():
        for start in range(0, coords_dim.shape[0], batch_size):
            stop = start + batch_size
            coords_t = torch.tensor(coords_dim[start:stop], dtype=torch.float32, device=device)
            state = compute_state(se_model, field_model, coords_t, dimless, require_spatial=False)
            value_parts.append(state[observed_variable].detach().cpu().numpy())
    n_probes = len(reference_probe["metadata"])
    n_times = len(reference_probe["times_d"])
    return np.concatenate(value_parts, axis=0).reshape(n_probes, n_times)


def predict_parameter_field_planes(
    field_model, parameter_plane_reference, dimless, batch_size, device
):
    predictions = {"planes": {}}
    field_model.eval()
    for plane_name, plane_ref in parameter_plane_reference["planes"].items():
        coords_phys = build_static_plane_coords(
            plane_name, plane_ref["coord1_1d"], plane_ref["coord2_1d"], plane_ref["fixed_value"]
        )
        coords_dim = phys_to_dim_coords_numpy(coords_phys, dimless)
        chunks = {name: [] for name in PARAMETER_FIELD_VARIABLES}
        with torch.no_grad():
            for start in range(0, coords_dim.shape[0], batch_size):
                stop = start + batch_size
                coords_t = torch.tensor(coords_dim[start:stop], dtype=torch.float32, device=device)
                local = evaluate_parameter_fields(field_model, coords_t)
                for field_name in PARAMETER_FIELD_VARIABLES:
                    chunks[field_name].append(local[field_name].detach().cpu().numpy())
        predictions["planes"][plane_name] = {
            "axis_names": plane_ref["axis_names"],
            "fixed_axis": plane_ref["fixed_axis"],
            "fixed_value": plane_ref["fixed_value"],
            "coord1_1d": plane_ref["coord1_1d"],
            "coord2_1d": plane_ref["coord2_1d"],
            "fields": {},
        }
        for field_name in PARAMETER_FIELD_VARIABLES:
            predictions["planes"][plane_name]["fields"][field_name] = (
                np.concatenate(chunks[field_name], axis=0)
                .reshape(len(plane_ref["coord2_1d"]), len(plane_ref["coord1_1d"]))
                .astype(np.float32)
            )
    return predictions


def predict_parameter_points(field_model, parameter_points, dimless, device, batch_size=None):
    coords_dim = phys_to_dim_coords_numpy(parameter_points["coords_phys"], dimless)
    batch_size = int(batch_size or coords_dim.shape[0])
    batch_size = max(1, batch_size)
    chunks = {field_name: [] for field_name in PARAMETER_POINT_VARIABLES}
    field_model.eval()
    with torch.no_grad():
        for start in range(0, coords_dim.shape[0], batch_size):
            stop = start + batch_size
            coords_t = torch.tensor(coords_dim[start:stop], dtype=torch.float32, device=device)
            local = evaluate_parameter_fields(field_model, coords_t)
            for field_name in PARAMETER_POINT_VARIABLES:
                chunks[field_name].append(
                    local[field_name].detach().cpu().numpy().astype(np.float32)
                )
    return {
        field_name: np.concatenate(chunks[field_name], axis=0)
        for field_name in PARAMETER_POINT_VARIABLES
    }


def collect_parameter_fit_rel_l2(
    field_model, parameter_evaluation_points, dimless, device, batch_size
):
    predictions = predict_parameter_points(
        field_model, parameter_evaluation_points, dimless, device, batch_size
    )
    return {
        field_name: float(
            metrics_for_pair(parameter_evaluation_points[field_name], predictions[field_name])[
                "rel_l2"
            ]
        )
        for field_name in PARAMETER_POINT_VARIABLES
    }


def collect_state_fit_rel_l2(
    se_model,
    field_model,
    plane_reference,
    dimless,
    batch_size,
    device,
    representative_times_local_d=STATE_ERROR_TIMES_LOCAL_D,
):
    predictions = predict_state_plane_fields(
        se_model, field_model, plane_reference, dimless, batch_size, device
    )
    ref_xz = plane_reference["planes"]["xz"]
    pred_xz = predictions["xz"]
    available_times = np.asarray(plane_reference["times_d"], dtype=np.float64).reshape(-1)
    time_indices = []
    for time_d in representative_times_local_d:
        matches = np.where(np.isclose(available_times, float(time_d), atol=1e-06))[0]
        if matches.size:
            time_indices.append(int(matches[0]))
    if not time_indices:
        time_indices = list(range(len(available_times)))
    state_rel_l2 = {}
    for variable in STATE_ERROR_VARIABLES:
        if variable not in ref_xz["fields"] or variable not in pred_xz["fields"]:
            continue
        rel_l2_values = [
            float(
                metrics_for_pair(ref_xz["fields"][variable][idx], pred_xz["fields"][variable][idx])[
                    "rel_l2"
                ]
            )
            for idx in time_indices
        ]
        state_rel_l2[variable] = float(np.mean(rel_l2_values))
    return state_rel_l2


def update_fit_error_history_row(
    row,
    se_model,
    field_model,
    plane_reference,
    parameter_evaluation_points,
    dimless,
    batch_size,
    device,
):
    parameter_rel_l2 = collect_parameter_fit_rel_l2(
        field_model, parameter_evaluation_points, dimless, device, batch_size
    )
    state_rel_l2 = collect_state_fit_rel_l2(
        se_model, field_model, plane_reference, dimless, batch_size, device
    )
    for field_name, value in parameter_rel_l2.items():
        row[f"param_field_rel_l2_{field_name}"] = float(value)
    for variable, value in state_rel_l2.items():
        row[f"state_rel_l2_{variable}"] = float(value)


def load_segment_references(
    manifest, segments_dir, dimless, device,
    theta_noise_std=THETA_NOISE_DEFAULT_STD,
    theta_noise_seed=THETA_NOISE_DEFAULT_SEED,
):
    segments = []
    for segment_index, spec in enumerate(manifest["segments"]):
        segment_root = Path(segments_dir) / spec["label"]
        probe = load_probe_dataset(segment_root / "probes_regular_5x3")
        theta_training = build_probe_training_tensors(
            probe, dimless, device, "theta_h",
            theta_noise_std=theta_noise_std,
            theta_noise_seed=int(theta_noise_seed) + segment_index,
        )
        segments.append(
            {
                "label": spec["label"],
                "spec": spec,
                "probe_group": "probes_regular_5x3",
                "theta_probe_group": "probes_regular_5x3",
                "h_probe_group": "probes_regular_5x3",
                "plane_reference": load_plane_reference_data(
                    segment_root / "planes",
                    spec["snapshot_times_local_d"],
                    manifest.get("plane_definitions"),
                    physical_times_d=spec.get("snapshot_times_physical_d"),
                ),
                "reference_probe": probe,
                "theta_reference_probe": probe,
                "h_reference_probe": probe,
                "probe_training": theta_training,
                "theta_probe_training": theta_training,
                "h_probe_training": build_probe_training_tensors(probe, dimless, device, "h"),
            }
        )
    return segments


def export_epoch_snapshot(
    se_models,
    field_model,
    dimless,
    segment_references,
    parameter_plane_reference,
    snapshot_parameter_evaluation_points,
    plane_batch_size,
    probe_batch_size,
    device,
    comparison_dir,
    probe_plot_seed,
    comparison_every,
    figure_dpi,
    epoch,
    snapshot_records,
):
    se_training_flags = {label: model.training for label, model in se_models.items()}
    field_was_training = field_model.training
    snapshot_root = ensure_dir(Path(comparison_dir))
    epoch_dir = ensure_dir(snapshot_root / f"e{int(epoch)}")
    epoch_segments_dir = ensure_dir(epoch_dir / "segments")
    epoch_parameter_fit_dir = ensure_dir(epoch_dir / "parameter_fit")
    segment_records = {}
    for segment in segment_references:
        label = segment["label"]
        segment_dir = epoch_segments_dir
        se_model = se_models[label]
        plane_predictions = predict_state_plane_fields(
            se_model, field_model, segment["plane_reference"], dimless, plane_batch_size, device
        )
        save_state_plane_comparison(
            segment["plane_reference"],
            plane_predictions,
            segment_dir,
            dpi=figure_dpi,
            plot_times_local_d=state_plot_times_for_segment(segment),
        )
        theta_probe_predictions = predict_probe_variable(
            se_model,
            field_model,
            segment["reference_probe"],
            dimless,
            probe_batch_size,
            device,
            "theta_h",
        )
        theta_probe_metrics = save_probe_theta_comparison(
            segment["reference_probe"],
            theta_probe_predictions,
            segment_dir,
            probe_plot_seed,
            dpi=figure_dpi,
        )
        h_probe_predictions = predict_probe_variable(
            se_model,
            field_model,
            segment["h_reference_probe"],
            dimless,
            probe_batch_size,
            device,
            "h",
        )
        h_probe_metrics = save_probe_variable_comparison(
            segment["h_reference_probe"],
            h_probe_predictions,
            segment_dir,
            probe_plot_seed,
            variable_name="h",
            file_prefix="probe_h",
            plot_prefix="probe_h",
            dpi=figure_dpi,
        )
        segment_records[label] = {
            "segments_dir": str(segment_dir.resolve()),
            "theta_probe_metrics": theta_probe_metrics,
            "h_probe_metrics": h_probe_metrics,
        }
    parameter_field_predictions = predict_parameter_field_planes(
        field_model, parameter_plane_reference, dimless, plane_batch_size, device
    )
    save_parameter_field_comparison(
        parameter_plane_reference,
        parameter_field_predictions,
        epoch_parameter_fit_dir,
        dpi=figure_dpi,
    )
    parameter_point_predictions = predict_parameter_points(
        field_model, snapshot_parameter_evaluation_points, dimless, device, plane_batch_size
    )
    save_parameter_point_comparison(
        snapshot_parameter_evaluation_points,
        parameter_point_predictions,
        epoch_parameter_fit_dir,
        dpi=figure_dpi,
    )
    record = {
        "epoch": int(epoch),
        "figure_dpi": int(figure_dpi),
        "parameter_point_evaluation_metadata": snapshot_parameter_evaluation_points.get(
            "evaluation_metadata", {}
        ),
        "segments": segment_records,
        "parameter_fit_dir": str(epoch_parameter_fit_dir.resolve()),
    }
    snapshot_records[:] = [item for item in snapshot_records if item["epoch"] != int(epoch)]
    snapshot_records.append(record)
    snapshot_records.sort(key=lambda item: item["epoch"])
    write_json(
        snapshot_root / "epoch_snapshot_manifest.json",
        {
            "comparison_every": int(comparison_every),
            "snapshot_figure_dpi": int(figure_dpi),
            "parameter_point_evaluation_metadata": snapshot_parameter_evaluation_points.get(
                "evaluation_metadata", {}
            ),
            "exports": snapshot_records,
        },
    )
    for label, was_training in se_training_flags.items():
        if was_training:
            se_models[label].train()
    if field_was_training:
        field_model.train()
    return record


def train(args):
    model_type = "baseline"
    activation = "sin"
    device = choose_device(args.device)
    set_seed(args.seed, deterministic=bool(args.deterministic))
    max_wall_hours = None if args.max_wall_hours is None else float(args.max_wall_hours)
    if max_wall_hours is not None and max_wall_hours <= 0.0:
        raise ValueError("--max-wall-hours must be positive when provided.")
    comparison_every = int(args.comparison_every)
    if comparison_every < 0:
        raise ValueError("--comparison-every must be non-negative.")
    theta_noise_std = validate_theta_noise_std(args.theta_noise_std)
    theta_noise_seed = int(args.theta_noise_seed)
    reference_manifest_path = resolve_portable_path(str(DEFAULT_REFERENCE_MANIFEST))
    segments_dir = resolve_portable_path(str(DEFAULT_SEGMENTS_DIR))
    parameter_field_dir = resolve_portable_path(str(DEFAULT_PARAMETER_FIELD_DIR))
    if not reference_manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {reference_manifest_path}")
    if not segments_dir.exists():
        raise FileNotFoundError(f"Segments reference dir not found: {segments_dir}")
    if not parameter_field_dir.exists():
        raise FileNotFoundError(f"Parameter field dir not found: {parameter_field_dir}")
    raw_manifest = load_input_manifest(reference_manifest_path)
    parameter_points_path = DEFAULT_PARAMETER_POINTS
    parameter_point_set = DEFAULT_PARAMETER_POINT_SET
    if not parameter_points_path.exists():
        raise FileNotFoundError(
            f"Missing bundled parameter observations: {parameter_points_path}. Restore reference_data from this package."
        )
    manifest = deepcopy(raw_manifest)
    if [s["label"] for s in manifest["segments"]] != ["rain1_0_20d"]:
        raise ValueError("This package contains only rain1_0_20d.")
    output_dir = ensure_dir(
        build_output_dir(model_type, args.epochs, "rain1_0_20d", args.output_dir)
    )
    training_log_path = output_dir / "training.log"
    training_log_path.write_text("", encoding="utf-8")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    comparison_dir = output_dir / "comparison"
    if comparison_dir.exists():
        shutil.rmtree(comparison_dir)
    comparison_dir = ensure_dir(comparison_dir)
    write_json(
        comparison_dir / "epoch_snapshot_manifest.json",
        {
            "comparison_every": int(comparison_every),
            "snapshot_figure_dpi": int(SNAPSHOT_FIGURE_DPI),
            "exports": [],
        },
    )
    field_global_stats = manifest["field_global_stats"]
    dimless = build_dimless(manifest, field_global_stats)
    dimless["comsol_physical_domain"] = manifest.get("comsol_physical_domain", {})
    segment_references = load_segment_references(
        manifest, segments_dir, dimless, device,
        theta_noise_std=theta_noise_std, theta_noise_seed=theta_noise_seed,
    )
    write_json(
        output_dir / "observation_noise.json",
        {
            "theta_noise_std": theta_noise_std,
            "theta_noise_seed": theta_noise_seed,
            "theta_noise_mode": THETA_NOISE_MODE,
            "theta_noise_clip_range": list(THETA_NOISE_CLIP_RANGE),
            "evaluation_reference": "clean",
            "normalization_reference": "clean_probe_minmax",
            "theta_h_stats_by_segment": {
                segment["label"]: segment["theta_probe_training"]["theta_stats"]
                for segment in segment_references
            },
        },
    )
    emit_log(
        f"Water-content observation noise: std={theta_noise_std} seed={theta_noise_seed} "
        f"mode={THETA_NOISE_MODE} clip={THETA_NOISE_CLIP_RANGE}; evaluation_reference=clean",
        training_log_path,
    )
    segment_labels = [segment["label"] for segment in segment_references]
    theta_probe_group = manifest.get(
        "theta_probe_group_dir_name",
        manifest.get("probe_group_dir_name", DEFAULT_THETA_PROBE_GROUP),
    )
    h_probe_group = manifest.get("h_probe_group_dir_name", DEFAULT_H_PROBE_GROUP)
    segment_local_t_end = float(manifest.get("local_domain", {}).get("t_end_day", DOMAIN["t_end"]))
    parameter_field_dataset = load_parameter_field_dataset(parameter_field_dir)
    parameter_plane_reference = build_parameter_field_plane_reference(
        parameter_field_dataset, manifest.get("plane_definitions")
    )
    parameter_evaluation_points = build_depth_filtered_parameter_grid_points(
        parameter_field_dataset
    )
    snapshot_parameter_evaluation_points = build_depth_filtered_parameter_grid_points(
        parameter_field_dataset, grid_shape=SNAPSHOT_PARAMETER_GRID_SHAPE_XZ
    )
    write_json(
        comparison_dir / "epoch_snapshot_manifest.json",
        {
            "comparison_every": int(comparison_every),
            "snapshot_figure_dpi": int(SNAPSHOT_FIGURE_DPI),
            "parameter_point_evaluation_metadata": snapshot_parameter_evaluation_points.get(
                "evaluation_metadata", {}
            ),
            "exports": [],
        },
    )
    parameter_points = load_parameter_point_targets(parameter_points_path)
    parameter_output_stats = build_parameter_output_stats(parameter_points)
    param_training = build_parameter_point_tensors(
        parameter_points, dimless, parameter_output_stats, device
    )
    se_models = torch.nn.ModuleDict(
        {
            label: RichardsPINN(model_type=model_type, activation=activation)
            for label in segment_labels
        }
    ).to(device)
    field_model = InverseFieldModelBundle(
        field_global_stats,
        parameter_model="shared64",
        parameter_output_stats=parameter_output_stats,
    ).to(device)
    parameter_model_metadata = field_model.parameter_model_metadata()
    trainable_parameter_fields = field_model.trainable_parameter_fields()
    trainable_parameter_loss_fields = field_model.trainable_loss_fields()
    parameter_loss_field_weights = {
        "lnalpha": float(args.lambda_param_alpha),
        "n": float(args.lambda_param_n),
        "theta_s": float(args.lambda_param_theta_s),
        "theta_r": float(args.lambda_param_theta_r),
        "lnks": float(args.lambda_param_lnks),
    }
    loss_plot_keys = build_loss_plot_keys(args, segment_labels, trainable_parameter_loss_fields)
    se_params = sum((count_parameters(model) for model in se_models.values()))
    field_params = count_parameters(field_model)
    total_params = se_params + field_params
    trainable_params = [
        param
        for param in list(se_models.parameters()) + list(field_model.parameters())
        if param.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    checkpoint_every = max(1, args.epochs // 30)
    scheduler = build_lr_scheduler(optimizer, args.epochs)
    print_every = (
        max(1, int(args.print_every))
        if args.print_every is not None
        else max(1, args.epochs // 1000)
    )
    emit_log(
        f"Training run started. output_dir={output_dir} seed={args.seed} model=baseline/sin parameter_model=shared64 data_window=rain1_0_20d parameter_points=5x3 theta_observations=5x3 BC/IC=disabled epochs={args.epochs} n_residual={args.n_residual} deterministic={args.deterministic}",
        training_log_path,
    )
    history = []
    epoch_snapshot_records = []
    best_loss = 1e309
    best_state = {
        "se_model_state_dicts": se_model_state_dicts(se_models),
        "field_model_state_dict": deepcopy(field_model.state_dict()),
    }
    train_start = time.perf_counter()
    max_wall_seconds = None if max_wall_hours is None else max_wall_hours * 3600.0
    stop_reason = "completed_requested_epochs"
    stopped_early = False
    last_epoch = 0
    se_models.train()
    field_model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        pde_losses = {}
        pde_values_for_mean = []
        segment_pde_weight = float(args.lambda_pde) / float(len(segment_references))
        residual_tau_cap_day = None
        for segment in segment_references:
            label = segment["label"]
            residual_points = sample_training_points(
                dimless, args.n_residual, device, tau_max_day=residual_tau_cap_day
            )
            pde_loss = compute_pde_loss(se_models[label], field_model, residual_points, dimless)
            pde_loss_value = float(pde_loss.detach().cpu())
            pde_losses[label] = pde_loss_value
            pde_values_for_mean.append(pde_loss_value)
            if segment_pde_weight != 0.0:
                (segment_pde_weight * pde_loss).backward()
        param_total_loss, param_components = compute_parameter_point_losses(
            field_model,
            param_training["coords_dim_t"],
            param_training["targets"],
            param_training["stats"],
            active_fields=trainable_parameter_loss_fields,
            field_weights=parameter_loss_field_weights,
        )
        if float(args.lambda_param) != 0.0 and trainable_parameter_loss_fields:
            (float(args.lambda_param) * param_total_loss).backward()
        probe_theta_values = {}
        if float(args.lambda_probe_theta) > 0.0:
            segment_probe_weight = float(args.lambda_probe_theta) / float(len(segment_references))
            for segment in segment_references:
                label = segment["label"]
                probe_training = segment["theta_probe_training"]
                total_probe = int(probe_training["coords_dim_t"].shape[0])
                probe_sum = 0.0
                for start in range(0, total_probe, int(args.probe_batch_size)):
                    stop = min(start + int(args.probe_batch_size), total_probe)
                    batch_coords = probe_training["coords_dim_t"][start:stop]
                    batch_targets = probe_training["value_targets_norm_t"][start:stop]
                    batch_loss = compute_probe_variable_loss(
                        se_models[label],
                        field_model,
                        batch_coords,
                        batch_targets,
                        probe_training["value_stats"],
                        dimless,
                        "theta_h",
                    )
                    weight = float(stop - start) / float(total_probe)
                    (segment_probe_weight * batch_loss * weight).backward()
                    probe_sum += float(batch_loss.detach().cpu()) * float(stop - start)
                probe_theta_values[label] = probe_sum / float(total_probe)
        else:
            probe_theta_values = {segment["label"]: 0.0 for segment in segment_references}
        optimizer.step()
        scheduler.step()
        pde_value = float(np.mean(pde_values_for_mean))
        probe_theta_value = float(np.mean([probe_theta_values[label] for label in segment_labels]))
        total_value = (
            float(args.lambda_pde) * pde_value
            + float(args.lambda_param) * float(param_total_loss.detach().cpu())
            + float(args.lambda_probe_theta) * probe_theta_value
        )
        if total_value < best_loss:
            best_loss = total_value
            best_state = {
                "se_model_state_dicts": se_model_state_dicts(se_models),
                "field_model_state_dict": deepcopy(field_model.state_dict()),
            }
            torch.save(
                {
                    **build_checkpoint_payload(
                        args,
                        best_state["se_model_state_dicts"],
                        best_state["field_model_state_dict"],
                        dimless,
                        best_loss,
                        field_global_stats,
                        manifest["segments"],
                        parameter_model_metadata,
                    ),
                    "epoch": epoch,
                },
                checkpoint_dir / "best.pt",
            )
        row = {
            "epoch": epoch,
            "elapsed_min": elapsed_minutes(train_start),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "total": total_value,
            "pde": pde_value,
            "probe_theta": probe_theta_value,
            "param": float(param_total_loss.detach().cpu()),
            "param_alpha": float(param_components["lnalpha"].detach().cpu()),
            "param_n": float(param_components["n"].detach().cpu()),
            "param_theta_s": float(param_components["theta_s"].detach().cpu()),
            "param_theta_r": float(param_components["theta_r"].detach().cpu()),
            "param_lnks": float(param_components["lnks"].detach().cpu()),
        }
        for label in segment_labels:
            row[f"pde_{label}"] = float(pde_losses[label])
            row[f"probe_theta_{label}"] = float(probe_theta_values[label])
        history.append(row)
        last_epoch = epoch
        elapsed_seconds = time.perf_counter() - train_start
        reached_wall_limit = (
            max_wall_seconds is not None
            and elapsed_seconds >= max_wall_seconds
            and (epoch < args.epochs)
        )
        if epoch == 1 or epoch % print_every == 0 or epoch == args.epochs or reached_wall_limit:
            emit_log(format_log_row(row), training_log_path)
        if epoch % checkpoint_every == 0 or epoch == args.epochs or reached_wall_limit:
            torch.save(
                {
                    **build_checkpoint_payload(
                        args,
                        se_model_state_dicts(se_models),
                        deepcopy(field_model.state_dict()),
                        dimless,
                        total_value,
                        field_global_stats,
                        manifest["segments"],
                        parameter_model_metadata,
                    ),
                    "epoch": epoch,
                    "total_loss": total_value,
                },
                checkpoint_dir / f"epoch_{epoch:05d}.pt",
            )
        if comparison_every > 0 and epoch % comparison_every == 0:
            update_fit_error_history_row(
                row,
                se_models[segment_labels[0]],
                field_model,
                segment_references[0]["plane_reference"],
                parameter_evaluation_points,
                dimless,
                args.plane_batch_size,
                device,
            )
            export_epoch_snapshot(
                se_models,
                field_model,
                dimless,
                segment_references,
                parameter_plane_reference,
                snapshot_parameter_evaluation_points,
                args.plane_batch_size,
                args.probe_batch_size,
                device,
                comparison_dir,
                args.probe_plot_seed,
                comparison_every,
                SNAPSHOT_FIGURE_DPI,
                epoch,
                epoch_snapshot_records,
            )
            save_loss_history(
                history, comparison_dir, plot_keys=loss_plot_keys, dpi=SNAPSHOT_FIGURE_DPI
            )
        if reached_wall_limit:
            stop_reason = "max_wall_hours_reached"
            stopped_early = True
            emit_log(
                f"Reached max wall-clock budget ({max_wall_hours:.3f} h) at epoch {epoch}; continuing with final exports.",
                training_log_path,
            )
            break
    if history and "param_field_rel_l2_lnks" not in history[-1]:
        update_fit_error_history_row(
            history[-1],
            se_models[segment_labels[0]],
            field_model,
            segment_references[0]["plane_reference"],
            parameter_evaluation_points,
            dimless,
            args.plane_batch_size,
            device,
        )
    load_se_model_state_dicts(se_models, best_state["se_model_state_dicts"])
    field_model.load_state_dict(best_state["field_model_state_dict"])
    se_models.eval()
    field_model.eval()
    save_loss_history(history, output_dir, plot_keys=loss_plot_keys)
    segment_metrics = {}
    comparison_segments_dir = ensure_dir(comparison_dir / "segments")
    for segment in segment_references:
        label = segment["label"]
        segment_dir = comparison_segments_dir
        plane_metrics = save_state_plane_comparison(
            segment["plane_reference"],
            predict_state_plane_fields(
                se_models[label],
                field_model,
                segment["plane_reference"],
                dimless,
                args.plane_batch_size,
                device,
            ),
            segment_dir,
            plot_times_local_d=state_plot_times_for_segment(segment),
        )
        theta_probe_metrics = save_probe_theta_comparison(
            segment["reference_probe"],
            predict_probe_variable(
                se_models[label],
                field_model,
                segment["reference_probe"],
                dimless,
                args.probe_batch_size,
                device,
                "theta_h",
            ),
            segment_dir,
            args.probe_plot_seed,
        )
        h_probe_metrics = save_probe_variable_comparison(
            segment["h_reference_probe"],
            predict_probe_variable(
                se_models[label],
                field_model,
                segment["h_reference_probe"],
                dimless,
                args.probe_batch_size,
                device,
                "h",
            ),
            segment_dir,
            args.probe_plot_seed,
            variable_name="h",
            file_prefix="probe_h",
            plot_prefix="probe_h",
        )
        segment_metrics[label] = {
            "plane_metrics": plane_metrics,
            "probe_metrics": theta_probe_metrics,
            "theta_probe_metrics": theta_probe_metrics,
            "h_probe_metrics": h_probe_metrics,
        }
    parameter_field_metrics = save_parameter_field_comparison(
        parameter_plane_reference,
        predict_parameter_field_planes(
            field_model, parameter_plane_reference, dimless, args.plane_batch_size, device
        ),
        comparison_dir / "parameter_fit",
    )
    parameter_point_metrics = save_parameter_point_comparison(
        parameter_evaluation_points,
        predict_parameter_points(
            field_model, parameter_evaluation_points, dimless, device, args.plane_batch_size
        ),
        comparison_dir / "parameter_fit",
    )
    best_payload = build_checkpoint_payload(
        args,
        best_state["se_model_state_dicts"],
        best_state["field_model_state_dict"],
        dimless,
        best_loss,
        field_global_stats,
        manifest["segments"],
        parameter_model_metadata,
    )
    torch.save(best_payload, output_dir / "inverse_fieldnet_best.pt")
    torch.save(best_payload, checkpoint_dir / "final_best.pt")
    write_json(
        output_dir / "training_summary.json",
        {
            "project_dir": str(PROJECT_DIR),
            "portable_runtime": True,
            "external_source_dirs_required": False,
            "reproducibility": {
                "seed": int(args.seed),
                "deterministic": bool(args.deterministic),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device": str(device),
                "command": [str(item) for item in sys.argv],
                "training_profile": "shared64_rain1_0_20d_param5x3_probe5x3_noBCIC",
            },
            "reference_manifest": str(reference_manifest_path),
            "segments_dir": str(segments_dir),
            "domain": {
                **manifest.get("domain", DOMAIN),
                "segment_count": len(segment_labels),
                "segment_local_t_end": segment_local_t_end,
            },
            "dimless": dimless,
            "model_type": model_type,
            "activation": activation,
            "parameter_model_variant": parameter_model_metadata["parameter_model_variant"],
            "shared_parameter_network": bool(parameter_model_metadata["shared_parameter_network"]),
            "parameter_output_mode": PARAMETER_OUTPUT_MODE,
            "n_output_mode": N_OUTPUT_MODE,
            "theta_h_loss_normalization": "probe_minmax",
            "theta_noise_std": float(theta_noise_std),
            "theta_noise_seed": int(theta_noise_seed),
            "theta_noise_mode": THETA_NOISE_MODE,
            "theta_noise_clip_range": list(THETA_NOISE_CLIP_RANGE),
            "h_loss_normalization": "probe_minmax",
            "parameter_normalization_source": "selected_parameter_points_expanded_25pct",
            "parameter_normalization_expansion_fraction": PARAMETER_NORMALIZATION_EXPANSION_FRACTION,
            "parameter_output_stats": parameter_output_stats,
            "theta_h_stats_by_segment": {
                segment["label"]: segment["theta_probe_training"]["theta_stats"]
                for segment in segment_references
            },
            "h_stats_by_segment": {
                segment["label"]: segment["h_probe_training"]["h_stats"]
                for segment in segment_references
            },
            "single_forward_network": len(segment_labels) == 1,
            "data_window": str("rain1_0_20d"),
            "available_data_windows": manifest.get("available_data_windows", []),
            "probe_group": theta_probe_group,
            "theta_probe_group": theta_probe_group,
            "theta_probe_count": int(
                segment_references[0]["theta_reference_probe"]["theta_h"].shape[0]
            ),
            "h_probe_group": h_probe_group,
            "h_probe_count": int(segment_references[0]["h_reference_probe"]["h"].shape[0]),
            "parameter_point_set": parameter_point_set,
            "parameter_points_path": str(parameter_points_path),
            "segment_labels": segment_labels,
            "segments": manifest["segments"],
            "epochs": int(args.epochs),
            "n_residual_per_segment": int(args.n_residual),
            "comparison_every": int(args.comparison_every),
            "snapshot_figure_dpi": int(SNAPSHOT_FIGURE_DPI),
            "lambda_pde": float(args.lambda_pde),
            "lambda_probe_theta": float(args.lambda_probe_theta),
            "lambda_param": float(args.lambda_param),
            "lambda_param_alpha": float(args.lambda_param_alpha),
            "lambda_param_n": float(args.lambda_param_n),
            "lambda_param_theta_s": float(args.lambda_param_theta_s),
            "lambda_param_theta_r": float(args.lambda_param_theta_r),
            "lambda_param_lnks": float(args.lambda_param_lnks),
            "lambda_param_ks": float(args.lambda_param_lnks),
            "parameter_loss_field_weights": parameter_loss_field_weights,
            "max_wall_hours": max_wall_hours,
            "lr_decay_count": int(LR_DECAY_COUNT),
            "lr_decay_gamma": float(LR_DECAY_GAMMA),
            "lr_decay_step_size": int(
                max(1, math.ceil(float(args.epochs) / float(LR_DECAY_COUNT)))
            ),
            "stop_reason": stop_reason,
            "stopped_early": bool(stopped_early),
            "last_epoch": int(last_epoch),
            "trainable_parameter_fields": trainable_parameter_fields,
            "trainable_parameter_loss_fields": trainable_parameter_loss_fields,
            "loss_plot_keys": loss_plot_keys,
            "se_hidden_dim": int(UNIFIED_HIDDEN_DIM),
            "se_hidden_layers": int(UNIFIED_HIDDEN_LAYERS),
            "parameter_hidden_dim": int(parameter_model_metadata["parameter_hidden_dim"]),
            "parameter_hidden_layers": int(parameter_model_metadata["parameter_hidden_layers"]),
            "parameter_head_count": int(parameter_model_metadata["parameter_head_count"]),
            "se_parameter_count_total": int(se_params),
            "field_parameter_count": int(field_params),
            "total_trainable_parameters": int(total_params),
            "probe_count": int(segment_references[0]["theta_reference_probe"]["theta_h"].shape[0]),
            "parameter_point_count": int(parameter_points["coords_phys"].shape[0]),
            "parameter_evaluation_point_count": int(
                parameter_evaluation_points["coords_phys"].shape[0]
            ),
            "parameter_evaluation_metadata": parameter_evaluation_points.get(
                "evaluation_metadata", {}
            ),
            "snapshot_parameter_evaluation_point_count": int(
                snapshot_parameter_evaluation_points["coords_phys"].shape[0]
            ),
            "snapshot_parameter_evaluation_metadata": snapshot_parameter_evaluation_points.get(
                "evaluation_metadata", {}
            ),
            "rain1_state_plot_times_local_d": list(RAIN1_STATE_PLOT_TIMES_LOCAL_D),
            "soil_sample_location_count": int(manifest.get("soil_sample_location_count", 0)),
            "parameter_supervision_truth": manifest.get("parameter_supervision_truth"),
            "best_total_loss": float(best_loss),
            "print_every": int(print_every),
            "training_log": str(training_log_path),
            "loss_history_csv": str(output_dir / "loss_history.csv"),
            "loss_history_png": str(output_dir / "loss_history.png"),
            "epoch_snapshot_manifest": str(comparison_dir / "epoch_snapshot_manifest.json"),
            "comparison_loss_history_csv": str(comparison_dir / "loss_history.csv"),
            "comparison_loss_curve_file": str(comparison_dir / "loss_history.png"),
            "comparison_segments_dir": str((comparison_dir / "segments").resolve()),
            "comparison_parameter_fit_dir": str((comparison_dir / "parameter_fit").resolve()),
            "segment_metrics": segment_metrics,
            "parameter_field_metrics": parameter_field_metrics,
            "parameter_point_metrics": parameter_point_metrics,
        },
    )
    emit_log(f"Training finished. Outputs -> {output_dir}", training_log_path)
    return output_dir


def main():
    args = build_parser().parse_args()
    try:
        train(args)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
