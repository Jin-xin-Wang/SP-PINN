import argparse
import json
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
    AVAILABLE_MODEL_TYPES,
    InverseFieldModelBundle,
    MINMAX_PARAMETER_FIELDS,
    N_OUTPUT_MODE,
    PARAMETER_NORMALIZATION_EXPANSION_FRACTION,
    PARAMETER_LOSS_FIELDS,
    PARAMETER_OUTPUT_MODE,
    PARAMETER_MODEL_CHOICES,
    RichardsPINN,
    UNIFIED_HIDDEN_DIM,
    UNIFIED_HIDDEN_LAYERS,
    normalize_activation_name,
    normalize_model_type,
)
from physical import (
    boundary_dimless_ranges,
    compute_bottom_free_drainage_loss,
    # Initial-head routines are preserved in physical.py but intentionally inactive
    # in the current theta-only initial-condition experiments.
    # compute_initial_head_and_theta_state_losses,
    # compute_initial_head_loss,
    compute_initial_theta_state_loss,
    compute_left_no_flow_loss,
    compute_parameter_point_losses,
    compute_pde_loss,
    compute_probe_theta_loss,
    compute_probe_variable_loss,
    compute_right_no_flow_loss,
    compute_state,
    compute_top_flux_loss,
    evaluate_parameter_fields,
    sample_bottom_boundary_points,
    sample_left_boundary_points,
    sample_physical_bottom_boundary_points,
    # sample_physical_initial_condition_points,
    sample_physical_top_boundary_points,
    sample_right_boundary_points,
    sample_top_boundary_points,
    sample_training_points,
)
from training_constraints import ConstraintRuntime, constraint_plan_metadata, run_segment_constraint
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
DEFAULT_PARAMETER_POINTS = REFERENCE_DATA_DIR / "parameter_points_regular_5x3" / "parameter_point_targets.npz"
DEFAULT_PARAMETER_POINT_SET = "regular_5x3"
DEFAULT_THETA_PROBE_GROUP = "probes_regular_5x3"
DEFAULT_H_PROBE_GROUP = "probes_regular_5x3"
DEFAULT_PROBE_GROUP = DEFAULT_THETA_PROBE_GROUP
DEFAULT_KNOWN_FIELD_MODEL_DIR = (PROJECT_DIR / "field_models").resolve()
LR_DECAY_COUNT = 20
LR_DECAY_GAMMA = 0.9
TIME_CAUSAL_EPOCH_BLOCK = 10000
TIME_CAUSAL_INCREMENT_DAY = 2.0
TIME_CAUSAL_MAX_DAY = 10.0
SNAPSHOT_FIGURE_DPI = 120
SNAPSHOT_PARAMETER_GRID_SHAPE_XZ = (50, 30)
RAIN1_STATE_PLOT_TIMES_LOCAL_D = (1.0, 3.0, 7.0, 12.0, 20.0)
THETA_NOISE_ALLOWED_STD = (0.0, 0.02)
THETA_NOISE_DEFAULT_SEED = 20260526
THETA_NOISE_MODE = "fixed_white_gaussian_train_target_only"
THETA_NOISE_CLIP_RANGE = (0.0, 1.0)

DOMAIN = {
    "x_min": 0.0,
    "x_max": 100.0,
    "z_min": -0.6,
    "z_max": 0.0,
    "t_end": 10.0,
}

LOSS_WEIGHTS = {
    "pde": 10.0,
    "top": 100.0,
    "bottom": 10.0,
    "left": 10.0,
    "right": 10.0,
    "probe_theta_h": 1.0,
    "probe_h": 0.0,
    "initial_theta": 0.0,
    "initial_head": 0.0,
    "initial_theta_state": 1.0,
    "param": 0.1,
    "param_alpha": 1.0,
    "param_n": 1.0,
    "param_theta_s": 1.0,
    "param_theta_r": 1.0,
    "param_lnks":1.0,
}
STATE_ERROR_VARIABLES = ("h", "k_h", "se", "theta_h")
STATE_ERROR_TIMES_LOCAL_D = (1.0, 5.0, 10.0)
OBSERVATION_VARIABLES = ("theta_h", "h")
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


def build_constraint_execution_plan(args):
    return {
        "top": ConstraintRuntime.sampled("top", args.lambda_top, args.n_top),
        "bottom": ConstraintRuntime.sampled("bottom", args.lambda_bottom, args.n_bottom),
        "left": ConstraintRuntime.sampled("left", args.lambda_left, args.n_left),
        "right": ConstraintRuntime.sampled("right", args.lambda_right, args.n_right),
        "initial_theta": ConstraintRuntime.data_backed(
            "initial_theta",
            args.lambda_initial_theta,
            bool(args.initial_theta_group),
            sample_count=args.n_initial_theta,
        ),
        # Keep the CLI values auditable, while preventing this retained historical
        # constraint from sampling points or entering the backward graph.
        "initial_head": ConstraintRuntime(
            name="initial_head",
            weight=float(args.lambda_initial_head),
            sample_count=int(args.n_initial_head),
            enabled=False,
            disabled_reason="commented_out_theta_only_experiment",
            activation_rule="temporarily commented out; use initial_theta_state only",
        ),
        "initial_theta_state": ConstraintRuntime.sampled(
            "initial_theta_state",
            args.lambda_initial_theta_state,
            args.n_initial_theta_state,
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the shared-parameter 2D SP-PINN for the 5x3/5x3 rain1 ICBC case."
    )
    parser.add_argument("--model-type", default="baseline", choices=AVAILABLE_MODEL_TYPES)
    parser.add_argument("--activation", default="sin", choices=("tanh", "silu", "sin"))
    parser.add_argument("--parameter-model", default="shared64", choices=PARAMETER_MODEL_CHOICES)
    parser.add_argument("--data-window", default="rain1_0_20d")
    parser.add_argument("--reference-manifest", default=str(DEFAULT_REFERENCE_MANIFEST))
    parser.add_argument("--segments-dir", default=str(DEFAULT_SEGMENTS_DIR))
    parser.add_argument("--parameter-field-dir", default=str(DEFAULT_PARAMETER_FIELD_DIR))
    parser.add_argument("--parameter-points", default=str(DEFAULT_PARAMETER_POINTS))
    parser.add_argument("--probe-group", default=None)
    parser.add_argument("--theta-probe-group", default=DEFAULT_THETA_PROBE_GROUP)
    parser.add_argument("--h-probe-group", default=DEFAULT_H_PROBE_GROUP)
    parser.add_argument("--parameter-point-set", default=DEFAULT_PARAMETER_POINT_SET)
    parser.add_argument("--kriging-metrics", default=None)
    parser.add_argument("--known-field-model-dir", default=str(DEFAULT_KNOWN_FIELD_MODEL_DIR))
    parser.add_argument("--known-ks", type=int, default=0, choices=(0, 1))
    parser.add_argument("--known-alpha", type=int, default=0, choices=(0, 1))
    parser.add_argument("--known-n", type=int, default=0, choices=(0, 1))
    parser.add_argument("--known-theta-s", type=int, default=0, choices=(0, 1))
    parser.add_argument("--known-theta-r", type=int, default=0, choices=(0, 1))
    parser.add_argument("--epochs", type=int, default=300006)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--n-residual", type=int, default=100000)
    parser.add_argument("--n-top", type=int, default=0)
    parser.add_argument("--n-bottom", type=int, default=0)
    parser.add_argument("--n-left", type=int, default=0)
    parser.add_argument("--n-right", type=int, default=0)
    parser.add_argument("--physical-boundary-frame", default="comsol", choices=("roi", "comsol"))
    parser.add_argument("--lambda-pde", type=float, default=LOSS_WEIGHTS["pde"])
    parser.add_argument("--lambda-top", type=float, default=0.0)
    parser.add_argument("--lambda-bottom", type=float, default=0.0)
    parser.add_argument("--lambda-left", type=float, default=0.0)
    parser.add_argument("--lambda-right", type=float, default=0.0)
    parser.add_argument("--lambda-probe-theta", type=float, default=LOSS_WEIGHTS["probe_theta_h"])
    parser.add_argument("--lambda-probe-h", type=float, default=0.0)
    parser.add_argument("--initial-theta-group", default=None)
    parser.add_argument("--lambda-initial-theta", type=float, default=0.0)
    parser.add_argument("--n-initial-theta", type=int, default=0)
    parser.add_argument("--lambda-initial-head", type=float, default=0.0)
    parser.add_argument("--n-initial-head", type=int, default=0)
    parser.add_argument("--lambda-initial-theta-state", type=float, default=0.0)
    parser.add_argument("--n-initial-theta-state", type=int, default=0)
    parser.add_argument("--initial-total-head-m", type=float, default=-5.0)
    parser.add_argument("--lambda-param", type=float, default=LOSS_WEIGHTS["param"])
    parser.add_argument("--lambda-param-alpha", type=float, default=LOSS_WEIGHTS["param_alpha"])
    parser.add_argument("--lambda-param-n", type=float, default=LOSS_WEIGHTS["param_n"])
    parser.add_argument("--lambda-param-theta-s", type=float, default=LOSS_WEIGHTS["param_theta_s"])
    parser.add_argument("--lambda-param-theta-r", type=float, default=LOSS_WEIGHTS["param_theta_r"])
    parser.add_argument("--lambda-param-lnks", type=float, default=LOSS_WEIGHTS["param_lnks"])
    parser.add_argument("--theta-noise-std", type=float, default=0.0)
    parser.add_argument("--theta-noise-seed", type=int, default=THETA_NOISE_DEFAULT_SEED)
    parser.add_argument("--time-causal", action="store_true")
    parser.add_argument("--probe-batch-size", type=int, default=20480)
    parser.add_argument("--plane-batch-size", type=int, default=4096)
    parser.add_argument("--probe-plot-seed", type=int, default=42)
    parser.add_argument("--comparison-every", type=int, default=20000)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument(
        "--with-icbc",
        action="store_true",
        help="Enable all four BC constraints and the full-domain initial theta constraint.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic torch kernels for cross-run reproducibility.",
    )
    return parser


def apply_icbc_defaults(args):
    """Activate the paper ICBC constraints without changing the noICBC default."""
    if not bool(args.with_icbc):
        return args
    args.lambda_top = 100.0
    args.lambda_bottom = 10.0
    args.lambda_left = 10.0
    args.lambda_right = 10.0
    args.n_top = 10000
    args.n_bottom = 10000
    args.n_left = 10000
    args.n_right = 10000
    args.lambda_initial_theta_state = 1.0
    args.n_initial_theta_state = 10000
    return args

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


def load_optional_json(path_text, base_dir=None):
    if not path_text:
        return {}, None
    path = resolve_portable_path(path_text, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path


def _metric_aliases_for_field(field_name):
    return {
        "lnalpha": ("lnalpha",),
        "lnn1": ("lnn1", "ln_n_minus_1"),
        "theta_s": ("theta_s", "logit_theta_s"),
        "theta_r": ("theta_r", "logit_theta_r_over_theta_s"),
        "lnks": ("lnks", "Ks"),
    }.get(field_name, (field_name,))


def extract_rel_l2_by_field(metrics_payload):
    rel_l2 = {}
    if not isinstance(metrics_payload, dict):
        return rel_l2
    nested = metrics_payload.get("metrics")
    for field_name in PARAMETER_FIELD_VARIABLES:
        value = None
        for alias in _metric_aliases_for_field(field_name):
            direct = metrics_payload.get(alias)
            if isinstance(direct, dict) and "rel_l2" in direct:
                value = direct["rel_l2"]
                break
        if value is None and isinstance(nested, dict):
            for alias in _metric_aliases_for_field(field_name):
                item = nested.get(alias)
                if isinstance(item, dict) and "rel_l2" in item:
                    value = item["rel_l2"]
                    break
        if value is not None:
            rel_l2[field_name] = float(value)
    return rel_l2


def infer_parameter_point_set(parameter_points_path, explicit_name=None):
    if explicit_name:
        return explicit_name
    parent_name = Path(parameter_points_path).parent.name
    if "3x3" in parent_name:
        return "regular_3x3"
    if "5x3" in parent_name:
        return "regular_5x3"
    if "9x3" in parent_name:
        return "regular_9x3"
    if "15x3" in parent_name:
        return "regular_15x3"
    if parent_name == "parameter_points":
        return DEFAULT_PARAMETER_POINT_SET
    return parent_name


def build_known_field_flags(args):
    return {
        "ks": bool(int(args.known_ks)),
        "alpha": bool(int(args.known_alpha)),
        "n": bool(int(args.known_n)),
        "theta_s": bool(int(args.known_theta_s)),
        "theta_r": bool(int(args.known_theta_r)),
    }


def build_loss_plot_keys(args, segment_labels, active_parameter_loss_fields):
    keys = ["total"]
    if float(args.lambda_pde) > 0.0 and int(args.n_residual) > 0:
        keys.append("pde")
        keys.extend(f"pde_{label}" for label in segment_labels)
    if float(args.lambda_top) > 0.0 and int(args.n_top) > 0:
        keys.append("top")
        keys.extend(f"top_{label}" for label in segment_labels)
    if float(args.lambda_bottom) > 0.0 and int(args.n_bottom) > 0:
        keys.append("bottom")
        keys.extend(f"bottom_{label}" for label in segment_labels)
    if float(args.lambda_left) > 0.0 and int(args.n_left) > 0:
        keys.append("left")
        keys.extend(f"left_{label}" for label in segment_labels)
    if float(args.lambda_right) > 0.0 and int(args.n_right) > 0:
        keys.append("right")
        keys.extend(f"right_{label}" for label in segment_labels)
    if float(args.lambda_probe_theta) > 0.0:
        keys.append("probe_theta")
        keys.extend(f"probe_theta_{label}" for label in segment_labels)
    if float(args.lambda_probe_h) > 0.0:
        keys.append("probe_h")
        keys.extend(f"probe_h_{label}" for label in segment_labels)
    if float(args.lambda_initial_theta) > 0.0 and args.initial_theta_group:
        keys.append("initial_theta")
    # Initial-head loss is intentionally inactive in the current training chain.
    # Keep the history row field at 0.0 but do not plot a non-existent loss curve.
    if float(args.lambda_initial_theta_state) > 0.0 and int(args.n_initial_theta_state) > 0:
        keys.append("initial_theta_state")
    if float(args.lambda_param) > 0.0 and active_parameter_loss_fields:
        keys.append("param")
        keys.extend(LOSS_COMPONENT_KEY_BY_FIELD[field] for field in PARAMETER_LOSS_FIELDS if field in active_parameter_loss_fields)
    return keys

def time_causal_residual_cap_day(epoch):
    if int(epoch) <= 0:
        return 0.0
    stage = int((int(epoch) - 1) // TIME_CAUSAL_EPOCH_BLOCK) + 1
    return float(min(TIME_CAUSAL_MAX_DAY, stage * TIME_CAUSAL_INCREMENT_DAY))


def build_lr_scheduler(optimizer, epochs):
    step_size = max(1, int(math.ceil(float(epochs) / float(LR_DECAY_COUNT))))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=LR_DECAY_GAMMA)


def validate_theta_noise_std(value):
    value = float(value)
    for allowed in THETA_NOISE_ALLOWED_STD:
        if abs(value - allowed) <= 1.0e-12:
            return float(allowed)
    choices = ", ".join(f"{allowed:g}" for allowed in THETA_NOISE_ALLOWED_STD)
    raise ValueError(f"--theta-noise-std must be one of {{{choices}}}; got {value:g}.")


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
    dimless["Cx"] = dimless["K0"] * dimless["T"] * dimless["H0"] / (
        dimless["Theta0"] * dimless["sx"] ** 2
    )
    dimless["Cz"] = dimless["K0"] * dimless["T"] / (dimless["Theta0"] * dimless["Lz"])
    return dimless


def lookup_top_flux_m_per_day(boundary, physical_time_day):
    schedule = boundary.get("rainfall_events") or boundary.get("top_flux_schedule") or []
    for item in schedule:
        start_day = float(item.get("start_day", 0.0))
        duration_day = float(item.get("duration_day", max(float(item.get("end_day", start_day)) - start_day, 0.0)))
        end_day = float(item.get("end_day", start_day + duration_day))
        if start_day <= float(physical_time_day) <= end_day and duration_day > 0.0:
            if "qmax_m_per_day" in item:
                phase = (float(physical_time_day) - start_day) / duration_day
                return float(item["qmax_m_per_day"]) * math.sin(math.pi * min(max(phase, 0.0), 1.0))
            return float(item.get("q_top_m_per_day", item.get("q_in", 0.0)))
    return float(boundary.get("q_in", 0.0))


def top_flux_star_for_points(manifest, segment_spec, top_points_dim, dimless):
    boundary = manifest.get("boundary", {})
    events = boundary.get("rainfall_events") or boundary.get("top_flux_schedule") or []
    physical_time_day = float(segment_spec.get("physical_start_day", 0.0)) + top_points_dim[:, 2:3] * float(dimless["T"])
    q_top = torch.zeros_like(physical_time_day)
    for item in events:
        start_day = float(item.get("start_day", 0.0))
        duration_day = float(item.get("duration_day", max(float(item.get("end_day", start_day)) - start_day, 0.0)))
        if duration_day <= 0.0:
            continue
        phase = (physical_time_day - start_day) / duration_day
        active = (phase >= 0.0) & (phase <= 1.0)
        if "qmax_m_per_day" in item:
            value = float(item["qmax_m_per_day"]) * torch.sin(float(math.pi) * torch.clamp(phase, 0.0, 1.0))
        else:
            value = torch.full_like(q_top, float(item.get("q_top_m_per_day", item.get("q_in", 0.0))))
        q_top = torch.where(active, q_top + value, q_top)
    return q_top / float(dimless["K0"])


def build_segment_top_flux_targets(manifest, dimless, enabled=False):
    boundary = manifest.get("boundary", {})
    targets = {}
    mode = "pointwise_half_sine_from_manifest" if enabled else "disabled_by_default"
    for spec in manifest.get("segments", []):
        label = spec["label"]
        start_day = float(spec.get("physical_start_day", 0.0))
        end_day = float(spec.get("physical_end_day", start_day))
        midpoint_day = 0.5 * (start_day + end_day)
        q_top_m_per_day = lookup_top_flux_m_per_day(boundary, midpoint_day)
        targets[label] = {
            "q_top_m_per_day": q_top_m_per_day,
            "q_top_star": q_top_m_per_day / float(dimless["K0"]),
            "mode": mode,
            "physical_start_day": start_day,
            "physical_end_day": end_day,
            "physical_midpoint_day": midpoint_day,
        }
    return targets


def build_output_dir(model_type, epochs, data_window, user_output_dir):
    if user_output_dir:
        return resolve_project_path(user_output_dir)
    return PROJECT_DIR / "outputs" / f"formal_{model_type}_inverse_fieldnet_{data_window}_e{int(epochs)}"


def select_data_window_manifest(manifest, data_window):
    segments = list(manifest.get("segments", []))
    available = [spec.get("label") for spec in segments]
    selected_segments = [spec for spec in segments if spec.get("label") == data_window]
    if len(selected_segments) != 1:
        aliases = sorted((manifest.get("data_window_aliases") or {}).keys())
        alias_text = f"; aliases for joint selection: {aliases}" if aliases else ""
        raise ValueError(f"Unknown --data-window {data_window!r}. Available windows: {available}{alias_text}")
    selected = deepcopy(manifest)
    selected["all_segments"] = segments
    selected["segments"] = selected_segments
    selected["segment_count"] = 1
    selected["selected_data_window"] = data_window
    selected["available_data_windows"] = manifest.get("available_data_windows", available)
    selected["local_domain"] = {"t_end_day": float(selected_segments[0].get("local_t_end_day", DOMAIN["t_end"]))}
    return selected


def apply_probe_group_overrides(manifest, theta_probe_group=None, h_probe_group=None, legacy_probe_group=None):
    available = manifest.get("probe_groups", {})
    default_theta_group = manifest.get(
        "theta_probe_group_dir_name",
        manifest.get("probe_group_dir_name", DEFAULT_THETA_PROBE_GROUP),
    )
    default_h_group = manifest.get("h_probe_group_dir_name", DEFAULT_H_PROBE_GROUP)
    if legacy_probe_group and theta_probe_group and str(legacy_probe_group) != str(theta_probe_group):
        raise ValueError(
            f"Conflicting theta probe groups: --probe-group={legacy_probe_group!r} and "
            f"--theta-probe-group={theta_probe_group!r}"
        )
    theta_group = str(theta_probe_group or legacy_probe_group or default_theta_group)
    h_group = str(h_probe_group or default_h_group)
    if available and theta_group not in available:
        raise ValueError(f"Unknown theta probe group {theta_group!r}. Available probe groups: {sorted(available)}")
    if available and h_group not in available:
        raise ValueError(f"Unknown h probe group {h_group!r}. Available probe groups: {sorted(available)}")
    selected = deepcopy(manifest)
    selected["probe_group"] = theta_group
    selected["probe_group_dir_name"] = theta_group
    selected["theta_probe_group"] = theta_group
    selected["theta_probe_group_dir_name"] = theta_group
    selected["h_probe_group"] = h_group
    selected["h_probe_group_dir_name"] = h_group
    for spec in selected.get("segments", []):
        spec["probe_group"] = theta_group
        spec["theta_probe_group"] = theta_group
        spec["h_probe_group"] = h_group
    return selected


def local_parameter_points_path(set_name):
    candidates = [
        REFERENCE_DATA_DIR / f"parameter_points_{set_name}" / "parameter_point_targets.npz",
    ]
    if str(set_name) == "regular_5x3":
        candidates.append(REFERENCE_DATA_DIR / "parameter_points" / "parameter_point_targets.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_parameter_points_from_manifest(manifest, args, manifest_path=None):
    if args.parameter_point_set:
        set_name = str(args.parameter_point_set)
        sets = manifest.get("parameter_point_sets", {})
        if set_name not in sets:
            raise ValueError(
                f"Unknown --parameter-point-set {set_name!r}. Available sets: {sorted(sets)}"
            )
        npz_path = sets[set_name].get("npz") or sets[set_name].get("path")
        if not npz_path:
            return local_parameter_points_path(set_name), set_name
        path = resolve_portable_path(npz_path, base_dir=None if manifest_path is None else Path(manifest_path).parent)
        if path.is_dir():
            path = path / "parameter_point_targets.npz"
        if not path.exists():
            path = local_parameter_points_path(set_name)
        return path.resolve(), set_name
    path = resolve_portable_path(args.parameter_points)
    return path, infer_parameter_point_set(path, None)


def resolve_default_kriging_metrics_path(manifest, parameter_point_set, manifest_path=None):
    local_path = (
        REFERENCE_DATA_DIR
        / "ordinary_kriging_baselines"
        / str(parameter_point_set)
        / "parameter_field_metrics.json"
    )
    if local_path.exists():
        return str(local_path)
    path_text = (
        manifest.get("parameter_point_sets", {})
        .get(parameter_point_set, {})
        .get("kriging_parameter_field_metrics")
    )
    if not path_text:
        return None
    return str(resolve_portable_path(path_text, base_dir=None if manifest_path is None else Path(manifest_path).parent))


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


def build_probe_training_tensors(
    reference_probe,
    dimless,
    device,
    observed_variable="theta_h",
    theta_noise_std=0.0,
    theta_noise_seed=None,
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
    safe = np.maximum(value, 1.0e-12)
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
        expanded_span = float(max(observed_range, 1.0e-6))
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
                else inverse_softplus_numpy(np.asarray(parameter_points["n"], dtype=np.float64) - 1.0)
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
                    coords_t = torch.tensor(coords_dim[start:stop], dtype=torch.float32, device=device)
                    state = compute_state(se_model, field_model, coords_t, dimless, require_spatial=True)
                    for var_name in plane_ref["fields"].keys():
                        chunks[var_name].append(state[var_name].detach().cpu().numpy())
            for var_name in plane_ref["fields"].keys():
                plane_pred["fields"][var_name].append(
                    np.concatenate(chunks[var_name], axis=0).reshape(
                        len(plane_ref["coord2_1d"]),
                        len(plane_ref["coord1_1d"]),
                    ).astype(np.float32)
                )
        for var_name in plane_ref["fields"].keys():
            plane_pred["fields"][var_name] = np.stack(plane_pred["fields"][var_name], axis=0)
        predictions[plane_name] = plane_pred
    return predictions


def predict_probe_variable(se_model, field_model, reference_probe, dimless, batch_size, device, observed_variable="theta_h"):
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


def predict_probe_theta(se_model, field_model, reference_probe, dimless, batch_size, device):
    return predict_probe_variable(se_model, field_model, reference_probe, dimless, batch_size, device, "theta_h")


def predict_parameter_field_planes(field_model, parameter_plane_reference, dimless, batch_size, device):
    predictions = {"planes": {}}
    field_model.eval()
    for plane_name, plane_ref in parameter_plane_reference["planes"].items():
        coords_phys = build_static_plane_coords(
            plane_name,
            plane_ref["coord1_1d"],
            plane_ref["coord2_1d"],
            plane_ref["fixed_value"],
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
            predictions["planes"][plane_name]["fields"][field_name] = np.concatenate(chunks[field_name], axis=0).reshape(
                len(plane_ref["coord2_1d"]),
                len(plane_ref["coord1_1d"]),
            ).astype(np.float32)
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
                chunks[field_name].append(local[field_name].detach().cpu().numpy().astype(np.float32))
    return {field_name: np.concatenate(chunks[field_name], axis=0) for field_name in PARAMETER_POINT_VARIABLES}


def find_time_index(times_d, target_time_d):
    for index, time_d in enumerate(np.asarray(times_d, dtype=np.float64).reshape(-1)):
        if np.isclose(float(time_d), float(target_time_d), atol=1.0e-6):
            return int(index)
    raise ValueError(f"Representative local time {target_time_d}d not found in plane reference times {list(times_d)}")


def collect_parameter_fit_rel_l2(field_model, parameter_evaluation_points, dimless, device, batch_size):
    predictions = predict_parameter_points(
        field_model,
        parameter_evaluation_points,
        dimless,
        device,
        batch_size,
    )
    return {
        field_name: float(
            metrics_for_pair(parameter_evaluation_points[field_name], predictions[field_name])["rel_l2"]
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
    predictions = predict_state_plane_fields(se_model, field_model, plane_reference, dimless, batch_size, device)
    ref_xz = plane_reference["planes"]["xz"]
    pred_xz = predictions["xz"]
    available_times = np.asarray(plane_reference["times_d"], dtype=np.float64).reshape(-1)
    time_indices = []
    for time_d in representative_times_local_d:
        matches = np.where(np.isclose(available_times, float(time_d), atol=1.0e-6))[0]
        if matches.size:
            time_indices.append(int(matches[0]))
    if not time_indices:
        time_indices = list(range(len(available_times)))
    state_rel_l2 = {}
    for variable in STATE_ERROR_VARIABLES:
        if variable not in ref_xz["fields"] or variable not in pred_xz["fields"]:
            continue
        rel_l2_values = [
            float(metrics_for_pair(ref_xz["fields"][variable][idx], pred_xz["fields"][variable][idx])["rel_l2"])
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
        field_model,
        parameter_evaluation_points,
        dimless,
        device,
        batch_size,
    )
    state_rel_l2 = collect_state_fit_rel_l2(
        se_model,
        field_model,
        plane_reference,
        dimless,
        batch_size,
        device,
    )
    for field_name, value in parameter_rel_l2.items():
        row[f"param_field_rel_l2_{field_name}"] = float(value)
    for variable, value in state_rel_l2.items():
        row[f"state_rel_l2_{variable}"] = float(value)


def load_segment_references(manifest, segments_dir, dimless, device, theta_noise_std=0.0, theta_noise_seed=THETA_NOISE_DEFAULT_SEED):
    segments = []
    theta_probe_group_dir_name = manifest.get(
        "theta_probe_group_dir_name",
        manifest.get("probe_group_dir_name", manifest.get("probe_group", DEFAULT_THETA_PROBE_GROUP)),
    )
    h_probe_group_dir_name = manifest.get("h_probe_group_dir_name", manifest.get("h_probe_group", DEFAULT_H_PROBE_GROUP))
    for segment_index, spec in enumerate(manifest["segments"]):
        label = spec["label"]
        segment_root = Path(segments_dir) / label
        plane_dir = segment_root / "planes"
        theta_probe_dir = segment_root / theta_probe_group_dir_name
        h_probe_dir = segment_root / h_probe_group_dir_name
        if not plane_dir.exists():
            raise FileNotFoundError(f"Segment plane reference dir not found: {plane_dir}")
        if not theta_probe_dir.exists():
            raise FileNotFoundError(f"Segment theta_h probe reference dir not found: {theta_probe_dir}")
        if not h_probe_dir.exists():
            raise FileNotFoundError(f"Segment h probe reference dir not found: {h_probe_dir}")
        plane_reference = load_plane_reference_data(
            plane_dir,
            spec["snapshot_times_local_d"],
            manifest.get("plane_definitions"),
            physical_times_d=spec.get("snapshot_times_physical_d"),
        )
        theta_reference_probe = load_probe_dataset(theta_probe_dir)
        h_reference_probe = load_probe_dataset(h_probe_dir)
        segments.append(
            {
                "label": label,
                "spec": spec,
                "probe_group": theta_probe_group_dir_name,
                "theta_probe_group": theta_probe_group_dir_name,
                "h_probe_group": h_probe_group_dir_name,
                "plane_reference": plane_reference,
                "reference_probe": theta_reference_probe,
                "theta_reference_probe": theta_reference_probe,
                "h_reference_probe": h_reference_probe,
                "probe_training": build_probe_training_tensors(
                    theta_reference_probe,
                    dimless,
                    device,
                    "theta_h",
                    theta_noise_std=theta_noise_std,
                    theta_noise_seed=int(theta_noise_seed) + segment_index,
                ),
                "theta_probe_training": build_probe_training_tensors(
                    theta_reference_probe,
                    dimless,
                    device,
                    "theta_h",
                    theta_noise_std=theta_noise_std,
                    theta_noise_seed=int(theta_noise_seed) + segment_index,
                ),
                "h_probe_training": build_probe_training_tensors(h_reference_probe, dimless, device, "h"),
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
    kriging_metrics=None,
    kriging_metrics_path=None,
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
            se_model,
            field_model,
            segment["plane_reference"],
            dimless,
            plane_batch_size,
            device,
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
        field_model,
        parameter_plane_reference,
        dimless,
        plane_batch_size,
        device,
    )
    save_parameter_field_comparison(
        parameter_plane_reference,
        parameter_field_predictions,
        epoch_parameter_fit_dir,
        kriging_metrics=kriging_metrics,
        kriging_metrics_path=kriging_metrics_path,
        dpi=figure_dpi,
    )
    parameter_point_predictions = predict_parameter_points(
        field_model,
        snapshot_parameter_evaluation_points,
        dimless,
        device,
        plane_batch_size,
    )
    save_parameter_point_comparison(
        snapshot_parameter_evaluation_points,
        parameter_point_predictions,
        epoch_parameter_fit_dir,
        kriging_metrics=kriging_metrics,
        kriging_metrics_path=kriging_metrics_path,
        dpi=figure_dpi,
    )

    record = {
        "epoch": int(epoch),
        "figure_dpi": int(figure_dpi),
        "parameter_point_evaluation_metadata": snapshot_parameter_evaluation_points.get("evaluation_metadata", {}),
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
            "parameter_point_evaluation_metadata": snapshot_parameter_evaluation_points.get("evaluation_metadata", {}),
            "exports": snapshot_records,
        },
    )
    for label, was_training in se_training_flags.items():
        if was_training:
            se_models[label].train()
    if field_was_training:
        field_model.train()
    return record


def load_initial_theta_training(manifest, group_name, dimless, device, n_initial_theta=0):
    if not group_name:
        return None
    groups = manifest.get("initial_theta_groups", {}) or {}
    if group_name not in groups:
        raise ValueError(f"Unknown --initial-theta-group {group_name!r}. Available groups: {sorted(groups)}")
    npz_path = resolve_portable_path(groups[group_name].get("npz") or (Path(groups[group_name].get("path", "")) / "initial_theta_targets.npz"))
    if not npz_path.exists():
        raise FileNotFoundError(f"Initial theta target npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        coords_phys = data["coords_phys"].astype(np.float32)
        local_time = data["local_time_day"].astype(np.float32).reshape(-1, 1) if "local_time_day" in data.files else np.zeros((coords_phys.shape[0], 1), dtype=np.float32)
        theta = data["theta_h"].astype(np.float32).reshape(-1, 1)
        clean = data["theta_h_clean"].astype(np.float32).reshape(-1, 1) if "theta_h_clean" in data.files else theta.copy()
    coords3 = np.column_stack([coords_phys[:, 0], coords_phys[:, 1], local_time.reshape(-1)]).astype(np.float32)
    if int(n_initial_theta) > 0 and int(n_initial_theta) < coords3.shape[0]:
        rng = np.random.default_rng(20260526)
        idx = np.sort(rng.choice(coords3.shape[0], size=int(n_initial_theta), replace=False))
        coords3 = coords3[idx]
        theta = theta[idx]
        clean = clean[idx]
    coords_dim = phys_to_dim_coords_numpy(coords3, dimless)
    return {
        "group": group_name,
        "npz_path": str(npz_path),
        "point_count": int(coords3.shape[0]),
        "coords_dim_t": torch.tensor(coords_dim, dtype=torch.float32, device=device),
        "theta_targets_t": torch.tensor(theta, dtype=torch.float32, device=device),
        "theta_clean_t": torch.tensor(clean, dtype=torch.float32, device=device),
        "theta_target_min": float(np.min(theta)),
        "theta_target_max": float(np.max(theta)),
    }


def compute_initial_theta_loss(se_model, field_model, initial_training, dimless):
    state = compute_state(se_model, field_model, initial_training["coords_dim_t"], dimless, require_spatial=False)
    return torch.mean((state["theta_h"] - initial_training["theta_targets_t"]) ** 2)


def load_initial_theta_state_training(manifest, dimless, device):
    """Load the embedded full-domain COMSOL t=0 theta field only when requested."""

    spec = manifest.get("initial_theta_state_full_domain")
    if not isinstance(spec, dict):
        raise ValueError(
            "Full-domain initial theta state is not configured in input_manifest.json. "
            "Run build_initial_state_targets.py before enabling --lambda-initial-theta-state."
        )
    npz_path = resolve_portable_path(spec.get("npz", ""))
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Full-domain initial theta state NPZ not found: {npz_path}. "
            "Run build_initial_state_targets.py before enabling this constraint."
        )
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"coords_phys", "theta_h", "x_values_m", "z_values_m"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Initial theta state NPZ is missing fields {missing}: {npz_path}")
        coords_phys = data["coords_phys"].astype(np.float32)
        theta = data["theta_h"].astype(np.float32).reshape(-1, 1)
        x_values = data["x_values_m"].astype(np.float32)
        z_values = data["z_values_m"].astype(np.float32)
    if coords_phys.ndim != 2 or coords_phys.shape[1] != 2 or theta.shape[0] != coords_phys.shape[0]:
        raise ValueError(f"Invalid initial theta state array shapes in {npz_path}.")
    expected_count = int(spec.get("point_count", coords_phys.shape[0]))
    if coords_phys.shape[0] != expected_count:
        raise ValueError(
            f"Initial theta state point count mismatch: manifest={expected_count}, npz={coords_phys.shape[0]}."
        )
    expected_shape = tuple(spec.get("grid_shape_xz", (x_values.size, z_values.size)))
    if expected_shape != (int(x_values.size), int(z_values.size)):
        raise ValueError(
            "Initial theta state grid shape mismatch between manifest and embedded NPZ: "
            f"manifest={expected_shape}, npz={(int(x_values.size), int(z_values.size))}."
        )
    if x_values.size * z_values.size != coords_phys.shape[0]:
        raise ValueError("Initial theta state coordinates do not form the declared regular grid.")
    if not np.all(np.isfinite(coords_phys)) or not np.all(np.isfinite(theta)):
        raise ValueError("Initial theta state contains non-finite coordinates or target values.")
    coords3 = np.column_stack(
        [coords_phys[:, 0], coords_phys[:, 1], np.zeros(coords_phys.shape[0], dtype=np.float32)]
    ).astype(np.float32)
    metadata_path = resolve_portable_path(spec.get("metadata", "")) if spec.get("metadata") else None
    metadata = None
    if metadata_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "npz_path": str(npz_path),
        "metadata_path": None if metadata_path is None else str(metadata_path),
        "metadata": metadata,
        "point_count": int(coords_phys.shape[0]),
        "grid_shape_xz": [int(x_values.size), int(z_values.size)],
        "x_range_m": [float(np.min(x_values)), float(np.max(x_values))],
        "z_range_m": [float(np.min(z_values)), float(np.max(z_values))],
        "theta_target_min": float(np.min(theta)),
        "theta_target_max": float(np.max(theta)),
        "theta_target_normalization": "(theta_pinn - theta_comsol) / Theta0",
        "sampling": str(spec.get("sampling", "uniform_without_replacement_from_full_comsol_grid")),
        "coords_dim_t": torch.tensor(phys_to_dim_coords_numpy(coords3, dimless), dtype=torch.float32, device=device),
        "z_phys_t": torch.tensor(coords_phys[:, 1:2], dtype=torch.float32, device=device),
        "theta_targets_t": torch.tensor(theta, dtype=torch.float32, device=device),
    }


def sample_initial_theta_state_batch(initial_training, sample_count):
    total_count = int(initial_training["point_count"])
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("Initial theta-state sample count must be positive when the constraint is enabled.")
    if sample_count > total_count:
        raise ValueError(
            f"Initial theta-state sample count {sample_count} exceeds the embedded grid size {total_count}; "
            "sampling is intentionally without replacement."
        )
    indices = torch.randperm(total_count, device=initial_training["coords_dim_t"].device)[:sample_count]
    return (
        initial_training["coords_dim_t"][indices],
        initial_training["z_phys_t"][indices],
        initial_training["theta_targets_t"][indices],
    )


def boundary_sampling_metadata(manifest, dimless, frame):
    comsol_domain = manifest.get("comsol_physical_domain", {}) if str(frame) == "comsol" else None
    return {
        "physical_boundary_frame": str(frame),
        "inverse_roi_domain": manifest.get("inverse_roi_domain", manifest.get("domain", {})),
        "comsol_physical_domain": manifest.get("comsol_physical_domain", {}),
        "boundary_from_manifest": manifest.get("boundary", {}).get("boundaries", {}),
        "sampled_boundary_dimless_ranges": boundary_dimless_ranges(dimless, comsol_domain),
        "note": "comsol frame samples true COMSOL boundaries and maps them using ROI scaling without clamping",
    }

def train(args):
    model_type = normalize_model_type(args.model_type)
    activation = normalize_activation_name(args.activation)
    device = choose_device(args.device)
    set_seed(args.seed, deterministic=bool(args.deterministic))
    known_field_flags = build_known_field_flags(args)
    max_wall_hours = None if args.max_wall_hours is None else float(args.max_wall_hours)
    if max_wall_hours is not None and max_wall_hours <= 0.0:
        raise ValueError("--max-wall-hours must be positive when provided.")
    comparison_every = int(args.comparison_every)
    if comparison_every < 0:
        raise ValueError("--comparison-every must be non-negative.")
    if int(args.n_top) < 0:
        raise ValueError("--n-top must be non-negative.")
    if int(args.n_bottom) < 0:
        raise ValueError("--n-bottom must be non-negative.")
    if int(args.n_left) < 0 or int(args.n_right) < 0:
        raise ValueError("--n-left/--n-right must be non-negative.")
    if int(args.n_initial_theta) < 0:
        raise ValueError("--n-initial-theta must be non-negative.")
    if int(args.n_initial_head) < 0:
        raise ValueError("--n-initial-head must be non-negative.")
    if int(args.n_initial_theta_state) < 0:
        raise ValueError("--n-initial-theta-state must be non-negative.")
    constraint_execution_plan = build_constraint_execution_plan(args)
    # The former paired initial-head/theta sample-count validation is intentionally
    # disabled with the initial-head constraint. Only theta-state IC is active.
    constraint_execution_metadata = constraint_plan_metadata(constraint_execution_plan)
    theta_noise_std = validate_theta_noise_std(args.theta_noise_std)
    theta_noise_seed = int(args.theta_noise_seed)

    reference_manifest_path = resolve_portable_path(args.reference_manifest)
    segments_dir = resolve_portable_path(args.segments_dir)
    parameter_field_dir = resolve_portable_path(args.parameter_field_dir)
    known_field_model_dir = resolve_project_path(args.known_field_model_dir)

    if not reference_manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {reference_manifest_path}")
    if not segments_dir.exists():
        raise FileNotFoundError(f"Segments reference dir not found: {segments_dir}")
    if not parameter_field_dir.exists():
        raise FileNotFoundError(f"Parameter field dir not found: {parameter_field_dir}")
    raw_manifest = load_input_manifest(reference_manifest_path)
    parameter_points_path, parameter_point_set = resolve_parameter_points_from_manifest(
        raw_manifest,
        args,
        reference_manifest_path,
    )
    if not parameter_points_path.exists():
        raise FileNotFoundError(
            f"Parameter point targets not found: {parameter_points_path}. Run build_inverse_inputs.py first."
        )
    if any(known_field_flags.values()) and not known_field_model_dir.exists():
        raise FileNotFoundError(f"Known field model dir not found: {known_field_model_dir}")
    # This publication package intentionally contains no Kriging baseline.
    # Historical training summaries remain comparable through the PINN metrics.
    kriging_metrics, kriging_metrics_path, kriging_rel_l2_by_field = {}, None, {}

    manifest = select_data_window_manifest(raw_manifest, args.data_window)
    manifest = apply_probe_group_overrides(
        manifest,
        theta_probe_group=args.theta_probe_group,
        h_probe_group=args.h_probe_group,
        legacy_probe_group=args.probe_group,
    )
    output_dir = ensure_dir(build_output_dir(model_type, args.epochs, args.data_window, args.output_dir))
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
        manifest,
        segments_dir,
        dimless,
        device,
        theta_noise_std=theta_noise_std,
        theta_noise_seed=theta_noise_seed,
    )
    segment_labels = [segment["label"] for segment in segment_references]
    theta_probe_group = manifest.get("theta_probe_group_dir_name", manifest.get("probe_group_dir_name", DEFAULT_THETA_PROBE_GROUP))
    h_probe_group = manifest.get("h_probe_group_dir_name", DEFAULT_H_PROBE_GROUP)
    segment_local_t_end = float(manifest.get("local_domain", {}).get("t_end_day", DOMAIN["t_end"]))
    boundary_metadata = manifest.get("boundary", {})
    top_flux_enabled = constraint_execution_plan["top"].enabled
    top_flux_targets = build_segment_top_flux_targets(manifest, dimless, enabled=top_flux_enabled)
    boundary_frame = str(getattr(args, "physical_boundary_frame", "roi"))
    boundary_enabled = any(
        constraint_execution_plan[name].enabled for name in ("top", "bottom", "left", "right")
    )
    comsol_boundary_domain = (
        manifest.get("comsol_physical_domain", {})
        if boundary_frame == "comsol" and boundary_enabled
        else None
    )
    comsol_initial_domain = manifest.get("comsol_physical_domain", {})
    initial_head_enabled = constraint_execution_plan["initial_head"].enabled
    initial_theta_state_enabled = constraint_execution_plan["initial_theta_state"].enabled
    boundary_metadata_payload = (
        boundary_sampling_metadata(manifest, dimless, boundary_frame)
        if boundary_enabled
        else {
            "physical_boundary_frame": boundary_frame,
            "enabled": False,
            "disabled_reason": "all_boundary_constraints_disabled",
        }
    )
    initial_theta_training = load_initial_theta_training(
        manifest,
        args.initial_theta_group,
        dimless,
        device,
        n_initial_theta=int(args.n_initial_theta),
    ) if constraint_execution_plan["initial_theta"].enabled else None
    initial_theta_state_training = (
        load_initial_theta_state_training(manifest, dimless, device)
        if initial_theta_state_enabled
        else None
    )

    parameter_field_dataset = load_parameter_field_dataset(parameter_field_dir)
    parameter_plane_reference = build_parameter_field_plane_reference(
        parameter_field_dataset,
        manifest.get("plane_definitions"),
    )
    parameter_evaluation_points = build_depth_filtered_parameter_grid_points(parameter_field_dataset)
    snapshot_parameter_evaluation_points = build_depth_filtered_parameter_grid_points(
        parameter_field_dataset,
        grid_shape=SNAPSHOT_PARAMETER_GRID_SHAPE_XZ,
    )
    write_json(
        comparison_dir / "epoch_snapshot_manifest.json",
        {
            "comparison_every": int(comparison_every),
            "snapshot_figure_dpi": int(SNAPSHOT_FIGURE_DPI),
            "parameter_point_evaluation_metadata": snapshot_parameter_evaluation_points.get("evaluation_metadata", {}),
            "exports": [],
        },
    )
    parameter_points = load_parameter_point_targets(parameter_points_path)
    parameter_output_stats = build_parameter_output_stats(parameter_points)
    param_training = build_parameter_point_tensors(parameter_points, dimless, parameter_output_stats, device)

    se_models = torch.nn.ModuleDict(
        {label: RichardsPINN(model_type=model_type, activation=activation) for label in segment_labels}
    ).to(device)
    field_model = InverseFieldModelBundle(
        field_global_stats,
        known_field_flags=known_field_flags,
        known_field_model_dir=known_field_model_dir if any(known_field_flags.values()) else None,
        parameter_model=args.parameter_model,
        parameter_output_stats=parameter_output_stats,
    ).to(device)
    parameter_model_metadata = field_model.parameter_model_metadata()
    trainable_parameter_fields = field_model.trainable_parameter_fields()
    frozen_parameter_fields = field_model.frozen_parameter_fields()
    trainable_parameter_loss_fields = field_model.trainable_loss_fields()
    parameter_loss_field_weights = {
        "lnalpha": float(args.lambda_param_alpha),
        "n": float(args.lambda_param_n),
        "theta_s": float(args.lambda_param_theta_s),
        "theta_r": float(args.lambda_param_theta_r),
        "lnks": float(args.lambda_param_lnks),
    }
    loss_plot_keys = build_loss_plot_keys(args, segment_labels, trainable_parameter_loss_fields)

    se_params = sum(count_parameters(model) for model in se_models.values())
    field_params = count_parameters(field_model)
    total_params = se_params + field_params

    trainable_params = [
        param
        for param in list(se_models.parameters()) + list(field_model.parameters())
        if param.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.learning_rate,
    )
    checkpoint_every = max(1, args.epochs // 30)
    scheduler = build_lr_scheduler(optimizer, args.epochs)
    print_every = max(1, int(args.print_every)) if args.print_every is not None else max(1, args.epochs // 1000)

    emit_log(
        (
            f"Training run started. output_dir={output_dir} model_type={model_type} "
            f"activation={activation} parameter_model={parameter_model_metadata['parameter_model_variant']} "
            f"data_window={args.data_window} epochs={int(args.epochs)} print_every={print_every} "
            f"theta_probe_group={theta_probe_group} h_probe_group={h_probe_group} "
            f"parameter_point_set={parameter_point_set} "
            f"parameter_hidden_dim={int(parameter_model_metadata['parameter_hidden_dim'])} "
            f"parameter_hidden_layers={int(parameter_model_metadata['parameter_hidden_layers'])} "
            f"parameter_head_count={int(parameter_model_metadata['parameter_head_count'])} "
            f"n_residual_per_segment={int(args.n_residual)} n_top_per_segment={int(args.n_top)} "
            f"n_bottom_per_segment={int(args.n_bottom)} n_left_per_segment={int(args.n_left)} n_right_per_segment={int(args.n_right)} lambda_top={float(args.lambda_top)} "
            f"lambda_bottom={float(args.lambda_bottom)} lambda_left={float(args.lambda_left)} lambda_right={float(args.lambda_right)} lambda_initial_theta={float(args.lambda_initial_theta)} lambda_initial_head={float(args.lambda_initial_head)} n_initial_head={int(args.n_initial_head)} lambda_initial_theta_state={float(args.lambda_initial_theta_state)} n_initial_theta_state={int(args.n_initial_theta_state)} initial_total_head_m={float(args.initial_total_head_m)} initial_head_enabled={initial_head_enabled} initial_theta_state_enabled={initial_theta_state_enabled} boundary_frame={boundary_frame} lambda_probe_theta={float(args.lambda_probe_theta)} "
            f"lambda_probe_h={float(args.lambda_probe_h)} time_causal={bool(args.time_causal)} "
            f"theta_noise_std={theta_noise_std} theta_noise_seed={theta_noise_seed} "
            f"deterministic={bool(args.deterministic)} "
            f"parameter_loss_field_weights={parameter_loss_field_weights} "
            f"parameter_output_mode={PARAMETER_OUTPUT_MODE} theta_h_loss_normalization=probe_minmax "
            f"h_loss_normalization=probe_minmax "
            f"snapshot_figure_dpi={SNAPSHOT_FIGURE_DPI} max_wall_hours={max_wall_hours} "
            f"known_field_flags={known_field_flags} trainable_parameter_fields={','.join(trainable_parameter_fields)} "
            f"constraint_execution_plan={constraint_execution_metadata} "
            f"segment_labels={','.join(segment_labels)}"
        ),
        training_log_path,
    )

    history = []
    epoch_snapshot_records = []
    best_loss = float("inf")
    best_state = {
        "se_model_state_dicts": se_model_state_dicts(se_models),
        "field_model_state_dict": deepcopy(field_model.state_dict()),
    }
    train_start = time.perf_counter()
    max_wall_seconds = None if max_wall_hours is None else max_wall_hours * 3600.0
    stop_reason = "completed_requested_epochs"
    stopped_early = False
    last_epoch = 0

    def build_top_loss(segment, residual_tau_cap_day):
        top_points = (
            sample_physical_top_boundary_points(
                dimless,
                constraint_execution_plan["top"].sample_count,
                device,
                comsol_boundary_domain,
                tau_max_day=residual_tau_cap_day,
            )
            if boundary_frame == "comsol"
            else sample_top_boundary_points(
                dimless,
                constraint_execution_plan["top"].sample_count,
                device,
            )
        )
        return compute_top_flux_loss(
            se_models[segment["label"]],
            field_model,
            top_points,
            dimless,
            top_flux_star_for_points(manifest, segment["spec"], top_points, dimless),
        )

    def build_bottom_loss(segment, residual_tau_cap_day):
        bottom_points = (
            sample_physical_bottom_boundary_points(
                dimless,
                constraint_execution_plan["bottom"].sample_count,
                device,
                comsol_boundary_domain,
                tau_max_day=residual_tau_cap_day,
            )
            if boundary_frame == "comsol"
            else sample_bottom_boundary_points(
                dimless,
                constraint_execution_plan["bottom"].sample_count,
                device,
            )
        )
        return compute_bottom_free_drainage_loss(
            se_models[segment["label"]],
            field_model,
            bottom_points,
            dimless,
        )

    def build_left_loss(segment, residual_tau_cap_day):
        left_points = sample_left_boundary_points(
            dimless,
            constraint_execution_plan["left"].sample_count,
            device,
            comsol_boundary_domain,
            tau_max_day=residual_tau_cap_day,
        )
        return compute_left_no_flow_loss(se_models[segment["label"]], field_model, left_points, dimless)

    def build_right_loss(segment, residual_tau_cap_day):
        right_points = sample_right_boundary_points(
            dimless,
            constraint_execution_plan["right"].sample_count,
            device,
            comsol_boundary_domain,
            tau_max_day=residual_tau_cap_day,
        )
        return compute_right_no_flow_loss(se_models[segment["label"]], field_model, right_points, dimless)

    se_models.train()
    field_model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        pde_losses = {}
        pde_values_for_mean = []
        segment_pde_weight = float(args.lambda_pde) / float(len(segment_references))
        residual_tau_cap_day = time_causal_residual_cap_day(epoch) if bool(args.time_causal) else None
        for segment in segment_references:
            label = segment["label"]
            residual_points = sample_training_points(dimless, args.n_residual, device, tau_max_day=residual_tau_cap_day)
            pde_loss = compute_pde_loss(se_models[label], field_model, residual_points, dimless)
            pde_loss_value = float(pde_loss.detach().cpu())
            pde_losses[label] = pde_loss_value
            pde_values_for_mean.append(pde_loss_value)
            if segment_pde_weight != 0.0:
                (segment_pde_weight * pde_loss).backward()

        top_losses, top_value = run_segment_constraint(
            constraint_execution_plan["top"],
            segment_references,
            build_top_loss,
            residual_tau_cap_day,
        )

        bottom_losses, bottom_value = run_segment_constraint(
            constraint_execution_plan["bottom"],
            segment_references,
            build_bottom_loss,
            residual_tau_cap_day,
        )

        left_losses, left_value = run_segment_constraint(
            constraint_execution_plan["left"],
            segment_references,
            build_left_loss,
            residual_tau_cap_day,
        )

        right_losses, right_value = run_segment_constraint(
            constraint_execution_plan["right"],
            segment_references,
            build_right_loss,
            residual_tau_cap_day,
        )

        initial_theta_value = 0.0
        if constraint_execution_plan["initial_theta"].enabled:
            anchor_label = segment_labels[0]
            initial_theta_loss = compute_initial_theta_loss(se_models[anchor_label], field_model, initial_theta_training, dimless)
            initial_theta_value = float(initial_theta_loss.detach().cpu())
            (float(args.lambda_initial_theta) * initial_theta_loss).backward()

        initial_head_value = 0.0
        initial_theta_state_value = 0.0
        if initial_theta_state_enabled:
            anchor_label = segment_labels[0]
            initial_points, _initial_z_phys, initial_theta_targets = sample_initial_theta_state_batch(
                initial_theta_state_training,
                int(args.n_initial_theta_state),
            )
            initial_theta_state_loss = compute_initial_theta_state_loss(
                se_models[anchor_label],
                field_model,
                initial_points,
                initial_theta_targets,
                dimless,
            )
            initial_theta_state_value = float(initial_theta_state_loss.detach().cpu())
            weighted_initial_state_loss = float(args.lambda_initial_theta_state) * initial_theta_state_loss
            weighted_initial_state_loss.backward()

        # Historical initial-head implementation retained for controlled recovery.
        # It is deliberately excluded from the active theta-only IC execution path.
        # if initial_head_enabled and initial_theta_state_enabled:
        #     initial_head_loss, _ = compute_initial_head_and_theta_state_losses(
        #         se_models[anchor_label], field_model, initial_points, _initial_z_phys,
        #         initial_theta_targets, dimless, float(args.initial_total_head_m),
        #     )
        #     initial_head_value = float(initial_head_loss.detach().cpu())
        #     (float(args.lambda_initial_head) * initial_head_loss).backward()
        # elif initial_head_enabled:
        #     initial_head_points, initial_head_z_phys = sample_physical_initial_condition_points(
        #         dimless, int(args.n_initial_head), device, comsol_domain=comsol_initial_domain,
        #     )
        #     initial_head_loss = compute_initial_head_loss(
        #         se_models[anchor_label], field_model, initial_head_points,
        #         initial_head_z_phys, dimless, float(args.initial_total_head_m),
        #     )
        #     initial_head_value = float(initial_head_loss.detach().cpu())
        #     (float(args.lambda_initial_head) * initial_head_loss).backward()

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

        probe_h_values = {}
        if float(args.lambda_probe_h) > 0.0:
            segment_probe_weight = float(args.lambda_probe_h) / float(len(segment_references))
            for segment in segment_references:
                label = segment["label"]
                probe_training = segment["h_probe_training"]
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
                        "h",
                    )
                    weight = float(stop - start) / float(total_probe)
                    (segment_probe_weight * batch_loss * weight).backward()
                    probe_sum += float(batch_loss.detach().cpu()) * float(stop - start)
                probe_h_values[label] = probe_sum / float(total_probe)
        else:
            probe_h_values = {segment["label"]: 0.0 for segment in segment_references}

        optimizer.step()
        scheduler.step()

        pde_value = float(np.mean(pde_values_for_mean))
        probe_theta_value = float(np.mean([probe_theta_values[label] for label in segment_labels]))
        probe_h_value = float(np.mean([probe_h_values[label] for label in segment_labels]))
        total_value = (
            float(args.lambda_pde) * pde_value
            + float(args.lambda_top) * top_value
            + float(args.lambda_bottom) * bottom_value
            + float(args.lambda_left) * left_value
            + float(args.lambda_right) * right_value
            + float(args.lambda_initial_theta) * initial_theta_value
            + float(args.lambda_initial_head) * initial_head_value
            + float(args.lambda_initial_theta_state) * initial_theta_state_value
            + float(args.lambda_param) * float(param_total_loss.detach().cpu())
            + float(args.lambda_probe_theta) * probe_theta_value
            + float(args.lambda_probe_h) * probe_h_value
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
            "top": top_value,
            "bottom": bottom_value,
            "left": left_value,
            "right": right_value,
            "initial_theta": initial_theta_value,
            "initial_head": initial_head_value,
            "initial_theta_state": initial_theta_state_value,
            "probe_theta": probe_theta_value,
            "probe_h": probe_h_value,
            "param": float(param_total_loss.detach().cpu()),
            "param_alpha": float(param_components["lnalpha"].detach().cpu()),
            "param_n": float(param_components["n"].detach().cpu()),
            "param_theta_s": float(param_components["theta_s"].detach().cpu()),
            "param_theta_r": float(param_components["theta_r"].detach().cpu()),
            "param_lnks": float(param_components["lnks"].detach().cpu()),
        }
        for label in segment_labels:
            row[f"pde_{label}"] = float(pde_losses[label])
            row[f"top_{label}"] = float(top_losses[label])
            row[f"bottom_{label}"] = float(bottom_losses[label])
            row[f"left_{label}"] = float(left_losses[label])
            row[f"right_{label}"] = float(right_losses[label])
            row[f"probe_theta_{label}"] = float(probe_theta_values[label])
            row[f"probe_h_{label}"] = float(probe_h_values[label])
        history.append(row)
        last_epoch = epoch
        elapsed_seconds = time.perf_counter() - train_start
        reached_wall_limit = (
            max_wall_seconds is not None
            and elapsed_seconds >= max_wall_seconds
            and epoch < args.epochs
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
                kriging_metrics=kriging_metrics,
                kriging_metrics_path=kriging_metrics_path,
            )
            save_loss_history(history, comparison_dir, plot_keys=loss_plot_keys, dpi=SNAPSHOT_FIGURE_DPI)

        if reached_wall_limit:
            stop_reason = "max_wall_hours_reached"
            stopped_early = True
            emit_log(
                (
                    f"Reached max wall-clock budget ({max_wall_hours:.3f} h) at epoch {epoch}; "
                    "continuing with final exports."
                ),
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
        predict_parameter_field_planes(field_model, parameter_plane_reference, dimless, args.plane_batch_size, device),
        comparison_dir / "parameter_fit",
        kriging_metrics=kriging_metrics,
        kriging_metrics_path=kriging_metrics_path,
    )
    parameter_point_metrics = save_parameter_point_comparison(
        parameter_evaluation_points,
        predict_parameter_points(
            field_model,
            parameter_evaluation_points,
            dimless,
            device,
            args.plane_batch_size,
        ),
        comparison_dir / "parameter_fit",
        kriging_metrics=kriging_metrics,
        kriging_metrics_path=kriging_metrics_path,
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
            "theta_noise_clip_range": [float(THETA_NOISE_CLIP_RANGE[0]), float(THETA_NOISE_CLIP_RANGE[1])],
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
            "data_window": str(args.data_window),
            "available_data_windows": manifest.get("available_data_windows", []),
            "probe_group": theta_probe_group,
            "theta_probe_group": theta_probe_group,
            "theta_probe_count": int(segment_references[0]["theta_reference_probe"]["theta_h"].shape[0]),
            "h_probe_group": h_probe_group,
            "h_probe_count": int(segment_references[0]["h_reference_probe"]["h"].shape[0]),
            "parameter_point_set": parameter_point_set,
            "parameter_points_path": str(parameter_points_path),
            "kriging_metrics_path": None if kriging_metrics_path is None else str(kriging_metrics_path),
            "kriging_rel_l2_by_field": kriging_rel_l2_by_field,
            "segment_labels": segment_labels,
            "segments": manifest["segments"],
            "epochs": int(args.epochs),
            "n_residual_per_segment": int(args.n_residual),
            "n_top_per_segment": int(args.n_top),
            "n_bottom_per_segment": int(args.n_bottom),
            "n_left_per_segment": int(args.n_left),
            "n_right_per_segment": int(args.n_right),
            "n_initial_theta": int(args.n_initial_theta),
            "n_initial_head": int(args.n_initial_head),
            "n_initial_theta_state": int(args.n_initial_theta_state),
            "comparison_every": int(args.comparison_every),
            "snapshot_figure_dpi": int(SNAPSHOT_FIGURE_DPI),
            "lambda_pde": float(args.lambda_pde),
            "lambda_top": float(args.lambda_top),
            "lambda_bottom": float(args.lambda_bottom),
            "lambda_left": float(args.lambda_left),
            "lambda_right": float(args.lambda_right),
            "lambda_initial_theta": float(args.lambda_initial_theta),
            "lambda_initial_head": float(args.lambda_initial_head),
            "lambda_initial_theta_state": float(args.lambda_initial_theta_state),
            "initial_total_head_m": float(args.initial_total_head_m),
            "initial_head_loss_normalization": "(h_m + z_m - H_init_total_m) / H0_m",
            "initial_head_domain": comsol_initial_domain,
            "initial_head_enabled": bool(initial_head_enabled),
            "initial_head_status": "commented_out_theta_only_experiment",
            "initial_theta_state_loss_normalization": "(theta_pinn - theta_comsol) / Theta0",
            "initial_theta_state_enabled": bool(initial_theta_state_enabled),
            "initial_theta_state_metadata": (
                None
                if initial_theta_state_training is None
                else {
                    key: value
                    for key, value in initial_theta_state_training.items()
                    if not key.endswith("_t")
                }
            ),
            "initial_theta_group": args.initial_theta_group,
            "initial_theta_enabled": initial_theta_training is not None,
            "initial_theta_metadata": None if initial_theta_training is None else {k: v for k, v in initial_theta_training.items() if k not in ("coords_dim_t", "theta_targets_t", "theta_clean_t")},
            "physical_boundary_frame": boundary_frame,
            "boundary_sampling_metadata": boundary_metadata_payload,
            "constraint_execution_plan": constraint_execution_metadata,
            "lambda_probe_theta": float(args.lambda_probe_theta),
            "lambda_probe_h": float(args.lambda_probe_h),
            "lambda_param": float(args.lambda_param),
            "lambda_param_alpha": float(args.lambda_param_alpha),
            "lambda_param_n": float(args.lambda_param_n),
            "lambda_param_theta_s": float(args.lambda_param_theta_s),
            "lambda_param_theta_r": float(args.lambda_param_theta_r),
            "lambda_param_lnks": float(args.lambda_param_lnks),
            "lambda_param_ks": float(args.lambda_param_lnks),
            "parameter_loss_field_weights": parameter_loss_field_weights,
            "max_wall_hours": max_wall_hours,
            "time_causal": bool(args.time_causal),
            "time_causal_epoch_block": int(TIME_CAUSAL_EPOCH_BLOCK),
            "time_causal_increment_day": float(TIME_CAUSAL_INCREMENT_DAY),
            "time_causal_max_day": float(TIME_CAUSAL_MAX_DAY),
            "lr_decay_count": int(LR_DECAY_COUNT),
            "lr_decay_gamma": float(LR_DECAY_GAMMA),
            "lr_decay_step_size": int(max(1, math.ceil(float(args.epochs) / float(LR_DECAY_COUNT)))),
            "stop_reason": stop_reason,
            "stopped_early": bool(stopped_early),
            "last_epoch": int(last_epoch),
            "top_flux_constraint": (
                "disabled" if float(args.lambda_top) <= 0.0 or int(args.n_top) <= 0 else "qz_star + q_top_star(t) = 0"
            ),
            "bottom_boundary_constraint": (
                "disabled" if float(args.lambda_bottom) <= 0.0 or int(args.n_bottom) <= 0 else "true bottom free drainage when physical_boundary_frame=comsol"
            ),
            "left_boundary_constraint": (
                "disabled" if float(args.lambda_left) <= 0.0 or int(args.n_left) <= 0 else "true left no-flow qx_star=0"
            ),
            "right_boundary_constraint": (
                "disabled" if float(args.lambda_right) <= 0.0 or int(args.n_right) <= 0 else "true right no-flow qx_star=0"
            ),
            "top_flux_schedule": manifest.get("boundary", {}).get("top_flux_schedule", []),
            "top_flux_targets_by_segment": top_flux_targets,
            "known_field_flags": known_field_flags,
            "trainable_parameter_fields": trainable_parameter_fields,
            "frozen_parameter_fields": frozen_parameter_fields,
            "trainable_parameter_loss_fields": trainable_parameter_loss_fields,
            "loss_plot_keys": loss_plot_keys,
            "known_field_model_dir": str(known_field_model_dir),
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
            "parameter_evaluation_point_count": int(parameter_evaluation_points["coords_phys"].shape[0]),
            "parameter_evaluation_metadata": parameter_evaluation_points.get("evaluation_metadata", {}),
            "snapshot_parameter_evaluation_point_count": int(snapshot_parameter_evaluation_points["coords_phys"].shape[0]),
            "snapshot_parameter_evaluation_metadata": snapshot_parameter_evaluation_points.get("evaluation_metadata", {}),
            "rain1_state_plot_times_local_d": list(RAIN1_STATE_PLOT_TIMES_LOCAL_D),
            "soil_sample_location_count": int(manifest.get("soil_sample_location_count", 0)),
            "parameter_supervision_truth": manifest.get("parameter_supervision_truth"),
            "source_prior_points_csv": manifest.get("source_prior_points_csv"),
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
    args = apply_icbc_defaults(build_parser().parse_args())
    try:
        train(args)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
