import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, write_dict_csv, write_json


STATE_PLANE_VARIABLES = ("se", "h", "theta_h", "k_h", "qz_h")
STATE_VARIABLE_LABELS = {
    "se": "Se",
    "h": "h",
    "theta_h": "theta_h",
    "k_h": "K(h)",
    "qz_h": "q_z",
}
STATE_VARIABLE_UNITS = {
    "h": "m",
    "k_h": "m/day",
    "qz_h": "m/day",
}
PROBE_VARIABLE_LABELS = {
    "theta_h": "theta_h",
    "h": "h",
}
PROBE_VARIABLE_UNITS = {
    "theta_h": None,
    "h": "m",
}
PARAMETER_POINT_VARIABLES = ("lnalpha", "lnn1", "theta_s", "theta_r", "lnks")
PARAMETER_FIELD_VARIABLES = ("lnalpha", "lnn1", "theta_s", "theta_r", "lnks")
PARAMETER_VARIABLE_LABELS = {
    "lnalpha": "lnalpha",
    "lnn1": "ln(n-1)",
    "theta_s": "theta_s",
    "theta_r": "theta_r",
    "lnks": "lnKs",
}
PARAMETER_VARIABLE_UNITS = {
    "lnalpha": "ln(1/m)",
    "lnn1": "-",
    "theta_s": "-",
    "theta_r": "-",
    "lnks": "ln(m/day)",
}
KRIGING_FIELD_ALIASES = {
    "lnalpha": ("lnalpha",),
    "lnn1": ("lnn1", "ln_n_minus_1"),
    "theta_s": ("theta_s",),
    "theta_r": ("theta_r",),
    "lnks": ("lnks",),
}
FIELD_FILES = {
    "alpha": "alpha_field_roi.txt",
    "n": "n_field_roi.txt",
    "theta_s": "theta_s_field_roi.txt",
    "theta_r": "theta_r_field_roi.txt",
    "ks": "ks_field_roi.txt",
}
PARAMETER_FIELD_PLOT_Z_FILTER_M = (-0.6, 0.0)
PARAMETER_POINT_EVALUATION_SOURCE = "full_parameter_grid_depth_0_100cm"
PARAMETER_POINT_EVALUATION_Z_FILTER_M = (-0.6, 0.0)


def _format_time_tag(time_d):
    return f"{float(time_d):08.4f}".replace(".", "p")


def phys_to_dim_coords_numpy(coords, dimless):
    coords = np.asarray(coords, dtype=np.float32).copy()
    if coords.shape[1] == 2:
        coords[:, 0] = (coords[:, 0] - float(dimless["x_mid"])) / float(dimless["sx"])
        coords[:, 1] = (coords[:, 1] - float(dimless["z_min"])) / float(dimless["Lz"])
        return coords
    if coords.shape[1] != 3:
        raise ValueError(f"coords must have shape [N,2] or [N,3], got {coords.shape}")
    coords[:, 0] = (coords[:, 0] - float(dimless["x_mid"])) / float(dimless["sx"])
    coords[:, 1] = (coords[:, 1] - float(dimless["z_min"])) / float(dimless["Lz"])
    coords[:, 2] = coords[:, 2] / float(dimless["T"])
    return coords


def load_input_manifest(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_probe_metadata(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clean = {(key or "").replace("\ufeff", "").strip(): value for key, value in row.items()}
            rows.append(
                {
                    "probe_name": str(clean["probe_name"]),
                    "point_index": int(clean.get("point_index", clean.get("column_index", 0))),
                    "x_m": float(clean["x_m"]),
                    "z_m": float(clean["z_m"]),
                    "slice_y_m": float(clean.get("slice_y_m", 50.0)),
                }
            )
    return rows


def _build_probe_coords(metadata, times_d, theta_h):
    n_probes = len(metadata)
    n_times = len(times_d)
    coords = np.zeros((n_probes * n_times, 3), dtype=np.float32)
    theta_values = np.zeros((n_probes * n_times, 1), dtype=np.float32)
    row = 0
    for probe_index, meta in enumerate(metadata):
        for time_index, time_d in enumerate(times_d):
            coords[row] = np.array([meta["x_m"], meta["z_m"], time_d], dtype=np.float32)
            theta_values[row, 0] = theta_h[probe_index, time_index]
            row += 1
    return coords, theta_values


def load_probe_dataset(path):
    path = Path(path)
    probe_npz = path / "probe_reference.npz" if path.is_dir() else path
    metadata_path = path / "probe_metadata.csv" if path.is_dir() else path.with_name("probe_metadata.csv")
    if not probe_npz.exists():
        raise FileNotFoundError(f"Probe reference dataset not found: {probe_npz}")
    metadata = load_probe_metadata(metadata_path)
    with np.load(probe_npz, allow_pickle=False) as data:
        times_d = data["times_day"].astype(np.float32)
        physical_times_d = data["physical_times_day"].astype(np.float32) if "physical_times_day" in data else times_d
        arrays = {
            name: data[name].astype(np.float32)
            for name in ("se", "h", "theta_h", "k_h", "qz_h")
            if name in data
        }
    theta_h = arrays.get("theta_h")
    h = arrays.get("h")
    primary_values = theta_h if theta_h is not None else h
    if primary_values is None:
        raise KeyError(f"Probe dataset {probe_npz} does not contain theta_h or h arrays.")
    coords, primary_coords = _build_probe_coords(metadata, times_d, primary_values)
    payload = {
        "metadata": metadata,
        "times_d": times_d,
        "physical_times_d": physical_times_d,
        "coords": coords,
        "primary_values": primary_coords,
        "theta_values": theta_h if theta_h is not None else primary_coords,
        "h_values": h if h is not None else primary_coords,
        "value_variables": sorted(arrays.keys()),
        "coords": coords,
    }
    payload.update(arrays)
    return payload


def load_plane_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Plane file must have columns (x,z,value): {path}")
    coord1_1d, i1 = np.unique(data[:, 0], return_inverse=True)
    coord2_1d, i2 = np.unique(data[:, 1], return_inverse=True)
    grid = np.zeros((len(coord2_1d), len(coord1_1d)), dtype=np.float32)
    grid[i2, i1] = data[:, 2]
    return coord1_1d, coord2_1d, grid


def _z_mask(z_1d, z_filter):
    z_1d = np.asarray(z_1d, dtype=np.float32)
    return (z_1d >= float(z_filter[0]) - 1.0e-7) & (z_1d <= float(z_filter[1]) + 1.0e-7)


def _depth_metadata(z_values, z_filter=PARAMETER_FIELD_PLOT_Z_FILTER_M):
    z_values = np.asarray(z_values, dtype=np.float32)
    return {
        "evaluation_depth_range": "0_100cm",
        "z_filter_m": [float(z_filter[0]), float(z_filter[1])],
        "z_actual_min_m": float(np.min(z_values)),
        "z_actual_max_m": float(np.max(z_values)),
        "z_layer_count": int(len(z_values)),
    }


def load_plane_reference_data(plane_dir, times_d, plane_definitions=None, physical_times_d=None):
    plane_dir = Path(plane_dir)
    times_array = np.asarray(times_d, dtype=np.float32)
    physical_times_array = (
        np.asarray(physical_times_d, dtype=np.float32)
        if physical_times_d is not None
        else times_array.copy()
    )
    data = {"times_d": times_array, "physical_times_d": physical_times_array, "planes": {}}
    plane_data = {
        "axis_names": ("x", "z"),
        "fixed_axis": "y",
        "fixed_value": float((plane_definitions or {}).get("xz", {}).get("fixed_value", 50.0)),
        "coord1_1d": None,
        "coord2_1d": None,
        "fields": {name: [] for name in STATE_PLANE_VARIABLES},
    }
    for time_d in times_d:
        tag = _format_time_tag(time_d)
        for var_name in STATE_PLANE_VARIABLES:
            x_1d, z_1d, grid = load_plane_csv(plane_dir / f"xz_{var_name}_t{tag}d.csv")
            mask = _z_mask(z_1d, PARAMETER_FIELD_PLOT_Z_FILTER_M)
            z_plot = z_1d[mask]
            grid = grid[mask, :]
            if plane_data["coord1_1d"] is None:
                plane_data["coord1_1d"] = x_1d
                plane_data["coord2_1d"] = z_plot
                plane_data["plot_depth_filter"] = _depth_metadata(z_plot)
            plane_data["fields"][var_name].append(grid)
    for var_name in STATE_PLANE_VARIABLES:
        plane_data["fields"][var_name] = np.stack(plane_data["fields"][var_name], axis=0)
    data["planes"]["xz"] = plane_data
    return data


def load_xzv_grid(path):
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"2D field file must have columns (x,z,value): {path}")
    x_1d64, ix = np.unique(data[:, 0], return_inverse=True)
    z_1d64, iz = np.unique(data[:, 1], return_inverse=True)
    field = np.zeros((len(z_1d64), len(x_1d64)), dtype=np.float32)
    field[iz, ix] = data[:, 2].astype(np.float32)
    x_1d = x_1d64.astype(np.float32)
    z_1d = z_1d64.astype(np.float32)
    return x_1d, z_1d, field


def load_parameter_field_dataset(field_dir):
    field_dir = Path(field_dir)
    dataset = {"x_1d": None, "z_1d": None, "fields": {}}
    for field_name, filename in FIELD_FILES.items():
        x_1d, z_1d, field = load_xzv_grid(field_dir / filename)
        if dataset["x_1d"] is None:
            dataset["x_1d"] = x_1d
            dataset["z_1d"] = z_1d
        elif not (np.array_equal(dataset["x_1d"], x_1d) and np.array_equal(dataset["z_1d"], z_1d)):
            raise ValueError(f"Parameter field grid mismatch: {field_dir / filename}")
        dataset["fields"][field_name] = field
    dataset["fields"]["lnalpha"] = np.log(np.clip(dataset["fields"]["alpha"], 1.0e-12, None)).astype(np.float32)
    dataset["fields"]["lnn1"] = np.log(np.clip(dataset["fields"]["n"] - 1.0, 1.0e-12, None)).astype(np.float32)
    dataset["fields"]["lnks"] = np.log(np.clip(dataset["fields"]["ks"], 1.0e-12, None)).astype(np.float32)
    return dataset


def build_parameter_field_plane_reference(field_dataset, plane_definitions=None):
    x_1d = field_dataset["x_1d"]
    z_1d = field_dataset["z_1d"]
    mask = _z_mask(z_1d, PARAMETER_FIELD_PLOT_Z_FILTER_M)
    z_plot = z_1d[mask]
    plane = {
        "axis_names": ("x", "z"),
        "fixed_axis": "y",
        "fixed_value": float((plane_definitions or {}).get("xz", {}).get("fixed_value", 50.0)),
        "coord1_1d": x_1d,
        "coord2_1d": z_plot,
        "fields": {},
        "plot_depth_filter": _depth_metadata(z_plot),
    }
    for field_name in PARAMETER_FIELD_VARIABLES:
        plane["fields"][field_name] = field_dataset["fields"][field_name][mask, :]
    return {"planes": {"xz": plane}, "plot_depth_filter": plane["plot_depth_filter"]}


def load_parameter_point_targets(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _uniform_cell_centers(source_1d, count):
    source_1d = np.sort(np.asarray(source_1d, dtype=np.float64).reshape(-1))
    if source_1d.size < 2:
        raise ValueError("At least two source coordinates are required for uniform center resampling.")
    if int(count) <= 0:
        raise ValueError(f"Grid count must be positive, got {count}.")
    spacing = float(np.median(np.diff(source_1d)))
    lower_edge = float(source_1d[0] - 0.5 * spacing)
    upper_edge = float(source_1d[-1] + 0.5 * spacing)
    return np.linspace(
        lower_edge + 0.5 * (upper_edge - lower_edge) / int(count),
        upper_edge - 0.5 * (upper_edge - lower_edge) / int(count),
        int(count),
        dtype=np.float64,
    )


def _bilinear_resample_parameter_field(x_source, z_source, values, x_target, z_target):
    x_order = np.argsort(np.asarray(x_source, dtype=np.float64))
    z_order = np.argsort(np.asarray(z_source, dtype=np.float64))
    x_source = np.asarray(x_source, dtype=np.float64)[x_order]
    z_source = np.asarray(z_source, dtype=np.float64)[z_order]
    values = np.asarray(values, dtype=np.float64)[np.ix_(z_order, x_order)]
    along_x = np.vstack([np.interp(x_target, x_source, row) for row in values])
    return np.column_stack([np.interp(z_target, z_source, along_x[:, index]) for index in range(len(x_target))])


def build_depth_filtered_parameter_grid_points(
    field_dataset,
    z_filter_m=PARAMETER_POINT_EVALUATION_Z_FILTER_M,
    grid_shape=None,
):
    x_1d = np.asarray(field_dataset["x_1d"], dtype=np.float64)
    z_1d = np.asarray(field_dataset["z_1d"], dtype=np.float64)
    mask = _z_mask(z_1d, z_filter_m)
    z_source = z_1d[mask]
    if grid_shape is None:
        x_plot = x_1d
        z_plot = z_source
        fields = {
            field_name: np.asarray(field_dataset["fields"][field_name][mask, :], dtype=np.float32)
            for field_name in PARAMETER_FIELD_VARIABLES
        }
        source = PARAMETER_POINT_EVALUATION_SOURCE
        resampling = "native_parameter_grid"
    else:
        n_x, n_z = (int(grid_shape[0]), int(grid_shape[1]))
        x_plot = _uniform_cell_centers(x_1d, n_x)
        z_plot = _uniform_cell_centers(z_source, n_z)
        fields = {
            field_name: _bilinear_resample_parameter_field(
                x_1d,
                z_source,
                field_dataset["fields"][field_name][mask, :],
                x_plot,
                z_plot,
            ).astype(np.float32)
            for field_name in PARAMETER_FIELD_VARIABLES
        }
        source = "uniform_parameter_grid_center_resample"
        resampling = "bilinear_reference_resample_at_uniform_cell_centers"

    xx, zz = np.meshgrid(x_plot, z_plot, indexing="xy")
    coords_phys = np.column_stack([xx.reshape(-1), zz.reshape(-1)]).astype(np.float32)
    payload = {
        "coords_phys": coords_phys,
        "x_m": coords_phys[:, 0],
        "z_m": coords_phys[:, 1],
        "probe_name": np.asarray([f"grid_{idx:06d}" for idx in range(coords_phys.shape[0])]),
        "point_index": np.arange(coords_phys.shape[0], dtype=np.int32),
        "evaluation_metadata": {
            "evaluation_point_source": source,
            "resampling": resampling,
            "grid_shape_xz": [int(len(x_plot)), int(len(z_plot))],
            "z_filter_m": [float(z_filter_m[0]), float(z_filter_m[1])],
            "z_actual_min_m": float(np.min(z_plot)),
            "z_actual_max_m": float(np.max(z_plot)),
            "x_actual_min_m": float(np.min(x_plot)),
            "x_actual_max_m": float(np.max(x_plot)),
            "evaluation_point_count": int(coords_phys.shape[0]),
        },
    }
    for field_name in PARAMETER_FIELD_VARIABLES:
        payload[field_name] = fields[field_name].reshape(-1, 1)
    return payload


def metrics_for_pair(reference, prediction):
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    diff = prediction - reference
    denom = float(np.linalg.norm(reference.reshape(-1)))
    centered = reference.reshape(-1) - float(np.mean(reference))
    ss_tot = float(np.sum(centered**2))
    ss_res = float(np.sum(diff.reshape(-1) ** 2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / max(denom, 1.0e-12)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-30 else 0.0,
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def parameter_axis_label(field_name):
    label = PARAMETER_VARIABLE_LABELS.get(field_name, field_name)
    unit = PARAMETER_VARIABLE_UNITS.get(field_name, "")
    return f"{label} [{unit}]" if unit else label


def state_axis_label(var_name):
    label = STATE_VARIABLE_LABELS.get(var_name, var_name)
    unit = STATE_VARIABLE_UNITS.get(var_name, "")
    return f"{label} [{unit}]" if unit else label


def _extract_kriging_metric(kriging_metrics, field_name, metric_name):
    if not kriging_metrics:
        return None
    aliases = KRIGING_FIELD_ALIASES.get(field_name, (field_name,))
    if isinstance(kriging_metrics, dict):
        candidates = [kriging_metrics.get(alias) for alias in aliases]
        metrics_payload = kriging_metrics.get("metrics")
        if isinstance(metrics_payload, dict):
            candidates.extend(metrics_payload.get(alias) for alias in aliases)
        if metric_name == "rel_l2":
            for payload_key in ("field_rel_l2", "point_rel_l2"):
                field_payload = kriging_metrics.get(payload_key)
                if isinstance(field_payload, dict):
                    for alias in aliases:
                        if alias in field_payload:
                            return float(field_payload[alias])
        if metric_name == "r2":
            for payload_key in ("field_r2", "point_r2"):
                field_payload = kriging_metrics.get(payload_key)
                if isinstance(field_payload, dict):
                    for alias in aliases:
                        if alias in field_payload:
                            return float(field_payload[alias])
    else:
        candidates = []
    for candidate in candidates:
        if isinstance(candidate, dict) and metric_name in candidate:
            return float(candidate[metric_name])
    return None


def _extract_kriging_rel_l2(kriging_metrics, field_name):
    return _extract_kriging_metric(kriging_metrics, field_name, "rel_l2")


def _extract_kriging_r2(kriging_metrics, field_name):
    return _extract_kriging_metric(kriging_metrics, field_name, "r2")


def _baseline_label(kriging_metrics):
    if not isinstance(kriging_metrics, dict):
        return "OK"
    label = kriging_metrics.get("baseline_label") or kriging_metrics.get("kriging_label")
    if label:
        return str(label)
    method = str(kriging_metrics.get("interpolation_method") or kriging_metrics.get("source") or "")
    if "rbf" in method.lower():
        return "RBF"
    return "OK"


def _ok_label(kriging_metrics, field_name):
    rel_l2 = _extract_kriging_rel_l2(kriging_metrics, field_name)
    r2 = _extract_kriging_r2(kriging_metrics, field_name)
    if rel_l2 is None and r2 is None:
        return ""
    label = _baseline_label(kriging_metrics)
    parts = []
    if rel_l2 is not None:
        parts.append(f"{label} relL2={rel_l2:.3e}")
    if r2 is not None:
        parts.append(f"{label} R$^2$={r2:.3f}")
    return "\n" + " ".join(parts)


def _baseline_root(kriging_metrics=None, kriging_metrics_path=None):
    candidates = []
    if kriging_metrics_path:
        candidates.append(Path(kriging_metrics_path).parent)
    if isinstance(kriging_metrics, dict):
        for key in (
            "baseline_dir",
            "output_dir",
            "kriging_baseline_dir",
            "ordinary_kriging_baseline_dir",
            "rbf_baseline_dir",
        ):
            value = kriging_metrics.get(key)
            if value:
                candidates.append(Path(value))
        for key in ("parameter_field_metrics", "parameter_point_metrics"):
            value = kriging_metrics.get(key)
            if value:
                candidates.append(Path(value).parent)
    for candidate in candidates:
        try:
            candidate = Path(candidate)
        except TypeError:
            continue
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _baseline_field_file_candidates(field_name, baseline_label):
    aliases = {
        "lnalpha": ("lnalpha",),
        "lnn1": ("lnn1", "ln_n_minus_1"),
        "theta_s": ("theta_s",),
        "theta_r": ("theta_r",),
        "lnks": ("lnks", "Ks"),
    }.get(field_name, (field_name,))
    label = str(baseline_label or "OK").lower()
    if label == "rbf":
        suffixes = ("rbf_interpolation_field", "rbf_field", "rbf_interp")
    else:
        suffixes = ("ordinary_kriging_field", "ok_field")
    names = []
    for alias in aliases:
        for suffix in suffixes:
            names.append(f"{alias}_{suffix}.txt")
    return names


def _load_xzv_field(path):
    try:
        data = np.loadtxt(path, dtype=np.float64)
    except ValueError:
        data = np.loadtxt(path, dtype=np.float64, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Baseline field file must have at least x,z,value columns: {path}")
    x_1d, ix = np.unique(data[:, 0], return_inverse=True)
    z_1d, iz = np.unique(data[:, 1], return_inverse=True)
    field = np.full((len(z_1d), len(x_1d)), np.nan, dtype=np.float32)
    field[iz, ix] = data[:, 2].astype(np.float32)
    return x_1d.astype(np.float32), z_1d.astype(np.float32), field


def _align_grid_field(source_x, source_z, source_field, target_x, target_z, field_name):
    source_x = np.asarray(source_x, dtype=np.float64)
    source_z = np.asarray(source_z, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    source_field = np.asarray(source_field, dtype=np.float32)
    if source_field.shape == (len(target_z), len(target_x)) and np.allclose(source_x, target_x) and np.allclose(source_z, target_z):
        return source_field.astype(np.float32)
    tolerance = 1.0e-6
    if (
        float(np.min(target_x)) < float(np.min(source_x)) - tolerance
        or float(np.max(target_x)) > float(np.max(source_x)) + tolerance
        or float(np.min(target_z)) < float(np.min(source_z)) - tolerance
        or float(np.max(target_z)) > float(np.max(source_z)) + tolerance
    ):
        raise ValueError(f"Baseline field grid for {field_name} does not cover the target grid.")
    return _bilinear_resample_parameter_field(source_x, source_z, source_field, target_x, target_z).astype(np.float32)


def load_baseline_field_planes(kriging_metrics=None, kriging_metrics_path=None, target_x=None, target_z=None):
    root = _baseline_root(kriging_metrics, kriging_metrics_path)
    if root is None:
        return None
    label = _baseline_label(kriging_metrics)
    fields = {}
    paths = {}
    for field_name in PARAMETER_FIELD_VARIABLES:
        field_path = None
        for filename in _baseline_field_file_candidates(field_name, label):
            candidate = root / filename
            if candidate.exists():
                field_path = candidate
                break
        if field_path is None:
            return None
        x_1d, z_1d, field = _load_xzv_field(field_path)
        if target_x is not None and target_z is not None:
            field = _align_grid_field(x_1d, z_1d, field, target_x, target_z, field_name)
        fields[field_name] = field
        paths[field_name] = str(field_path)
    return {
        "baseline_label": label,
        "root": str(root),
        "field_paths": paths,
        "fields": fields,
    }


def _baseline_values_at_points(baseline_fields, x_1d, z_1d, coords_phys):
    if not baseline_fields:
        return None
    coords_phys = np.asarray(coords_phys, dtype=np.float64)
    x_index = {round(float(value), 9): index for index, value in enumerate(np.asarray(x_1d, dtype=np.float64))}
    z_index = {round(float(value), 9): index for index, value in enumerate(np.asarray(z_1d, dtype=np.float64))}
    values = {}
    for field_name, field in baseline_fields["fields"].items():
        field = np.asarray(field, dtype=np.float32)
        sampled = np.empty((coords_phys.shape[0],), dtype=np.float32)
        for idx, (x_value, z_value) in enumerate(coords_phys[:, :2]):
            ix = x_index.get(round(float(x_value), 9))
            iz = z_index.get(round(float(z_value), 9))
            if ix is None or iz is None:
                return None
            sampled[idx] = field[iz, ix]
        values[field_name] = sampled
    return values


def save_loss_history(history, output_dir, plot_keys=None, dpi=180):
    output_dir = ensure_dir(Path(output_dir))
    if not history:
        return
    keys = []
    seen = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    write_dict_csv(output_dir / "loss_history.csv", keys, history)
    plot_key_set = None if plot_keys is None else set(plot_keys)
    fig = plt.figure(figsize=(14.5, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.35, 1.25], hspace=0.28, wspace=0.28)
    ax_loss = fig.add_subplot(grid[:, 0])
    ax_param = fig.add_subplot(grid[0, 1])
    ax_state = fig.add_subplot(grid[1, 1])

    def collect_series(metric_key):
        xs = []
        ys = []
        for row in history:
            if metric_key not in row:
                continue
            value = row.get(metric_key)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            xs.append(int(row["epoch"]))
            ys.append(max(value, 1.0e-30))
        return xs, ys

    plotted_loss = False
    for key in keys:
        if key in ("epoch", "elapsed_min", "lr"):
            continue
        if plot_key_set is not None and key not in plot_key_set:
            continue
        raw_values = [float(row.get(key, 0.0)) for row in history]
        if all(abs(value) <= 1.0e-30 for value in raw_values):
            continue
        epochs = [int(row["epoch"]) for row in history]
        values = [max(value, 1.0e-30) for value in raw_values]
        ax_loss.semilogy(epochs, values, label=key)
        plotted_loss = True
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(alpha=0.25)
    if plotted_loss:
        ax_loss.legend(fontsize=7, ncol=2)
    else:
        ax_loss.text(0.5, 0.5, "No active nonzero loss terms", ha="center", va="center", transform=ax_loss.transAxes)

    parameter_curve_keys = (
        ("lnalpha", "param_field_rel_l2_lnalpha"),
        ("ln(n-1)", "param_field_rel_l2_lnn1"),
        ("theta_s", "param_field_rel_l2_theta_s"),
        ("theta_r", "param_field_rel_l2_theta_r"),
        ("lnks", "param_field_rel_l2_lnks"),
    )
    plotted_param = False
    for label, metric_key in parameter_curve_keys:
        xs, ys = collect_series(metric_key)
        if not xs:
            continue
        ax_param.semilogy(xs, ys, label=label)
        plotted_param = True
    ax_param.set_xlabel("epoch")
    ax_param.set_ylabel("relL2")
    ax_param.set_title("Parameter Field relL2")
    ax_param.grid(alpha=0.25)
    if plotted_param:
        ax_param.legend(fontsize=7, ncol=2)
    else:
        ax_param.text(0.5, 0.5, "No parameter relL2 samples", ha="center", va="center", transform=ax_param.transAxes)

    state_curve_keys = (
        ("h", "state_rel_l2_h"),
        ("k_h", "state_rel_l2_k_h"),
        ("se", "state_rel_l2_se"),
        ("theta_h", "state_rel_l2_theta_h"),
    )
    plotted_state = False
    for label, metric_key in state_curve_keys:
        xs, ys = collect_series(metric_key)
        if not xs:
            continue
        ax_state.semilogy(xs, ys, label=label)
        plotted_state = True
    ax_state.set_xlabel("epoch")
    ax_state.set_ylabel("relL2")
    ax_state.set_title("State Field relL2")
    ax_state.grid(alpha=0.25)
    if plotted_state:
        ax_state.legend(fontsize=7, ncol=2)
    else:
        ax_state.text(0.5, 0.5, "No state relL2 samples", ha="center", va="center", transform=ax_state.transAxes)

    fig.savefig(output_dir / "loss_history.png", dpi=int(dpi))
    plt.close(fig)


def _imshow_xz(ax, x_1d, z_1d, values, title, cmap="turbo", vmin=None, vmax=None):
    im = ax.imshow(
        values,
        origin="lower",
        extent=[float(x_1d[0]), float(x_1d[-1]), float(z_1d[0]), float(z_1d[-1])],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    return im


def save_state_plane_comparison(plane_reference, plane_predictions, output_dir, dpi=180, plot_times_local_d=None):
    output_dir = ensure_dir(Path(output_dir))
    metrics_rows = []
    metrics_json = {}
    ref = plane_reference["planes"]["xz"]
    pred = plane_predictions["xz"]
    x_1d = ref["coord1_1d"]
    z_1d = ref["coord2_1d"]
    times = plane_reference["times_d"]
    physical_times = plane_reference.get("physical_times_d", times)
    if plot_times_local_d is None:
        plot_indices = list(range(len(times)))
    else:
        plot_indices = []
        for target_time_d in plot_times_local_d:
            matches = np.where(np.isclose(times, float(target_time_d), atol=1.0e-6))[0]
            if matches.size:
                plot_indices.append(int(matches[0]))
        if not plot_indices:
            raise ValueError(
                f"None of the requested state-plot times {list(plot_times_local_d)} are available in {list(times)}."
            )
    for var_name in ref["fields"].keys():
        metrics_json[var_name] = {}
        axis_label = state_axis_label(var_name)
        for i, time_d in enumerate(times):
            physical_time_d = float(physical_times[i])
            truth = ref["fields"][var_name][i]
            predicted = pred["fields"][var_name][i]
            err = predicted - truth
            metric = metrics_for_pair(truth, predicted)
            metrics_json[var_name][str(physical_time_d)] = {
                "local_time_d": float(time_d),
                "physical_time_d": physical_time_d,
                **metric,
            }
            metrics_rows.append(
                {
                    "variable": var_name,
                    "local_time_d": float(time_d),
                    "physical_time_d": physical_time_d,
                    **metric,
                }
            )
        fig, axes = plt.subplots(
            len(plot_indices),
            3,
            figsize=(11.5, max(2.2 * len(plot_indices), 3.0)),
            squeeze=False,
        )
        for row_index, i in enumerate(plot_indices):
            time_d = times[i]
            physical_time_d = float(physical_times[i])
            truth = ref["fields"][var_name][i]
            predicted = pred["fields"][var_name][i]
            err = predicted - truth
            metric = metrics_json[var_name][str(physical_time_d)]
            clim = (float(min(np.min(truth), np.min(predicted))), float(max(np.max(truth), np.max(predicted))))
            err_abs = float(np.max(np.abs(err)))
            panels = (
                ("Reference", truth, "turbo", clim[0], clim[1]),
                ("PINN", predicted, "turbo", clim[0], clim[1]),
                ("Error", err, "coolwarm", -err_abs, err_abs),
            )
            for col, (title, values, cmap, vmin, vmax) in enumerate(panels):
                label = f"{title} {axis_label} t={physical_time_d:g}d"
                if not np.isclose(float(time_d), physical_time_d):
                    label += f" (local {float(time_d):g}d)"
                if col == 2:
                    label += f"\nrelL2={metric['rel_l2']:.3g}"
                im = _imshow_xz(ax=axes[row_index, col], x_1d=x_1d, z_1d=z_1d, values=values, title=label, cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=axes[row_index, col], shrink=0.78)
                if var_name in STATE_VARIABLE_UNITS:
                    cbar.set_label(axis_label, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"xz_{var_name}_comparison.png", dpi=int(dpi))
        plt.close(fig)
    if metrics_rows:
        write_dict_csv(output_dir / "plane_metrics.csv", list(metrics_rows[0].keys()), metrics_rows)
    write_json(output_dir / "plane_metrics.json", metrics_json)
    return metrics_json


def _probe_axis_label(variable_name):
    label = PROBE_VARIABLE_LABELS.get(variable_name, variable_name)
    unit = PROBE_VARIABLE_UNITS.get(variable_name)
    return label if not unit else f"{label} ({unit})"


def save_probe_variable_comparison(reference_probe, prediction_values, output_dir, plot_seed, variable_name="theta_h", file_prefix=None, plot_prefix=None, dpi=180):
    output_dir = ensure_dir(Path(output_dir))
    if variable_name not in reference_probe:
        raise KeyError(f"Probe dataset does not contain observed variable {variable_name!r}.")
    file_prefix = "probe" if file_prefix is None else str(file_prefix)
    plot_prefix = file_prefix if plot_prefix is None else str(plot_prefix)
    metrics = {variable_name: metrics_for_pair(reference_probe[variable_name], prediction_values)}
    write_json(output_dir / f"{file_prefix}_metrics.json", metrics)
    write_dict_csv(
        output_dir / f"{file_prefix}_metrics.csv",
        ["variable", *metrics[variable_name].keys()],
        [{"variable": variable_name, **metrics[variable_name]}],
    )
    plot_times = reference_probe.get("physical_times_d", reference_probe["times_d"])
    rng = np.random.default_rng(int(plot_seed))
    n_show = min(12, len(reference_probe["metadata"]))
    indices = np.sort(rng.choice(len(reference_probe["metadata"]), size=n_show, replace=False))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for offset, index in enumerate(indices):
        ax.plot(
            plot_times,
            reference_probe[variable_name][index],
            color="black",
            alpha=0.38,
            linewidth=0.9,
            linestyle="-",
            label="Reference" if offset == 0 else None,
        )
        ax.plot(
            plot_times,
            prediction_values[index],
            color="red",
            alpha=0.65,
            linewidth=0.9,
            linestyle="--",
            label="PINN" if offset == 0 else None,
        )
    ax.set_xlabel("physical time (day)")
    ax.set_ylabel(_probe_axis_label(variable_name))
    ax.set_title(f"{_probe_axis_label(variable_name)} probe comparison relL2={metrics[variable_name]['rel_l2']:.3e}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{plot_prefix}_comparison.png", dpi=int(dpi))
    plt.close(fig)
    return metrics


def save_probe_theta_comparison(reference_probe, prediction_theta_h, output_dir, plot_seed, dpi=180):
    return save_probe_variable_comparison(
        reference_probe,
        prediction_theta_h,
        output_dir,
        plot_seed,
        variable_name="theta_h",
        file_prefix="probe",
        plot_prefix="probe_theta",
        dpi=dpi,
    )


def save_parameter_field_comparison(reference_planes, predicted_planes, output_dir, kriging_metrics=None, kriging_metrics_path=None, dpi=180):
    output_dir = ensure_dir(Path(output_dir))
    for child in list(output_dir.iterdir()) if output_dir.exists() else []:
        if child.is_dir():
            shutil.rmtree(child)
    ref = reference_planes["planes"]["xz"]
    pred = predicted_planes["planes"]["xz"]
    x_1d = ref["coord1_1d"]
    z_1d = ref["coord2_1d"]
    baseline_planes = load_baseline_field_planes(
        kriging_metrics=kriging_metrics,
        kriging_metrics_path=kriging_metrics_path,
        target_x=x_1d,
        target_z=z_1d,
    )
    baseline_label = _baseline_label(kriging_metrics)
    has_baseline_fields = baseline_planes is not None
    metrics_rows = []
    metrics_json = {"plot_depth_filter": ref.get("plot_depth_filter", {})}
    if has_baseline_fields:
        metrics_json["baseline_field_source"] = {
            "baseline_label": baseline_label,
            "root": baseline_planes["root"],
            "field_paths": baseline_planes["field_paths"],
        }
    else:
        metrics_json["baseline_field_source"] = {
            "baseline_label": baseline_label,
            "available": False,
        }
    n_cols = 4 if has_baseline_fields else 3
    fig, axes = plt.subplots(
        len(PARAMETER_FIELD_VARIABLES),
        n_cols,
        figsize=(15.6 if has_baseline_fields else 12.0, 13.0),
        squeeze=False,
    )
    for row, field_name in enumerate(PARAMETER_FIELD_VARIABLES):
        truth = ref["fields"][field_name]
        predicted = pred["fields"][field_name]
        err = predicted - truth
        metric = metrics_for_pair(truth, predicted)
        kriging_rel_l2 = _extract_kriging_rel_l2(kriging_metrics, field_name)
        if kriging_rel_l2 is not None:
            metric["kriging_rel_l2"] = float(kriging_rel_l2)
        kriging_r2 = _extract_kriging_r2(kriging_metrics, field_name)
        if kriging_r2 is not None:
            metric["kriging_r2"] = float(kriging_r2)
        metric["baseline_label"] = baseline_label
        baseline = None if not has_baseline_fields else baseline_planes["fields"].get(field_name)
        baseline_metric = None
        if baseline is not None:
            baseline_metric = metrics_for_pair(truth, baseline)
            metric["baseline_field_file"] = baseline_planes["field_paths"].get(field_name)
            metric["baseline_field_rel_l2"] = float(baseline_metric["rel_l2"])
            metric["baseline_field_r2"] = float(baseline_metric["r2"])
            if kriging_rel_l2 is None:
                metric["kriging_rel_l2"] = float(baseline_metric["rel_l2"])
            if kriging_r2 is None:
                metric["kriging_r2"] = float(baseline_metric["r2"])
        metrics_json[field_name] = {**metric, **ref.get("plot_depth_filter", {})}
        metrics_rows.append({"field": field_name, **metric, **ref.get("plot_depth_filter", {})})
        clim_arrays = [truth, predicted]
        if baseline is not None:
            clim_arrays.append(baseline)
        clim = (
            float(min(np.min(values) for values in clim_arrays)),
            float(max(np.max(values) for values in clim_arrays)),
        )
        err_abs = float(np.max(np.abs(err)))
        axis_label = parameter_axis_label(field_name)
        panels = [
            ("Reference", truth, "jet", clim[0], clim[1]),
            ("PINN", predicted, "jet", clim[0], clim[1]),
        ]
        if baseline is not None:
            panels.append((baseline_label, baseline, "jet", clim[0], clim[1]))
        panels.append(("PINN - Reference", err, "coolwarm", -err_abs, err_abs))
        for col, (title, values, cmap, vmin, vmax) in enumerate(panels):
            label = f"{title} {axis_label}"
            if title == "PINN":
                label += f"\nrelL2={metric['rel_l2']:.3e} R$^2$={metric['r2']:.3f}"
            elif baseline is not None and title == baseline_label and baseline_metric is not None:
                label += f"\nrelL2={baseline_metric['rel_l2']:.3e} R$^2$={baseline_metric['r2']:.3f}"
            elif title == "PINN - Reference":
                label += f"\nrelL2={metric['rel_l2']:.3e} R$^2$={metric['r2']:.3f}"
            im = _imshow_xz(axes[row, col], x_1d, z_1d, values, label, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axes[row, col], shrink=0.78)
            cbar.set_label(axis_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "xz_parameter_fields_overview.png", dpi=int(dpi))
    plt.close(fig)
    write_json(output_dir / "parameter_field_metrics.json", metrics_json)
    write_dict_csv(output_dir / "parameter_field_metrics.csv", list(metrics_rows[0].keys()), metrics_rows)
    return metrics_json


def save_parameter_point_comparison(targets, predictions, output_dir, kriging_metrics=None, kriging_metrics_path=None, dpi=180):
    output_dir = ensure_dir(Path(output_dir))
    metric_rows = []
    per_point_rows = []
    n_fields = len(PARAMETER_POINT_VARIABLES)
    metadata = targets.get("evaluation_metadata", {})
    coords_phys = np.asarray(targets["coords_phys"], dtype=np.float32)
    x_1d = np.unique(coords_phys[:, 0]).astype(np.float32)
    z_1d = np.unique(coords_phys[:, 1]).astype(np.float32)
    baseline_planes = load_baseline_field_planes(
        kriging_metrics=kriging_metrics,
        kriging_metrics_path=kriging_metrics_path,
        target_x=x_1d,
        target_z=z_1d,
    )
    baseline_values = _baseline_values_at_points(baseline_planes, x_1d, z_1d, coords_phys)
    if baseline_values is None:
        baseline_planes = None
    has_baseline_fields = baseline_values is not None
    baseline_label = _baseline_label(kriging_metrics)
    n_rows = 2 if has_baseline_fields else 1
    fig, axes = plt.subplots(
        n_rows,
        n_fields,
        figsize=(3.2 * n_fields, 6.4 if has_baseline_fields else 3.2),
        squeeze=False,
    )
    for col, field_name in enumerate(PARAMETER_POINT_VARIABLES):
        truth = np.asarray(targets[field_name], dtype=np.float32).reshape(-1)
        pred = np.asarray(predictions[field_name], dtype=np.float32).reshape(-1)
        metric = metrics_for_pair(truth, pred)
        kriging_rel_l2 = _extract_kriging_rel_l2(kriging_metrics, field_name)
        if kriging_rel_l2 is not None:
            metric["kriging_rel_l2"] = float(kriging_rel_l2)
        kriging_r2 = _extract_kriging_r2(kriging_metrics, field_name)
        if kriging_r2 is not None:
            metric["kriging_r2"] = float(kriging_r2)
        metric["baseline_label"] = baseline_label
        baseline = None if baseline_values is None else baseline_values.get(field_name)
        baseline_metric = None
        if baseline is not None:
            baseline_metric = metrics_for_pair(truth, baseline)
            metric["baseline_point_rel_l2"] = float(baseline_metric["rel_l2"])
            metric["baseline_point_r2"] = float(baseline_metric["r2"])
            if baseline_planes is not None:
                metric["baseline_field_file"] = baseline_planes["field_paths"].get(field_name)
            if kriging_rel_l2 is None:
                metric["kriging_rel_l2"] = float(baseline_metric["rel_l2"])
            if kriging_r2 is None:
                metric["kriging_r2"] = float(baseline_metric["r2"])
        metric_rows.append({"field": field_name, **metric, **metadata})
        ax = axes[0, col]
        ax.scatter(truth, pred, s=3, alpha=0.25)
        value_arrays = [truth, pred]
        if baseline is not None:
            value_arrays.append(baseline)
        lo = float(min(np.min(values) for values in value_arrays))
        hi = float(max(np.max(values) for values in value_arrays))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9)
        axis_label = parameter_axis_label(field_name)
        ax.set_xlabel(f"Reference {axis_label}")
        ax.set_ylabel(f"PINN {axis_label}")
        ax.set_title(f"PINN {axis_label}\nrelL2={metric['rel_l2']:.3e} R$^2$={metric['r2']:.3f}")
        ax.grid(alpha=0.22)
        if baseline is not None and baseline_metric is not None:
            ax_ok = axes[1, col]
            ax_ok.scatter(truth, baseline, s=3, alpha=0.25, color="tab:green")
            ax_ok.plot([lo, hi], [lo, hi], "k--", linewidth=0.9)
            ax_ok.set_xlabel(f"Reference {axis_label}")
            ax_ok.set_ylabel(f"{baseline_label} {axis_label}")
            ax_ok.set_title(f"{baseline_label} {axis_label}\nrelL2={baseline_metric['rel_l2']:.3e} R$^2$={baseline_metric['r2']:.3f}")
            ax_ok.grid(alpha=0.22)
        for idx in range(truth.shape[0]):
            row = {
                "field": field_name,
                "point_index": int(targets["point_index"][idx]) if "point_index" in targets else idx,
                "x_m": float(targets["x_m"][idx]) if "x_m" in targets else float(targets["coords_phys"][idx, 0]),
                "z_m": float(targets["z_m"][idx]) if "z_m" in targets else float(targets["coords_phys"][idx, 1]),
                "reference": float(truth[idx]),
                "prediction": float(pred[idx]),
                "error": float(pred[idx] - truth[idx]),
            }
            if baseline is not None:
                row["baseline_label"] = baseline_label
                row["baseline_prediction"] = float(baseline[idx])
                row["baseline_error"] = float(baseline[idx] - truth[idx])
            per_point_rows.append(row)
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_point_scatter.png", dpi=int(dpi))
    plt.close(fig)
    write_json(
        output_dir / "parameter_point_metrics.json",
        {
            "evaluation_metadata": metadata,
            "metrics": {row["field"]: {k: v for k, v in row.items() if k != "field"} for row in metric_rows},
        },
    )
    write_dict_csv(output_dir / "parameter_point_metrics.csv", list(metric_rows[0].keys()), metric_rows)
    write_dict_csv(output_dir / "parameter_point_predictions.csv", list(per_point_rows[0].keys()), per_point_rows)
    return {row["field"]: {k: v for k, v in row.items() if k != "field"} for row in metric_rows}
