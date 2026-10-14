from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr

FEATURE_NAMES = ["USSAND", "USSILT", "USCLAY", "DEPTH_M"]
STAT_PARAMETERS = [
    ("theta_r", "theta_r", "cm3/cm3", False),
    ("theta_s", "theta_s", "cm3/cm3", False),
    ("alpha_cm_inv", "ln(alpha)", "ln(cm^-1)", True),
    ("alpha_cm_inv", "alpha", "cm^-1", False),
    ("n", "n", "-", False),
    ("Ks_cm_d", "ln(Ks)", "ln(cm/day)", True),
    ("Ks_cm_d", "Ks", "cm/day", False),
]


def dict_get(obj: dict[Any, Any], key: str) -> Any:
    for candidate, value in obj.items():
        if str(candidate) == key:
            return value
    raise KeyError(key)


def load_rdata_object(path: Path, object_name: str) -> Any:
    try:
        import rdata
        from rdata import conversion
    except ImportError as exc:
        raise RuntimeError(
            "Reading the official euptfv2 .rdata model objects requires the Python package 'rdata'. Install it with: python -m pip install rdata"
        ) from exc
    constructors = dict(conversion.DEFAULT_CLASS_MAP)
    for class_name in ["srcfilecopy", "srcfile", "srcref", "POSIXct", "POSIXt", "parseData"]:
        constructors[class_name] = lambda obj, attrs: obj
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(path)
        converted = conversion.convert(
            parsed,
            constructor_dict=constructors,
            default_encoding="utf-8",
            force_default_encoding=True,
        )
    return dict_get(converted, object_name)


def infer_split_index_offset(var_ids: np.ndarray, n_features: int) -> int:
    nonterminal = var_ids[var_ids > 0]
    if nonterminal.size == 0:
        return 0
    if int(nonterminal.min()) >= 1 and int(nonterminal.max()) <= n_features:
        return 1
    if int(nonterminal.min()) >= 0 and int(nonterminal.max()) < n_features:
        return 0
    raise ValueError(
        f"Cannot map ranger split variable IDs {nonterminal.min()}..{nonterminal.max()} to {n_features} predictor columns."
    )


def ranger_regression_predict(
    model: dict[Any, Any], predictors: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    forest = dict_get(model, "forest")
    forest_feature_names = [
        str(v) for v in np.asarray(dict_get(forest, "independent.variable.names")).tolist()
    ]
    column_index = [feature_names.index(name) for name in forest_feature_names]
    x = np.asarray(predictors[:, column_index], dtype=float)
    child_node_ids = dict_get(forest, "child.nodeIDs")
    split_var_ids = dict_get(forest, "split.varIDs")
    split_values = dict_get(forest, "split.values")
    n_trees = len(split_var_ids)
    prediction_sum = np.zeros(x.shape[0], dtype=float)
    for tree_idx in range(n_trees):
        left = np.asarray(child_node_ids[tree_idx][0], dtype=np.int64)
        right = np.asarray(child_node_ids[tree_idx][1], dtype=np.int64)
        var_ids = np.asarray(split_var_ids[tree_idx], dtype=np.int64)
        values = np.asarray(split_values[tree_idx], dtype=float)
        offset = infer_split_index_offset(var_ids, x.shape[1])
        nodes = np.zeros(x.shape[0], dtype=np.int64)
        tree_prediction = np.empty(x.shape[0], dtype=float)
        active = np.ones(x.shape[0], dtype=bool)
        for _ in range(len(var_ids) + 1):
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break
            current_nodes = nodes[active_idx]
            terminal = (left[current_nodes] == 0) & (right[current_nodes] == 0)
            if np.any(terminal):
                terminal_idx = active_idx[terminal]
                tree_prediction[terminal_idx] = values[current_nodes[terminal]]
                active[terminal_idx] = False
            branch_idx = active_idx[~terminal]
            if branch_idx.size == 0:
                continue
            branch_nodes = nodes[branch_idx]
            feature_idx = var_ids[branch_nodes] - offset
            split = values[branch_nodes]
            go_left = x[branch_idx, feature_idx] <= split
            nodes[branch_idx] = np.where(go_left, left[branch_nodes], right[branch_nodes])
        else:
            raise RuntimeError(f"Tree {tree_idx} traversal did not terminate.")
        if np.any(active):
            raise RuntimeError(
                f"Tree {tree_idx} traversal left {int(np.sum(active))} active samples."
            )
        prediction_sum += tree_prediction
    return prediction_sum / float(n_trees)


def build_predictor_table(ds: xr.Dataset) -> tuple[np.ndarray, tuple[int, int]]:
    sand = ds["sand"].values
    silt = ds["silt"].values
    clay = ds["clay"].values
    shape = sand.shape
    psd_sum = sand + silt + clay
    max_error = float(np.max(np.abs(psd_sum - 100.0)))
    if max_error > 1.0:
        raise ValueError(f"sand+silt+clay must sum to about 100%; max error is {max_error:.3g}%.")
    depth_cm = np.broadcast_to(ds["z_cm"].values[:, None], shape)
    predictors = np.column_stack(
        [sand.reshape(-1), silt.reshape(-1), clay.reshape(-1), depth_cm.reshape(-1)]
    )
    return (predictors, shape)


def write_csv(ds: xr.Dataset, path: Path) -> None:
    keep = [
        "sand",
        "silt",
        "clay",
        "depth_cm",
        "theta_r",
        "theta_s",
        "alpha_cm_inv",
        "n",
        "m",
        "Ks_cm_d",
        "log10_Ks_cm_d",
    ]
    ds[keep].to_dataframe().reset_index().to_csv(path, index=False)


def mean_autocorrelation(arr: np.ndarray, axis: int) -> np.ndarray:
    data = np.moveaxis(np.asarray(arr, dtype=float), axis, -1)
    lines = data.reshape(-1, data.shape[-1])
    n = lines.shape[-1]
    acfs = []
    for line in lines:
        centered = line - np.mean(line)
        var = np.var(centered)
        if var <= 0.0:
            continue
        corr = np.correlate(centered, centered, mode="full")[n - 1 :]
        corr = corr / (var * np.arange(n, 0, -1))
        corr[0] = 1.0
        acfs.append(corr)
    if not acfs:
        return np.full(n, np.nan)
    return np.nanmean(np.vstack(acfs), axis=0)


def efold_length(acf: np.ndarray, spacing: float) -> float:
    threshold = 1.0 / np.e
    if not np.isfinite(acf[0]):
        return np.nan
    below = np.where(acf <= threshold)[0]
    if below.size == 0:
        return np.nan
    i = int(below[0])
    if i == 0:
        return 0.0
    x0 = (i - 1) * spacing
    x1 = i * spacing
    y0 = acf[i - 1]
    y1 = acf[i]
    if y1 == y0:
        return x1
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def summarize_statistics(ds: xr.Dataset) -> pd.DataFrame:
    dx = float(ds["x_m"].values[1] - ds["x_m"].values[0])
    dz = float(ds["z_cm"].values[1] - ds["z_cm"].values[0])
    rows = []
    for var, label, unit, use_log in STAT_PARAMETERS:
        arr = ds[var].values
        if use_log:
            arr = np.log(arr)
        acf_x = mean_autocorrelation(arr, axis=1)
        acf_z = mean_autocorrelation(arr, axis=0)
        rows.append(
            {
                "parameter": label,
                "unit": unit,
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "variance": float(np.var(arr, ddof=1)),
                "corr_length_x_m": efold_length(acf_x, dx),
                "corr_length_z_cm": efold_length(acf_z, dz),
            }
        )
    return pd.DataFrame(rows)


def write_summary(ds: xr.Dataset, stats: pd.DataFrame, path: Path) -> None:
    summary = {
        "model": ds.attrs["model"],
        "ptf": ds.attrs["ptf"],
        "target": ds.attrs["target"],
        "predictors": ds.attrs["predictors"],
        "variables": {},
        "correlations": {},
        "statistics_table": stats.to_dict(orient="records"),
    }
    for name in ["theta_r", "theta_s", "alpha_cm_inv", "n", "m", "Ks_cm_d", "log10_Ks_cm_d"]:
        arr = ds[name].values
        summary["variables"][name] = {
            "units": ds[name].attrs.get("units", ""),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    theta_r = ds["theta_r"].values.reshape(-1)
    alpha = ds["alpha_cm_inv"].values.reshape(-1)
    n_minus_1 = (ds["n"].values - 1.0).reshape(-1)
    ks = ds["Ks_cm_d"].values.reshape(-1)
    summary["correlations"] = {
        "ln_Ks_vs_ln_alpha": float(np.corrcoef(np.log(ks), np.log(alpha))[0, 1]),
        "ln_Ks_vs_ln_n_minus_1": float(np.corrcoef(np.log(ks), np.log(n_minus_1))[0, 1]),
        "theta_r_vs_ln_alpha": float(np.corrcoef(theta_r, np.log(alpha))[0, 1]),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
