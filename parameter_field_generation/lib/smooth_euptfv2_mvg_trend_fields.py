from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter


def logit(values: np.ndarray, eps: float = 1e-09) -> np.ndarray:
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def transformed_fields(ds: xr.Dataset) -> dict[str, np.ndarray]:
    theta_s = ds["theta_s"].values
    ratio = ds["theta_r"].values / theta_s
    return {
        "logit_theta_s": logit(theta_s),
        "logit_theta_r_over_theta_s": logit(ratio),
        "ln_alpha": np.log(ds["alpha_cm_inv"].values),
        "ln_n_minus_1": np.log(ds["n"].values - 1.0),
        "ln_Ks": np.log(ds["Ks_cm_d"].values),
    }


def inverse_transformed_fields(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    theta_s = logistic(fields["logit_theta_s"])
    theta_r = theta_s * logistic(fields["logit_theta_r_over_theta_s"])
    alpha = np.exp(fields["ln_alpha"])
    n_param = 1.0 + np.exp(fields["ln_n_minus_1"])
    ks = np.exp(fields["ln_Ks"])
    return {
        "theta_r": theta_r,
        "theta_s": theta_s,
        "alpha_cm_inv": alpha,
        "n": n_param,
        "m": 1.0 - 1.0 / n_param,
        "K0_cm_d": ks,
        "Ks_cm_d": ks,
        "log10_alpha_cm_inv": np.log10(alpha),
        "log10_Ks_cm_d": np.log10(ks),
    }


def smooth_transformed_fields(
    ds: xr.Dataset, sigma_x_m: float, sigma_z_cm: float
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    dx_m = float(ds["x_m"].values[1] - ds["x_m"].values[0])
    dz_cm = float(ds["z_cm"].values[1] - ds["z_cm"].values[0])
    sigma = (sigma_z_cm / dz_cm, sigma_x_m / dx_m)
    smoothed = {}
    for name, values in transformed_fields(ds).items():
        smoothed[name] = gaussian_filter(values, sigma=sigma, mode="nearest")
    meta = {
        "sigma_x_m": float(sigma_x_m),
        "sigma_z_cm": float(sigma_z_cm),
        "sigma_x_cells": float(sigma[1]),
        "sigma_z_cells": float(sigma[0]),
    }
    return (smoothed, meta)


def make_dataset(
    ds: xr.Dataset, smoothed_transforms: dict[str, np.ndarray], meta: dict[str, float]
) -> xr.Dataset:
    physical = inverse_transformed_fields(smoothed_transforms)
    out = xr.Dataset(coords=ds.coords, attrs=dict(ds.attrs))
    out.attrs.update(
        {
            "title": "Trend-smoothed EUPTFv2 PTF01_MVG fields",
            "postprocess": "Anisotropic Gaussian low-pass smoothing in transformed hydraulic parameter space.",
            "transform_space": "logit(theta_s), logit(theta_r/theta_s), ln(alpha), ln(n-1), ln(Ks)",
            "postprocess_json": json.dumps(meta, ensure_ascii=True),
        }
    )
    for source_name in ("sand", "silt", "clay"):
        if source_name in ds:
            out[source_name] = ds[source_name]
            out[source_name].attrs["units"] = ds[source_name].attrs.get("units", "%")
    if "depth_cm" in ds:
        out["depth_cm"] = ds["depth_cm"]
    else:
        depth_cm = np.broadcast_to(ds["z_cm"].values[:, None], (ds.sizes["z_cm"], ds.sizes["x_m"]))
        out["depth_cm"] = (("z_cm", "x_m"), depth_cm)
    out["depth_cm"].attrs["units"] = "cm"
    units = {
        "theta_r": "cm3 cm-3",
        "theta_s": "cm3 cm-3",
        "alpha_cm_inv": "cm-1",
        "n": "-",
        "m": "-",
        "K0_cm_d": "cm d-1",
        "Ks_cm_d": "cm d-1",
        "log10_alpha_cm_inv": "log10(cm-1)",
        "log10_Ks_cm_d": "log10(cm d-1)",
    }
    for name, values in physical.items():
        out[name] = (("z_cm", "x_m"), values)
        out[name].attrs["units"] = units[name]
    out["Ks_cm_d"].attrs[
        "long_name"
    ] = "Trend-smoothed MVG K0 used as saturated hydraulic conductivity field"
    out["K0_cm_d"].attrs["long_name"] = "Trend-smoothed Mualem-van Genuchten K0"
    return out


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "image.cmap": "jet",
            "figure.dpi": 300,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.edgecolor": "#2B3B3B",
            "xtick.major.size": 1.5,
            "ytick.major.size": 1.5,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.0,
            "font.family": "Calibri",
            "axes.titlesize": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
        }
    )


def plot_maps_actual(ds: xr.Dataset, path: Path) -> None:
    configure_paper_style()
    x_m = np.asarray(ds["x_m"].values, dtype=float)
    z_m = -np.asarray(ds["z_cm"].values, dtype=float) / 100.0
    figure = plt.figure(figsize=(14.0 / 2.54, 7.0 / 2.54), tight_layout=False)
    if hasattr(figure, "set_layout_engine"):
        figure.set_layout_engine(None)
    layout = figure.add_gridspec(
        2,
        8,
        width_ratios=(1.0, 0.045, 0.18, 1.0, 0.045, 0.18, 1.0, 0.045),
        left=0.075,
        right=0.92,
        bottom=0.13,
        top=0.88,
        wspace=0.04,
        hspace=0.42,
    )
    panel_positions = ((0, 0), (0, 3), (0, 6), (1, 0), (1, 3))
    colorbar_positions = ((0, 1), (0, 4), (0, 7), (1, 1), (1, 4))
    fields = (
        (
            "theta_s",
            "$\\theta_s$",
            lambda values: values,
            ("#5E839B", "#A9C2CC", "#F1EEE6", "#E4C18C", "#C28750"),
            0.37,
            0.58,
            (0.4, 0.45, 0.5, 0.55),
        ),
        (
            "theta_r",
            "$\\theta_r$",
            lambda values: values,
            ("#758CA9", "#B6C5D3", "#F4EFF0", "#D8B0B9", "#AD7585"),
            None,
            None,
            None,
        ),
        (
            "alpha_cm_inv",
            "$\\alpha$",
            lambda values: values * 100.0,
            ("#648B90", "#B2CFCA", "#F2F1E8", "#D5C789", "#8FA05E"),
            None,
            None,
            (2.0, 4.0, 6.0),
        ),
        (
            "n",
            "$n$",
            lambda values: values,
            ("#6D8FAB", "#B7C9D9", "#F5F1E8", "#E1BD7C", "#B37D40"),
            1.07,
            1.33,
            (1.1, 1.2, 1.3),
        ),
        (
            "Ks_cm_d",
            "$K_s$",
            lambda values: values * 0.01,
            ("#6386AA", "#B1C8DD", "#F2F0EB", "#D5AD97", "#AF6A5C"),
            None,
            None,
            (0.2, 0.6, 1.0),
        ),
    )
    for index, (name, title, transform, colors, vmin, vmax, ticks) in enumerate(fields):
        row, column = panel_positions[index]
        cbar_row, cbar_column = colorbar_positions[index]
        axis = figure.add_subplot(layout[row, column])
        colorbar_axis = figure.add_subplot(layout[cbar_row, cbar_column])
        values = np.asarray(transform(ds[name].values), dtype=float)
        cmap = LinearSegmentedColormap.from_list(
            f"paper_{name}", list(zip((0.0, 0.26, 0.52, 0.76, 1.0), colors)), N=256
        )
        mesh = axis.pcolormesh(
            centers_to_edges(x_m),
            centers_to_edges(z_m),
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            antialiased=False,
        )
        axis.set_title(title)
        axis.set_xlim(-5.0, 105.0)
        axis.set_ylim(-1.2, 0.0)
        axis.set_xticks((0.0, 25.0, 50.0, 75.0, 100.0))
        axis.set_yticks((0.0, -0.4, -0.8, -1.2))
        axis.set_xlabel("x (m)" if row == 1 else "")
        axis.set_ylabel("z (m)" if column == 0 else "")
        axis.set_xticklabels(("0", "25", "50", "75", "100") if row == 1 else [])
        axis.set_yticklabels(("0", "-0.4", "-0.8", "-1.2") if column == 0 else [])
        axis.tick_params(
            width=0.6, length=1.5, labelsize=7, pad=1.2, direction="in", top=False, right=False
        )
        colorbar = figure.colorbar(mesh, cax=colorbar_axis)
        colorbar.formatter = FormatStrFormatter("%.2f")
        if ticks is not None:
            colorbar.set_ticks(ticks)
        colorbar.update_ticks()
        colorbar.ax.tick_params(width=0.6, length=1.5, labelsize=7, pad=1.2, direction="in")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    figure.savefig(
        path.with_suffix(".eps"), format="eps", dpi=600, bbox_inches="tight", pad_inches=0.0
    )
    plt.close(figure)


def write_comparison_stats(raw: xr.Dataset, trend: xr.Dataset, path: Path) -> None:
    rows = []
    for name, unit in [
        ("theta_r", "cm3/cm3"),
        ("theta_s", "cm3/cm3"),
        ("alpha_cm_inv", "cm^-1"),
        ("n", "-"),
        ("Ks_cm_d", "cm/day"),
    ]:
        raw_arr = raw[name].values.reshape(-1)
        trend_arr = trend[name].values.reshape(-1)
        rows.append(
            {
                "parameter": name,
                "unit": unit,
                "raw_mean": float(np.mean(raw_arr)),
                "trend_mean": float(np.mean(trend_arr)),
                "raw_std": float(np.std(raw_arr, ddof=1)),
                "trend_std": float(np.std(trend_arr, ddof=1)),
                "raw_min": float(np.min(raw_arr)),
                "trend_min": float(np.min(trend_arr)),
                "raw_max": float(np.max(raw_arr)),
                "trend_max": float(np.max(trend_arr)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
