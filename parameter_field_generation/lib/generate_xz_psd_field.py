from __future__ import annotations
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

EPS = 1e-09
ILR_BASIS = np.array(
    [
        [math.sqrt(1.0 / 2.0), -math.sqrt(1.0 / 2.0), 0.0],
        [math.sqrt(1.0 / 6.0), math.sqrt(1.0 / 6.0), -math.sqrt(2.0 / 3.0)],
    ],
    dtype=float,
)


def composition_to_ilr(parts: np.ndarray) -> np.ndarray:
    parts = np.asarray(parts, dtype=float)
    parts = np.clip(parts, EPS, None)
    parts = parts / parts.sum(axis=-1, keepdims=True)
    return np.log(parts) @ ILR_BASIS.T


def ilr_to_composition(z: np.ndarray) -> np.ndarray:
    log_parts = np.asarray(z, dtype=float) @ ILR_BASIS
    log_parts = log_parts - log_parts.max(axis=-1, keepdims=True)
    parts = np.exp(log_parts)
    return parts / parts.sum(axis=-1, keepdims=True)


def enforce_composition_floor(parts: np.ndarray, floor: float) -> np.ndarray:
    parts = np.maximum(parts, floor)
    return parts / parts.sum(axis=-1, keepdims=True)


def nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return a positive-definite correlation matrix close to matrix."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, 1e-06)
    repaired = vecs * vals @ vecs.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def standardize(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    std = float(field.std())
    if std < EPS:
        return field - field.mean()
    return (field - field.mean()) / std


@dataclass(frozen=True)
class XZPSDConfig:
    nx: int = 110
    nz: int = 120
    dx_m: float = 1.0
    dz_cm: float = 1.0
    seed: int = 20260519
    horizontal_range_m: float = 30.0
    vertical_range_cm: float = 20.0
    covariance_model: str = "gaussian_rff"
    rff_modes: int = 2048
    top_psd_percent: tuple[float, float, float] = (30.0, 50.0, 20.0)
    bottom_psd_percent: tuple[float, float, float] = (30.0, 50.0, 20.0)
    ilr_sd: tuple[float, float] = (0.3, 0.2)
    ilr_corr: float = -0.2
    min_fraction: float = 0.02


def anisotropic_grf(cfg: XZPSDConfig, rng: np.random.Generator) -> np.ndarray:
    x_m = (np.arange(cfg.nx, dtype=float) + 0.5) * cfg.dx_m
    z_cm = (np.arange(cfg.nz, dtype=float) + 0.5) * cfg.dz_cm
    zz, xx = np.meshgrid(z_cm, x_m, indexing="ij")
    modes = int(cfg.rff_modes)
    kx = rng.normal(0.0, math.sqrt(2.0) / max(cfg.horizontal_range_m, 1e-09), size=modes)
    kz = rng.normal(0.0, math.sqrt(2.0) / max(cfg.vertical_range_cm, 1e-09), size=modes)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=modes)
    weights = rng.standard_normal(modes)
    field = np.zeros((cfg.nz, cfg.nx), dtype=float)
    batch_size = 128
    scale = math.sqrt(2.0 / modes)
    for start in range(0, modes, batch_size):
        stop = min(start + batch_size, modes)
        arg = xx[:, :, None] * kx[start:stop] + zz[:, :, None] * kz[start:stop] + phase[start:stop]
        field += scale * np.sum(weights[start:stop] * np.cos(arg), axis=2)
    return standardize(field)


def generate_xz_psd(cfg: XZPSDConfig) -> xr.Dataset:
    rng = np.random.default_rng(cfg.seed)
    corr = nearest_correlation(np.array([[1.0, cfg.ilr_corr], [cfg.ilr_corr, 1.0]]))
    chol = np.linalg.cholesky(corr)
    independent = np.stack([anisotropic_grf(cfg, rng), anisotropic_grf(cfg, rng)], axis=0)
    mixed = (chol @ independent.reshape(2, -1)).reshape(2, cfg.nz, cfg.nx)
    z_norm = np.linspace(0.0, 1.0, cfg.nz)
    top = np.asarray(cfg.top_psd_percent, dtype=float) / 100.0
    bottom = np.asarray(cfg.bottom_psd_percent, dtype=float) / 100.0
    mean_comp = top[None, :] * (1.0 - z_norm[:, None]) + bottom[None, :] * z_norm[:, None]
    mean_comp = mean_comp / mean_comp.sum(axis=1, keepdims=True)
    mean_ilr = composition_to_ilr(mean_comp)
    sd = np.asarray(cfg.ilr_sd, dtype=float)[:, None, None]
    ilr = mean_ilr.T[:, :, None] + sd * mixed
    comp = ilr_to_composition(np.moveaxis(ilr, 0, -1))
    comp = enforce_composition_floor(comp, cfg.min_fraction)
    x_m = (np.arange(cfg.nx, dtype=float) + 0.5) * cfg.dx_m
    z_cm = (np.arange(cfg.nz, dtype=float) + 0.5) * cfg.dz_cm
    ds = xr.Dataset(
        data_vars={
            "sand": (("z_cm", "x_m"), comp[:, :, 0] * 100.0),
            "silt": (("z_cm", "x_m"), comp[:, :, 1] * 100.0),
            "clay": (("z_cm", "x_m"), comp[:, :, 2] * 100.0),
            "ilr_1": (("z_cm", "x_m"), ilr[0]),
            "ilr_2": (("z_cm", "x_m"), ilr[1]),
        },
        coords={"z_cm": z_cm, "x_m": x_m},
        attrs={
            "title": "Synthetic x-z PSD compositional random field",
            "config_json": json.dumps(asdict(cfg), ensure_ascii=True),
            "composition_order": "sand, silt, clay",
            "z_positive": "downward",
            "domain": f"{cfg.nx * cfg.dx_m:g} m horizontal by {cfg.nz * cfg.dz_cm:g} cm depth, stored as x_m and z_cm cell centers",
            "method": "anisotropic logistic-normal random field in ilr space with Gaussian random Fourier feature covariance simulation",
            "covariance_note": "Gaussian kernel C(r)=exp(-r^2); simulated directly by random Fourier features, not periodic FFT convolution.",
        },
    )
    for var in ("sand", "silt", "clay"):
        ds[var].attrs["units"] = "%"
    return ds


def write_csv(ds: xr.Dataset, path: Path) -> Path:
    df = ds[["sand", "silt", "clay"]].to_dataframe().reset_index()
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def write_summary(ds: xr.Dataset, cfg: XZPSDConfig, path: Path) -> None:
    psd_sum = ds["sand"].values + ds["silt"].values + ds["clay"].values
    summary = {
        "config": asdict(cfg),
        "max_abs_sum_error_percent": float(np.max(np.abs(psd_sum - 100.0))),
        "variables": {},
        "correlations": {
            "sand_silt": float(
                np.corrcoef(ds["sand"].values.reshape(-1), ds["silt"].values.reshape(-1))[0, 1]
            ),
            "sand_clay": float(
                np.corrcoef(ds["sand"].values.reshape(-1), ds["clay"].values.reshape(-1))[0, 1]
            ),
            "silt_clay": float(
                np.corrcoef(ds["silt"].values.reshape(-1), ds["clay"].values.reshape(-1))[0, 1]
            ),
        },
    }
    for var in ("sand", "silt", "clay"):
        arr = ds[var].values
        summary["variables"][var] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
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


def summarize_psd_statistics(ds: xr.Dataset) -> pd.DataFrame:
    dx = float(ds["x_m"].values[1] - ds["x_m"].values[0])
    dz = float(ds["z_cm"].values[1] - ds["z_cm"].values[0])
    rows = []
    for var in ("sand", "silt", "clay"):
        arr = ds[var].values
        acf_x = mean_autocorrelation(arr, axis=1)
        acf_z = mean_autocorrelation(arr, axis=0)
        flat = arr.reshape(-1)
        rows.append(
            {
                "variable": var,
                "unit": "%",
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat, ddof=1)),
                "variance": float(np.var(flat, ddof=1)),
                "min": float(np.min(flat)),
                "p01": float(np.percentile(flat, 1)),
                "p05": float(np.percentile(flat, 5)),
                "p50": float(np.percentile(flat, 50)),
                "p95": float(np.percentile(flat, 95)),
                "p99": float(np.percentile(flat, 99)),
                "max": float(np.max(flat)),
                "corr_length_x_m": efold_length(acf_x, dx),
                "corr_length_z_cm": efold_length(acf_z, dz),
            }
        )
    return pd.DataFrame(rows)


def plot_profile(ds: xr.Dataset, path: Path) -> None:
    configure_paper_style()
    x_m = np.asarray(ds["x_m"].values, dtype=float)
    z_m = -np.asarray(ds["z_cm"].values, dtype=float) / 100.0
    figure = plt.figure(figsize=(14.0 / 2.54, 5.4 / 2.54), tight_layout=False)
    if hasattr(figure, "set_layout_engine"):
        figure.set_layout_engine(None)
    layout = figure.add_gridspec(
        1,
        8,
        width_ratios=(1.0, 0.045, 0.15, 1.0, 0.045, 0.15, 1.0, 0.045),
        left=0.075,
        right=0.96,
        bottom=0.24,
        top=0.88,
        wspace=0.04,
    )
    map_columns, colorbar_columns = ((0, 3, 6), (1, 4, 7))
    axes = [figure.add_subplot(layout[0, column]) for column in map_columns]
    colorbar_axes = [figure.add_subplot(layout[0, column]) for column in colorbar_columns]
    fields = (
        ("sand", "(a) Sand (%)", ("#FFFFFF", "#FFF0B0", "#E8841A", "#8B3008"), (20.0, 30.0, 40.0)),
        ("silt", "(b) Silt (%)", ("#FFFFFF", "#DFF3EF", "#51A99B", "#0B5F59"), (30.0, 50.0, 70.0)),
        ("clay", "(c) Clay (%)", ("#FFFFFF", "#F7DCEC", "#B54D92", "#5D1C51"), (12.0, 20.0, 28.0)),
    )
    for index, (name, title, colors, ticks) in enumerate(fields):
        cmap = LinearSegmentedColormap.from_list(
            name, list(zip((0.0, 0.42, 0.7, 1.0), colors)), N=256
        )
        values = np.asarray(ds[name].values, dtype=float)
        mesh = axes[index].pcolormesh(
            centers_to_edges(x_m),
            centers_to_edges(z_m),
            values,
            shading="auto",
            cmap=cmap,
            vmin=float(np.min(values)),
            vmax=float(np.max(values)),
            rasterized=True,
            antialiased=False,
        )
        axis = axes[index]
        axis.text(
            0.02,
            0.97,
            title,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#1E2929",
        )
        axis.set_xlim(-5.0, 105.0)
        axis.set_ylim(-1.2, 0.0)
        axis.set_box_aspect(0.7)
        axis.set_xticks((0.0, 25.0, 50.0, 75.0, 100.0))
        axis.set_yticks((0.0, -0.4, -0.8, -1.2))
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)" if index == 0 else "")
        axis.set_yticklabels(("0", "-0.4", "-0.8", "-1.2") if index == 0 else [])
        axis.tick_params(direction="in", top=False, right=False, width=0.6, length=1.5, pad=1.2)
        colorbar = figure.colorbar(mesh, cax=colorbar_axes[index], orientation="vertical")
        colorbar.set_ticks(ticks)
        colorbar.ax.tick_params(direction="in", width=0.6, length=1.5, labelsize=7, pad=1.2)
    figure.canvas.draw()
    for axis, colorbar_axis in zip(axes, colorbar_axes):
        map_position = axis.get_position()
        colorbar_position = colorbar_axis.get_position()
        colorbar_axis.set_position(
            [
                colorbar_position.x0 + 0.007,
                map_position.y0,
                colorbar_position.width,
                map_position.height,
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    figure.savefig(
        path.with_suffix(".eps"), format="eps", dpi=600, bbox_inches="tight", pad_inches=0.0
    )
    plt.close(figure)
