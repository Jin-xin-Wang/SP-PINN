from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_NC = (
    PROJECT_DIR
    / "outputs"
    / "psd_30_50_20_domain_110x120"
    / "euptfv2_ptf01_mvg_trend_smooth_fields.nc"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "comsol_field_data"
FIELD_FILES = {
    "theta_s": "theta_s_field_domain.txt",
    "theta_r": "theta_r_field_domain.txt",
    "alpha": "alpha_field_domain.txt",
    "n": "n_field_domain.txt",
    "ks": "ks_field_domain.txt",
}


def summarize_values(values: np.ndarray, units: str) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "units": units,
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def write_xyz_field(
    path: Path, x_m: np.ndarray, z_cm: np.ndarray, values: np.ndarray
) -> np.ndarray:
    x_grid, z_grid = np.meshgrid(x_m, z_cm)
    y_grid = -z_grid / 100.0
    rows = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), np.asarray(values, dtype=np.float64).ravel()]
    )
    np.savetxt(path, rows, fmt="%.16g")
    return rows


def validate_input_dataset(ds: xr.Dataset) -> None:
    required_coords = {"x_m", "z_cm"}
    required_vars = {"theta_r", "theta_s", "alpha_cm_inv", "n", "Ks_cm_d"}
    missing_coords = sorted(required_coords.difference(ds.coords))
    missing_vars = sorted(required_vars.difference(ds.data_vars))
    if missing_coords or missing_vars:
        raise KeyError(f"Missing required coords={missing_coords} vars={missing_vars}")


def export_fields(input_nc: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(input_nc) as ds:
        validate_input_dataset(ds)
        x_m = ds["x_m"].values.astype(np.float64)
        z_cm = ds["z_cm"].values.astype(np.float64)
        fields = {
            "theta_s": (ds["theta_s"].values, "1", "theta_s"),
            "theta_r": (ds["theta_r"].values, "1", "theta_r"),
            "alpha": (ds["alpha_cm_inv"].values * 100.0, "1/m", "alpha_cm_inv*100"),
            "n": (ds["n"].values, "1", "n"),
            "ks": (ds["Ks_cm_d"].values / 100.0, "m/day", "Ks_cm_d/100"),
        }
        rows_by_field: dict[str, np.ndarray] = {}
        for key, (values, _units, _source) in fields.items():
            rows_by_field[key] = write_xyz_field(output_dir / FIELD_FILES[key], x_m, z_cm, values)
        theta_s = fields["theta_s"][0]
        theta_r = fields["theta_r"][0]
        alpha = fields["alpha"][0]
        n_field = fields["n"][0]
        ks = fields["ks"][0]
        y_values = rows_by_field["theta_s"][:, 1]
        max_composition_error = None
        if {"sand", "silt", "clay"}.issubset(ds.data_vars):
            max_composition_error = float(
                np.max(np.abs(ds["sand"].values + ds["silt"].values + ds["clay"].values - 100.0))
            )
    checks = {
        "theta_r_lt_theta_s": bool(np.all(theta_r < theta_s)),
        "alpha_positive": bool(np.all(alpha > 0.0)),
        "n_gt_1": bool(np.all(n_field > 1.0)),
        "ks_positive": bool(np.all(ks > 0.0)),
        "max_abs_sand_silt_clay_minus_100": max_composition_error,
        "field_y_min_m": float(np.min(y_values)),
        "field_y_max_m": float(np.max(y_values)),
    }
    if not all(
        [
            checks["theta_r_lt_theta_s"],
            checks["alpha_positive"],
            checks["n_gt_1"],
            checks["ks_positive"],
        ]
    ):
        raise ValueError(f"Invalid hydraulic fields: {checks}")
    summary: dict[str, object] = {
        "source_nc": str(input_nc.resolve()),
        "field_dir": str(output_dir.resolve()),
        "grid": {
            "nx": int(x_m.size),
            "nz": int(z_cm.size),
            "x_min_m": float(np.min(x_m)),
            "x_max_m": float(np.max(x_m)),
            "z_cm_min_positive_down": float(np.min(z_cm)),
            "z_cm_max_positive_down": float(np.max(z_cm)),
            "y_min_m_comsol": float(np.min(y_values)),
            "y_max_m_comsol": float(np.max(y_values)),
        },
        "unit_conversions": {
            "x": "x_m",
            "y": "-z_cm/100",
            "theta_r": "dimensionless",
            "theta_s": "dimensionless",
            "alpha": "alpha_cm_inv*100 -> 1/m",
            "n": "dimensionless",
            "Ks": "Ks_cm_d/100 -> m/day",
        },
        "variables": {
            "theta_s": summarize_values(theta_s, "1"),
            "theta_r": summarize_values(theta_r, "1"),
            "alpha": summarize_values(alpha, "1/m"),
            "n": summarize_values(n_field, "1"),
            "Ks": summarize_values(ks, "m/day"),
        },
        "checks": checks,
    }
    (output_dir / "field_scaling_metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Export AA hydraulic parameter fields from generated NetCDF to COMSOL xyz txt."
    )
    parser.add_argument("--input-nc", type=Path, default=DEFAULT_INPUT_NC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_fields(args.input_nc, args.output_dir)
    print(
        json.dumps(
            {"field_dir": summary["field_dir"], "checks": summary["checks"]},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
