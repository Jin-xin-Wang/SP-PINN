from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "psd_30_50_20_domain_110x120"
DEFAULT_COMSOL_FIELD_DIR = PROJECT_DIR / "outputs" / "comsol_field_data"
DEFAULT_REFERENCE_ROI_DIR = PACKAGE_DIR / "reference_data" / "parameter_fields_roi"
FIELD_SPECS = {
    "theta_s": ("theta_s", 1.0, "theta_s_field_domain.txt", "theta_s_field_roi.txt"),
    "theta_r": ("theta_r", 1.0, "theta_r_field_domain.txt", "theta_r_field_roi.txt"),
    "alpha": ("alpha_cm_inv", 100.0, "alpha_field_domain.txt", "alpha_field_roi.txt"),
    "n": ("n", 1.0, "n_field_domain.txt", "n_field_roi.txt"),
    "ks": ("Ks_cm_d", 0.01, "ks_field_domain.txt", "ks_field_roi.txt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate locally generated full-domain hydraulic fields without modifying SP-PINN reference data."
    )
    parser.add_argument(
        "--input-nc",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "euptfv2_ptf01_mvg_trend_smooth_fields.nc",
    )
    parser.add_argument("--comsol-field-dir", type=Path, default=DEFAULT_COMSOL_FIELD_DIR)
    parser.add_argument("--reference-roi-dir", type=Path, default=DEFAULT_REFERENCE_ROI_DIR)
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_OUTPUT_DIR / "validation_report.json"
    )
    parser.add_argument("--tolerance", type=float, default=1e-06)
    return parser.parse_args()


def load_triplets(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing field file: {path}")
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or (not np.all(np.isfinite(values))):
        raise ValueError(f"Expected finite x, z, value triplets in: {path}")
    return values


def sort_triplets(values: np.ndarray) -> np.ndarray:
    return values[np.lexsort((values[:, 0], values[:, 1]))]


def generated_roi_triplets(ds: xr.Dataset, variable: str, scale: float) -> np.ndarray:
    x_m = np.asarray(ds["x_m"].values, dtype=np.float64)
    z_cm = np.asarray(ds["z_cm"].values, dtype=np.float64)
    x_grid, z_cm_grid = np.meshgrid(x_m, z_cm)
    z_m_grid = -z_cm_grid / 100.0
    values = np.asarray(ds[variable].values, dtype=np.float64) * scale
    mask = (x_grid >= 0.0) & (x_grid <= 100.0) & (z_m_grid >= -0.6) & (z_m_grid <= 0.0)
    return np.column_stack((x_grid[mask], z_m_grid[mask], values[mask]))


def coordinate_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.shape[0]),
        "x_min_m": float(np.min(values[:, 0])),
        "x_max_m": float(np.max(values[:, 0])),
        "z_min_m": float(np.min(values[:, 1])),
        "z_max_m": float(np.max(values[:, 1])),
    }


def validate(args: argparse.Namespace) -> dict[str, object]:
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")
    if not args.input_nc.is_file():
        raise FileNotFoundError(f"Missing generated NetCDF: {args.input_nc}")
    with xr.open_dataset(args.input_nc) as dataset:
        required_variables = {spec[0] for spec in FIELD_SPECS.values()}
        missing_variables = sorted(required_variables.difference(dataset.data_vars))
        if missing_variables:
            raise KeyError(f"Missing generated parameter variables: {missing_variables}")
        x_m = np.asarray(dataset["x_m"].values, dtype=np.float64)
        z_cm = np.asarray(dataset["z_cm"].values, dtype=np.float64)
        expected_shape = (120, 110)
        shape_checks = {
            key: list(np.asarray(dataset[variable].values).shape) == list(expected_shape)
            for key, (variable, *_rest) in FIELD_SPECS.items()
        }
        full_domain = {
            "x_count": int(x_m.size),
            "z_count": int(z_cm.size),
            "x_min_m": float(x_m.min()),
            "x_max_m": float(x_m.max()),
            "z_cm_min_positive_down": float(z_cm.min()),
            "z_cm_max_positive_down": float(z_cm.max()),
            "field_shape": list(expected_shape),
        }
        domain_ok = bool(
            x_m.size == 110
            and z_cm.size == 120
            and np.isclose(x_m.min(), -4.5)
            and np.isclose(x_m.max(), 104.5)
            and np.isclose(z_cm.min(), 0.5)
            and np.isclose(z_cm.max(), 119.5)
        )
        theta_s = np.asarray(dataset["theta_s"].values, dtype=np.float64)
        theta_r = np.asarray(dataset["theta_r"].values, dtype=np.float64)
        alpha = np.asarray(dataset["alpha_cm_inv"].values, dtype=np.float64)
        n_value = np.asarray(dataset["n"].values, dtype=np.float64)
        ks = np.asarray(dataset["Ks_cm_d"].values, dtype=np.float64)
        physical_checks = {
            "all_finite": bool(
                all(
                    (
                        np.all(np.isfinite(values))
                        for values in (theta_s, theta_r, alpha, n_value, ks)
                    )
                )
            ),
            "theta_r_lt_theta_s": bool(np.all(theta_r < theta_s)),
            "alpha_positive": bool(np.all(alpha > 0.0)),
            "n_gt_1": bool(np.all(n_value > 1.0)),
            "ks_positive": bool(np.all(ks > 0.0)),
        }
        roi_comparison: dict[str, object] = {}
        for field_name, (variable, scale, _domain_file, roi_file) in FIELD_SPECS.items():
            generated = sort_triplets(generated_roi_triplets(dataset, variable, scale))
            reference = sort_triplets(load_triplets(args.reference_roi_dir / roi_file))
            if generated.shape != reference.shape:
                raise ValueError(
                    f"ROI shape mismatch for {field_name}: generated={generated.shape}, reference={reference.shape}"
                )
            coordinate_max_abs_diff = float(np.max(np.abs(generated[:, :2] - reference[:, :2])))
            value_max_abs_diff = float(np.max(np.abs(generated[:, 2] - reference[:, 2])))
            roi_comparison[field_name] = {
                "generated_grid": coordinate_summary(generated),
                "reference_grid": coordinate_summary(reference),
                "coordinate_max_abs_diff": coordinate_max_abs_diff,
                "value_max_abs_diff": value_max_abs_diff,
                "within_tolerance": bool(
                    coordinate_max_abs_diff <= args.tolerance
                    and value_max_abs_diff <= args.tolerance
                ),
            }
    comsol_exports: dict[str, object] = {}
    for field_name, (_variable, _scale, domain_file, _roi_file) in FIELD_SPECS.items():
        values = load_triplets(args.comsol_field_dir / domain_file)
        comsol_exports[field_name] = {
            **coordinate_summary(values),
            "complete_110x120_grid": bool(values.shape == (13200, 3)),
        }
    all_roi_matches = all((item["within_tolerance"] for item in roi_comparison.values()))
    all_export_grids = all((item["complete_110x120_grid"] for item in comsol_exports.values()))
    all_physical_checks = all(physical_checks.values())
    all_shape_checks = all(shape_checks.values())
    status = (
        "ok"
        if domain_ok
        and all_shape_checks
        and all_physical_checks
        and all_export_grids
        and all_roi_matches
        else "failed"
    )
    return {
        "status": status,
        "tolerance": float(args.tolerance),
        "input_nc": str(args.input_nc.resolve()),
        "comsol_field_dir": str(args.comsol_field_dir.resolve()),
        "reference_roi_dir": str(args.reference_roi_dir.resolve()),
        "full_domain": full_domain,
        "domain_ok": domain_ok,
        "shape_checks": shape_checks,
        "physical_checks": physical_checks,
        "comsol_exports": comsol_exports,
        "roi_comparison": roi_comparison,
    }


def main() -> None:
    args = parse_args()
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report.resolve())}, indent=2))
    if report["status"] != "ok":
        raise SystemExit("Validation failed; see the generated report for details.")


if __name__ == "__main__":
    main()
