from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import xarray as xr

PROJECT_DIR = Path(__file__).resolve().parent
LIB_DIR = PROJECT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
from generate_xz_psd_field import (
    XZPSDConfig,
    generate_xz_psd,
    plot_profile,
    summarize_psd_statistics,
    write_csv as write_psd_csv,
    write_summary as write_psd_summary,
)
from predict_euptfv2_mvg_ptf01_fields import predict_euptfv2_mvg_ptf01_from_psd
from smooth_euptfv2_mvg_trend_fields import (
    make_dataset,
    plot_maps_actual,
    smooth_transformed_fields,
    write_comparison_stats,
)
from predict_euptfv2_ptf01_fields import (
    summarize_statistics,
    write_csv as write_euptf_csv,
    write_summary as write_euptf_summary,
)

OUT_DIR = PROJECT_DIR / "outputs" / "psd_30_50_20_domain_110x120"
FIG_DIR = PROJECT_DIR / "figures" / "psd_30_50_20_euptfv2_smooth_domain_110x120"
MODEL_DIR = PROJECT_DIR / "data" / "euptf2_official"


def shift_x_domain(ds: xr.Dataset, x_min_m: float) -> xr.Dataset:
    out = ds.copy()
    out = out.assign_coords(x_m=out["x_m"] + x_min_m)
    out.attrs["domain"] = (
        "110 m horizontal by 120 cm depth, stored as x_m and z_cm cell centers; x domain shifted to -5..105 m"
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = XZPSDConfig(
        nx=110,
        nz=120,
        dx_m=1.0,
        dz_cm=1.0,
        seed=20260519,
        horizontal_range_m=30.0,
        vertical_range_cm=20.0,
        covariance_model="gaussian_rff",
        rff_modes=2048,
        top_psd_percent=(30.0, 50.0, 20.0),
        bottom_psd_percent=(30.0, 50.0, 20.0),
        ilr_sd=(0.3, 0.2),
        ilr_corr=-0.2,
        min_fraction=0.02,
    )
    ds_psd = shift_x_domain(generate_xz_psd(cfg), x_min_m=-5.0)
    psd_nc = OUT_DIR / "xz_psd_field.nc"
    psd_csv = OUT_DIR / "xz_psd_field.csv"
    psd_summary = OUT_DIR / "xz_psd_field_summary.json"
    psd_stats = OUT_DIR / "psd_statistics.csv"
    ds_psd.to_netcdf(psd_nc)
    write_psd_csv(ds_psd, psd_csv)
    write_psd_summary(ds_psd, cfg, psd_summary)
    stats_psd = summarize_psd_statistics(ds_psd)
    stats_psd.to_csv(psd_stats, index=False)
    plot_profile(ds_psd, FIG_DIR / "PSD_spatial_distribution_1x3.png")
    ds_raw = predict_euptfv2_mvg_ptf01_from_psd(ds_psd, model_dir=MODEL_DIR)
    raw_nc = OUT_DIR / "euptfv2_ptf01_mvg_fields.nc"
    raw_csv = OUT_DIR / "euptfv2_ptf01_mvg_fields.csv"
    raw_summary = OUT_DIR / "euptfv2_ptf01_mvg_fields_summary.json"
    raw_stats_path = OUT_DIR / "euptfv2_ptf01_mvg_statistics_ln.csv"
    raw_stats = summarize_statistics(ds_raw)
    ds_raw.to_netcdf(raw_nc)
    write_euptf_csv(ds_raw, raw_csv)
    write_euptf_summary(ds_raw, raw_stats, raw_summary)
    raw_stats.to_csv(raw_stats_path, index=False)
    smoothed, smooth_meta = smooth_transformed_fields(ds_raw, sigma_x_m=5.0, sigma_z_cm=5.0)
    ds_trend = make_dataset(ds_raw, smoothed, smooth_meta)
    trend_stats = summarize_statistics(ds_trend)
    trend_nc = OUT_DIR / "euptfv2_ptf01_mvg_trend_smooth_fields.nc"
    trend_csv = OUT_DIR / "euptfv2_ptf01_mvg_trend_smooth_fields.csv"
    trend_summary = OUT_DIR / "euptfv2_ptf01_mvg_trend_smooth_fields_summary.json"
    trend_stats_path = OUT_DIR / "euptfv2_ptf01_mvg_trend_smooth_statistics_ln.csv"
    trend_compare = OUT_DIR / "euptfv2_ptf01_mvg_trend_smooth_raw_vs_trend_statistics.csv"
    ds_trend.to_netcdf(trend_nc)
    write_euptf_csv(ds_trend, trend_csv)
    write_euptf_summary(ds_trend, trend_stats, trend_summary)
    trend_stats.to_csv(trend_stats_path, index=False)
    write_comparison_stats(ds_raw, ds_trend, trend_compare)
    plot_maps_actual(ds_trend, FIG_DIR / "hydraulic_parameter_fields_physical_2x3.png")
    manifest = {
        "domain": {
            "x_min_m": -5.0,
            "x_max_m": 105.0,
            "z_top_m": 0.0,
            "z_bottom_m": -1.2,
            "nx": 110,
            "nz": 120,
            "dx_m": 1.0,
            "dz_m": 0.01,
            "note": "110x120 grid: x cell centers every 1 m, z cell centers every 1 cm.",
        },
        "psd": {
            "sand_silt_clay_percent": [30.0, 50.0, 20.0],
            "ilr_sd": [0.3, 0.2],
            "ilr_corr": -0.2,
            "seed": 20260519,
            "horizontal_range_m": 30.0,
            "vertical_range_cm": 20.0,
        },
        "ptf": "EUPTFv2 PTF01 MVG",
        "smoothing": {"sigma_x_m": 5.0, "sigma_z_cm": 5.0, **smooth_meta},
        "outputs": {
            "psd_nc": str(psd_nc.resolve()),
            "raw_euptfv2_nc": str(raw_nc.resolve()),
            "trend_smooth_nc": str(trend_nc.resolve()),
            "trend_smooth_csv": str(trend_csv.resolve()),
            "figures": str(FIG_DIR.resolve()),
            "psd_spatial_figure": str((FIG_DIR / "PSD_spatial_distribution_1x3.png").resolve()),
            "parameter_fields_physical_figure": str(
                (FIG_DIR / "hydraulic_parameter_fields_physical_2x3.png").resolve()
            ),
        },
        "trend_statistics": trend_stats.to_dict(orient="records"),
    }
    (OUT_DIR / "domain_110x120_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote expanded PSD/EUPTFv2 smooth field -> {trend_nc}")
    print(f"Wrote figures -> {FIG_DIR}")
    print(pd.DataFrame(manifest["trend_statistics"]).to_string(index=False))


if __name__ == "__main__":
    main()
