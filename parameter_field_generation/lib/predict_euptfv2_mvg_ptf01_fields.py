from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
from predict_euptfv2_ptf01_fields import (
    FEATURE_NAMES,
    build_predictor_table,
    dict_get,
    load_rdata_object,
    ranger_regression_predict,
)

MVG_MODEL_URL = "https://github.com/tkdweber/euptf2/raw/1.0.0/data/MVG_PTF01.rdata"
MVG_MODEL_NAMES = {
    "theta_s": "MVG_THS_PTF01",
    "theta_r_log10_plus_1": "MVG_log10THR_1_PTF01",
    "alpha_log10": "MVG_log10ALP_PTF01",
    "n_log10_minus_1": "MVG_log10N_1_PTF01",
    "k0_log10": "MVG_log10K0_PTF01",
    "l": "MVG_L_PTF01",
}


def ensure_mvg_model(model_dir: Path) -> Path:
    path = model_dir / "MVG_PTF01.rdata"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Missing bundled EUPTFv2 MVG model file: {path}. This package is offline-only; restore the bundled MVG_PTF01.rdata file."
    )


def predict_euptfv2_mvg_ptf01_from_psd(ds: xr.Dataset, model_dir: Path) -> xr.Dataset:
    model_path = ensure_mvg_model(model_dir)
    mvg_models = load_rdata_object(model_path, "MVG_PTF01")
    predictors, shape = build_predictor_table(ds)
    theta_s = ranger_regression_predict(
        dict_get(mvg_models, MVG_MODEL_NAMES["theta_s"]), predictors, FEATURE_NAMES
    )
    theta_r_trans = ranger_regression_predict(
        dict_get(mvg_models, MVG_MODEL_NAMES["theta_r_log10_plus_1"]), predictors, FEATURE_NAMES
    )
    alpha_trans = ranger_regression_predict(
        dict_get(mvg_models, MVG_MODEL_NAMES["alpha_log10"]), predictors, FEATURE_NAMES
    )
    n_trans = ranger_regression_predict(
        dict_get(mvg_models, MVG_MODEL_NAMES["n_log10_minus_1"]), predictors, FEATURE_NAMES
    )
    log10_k0 = ranger_regression_predict(
        dict_get(mvg_models, MVG_MODEL_NAMES["k0_log10"]), predictors, FEATURE_NAMES
    )
    l_param = ranger_regression_predict(
        dict_get(mvg_models, MVG_MODEL_NAMES["l"]), predictors, FEATURE_NAMES
    )
    theta_r = np.power(10.0, theta_r_trans) - 1.0
    alpha = np.power(10.0, alpha_trans)
    n_param = np.power(10.0, n_trans) + 1.0
    k0 = np.power(10.0, log10_k0)
    m_param = 1.0 - 1.0 / n_param
    depth_cm = predictors[:, FEATURE_NAMES.index("DEPTH_M")]
    out = xr.Dataset(coords=ds.coords, attrs=dict(ds.attrs))
    out.attrs.update(
        {
            "title": "EUPTFv2 PTF01 MVG fields predicted from x-z PSD and depth",
            "model": "euptfv2 / euptf2 official MVG_PTF01 random forest objects",
            "ptf": "PTF01",
            "target": "MVG",
            "predictors": "USSAND, USSILT, USCLAY, DEPTH_M",
            "depth_definition": "The official predictor is named DEPTH_M, but euptf2 sample_data uses cm-scale depth values; here DEPTH_M is the z_cm cell center.",
            "source_package": "tkdweber/euptf2 version 1.0.0",
            "source_url": MVG_MODEL_URL,
            "transforms": "theta_s=THS; theta_r=10^MVG_log10THR_1-1; alpha=10^MVG_log10ALP; n=10^MVG_log10N_1+1; K0=10^MVG_log10K0; L=L",
            "ks_note": "Ks_cm_d is set equal to MVG K0_cm_d for five-field VG+K usage. This is not the separate KS_PTF01 target.",
        }
    )
    for source_name in ("sand", "silt", "clay"):
        out[source_name] = ds[source_name]
        out[source_name].attrs["units"] = "%"
    variables = {
        "depth_cm": (depth_cm, "cm"),
        "theta_r": (theta_r, "cm3 cm-3"),
        "theta_s": (theta_s, "cm3 cm-3"),
        "alpha_cm_inv": (alpha, "cm-1"),
        "n": (n_param, "-"),
        "m": (m_param, "-"),
        "K0_cm_d": (k0, "cm d-1"),
        "Ks_cm_d": (k0, "cm d-1"),
        "L": (l_param, "-"),
        "log10_Ks_cm_d": (log10_k0, "log10(cm d-1)"),
        "log10_alpha_cm_inv": (alpha_trans, "log10(cm-1)"),
    }
    for name, (flat_values, unit) in variables.items():
        out[name] = (("z_cm", "x_m"), np.asarray(flat_values).reshape(shape))
        out[name].attrs["units"] = unit
    out["Ks_cm_d"].attrs["long_name"] = "MVG K0 used as saturated hydraulic conductivity field"
    out["K0_cm_d"].attrs["long_name"] = "Mualem-van Genuchten K0"
    return out
