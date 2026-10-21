import torch
from model import PARAMETER_LOSS_FIELDS


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True
    )[0]


def _scales(dimless):
    sx = float(dimless.get("sx", dimless["Lx"]))
    return {
        "Lx": float(dimless["Lx"]),
        "Lz": float(dimless["Lz"]),
        "H0": float(dimless["H0"]),
        "Theta0": float(dimless["Theta0"]),
        "K0": float(dimless["K0"]),
        "Cx": float(dimless["Cx"]),
        "Cz": float(dimless["Cz"]),
        "gamma_x": float(dimless["H0"]) / sx,
        "gamma_z": float(dimless["H0"]) / float(dimless["Lz"]),
    }


def se_to_theta(se, theta_s, theta_r):
    return theta_r + (theta_s - theta_r) * se


def se_to_h(se, alpha, n, m):
    se_safe = torch.clamp(se, min=1e-08, max=1.0 - 1e-08)
    alpha_safe = torch.clamp(alpha, min=1e-08)
    n_safe = torch.clamp(n, min=1.01 + 1e-06, max=10.0)
    m_safe = torch.clamp(m, min=0.01, max=0.999)
    term = torch.pow(se_safe, -1.0 / m_safe) - 1.0
    h_unsat = -(1.0 / alpha_safe) * torch.pow(torch.clamp(term, min=1e-12), 1.0 / n_safe)
    return torch.where(se < 1.0, h_unsat, torch.zeros_like(se))


def se_to_kr(se, m, l=0.5):
    se_safe = torch.clamp(se, min=1e-08, max=1.0)
    m_safe = torch.clamp(m, min=0.01, max=0.999)
    inner = 1.0 - torch.pow(1.0 - torch.pow(se_safe, 1.0 / m_safe), m_safe)
    return torch.pow(se_safe, l) * torch.pow(torch.clamp(inner, min=0.0, max=1.0), 2.0)


def evaluate_parameter_fields(field_model, xz_dim):
    return field_model.evaluate_fields(xz_dim)


def compute_state(se_model, field_model, coords_dim, dimless, require_spatial=False):
    scales = _scales(dimless)
    coords_use = coords_dim.clone().detach().requires_grad_(True) if require_spatial else coords_dim
    local = evaluate_parameter_fields(field_model, coords_use[:, 0:2])
    se = se_model(coords_use)
    h = se_to_h(se, local["alpha"], local["n"], local["m"])
    psi = h / scales["H0"]
    kr_h = se_to_kr(se, local["m"], l=0.5)
    k_h = local["ks_eff"] * kr_h
    state = {
        "coords": coords_use,
        "se": se,
        "h": h,
        "psi": psi,
        "theta_h": se_to_theta(se, local["theta_s"], local["theta_r"]),
        "kr_h": kr_h,
        "ks_eff": local["ks_eff"],
        "k_h": k_h,
        "kappa_s": local["ks_eff"] / scales["K0"],
        "kappa": k_h / scales["K0"],
        "theta_s": local["theta_s"],
        "theta_r": local["theta_r"],
        "alpha": local["alpha"],
        "lnalpha": local["lnalpha"],
        "n": local["n"],
        "lnn1": local["lnn1"],
        "m": local["m"],
        "lnks": local["lnks"],
        "delta_theta_ratio": local["delta_theta"] / scales["Theta0"],
    }
    if not require_spatial:
        return state
    se_grad = grad(se, coords_use)
    psi_grad = grad(psi, coords_use)
    se_tau = se_grad[:, 2:3]
    psi_xi = psi_grad[:, 0:1]
    psi_zeta = psi_grad[:, 1:2]
    qx_core = -state["kappa"] * psi_xi
    qx_star = scales["gamma_x"] * qx_core
    qz_star = -state["kappa"] * (scales["gamma_z"] * psi_zeta + 1.0)
    state.update(
        {
            "se_tau": se_tau,
            "psi_xi": psi_xi,
            "psi_zeta": psi_zeta,
            "h_x": scales["gamma_x"] * psi_xi,
            "h_z": scales["gamma_z"] * psi_zeta,
            "qx_core": qx_core,
            "qx_star": qx_star,
            "qz_star": qz_star,
            "qx_h": scales["K0"] * qx_star,
            "qz_h": scales["K0"] * qz_star,
        }
    )
    return state


def compute_pde_loss(se_model, field_model, coords_dim, dimless):
    scales = _scales(dimless)
    state = compute_state(se_model, field_model, coords_dim, dimless, require_spatial=True)
    coords = state["coords"]
    qx_core_xi = grad(state["qx_core"], coords)[:, 0:1]
    qz_star_zeta = grad(state["qz_star"], coords)[:, 1:2]
    residual = (
        state["delta_theta_ratio"] * state["se_tau"]
        + scales["Cx"] * qx_core_xi
        + scales["Cz"] * qz_star_zeta
    )
    return torch.mean(residual.pow(2))


def compute_probe_variable_loss(
    se_model,
    field_model,
    probe_coords_dim,
    probe_targets_norm,
    probe_stats,
    dimless,
    observed_variable,
):
    state = compute_state(se_model, field_model, probe_coords_dim, dimless, require_spatial=False)
    if observed_variable not in state:
        raise KeyError(f"Observed variable {observed_variable!r} not available in state outputs.")
    value_pred_norm = (state[observed_variable] - float(probe_stats["min"])) / float(
        probe_stats["range"]
    )
    return torch.mean((value_pred_norm - probe_targets_norm) ** 2)


def compute_parameter_point_losses(
    field_model,
    point_coords_dim,
    point_targets,
    point_stats,
    active_fields=None,
    field_weights=None,
):
    raw_prediction = field_model.raw_field_outputs(point_coords_dim)
    active_fields = set(PARAMETER_LOSS_FIELDS if active_fields is None else active_fields)
    field_weights = {} if field_weights is None else dict(field_weights)
    components = {}
    zero = torch.sum(point_coords_dim[:, :1]) * 0.0
    total = zero
    for field_name in PARAMETER_LOSS_FIELDS:
        if field_name not in active_fields:
            components[field_name] = zero
            continue
        pred = raw_prediction[field_name]
        target = point_targets[field_name]
        components[field_name] = torch.mean((pred - target) ** 2)
        total = total + float(field_weights.get(field_name, 1.0)) * components[field_name]
    return (total, components)


def sample_training_points(dimless, n_residual, device, tau_max_day=None):
    tau_end = float(dimless["tau_end"])
    if tau_max_day is None:
        tau_max = tau_end
    else:
        tau_max = min(float(tau_end), float(tau_max_day) / float(dimless["T"]))
        tau_max = max(tau_max, 0.0)
    return torch.cat(
        [
            -1.0 + 2.0 * torch.rand((n_residual, 1), device=device),
            torch.rand((n_residual, 1), device=device),
            tau_max * torch.rand((n_residual, 1), device=device),
        ],
        dim=1,
    )
