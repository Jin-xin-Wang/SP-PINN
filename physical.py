import torch

from model import PARAMETER_LOSS_FIELDS, transform_raw_parameter_outputs


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
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


def h_to_se(h, alpha, n, m):
    se = torch.ones_like(h)
    mask = h < 0.0
    if torch.any(mask):
        h_abs = torch.abs(h[mask]).clamp_min(1.0e-12)
        se[mask] = torch.pow(1.0 + torch.pow(alpha[mask] * h_abs, n[mask]), -m[mask])
    return se


def se_to_theta(se, theta_s, theta_r):
    return theta_r + (theta_s - theta_r) * se


def se_to_h(se, alpha, n, m):
    se_safe = torch.clamp(se, min=1.0e-8, max=1.0 - 1.0e-8)
    alpha_safe = torch.clamp(alpha, min=1.0e-8)
    n_safe = torch.clamp(n, min=1.01 + 1.0e-6, max=10.0)
    m_safe = torch.clamp(m, min=0.01, max=0.999)
    term = torch.pow(se_safe, -1.0 / m_safe) - 1.0
    h_unsat = -(1.0 / alpha_safe) * torch.pow(torch.clamp(term, min=1.0e-12), 1.0 / n_safe)
    return torch.where(se < 1.0, h_unsat, torch.zeros_like(se))


def se_to_kr(se, m, l=0.5):
    se_safe = torch.clamp(se, min=1.0e-8, max=1.0)
    m_safe = torch.clamp(m, min=0.01, max=0.999)
    inner = 1.0 - torch.pow(1.0 - torch.pow(se_safe, 1.0 / m_safe), m_safe)
    return torch.pow(se_safe, l) * torch.pow(torch.clamp(inner, min=0.0, max=1.0), 2.0)


def evaluate_parameter_fields(field_model, xz_dim):
    if hasattr(field_model, "evaluate_fields"):
        return field_model.evaluate_fields(xz_dim)

    raw = field_model.raw_field_outputs(xz_dim)
    return transform_raw_parameter_outputs(raw)


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


def compute_initial_head_loss(se_model, field_model, coords_dim, z_phys, dimless, total_head_m):
    """Constrain the physical initial total head h + z to a prescribed value."""

    state = compute_state(se_model, field_model, coords_dim, dimless, require_spatial=False)
    total_head_error = (state["h"] + z_phys - float(total_head_m)) / float(dimless["H0"])
    return torch.mean(total_head_error.pow(2))


def compute_initial_head_and_theta_state_losses(
    se_model,
    field_model,
    coords_dim,
    z_phys,
    theta_targets,
    dimless,
    total_head_m,
):
    """Evaluate paired t=0 total-head and COMSOL theta losses in one forward pass."""

    state = compute_state(se_model, field_model, coords_dim, dimless, require_spatial=False)
    total_head_error = (state["h"] + z_phys - float(total_head_m)) / float(dimless["H0"])
    theta_error = (state["theta_h"] - theta_targets) / float(dimless["Theta0"])
    return torch.mean(total_head_error.pow(2)), torch.mean(theta_error.pow(2))


def compute_initial_theta_state_loss(se_model, field_model, coords_dim, theta_targets, dimless):
    """Constrain the initial water-content state without evaluating total head."""

    state = compute_state(se_model, field_model, coords_dim, dimless, require_spatial=False)
    theta_error = (state["theta_h"] - theta_targets) / float(dimless["Theta0"])
    return torch.mean(theta_error.pow(2))


def compute_bottom_free_drainage_loss(se_model, field_model, bottom_points_dim, dimless):
    state = compute_state(se_model, field_model, bottom_points_dim, dimless, require_spatial=True)
    return torch.mean((state["qz_star"] + state["kappa"]) ** 2)


def compute_top_flux_loss(se_model, field_model, top_points_dim, dimless, q_top_star):
    state = compute_state(se_model, field_model, top_points_dim, dimless, require_spatial=True)
    if not torch.is_tensor(q_top_star):
        q_top_star = torch.as_tensor(q_top_star, dtype=state["qz_star"].dtype, device=state["qz_star"].device)
    q_top_star = q_top_star.to(dtype=state["qz_star"].dtype, device=state["qz_star"].device)
    return torch.mean((state["qz_star"] + q_top_star) ** 2)


def compute_left_no_flow_loss(se_model, field_model, left_points_dim, dimless):
    state = compute_state(se_model, field_model, left_points_dim, dimless, require_spatial=True)
    return torch.mean(state["qx_star"].pow(2))


def compute_right_no_flow_loss(se_model, field_model, right_points_dim, dimless):
    state = compute_state(se_model, field_model, right_points_dim, dimless, require_spatial=True)
    return torch.mean(state["qx_star"].pow(2))


def compute_probe_theta_loss(se_model, field_model, probe_coords_dim, probe_theta_targets_norm, probe_theta_stats, dimless):
    return compute_probe_variable_loss(
        se_model,
        field_model,
        probe_coords_dim,
        probe_theta_targets_norm,
        probe_theta_stats,
        dimless,
        "theta_h",
    )


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
    value_pred_norm = (state[observed_variable] - float(probe_stats["min"])) / float(probe_stats["range"])
    return torch.mean((value_pred_norm - probe_targets_norm) ** 2)


def compute_parameter_point_losses(field_model, point_coords_dim, point_targets, point_stats, active_fields=None, field_weights=None):
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
    return total, components


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


def sample_bottom_boundary_points(dimless, n_bottom, device):
    tau_end = float(dimless["tau_end"])
    return torch.cat(
        [
            -1.0 + 2.0 * torch.rand((n_bottom, 1), device=device),
            torch.zeros((n_bottom, 1), device=device),
            tau_end * torch.rand((n_bottom, 1), device=device),
        ],
        dim=1,
    )


def sample_top_boundary_points(dimless, n_top, device):
    tau_end = float(dimless["tau_end"])
    tau_min = min(0.01, tau_end)
    return torch.cat(
        [
            -1.0 + 2.0 * torch.rand((n_top, 1), device=device),
            torch.ones((n_top, 1), device=device),
            tau_min + (tau_end - tau_min) * torch.rand((n_top, 1), device=device),
        ],
        dim=1,
    )


def _physical_to_dim_torch(x_phys, z_phys, tau_dim, dimless):
    x_dim = (x_phys - float(dimless["x_mid"])) / float(dimless["sx"])
    z_dim = (z_phys - float(dimless["z_min"])) / float(dimless["Lz"])
    return torch.cat([x_dim, z_dim, tau_dim], dim=1)


def _rand_uniform(n_points, low, high, device):
    return float(low) + (float(high) - float(low)) * torch.rand((int(n_points), 1), device=device)


def sample_physical_boundary_points(dimless, boundary_name, n_points, device, boundary_metadata=None, tau_max_day=None):
    """Sample true COMSOL physical boundaries and map them with ROI dimless scaling.

    The returned coordinates are not clamped to the inverse ROI. This is intentional:
    true left/right/bottom boundaries of the expanded COMSOL model can land outside
    the PDE sampling box after ROI-based nondimensionalization.
    """

    boundary_name = str(boundary_name).lower()
    if boundary_metadata is None:
        boundary_metadata = {}
    boundaries = boundary_metadata.get("boundaries", {}) if isinstance(boundary_metadata, dict) else {}
    spec = boundaries.get(boundary_name, {})

    def _range(axis, default):
        values = spec.get(axis)
        if isinstance(values, (list, tuple)) and len(values) == 2:
            return float(values[0]), float(values[1])
        return default

    tau_end = float(dimless["tau_end"])
    if tau_max_day is None:
        tau_max = tau_end
    else:
        tau_max = min(tau_end, max(0.0, float(tau_max_day) / float(dimless["T"])))
    tau = tau_max * torch.rand((int(n_points), 1), device=device)

    if boundary_name == "top":
        x0, x1 = _range("x_m", (float(dimless["x_min"]), float(dimless["x_max"])))
        z0, z1 = _range("z_m", (float(dimless["z_max"]), float(dimless["z_max"])))
        x_phys = _rand_uniform(n_points, x0, x1, device)
        z_phys = torch.full_like(x_phys, 0.5 * (z0 + z1))
    elif boundary_name == "bottom":
        x0, x1 = _range("x_m", (float(dimless["x_min"]), float(dimless["x_max"])))
        z0, z1 = _range("z_m", (float(dimless["z_min"]), float(dimless["z_min"])))
        x_phys = _rand_uniform(n_points, x0, x1, device)
        z_phys = torch.full_like(x_phys, 0.5 * (z0 + z1))
    elif boundary_name == "left":
        x0, x1 = _range("x_m", (float(dimless["x_min"]), float(dimless["x_min"])))
        z0, z1 = _range("z_m", (float(dimless["z_min"]), float(dimless["z_max"])))
        x_phys = torch.full((int(n_points), 1), 0.5 * (x0 + x1), device=device)
        z_phys = _rand_uniform(n_points, z0, z1, device)
    elif boundary_name == "right":
        x0, x1 = _range("x_m", (float(dimless["x_max"]), float(dimless["x_max"])))
        z0, z1 = _range("z_m", (float(dimless["z_min"]), float(dimless["z_max"])))
        x_phys = torch.full((int(n_points), 1), 0.5 * (x0 + x1), device=device)
        z_phys = _rand_uniform(n_points, z0, z1, device)
    else:
        raise ValueError(f"Unsupported boundary_name {boundary_name!r}")
    return _physical_to_dim_torch(x_phys, z_phys, tau, dimless)


def sample_physical_initial_condition_points(dimless, n_points, device, comsol_domain=None):
    """Sample t=0 points over the physical COMSOL domain without ROI clamping."""

    domain = dict(comsol_domain or {})
    x_min = float(domain.get("x_min", dimless["x_min"]))
    x_max = float(domain.get("x_max", dimless["x_max"]))
    z_min = float(domain.get("z_min", dimless["z_min"]))
    z_max = float(domain.get("z_max", dimless["z_max"]))
    x_phys = _rand_uniform(n_points, x_min, x_max, device)
    z_phys = _rand_uniform(n_points, z_min, z_max, device)
    tau = torch.zeros((int(n_points), 1), device=device)
    return _physical_to_dim_torch(x_phys, z_phys, tau, dimless), z_phys


def physical_to_dimless_xy(x_m, z_m, dimless):
    x_dim = (float(x_m) - float(dimless["x_mid"])) / float(dimless["sx"])
    z_dim = (float(z_m) - float(dimless["z_min"])) / float(dimless["Lz"])
    return x_dim, z_dim


def boundary_dimless_ranges(dimless, comsol_domain=None):
    comsol_domain = dict(comsol_domain or {})
    x_left = float(comsol_domain.get("x_min", dimless["x_min"]))
    x_right = float(comsol_domain.get("x_max", dimless["x_max"]))
    z_bottom = float(comsol_domain.get("z_min", dimless["z_min"]))
    z_top = float(comsol_domain.get("z_max", dimless["z_max"]))
    specs = {
        "top": ((float(dimless["x_min"]), float(dimless["x_max"])), (z_top, z_top)),
        "bottom": ((float(dimless["x_min"]), float(dimless["x_max"])), (z_bottom, z_bottom)),
        "left": ((x_left, x_left), (z_bottom, z_top)),
        "right": ((x_right, x_right), (z_bottom, z_top)),
    }
    ranges = {}
    for name, (x_range, z_range) in specs.items():
        x_dim = [physical_to_dimless_xy(x, z_range[0], dimless)[0] for x in x_range]
        z_dim = [physical_to_dimless_xy(x_range[0], z, dimless)[1] for z in z_range]
        ranges[name] = {
            "x_m": [float(x_range[0]), float(x_range[1])],
            "z_m": [float(z_range[0]), float(z_range[1])],
            "x_dim_min": float(min(x_dim)),
            "x_dim_max": float(max(x_dim)),
            "z_dim_min": float(min(z_dim)),
            "z_dim_max": float(max(z_dim)),
        }
    return ranges



def _tau_samples(dimless, n, device, tau_max_day=None, tau_min=0.0):
    tau_end = float(dimless["tau_end"])
    if tau_max_day is None:
        tau_max = tau_end
    else:
        tau_max = min(float(tau_end), float(tau_max_day) / float(dimless["T"]))
        tau_max = max(tau_max, 0.0)
    tau_min = min(float(tau_min), tau_max)
    return tau_min + (tau_max - tau_min) * torch.rand((n, 1), device=device)


def sample_physical_bottom_boundary_points(dimless, n_bottom, device, comsol_domain=None, tau_max_day=None):
    ranges = boundary_dimless_ranges(dimless, comsol_domain)
    bottom = ranges["bottom"]
    x = float(bottom["x_dim_min"]) + (float(bottom["x_dim_max"]) - float(bottom["x_dim_min"])) * torch.rand((n_bottom, 1), device=device)
    z = torch.full((n_bottom, 1), float(bottom["z_dim_min"]), device=device)
    return torch.cat([x, z, _tau_samples(dimless, n_bottom, device, tau_max_day=tau_max_day)], dim=1)


def sample_physical_top_boundary_points(dimless, n_top, device, comsol_domain=None, tau_max_day=None):
    ranges = boundary_dimless_ranges(dimless, comsol_domain)
    top = ranges["top"]
    x = float(top["x_dim_min"]) + (float(top["x_dim_max"]) - float(top["x_dim_min"])) * torch.rand((n_top, 1), device=device)
    z = torch.full((n_top, 1), float(top["z_dim_min"]), device=device)
    tau_min = min(0.01, float(dimless["tau_end"]))
    return torch.cat([x, z, _tau_samples(dimless, n_top, device, tau_max_day=tau_max_day, tau_min=tau_min)], dim=1)


def sample_left_boundary_points(dimless, n_left, device, comsol_domain=None, tau_max_day=None):
    ranges = boundary_dimless_ranges(dimless, comsol_domain)
    left = ranges["left"]
    x = torch.full((n_left, 1), float(left["x_dim_min"]), device=device)
    z = float(left["z_dim_min"]) + (float(left["z_dim_max"]) - float(left["z_dim_min"])) * torch.rand((n_left, 1), device=device)
    return torch.cat([x, z, _tau_samples(dimless, n_left, device, tau_max_day=tau_max_day)], dim=1)


def sample_right_boundary_points(dimless, n_right, device, comsol_domain=None, tau_max_day=None):
    ranges = boundary_dimless_ranges(dimless, comsol_domain)
    right = ranges["right"]
    x = torch.full((n_right, 1), float(right["x_dim_min"]), device=device)
    z = float(right["z_dim_min"]) + (float(right["z_dim_max"]) - float(right["z_dim_min"])) * torch.rand((n_right, 1), device=device)
    return torch.cat([x, z, _tau_samples(dimless, n_right, device, tau_max_day=tau_max_day)], dim=1)
