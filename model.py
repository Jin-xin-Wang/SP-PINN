import math
from pathlib import Path

import torch
import torch.nn as nn

from field_models import evaluate_frozen_field, load_frozen_field_models


UNIFIED_HIDDEN_LAYERS = 4
UNIFIED_HIDDEN_DIM = 100
EXPLICIT_FOURIER_FEATURE_DIM = 11
PARAMETER_HIDDEN_LAYERS = 3
PARAMETER_HIDDEN_DIM = 32
PARAMETER_SHARED_HIDDEN_LAYERS = 3
PARAMETER_SHARED_HIDDEN_DIM = 64
PARAMETER_MODEL_CHOICES = ("independent32", "shared64")
MINMAX_PARAMETER_FIELDS = ("lnalpha", "theta_s", "theta_r", "lnks")
PARAMETER_LOSS_FIELDS = ("lnalpha", "n", "theta_s", "theta_r", "lnks")
PARAMETER_NORMALIZATION_EXPANSION_FRACTION = 0.25
PARAMETER_OUTPUT_MODE = "expanded_normalized_lnalpha_raw_n_expanded_normalized_theta_s_theta_r_lnks"
N_OUTPUT_MODE = "raw_n supervised directly; physics uses n = 1 + softplus(raw_n)"
THETA_EPS = 1.0e-6
PHYSICAL_FIELD_NAMES = ("ks", "alpha", "n", "theta_s", "theta_r")
LOSS_FIELD_BY_PHYSICAL_FIELD = {
    "ks": "lnks",
    "alpha": "lnalpha",
    "n": "n",
    "theta_s": "theta_s",
    "theta_r": "theta_r",
}
PHYSICAL_FIELD_BY_LOSS_FIELD = {loss: physical for physical, loss in LOSS_FIELD_BY_PHYSICAL_FIELD.items()}


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def normalize_activation_name(activation_name):
    key = activation_name.strip().lower().replace("-", "_")
    aliases = {
        "tanh": "tanh",
        "silu": "silu",
        "swish": "silu",
        "sin": "sin",
        "sine": "sin",
    }
    if key not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported activation '{activation_name}'. Valid options: {valid}")
    return aliases[key]


def build_activation(activation_name):
    key = normalize_activation_name(activation_name)
    if key == "tanh":
        return nn.Tanh()
    if key == "sin":
        return SinActivation()
    return nn.SiLU()


def normalize_model_type(model_type):
    key = model_type.strip().lower().replace("-", "_")
    aliases = {
        "baseline": "baseline",
        "richards": "baseline",
        "richards_pinn": "baseline",
        "baseline_ff": "baseline_ff",
        "baseline_rf": "baseline_rf",
        "tsff": "tsff",
        "tsff_branch": "tsff",
    }
    if key not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported model_type '{model_type}'. Valid options: {valid}")
    return aliases[key]


def normalize_parameter_model(parameter_model):
    key = str(parameter_model).strip().lower().replace("-", "_")
    aliases = {
        "independent32": "independent32",
        "independent_32": "independent32",
        "shared64": "shared64",
        "shared_64": "shared64",
    }
    if key not in aliases:
        valid = ", ".join(PARAMETER_MODEL_CHOICES)
        raise ValueError(f"Unsupported parameter_model '{parameter_model}'. Valid options: {valid}")
    return aliases[key]


class ModelArgs:
    def __init__(self, activation_name="tanh"):
        self.in_dim = 3
        self.out_dim = 1
        self.hid_layer_dim = UNIFIED_HIDDEN_DIM
        self.hid_layer_num = UNIFIED_HIDDEN_LAYERS
        self.activation_name = normalize_activation_name(activation_name)


class BaseSeModel(nn.Module):
    def __init__(self, s_min=1.0e-6, s_max=1.0 - 1.0e-6):
        super().__init__()
        self.s_min = float(s_min)
        self.s_max = float(s_max)

    def normalize_coords(self, coords):
        return coords

    def scale_output(self, raw):
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(raw)


class _RichardsBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation = build_activation(args.activation_name)
        self.input_layer = nn.Linear(args.in_dim, args.hid_layer_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(args.hid_layer_dim, args.hid_layer_dim)
                for _ in range(args.hid_layer_num - 1)
            ]
        )
        self.output_layer = nn.Linear(args.hid_layer_dim, args.out_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        hidden = self.activation(self.input_layer(inputs))
        for index, layer in enumerate(self.hidden_layers):
            residual = hidden
            hidden = self.activation(layer(hidden))
            if index % 2 == 1:
                hidden = hidden + residual
        return self.output_layer(hidden)


class _RichardsBackbone_FF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation = build_activation(args.activation_name)
        self.input_layer = nn.Linear(EXPLICIT_FOURIER_FEATURE_DIM, args.hid_layer_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(args.hid_layer_dim, args.hid_layer_dim)
                for _ in range(args.hid_layer_num - 1)
            ]
        )
        self.output_layer = nn.Linear(args.hid_layer_dim, args.out_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_features(self, inputs):
        x_hat = inputs[..., 0:1]
        z_hat = inputs[..., 1:2]
        tau = inputs[..., 2:3]
        return torch.cat(
            (
                x_hat,
                z_hat,
                tau,
                torch.sin(torch.pi * x_hat),
                torch.cos(torch.pi * x_hat),
                torch.sin(2.0 * torch.pi * x_hat),
                torch.cos(2.0 * torch.pi * x_hat),
                torch.sin(torch.pi * z_hat),
                torch.cos(torch.pi * z_hat),
                torch.sin(2.0 * torch.pi * z_hat),
                torch.cos(2.0 * torch.pi * z_hat),
            ),
            dim=-1,
        )

    def forward(self, inputs):
        hidden = self.activation(self.input_layer(self._build_features(inputs)))
        for index, layer in enumerate(self.hidden_layers):
            residual = hidden
            hidden = self.activation(layer(hidden))
            if index % 2 == 1:
                hidden = hidden + residual
        return self.output_layer(hidden)


class _RichardsBackbone_RF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation = build_activation(args.activation_name)
        f1 = torch.randn((3, args.hid_layer_dim // 4)).float() * torch.tensor([[1], [1], [1]])
        f2 = torch.randn((3, args.hid_layer_dim // 4)).float() * torch.tensor([[10], [1], [1]])
        self.F = nn.Parameter(torch.hstack((f1, f2)), requires_grad=False)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(args.hid_layer_dim, args.hid_layer_dim)
                for _ in range(args.hid_layer_num - 1)
            ]
        )
        self.output_layer = nn.Linear(args.hid_layer_dim, args.out_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        hidden = torch.cat((torch.sin(inputs @ self.F), torch.cos(inputs @ self.F)), dim=-1)
        for index, layer in enumerate(self.hidden_layers):
            residual = hidden
            hidden = self.activation(layer(hidden))
            if index % 2 == 1:
                hidden = hidden + residual
        return self.output_layer(hidden)


class _TSFFBranchBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.register_buffer("ZT_F1", torch.randn((2, args.hid_layer_dim // 2)) * 1.0)
        self.register_buffer("XZ_F1", torch.randn((2, args.hid_layer_dim // 2)) * 1.0)
        self.register_buffer("XZ_F2", torch.randn((2, args.hid_layer_dim // 2)) * 10.0)

        zt_layers = []
        for _ in range(args.hid_layer_num - 1):
            zt_layers.append(nn.Linear(args.hid_layer_dim, args.hid_layer_dim))
            zt_layers.append(build_activation(args.activation_name))
        self.densenet_zt = nn.Sequential(*zt_layers)

        xz_layers = []
        for _ in range(args.hid_layer_num):
            xz_layers.append(nn.Linear(args.hid_layer_dim, args.hid_layer_dim))
            xz_layers.append(build_activation(args.activation_name))
        self.densenet_xz = nn.Sequential(*xz_layers)

        self.out_layer = nn.Linear(2 * args.hid_layer_dim, args.out_dim)

        for net in (self.densenet_zt, self.densenet_xz):
            for module in net:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.kaiming_normal_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, inputs):
        xz = inputs[..., 0:2]
        zt = inputs[..., 1:3]

        h_zt = torch.cat((torch.sin(zt @ self.ZT_F1), torch.cos(zt @ self.ZT_F1)), dim=-1)
        h_xz_1 = torch.cat((torch.sin(xz @ self.XZ_F1), torch.cos(xz @ self.XZ_F1)), dim=-1)
        h_xz_2 = torch.cat((torch.sin(xz @ self.XZ_F2), torch.cos(xz @ self.XZ_F2)), dim=-1)

        h_zt = self.densenet_zt(h_zt)
        h_xz_1 = self.densenet_xz(h_xz_1)
        h_xz_2 = self.densenet_xz(h_xz_2)

        branch_1 = h_zt * h_xz_1
        branch_2 = h_zt * h_xz_2
        return self.out_layer(torch.cat((branch_1, branch_2), dim=-1))


class RichardsPINN(BaseSeModel):
    def __init__(self, model_type="baseline", activation="tanh", s_min=1.0e-6, s_max=1.0 - 1.0e-6):
        super().__init__(s_min=s_min, s_max=s_max)
        self.model_type = normalize_model_type(model_type)
        self.activation = normalize_activation_name(activation)

        args = ModelArgs(activation_name=self.activation)
        if self.model_type == "baseline":
            self.backbone = _RichardsBackbone(args)
        elif self.model_type == "baseline_ff":
            self.backbone = _RichardsBackbone_FF(args)
        elif self.model_type == "baseline_rf":
            self.backbone = _RichardsBackbone_RF(args)
        else:
            self.backbone = _TSFFBranchBackbone(args)

    def forward(self, coords_dim):
        return self.scale_output(self.backbone(self.normalize_coords(coords_dim)))


def inverse_sigmoid(value, eps=1.0e-6):
    value = min(max(float(value), eps), 1.0 - eps)
    return math.log(value / (1.0 - value))


def inverse_softplus(value):
    value = max(float(value), 1.0e-12)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class ParameterInverseNet(nn.Module):
    def __init__(self, bias_init=0.0, hidden_dim=PARAMETER_HIDDEN_DIM, hidden_layers=PARAMETER_HIDDEN_LAYERS):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.network[-1].bias, float(bias_init))

    def forward(self, xyz_dim):
        return self.network(xyz_dim)


class AlphaInverseNet(ParameterInverseNet):
    pass


class NInverseNet(ParameterInverseNet):
    pass


class ThetaSInverseNet(ParameterInverseNet):
    pass


class ThetaRInverseNet(ParameterInverseNet):
    pass


class LnKsInverseNet(ParameterInverseNet):
    pass


class SharedParameterInverseNet(nn.Module):
    def __init__(self, output_names, bias_init_by_output=None):
        super().__init__()
        self.output_names = tuple(output_names)
        bias_init_by_output = {} if bias_init_by_output is None else dict(bias_init_by_output)

        layers = [nn.Linear(2, PARAMETER_SHARED_HIDDEN_DIM), nn.Tanh()]
        for _ in range(PARAMETER_SHARED_HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(PARAMETER_SHARED_HIDDEN_DIM, PARAMETER_SHARED_HIDDEN_DIM))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(PARAMETER_SHARED_HIDDEN_DIM, len(self.output_names)))
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            for index, field_name in enumerate(self.output_names):
                self.network[-1].bias[index] = float(bias_init_by_output.get(field_name, 0.0))

    def forward(self, xz_dim):
        outputs = self.network(xz_dim)
        return {
            field_name: outputs[:, index : index + 1]
            for index, field_name in enumerate(self.output_names)
        }


def _denormalize_parameter(raw_value, stats, field_name):
    field_stats = stats[field_name]
    return 0.5 * (raw_value + 1.0) * float(field_stats["range"]) + float(field_stats["min"])


def transform_raw_parameter_outputs(raw, known_values=None, parameter_output_stats=None):
    known_values = {} if known_values is None else dict(known_values)
    parameter_output_stats = {} if parameter_output_stats is None else dict(parameter_output_stats)

    if parameter_output_stats:
        theta_r = (
            known_values["theta_r"]
            if "theta_r" in known_values
            else _denormalize_parameter(raw["theta_r"], parameter_output_stats, "theta_r")
        )
        theta_s = (
            known_values["theta_s"]
            if "theta_s" in known_values
            else _denormalize_parameter(raw["theta_s"], parameter_output_stats, "theta_s")
        )
        if "alpha" in known_values:
            alpha = known_values["alpha"]
            lnalpha = torch.log(torch.clamp(alpha, min=1.0e-12))
        else:
            lnalpha = _denormalize_parameter(raw["lnalpha"], parameter_output_stats, "lnalpha")
            alpha = torch.exp(lnalpha)
        if "n" in known_values:
            n = known_values["n"]
            lnn1 = torch.log(torch.clamp(n - 1.0, min=1.0e-12))
        else:
            n = 1.0 + torch.nn.functional.softplus(raw["n"])
            lnn1 = torch.log(torch.clamp(n - 1.0, min=1.0e-12))
    else:
        theta_r = known_values["theta_r"] if "theta_r" in known_values else torch.sigmoid(raw["theta_r"])
        theta_s = known_values["theta_s"] if "theta_s" in known_values else torch.sigmoid(raw["theta_s"])
        if "alpha" in known_values:
            alpha = known_values["alpha"]
            lnalpha = torch.log(torch.clamp(alpha, min=1.0e-12))
        else:
            lnalpha = raw.get("lnalpha", raw.get("alpha"))
            alpha = torch.exp(lnalpha)
        if "n" in known_values:
            n = known_values["n"]
            lnn1 = torch.log(torch.clamp(n - 1.0, min=1.0e-12))
        else:
            n = 1.0 + torch.nn.functional.softplus(raw["n"])
            lnn1 = torch.log(torch.clamp(n - 1.0, min=1.0e-12))
    m = 1.0 - 1.0 / n

    if "ks" in known_values:
        ks = known_values["ks"]
        lnks = torch.log(ks)
    elif "lnks" in known_values:
        lnks = known_values["lnks"]
        ks = torch.exp(lnks)
    elif parameter_output_stats:
        lnks = _denormalize_parameter(raw["lnks"], parameter_output_stats, "lnks")
        ks = torch.exp(lnks)
    else:
        lnks = raw["lnks"]
        ks = torch.exp(lnks)

    return {
        "theta_s": theta_s,
        "theta_r": theta_r,
        "alpha": alpha,
        "lnalpha": lnalpha,
        "n": n,
        "lnn1": lnn1,
        "m": m,
        "lnks": lnks,
        "ks": ks,
        "ks_eff": ks,
        "delta_theta": theta_s - theta_r,
    }


def _normalize_known_field_flags(known_field_flags):
    known_field_flags = known_field_flags or {}
    return {
        field_name: bool(int(known_field_flags.get(field_name, 0)))
        for field_name in PHYSICAL_FIELD_NAMES
    }


class InverseFieldModelBundle(nn.Module):
    def __init__(
        self,
        field_global_stats,
        known_field_flags=None,
        known_field_model_dir=None,
        parameter_model="independent32",
        parameter_output_stats=None,
    ):
        super().__init__()
        self.field_global_stats = field_global_stats
        self.parameter_output_stats = {} if parameter_output_stats is None else dict(parameter_output_stats)
        self.known_field_flags = _normalize_known_field_flags(known_field_flags)
        self.known_field_model_dir = None if known_field_model_dir is None else str(Path(known_field_model_dir).resolve())
        self.parameter_model_variant = normalize_parameter_model(parameter_model)
        self.frozen_stats = None
        self.frozen_generation = {}
        self.frozen_models = nn.ModuleDict()
        self.nets = nn.ModuleDict()
        self.shared_net = None

        frozen_field_names = [field_name for field_name, is_known in self.known_field_flags.items() if is_known]
        if frozen_field_names:
            if not self.known_field_model_dir:
                raise ValueError("known_field_model_dir is required when any parameter field is marked known.")
            frozen_stats, frozen_models, frozen_generation = load_frozen_field_models(
                self.known_field_model_dir,
                frozen_field_names,
                torch.device("cpu"),
            )
            self.frozen_stats = frozen_stats
            self.frozen_generation = frozen_generation
            self.frozen_models = nn.ModuleDict(frozen_models)

        theta_r_median = float(field_global_stats["theta_r"]["median"])
        theta_s_median = float(field_global_stats["theta_s"]["median"])
        def _normalized_bias(field_name, fallback):
            if field_name not in self.parameter_output_stats:
                return float(fallback)
            stats = self.parameter_output_stats[field_name]
            return 2.0 * (float(stats["median"]) - float(stats["min"])) / float(stats["range"]) - 1.0

        bias_init_by_output = {
            "lnalpha": _normalized_bias("lnalpha", math.log(max(float(field_global_stats["alpha"]["median"]), 1.0e-12))),
            "n": inverse_softplus(max(float(field_global_stats["n"]["median"]) - 1.0, 1.0e-6)),
            "theta_s": _normalized_bias("theta_s", theta_s_median),
            "theta_r": _normalized_bias("theta_r", theta_r_median),
            "lnks": _normalized_bias("lnks", float(field_global_stats["lnks"]["median"])),
        }
        trainable_loss_fields = [
            field_name
            for field_name in PARAMETER_LOSS_FIELDS
            if not self.known_field_flags[PHYSICAL_FIELD_BY_LOSS_FIELD[field_name]]
        ]

        if self.parameter_model_variant == "independent32":
            if not self.known_field_flags["alpha"]:
                self.nets["lnalpha"] = AlphaInverseNet(
                    bias_init=bias_init_by_output["lnalpha"],
                )
            if not self.known_field_flags["n"]:
                self.nets["n"] = NInverseNet(
                    bias_init=bias_init_by_output["n"],
                )
            if not self.known_field_flags["theta_s"]:
                self.nets["theta_s"] = ThetaSInverseNet(
                    bias_init=bias_init_by_output["theta_s"],
                )
            if not self.known_field_flags["theta_r"]:
                self.nets["theta_r"] = ThetaRInverseNet(
                    bias_init=bias_init_by_output["theta_r"],
                )
            if not self.known_field_flags["ks"]:
                self.nets["lnks"] = LnKsInverseNet(
                    bias_init=bias_init_by_output["lnks"],
                )
        elif trainable_loss_fields:
            self.shared_net = SharedParameterInverseNet(
                trainable_loss_fields,
                bias_init_by_output=bias_init_by_output,
            )
        else:
            self.shared_net = None

    def parameter_model_metadata(self):
        if self.parameter_model_variant == "shared64":
            parameter_hidden_dim = int(PARAMETER_SHARED_HIDDEN_DIM)
            parameter_hidden_layers = int(PARAMETER_SHARED_HIDDEN_LAYERS)
            parameter_head_count = 0 if self.shared_net is None else int(len(self.shared_net.output_names))
            shared_parameter_network = True
        else:
            parameter_hidden_dim = int(PARAMETER_HIDDEN_DIM)
            parameter_hidden_layers = int(PARAMETER_HIDDEN_LAYERS)
            parameter_head_count = int(len(self.nets))
            shared_parameter_network = False
        return {
            "parameter_model_variant": self.parameter_model_variant,
            "parameter_hidden_dim": parameter_hidden_dim,
            "parameter_hidden_layers": parameter_hidden_layers,
            "parameter_head_count": parameter_head_count,
            "shared_parameter_network": shared_parameter_network,
            "parameter_output_mode": PARAMETER_OUTPUT_MODE,
            "n_output_mode": N_OUTPUT_MODE,
            "parameter_normalization_source": "selected_parameter_points_expanded_25pct",
            "parameter_normalization_expansion_fraction": PARAMETER_NORMALIZATION_EXPANSION_FRACTION,
        }

    def raw_field_outputs(self, xyz_dim):
        if self.shared_net is not None:
            return self.shared_net(xyz_dim)
        return {field_name: net(xyz_dim) for field_name, net in self.nets.items()}

    def _frozen_field(self, field_name, xyz_dim):
        return evaluate_frozen_field(
            field_name,
            xyz_dim,
            self.frozen_models,
            self.frozen_stats,
            self.frozen_generation,
        )

    def trainable_parameter_fields(self):
        return [
            field_name
            for field_name in PHYSICAL_FIELD_NAMES
            if not self.known_field_flags[field_name]
        ]

    def frozen_parameter_fields(self):
        return [
            field_name
            for field_name in PHYSICAL_FIELD_NAMES
            if self.known_field_flags[field_name]
        ]

    def trainable_loss_fields(self):
        return [
            field_name
            for field_name in PARAMETER_LOSS_FIELDS
            if PHYSICAL_FIELD_BY_LOSS_FIELD[field_name] in self.trainable_parameter_fields()
        ]

    def evaluate_fields(self, xyz_dim):
        raw = self.raw_field_outputs(xyz_dim)
        known_values = {}
        if self.known_field_flags["theta_r"]:
            known_values["theta_r"] = torch.clamp(
                self._frozen_field("theta_r", xyz_dim),
                min=THETA_EPS,
                max=1.0 - 2.0 * THETA_EPS,
            )
        if self.known_field_flags["theta_s"]:
            known_values["theta_s"] = torch.clamp(
                self._frozen_field("theta_s", xyz_dim),
                min=THETA_EPS,
                max=1.0 - THETA_EPS,
            )
        if self.known_field_flags["alpha"]:
            known_values["alpha"] = torch.clamp(self._frozen_field("alpha", xyz_dim), min=1.0e-8)
        if self.known_field_flags["n"]:
            known_values["n"] = torch.clamp(self._frozen_field("n", xyz_dim), min=1.01 + THETA_EPS, max=10.0)
        if self.known_field_flags["ks"]:
            known_values["ks"] = torch.clamp(self._frozen_field("ks", xyz_dim), min=1.0e-12, max=1.0e6)
        return transform_raw_parameter_outputs(
            raw,
            known_values=known_values,
            parameter_output_stats=self.parameter_output_stats,
        )


AVAILABLE_MODEL_TYPES = ("baseline", "baseline_ff", "baseline_rf", "tsff")
