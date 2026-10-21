import math
import torch
import torch.nn as nn

UNIFIED_HIDDEN_LAYERS = 4
UNIFIED_HIDDEN_DIM = 100
PARAMETER_SHARED_HIDDEN_LAYERS = 3
PARAMETER_SHARED_HIDDEN_DIM = 64
MINMAX_PARAMETER_FIELDS = ("lnalpha", "theta_s", "theta_r", "lnks")
PARAMETER_LOSS_FIELDS = ("lnalpha", "n", "theta_s", "theta_r", "lnks")
PARAMETER_NORMALIZATION_EXPANSION_FRACTION = 0.25
PARAMETER_OUTPUT_MODE = "expanded_normalized_lnalpha_raw_n_expanded_normalized_theta_s_theta_r_lnks"
N_OUTPUT_MODE = "raw_n supervised directly; physics uses n = 1 + softplus(raw_n)"
PHYSICAL_FIELD_NAMES = ("ks", "alpha", "n", "theta_s", "theta_r")


def normalize_activation_name(activation_name):
    if activation_name != "sin":
        raise ValueError("This release uses the sin state network.")
    return "sin"


def build_activation(activation_name):
    normalize_activation_name(activation_name)
    return SinActivation()


class SinActivation(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class ModelArgs:

    def __init__(self, activation_name="sin"):
        self.in_dim = 3
        self.out_dim = 1
        self.hid_layer_dim = UNIFIED_HIDDEN_DIM
        self.hid_layer_num = UNIFIED_HIDDEN_LAYERS
        self.activation_name = normalize_activation_name(activation_name)


class BaseSeModel(nn.Module):

    def __init__(self, s_min=1e-06, s_max=1.0 - 1e-06):
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


def inverse_softplus(value):
    value = max(float(value), 1e-12)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


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


class RichardsPINN(BaseSeModel):

    def __init__(self, model_type="baseline", activation="sin", s_min=1e-06, s_max=1.0 - 1e-06):
        super().__init__(s_min=s_min, s_max=s_max)
        if model_type != "baseline":
            raise ValueError("This release uses the baseline state network.")
        self.model_type = model_type
        self.activation = normalize_activation_name(activation)
        self.backbone = _RichardsBackbone(ModelArgs(activation_name=self.activation))

    def forward(self, coords_dim):
        return self.scale_output(self.backbone(self.normalize_coords(coords_dim)))


def transform_raw_parameter_outputs(raw, parameter_output_stats):
    theta_r = _denormalize_parameter(raw["theta_r"], parameter_output_stats, "theta_r")
    theta_s = _denormalize_parameter(raw["theta_s"], parameter_output_stats, "theta_s")
    lnalpha = _denormalize_parameter(raw["lnalpha"], parameter_output_stats, "lnalpha")
    alpha = torch.exp(lnalpha)
    n = 1.0 + torch.nn.functional.softplus(raw["n"])
    lnn1 = torch.log(torch.clamp(n - 1.0, min=1e-12))
    m = 1.0 - 1.0 / n
    lnks = _denormalize_parameter(raw["lnks"], parameter_output_stats, "lnks")
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


class InverseFieldModelBundle(nn.Module):

    def __init__(self, field_global_stats, parameter_model="shared64", parameter_output_stats=None):
        super().__init__()
        if parameter_model != "shared64" or not parameter_output_stats:
            raise ValueError(
                "This release requires shared64 and observation-derived normalization statistics."
            )
        self.field_global_stats = field_global_stats
        self.parameter_output_stats = dict(parameter_output_stats)
        self.parameter_model_variant = "shared64"

        def normalized_bias(field_name):
            stats = self.parameter_output_stats[field_name]
            return (
                2.0 * (float(stats["median"]) - float(stats["min"])) / float(stats["range"]) - 1.0
            )

        bias_init_by_output = {
            "lnalpha": normalized_bias("lnalpha"),
            "n": inverse_softplus(max(float(field_global_stats["n"]["median"]) - 1.0, 1e-06)),
            "theta_s": normalized_bias("theta_s"),
            "theta_r": normalized_bias("theta_r"),
            "lnks": normalized_bias("lnks"),
        }
        self.shared_net = SharedParameterInverseNet(
            PARAMETER_LOSS_FIELDS, bias_init_by_output=bias_init_by_output
        )

    def parameter_model_metadata(self):
        return {
            "parameter_model_variant": "shared64",
            "parameter_hidden_dim": PARAMETER_SHARED_HIDDEN_DIM,
            "parameter_hidden_layers": PARAMETER_SHARED_HIDDEN_LAYERS,
            "parameter_head_count": len(self.shared_net.output_names),
            "shared_parameter_network": True,
            "parameter_output_mode": PARAMETER_OUTPUT_MODE,
            "n_output_mode": N_OUTPUT_MODE,
            "parameter_normalization_source": "selected_parameter_points_expanded_25pct",
            "parameter_normalization_expansion_fraction": PARAMETER_NORMALIZATION_EXPANSION_FRACTION,
        }

    def raw_field_outputs(self, xyz_dim):
        return self.shared_net(xyz_dim)

    def trainable_parameter_fields(self):
        return list(PHYSICAL_FIELD_NAMES)

    def trainable_loss_fields(self):
        return list(PARAMETER_LOSS_FIELDS)

    def evaluate_fields(self, xyz_dim):
        return transform_raw_parameter_outputs(
            self.raw_field_outputs(xyz_dim), self.parameter_output_stats
        )
