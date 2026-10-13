from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintRuntime:
    name: str
    weight: float
    sample_count: int | None
    enabled: bool
    disabled_reason: str | None
    activation_rule: str

    @classmethod
    def sampled(cls, name, weight, sample_count):
        weight = float(weight)
        sample_count = int(sample_count)
        if weight <= 0.0:
            enabled = False
            reason = "zero_weight"
        elif sample_count <= 0:
            enabled = False
            reason = "zero_samples"
        else:
            enabled = True
            reason = None
        return cls(
            name=str(name),
            weight=weight,
            sample_count=sample_count,
            enabled=enabled,
            disabled_reason=reason,
            activation_rule="weight > 0 and sample_count > 0",
        )

    @classmethod
    def data_backed(cls, name, weight, has_data, sample_count=None):
        weight = float(weight)
        if weight <= 0.0:
            enabled = False
            reason = "zero_weight"
        elif not bool(has_data):
            enabled = False
            reason = "missing_initial_group"
        else:
            enabled = True
            reason = None
        return cls(
            name=str(name),
            weight=weight,
            sample_count=None if sample_count is None else int(sample_count),
            enabled=enabled,
            disabled_reason=reason,
            activation_rule="weight > 0 and initial data group is configured",
        )

    def to_dict(self):
        return {
            "name": self.name,
            "enabled": bool(self.enabled),
            "weight": float(self.weight),
            "sample_count": self.sample_count,
            "disabled_reason": self.disabled_reason,
            "activation_rule": self.activation_rule,
        }


def constraint_plan_metadata(plan):
    return {name: runtime.to_dict() for name, runtime in plan.items()}


def run_segment_constraint(runtime, segment_references, loss_builder, *loss_args, **loss_kwargs):
    labels = [str(segment["label"]) for segment in segment_references]
    if not labels:
        raise ValueError(f"Constraint {runtime.name!r} received no segments.")
    if not runtime.enabled:
        return {label: 0.0 for label in labels}, 0.0

    segment_weight = float(runtime.weight) / float(len(segment_references))
    loss_values = {}
    for segment in segment_references:
        label = str(segment["label"])
        loss = loss_builder(segment, *loss_args, **loss_kwargs)
        loss_value = float(loss.detach().cpu())
        loss_values[label] = loss_value
        (segment_weight * loss).backward()
    return loss_values, float(sum(loss_values.values()) / len(loss_values))
