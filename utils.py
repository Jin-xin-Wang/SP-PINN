import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


SECONDS_PER_DAY = 86400.0


def choose_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed, deterministic=False):
    """Seed all runtime RNGs and optionally request deterministic torch kernels."""
    if deterministic:
        # Set this before torch.manual_seed can initialize a CUDA context.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        elif hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def elapsed_minutes(start_time):
    return (time.perf_counter() - start_time) / 60.0


def scientific(value):
    return f"{float(value):.2e}"


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def format_log_row(row):
    pieces = [
        f"epoch={int(row['epoch']):05d}",
        f"elapsed_min={scientific(row['elapsed_min'])}",
        f"lr={scientific(row['lr'])}",
        f"total={scientific(row['total'])}",
        f"pde={scientific(row['pde'])}",
    ]
    if "top" in row:
        pieces.append(f"top={scientific(row['top'])}")
    if "bottom" in row:
        pieces.append(f"bottom={scientific(row['bottom'])}")
    if "initial_theta" in row:
        pieces.append(f"initial_theta={scientific(row['initial_theta'])}")
    if "initial_head" in row:
        pieces.append(f"initial_head={scientific(row['initial_head'])}")
    if "initial_theta_state" in row:
        pieces.append(f"initial_theta_state={scientific(row['initial_theta_state'])}")
    pieces.extend(
        [
            f"probe_theta={scientific(row['probe_theta'])}",
            f"probe_h={scientific(row['probe_h'])}" if "probe_h" in row else None,
            f"param={scientific(row['param'])}",
        ]
    )
    pieces = [piece for piece in pieces if piece is not None]
    for key in ("param_alpha", "param_n", "param_theta_s", "param_theta_r", "param_lnks"):
        if key in row:
            pieces.append(f"{key}={scientific(row[key])}")
    for prefix in ("pde_", "top_", "bottom_", "probe_theta_", "probe_h_"):
        for key in sorted(k for k in row if k.startswith(prefix)):
            pieces.append(f"{key}={scientific(row[key])}")
    return " ".join(pieces)


def write_json(path, payload):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_dict_csv(path, fieldnames, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path, header, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
