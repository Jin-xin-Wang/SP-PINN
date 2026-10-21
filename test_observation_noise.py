"""Verify observation noise using bundled data, without launching training."""

import copy
import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import physical

ROOT = Path(__file__).resolve().parent
ENTRIES = ["main"] + (["main_sfli"] if (ROOT / "main_sfli.py").exists() else [])


class ObservationNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.modules = [importlib.import_module(name) for name in ENTRIES]
        cls.manifest = json.loads((ROOT / "reference_data/input_manifest.json").read_text(encoding="utf-8"))
        cls.dimless = cls.modules[0].build_dimless(cls.manifest, cls.manifest["field_global_stats"])
        cls.probe = cls.modules[0].load_probe_dataset(
            ROOT / "reference_data/segments/rain1_0_20d/probes_regular_5x3"
        )

    def build(self, module, **kwargs):
        return module.build_probe_training_tensors(self.probe, self.dimless, "cpu", **kwargs)

    def test_defaults_and_validation(self):
        for module in self.modules:
            with self.subTest(entry=module.__name__):
                args = module.build_parser().parse_args([])
                self.assertEqual(args.theta_noise_std, 0.02)
                self.assertEqual(args.theta_noise_seed, 20260526)
                self.assertEqual(module.validate_theta_noise_std(0), 0.0)
                self.assertEqual(module.validate_theta_noise_std(0.02), 0.02)
                for invalid in (-0.02, 0.01, float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        module.validate_theta_noise_std(invalid)

    def test_real_targets_match_fixed_absolute_gaussian_noise(self):
        clean = self.probe["theta_h"].astype(np.float32).reshape(-1, 1)
        minimum, span = float(clean.min()), float(clean.max()) - float(clean.min())
        noise = np.random.default_rng(20260526).normal(0, 0.02, clean.shape).astype(np.float32)
        expected = torch.from_numpy((np.clip(clean + noise, 0, 1) - minimum) / span)
        for module in self.modules:
            with self.subTest(entry=module.__name__):
                payload = self.build(module)
                torch.testing.assert_close(payload["theta_targets_norm_t"], expected, rtol=0, atol=0)
                self.assertIs(payload["value_targets_norm_t"], payload["theta_targets_norm_t"])
                stats = payload["theta_stats"]
                self.assertEqual(stats["theta_noise_seed"], 20260526)
                self.assertEqual(stats["min"], minimum)
                self.assertEqual(stats["range"], span)
                self.assertAlmostEqual(stats["noise_std_empirical"], 0.02, delta=0.0005)

    def test_reproducibility_and_independence_from_training_seed(self):
        for module in self.modules:
            first = self.build(module)["theta_targets_norm_t"]
            for seed in (42, 43):
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.testing.assert_close(self.build(module)["theta_targets_norm_t"], first, rtol=0, atol=0)
            other = self.build(module, theta_noise_seed=20260527)["theta_targets_norm_t"]
            self.assertFalse(torch.equal(first, other))

    def test_private_generator_preserves_training_rng_streams(self):
        for module in self.modules:
            np.random.seed(42)
            torch.manual_seed(42)
            numpy_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
            self.build(module)
            numpy_after = np.random.get_state()
            self.assertEqual(numpy_before[0], numpy_after[0])
            np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
            self.assertEqual(numpy_before[2:], numpy_after[2:])
            self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_zero_noise_override_and_clean_pressure_targets(self):
        for module in self.modules:
            args = module.build_parser().parse_args(["--theta-noise-std", "0"])
            for variable, kwargs in (("theta_h", {"theta_noise_std": args.theta_noise_std}), ("h", {})):
                with self.subTest(entry=module.__name__, variable=variable):
                    payload = self.build(module, observed_variable=variable, **kwargs)
                    clean = self.probe[variable].astype(np.float32).reshape(-1, 1)
                    stats = payload["value_stats"]
                    expected = torch.from_numpy((clean - stats["min"]) / stats["range"])
                    torch.testing.assert_close(payload["value_targets_norm_t"], expected, rtol=0, atol=0)

    def test_clipping_and_clean_evaluation_reference(self):
        for module in self.modules:
            probe = copy.deepcopy(self.probe)
            probe["theta_h"].flat[::2], probe["theta_h"].flat[1::2] = 0.0, 1.0
            before = {k: v.copy() for k, v in probe.items() if isinstance(v, np.ndarray)}
            payload = module.build_probe_training_tensors(probe, self.dimless, "cpu")
            clean = before["theta_h"].reshape(-1, 1)
            noise = np.random.default_rng(20260526).normal(0, 0.02, clean.shape).astype(np.float32)
            torch.testing.assert_close(payload["theta_targets_norm_t"], torch.from_numpy(np.clip(clean + noise, 0, 1)), rtol=0, atol=0)
            self.assertEqual(payload["theta_stats"]["theta_noise_clip_range"], [0.0, 1.0])
            for key, value in before.items():
                np.testing.assert_array_equal(probe[key], value)

    def test_loader_wires_noise_and_preserves_clean_references(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["segments"] = [copy.deepcopy(manifest["segments"][0]) for _ in range(2)]
        manifest["segments"][1]["label"] += "_second"
        for module in self.modules:
            with patch.object(module, "load_probe_dataset", side_effect=lambda *a: copy.deepcopy(self.probe)), \
                 patch.object(module, "load_plane_reference_data", return_value={}):
                segments = module.load_segment_references(manifest, ROOT, self.dimless, "cpu")
            for index, segment in enumerate(segments):
                payload = segment["theta_probe_training"]
                self.assertIs(segment["probe_training"], payload)
                self.assertEqual(payload["theta_stats"]["theta_noise_seed"], 20260526 + index)
                expected = self.build(module, theta_noise_seed=20260526 + index)
                torch.testing.assert_close(payload["theta_targets_norm_t"], expected["theta_targets_norm_t"], rtol=0, atol=0)
                for key in ("theta_h", "h", "coords"):
                    np.testing.assert_array_equal(segment["reference_probe"][key], self.probe[key])
                self.assertIs(segment["theta_reference_probe"], segment["reference_probe"])
                expected_h = self.build(module, observed_variable="h")
                torch.testing.assert_close(segment["h_probe_training"]["h_targets_norm_t"], expected_h["h_targets_norm_t"], rtol=0, atol=0)

    def test_observation_loss_uses_noisy_targets(self):
        clean = torch.from_numpy(self.probe["theta_h"].reshape(-1, 1).copy())
        for module in self.modules:
            losses = []
            for std in (0.0, 0.02):
                payload = self.build(module, theta_noise_std=std)
                with patch.object(physical, "compute_state", return_value={"theta_h": clean}):
                    loss = module.compute_probe_variable_loss(
                        None, None, payload["coords_dim_t"], payload["theta_targets_norm_t"],
                        payload["theta_stats"], self.dimless, "theta_h",
                    )
                losses.append(float(loss))
            self.assertEqual(losses[0], 0.0)
            self.assertGreater(losses[1], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
