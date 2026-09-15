import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from proxy_calibrator_axicon import (
    AxiconProxyParameters,
    ProxyCalibrationDataset,
    split_axicon_transfer_by_z,
    transfer_for_z_offset,
)


def make_proxy(kernel_size: int, subpixel_factor: int = 1) -> AxiconProxyParameters:
    return AxiconProxyParameters(
        num_zernike=1,
        source_grid_shape=(2, 2),
        crosstalk_kernel_size=kernel_size,
        crosstalk_subpixel_factor=subpixel_factor,
        transfer_radial_bins=2,
        transfer_azimuthal_order=0,
        transfer_angular_samples=1,
    )


class SlmCrosstalkTests(unittest.TestCase):
    def test_one_by_one_kernel_preserves_previous_phase_path(self):
        proxy = make_proxy(1)
        drive = torch.tensor([[0.0, 0.25], [0.5, 1.0]])

        filtered = proxy.filter_slm_drive(drive)
        phase = proxy.slm_phase_from_drive(drive)

        torch.testing.assert_close(filtered, drive)
        torch.testing.assert_close(phase, drive * (2.0 * math.pi))
        self.assertAlmostEqual(float(phase[-1, -1]), 2.0 * math.pi, places=5)

    def test_uniform_drive_and_dc_response_are_preserved(self):
        proxy = make_proxy(3)
        drive = torch.full((7, 9), 0.73)

        kernel = proxy.slm_crosstalk_kernel()
        filtered = proxy.filter_slm_drive(drive)

        torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))
        torch.testing.assert_close(filtered, drive)

    def test_subpixel_factor_expands_drive_and_kernel_not_parameters_per_pixel(self):
        proxy = make_proxy(3, subpixel_factor=2)
        drive = torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 0.125]])

        filtered = proxy.filter_slm_drive(drive)
        expected = drive.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

        self.assertEqual(proxy.slm_crosstalk_kernel().shape, (6, 6))
        self.assertEqual(proxy.crosstalk_kernel_residual.numel(), 36)
        torch.testing.assert_close(filtered, expected)

    def test_signed_asymmetric_lobes_are_representable_with_unit_sum(self):
        proxy = make_proxy(3, subpixel_factor=2)
        anchor = (proxy.crosstalk_effective_kernel_size - 1) // 2
        with torch.no_grad():
            proxy.crosstalk_kernel_residual[anchor, anchor - 1] = -0.2
            proxy.crosstalk_kernel_residual[anchor, anchor + 1] = 0.2

        kernel = proxy.slm_crosstalk_kernel()

        self.assertLess(float(kernel.min()), 0.0)
        torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))

    def test_filter_operates_across_zero_two_pi_boundary_before_phase(self):
        proxy = make_proxy(3)
        with torch.no_grad():
            uniform_kernel = torch.full((3, 3), 1.0 / 9.0)
            proxy.crosstalk_kernel_residual.copy_(
                uniform_kernel - proxy.slm_crosstalk_identity()
            )
        drive = torch.zeros(3, 3)
        drive[1, 1] = 1.0

        phase = proxy.slm_phase_from_drive(drive)

        expected = torch.full((3, 3), 2.0 * math.pi / 9.0)
        torch.testing.assert_close(phase, expected)
        self.assertGreater(float(phase[1, 1]), 0.0)

    def test_visual_gradient_reaches_every_kernel_tap(self):
        proxy = make_proxy(3)
        torch.manual_seed(7)
        drive = torch.rand(8, 9)
        spatial_weights = torch.rand(8, 9)

        loss = (proxy.filter_slm_drive(drive) * spatial_weights).sum()
        loss.backward()

        gradient = proxy.crosstalk_kernel_residual.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool((gradient.abs() > 1e-8).all()))

    def test_dataset_keeps_full_scale_command_distinct_from_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.Phase_Masks").mkdir()
            (root / "3.Aligned_Camera").mkdir()
            np.save(
                root / "0.Phase_Masks" / "sample_0.npy",
                np.array([[0.0, 1023.0]], dtype=np.float32),
            )
            np.save(
                root / "3.Aligned_Camera" / "sample_0.npy",
                np.ones((1, 2), dtype=np.float32),
            )
            dataset = ProxyCalibrationDataset(
                root,
                fov_crop_size=None,
                phase_transpose=False,
                phase_flip_first_axis=False,
                expected_phase_shape=(1, 2),
                camera_scale=1.0,
            )

            drive = dataset[0]["slm_drive"]

            self.assertEqual(float(drive[0, 0]), 0.0)
            self.assertEqual(float(drive[0, 1]), 1.0)

    def test_even_kernel_size_preserves_shape_and_dc_response(self):
        proxy = make_proxy(4)
        drive = torch.full((6, 7), 0.41)

        filtered = proxy.filter_slm_drive(drive)

        self.assertEqual(filtered.shape, drive.shape)
        torch.testing.assert_close(filtered, drive)


class MultiZDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_pair(root: Path, pattern_id: str, z_folder: str) -> None:
        np.save(
            root / "0.Phase_Masks" / f"{pattern_id}.npy",
            np.array([[0.0, 1023.0]], dtype=np.float32),
        )
        camera_dir = root / "3.Aligned_Camera" / z_folder
        camera_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            camera_dir / f"{pattern_id}.npy",
            np.ones((1, 2), dtype=np.float32),
        )

    @staticmethod
    def _dataset(root: Path, **kwargs) -> ProxyCalibrationDataset:
        return ProxyCalibrationDataset(
            root,
            fov_crop_size=None,
            phase_transpose=False,
            phase_flip_first_axis=False,
            expected_phase_shape=(1, 2),
            camera_scale=1.0,
            **kwargs,
        )

    def test_shared_phase_is_paired_with_every_z_and_complete_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.Phase_Masks").mkdir()
            for z_folder in ("z_-0.5", "z_0", "z_+0.5"):
                self._write_pair(root, "real_0", z_folder)
            self._write_pair(root, "real_1", "z_0")

            dataset = self._dataset(root, require_all_z=True)

            self.assertEqual(dataset.z_positions_mm, [-0.5, 0.0, 0.5])
            self.assertEqual(len(dataset), 3)
            self.assertEqual({s["pattern_id"] for s in dataset.samples}, {"real_0"})
            self.assertEqual(
                {float(dataset[index]["z_mm"]) for index in range(len(dataset))},
                {-0.5, 0.0, 0.5},
            )
            self.assertTrue(all("__z_" in s["id"] for s in dataset.samples))

    def test_z_and_pattern_selection_retain_all_views_of_chosen_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.Phase_Masks").mkdir()
            for pattern_id in ("real_0", "real_1", "other_0"):
                for z_folder in ("z_-1.0", "z_0", "z_+1.0"):
                    self._write_pair(root, pattern_id, z_folder)

            dataset = self._dataset(
                root,
                z_offsets_mm=[-1.0, 1.0],
                pattern_globs=["real_*"],
                max_patterns=1,
                seed=7,
            )

            self.assertEqual(dataset.z_positions_mm, [-1.0, 1.0])
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len({s["pattern_id"] for s in dataset.samples}), 1)

    def test_sparse_transfer_is_sliced_to_one_plane_per_sample(self):
        transfer = torch.arange(12).reshape(4, 3)
        h_asm = {
            "type": "sparse",
            "H_asm_sparse": transfer,
            "ring_mask": torch.ones(2, 2, dtype=torch.bool),
            "z_query": torch.tensor([0.005, 0.006, 0.007]),
        }
        planes = split_axicon_transfer_by_z(h_asm, [-1.0, 0.0, 1.0])
        physics = {"h_asm_by_z_mm": planes}

        selected = transfer_for_z_offset(physics, torch.tensor(1.0))

        self.assertEqual(selected["H_asm_sparse"].shape, (4, 1))
        torch.testing.assert_close(selected["H_asm_sparse"], transfer[:, 2:3])
        torch.testing.assert_close(selected["z_query"], torch.tensor([0.007]))


if __name__ == "__main__":
    unittest.main()
