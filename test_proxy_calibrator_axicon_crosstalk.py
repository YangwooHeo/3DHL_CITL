import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from proxy_calibrator_axicon import AxiconProxyParameters, ProxyCalibrationDataset


def make_proxy(kernel_size: int) -> AxiconProxyParameters:
    return AxiconProxyParameters(
        num_zernike=1,
        source_grid_shape=(2, 2),
        crosstalk_kernel_size=kernel_size,
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

        self.assertTrue(bool(torch.all(kernel >= 0)))
        torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))
        torch.testing.assert_close(filtered, drive)

    def test_filter_operates_across_zero_two_pi_boundary_before_phase(self):
        proxy = make_proxy(3)
        with torch.no_grad():
            proxy.crosstalk_kernel_logits.zero_()
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

        gradient = proxy.crosstalk_kernel_logits.grad
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


if __name__ == "__main__":
    unittest.main()
