"""Numerical contracts that must hold before connecting a live camera/SLM."""
import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from citl_proxy import (
    ProxyAdapter, make_camera_loss, measured_proxy_gradient, native_center_crop,
    phase_to_slm, quantize_phase_ste, slm_to_phase,
)


class PhaseContractTests(unittest.TestCase):
    def test_upload_round_trip_has_exact_file_orientation(self):
        commands = np.array([[0, 3, 100, 1022], [40, 500, 900, 17]], dtype=np.uint16)
        phase = slm_to_phase(commands)
        self.assertEqual(phase.shape, (4, 2))
        np.testing.assert_allclose(phase[:, 0].numpy(), commands[0, ::-1] * (2 * math.pi / 1023))
        np.testing.assert_array_equal(phase_to_slm(phase), commands)
        self.assertEqual(phase_to_slm(phase).dtype, np.uint16)

    def test_proxy_forward_evaluates_uploaded_phase_and_keeps_derivative(self):
        phase = torch.tensor([[-.1, .21], [3.1456, 7.1]], requires_grad=True)
        displayed = slm_to_phase(phase_to_slm(phase))
        proxy_phase = quantize_phase_ste(phase)
        torch.testing.assert_close(proxy_phase, displayed, rtol=0, atol=0)
        (proxy_phase.square().sum()).backward()
        torch.testing.assert_close(phase.grad, 2 * displayed)

    def test_options_round_trip_and_invalid_drive_values(self):
        commands = np.array([[1, 2], [15, 19], [24, 70]], dtype=np.uint16)
        for transpose in (False, True):
            for flip in (False, True):
                phase = slm_to_phase(commands, transpose=transpose, flip_first_axis=flip)
                np.testing.assert_array_equal(
                    phase_to_slm(phase, transpose=transpose, flip_first_axis=flip), commands)
        with self.assertRaises(ValueError):
            slm_to_phase(np.array([[1.1]]))
        with self.assertRaises(ValueError):
            phase_to_slm(torch.zeros((2, 2)), level_max=1023.5)


class MeasuredLossTests(unittest.TestCase):
    def test_measured_derivative_times_proxy_jacobian_not_proxy_loss(self):
        phase = torch.tensor([[.4, .8], [1.2, 1.6]], requires_grad=True)
        prediction = (3 * phase.square())[None, None]
        measured = torch.full_like(prediction, 5.)
        target = torch.full_like(prediction, 2.)
        mask = torch.tensor([[[[1., 0.], [1., 1.]]]])
        data_loss, upstream = measured_proxy_gradient(prediction, measured, target,
                                                      make_camera_loss("mse"), mask)
        prediction.backward(upstream)
        expected_upstream = mask * 2 * (measured - target) / mask.sum()
        torch.testing.assert_close(upstream, expected_upstream)
        torch.testing.assert_close(phase.grad, (expected_upstream * 6 * phase)[0, 0])
        self.assertAlmostEqual(float(data_loss), 9.)
        self.assertIsNone(measured.grad)

    def test_shape_all_valid_matches_offline_formula(self):
        p = torch.tensor([[[[1., 2.], [3., 7.]]]])
        t = torch.tensor([[[[2., 4.], [1., 9.]]]])
        actual = make_camera_loss("shape")(p, t)
        expected = F.smooth_l1_loss(p / p.mean(), t / t.mean())
        torch.testing.assert_close(actual, expected)

    def test_invalid_intensities_do_not_affect_shape_means_or_derivatives(self):
        p = torch.tensor([[[[2., float("nan")], [4., 999.]]]], requires_grad=True)
        t = torch.tensor([[[[1., float("nan")], [4., 999.]]]])
        mask = torch.tensor([[[[1., 0.], [1., 0.]]]])
        value = make_camera_loss("shape")(p, t, mask)
        expected = F.smooth_l1_loss(torch.tensor([2., 4.]) / 3., torch.tensor([1., 4.]) / 2.5)
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[0, 0, 0, 1]), 0.)
        self.assertEqual(float(p.grad[0, 0, 1, 1]), 0.)

    def test_roi_and_empty_mask_are_applied_together(self):
        p = torch.ones((1, 1, 4, 4))
        t = p.clone()
        p[..., 0, :] = 10
        self.assertEqual(float(make_camera_loss("mse", .5)(p, t)), 0.)
        mask = torch.ones_like(p)
        mask[..., 1:3, 1:3] = 0
        with self.assertRaisesRegex(ValueError, "No valid"):
            make_camera_loss("mse", .5)(p, t, mask)

    def test_weights_use_weighted_pixel_count(self):
        p = torch.tensor([[[[1., 4.]]]])
        t = torch.ones_like(p)
        mask = torch.tensor([[[[1., .5]]]])
        self.assertAlmostEqual(float(make_camera_loss("mse")(p, t, mask)), 3.)
        with self.assertRaises(ValueError):
            make_camera_loss("training")


class CameraContractTests(unittest.TestCase):
    @staticmethod
    def adapter(mode="global_percentile"):
        adapter = ProxyAdapter()
        adapter.device = torch.device("cpu")
        adapter.model_size = 2
        adapter.camera_scale = 4.
        adapter.camera_scale_mode = mode
        adapter.camera_black_level = 2.
        return adapter

    def test_native_crop_black_level_scale_and_mask_share_coordinates(self):
        adapter = self.adapter()
        raw = np.arange(16, dtype=np.float32).reshape(4, 4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 2] = True
        camera = adapter.camera_tensor(raw)
        torch.testing.assert_close(camera[0, 0], torch.tensor([[.75, 1.], [1.75, 2.]]))
        torch.testing.assert_close(adapter.camera_mask(mask)[0, 0], torch.tensor([[0., 1.], [0., 0.]]))

    def test_scaling_is_fixed_across_frames_and_raw_mode_omits_division(self):
        adapter = self.adapter()
        first = adapter.camera_tensor(np.full((2, 2), 6., dtype=np.float32))
        second = adapter.camera_tensor(np.full((2, 2), 10., dtype=np.float32))
        torch.testing.assert_close(second, first * 2)
        adapter.camera_scale_mode = "raw"
        torch.testing.assert_close(adapter.camera_tensor(np.ones((2, 2))), torch.zeros_like(first))
        torch.testing.assert_close(adapter.camera_tensor(np.full((2, 2), 6.)), torch.full_like(first, 4.))

    def test_nonfinite_data_requires_invalid_mask_and_bad_sampling_fails(self):
        adapter = self.adapter()
        raw = np.array([[6., float("nan")], [6., 6.]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            adapter.camera_tensor(raw)
        camera = adapter.camera_tensor(raw, valid_mask=np.isfinite(raw))
        self.assertTrue(torch.isfinite(camera).all())
        with self.assertRaisesRegex(ValueError, "parity"):
            adapter.camera_tensor(np.ones((3, 3)))
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            native_center_crop(torch.ones((1, 2)), 2)


if __name__ == "__main__":
    unittest.main()
