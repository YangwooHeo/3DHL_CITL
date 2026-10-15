"""Exercise the live device boundary using fakes; never import a vendor SDK."""

import ctypes
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from citl_hardware import HardwareConfig, HololithCameraHardware, prepare_slm_commands


class Node:
    def __init__(self, value=0):
        self.value = value
        self.clamp = None

    def SetValue(self, value):
        self.value = value if self.clamp is None else self.clamp

    def GetValue(self):
        return self.value


class Grab:
    def __init__(self, array, success=True):
        self.Array = array
        self.success = success
        self.released = False

    def GrabSucceeded(self):
        return self.success

    def GetErrorCode(self):
        return 7

    def GetErrorDescription(self):
        return "fake acquisition failure"

    def Release(self):
        self.released = True


class Camera:
    def __init__(self):
        for name in ("PixelFormat", "ExposureAuto", "GainAuto", "TriggerMode", "AcquisitionMode",
                     "ExposureMode", "ExposureTime", "Gain", "BlackLevel"):
            setattr(self, name, Node())
        self.grabbing = False
        self.opens = self.closes = self.starts = 0
        self.results = []
        self.sequence = []
        self.exposures = []

    def Open(self):
        self.opens += 1

    def Close(self):
        self.closes += 1

    def StartGrabbing(self, strategy):
        self.grabbing = True
        self.starts += 1
        self.exposures.append(self.ExposureTime.value)

    def StopGrabbing(self):
        self.grabbing = False

    def IsGrabbing(self):
        return self.grabbing

    def GetDeviceInfo(self):
        return SimpleNamespace(GetModelName=lambda: "fake Basler", GetSerialNumber=lambda: "123")

    def RetrieveResult(self, timeout, handling):
        result = self.sequence.pop(0) if self.sequence else Grab(
            np.full((2, 4), int(self.ExposureTime.value), dtype=np.uint16))
        self.results.append(result)
        return result


class Slm:
    SLM_OK = 0
    SLM_BS = 2
    MODE_MEMORY = 0

    def __init__(self):
        self.calls = []
        self.uploads = []
        self.fail_upload = False

    def SLM_Ctrl_ReadSU(self, number):
        return self.SLM_OK

    def SLM_Ctrl_ReadVI(self, number, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32)).contents.value = self.MODE_MEMORY
        return self.SLM_OK

    def SLM_Ctrl_WriteMI(self, number, slot, width, height, flags, pointer):
        self.calls.append(("SLM_Ctrl_WriteMI", (number, slot, width, height)))
        self.uploads.append(np.ctypeslib.as_array(pointer, shape=(width * height,)).copy().reshape(height, width))
        return 1 if self.fail_upload else self.SLM_OK

    def __getattr__(self, name):
        if name.startswith("SLM_Ctrl_"):
            def call(*args):
                self.calls.append((name, args))
                return self.SLM_OK
            return call
        raise AttributeError(name)


class Arduino:
    def __init__(self):
        self.commands = []
        self.opens = self.closes = 0

    def openPort(self):
        self.opens += 1

    def closePort(self):
        self.closes += 1

    def writeCommand(self, text):
        self.commands.append(text)


def fake_hardware(**kwargs):
    camera, slm, arduino = Camera(), Slm(), Arduino()
    factory = SimpleNamespace(CreateFirstDevice=lambda: object())
    pylon = SimpleNamespace(TlFactory=SimpleNamespace(GetInstance=lambda: factory),
                            InstantCamera=lambda device: camera,
                            GrabStrategy_LatestImageOnly=1, TimeoutHandling_ThrowException=1)
    sdk = SimpleNamespace(pylon=pylon, slm=slm,
                          read_config=lambda path: SimpleNamespace(slm=SimpleNamespace(slm_number=1)),
                          arduino_class=lambda cfg: arduino)
    config = HardwareConfig(config_path="fake.yaml", raw_shape=(2, 4), slm_shape=(4, 6),
                            slm_settle_s=0, laser_settle_s=0)
    hw = HololithCameraHardware(replace(config, **kwargs), _sdk=sdk)
    return hw, camera, slm, arduino


class HardwareTests(unittest.TestCase):
    def test_upload_preserves_orientation_and_exact_integer_codes(self):
        source = np.array([[0, 1023, 15], [9, 512, 88]], dtype=np.uint16)
        padded = prepare_slm_commands(source, (4, 6))
        np.testing.assert_array_equal(padded[1:3, 1:4], source)
        self.assertEqual(np.count_nonzero(padded), np.count_nonzero(source))
        self.assertTrue(padded.flags.c_contiguous)
        self.assertEqual(padded.dtype, np.uint16)
        for invalid in (source.astype(float), np.array([[-1]]), np.array([[1024]])):
            with self.assertRaises((ValueError, TypeError)):
                prepare_slm_commands(invalid, (4, 6))

    def test_devices_open_once_and_slot_is_freed_before_every_upload(self):
        hw, camera, slm, arduino = fake_hardware()
        with hw:
            for value in (80, 90):
                hw.upload_phase(np.full((4, 6), value, dtype=np.uint16))
                capture = hw.capture_bracket([300, 1000])
                self.assertEqual(set(capture), {300, 1000})
                np.testing.assert_array_equal(capture[300], np.full((2, 4), 300))
                self.assertFalse(camera.grabbing)
                self.assertEqual(arduino.commands[-1], "laser 0 0")
        self.assertEqual(camera.opens, 1)
        self.assertEqual(camera.closes, 1)
        self.assertEqual(arduino.opens, 1)
        self.assertEqual(arduino.closes, 1)
        self.assertEqual(len(slm.uploads), 2)
        for i, (name, _) in enumerate(slm.calls):
            if name == "SLM_Ctrl_WriteMI":
                self.assertEqual(slm.calls[i - 1][0], "SLM_Ctrl_WriteGS")
                self.assertEqual(slm.calls[i + 1][0], "SLM_Ctrl_WriteDS")
        self.assertTrue(all(result.released for result in camera.results))
        self.assertEqual(camera.exposures, [300, 1000, 300, 1000])

    def test_flush_then_capture_preserves_exact_dn_for_inverse_lut_and_hdr(self):
        hw, camera, _, _ = fake_hardware(flush_frames=1)
        stale = Grab(np.full((2, 4), 9, dtype=np.uint16))
        first = np.full((2, 4), 100, dtype=np.uint16)
        first[0, 0] = 4095
        first[0, 1] = 4088
        camera.sequence = [stale, Grab(first)]
        with hw:
            hw.upload_phase(np.zeros((4, 6), dtype=np.uint16))
            image = hw.capture_bracket([300])[300]
        np.testing.assert_array_equal(image, first)
        self.assertTrue(stale.released)

    def test_config_rejects_fractional_nonfinite_and_raw_averaging_settings(self):
        for kwargs in (dict(average_frames=2), dict(average_frames=1.0),
                       dict(laser_pwm=152.5), dict(flush_frames=True), dict(timeout_ms=2.5),
                       dict(slm_settle_s=float("nan")), dict(laser_settle_s=float("inf")),
                       dict(slm_poll_s=0), dict(camera_gain=float("nan")),
                       dict(raw_shape=(2.0, 4)), dict(slm_shape=(4, False))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fake_hardware(**kwargs)

    def test_failure_releases_grab_disables_laser_and_closes_devices(self):
        hw, camera, slm, arduino = fake_hardware(flush_frames=0)
        failed = Grab(np.zeros((2, 4), dtype=np.uint16), success=False)
        camera.sequence = [failed]
        with self.assertRaisesRegex(RuntimeError, "fake acquisition failure"):
            with hw:
                hw.upload_phase(np.zeros((4, 6), dtype=np.uint16))
                hw.capture_bracket([300])
        self.assertTrue(failed.released)
        self.assertEqual(arduino.commands[-1], "laser 0 0")
        self.assertEqual(camera.closes, 1)
        self.assertEqual(arduino.closes, 1)
        self.assertEqual(slm.calls[-1][0], "SLM_Ctrl_Close")

    def test_upload_error_never_captures_or_enables_laser(self):
        hw, camera, slm, arduino = fake_hardware()
        slm.fail_upload = True
        with self.assertRaisesRegex(RuntimeError, "SLM_Ctrl_WriteMI"):
            with hw:
                hw.upload_phase(np.zeros((4, 6), dtype=np.uint16))
        self.assertEqual(camera.starts, 0)
        self.assertNotIn("laser 152 0", arduino.commands)

    def test_invalid_brackets_fail_before_laser_enable(self):
        hw, _, _, arduino = fake_hardware()
        with hw:
            hw.upload_phase(np.zeros((4, 6), dtype=np.uint16))
            for exposures in ([], [300, 300], [300.5], [300.0], [True], [0], ["300"]):
                with self.subTest(exposures=exposures), self.assertRaises(ValueError):
                    hw.capture_bracket(exposures)
        self.assertNotIn("laser 152 0", arduino.commands)

    def test_exposure_clamping_is_not_silently_used_for_hdr(self):
        hw, camera, _, arduino = fake_hardware()
        camera.ExposureTime.clamp = 200
        with self.assertRaisesRegex(RuntimeError, "camera uses 200"):
            with hw:
                hw.upload_phase(np.zeros((4, 6), dtype=np.uint16))
                hw.capture_bracket([300])
        self.assertEqual(arduino.commands[-1], "laser 0 0")
        self.assertEqual(camera.starts, 0)

    def test_reject_wrong_sensor_geometry_without_leaving_laser_on(self):
        hw, camera, _, arduino = fake_hardware(flush_frames=0)
        camera.sequence = [Grab(np.zeros((4, 2), dtype=np.uint16))]
        with self.assertRaisesRegex(ValueError, "check camera ROI"):
            with hw:
                hw.upload_phase(np.zeros((4, 6), dtype=np.uint16))
                hw.capture_bracket([300])
        self.assertEqual(arduino.commands[-1], "laser 0 0")


if __name__ == "__main__":
    unittest.main()
