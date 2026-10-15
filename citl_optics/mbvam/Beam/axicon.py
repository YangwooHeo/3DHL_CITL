"""Refractive axicon parameters shared by simulation, CGH and calibration.

AX2510-A: UV fused silica, 10 degree base angle, 25.4 mm diameter,
350--700 nm AR coating. https://www.thorlabs.com/item/AX2510-A
The equivalent pitch describes a continuous phase ramp, not a binary grating.
"""

from dataclasses import dataclass
import math


def fused_silica_index(wavelength_m):
    """Malitson Sellmeier fit (wavelength in vacuum), at approximately 20 C.

    Source: https://doi.org/10.1364/JOSA.57.000674 (0.21--3.71 um).
    """
    wavelength_um = float(wavelength_m) * 1e6
    if not 0.21 <= wavelength_um <= 3.71:
        raise ValueError("Fused-silica fit requires a wavelength in [0.21, 3.71] um.")
    l2 = wavelength_um ** 2
    return math.sqrt(1 + sum(b * l2 / (l2 - c ** 2) for b, c in (
        (0.6961663, 0.0684043), (0.4079426, 0.1162414),
        (0.8974794, 9.896161))))


@dataclass(frozen=True)
class RefractiveAxicon:
    """Ideal plano-first axicon, collimated axial incidence.

    exit_index is the medium touching the CONICAL surface. For an air-mounted
    axicon followed by a flat resin interface, leave exit_index=1.0 and pass
    the resin index to cone_angle()/the ASM separately. This is an effective
    exit-plane scalar phase model; tip rounding and Fresnel losses are omitted.
    """
    wavelength_m: float = 473e-9
    base_angle_deg: float = 10.0
    exit_index: float = 1.0
    diameter_m: float = 25.4e-3

    def __post_init__(self):
        if not math.isfinite(self.exit_index) or self.exit_index <= 0:
            raise ValueError("exit_index must be finite and positive.")
        if not 0 < self.base_angle_deg < 90:
            raise ValueError("base_angle_deg must be in (0, 90).")
        if not math.isfinite(self.diameter_m) or self.diameter_m <= 0:
            raise ValueError("diameter_m must be finite and positive.")
        if self.glass_index * math.sin(self.base_angle_rad) >= self.exit_index:
            raise ValueError("Conical exit surface has total internal reflection.")

    @property
    def glass_index(self):
        return fused_silica_index(self.wavelength_m)

    @property
    def base_angle_rad(self):
        return math.radians(self.base_angle_deg)

    @property
    def deflection_rad(self):
        alpha = self.base_angle_rad
        return math.asin(self.glass_index / self.exit_index * math.sin(alpha)) - alpha

    @property
    def transverse_na(self):
        """Signed n*sin(beta); conserved across subsequent planar interfaces."""
        return self.exit_index * math.sin(self.deflection_rad)

    @property
    def radial_frequency(self):
        return self.transverse_na / self.wavelength_m

    @property
    def equivalent_pitch_m(self):
        return math.inf if self.radial_frequency == 0 else 1 / abs(self.radial_frequency)

    def cone_angle(self, n_medium=1.0):
        if not math.isfinite(n_medium) or n_medium <= abs(self.transverse_na):
            raise ValueError("n_medium must be positive and support this transverse NA.")
        return math.asin(self.transverse_na / n_medium)

    def sampling(self, pixel_pitch_m, slm_bandwidth=None):
        """Nyquist lower bounds, not an error guarantee for pixelated fields.

        Default bandwidth covers the diagonal of the SLM's principal Nyquist
        square. Pixel-edge harmonics still require a convergence check.
        """
        if not math.isfinite(pixel_pitch_m) or pixel_pitch_m <= 0:
            raise ValueError("pixel_pitch_m must be finite and positive.")
        bandwidth = math.sqrt(2) / (2 * pixel_pitch_m) if slm_bandwidth is None else slm_bandwidth
        if not math.isfinite(bandwidth) or bandwidth < 0:
            raise ValueError("slm_bandwidth must be finite and nonnegative.")
        carrier = abs(self.radial_frequency)
        return {
            "carrier_field_min": math.ceil(2 * pixel_pitch_m * carrier),
            "carrier_intensity_min": math.ceil(4 * pixel_pitch_m * carrier),
            "cgh_intensity_min": math.ceil(4 * pixel_pitch_m * (carrier + bandwidth)),
        }


# Physical defaults for the current 8 um SLM and 0.8x relay.
AX2510_UPSAMPLE = 8
AX2510_ROI_WIDTH_M = 1.28e-3


def roi_pixels(width_m, pixel_pitch_m, upsample_factor):
    """Use a fixed physical output width when comparing sampling factors."""
    return max(2, 2 * round(width_m * upsample_factor / pixel_pitch_m / 2))
