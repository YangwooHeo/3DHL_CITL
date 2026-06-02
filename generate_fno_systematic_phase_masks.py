import csv
import math
from pathlib import Path

import numpy as np


OUTPUT_ROOT = Path(r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\FNO Sample Generation")
PHASE_DIR = OUTPUT_ROOT / "0.Phase_Mask"
MANIFEST_PATH = OUTPUT_ROOT / "systematic_phase_manifest.csv"

HEIGHT = 1200
WIDTH = 1600
LEVEL_MAX = 1023
RNG_SEED = 20260528


def coordinate_grid():
    y = np.arange(HEIGHT, dtype=np.float32) - (HEIGHT - 1) / 2
    x = np.arange(WIDTH, dtype=np.float32) - (WIDTH - 1) / 2
    return np.meshgrid(x, y)


X, Y = coordinate_grid()


def sanitize_level(arr):
    arr = np.nan_to_num(arr, copy=False)
    arr = np.clip(np.rint(arr), 0, LEVEL_MAX)
    return arr.astype(np.uint16, copy=False)


def phase_rad_to_level(phase_rad):
    wrapped = np.mod(phase_rad, 2 * np.pi)
    return sanitize_level(wrapped * (LEVEL_MAX / (2 * np.pi)))


def save_mask(name, arr, manifest, category, **params):
    path = PHASE_DIR / f"{name}.npy"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing mask: {path}")
    np.save(path, sanitize_level(arr))
    row = {"filename": path.name, "category": category}
    row.update(params)
    manifest.append(row)


def projection(angle_deg):
    theta = np.deg2rad(angle_deg)
    return X * np.cos(theta) + Y * np.sin(theta)


def projection_pair(angle_deg):
    theta = np.deg2rad(angle_deg)
    u = X * np.cos(theta) + Y * np.sin(theta)
    v = -X * np.sin(theta) + Y * np.cos(theta)
    return u, v


def generate_sine_sweeps(manifest):
    periods = [8, 16, 32]
    angles = [0, 45, 90, 135]
    offsets = [0.0, 0.25, 0.50, 0.75]
    depths = [256, 512, 1023]

    for period in periods:
        for angle in angles:
            coord = projection(angle)
            for offset_idx, offset_frac in enumerate(offsets):
                offset = offset_frac * period
                carrier = np.cos(2 * np.pi * (coord - offset) / period)
                for depth in depths:
                    arr = 0.5 * depth * (carrier + 1.0)
                    save_mask(
                        f"sys_sine_p{period:03d}_a{angle:03d}_off{offset_idx}_d{depth:04d}",
                        arr, manifest, "sine",
                        period_px=period, angle_deg=angle,
                        offset_fraction=offset_frac, depth_level=depth,
                    )


def generate_linear_ramps(manifest):
    periods = [16, 32, 64]
    angles = [0, 45, 90, 135]
    offsets = [0.0, 0.25, 0.50, 0.75]
    depths = [512, 1023]

    for period in periods:
        for angle in angles:
            coord = projection(angle)
            for offset_idx, offset_frac in enumerate(offsets):
                offset = offset_frac * period
                ramp = np.mod(coord - offset, period) / period
                for depth in depths:
                    save_mask(
                        f"sys_ramp_p{period:03d}_a{angle:03d}_off{offset_idx}_d{depth:04d}",
                        depth * ramp, manifest, "linear_ramp",
                        period_px=period, angle_deg=angle,
                        offset_fraction=offset_frac, depth_level=depth,
                    )


def generate_checkerboards(manifest):
    periods = [8, 16, 32]
    angles = [0, 45, 90, 135]
    offsets = [(0.0, 0.0), (0.5, 0.5)]
    depths = [128] #, 256, 512]

    for period in periods:
        for angle in angles:
            u, v = projection_pair(angle)
            for offset_idx, (off_u, off_v) in enumerate(offsets):
                squares = (
                    np.floor((u - off_u * period) / period) +
                    np.floor((v - off_v * period) / period)
                )
                binary = np.mod(squares, 2)
                for depth in depths:
                    save_mask(
                        f"sys_checker_p{period:03d}_a{angle:03d}_off{offset_idx}_d{depth:04d}",
                        depth * binary, manifest, "checkerboard",
                        period_px=period, angle_deg=angle,
                        offset_u_fraction=off_u, offset_v_fraction=off_v,
                        depth_level=depth,
                    )


def generate_square_rings(manifest):
    sizes = [160, 240, 320, 480]
    thicknesses = [12, 24, 36]
    angles = [0, 45]
    depths = [256, 512] #, 1023]

    for size in sizes:
        for thickness in thicknesses:
            for angle in angles:
                u, v = projection_pair(angle)
                outer = np.maximum(np.abs(u), np.abs(v)) <= size / 2
                inner = np.maximum(np.abs(u), np.abs(v)) < max(size / 2 - thickness, 1)
                ring = (outer & ~inner).astype(np.float32)
                for depth in depths:
                    save_mask(
                        f"sys_square_ring_s{size:03d}_t{thickness:03d}_a{angle:03d}_d{depth:04d}",
                        depth * ring, manifest, "square_ring",
                        size_px=size, thickness_px=thickness,
                        angle_deg=angle, depth_level=depth,
                    )


NOLL_TO_NM = {
    2: (1, -1), 3: (1, 1),
    4: (2, -2), 5: (2, 0), 6: (2, 2),
    7: (3, -3), 8: (3, -1), 9: (3, 1), 10: (3, 3),
    11: (4, -4), 12: (4, -2), 13: (4, 0), 14: (4, 2),
}


def zernike_radial(n, m_abs, rho):
    out = np.zeros_like(rho, dtype=np.float32)
    for k in range((n - m_abs) // 2 + 1):
        coef = ((-1) ** k * math.factorial(n - k) /
                (math.factorial(k) *
                 math.factorial((n + m_abs) // 2 - k) *
                 math.factorial((n - m_abs) // 2 - k)))
        out += coef * rho ** (n - 2 * k)
    return out


def zernike(n, m, rho, theta):
    m_abs = abs(m)
    radial = zernike_radial(n, m_abs, rho)
    if m < 0:
        z = radial * np.sin(m_abs * theta)
    elif m > 0:
        z = radial * np.cos(m_abs * theta)
    else:
        z = radial
    return z.astype(np.float32)


def zernike_basis():
    radius = 0.48 * min(HEIGHT, WIDTH)
    xn = X / radius
    yn = Y / radius
    rho = np.sqrt(xn ** 2 + yn ** 2)
    theta = np.arctan2(yn, xn)
    aperture = rho <= 1.0

    basis = {}
    for j, (n, m) in NOLL_TO_NM.items():
        z = np.zeros_like(rho, dtype=np.float32)
        z[aperture] = zernike(n, m, rho[aperture], theta[aperture])
        basis[j] = z
    return basis, aperture


def generate_zernike_single(manifest, basis):
    coeffs_waves = [0.10, 0.25, 0.50]
    for j, z in basis.items():
        n, m = NOLL_TO_NM[j]
        for coeff in coeffs_waves:
            for sign in [-1, 1]:
                signed_coeff = sign * coeff
                phase = 2 * np.pi * signed_coeff * z
                sign_name = "pos" if sign > 0 else "neg"
                save_mask(
                    f"sys_zernike_j{j:02d}_{sign_name}_c{int(coeff * 100):03d}",
                    phase_rad_to_level(phase), manifest, "zernike_single",
                    noll_index=j, radial_order=n, azimuthal_order=m,
                    coefficient_waves=signed_coeff,
                )


def generate_zernike_combos(manifest, basis, aperture):
    rng = np.random.default_rng(RNG_SEED)
    modes = sorted(basis)
    rms_targets = [0.15, 0.30, 0.50]

    for combo_idx in range(32):
        weights = rng.normal(0.0, 1.0, size=len(modes)).astype(np.float32)
        weights *= np.linspace(1.0, 0.45, len(modes), dtype=np.float32)
        wavefront = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        for w, mode in zip(weights, modes):
            wavefront += w * basis[mode]
        rms = float(np.sqrt(np.mean(wavefront[aperture] ** 2)))
        wavefront /= max(rms, 1e-8)

        for rms_target in rms_targets:
            phase = 2 * np.pi * rms_target * wavefront
            save_mask(
                f"sys_zernike_combo_{combo_idx:02d}_rms{int(rms_target * 100):03d}",
                phase_rad_to_level(phase), manifest, "zernike_combo",
                combo_index=combo_idx, rms_waves=rms_target,
            )


def generate_radial_rings(manifest):
    radius = np.sqrt(X ** 2 + Y ** 2)
    periods = [16, 32, 64]
    offsets = [0.0, 0.25, 0.50, 0.75]
    depths = [256, 512, 1023]

    for period in periods:
        for offset_idx, offset_frac in enumerate(offsets):
            offset = offset_frac * period
            carrier = np.cos(2 * np.pi * (radius - offset) / period)
            for depth in depths:
                save_mask(
                    f"sys_radial_ring_p{period:03d}_off{offset_idx}_d{depth:04d}",
                    0.5 * depth * (carrier + 1.0), manifest, "radial_ring",
                    period_px=period, offset_fraction=offset_frac,
                    depth_level=depth,
                )


def generate_local_bumps(manifest):
    rng = np.random.default_rng(RNG_SEED + 1)
    coeffs_waves = [0.15, 0.35, 0.70]

    for bump_idx in range(20):
        x0 = rng.uniform(-0.35 * WIDTH, 0.35 * WIDTH)
        y0 = rng.uniform(-0.35 * HEIGHT, 0.35 * HEIGHT)
        sigma_x = rng.uniform(35.0, 150.0)
        sigma_y = rng.uniform(35.0, 150.0)
        angle = rng.uniform(0.0, np.pi)
        ca, sa = np.cos(angle), np.sin(angle)
        u = (X - x0) * ca + (Y - y0) * sa
        v = -(X - x0) * sa + (Y - y0) * ca
        bump = np.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2)).astype(np.float32)
        for coeff in coeffs_waves:
            phase = 2 * np.pi * coeff * bump
            save_mask(
                f"sys_local_bump_{bump_idx:02d}_c{int(coeff * 100):03d}",
                phase_rad_to_level(phase), manifest, "local_bump",
                bump_index=bump_idx, coefficient_waves=coeff,
                x0_px=float(x0), y0_px=float(y0),
                sigma_x_px=float(sigma_x), sigma_y_px=float(sigma_y),
                angle_rad=float(angle),
            )


def write_manifest(manifest):
    keys = ["filename", "category"]
    for row in manifest:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in manifest:
            writer.writerow(row)


def main():
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    #generate_sine_sweeps(manifest)
    #generate_linear_ramps(manifest)
    generate_checkerboards(manifest)
    generate_square_rings(manifest)
    #basis, aperture = zernike_basis()
    #generate_zernike_single(manifest, basis)
    #generate_zernike_combos(manifest, basis, aperture)
    #generate_radial_rings(manifest)
    #generate_local_bumps(manifest)
    #write_manifest(manifest)

    counts = {}
    for row in manifest:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"Saved {len(manifest)} phase masks to {PHASE_DIR}")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
