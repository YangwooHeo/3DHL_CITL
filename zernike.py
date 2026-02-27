
import math
import torch
import numpy as np
import torch.fft
from aotools.functions import zernikeArray

def combine_zernike_basis(coeffs, basis, return_phase=False):
    """
    Multiplies the Zernike coefficients and basis functions while preserving
    dimensions

    :param coeffs: torch tensor with coeffs, see propagation_ASM_zernike
    :param basis: the output of compute_zernike_basis, must be same length as coeffs
    :param return_phase:
    :return: A Complex64 tensor that combines coeffs and basis.
    """

    if len(coeffs.shape) < 3:
        coeffs = torch.reshape(coeffs, (coeffs.shape[0], 1, 1))

    # combine zernike basis and coefficients
    zernike = (coeffs * basis).sum(0, keepdim=True)

    # shape to [1, len(coeffs), H, W]
    zernike = zernike.unsqueeze(0)

    # convert to Pytorch Complex tensor
    real, imag = polar_to_rect(torch.ones_like(zernike), zernike)
    return torch.complex(real, imag)


def compute_zernike_basis(num_polynomials, field_res, dtype=torch.float32, wo_piston=False):
    """Computes a set of Zernike basis function with resolution field_res

    num_polynomials: number of Zernike polynomials in this basis
    field_res: [height, width] in px, any list-like object
    dtype: torch dtype for computation at different precision
    """

    # size the zernike basis to avoid circular masking
    zernike_diam = int(np.ceil(np.sqrt(field_res[0]**2 + field_res[1]**2)))

    # create zernike functions

    if not wo_piston:
        zernike = zernikeArray(num_polynomials, zernike_diam)
    else:  # 200427 - exclude pistorn term
        idxs = range(2, 2 + num_polynomials)
        zernike = zernikeArray(idxs, zernike_diam)

    zernike = crop_image(zernike, field_res, pytorch=False)

    # convert to tensor and create phase
    zernike = torch.tensor(zernike, dtype=dtype, requires_grad=False)

    return zernike

def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field
