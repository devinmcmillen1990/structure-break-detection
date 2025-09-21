import numpy as np
import pywt


def compute_multilevel_dwt(time_series, wavelet='db1', level=None):
    max_level = pywt.dwt_max_level(len(time_series), pywt.Wavelet(wavelet).dec_len)
    if level is None or level > max_level:
        level = max_level
    coeffs = pywt.wavedec(time_series, wavelet, level=level)
    return coeffs


def filter_dwt_coefficients(coeffs, levels_to_keep):
    filtered = []
    for idx, c in enumerate(coeffs):
        if idx in levels_to_keep:
            filtered.append(c)
        else:
            filtered.append(np.zeros_like(c))
    return filtered


def inverse_dwt(filtered_coeffs, wavelet='db1'):
    """
    Compute inverse DWT to reconstruct time-domain signal from filtered coefficients.

    Args:
        filtered_coeffs (list): Filtered wavelet coeffs list [cA_n, cD_n, ..., cD_1].
        wavelet (str): Wavelet name used originally.

    Returns:
        ndarray: Reconstructed time domain signal.
    """
    reconstructed = pywt.waverec(filtered_coeffs, wavelet)
    return reconstructed
