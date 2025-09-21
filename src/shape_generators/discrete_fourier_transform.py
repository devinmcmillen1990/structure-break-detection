import numpy as np
from scipy.fft import fft, fftfreq, ifft


def compute_dft(time_series, sampling_rate=1.0):
    time_series = np.asarray(time_series)
    dft_coefficients = fft(time_series)
    frequencies = fftfreq(len(time_series), d=1 / sampling_rate)
    return frequencies, dft_coefficients


def filter_dft_coefficients(freqs, coeffs, freq_bounds=None, magnitude_threshold=None):
    filtered = coeffs.copy()

    if freq_bounds is not None:
        mask = (freqs >= freq_bounds[0]) & (freqs <= freq_bounds[1])
        filtered[~mask] = 0

    if magnitude_threshold is not None:
        filtered[np.abs(filtered) < magnitude_threshold] = 0

    return filtered


def inverse_dft(filtered_coeffs):
    """
    Compute inverse DFT to reconstruct time-domain signal from filtered coefficients.

    Args:
        filtered_coeffs (ndarray): Complex filtered DFT coefficients.

    Returns:
        ndarray: Real-valued reconstructed time-domain signal.
    """
    reconstructed = ifft(filtered_coeffs)
    return np.real(reconstructed)
