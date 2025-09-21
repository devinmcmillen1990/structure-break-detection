import numpy as np
from scipy.fft import fft, fftfreq

def compute_dft(time_series, sampling_rate=1.0):
    """
    Computes the Discrete Fourier Transform (DFT) for a given time series.

    Parameters:
        time_series (array-like): The input time series data.
        sampling_rate (float): Sampling rate in Hz (default: 1.0).

    Returns:
        frequencies (ndarray): Array of frequency bins.
        dft_coefficients (ndarray): Complex DFT coefficients.
    """
    time_series = np.asarray(time_series)
    dft_coefficients = fft(time_series)
    frequencies = fftfreq(len(time_series), d=1/sampling_rate)
    return frequencies, dft_coefficients
