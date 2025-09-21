import numpy as np
from src.shape_generators.discrete_fourier_transform import (
    compute_dft, filter_dft_coefficients, inverse_dft
)
from src.shape_generators.discrete_wavelet_transform import (
    compute_multilevel_dwt, filter_dwt_coefficients, inverse_dwt
)
from src.visualizations import overlay_shapes
import pywt

# Generate an example dataset
fs = 1000
T = 1.0
t = np.linspace(0, T, int(fs * T), endpoint=False)
dataset = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# Compute DFT and filter (example freq bounds around signals)
freqs, dft_coeffs = compute_dft(dataset, sampling_rate=fs)
filtered_dft = filter_dft_coefficients(freqs, dft_coeffs, freq_bounds=(40, 130))

# Reconstruct time-domain DFT shape
reconstructed_dft = inverse_dft(filtered_dft)

# Compute DWT and filter (keep approximation and detail levels as desired)
wavelet_name = 'db4'
dwt_coeffs = compute_multilevel_dwt(dataset, wavelet=wavelet_name, level=4)
filtered_dwt = filter_dwt_coefficients(dwt_coeffs, levels_to_keep=[0, 1])  # example keep approx and first detail

# Reconstruct time-domain DWT shape
reconstructed_dwt = inverse_dwt(filtered_dwt, wavelet=wavelet_name)

# Prepare overlay data (same length now)
spectra = [
    (t, reconstructed_dft, "Reconstructed DFT Shape", "blue"),
    (t, reconstructed_dwt[:len(t)], "Reconstructed DWT Shape", "orange"),
]

overlay_shapes(spectra, title="Overlay: Reconstructed DFT vs DWT Shapes")
