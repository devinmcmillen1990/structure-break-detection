from src.shape_generators.discrete_fourier_transform import compute_dft
from src.visualizations import overlay_shapes

import numpy as np

# Generate example time series data
fs = 1000
T = 1.0
t = np.linspace(0, T, int(fs * T), endpoint=False)
signal1 = np.sin(2 * np.pi * 50 * t)
signal2 = np.sin(2 * np.pi * 120 * t)

# Compute DFTs
freqs1, dft1 = compute_dft(signal1, sampling_rate=fs)
freqs2, dft2 = compute_dft(signal2, sampling_rate=fs)

# Prepare for visualization (only positive frequencies for clarity)
mask = freqs1 > 0
spectra = [
    (freqs1[mask], dft1[mask], "Signal 1", "blue"),
    (freqs2[mask], dft2[mask], "Signal 2", "orange"),
]
overlay_shapes(spectra, title="Overlay: DFT Shapes")

# To run: from the project root
# > python main.py
