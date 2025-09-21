import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Generate a sample signal
fs = 1000  # Sampling frequency (Hz)
T = 1.0    # Duration (seconds)
t = np.linspace(0, T, int(fs * T), endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# Compute DFT using FFT
signal_fft = fft(signal)
frequencies = fftfreq(len(signal), 1 / fs)

# Plot frequency spectrum (magnitude)
plt.figure(figsize=(10, 5))
plt.plot(frequencies, np.abs(signal_fft))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('DFT: Signal Frequency Spectrum')
plt.grid(True)
plt.show()
