import numpy as np
import matplotlib.pyplot as plt

def plot_dft(frequencies, dft_coefficients, label="DFT", color="blue"):
    """
    Plots the magnitude spectrum of the DFT.

    Parameters:
        frequencies (ndarray): Frequency bins.
        dft_coefficients (ndarray): DFT result (complex values).
        label (str): Legend label for the plot.
        color (str): Color for the plot line.
    """
    plt.plot(frequencies, np.abs(dft_coefficients), label=label, color=color)

def overlay_shapes(list_of_spectra, title="Overlay of Shape Spectra"):
    """
    Plots multiple shape spectra on a single window for overlay comparison.

    Parameters:
        list_of_spectra (list of tuples): Each tuple must be (frequencies, coefficients, label, color)
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 5))
    for freqs, coeffs, label, color in list_of_spectra:
        plot_dft(freqs, coeffs, label=label, color=color)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
