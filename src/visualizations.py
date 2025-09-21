import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def normalize_min_max(arr):
    arr = np.array(arr)
    if np.ptp(arr) == 0:  # avoid div by zero if constant array
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / np.ptp(arr)


def upsample_to_length(arr, target_length):
    x_orig = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_length)
    f_interp = interp1d(x_orig, arr, kind='linear')
    return f_interp(x_new)


def plot_shape(x_axis, values, label="Shape", color="blue"):
    """
    Generic plot function for any 1D shape spectrum or coefficients.

    Parameters:
        x_axis (array-like): X-axis values (frequency, index, time, etc.)
        values (array-like): Magnitude or coefficient values.
        label (str): Legend label.
        color (str): Line color.
    """
    plt.plot(x_axis, values, label=label, color=color)


def overlay_shapes(list_of_shapes, title="Overlay of Shape Spectra"):
    plt.figure(figsize=(10, 5))

    max_length = max(len(values) for _, values, _, _ in list_of_shapes)
    for i, (x, vals, label, color) in enumerate(list_of_shapes):
        print(f"Shape {i} '{label}': original length {len(vals)}, x length {len(x)}, min {np.min(vals)}, max {np.max(vals)}")
        norm_vals = normalize_min_max(vals)
        print(f"Shape {i} '{label}': normalized min {np.min(norm_vals)}, max {np.max(norm_vals)}")

        if len(vals) != max_length:
            norm_vals = upsample_to_length(norm_vals, max_length)
            print(f"Shape {i} '{label}': upsampled to length {len(norm_vals)}")

        if len(x) != max_length:
            x_new = np.linspace(x[0], x[-1], max_length)
        else:
            x_new = x

        plot_shape(x_new, norm_vals, label=label, color=color)

    plt.xlabel("Index / Frequency")
    plt.ylabel("Normalized Magnitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
