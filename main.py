import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from src.shape_generators.discrete_fourier_transform import (
    compute_dft, filter_dft_coefficients, inverse_dft
)
from src.shape_generators.discrete_wavelet_transform import (
    compute_multilevel_dwt, filter_dwt_coefficients, inverse_dwt
)
from src.visualizations import (
    overlay_shapes, sweep_tube_around_trajectory, plot_swept_surface
)
from src.trajection import create_trajectory_from_complex

# Dataset generation as before
fs = 1000
T = 1.0
t = np.linspace(0, T, int(fs * T), endpoint=False)
dataset = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# DFT computation and filter
freqs, dft_coeffs = compute_dft(dataset, sampling_rate=fs)
filtered_dft = filter_dft_coefficients(freqs, dft_coeffs, freq_bounds=(40, 130))
reconstructed_dft_signal = inverse_dft(filtered_dft)

# DWT computation and filter
wavelet_name = 'db4'
dwt_coeffs = compute_multilevel_dwt(dataset, wavelet=wavelet_name, level=4)
filtered_dwt = filter_dwt_coefficients(dwt_coeffs, levels_to_keep=[0, 1])  # approx + detail1
reconstructed_dwt_signal = inverse_dwt(filtered_dwt, wavelet=wavelet_name)

x_dft, y_dft, z_dft = create_trajectory_from_complex(filtered_dft)
X_dft, Y_dft, Z_dft = sweep_tube_around_trajectory(x_dft, y_dft, z_dft, radius=0.05)

x_dwt, y_dwt, z_dwt = create_trajectory_from_complex(np.concatenate(filtered_dwt))
X_dwt, Y_dwt, Z_dwt = sweep_tube_around_trajectory(x_dwt, y_dwt, z_dwt, radius=0.05)

# Plot DFT swept surface
plot_swept_surface(X_dft, Y_dft, Z_dft, title="DFT Trajectory Swept Surface")

# Plot DWT swept surface
plot_swept_surface(X_dwt, Y_dwt, Z_dwt, title="DWT Trajectory Swept Surface")




# # Overlay in 2D as before
# spectra = [
#     (t, reconstructed_dft_signal, "Reconstructed DFT", "blue"),
#     (t, reconstructed_dwt_signal[:len(t)], "Reconstructed DWT", "orange"),
# ]
# overlay_shapes(spectra, title="Overlay: Reconstructed DFT vs DWT")

# # Prepare 3D trajectoid coordinates
# x_dft, y_dft, z_dft = create_trajectory_from_complex(filtered_dft)
# x_dwt, y_dwt, z_dwt = create_trajectory_from_complex(np.concatenate(filtered_dwt))  # flatten coeffs list

# # Create 3D figure and axis
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot both trajectoids on same axis
# ax.plot(x_dft, y_dft, z_dft, label='DFT Trajectoid', color='blue')
# ax.plot(x_dwt, y_dwt, z_dwt, label='DWT Trajectoid', color='orange')

# # Set labels and legend
# ax.set_xlabel('Index / Time')
# ax.set_ylabel('Normalized Magnitude')
# ax.set_zlabel('Phase')
# ax.set_title('Overlay: DFT vs DWT Trajectoids')
# ax.legend()

# plt.show()
