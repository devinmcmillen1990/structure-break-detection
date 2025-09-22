import numpy as np
from src.shape_generators.discrete_fourier_transform import (
    compute_dft, filter_dft_coefficients
)
from src.shape_generators.discrete_wavelet_transform import (
    compute_multilevel_dwt, filter_dwt_coefficients
)
from src.trajection import create_trajectory_from_complex
from src.trajectoid_bodies import (
    star_body_from_points, rounded_hull_support_mesh, plot_meshes
)
from src.visualizations import (
    sweep_tube_around_trajectory
)


# Dataset generation as before
fs = 1000
T = 1.0
t = np.linspace(0, T, int(fs * T), endpoint=False)
dataset = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# DFT computation and filter
freqs, dft_coeffs = compute_dft(dataset, sampling_rate=fs)
filtered_dft = filter_dft_coefficients(freqs, dft_coeffs, freq_bounds=(40, 130))

# DWT computation and filter
wavelet_name = 'db4'
dwt_coeffs = compute_multilevel_dwt(dataset, wavelet=wavelet_name, level=4)
filtered_dwt = filter_dwt_coefficients(dwt_coeffs, levels_to_keep=[0, 1])  # approx + detail1

x_dft, y_dft, z_dft = create_trajectory_from_complex(filtered_dft)
X_dft, Y_dft, Z_dft = sweep_tube_around_trajectory(x_dft, y_dft, z_dft, radius=0.05)

x_dwt, y_dwt, z_dwt = create_trajectory_from_complex(np.concatenate(filtered_dwt))
X_dwt, Y_dwt, Z_dwt = sweep_tube_around_trajectory(x_dwt, y_dwt, z_dwt, radius=0.05)

pts_dft = np.column_stack([x_dft, y_dft, z_dft]).astype(np.float32)
pts_dwt = np.column_stack([x_dwt, y_dwt, z_dwt]).astype(np.float32)


# --- Rounded hull overlay ---
verts_rh_dft, faces_rh_dft = rounded_hull_support_mesh(pts_dft, r=0.08, ico_subdiv=3)
verts_rh_dwt, faces_rh_dwt = rounded_hull_support_mesh(pts_dwt, r=0.08, ico_subdiv=3)

plot_meshes([
    (verts_rh_dft, faces_rh_dft, "blue", "DFT Rounded Hull"),
    (verts_rh_dwt, faces_rh_dwt, "orange", "DWT Rounded Hull"),
], title="Overlay: Rounded Convex Hull Trajectoids")


# --- Star/support overlay ---
verts_sb_dft, faces_sb_dft = star_body_from_points(pts_dft, quantile=0.90, ico_subdiv=3)
verts_sb_dwt, faces_sb_dwt = star_body_from_points(pts_dwt, quantile=0.90, ico_subdiv=3)

plot_meshes([
    (verts_sb_dft, faces_sb_dft, "blue", "DFT Star Body"),
    (verts_sb_dwt, faces_sb_dwt, "orange", "DWT Star Body"),
], title="Overlay: Star/Support Trajectoids")
