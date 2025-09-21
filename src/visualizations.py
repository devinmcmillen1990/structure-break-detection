import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D


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

def plot_trajectoid(x, y, z, label="Trajectoid", color="blue"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label=label, color=color)
    ax.set_xlabel('Time / Index')
    ax.set_ylabel('Normalized Magnitude')
    ax.set_zlabel('Phase or Scale')
    ax.legend()
    plt.title(label)
    plt.show()

def sweep_tube_around_trajectory(x, y, z, radius=0.05, num_circle_points=20):
    """
    Generate a tubular mesh sweeping a circle of given radius along trajectory curve.

    Args:
        x, y, z: 1D arrays representing trajectory coordinates.
        radius: radius of circular cross section.
        num_circle_points: number of points to define circular cross-section.

    Returns:
        X, Y, Z: 2D arrays meshgrid of swept surface points.
    """
    # Parameter t along curve
    t = np.arange(len(x))

    # Tangent vectors via finite differences
    tangents = np.gradient(np.vstack([x, y, z]), axis=1).T
    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]

    # Arbitrary reference vector (not parallel to tangent)
    ref_vec = np.array([0, 0, 1])

    # Compute vectors normal to curve for cross-section circle planes
    normals = np.cross(tangents, ref_vec)
    norm_lengths = np.linalg.norm(normals, axis=1)
    zero_norms = norm_lengths < 1e-8

    # Avoid zero length norms by switching reference if needed
    normals[zero_norms] = np.cross(tangents[zero_norms], np.array([0,1,0]))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    binormals = np.cross(tangents, normals)

    # Circle parameter theta for cross-section points
    theta = np.linspace(0, 2*np.pi, num_circle_points)

    # Initialize mesh points
    X = np.zeros((len(t), num_circle_points))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)

    for i in range(len(t)):
        # Circle in normal-binormal plane
        circle_points = (
            radius * (np.outer(np.cos(theta), normals[i]) + np.outer(np.sin(theta), binormals[i]))
        )
        # Sweep circle center along trajectory curve point
        center = np.array([x[i], y[i], z[i]])[:, np.newaxis]  # Shape (3,1)
        points = center + circle_points.T  # Both (3,1) + (3,20) broadcast to (3,20)

        X[i, :] = points[0]
        Y[i, :] = points[1]
        Z[i, :] = points[2]

    return X, Y, Z


def plot_swept_surface(X, Y, Z, title="Swept Surface from Trajectory"):
    """
    Plot swept surface given mesh grid points.

    Args:
        X, Y, Z: 2D arrays of surface coordinates.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.show()