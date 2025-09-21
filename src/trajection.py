import numpy as np

def create_trajectory_from_signal(signal, phase=None):
    x = np.arange(len(signal))
    y = (signal - np.min(signal)) / (np.ptp(signal) + 1e-12)
    z = phase if phase is not None else np.zeros_like(signal)
    return x, y, z

def create_trajectory_from_complex(coefficients):
    mag = np.abs(coefficients)
    phase = np.angle(coefficients)
    return create_trajectory_from_signal(mag, phase)

def stack_points(x, y, z):
    return np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
