import os
import numpy as np
import csv
import pytest

from src.shape_generators.discrete_fourier_transform import compute_dft

def load_csv_as_array(filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        return np.array([float(row[0]) for row in reader])

def test_dft_constant_signal():
    dataset_path = os.path.join("tests", ".datasets", "01_clean_constant_dataset.csv")
    time_series = load_csv_as_array(dataset_path)
    frequencies, dft_coeffs = compute_dft(time_series, sampling_rate=1.0)

    dc_component = dft_coeffs[0]
    other_components = dft_coeffs[1:]

    # DC component magnitude should be sum of constant values
    assert np.isclose(np.abs(dc_component), np.sum(time_series))

    # All other frequency magnitudes should be close to zero
    assert np.allclose(np.abs(other_components), 0, atol=1e-10)

def test_dft_linear_signal():
    dataset_path = os.path.join("tests", ".datasets", "02_clean_linear_dataset.csv")
    time_series = load_csv_as_array(dataset_path)
    frequencies, dft_coeffs = compute_dft(time_series, sampling_rate=1.0)

    # DC component magnitude should approx equal mean * length 
    expected_dc = np.mean(time_series) * len(time_series)
    dc_component = dft_coeffs[0]
    assert np.isclose(np.abs(dc_component), expected_dc, rtol=1e-2)

    # Energy concentrated in low frequencies: sum of magnitudes mostly in first half
    magnitudes = np.abs(dft_coeffs)
    low_freq_energy = np.sum(magnitudes[:len(magnitudes)//2])
    total_energy = np.sum(magnitudes)
    assert low_freq_energy / total_energy > 0.60
