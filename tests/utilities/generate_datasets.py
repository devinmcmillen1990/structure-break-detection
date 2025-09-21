import numpy as np
from tests.utilities.save_as_csv import save_as_csv

def generate_clean_constant_dataset(length):
    constant_array = np.full(length, 7.0)
    file_path = save_as_csv(constant_array, "01_clean_constant_dataset.csv")
    print(f"Saved test dataset {file_path}")

def generate_clean_linear_dataset(length):
    linear_array = np.linspace(0, length, length)
    file_path = save_as_csv(linear_array, "02_clean_linear_dataset.csv")
    print(f"Saved test dataset {file_path}")

if __name__ == "__main__":
    generate_clean_constant_dataset(100)
    generate_clean_linear_dataset(100)