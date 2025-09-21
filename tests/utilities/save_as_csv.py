import os
import csv

def get_datasets_dir():
    """
    Returns the absolute path to the sibling '.datasets' directory of the 'tests' folder,
    assuming this script lives inside 'tests/.utilities/'.
    Creates the directory if it does not exist.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # e.g., tests/.utilities
    tests_dir = os.path.abspath(os.path.join(current_dir, ".."))  # tests/
    datasets_dir = os.path.join(tests_dir, ".datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    return datasets_dir

def save_as_csv(array, filename):
    """
    Saves a 1D array as CSV under the datasets directory with the given filename.
    """
    directory = get_datasets_dir()
    file_path = os.path.join(directory, filename)
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        for val in array:
            writer.writerow([val])
    return file_path
