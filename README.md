# structure-break-detection
## Run Commands
    1. Fetch Python requirements: ```pip install -r requirements.txt```
    2. Run Shape Overlay
        - ```python ./overlay_signals.py```
        - ```python ./overlay_trajectories.py```
        - ```python ./overlay_trajectory_polar_projections.py```
    3. Generate unit test files '''python -m tests.utilities.generate_datasets'''
        - Will generate required test files under ```tests/.datasets/``` in CSV format
    4. Run unit test:
        - '''pytest tests/test_shape_generators_discrete_fourier_transform.py'''