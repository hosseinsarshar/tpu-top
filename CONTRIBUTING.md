# Contributing to TPU-TOP

Thank you for your interest in contributing to `tpu-top`! This document provides guidelines and instructions for setting up your development environment and extending the project.

## Development Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/hosseinsarshar/tpu-top.git
    cd tpu-top
    ```

2.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```
    *(Note: If you are on a Google Corp machine, you may need to use specific internal instructions for installing dependencies if standard pip fails).*

## Code Structure

The project follows a standard PyPI layout:
*   `src/tpu_top/tputop.py`: This is the core of the application. It contains the TUI layout definition (using `rich`), the metrics gathering loop, and the rendering logic.
*   `tests/`: Contains unit tests.

## Extending the Dashboard

### Adding New Metrics
If you want to add new metrics (e.g., HLO queue size or transfer latencies as discussed):
1.  Locate the metrics gathering section in the `main()` function of `tputop.py`.
2.  Use `tpu_info` (or fallbacks to `libtpu`) to fetch the new data.
3.  Add a new panel in `make_layout()` or add columns to `make_device_table()`.
4.  Pass the new data to the rendering functions.

### Mock Mode
For development on machines without access to physical TPUs, you can run in mock mode:
```bash
TPU_TOP_MOCK=1 tpu-top
```
You can extend the mock data generators (`get_mock_metrics`) in `tputop.py` to simulate the new metrics you are adding.

## Running Tests

To validate changes, run the unit tests:
```bash
python -m unittest tests/test_main.py
```

## Building and Publishing

To build and push a new version to PyPI:

1.  Bump the version in `pyproject.toml`.
2.  Clean old builds: `rm -rf dist/ build/ src/tpu_top.egg-info/`
3.  Build the package:
    ```bash
    python3 -m build
    ```
    *If you are on a Google Corp machine and see index errors, use:*
    ```bash
    python3 -m build --no-isolation
    ```
4.  Upload to PyPI:
    ```bash
    python3 -m twine upload dist/*
    ```
