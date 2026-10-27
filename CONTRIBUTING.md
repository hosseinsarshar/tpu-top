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
*   `src/tpu_top/tputop.py`: The Textual app. It lays out the screen, runs the metrics loop in a background thread, and handles key bindings and the command-line flags.
*   `src/tpu_top/widgets.py`: Textual widgets: the header bar, history graphs, the process table and the help / hardware info dialogs.
*   `src/tpu_top/ui.py`: Rich renderables and helpers: gauges, braille graphs, the time axis and the devices table.
*   `src/tpu_top/theme.py`: The color palette. Metric colors stay fixed; neutral colors follow the active Textual theme.
*   `src/tpu_top/metrics.py`: Collects host, TPU and process metrics (and the mock data).
*   `src/tpu_top/state.py`: Keeps the metric history used by the graphs.
*   `src/tpu_top/hardware.py`: Reads the TPU hardware specs shown with `i`.
*   `tests/`: Contains unit tests.

## Extending the Dashboard

### Adding New Metrics
If you want to add new metrics (e.g., HLO queue size or transfer latencies as discussed):
1.  Collect the value in `MetricsCollector.collect_metrics()` in `metrics.py`, using `tpu_info` (or fallbacks to `libtpu`).
2.  If it needs a history graph, add a series to `MetricsHistory` in `state.py` and append to it in `TpuTopApp.update_ui()`.
3.  Show it: add a column to `make_device_table()` in `ui.py`, or a new `MetricGraph` in `TpuTopApp.compose()`.
4.  Pick its color in `theme.py` so it looks the same everywhere it shows up.

### Mock Mode
For development on machines without access to physical TPUs, you can run in mock mode:
```bash
tpu-top --mock        # or: TPU_TOP_MOCK=1 tpu-top
```
You can extend the mock data generator (`_MockTPU`) in `metrics.py` to simulate the new metrics you are adding.

## Running Tests

To validate changes, run the unit tests:
```bash
python -m unittest discover tests
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
