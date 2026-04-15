# TPU-TOP

A modern, terminal-based monitoring dashboard for Google Cloud TPUs, designed to give you real-time visibility into your machine's performance.

[![PyPI version](https://img.shields.io/pypi/v/tpu-top.svg)](https://pypi.org/project/tpu-top/)
[GitHub Repository](https://github.com/hosseinsarshar/tpu-top) | [PyPI Project](https://pypi.org/project/tpu-top/)

## Project Overview

`tpu-top` provides a visual, interactive TUI (Terminal User Interface) to monitor system and TPU resources. It is specifically tailored for high-performance computing environments like GKE (Google Kubernetes Engine) where deep learning models are trained on TPUs.

![tpu-top UI](https://raw.githubusercontent.com/hosseinsarshar/tpu-top/main/images/image.png)

### Key Features

*   **Responsive Layout**: Automatically switches between 2x2 grid and 1x4 list for graphs based on terminal height.
*   **Priority-Based Rendering**: Dynamically shrinks or drops panels (Processes panel shrinks first) to fit severe vertical constraints.
*   **Device Status Table**: Detailed per-device breakdown showing duty cycle and memory usage. Compact 1-line layout for single-core TPUs (v6e) and multi-line for multi-core (v7).
*   **Process Monitor**: Lists active processes, prioritized by TPU processes (sorted by TPU index) and then CPU processes (sorted by memory usage).
*   **Google Brand Styling**: Color-coded graphs using Google's signature color palette (Blue, Red, Yellow, Green).
*   **Version Awareness**: Displays `tpu-top`, `libtpu`, and `tpu-info` versions directly in the header alongside detected TPU generation.

## Installation

### From PyPI (Recommended)

```bash
pip install tpu-top
```

### From Source

You can also install `tpu-top` directly from the source directory.

### Prerequisites

Ensure you have Python 3.12+ and access to a Cloud TPU environment. The tool relies on `tpu-info` to communicate with the TPU driver.

### Standard Source Install

Navigate to the project root directory and run:

```bash
pip install .
```

### Developer Install

If you are making modifications and want them to reflect immediately:

```bash
pip install -e .
```

## How to Use

Once installed, you can launch the dashboard from anywhere in your terminal:

```bash
tpu-top
```

### Environment Variables

*   `TPU_TOP_MOCK=1`: Force mock mode for testing on machines without physical TPUs.
*   `TPU_TOP_ITERATIONS=N`: Run for exactly `N` refresh cycles and exit (useful for automated tests).

## Running Tests

To validate changes, run the unit tests:

```bash
python -m unittest test_main.py
```
(Note: If testing inside a GKE container, ensure dependencies are installed in your target environment).

