# TPU-TOP

A modern, terminal-based monitoring dashboard for Google Cloud TPUs, designed to give you real-time visibility into your machine's performance.

[![PyPI version](https://img.shields.io/pypi/v/tpu-top.svg)](https://pypi.org/project/tpu-top/)
[GitHub Repository](https://github.com/hosseinsarshar/tpu-top) | [PyPI Project](https://pypi.org/project/tpu-top/)

## Project Overview

`tpu-top` provides a visual, interactive TUI (Terminal User Interface) to monitor system and TPU resources. It is specifically tailored for high-performance computing environments like GKE (Google Kubernetes Engine) where deep learning models are trained on TPUs.

![tpu-top UI](https://raw.githubusercontent.com/hosseinsarshar/tpu-top/main/images/image.png)

### What You Can See

*   **TPU Memory & Utilization**: Real-time memory usage and duty cycle for each TPU device.
*   **History Timebars**: Visual graphs with timeline markers showing the history of CPU, RAM, and TPU usage.
*   **PIDs per TPU**: A dedicated process list showing which PIDs are utilizing specific TPU devices.
*   **Active HLO Ops**: Current HLO operations executing on each TPU core.

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

## Running Tests

To validate changes, run the unit tests:

```bash
python -m unittest test_main.py
```
(Note: If testing inside a GKE container, ensure dependencies are installed in your target environment).

