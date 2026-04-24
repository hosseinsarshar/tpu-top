# TPU-TOP

A modern, terminal-based monitoring dashboard for Google Cloud TPUs, designed to give you real-time visibility into your machine's performance.

> [!NOTE]
> This tool was inspired by the [nvitop](https://github.com/XuehaiPan/nvitop) project for GPUs. This is a community project and not an official Google product.

[![PyPI version](https://img.shields.io/pypi/v/tpu-top.svg)](https://pypi.org/project/tpu-top/)
[GitHub Repository](https://github.com/hosseinsarshar/tpu-top) | [PyPI Project](https://pypi.org/project/tpu-top/)

## Project Overview

`tpu-top` provides a visual, interactive TUI (Terminal User Interface) to monitor system and TPU resources. It is specifically tailored for high-performance computing environments like GKE (Google Kubernetes Engine) where deep learning models are trained on TPUs.

![tpu-top UI](https://raw.githubusercontent.com/hosseinsarshar/tpu-top/main/images/image.png)

### What You Can See

*   **TPU Memory & Utilization**: Real-time memory usage, TensorCore utilization, and raw duty cycle for each TPU device.
*   **History Graphs**: Visual graphs with timeline markers showing the history of CPU (with core count), RAM (with GiB usage), and TPU usage.
*   **Duty Cycle History**: A dedicated panel showing the history of TPU duty cycle.
*   **PIDs per TPU**: A dedicated process list showing which PIDs are utilizing specific TPU devices, including their host RAM and CPU impact.
*   **Active HLO Ops**: Current HLO operations executing on each TPU core (Tensor Cores and Sparse Cores).

## Calculations Explained

### TensorCore Utilization
This metric measures the percentage of time the Tensor Cores on the TPU chip were actively executing a program. It is read from the `libtpu` library (if available). If `libtpu` metrics are not available, it falls back to reporting the raw duty cycle.

### Duty Cycle
This metric measures the active execution time of the TPU chip as reported by the TPU driver via the `tpu-info` library. It represents the overall proportion of time the accelerator was busy, regardless of whether it was executing massive matrix multiplications or standard operations.

## Installation

### From PyPI (Recommended)

```bash
pip install tpu-top
```

### From Source

You can also install `tpu-top` directly from the source directory.

### Prerequisites

Ensure you have Python 3.10+ and access to a Cloud TPU environment. The tool relies on `tpu-info` to communicate with the TPU driver.

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

