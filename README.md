# TPU-TOP

A simple terminal-based monitoring dashboard for Google Cloud TPUs, designed to give you real-time visibility into your machine's performance both on the host and the device.

> [!NOTE]
> This tool was inspired by the [nvitop](https://github.com/XuehaiPan/nvitop) project for GPUs. This is a community project and not an official Google product.

[![PyPI version](https://img.shields.io/pypi/v/tpu-top.svg)](https://pypi.org/project/tpu-top/)
[GitHub Repository](https://github.com/hosseinsarshar/tpu-top) | [PyPI Project](https://pypi.org/project/tpu-top/)

## Project Overview

`tpu-top` provides a visual, TUI (Terminal User Interface) to monitor system and TPU resources. It is tailored to run it directly on a TPU instance either on a GCE VM or a GKE Pod.

![tpu-top UI](https://raw.githubusercontent.com/hosseinsarshar/tpu-top/main/images/image.png)



### What You Can See

*   **TPU Memory & Utilization**: Real-time memory usage, TensorCore utilization, and raw duty cycle for each TPU device.
*   **History Graphs**: Visual graphs with timeline markers showing the history of CPU (with core count), RAM (with GiB usage), and TPU usage.
*   **Duty Cycle History**: A dedicated panel showing the history of TPU duty cycle.
*   **PIDs per TPU**: A dedicated process list showing which PIDs are utilizing specific TPU devices, including their host RAM and CPU impact.
*   **Active HLO Ops**: Current HLO operations executing on each TPU core (Tensor Cores and Sparse Cores).

## Calculations Explained

### Duty Cycle
**Duty Cycle** represents the percentage of time the TPU is "busy" (not idle) during a given sampling window. 

Performance Insights:
- High Duty Cycle (e.g., >90%): The TPU is constantly running kernels and is not waiting on the host.
- Low Duty Cycle (e.g., <30%): This is often a sign of "data starvation." The TPU is idle because it is waiting for the CPU to provide input data.

### TensorCore Utilization
TensorCore Utilization measures the computational intensity of the workload. It tracks what percentage of the TPU's peak theoretical matrix-multiplication capacity is actually being used while the chip is active.

Performance Insights:
- Low TensorCore Utilization: If your Duty Cycle is high but your TensorCore Utilization is low, your TPU is "busy," but it isn't doing much math. This often occurs when:
  - Batch sizes are too small to saturate the hardware.
  - The model is limited by memory bandwidth rather than compute.
  - The code spends a lot of time on non-matrix operations (e.g., scalar transposes).


### How to use them together
- **Low Duty Cycle + Low TensorCore Util:** Your TPU is mostly idle, likely waiting for data from the CPU.
- **High Duty Cycle + Low TensorCore Util:** Your TPU is constantly working, but the specific operations (kernels) you are running are not computationally dense (likely memory-bound or using small batch sizes).
- **High Duty Cycle + High TensorCore Util:** Ideal performance; you are keeping the TPU busy and fully utilizing its matrix-multiplication hardware.


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

