# TPU-TOP

A terminal dashboard for Google Cloud TPUs. It gives you real-time visibility into your machine's performance, both on the host and on the TPU chips.

> [!NOTE]
> This tool was inspired by the [nvitop](https://github.com/XuehaiPan/nvitop) project for GPUs. This is a community project and not an official Google product.

[![PyPI version](https://img.shields.io/pypi/v/tpu-top.svg)](https://pypi.org/project/tpu-top/)
[![Python versions](https://img.shields.io/pypi/pyversions/tpu-top.svg)](https://pypi.org/project/tpu-top/)
[GitHub Repository](https://github.com/hosseinsarshar/tpu-top) | [PyPI Project](https://pypi.org/project/tpu-top/)

![tpu-top UI](https://raw.githubusercontent.com/hosseinsarshar/tpu-top/refs/heads/main/images/image.png)

## Project Overview

`tpu-top` is a TUI (Terminal User Interface) to monitor system and TPU resources. It is made to run directly on a TPU host, either a GCE TPU VM or a GKE Pod.

### What You Can See

*   **Per-chip gauges**: HBM usage, TensorCore utilization and duty cycle for every chip, with a short trend line under each gauge.
*   **Active HLO ops**: The HLO op and computation each core is running right now (TensorCores and SparseCores).
*   **History graphs**: TensorCore utilization, duty cycle and HBM (averaged over all chips), plus host CPU and RAM. Graphs use braille dots for a sharper, longer history; press `g` for solid blocks.
*   **Processes**: Which PIDs hold the TPUs (one row per process, with all of its chips, e.g. `TPU 0-3`), plus processes that are busy on the CPU. Shows user, CPU, memory and the full command. Sortable.
*   **TPU hardware info**: VMEM, HBM, memory bandwidth, peak ops, SparseCore details and more. Press `i`.
*   **Themes**: Open the command palette with `ctrl+p` and pick "Change theme". The metric colors stay the same in every theme.
*   **Adaptive layout**: The screen rearranges itself for small or large terminals.

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

Press `?` in the app for a short version of this guide.

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

### Options

| Option | Description |
| --- | --- |
| `-n`, `--interval SECONDS` | Time between updates (default: `0.5`). |
| `--mock` | Show simulated data, so you can try it without a TPU. Same as `TPU_TOP_MOCK=1`. |
| `-V`, `--version` | Print the version and exit. |

If `tpu-info` is not installed, `tpu-top` starts with mock data and shows a **MOCK DATA** badge in the header.

### Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `i` | Show or hide the TPU hardware info. |
| `p` or `space` | Pause or resume updates. |
| `s` | Change the process sort order: device, CPU, memory, PID. You can also click a column header. |
| `g` | Switch the graph style: braille dots or solid blocks. |
| `↑` `↓` `PgUp` `PgDn` | Scroll the process list. |
| `ctrl+p` | Open the command palette (for example, to change the theme). |
| `?` | Show the help. |
| `escape` | Close the open dialog and return to the default view. |
| `q` or `ctrl+c` | Quit the application. |

## Running Tests

To validate changes, run the unit tests:

```bash
python -m unittest discover tests
```
(Note: If testing inside a GKE container, ensure dependencies are installed in your target environment).
