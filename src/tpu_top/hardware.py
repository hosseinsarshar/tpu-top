"""TPU hardware specs, read from JAX's Pallas TPU info tables.

``jax`` is imported lazily: it is slow to import and only needed when the
hardware info screen is opened.
"""

from __future__ import annotations

from typing import List, Tuple

Section = Tuple[str, List[Tuple[str, str]]]

# tpu_info.device.TpuChip name -> jax ChipVersion name
_CHIP_VERSIONS = (
    ("V2", "TPU_V2"),
    ("V3", "TPU_V3"),
    ("V4", "TPU_V4"),
    ("V5E", "TPU_V5E"),
    ("V5P", "TPU_V5P"),
    ("V6E", "TPU_V6E"),
    ("V7X", "TPU_7X"),
)


class HardwareInfoUnavailable(Exception):
    """Raised when the hardware specs can't be read on this machine."""


def fmt_bytes(b: float) -> str:
    if b == 0:
        return "0 B"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PiB"


def fmt_ops(o: float) -> str:
    if o == 0:
        return "0"
    for unit in ["", "K", "M", "G", "T", "P"]:
        if o < 1000:
            return f"{o:.2f} {unit}Op/s"
        o /= 1000
    return f"{o:.2f} EOp/s"


def _yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


def collect_hardware_info() -> List[Section]:
    """Return the hardware specs of the local TPU as titled sections.

    Raises:
        HardwareInfoUnavailable: if ``jax`` or ``tpu-info`` is missing, or no
            supported TPU chip is found.
    """
    try:
        from jax._src.pallas.mosaic.tpu_info import ChipVersion, get_tpu_info_for_chip
    except ImportError as e:
        raise HardwareInfoUnavailable("Install jax[tpu] to see hardware details.") from e
    try:
        from tpu_info.device import TpuChip, get_local_chips
    except ImportError as e:
        raise HardwareInfoUnavailable("Install tpu-info to see hardware details.") from e

    chip_type, count = get_local_chips()
    if chip_type is None:
        raise HardwareInfoUnavailable("No TPU chips found on this machine.")

    mapping = {
        getattr(TpuChip, chip): getattr(ChipVersion, version)
        for chip, version in _CHIP_VERSIONS
        if hasattr(TpuChip, chip) and hasattr(ChipVersion, version)
    }
    jax_chip_version = mapping.get(chip_type)
    if jax_chip_version is None:
        raise HardwareInfoUnavailable(f"No hardware table for {chip_type.value.name}.")

    num_cores = chip_type.value.devices_per_chip
    single_core = {
        getattr(ChipVersion, name)
        for name in ("TPU_V2", "TPU_V3", "TPU_7", "TPU_7X")
        if hasattr(ChipVersion, name)
    }
    if jax_chip_version in single_core or jax_chip_version.is_lite:
        num_cores = 1

    info = get_tpu_info_for_chip(jax_chip_version, num_cores)

    sections: List[Section] = [
        (
            "Chip",
            [
                ("Chip Version", str(info.chip_version)),
                ("Generation", str(info.generation)),
                ("Total Chips", str(count // chip_type.value.devices_per_chip)),
                ("Physical Cores per Chip", str(info.chip_version.num_physical_tensor_cores_per_chip)),
                ("Megacore Mode", _yes_no(info.is_megacore)),
                ("Supports Megacore", _yes_no(info.chip_version.supports_megacore)),
                ("Num Cores (Logical)", str(info.num_cores)),
            ],
        ),
        (
            "Core layout",
            [
                ("Num Lanes (per Core)", str(info.num_lanes)),
                ("Num Sublanes (per Core)", str(info.num_sublanes)),
                ("MXU Column Size (per Core)", str(info.mxu_column_size)),
            ],
        ),
        (
            "Memory (per core)",
            [
                ("VMEM Capacity", fmt_bytes(info.vmem_capacity_bytes)),
                ("CMEM Capacity", fmt_bytes(info.cmem_capacity_bytes)),
                ("SMEM Capacity", fmt_bytes(info.smem_capacity_bytes)),
                ("HBM Capacity", fmt_bytes(info.hbm_capacity_bytes)),
                ("Memory Bandwidth", f"{fmt_bytes(info.mem_bw_bytes_per_second)}/s"),
            ],
        ),
        (
            "Peak compute (per core)",
            [
                ("BF16", fmt_ops(info.bf16_ops_per_second)),
                ("INT8", fmt_ops(info.int8_ops_per_second)),
                ("FP8", fmt_ops(info.fp8_ops_per_second)),
                ("INT4", fmt_ops(info.int4_ops_per_second)),
            ],
        ),
    ]
    if info.sparse_core:
        sc = info.sparse_core
        sections.append(
            (
                "SparseCore",
                [
                    ("Cores", str(sc.num_cores)),
                    ("Subcores", str(sc.num_subcores)),
                    ("Lanes", str(sc.num_lanes)),
                    ("DMA Granule", f"{sc.dma_granule_size_bytes} B"),
                ],
            )
        )
    return sections
