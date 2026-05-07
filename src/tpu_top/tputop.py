import os
import time
import asyncio
from typing import Dict, Any
import psutil

from textual.app import App, ComposeResult
from textual.widgets import Footer, Static
from textual.containers import Grid
from textual import work
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.table import Table

from tpu_top.state import MetricsHistory
from tpu_top.metrics import MetricsCollector, HAS_TPU_INFO

try:
    from jax._src.pallas.mosaic.tpu_info import get_tpu_info_for_chip, ChipVersion
    HAS_JAX_TPU_INFO = True
except ImportError:
    HAS_JAX_TPU_INFO = False
from tpu_top.ui import (
    make_device_table, make_process_table, 
    vertical_bar_chart, make_timeline
)

class TpuTopApp(App):
    """A Textual app that preserves the original UI/UX using Rich components."""

    TITLE = "TPU-TOP"
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit"), ("i", "toggle_info", "TPU Info"), ("escape", "default_view", "Default View")]

    CSS = """
    #header-container {
        height: 4;
        border: none;
    }
    #devices-container {
        height: auto;
        border: none;
    }
    #graphs-container {
        layout: grid;
        grid-size: 2;
        height: auto;
        border: none;
    }
    #graphs-container.row-layout {
        grid-size: 5;
        height: 7;
    }
    #duty-cycle-container {
        column-span: 2;
        height: 7;
        border: none;
    }
    #graphs-container.row-layout #duty-cycle-container {
        column-span: 1;
    }
    #processes-container {
        height: 1fr;
        border: none;
    }
    Static {
        padding: 0;
        margin: 0;
    }
    """

    def __init__(self, use_mock: bool = False):
        super().__init__()
        self.use_mock = use_mock
        self.collector = MetricsCollector(use_mock=use_mock)
        self.history = MetricsHistory()
        self.show_info = False

    def compose(self) -> ComposeResult:
        yield Static(id="header-container")
        yield Static(id="devices-container")
        
        with Grid(id="graphs-container"):
            yield Static(id="cpu-graph")
            yield Static(id="tpu-util-graph")
            yield Static(id="ram-graph")
            yield Static(id="tpu-mem-graph")
            yield Static(id="duty-cycle-container")
            
        yield Static(id="processes-container")
        yield Footer()

    def on_mount(self) -> None:
        # Set up versions for header
        try:
            import libtpu
            self.libtpu_version = libtpu.__version__
        except ImportError:
            self.libtpu_version = "--"
            
        try:
            import tpu_info
            self.tpu_info_version = getattr(tpu_info, "__version__", None)
            if not self.tpu_info_version:
                import importlib.metadata
                try:
                    self.tpu_info_version = importlib.metadata.version("tpu-info")
                except importlib.metadata.PackageNotFoundError:
                    self.tpu_info_version = "--"
        except ImportError:
            self.tpu_info_version = "--"
            
        try:
            import importlib.metadata
            self.tpu_top_version = importlib.metadata.version("tpu-top")
        except (importlib.metadata.PackageNotFoundError, ImportError):
            self.tpu_top_version = "--"

        self.tpu_topology = os.environ.get("TPU_ACCELERATOR_TYPE")
        if self.tpu_topology:
            self.tpu_topology = self.tpu_topology.upper()
        if not self.tpu_topology:
            if self.use_mock:
                self.tpu_topology = "TPU Mock"
            else:
                try:
                    from tpu_info import device
                    chip_type, _ = device.get_local_chips()
                    self.tpu_topology = f"TPU {chip_type.value.name}"
                except Exception:
                    self.tpu_topology = "TPU Unknown"

        self.collect_metrics_worker()

    def action_toggle_info(self) -> None:
        """Toggle the display of TPU info."""
        self.show_info = not self.show_info

    def action_default_view(self) -> None:
        """Return to the default view (hide info)."""
        self.show_info = False

    def get_tpu_info_table(self) -> Table | None:
        if not HAS_JAX_TPU_INFO:
            return None
        if not HAS_TPU_INFO:
            return None
            
        try:
            from tpu_info.device import get_local_chips, TpuChip
            
            chip_type, count = get_local_chips()
            if chip_type is None:
                return None
                
            mapping = {
                TpuChip.V2: ChipVersion.TPU_V2,
                TpuChip.V3: ChipVersion.TPU_V3,
                TpuChip.V4: ChipVersion.TPU_V4,
                TpuChip.V5E: ChipVersion.TPU_V5E,
                TpuChip.V5P: ChipVersion.TPU_V5P,
                TpuChip.V6E: ChipVersion.TPU_V6E,
                TpuChip.V7X: ChipVersion.TPU_7X,
            }
            
            jax_chip_version = mapping.get(chip_type)
            if jax_chip_version is None:
                 return None
                 
            num_cores = chip_type.value.devices_per_chip
            
            if (
                jax_chip_version in {
                    ChipVersion.TPU_V2,
                    ChipVersion.TPU_V3,
                    ChipVersion.TPU_7,
                    ChipVersion.TPU_7X,
                }
                or jax_chip_version.is_lite
            ):
                num_cores = 1
                
            info = get_tpu_info_for_chip(jax_chip_version, num_cores)
            
            table = Table(box=box.ROUNDED, expand=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            def fmt_bytes(b):
                if b == 0: return "0 B"
                for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
                    if b < 1024: return f"{b:.2f} {unit}"
                    b /= 1024
                return f"{b:.2f} PiB"

            def fmt_ops(o):
                if o == 0: return "0"
                for unit in ['', 'K', 'M', 'G', 'T', 'P']:
                    if o < 1000: return f"{o:.2f} {unit}Op/s"
                    o /= 1000
                return f"{o:.2f} EOp/s"
                
            table.add_row("Chip Version", str(info.chip_version))
            table.add_row("Generation", str(info.generation))
            table.add_row("Total Chips", str(count // chip_type.value.devices_per_chip))
            table.add_row("Physical Cores per Chip", str(info.chip_version.num_physical_tensor_cores_per_chip))
            table.add_row("Megacore Mode", "Yes" if info.is_megacore else "No")
            table.add_row("Supports Megacore", "Yes" if info.chip_version.supports_megacore else "No")
            table.add_row("Num Cores (Logical)", str(info.num_cores))
            table.add_row("Num Lanes (per Core)", str(info.num_lanes))
            table.add_row("Num Sublanes (per Core)", str(info.num_sublanes))
            table.add_row("MXU Column Size (per Core)", str(info.mxu_column_size))
            table.add_row("VMEM Capacity (per Core)", fmt_bytes(info.vmem_capacity_bytes))
            table.add_row("CMEM Capacity (per Core)", fmt_bytes(info.cmem_capacity_bytes))
            table.add_row("SMEM Capacity (per Core)", fmt_bytes(info.smem_capacity_bytes))
            table.add_row("HBM Capacity (per Core)", fmt_bytes(info.hbm_capacity_bytes))
            table.add_row("Memory Bandwidth (per Core)", f"{fmt_bytes(info.mem_bw_bytes_per_second)}/s")
            table.add_row("BF16 Ops (per Core)", fmt_ops(info.bf16_ops_per_second))
            table.add_row("INT8 Ops (per Core)", fmt_ops(info.int8_ops_per_second))
            table.add_row("FP8 Ops (per Core)", fmt_ops(info.fp8_ops_per_second))
            table.add_row("INT4 Ops (per Core)", fmt_ops(info.int4_ops_per_second))
            
            if info.sparse_core:
                table.add_row("Sparse Core", f"Cores: {info.sparse_core.num_cores}, Subcores: {info.sparse_core.num_subcores}, Lanes: {info.sparse_core.num_lanes}, DMA Granule: {info.sparse_core.dma_granule_size_bytes} B")
                
            return table
        except Exception as e:
            table = Table(box=box.ROUNDED, expand=True)
            table.add_column("Error", style="red")
            table.add_row(f"Failed to get TPU info: {e}")
            return table

    @work(thread=True)
    def collect_metrics_worker(self) -> None:
        while self.is_running:
            try:
                metrics_data = self.collector.collect_metrics()
                self.call_from_thread(self.update_ui, metrics_data)
            except Exception as e:
                pass
            time.sleep(0.5)

    def _update_graph(self, widget_id: str, panel_title: str, header_str: str, history_data: list, color: str, width: int, timeline: str):
        """Helper to update a graph panel to avoid repeated logic."""
        bars = vertical_bar_chart(history_data, width=width, height=3)
        text = Text(header_str, style=f"bold {color}")
        for line in bars:
            text.append(line + "\n", style=color)
        text.append(timeline, style=color)
        
        self.query_one(widget_id, Static).update(
            Panel(text, title=panel_title, box=box.ROUNDED, border_style=color)
        )

    def update_ui(self, metrics_data: Dict[str, Any]) -> None:
        # Update history
        self.history.append_cpu(metrics_data["cpu_usage"])
        self.history.append_ram(metrics_data["ram_usage"]["percent"])
        
        devices = metrics_data["devices"]
        num_devices = len(devices)
        avg_util = sum(d["tensorcore_util"] for d in devices) / len(devices) if devices else 0
        avg_duty_cycle = sum(d["duty_cycle"] for d in devices) / len(devices) if devices else 0
        avg_mem_pct = sum(d["memory_usage"] / d["total_memory"] * 100 for d in devices) / len(devices) if devices else 0
        total_hbm_gb = sum(d["total_memory"] for d in devices) / (1024**3) if devices else 0.0
        total_ram_gb = metrics_data['ram_usage']['total'] / (1024**3)
        
        self.history.append_tpu_util(avg_util)
        self.history.append_tpu_mem(avg_mem_pct)
        self.history.append_tpu_duty_cycle(avg_duty_cycle)

        # Dynamic Layout Logic based on height
        graphs_container = self.query_one("#graphs-container", Grid)
        if self.console.height < 55:
            graphs_container.add_class("row-layout")
            col_width = self.console.width // 5
        else:
            graphs_container.remove_class("row-layout")
            col_width = self.console.width // 2

        # Update Banner
        header_text = Text(f"TPU-TOP - TPU Utilization Monitor ({self.tpu_topology})", justify="center", style="bold green")
        header_text.append(f"\ntpu-top: {self.tpu_top_version} | libtpu: {self.libtpu_version} | tpu-info: {self.tpu_info_version}", style="dim")
        
        self.query_one("#header-container", Static).update(Panel(header_text, box=box.ROUNDED))

        # Update Devices Table
        dev_table = make_device_table(devices, metrics_data["hlo_map"], metrics_data["devices_per_chip"])
        self.query_one("#devices-container", Static).update(Panel(dev_table, title="Devices", box=box.ROUNDED))

        # Update Graphs
        graph_width = max(10, col_width - 6)
        timeline_str = make_timeline(graph_width)

        # CPU
        cpu_count = psutil.cpu_count() or 1
        self._update_graph("#cpu-graph", "AVG CPU Activity", f"CPU ({cpu_count} cores): {metrics_data['cpu_usage']:5.1f}%\n", self.history.cpu, "#4285F4", graph_width, timeline_str)
        
        # TPU Util
        self._update_graph("#tpu-util-graph", "AVG TPU (TC) UTL", f"UTL ({num_devices} Chips): {avg_util:5.1f}%\n", self.history.tpu_util, "#34A853", graph_width, timeline_str)
        
        # RAM
        self._update_graph("#ram-graph", "RAM Usage", f"RAM ({total_ram_gb:.1f} GB): {metrics_data['ram_usage']['percent']:5.1f}%\n", self.history.ram, "#EA4335", graph_width, timeline_str)
        
        # TPU Mem
        self._update_graph("#tpu-mem-graph", "AVG TPU Mem", f"HBM ({total_hbm_gb:.1f} GB): {avg_mem_pct:5.1f}%\n", self.history.tpu_mem, "#FBBC05", graph_width, timeline_str)

        # Duty Cycle
        if self.console.height < 55:
            dc_graph_width = graph_width
            dc_timeline_str = timeline_str
        else:
            dc_graph_width = max(10, self.console.width - 6)
            dc_timeline_str = make_timeline(dc_graph_width)
            
        self._update_graph("#duty-cycle-container", "AVG TPU DUTY CYCLE", f"DC ({num_devices} Chips): {avg_duty_cycle:5.1f}%\n", self.history.tpu_duty_cycle, "#E066FF", dc_graph_width, dc_timeline_str)

        # Update Processes Table
        current_pid = os.getpid()
        processes = [p for p in metrics_data["processes"] if p["pid"] != current_pid]
        processes = [p for p in processes if "tpu-top" not in p["name"] and "tputop" not in p["name"]]
        
        tpu_procs = [p for p in processes if "TPU" in p["device"]]
        cpu_procs = [p for p in processes if p["device"] == "CPU"]
        
        tpu_procs.sort(key=lambda x: (int(x["device"].split()[1]), -x["memory"]))
        cpu_procs.sort(key=lambda x: x["memory"], reverse=True)
        
        processes = tpu_procs + cpu_procs

        if self.show_info:
            info_table = self.get_tpu_info_table()
            if info_table:
                self.query_one("#processes-container", Static).update(Panel(info_table, title="TPU Info", box=box.ROUNDED))
            else:
                self.query_one("#processes-container", Static).update(Panel(Text("TPU Info not available", style="red"), title="TPU Info", box=box.ROUNDED))
        else:
            proc_table = make_process_table(processes)
            self.query_one("#processes-container", Static).update(Panel(proc_table, title="Processes", box=box.ROUNDED))

def main():
    use_mock = not HAS_TPU_INFO or os.environ.get("TPU_TOP_MOCK") == "1"
    app = TpuTopApp(use_mock=use_mock)
    app.run()

if __name__ == "__main__":
    main()
