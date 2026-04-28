import os
import time
import asyncio
from typing import Dict, Any

from textual.app import App, ComposeResult
from textual.widgets import Footer, Static
from textual.containers import Grid
from textual import work
from rich.panel import Panel
from rich import box
from rich.text import Text

from tpu_top.state import MetricsHistory
from tpu_top.metrics import MetricsCollector, HAS_TPU_INFO
from tpu_top.ui import (
    make_device_table, make_process_table, 
    vertical_bar_chart, make_timeline
)

class TpuTopApp(App):
    """A Textual app that preserves the original UI/UX using Rich components."""

    TITLE = "TPU-TOP"
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

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
            self.tpu_top_version = "0.1.7"

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

    @work(thread=True)
    async def collect_metrics_worker(self) -> None:
        while True:
            try:
                metrics_data = self.collector.collect_metrics()
                self.call_from_thread(self.update_ui, metrics_data)
            except Exception as e:
                pass
            await asyncio.sleep(0.5)

    def update_ui(self, metrics_data: Dict[str, Any]) -> None:
        # Update history
        self.history.append_cpu(metrics_data["cpu_usage"])
        self.history.append_ram(metrics_data["ram_usage"]["percent"])
        
        devices = metrics_data["devices"]
        avg_util = sum(d["tensorcore_util"] for d in devices) / len(devices) if devices else 0
        avg_duty_cycle = sum(d["duty_cycle"] for d in devices) / len(devices) if devices else 0
        avg_mem_pct = sum(d["memory_usage"] / d["total_memory"] * 100 for d in devices) / len(devices) if devices else 0
        
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
        cpu_bars = vertical_bar_chart(self.history.cpu, width=graph_width, height=3)
        cpu_text = Text(f"CPU: {metrics_data['cpu_usage']:5.1f}%\n", style="bold #4285F4")
        for line in cpu_bars:
            cpu_text.append(line + "\n", style="#4285F4")
        cpu_text.append(timeline_str, style="#4285F4")
        self.query_one("#cpu-graph", Static).update(Panel(cpu_text, title="CPU Activity", box=box.ROUNDED, border_style="#4285F4"))

        # TPU Util
        util_bars = vertical_bar_chart(self.history.tpu_util, width=graph_width, height=3)
        util_text = Text(f"UTL: {avg_util:5.1f}%\n", style="bold #34A853")
        for line in util_bars:
            util_text.append(line + "\n", style="#34A853")
        util_text.append(timeline_str, style="#34A853")
        self.query_one("#tpu-util-graph", Static).update(Panel(util_text, title="AVG TPU (TC) UTL", box=box.ROUNDED, border_style="#34A853"))

        # RAM
        ram_bars = vertical_bar_chart(self.history.ram, width=graph_width, height=3)
        ram_text = Text(f"RAM: {metrics_data['ram_usage']['percent']:5.1f}%\n", style="bold #EA4335")
        for line in ram_bars:
            ram_text.append(line + "\n", style="#EA4335")
        ram_text.append(timeline_str, style="#EA4335")
        self.query_one("#ram-graph", Static).update(Panel(ram_text, title="RAM Activity", box=box.ROUNDED, border_style="#EA4335"))

        # TPU Mem
        mem_bars = vertical_bar_chart(self.history.tpu_mem, width=graph_width, height=3)
        mem_text = Text(f"HBM: {avg_mem_pct:5.1f}%\n", style="bold #FBBC05")
        for line in mem_bars:
            mem_text.append(line + "\n", style="#FBBC05")
        mem_text.append(timeline_str, style="#FBBC05")
        self.query_one("#tpu-mem-graph", Static).update(Panel(mem_text, title="AVG TPU Mem", box=box.ROUNDED, border_style="#FBBC05"))

        # Duty Cycle
        if self.console.height < 55:
            dc_graph_width = graph_width
            dc_timeline_str = timeline_str
        else:
            dc_graph_width = max(10, self.console.width - 6)
            dc_timeline_str = make_timeline(dc_graph_width)
            
        dc_bars = vertical_bar_chart(self.history.tpu_duty_cycle, width=dc_graph_width, height=3)
        dc_text = Text(f"DC: {avg_duty_cycle:5.1f}%\n", style="bold #E066FF")
        for line in dc_bars:
            dc_text.append(line + "\n", style="#E066FF")
        dc_text.append(dc_timeline_str, style="#E066FF")
        self.query_one("#duty-cycle-container", Static).update(Panel(dc_text, title="AVG TPU DUTY CYCLE", box=box.ROUNDED, border_style="#E066FF"))

        # Update Processes Table
        current_pid = os.getpid()
        processes = [p for p in metrics_data["processes"] if p["pid"] != current_pid]
        processes = [p for p in processes if "tpu-top" not in p["name"] and "tputop" not in p["name"]]
        
        tpu_procs = [p for p in processes if "TPU" in p["device"]]
        cpu_procs = [p for p in processes if p["device"] == "CPU"]
        
        tpu_procs.sort(key=lambda x: (int(x["device"].split()[1]), -x["memory"]))
        cpu_procs.sort(key=lambda x: x["memory"], reverse=True)
        
        processes = tpu_procs + cpu_procs

        proc_table = make_process_table(processes)
        self.query_one("#processes-container", Static).update(Panel(proc_table, title="Processes", box=box.ROUNDED))

def main():
    use_mock = not HAS_TPU_INFO or os.environ.get("TPU_TOP_MOCK") == "1"
    app = TpuTopApp(use_mock=use_mock)
    app.run()

if __name__ == "__main__":
    main()
