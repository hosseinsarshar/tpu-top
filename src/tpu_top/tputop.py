import argparse
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
from rich.console import Group
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Footer, Static
from textual.worker import get_current_worker

from tpu_top.hardware import HardwareInfoUnavailable, collect_hardware_info
from tpu_top.metrics import MetricsCollector, HAS_TPU_INFO
from tpu_top.state import MetricsHistory
from tpu_top.theme import THEME_NAME, TPU_TOP_THEME, Palette, build_palette
from tpu_top.ui import GIB, build_process_rows, make_device_table, make_info_sheet
from tpu_top.widgets import (
    HeaderBar,
    HelpScreen,
    InfoScreen,
    MetricGraph,
    ProcessTable,
    make_help,
)

# Rows the process list should keep before the graphs are stacked on two rows.
PROCESSES_MIN_ROWS = 12


def _package_version(name: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return "--"


class MetricsReady(Message):
    """A new metrics sample, posted by the collector thread."""

    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__()
        self.data = data


class MetricsFailed(Message):
    """Collecting metrics failed, posted by the collector thread."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class HardwareInfoReady(Message):
    """Hardware specs (or the reason they are unavailable)."""

    def __init__(self, sections: Optional[list], error: str = "") -> None:
        super().__init__()
        self.sections = sections
        self.error = error


class TpuTopApp(App):
    """A live dashboard for the TPUs and host of a Cloud TPU VM or GKE pod."""

    TITLE = "tpu-top"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("i", "toggle_info", "TPU info"),
        Binding("p", "toggle_pause", "Pause"),
        Binding("space", "toggle_pause", "Pause", show=False),
        Binding("s", "cycle_sort", "Sort"),
        Binding("g", "toggle_graph_style", "Graph style"),
        Binding("t", "change_theme", "Theme"),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "default_view", "Default view", show=False),
    ]

    CSS = """
    #devices {
        height: auto;
        border: round $panel-lighten-2;
        border-title-align: left;
        border-subtitle-align: right;
        padding: 0 1;
    }
    #graphs {
        height: auto;
    }
    #graphs Horizontal {
        height: 7;
    }
    #graphs MetricGraph {
        width: 1fr;
    }
    #graphs.row {
        layout: horizontal;
        height: 7;
    }
    #graphs.row.short, #graphs.row.short Horizontal {
        height: 6;
    }
    #graphs.row #tpu-graphs {
        width: 3fr;
    }
    #graphs.row #host-graphs {
        width: 2fr;
    }
    #processes {
        height: 1fr;
        min-height: 6;
        border: round $panel-lighten-2;
        border-title-align: left;
        border-subtitle-align: right;
    }
    ProcessTable:focus {
        background-tint: $foreground 0%;
    }
    ProcessTable > .datatable--header,
    ProcessTable:focus > .datatable--header {
        background: $background;
        background-tint: $foreground 0%;
        color: $text-muted;
        text-style: bold;
    }
    ProcessTable > .datatable--even-row {
        background: $foreground 3%;
    }
    ProcessTable > .datatable--cursor {
        background: $primary 10%;
        color: $foreground;
        text-style: none;
    }
    ProcessTable:focus > .datatable--cursor {
        background: $primary 22%;
        color: $foreground;
        text-style: none;
    }
    ProcessTable > .datatable--hover {
        background: $primary 8%;
    }
    """

    def __init__(
        self,
        use_mock: bool = False,
        interval: float = 0.5,
        mock_requested: bool = True,
        initial_theme: str = THEME_NAME,
    ):
        super().__init__()
        self.use_mock = use_mock
        self.mock_requested = mock_requested
        self.interval = interval
        self.initial_theme = initial_theme or THEME_NAME
        self.collector = MetricsCollector(use_mock=use_mock)
        self.history = MetricsHistory()
        self.palette: Palette = build_palette()
        self.paused = False
        self.graph_style = "braille"
        self.compact = False
        self.last_data: Optional[Dict[str, Any]] = None
        self.last_error = ""
        self._hardware_info: Optional[HardwareInfoReady] = None
        self._stop = threading.Event()
        self._setup_done = False
        self._terminal_size: Optional[Size] = None

    # ------------------------------------------------------------------ setup

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        yield Static(id="devices")
        with Container(id="graphs"):
            with Horizontal(id="tpu-graphs"):
                yield MetricGraph("TensorCore util", "tensorcore", short_label="TC util", id="tc-graph")
                yield MetricGraph("Duty cycle", "duty_cycle", short_label="Duty", id="dc-graph")
                yield MetricGraph("HBM", "hbm", id="hbm-graph")
            with Horizontal(id="host-graphs"):
                yield MetricGraph("CPU", "cpu", id="cpu-graph")
                yield MetricGraph("RAM", "ram", id="ram-graph")
        with Vertical(id="processes"):
            yield ProcessTable(id="process-table")
        yield Footer()

    def on_mount(self) -> None:
        # Keep references to the main screen's widgets: on some Textual versions
        # `App.query_one` searches the active screen, which may be a dialog.
        self.header_bar = self.query_one(HeaderBar)
        self.devices_panel = self.query_one("#devices", Static)
        self.graphs_box = self.query_one("#graphs", Container)
        self.graph_views = {graph.id: graph for graph in self.query(MetricGraph)}
        self.processes_panel = self.query_one("#processes", Vertical)
        self.process_table = self.query_one(ProcessTable)

        self.register_theme(TPU_TOP_THEME)
        self.theme = self.initial_theme if self.initial_theme in self.available_themes else THEME_NAME
        self.theme_changed_signal.subscribe(self, lambda _theme: self._apply_palette())

        # Set up versions for header
        try:
            import libtpu

            self.libtpu_version = getattr(libtpu, "__version__", None) or _package_version("libtpu")
        except ImportError:
            self.libtpu_version = "--"

        try:
            import tpu_info

            self.tpu_info_version = getattr(tpu_info, "__version__", None) or _package_version("tpu-info")
        except ImportError:
            self.tpu_info_version = "--"

        self.tpu_top_version = _package_version("tpu-top")

        self.tpu_topology = os.environ.get("TPU_ACCELERATOR_TYPE")
        if self.tpu_topology:
            self.tpu_topology = f"TPU {self.tpu_topology}"
        if not self.tpu_topology:
            if self.use_mock:
                self.tpu_topology = "TPU mock"
            else:
                try:
                    from tpu_info import device

                    chip_type, _ = device.get_local_chips()
                    self.tpu_topology = f"TPU {chip_type.value.name}"
                except Exception:
                    self.tpu_topology = "TPU unknown"

        self.host_name = socket.gethostname()
        worker_id = os.environ.get("TPU_WORKER_ID")
        if worker_id:
            self.host_name = f"{self.host_name} · worker {worker_id}"

        self._setup_done = True
        self._apply_palette()
        self._update_layout()
        self._render_all()
        if self.use_mock and not self.mock_requested:
            self.notify(
                "tpu-info is not installed, so tpu-top is showing mock data.",
                title="Mock data",
                severity="warning",
                timeout=8,
            )
        self.collect_metrics_worker()

    def _apply_palette(self) -> None:
        if not self._setup_done:
            return
        variables = self.theme_variables
        self.palette = build_palette(
            background=variables.get("background"),
            foreground=variables.get("foreground"),
            primary=variables.get("primary"),
            dark=self.current_theme.dark,
        )
        border = ("round", self.palette.faint)
        self.devices_panel.styles.border = border
        self.processes_panel.styles.border = border
        for graph in self.graph_views.values():
            graph.apply_palette(self.palette)
        self._render_all()

    # --------------------------------------------------------------- metrics

    @work(thread=True, exclusive=True, group="metrics")
    def collect_metrics_worker(self) -> None:
        worker = get_current_worker()
        while not worker.is_cancelled and not self._stop.is_set():
            started = time.monotonic()
            if not self.paused:
                try:
                    message: Message = MetricsReady(self.collector.collect_metrics())
                except Exception as e:
                    message = MetricsFailed(str(e) or type(e).__name__)
                try:
                    self.post_message(message)
                except RuntimeError:
                    break  # The app is shutting down.
            self._stop.wait(max(0.05, self.interval - (time.monotonic() - started)))

    def on_metrics_ready(self, message: MetricsReady) -> None:
        if self.paused:
            return
        self.update_ui(message.data)

    def on_metrics_failed(self, message: MetricsFailed) -> None:
        self.last_error = message.error
        self._render_all()

    def update_ui(self, metrics_data: Dict[str, Any]) -> None:
        # Update history
        self.history.tick(time.monotonic())
        self.history.append_cpu(metrics_data["cpu_usage"])
        self.history.append_ram(metrics_data["ram_usage"]["percent"])

        devices = metrics_data["devices"]
        n = len(devices)
        avg_util = sum(d["tensorcore_util"] for d in devices) / n if n else 0.0
        avg_duty_cycle = sum(d["duty_cycle"] for d in devices) / n if n else 0.0
        avg_mem_pct = sum(_mem_pct(d) for d in devices) / n if n else 0.0
        self.history.append_tpu_util(avg_util)
        self.history.append_tpu_mem(avg_mem_pct)
        self.history.append_tpu_duty_cycle(avg_duty_cycle)
        for d in devices:
            self.history.append_chip(d["id"], d["tensorcore_util"], d["duty_cycle"], _mem_pct(d))

        chips_changed = self.last_data is None or len(self.last_data["devices"]) != n
        self.last_data = metrics_data
        self.last_error = ""
        if chips_changed:
            self._update_layout()
        self._render_all()

    # ------------------------------------------------------------- rendering

    @property
    def _state(self) -> str:
        if self.paused:
            return "paused"
        if self.last_error:
            return "error"
        if self.last_data is None:
            return "starting"
        return "live"

    def _render_all(self) -> None:
        if not self._setup_done:
            return
        self._render_header()
        self._render_devices()
        self._render_graphs()
        self._render_processes()

    def _render_header(self) -> None:
        data = self.last_data
        summary = ""
        if data and data["devices"]:
            chips = len(data["devices"])
            cores = chips * max(1, data.get("devices_per_chip", 1))
            summary = f"{chips} chip{'s' if chips != 1 else ''}"
            if cores != chips:
                summary += f" · {cores} cores"
        versions: List[Tuple[str, str]] = [
            (name, version)
            for name, version in (
                ("libtpu", self.libtpu_version),
                ("tpu-info", self.tpu_info_version),
                ("tpu-top", self.tpu_top_version),
            )
            if version and version != "--"
        ]
        self.header_bar.show(
            self.palette,
            topology=self.tpu_topology,
            summary=summary,
            host=self.host_name,
            versions=versions,
            state=self._state,
            mock=self.use_mock,
            interval=self.interval,
        )

    def _render_devices(self) -> None:
        p = self.palette
        panel = self.devices_panel
        panel.border_title = Text(" Devices ", style=f"bold {p.text}")
        data = self.last_data
        if data is None:
            panel.border_subtitle = None
            if self.last_error:
                hint = "Run tpu-top on a Cloud TPU VM or GKE TPU pod, or try `tpu-top --mock`."
                panel.update(
                    Group(
                        Text(f"Couldn't read TPU metrics: {self.last_error}", style=f"bold {p.bad}"),
                        Text(hint, style=p.muted),
                    )
                )
            else:
                panel.update(Text("Waiting for TPU metrics…", style=p.muted))
            return

        devices = data["devices"]
        used = sum(d["memory_usage"] for d in devices) / GIB
        total = sum(d["total_memory"] for d in devices) / GIB
        panel.border_subtitle = Text(f" HBM {used:.1f} / {total:.1f} GiB ", style=p.muted) if devices else None
        table = make_device_table(
            devices,
            data["hlo_map"],
            data["devices_per_chip"],
            palette=p,
            history=self.history,
            compact=self.compact,
            narrow=self.terminal_size.width < 110,
        )
        panel.update(table)

    def _render_graphs(self) -> None:
        data = self.last_data
        h = self.history
        period = h.sample_period or self.interval
        n = len(data["devices"]) if data else 0
        chips = (f"avg of {n} chips", f"{n} chips") if n > 1 else ("", "")

        hbm = ram = ("", "")
        if data:
            used = sum(d["memory_usage"] for d in data["devices"]) / GIB
            total = sum(d["total_memory"] for d in data["devices"]) / GIB
            if total:
                hbm = (f"{used:.1f} / {total:.1f} GiB", f"of {total:.0f} GiB")
            ram_total = data["ram_usage"].get("total", 0) / GIB
            ram_used = ram_total * data["ram_usage"]["percent"] / 100
            if ram_total:
                ram = (f"{ram_used:.1f} / {ram_total:.1f} GiB", f"of {ram_total:.0f} GiB")

        cores = f"{psutil.cpu_count() or 1} cores"
        for graph_id, series, (detail, short_detail) in (
            ("tc-graph", h.tpu_util, chips),
            ("dc-graph", h.tpu_duty_cycle, chips),
            ("hbm-graph", h.tpu_mem, hbm),
            ("cpu-graph", h.cpu, (cores, cores)),
            ("ram-graph", h.ram, ram),
        ):
            self.graph_views[graph_id].show(series, detail, period, self.graph_style, short_detail)

    def _render_processes(self) -> None:
        p = self.palette
        panel = self.processes_panel
        table = self.process_table
        rows = []
        if self.last_data:
            rows = build_process_rows(
                self.last_data["processes"],
                cpu_count=psutil.cpu_count() or 1,
                exclude_pids={os.getpid()},
            )
        tpu_rows = sum(1 for r in rows if r.on_tpu)
        title = Text.assemble((" Processes ", f"bold {p.text}"))
        if rows:
            title.append(f"{tpu_rows} on TPU · {len(rows) - tpu_rows} busy on CPU ", style=p.muted)
        panel.border_title = title
        panel.border_subtitle = Text.assemble(
            (" sort ", p.muted), (table.sort_mode, f"bold {p.text}"), ("  s ", p.faint)
        )
        table.show(rows, p)

    # --------------------------------------------------------------- layout

    def on_resize(self, event: events.Resize) -> None:
        # Newer Textual versions update `App.size` after a short delay, so use
        # the size carried by the event.
        self._terminal_size = event.size
        if not self._setup_done:
            return
        self._update_layout()
        self._render_all()
        # Render again once the widgets have their new sizes (graph labels depend on them).
        self.set_timer(0.3, self._render_all)

    @property
    def terminal_size(self) -> Size:
        return self._terminal_size or self.size

    def _update_layout(self) -> None:
        """Pick the richest layout that still leaves room for the process list."""
        height = self.terminal_size.height
        data = self.last_data
        chips = len(data["devices"]) if data else 4
        cores = max(1, data.get("devices_per_chip", 1)) if data else 1
        # Devices panel: borders, header and rule, then the chip rows.
        devices_full = 4 + chips * max(2, cores)
        devices_compact = 4 + chips

        # (one line per chip, graph layout, graph height, process rows to keep)
        options = (
            (False, "split", 14, PROCESSES_MIN_ROWS),
            (False, "row", 7, 8),
            (False, "short", 6, 8),
            (True, "row", 7, 8),
            (True, "short", 6, 6),
        )
        for compact, mode, graphs_height, process_rows in options:
            devices = devices_compact if compact else devices_full
            if 1 + devices + graphs_height + process_rows + 1 <= height:
                break
        else:
            compact, mode = True, "hidden"

        self.compact = compact
        graphs = self.graphs_box
        graphs.set_class(mode != "split", "row")
        graphs.set_class(mode == "short", "short")
        graphs.display = mode != "hidden"

    # -------------------------------------------------------------- actions

    async def action_quit(self) -> None:
        self._stop.set()
        self.exit()

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        if not self.paused:
            self.history.reset_clock()
        self._render_header()

    def action_toggle_graph_style(self) -> None:
        self.graph_style = "block" if self.graph_style == "braille" else "braille"
        self._render_graphs()
        self.notify(f"Graph style: {'blocks' if self.graph_style == 'block' else 'braille'}", timeout=2)

    def action_cycle_sort(self) -> None:
        self.process_table.cycle_sort()
        self._render_processes()

    def on_process_table_sort_changed(self, _message: ProcessTable.SortChanged) -> None:
        self._render_processes()

    def action_help(self) -> None:
        if isinstance(self.screen, HelpScreen):
            self.screen.dismiss()
            return
        self.push_screen(HelpScreen(make_help(self.palette), "Help"))

    def action_toggle_info(self) -> None:
        """Toggle the display of TPU info."""
        if isinstance(self.screen, InfoScreen):
            self.screen.dismiss()
            return
        if self._hardware_info is not None:
            self._show_hardware_info(self._hardware_info)
            return
        self.notify("Reading TPU hardware info…", timeout=2)
        self._load_hardware_info()

    def action_default_view(self) -> None:
        """Return to the default view (close any dialog)."""
        if isinstance(self.screen, (InfoScreen, HelpScreen)):
            self.screen.dismiss()

    @work(thread=True, exclusive=True, group="hardware")
    def _load_hardware_info(self) -> None:
        try:
            message = HardwareInfoReady(collect_hardware_info())
        except HardwareInfoUnavailable as e:
            message = HardwareInfoReady(None, str(e))
        except Exception as e:
            message = HardwareInfoReady(None, f"Failed to get TPU info: {e}")
        self.post_message(message)

    def on_hardware_info_ready(self, message: HardwareInfoReady) -> None:
        if message.sections is not None:
            self._hardware_info = message
        self._show_hardware_info(message)

    def _show_hardware_info(self, info: HardwareInfoReady) -> None:
        p = self.palette
        if info.sections is not None:
            body: Any = make_info_sheet(info.sections, p)
        else:
            body = Group(
                Text("TPU info not available", style=f"bold {p.bad}"),
                Text(info.error, style=p.muted),
            )
        if not isinstance(self.screen, InfoScreen):
            self.push_screen(InfoScreen(body, f"{self.tpu_topology} · hardware"))


def _mem_pct(device: Dict[str, Any]) -> float:
    total = device.get("total_memory") or 0
    return device["memory_usage"] / total * 100 if total > 0 else 0.0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tpu-top",
        description="Live TPU and host monitoring for Cloud TPU VMs and GKE pods.",
    )
    parser.add_argument(
        "-n",
        "--interval",
        type=float,
        default=0.5,
        help="seconds between updates (default: 0.5)",
    )
    parser.add_argument(
        "-t",
        "--theme",
        default=os.environ.get("TPU_TOP_THEME", THEME_NAME),
        help=f"color theme name, e.g. {THEME_NAME}, dracula, nord, tokyo-night, textual-light",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="show simulated data; no TPU needed (same as TPU_TOP_MOCK=1)",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {_package_version('tpu-top')}"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    mock_requested = args.mock or os.environ.get("TPU_TOP_MOCK") == "1"
    use_mock = mock_requested or not HAS_TPU_INFO
    app = TpuTopApp(
        use_mock=use_mock,
        interval=max(0.1, args.interval),
        mock_requested=mock_requested,
        initial_theme=args.theme,
    )
    app.run()


if __name__ == "__main__":
    main()
