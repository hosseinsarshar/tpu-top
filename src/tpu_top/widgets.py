"""Textual widgets that make up the tpu-top screen."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import DataTable, Static

from tpu_top.theme import DEFAULT_PALETTE, Palette
from tpu_top.ui import (
    SORT_MODES,
    ProcessRow,
    braille_graph,
    format_bytes,
    make_timeline,
    sort_process_rows,
    vertical_bar_chart,
)


class HeaderBar(Static):
    """One-line title bar: what is being monitored, versions and live status."""

    DEFAULT_CSS = """
    HeaderBar {
        height: 1;
        padding: 0 1 0 0;
        background: $panel;
    }
    """

    def show(
        self,
        palette: Palette,
        topology: str,
        summary: str,
        host: str,
        versions: Sequence[Tuple[str, str]],
        state: str,
        mock: bool,
        interval: float = 0.0,
    ) -> None:
        p = palette
        left = Text(no_wrap=True, overflow="ellipsis")
        left.append(" ◆ tpu-top ", style=f"bold {p.background} on {p.primary}")
        left.append("  ")
        left.append(topology, style=f"bold {p.text}")
        if summary:
            left.append(f"  {summary}", style=p.muted)
        if host:
            left.append("  │  ", style=p.faint)
            left.append(host, style=p.muted)

        right = Text(no_wrap=True, justify="right")
        if self.size.width >= 120:  # Versions are the first thing to go on narrow screens.
            for name, version in versions:
                right.append(f"{name} ", style=p.faint)
                right.append(f"{version}  ", style=p.muted)
        if mock:
            right.append(" MOCK DATA ", style=f"bold {p.background} on {p.warn}")
            right.append("  ")
        if state == "paused":
            right.append("❚❚ paused", style=f"bold {p.warn}")
        elif state == "error":
            right.append("● no data", style=f"bold {p.bad}")
        elif state == "starting":
            right.append("● starting", style=f"bold {p.muted}")
        else:
            right.append("● live", style=f"bold {p.good}")
            if interval:
                right.append(f" {interval:g}s", style=p.faint)

        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        grid.add_column(no_wrap=True, justify="right")
        grid.add_row(left, right)
        self.update(grid)


class MetricGraph(Widget):
    """A bordered history graph that redraws to fit its size."""

    DEFAULT_CSS = """
    MetricGraph {
        height: 1fr;
        min-width: 10;
        border: round $panel-lighten-2;
        border-title-align: left;
        border-subtitle-align: right;
        padding: 0 1;
    }
    """

    def __init__(self, label: str, hue: str, short_label: str = "", id: Optional[str] = None) -> None:
        super().__init__(id=id)
        self.label = label
        self.short_label = short_label or label
        self.hue_name = hue
        self.data: Sequence[float] = ()
        self.sample_period = 0.5
        self.graph_style = "braille"
        self.palette: Palette = DEFAULT_PALETTE

    def apply_palette(self, palette: Palette) -> None:
        self.palette = palette
        self.styles.border = ("round", getattr(palette, self.hue_name).border)
        self.refresh()

    def show(
        self,
        data: Sequence[float],
        detail: str,
        sample_period: float,
        graph_style: str,
        short_detail: str = "",
    ) -> None:
        p = self.palette
        hue = getattr(p, self.hue_name)
        self.data = data
        self.sample_period = sample_period
        self.graph_style = graph_style

        # Border labels get the outer width minus the corners and some blanks.
        room = self.outer_size.width - 5
        value = ""
        if data:
            value = f"{data[-1]:5.1f}% " if room >= 22 else f"{data[-1]:.0f}% "
        labels = (self.label, self.short_label, self.short_label.split()[0])
        label = next((l for l in labels if len(l) + len(value) + 2 <= room), labels[-1])
        self.border_title = Text.assemble((f" {label} ", f"bold {hue.fg}"), (value, f"bold {p.text}"))

        fitting = [d for d in (detail, short_detail) if d and len(d) + 2 <= room]
        self.border_subtitle = Text(f" {fitting[0]} ", style=p.muted) if fitting else None
        self.refresh()

    def render(self) -> RenderableType:
        width, height = self.content_size
        if width <= 0 or height <= 0:
            return Text("")
        hue = getattr(self.palette, self.hue_name)
        show_axis = height >= 3
        chart_height = height - 1 if show_axis else height
        if self.graph_style == "block":
            rows = vertical_bar_chart(list(self.data), width, chart_height)
            seconds_per_col = self.sample_period
        else:
            rows = braille_graph(self.data, width, chart_height)
            seconds_per_col = self.sample_period * 2

        text = Text(no_wrap=True, overflow="crop")
        for i, row in enumerate(rows):
            level = 1.0 - i / max(1, chart_height - 1)
            text.append(row, style=hue.at(0.3 + 0.7 * level))
            if i < len(rows) - 1 or show_axis:
                text.append("\n")
        if show_axis:
            # Narrow graphs get labels closer together so they still show a scale.
            gap = min(16, max(8, width // 3))
            text.append(make_timeline(width, seconds_per_col, min_gap=gap), style=self.palette.faint)
        return text


# (label, key, width); a width of None fills the remaining space.
_PROCESS_COLUMNS: Tuple[Tuple[str, str, Optional[int]], ...] = (
    ("PID", "pid", 8),
    ("USER", "user", 9),
    ("DEVICE", "device", 9),
    ("CPU%", "cpu", 6),
    ("MEM", "memory", 10),
    ("NAME", "name", 20),
    ("COMMAND", "command", None),
)
_RIGHT_ALIGNED = {"pid", "cpu", "memory"}
_SORT_BY_COLUMN = {"pid": "pid", "device": "device", "cpu": "cpu", "memory": "memory"}
# Columns dropped, in order, when the table is too narrow.
_OPTIONAL_COLUMNS = ("user", "command")
_MIN_FILL_WIDTH = 16


class ProcessTable(DataTable):
    """Processes using the TPUs, plus busy CPU processes. Sortable."""

    DEFAULT_CSS = """
    ProcessTable {
        height: 1fr;
        background: transparent;
        overflow-x: hidden;
    }
    """

    def __init__(self, id: Optional[str] = None) -> None:
        super().__init__(id=id, cursor_type="row", zebra_stripes=True)
        self.sort_mode = "device"
        self.palette: Palette = DEFAULT_PALETTE
        self._rows: List[ProcessRow] = []
        self._signatures: Dict[str, tuple] = {}
        self._layout_width = 0
        self._columns: List[Tuple[str, str, int]] = []

    def on_mount(self) -> None:
        self._build_columns()

    def on_resize(self) -> None:
        if self.size.width != self._layout_width:
            self._build_columns()

    def _fit_columns(self, width: int) -> List[Tuple[str, str, int]]:
        """Choose columns and widths so the last one fills the table width."""
        columns = list(_PROCESS_COLUMNS)
        dropped: List[str] = []
        while True:
            fill_key = "command" if "command" not in dropped else "name"
            fixed = sum(w for _, key, w in columns if w and key != fill_key)
            padding = 2 * self.cell_padding * len(columns)
            # Leave room for a vertical scrollbar.
            fill = width - fixed - padding - 2
            optional = [key for key in _OPTIONAL_COLUMNS if key not in dropped]
            if fill >= _MIN_FILL_WIDTH or not optional:
                break
            dropped.append(optional[0])
            columns = [c for c in columns if c[1] not in dropped]
        return [(label, key, w if key != fill_key else max(8, fill)) for label, key, w in columns]

    def _build_columns(self) -> None:
        self._layout_width = self.size.width
        self._columns = self._fit_columns(self.size.width)
        self.clear(columns=True)
        self._signatures.clear()
        for label, key, width in self._columns:
            header = Text(label, justify="right" if key in _RIGHT_ALIGNED else "left")
            self.add_column(header, key=key, width=width)
        self._sync()

    def show(self, rows: List[ProcessRow], palette: Palette) -> None:
        if palette is not self.palette:
            self.palette = palette
            self._signatures.clear()
        self._rows = rows
        self._sync()

    def cycle_sort(self) -> str:
        index = SORT_MODES.index(self.sort_mode) if self.sort_mode in SORT_MODES else -1
        self.sort_mode = SORT_MODES[(index + 1) % len(SORT_MODES)]
        self._sync()
        return self.sort_mode

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        mode = _SORT_BY_COLUMN.get(str(event.column_key.value))
        if mode:
            self.sort_mode = mode
            self._sync()
            self.post_message(self.SortChanged(mode))

    class SortChanged(Message):
        """Posted when the sort column is changed by clicking a header."""

        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    def _cells(self, row: ProcessRow) -> List[Text]:
        """Cells for the columns currently shown."""
        p = self.palette
        command = row.command
        if command.strip() == row.name:
            command = ""
        cpu_style = p.warn if row.cpu >= 50 else p.text
        cells = {
            "pid": Text(str(row.pid), style=p.muted, justify="right"),
            "user": Text(row.user or "—", style=p.muted, no_wrap=True, overflow="ellipsis"),
            "device": Text(row.device_label, style=f"bold {p.primary}" if row.on_tpu else p.muted),
            "cpu": Text(f"{row.cpu:.1f}", style=cpu_style, justify="right"),
            "memory": Text(format_bytes(row.memory) if row.memory else "—", style=p.text, justify="right"),
            "name": Text(row.name, style=f"bold {p.text}", no_wrap=True, overflow="ellipsis"),
            "command": Text(command, style=p.muted, no_wrap=True, overflow="ellipsis"),
        }
        return [cells[key] for _, key, _ in self._columns]

    def _sync(self) -> None:
        if not self.columns:
            return
        ordered = sort_process_rows(self._rows, self.sort_mode)
        wanted = {str(r.pid): r for r in ordered}

        selected: Optional[str] = None
        if self.row_count:
            try:
                selected = self.coordinate_to_cell_key(self.cursor_coordinate).row_key.value
            except Exception:
                selected = None

        for key in [k.value for k in self.rows if k.value not in wanted]:
            self.remove_row(key)
            self._signatures.pop(key, None)

        column_keys = [key for _, key, _ in self._columns]
        for key, row in wanted.items():
            signature = (row.user, row.device_label, round(row.cpu, 1), row.memory, row.name, row.command)
            if key not in self.rows:
                self.add_row(*self._cells(row), key=key)
            elif self._signatures.get(key) != signature:
                for column, value in zip(column_keys, self._cells(row)):
                    self.update_cell(key, column, value)
            self._signatures[key] = signature

        order = list(wanted)
        if [r.key.value for r in self.ordered_rows] != order:
            rank = {key: i for i, key in enumerate(order)}
            self.sort("pid", key=lambda cell: rank.get(cell.plain.strip(), len(rank)))

        if selected in wanted:
            index = self.get_row_index(selected)
            if index != self.cursor_row:
                self.move_cursor(row=index, animate=False, scroll=False)


class _Dialog(ModalScreen[None]):
    """A centered, scrollable dialog that closes with escape."""

    DEFAULT_CSS = """
    _Dialog {
        align: center middle;
        background: $background 70%;
    }
    _Dialog > VerticalScroll {
        width: 84;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        border: round $primary;
        border-title-align: left;
        border-subtitle-align: right;
        background: $surface;
        padding: 1 2;
    }
    """

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, body: RenderableType, title: str) -> None:
        super().__init__()
        self.body = body
        self.title_text = title

    def compose(self) -> ComposeResult:
        scroll = VerticalScroll()
        scroll.border_title = f" {self.title_text} "
        scroll.border_subtitle = " esc to close "
        with scroll:
            yield Static(self.body)

    def action_close(self) -> None:
        self.dismiss()


class InfoScreen(_Dialog):
    """TPU hardware specs."""

    BINDINGS = [Binding("i", "close", "Close", show=False)]


class HelpScreen(_Dialog):
    """Key bindings and how to read the metrics."""

    BINDINGS = [Binding("question_mark", "close", "Close", show=False)]


def make_help(palette: Palette) -> RenderableType:
    """Help text: key bindings and a short guide to the TPU metrics."""
    p = palette
    keys = Table.grid(padding=(0, 2))
    keys.add_column(style=f"bold {p.primary}", no_wrap=True)
    keys.add_column(style=p.text)
    for key, description in (
        ("i", "TPU hardware info"),
        ("p / space", "Pause or resume updates"),
        ("s", "Change process sort order (or click a column header)"),
        ("g", "Switch graph style (braille / blocks)"),
        ("↑ ↓ PgUp PgDn", "Scroll the process list"),
        ("ctrl+p", "Command palette (change theme, ...)"),
        ("?", "This help"),
        ("q / ctrl+c", "Quit"),
    ):
        keys.add_row(key, description)

    guide = Table.grid(padding=(0, 2))
    guide.add_column(no_wrap=True)
    guide.add_column(style=p.text, ratio=1)
    guide.add_row(
        Text("Duty cycle", style=f"bold {p.duty_cycle.fg}"),
        "Share of time the TPU was busy running work. Low values usually mean the "
        "TPU is waiting on the host, e.g. for input data.",
    )
    guide.add_row(
        Text("TensorCore util", style=f"bold {p.tensorcore.fg}"),
        "How much of the peak matrix-multiply capacity is used while busy. Low values "
        "with a high duty cycle point to small batches or memory-bound ops.",
    )
    guide.add_row(
        Text("HBM", style=f"bold {p.hbm.fg}"),
        "High-bandwidth memory in use on each chip.",
    )

    body = Table.grid()
    body.add_column()
    body.add_row(Text("Keys", style=f"bold {p.text}"))
    body.add_row(keys)
    body.add_row("")
    body.add_row(Text("Reading the metrics", style=f"bold {p.text}"))
    body.add_row(guide)
    body.add_row("")

    combos = Table.grid(padding=(0, 2))
    combos.add_column(style=p.muted, no_wrap=True)
    combos.add_column(style=p.text)
    combos.add_row("Low duty cycle, low util", "Mostly idle, likely waiting for data.")
    combos.add_row("High duty cycle, low util", "Busy, but the ops are not math-heavy.")
    combos.add_row("High duty cycle, high util", "The TPU is well used.")
    body.add_row(combos)
    return body
