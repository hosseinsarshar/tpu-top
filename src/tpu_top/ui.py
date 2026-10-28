"""Rich renderables used by the tpu-top dashboard."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.measure import Measurement
from rich.style import Style
from rich.table import Table
from rich.text import Text

from tpu_top.theme import DEFAULT_PALETTE, Hue, Palette

GIB = 1024**3

# ---------------------------------------------------------------------------
# Sparklines and graphs
# ---------------------------------------------------------------------------


def sparkline(data: List[float], width: int = 40) -> str:
    """Generate a sparkline string from data."""
    bars = " ▂▃▄▅▆▇█"
    if not data:
        return " " * width

    if len(data) < width:
        data = [0.0] * (width - len(data)) + data
    else:
        data = data[-width:]

    min_val = 0.0
    max_val = 100.0
    extent = max_val - min_val

    res = []
    for v in data:
        idx = int(min(max(v - min_val, 0), extent) / extent * (len(bars) - 1))
        res.append(bars[idx])
    return "".join(res)


def vertical_bar_chart(data: List[float], width: int, height: int = 3) -> List[str]:
    """Generate a multi-line vertical bar chart from data."""
    bars = " ▂▃▄▅▆▇█"
    if not data:
        return [" " * width] * height

    if len(data) < width:
        data = [0.0] * (width - len(data)) + data
    else:
        data = data[-width:]

    lines = []
    for h in range(height - 1, -1, -1):  # From top to bottom
        line = []
        for v in data:
            s = (v / 100.0) * height
            if s >= h + 1:
                line.append("█")
            elif s <= h:
                line.append(" ")
            else:
                f = s - h
                idx = int(f * 8)
                idx = min(idx, 7)
                line.append(bars[idx])
        lines.append("".join(line))
    return lines


# Braille dot bits for the left and right halves of a cell, bottom to top.
_BRAILLE_LEFT = (0x40, 0x04, 0x02, 0x01)
_BRAILLE_RIGHT = (0x80, 0x20, 0x10, 0x08)


def braille_graph(data: Sequence[float], width: int, height: int = 3) -> List[str]:
    """Draw an area graph with braille dots.

    Each character holds two samples side by side and four levels top to
    bottom, so the graph shows twice as much history as a block chart of the
    same width, with four times the vertical resolution. The newest sample
    is in the right-most column.
    """
    if width <= 0 or height <= 0:
        return []
    samples = width * 2
    values = list(data)[-samples:]
    values = [0.0] * (samples - len(values)) + values
    levels = height * 4

    dots = []
    for v in values:
        v = min(max(float(v), 0.0), 100.0)
        level = round(v / 100.0 * levels)
        if v > 0 and level == 0:
            level = 1  # Keep a visible baseline for small non-zero values.
        dots.append(level)

    lines = []
    for row in range(height - 1, -1, -1):
        floor = row * 4
        chars = []
        for col in range(width):
            left = min(max(dots[2 * col] - floor, 0), 4)
            right = min(max(dots[2 * col + 1] - floor, 0), 4)
            code = 0
            for i in range(left):
                code |= _BRAILLE_LEFT[i]
            for i in range(right):
                code |= _BRAILLE_RIGHT[i]
            chars.append(chr(0x2800 + code) if code else " ")
        lines.append("".join(chars))
    return lines


_TICK_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)


def format_duration(seconds: float) -> str:
    """Short label for a time offset: ``45s``, ``2m``, ``1m30s``."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    minutes, rest = divmod(s, 60)
    if minutes < 60:
        return f"{minutes}m" if rest == 0 else f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h" if minutes == 0 else f"{hours}h{minutes:02d}m"


def make_timeline(width: int, seconds_per_col: float = 0.5, min_gap: int = 16) -> str:
    """Build a time axis for a graph whose newest sample is on the right.

    Args:
        width: Number of columns.
        seconds_per_col: How much time one column covers.
        min_gap: Minimum number of columns between two labels.
    """
    if width <= 0:
        return ""
    chars = ["─"] * width
    seconds_per_col = max(seconds_per_col, 1e-3)

    def place(label: str, start: int) -> None:
        if start > 0:
            chars[start - 1] = " "
        for i, ch in enumerate(label):
            chars[start + i] = ch
        end = start + len(label)
        if end < width:
            chars[end] = " "

    if width >= 3:
        place("0s", width - 2)
    elif width == 2:
        chars[:] = ["0", "s"]
        return "".join(chars)

    step = next((s for s in _TICK_STEPS if s / seconds_per_col >= min_gap), _TICK_STEPS[-1])
    limit = width - 3  # Labels must end before the space in front of "0s".
    t = step
    while True:
        label = format_duration(t)
        col = width - 1 - round(t / seconds_per_col)
        start = col - len(label) // 2
        if start < 1:
            break
        if start + len(label) < limit:
            place(label, start)
            limit = start - 1
        t += step
    return "".join(chars)


# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

_EIGHTHS = " ▏▎▍▌▋▊▉"


class Gauge:
    """A horizontal bar that fills the width it is given, followed by a label.

    The filled part fades from a faint to a strong tone of the metric's hue,
    so higher values also look brighter. ``thin`` draws a line instead of a
    solid bar, which keeps bars on adjacent rows apart.
    """

    def __init__(
        self,
        value: float,
        hue: Hue,
        track: str,
        label: Optional[str] = None,
        label_style: Optional[str] = None,
        label_width: int = 0,
        min_bar: int = 4,
        thin: bool = False,
    ) -> None:
        self.value = value
        self.hue = hue
        self.track = track
        self.label = f"{value:.1f}%" if label is None else label
        self.label_style = label_style or f"bold {hue.fg}"
        self.label_width = max(label_width, cell_len(self.label))
        self.min_bar = min_bar
        self.thin = thin

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(self.min_bar + self.label_width + 1, options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.render(options.max_width)

    def _color(self, cell: int, bar_w: int) -> str:
        return self.hue.at(0.35 + 0.65 * (cell + 1) / bar_w)

    def render(self, width: int) -> Text:
        """Render the gauge as exactly ``width`` cells (when it fits)."""
        label_w = self.label_width if width > self.label_width + 1 else 0
        bar_w = max(1, width - (label_w + 1 if label_w else 0))
        fraction = min(max(self.value / 100.0, 0.0), 1.0)

        text = Text(no_wrap=True, overflow="crop")
        if self.thin:
            full, half = divmod(round(fraction * bar_w * 2), 2)
            for i in range(full):
                text.append("━", self._color(i, bar_w))
            used = full
            if half and used < bar_w:
                text.append("╸", self._color(used, bar_w))
                used += 1
            if used < bar_w:
                text.append("━" * (bar_w - used), self.track)
        else:
            full, part = divmod(round(fraction * bar_w * 8), 8)
            for i in range(full):
                text.append(" ", Style(bgcolor=self._color(i, bar_w)))
            used = full
            if part and used < bar_w:
                text.append(_EIGHTHS[part], Style(color=self._color(used, bar_w), bgcolor=self.track))
                used += 1
            if used < bar_w:
                text.append(" " * (bar_w - used), Style(bgcolor=self.track))
        if label_w:
            text.append(" ")
            text.append(self.label.rjust(label_w), self.label_style)
        return text


class Trend:
    """A one-line braille sparkline that fills the width it is given."""

    def __init__(self, data: Sequence[float], hue: Hue, min_width: int = 4) -> None:
        self.data = data
        self.hue = hue
        self.min_width = min_width

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(self.min_width, options.max_width)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self.render(options.max_width)

    def render(self, width: int) -> Text:
        if width <= 0:
            return Text("")
        line = braille_graph(self.data, width, height=1)[0]
        return Text(line, style=self.hue.at(0.45), no_wrap=True, overflow="crop")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_bytes(num: float, precision: int = 1) -> str:
    """Human readable binary size: ``512.0 MiB``, ``26.7 GiB``."""
    if num <= 0:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.{precision}f} {unit}"
        num /= 1024
    return f"{num:.{precision}f} PiB"


def format_device_list(ids: Iterable[int]) -> str:
    """Compact list of chip ids: ``[0, 1, 2, 3, 5]`` becomes ``0-3,5``."""
    ordered = sorted(set(ids))
    if not ordered:
        return ""
    parts = []
    start = prev = ordered[0]
    for i in ordered[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


_HLO_RE = re.compile(r"^\s*HLO:\s*(?P<op>.*?)\s*;\s*HLO computation:\s*(?P<comp>.*?)\s*$")


def parse_hlo_location(location: Optional[str]) -> Tuple[str, str]:
    """Split a tpu-info HLO location into ``(op, computation)``.

    Locations look like ``"HLO: fusion.12; HLO computation: main.579_spmd"``.
    Anything else is returned as the op name.
    """
    if not location or location in ("N/A", "no HLO mapping"):
        return "", ""
    match = _HLO_RE.match(location)
    if match:
        return match.group("op"), match.group("comp")
    return location.strip(), ""


def hlo_lines_for_device(
    device: Dict[str, Any], hlo_map: Dict[int, Dict[str, str]], devices_per_chip: int
) -> List[Tuple[str, str]]:
    """Return ``(core label, HLO location)`` pairs to show for one chip."""
    hlo_dict = hlo_map.get(device["id"], {})
    has_real_hlo = any(loc and loc != "N/A" for loc in hlo_dict.values())
    if not has_real_hlo and device["id"] != 0:
        if device["tensorcore_util"] > 10 and 0 in hlo_map:
            hlo_dict = hlo_map[0]

    lines = []
    for i in range(devices_per_chip):
        label = f"TC{i}"
        lines.append((label, hlo_dict.get(label, "N/A")))
    for label in sorted(hlo_dict.keys()):
        if not label.startswith("TC"):
            loc = hlo_dict[label]
            if loc and loc != "N/A":
                lines.append((label, loc))
    return lines


# ---------------------------------------------------------------------------
# Devices table
# ---------------------------------------------------------------------------


def _hlo_text(lines: List[Tuple[str, str]], palette: Palette, compact: bool = False) -> Text:
    """HLO op per core, one line each (or all on one line when ``compact``)."""
    text = Text(no_wrap=True, overflow="ellipsis")
    if compact:
        for i, (label, location) in enumerate(lines):
            if i:
                text.append("   ")
            op, _ = parse_hlo_location(location)
            text.append(label + " ", palette.muted)
            text.append(op or "—", palette.text if op else palette.faint)
        return text

    label_width = max((len(label) for label, _ in lines), default=3)
    for i, (label, location) in enumerate(lines):
        if i:
            text.append("\n")
        text.append(label.ljust(label_width) + "  ", palette.muted)
        op, computation = parse_hlo_location(location)
        if not op:
            text.append("—", palette.faint)
            continue
        text.append(op, palette.text)
        if computation:
            text.append(" · ", palette.faint)
            text.append(computation, palette.muted)
    return text


def make_device_table(
    devices: List[Dict[str, Any]],
    hlo_map: Dict[int, Dict[str, str]],
    devices_per_chip: int,
    palette: Optional[Palette] = None,
    history: Any = None,
    compact: bool = False,
    narrow: bool = False,
) -> Table:
    """One row per chip: HBM, TensorCore utilization, duty cycle and HLO ops.

    Args:
        devices: Per-chip metrics from ``MetricsCollector``.
        hlo_map: HLO location per chip and core.
        devices_per_chip: TensorCores per chip.
        palette: Colors to use.
        history: Optional ``MetricsHistory`` used to draw per-chip trends.
        compact: Use a single line per chip.
        narrow: Use shorter column headers.
    """
    p = palette or DEFAULT_PALETTE
    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        border_style=p.faint,
        header_style=f"bold {p.muted}",
        row_styles=["", f"on {p.stripe}"],
    )
    table.add_column("TPU", width=8 if not compact else 6, no_wrap=True)
    table.add_column("HBM", ratio=24, no_wrap=True)
    table.add_column("TC util" if narrow else "TensorCore util", ratio=19, no_wrap=True)
    table.add_column("Duty" if narrow else "Duty cycle", ratio=19, no_wrap=True)
    table.add_column("HLO op" if narrow else "HLO op per core", ratio=38, no_wrap=True, overflow="ellipsis")

    track = p.faint if compact else p.track
    for d in devices:
        total = d["total_memory"] or 0
        mem_pct = d["memory_usage"] / total * 100 if total > 0 else 0.0
        util = d["tensorcore_util"]
        duty = d["duty_cycle"]

        hbm = Gauge(mem_pct, p.hbm, track, label_width=6, thin=compact)
        tc = Gauge(util, p.tensorcore, track, label=f"{util:.0f}%", label_width=4, thin=compact)
        dc = Gauge(duty, p.duty_cycle, track, label=f"{duty:.0f}%", label_width=4, thin=compact)

        name = Text.assemble(("TPU ", p.muted), (str(d["id"]), f"bold {p.text}"))
        if compact:
            tpu_cell: Any = name
            hbm_cell: Any = hbm
            tc_cell: Any = tc
            dc_cell: Any = dc
        else:
            active = duty >= 1.0 or util >= 1.0
            status = Text.assemble(
                ("● " if active else "○ ", p.good if active else p.faint),
                ("active" if active else "idle", p.muted if active else p.faint),
            )
            tpu_cell = Group(name, status)
            hbm_detail = Text(
                f"{d['memory_usage'] / GIB:.1f} / {total / GIB:.1f} GiB",
                style=p.muted,
                no_wrap=True,
                overflow="ellipsis",
            )
            hbm_cell = Group(hbm, hbm_detail)
            chip_util = history.chip_util.get(d["id"], []) if history is not None else []
            chip_duty = history.chip_duty_cycle.get(d["id"], []) if history is not None else []
            tc_cell = Group(tc, Trend(chip_util, p.tensorcore))
            dc_cell = Group(dc, Trend(chip_duty, p.duty_cycle))

        hlo = _hlo_text(hlo_lines_for_device(d, hlo_map, devices_per_chip), p, compact=compact)
        table.add_row(tpu_cell, hbm_cell, tc_cell, dc_cell, hlo)
    return table


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

SORT_MODES = ("device", "cpu", "memory", "pid")


@dataclass
class ProcessRow:
    """One process in the processes table. TPU owners are merged by PID."""

    pid: int
    name: str
    user: str = ""
    command: str = ""
    devices: List[int] = field(default_factory=list)
    cpu: float = 0.0
    memory: int = 0

    @property
    def on_tpu(self) -> bool:
        return bool(self.devices)

    @property
    def device_label(self) -> str:
        return f"TPU {format_device_list(self.devices)}" if self.devices else "CPU"


def build_process_rows(
    processes: List[Dict[str, Any]],
    cpu_count: int = 1,
    exclude_pids: Iterable[int] = (),
) -> List[ProcessRow]:
    """Turn raw process metrics into table rows.

    A process that owns several TPU chips shows up once with all of its chips.
    CPU usage is scaled to the whole host (100% means every core is busy).
    """
    excluded = set(exclude_pids)
    cpu_count = max(1, cpu_count or 1)
    rows: Dict[int, ProcessRow] = {}
    for p in processes:
        pid = p["pid"]
        name = p.get("name", "")
        if pid in excluded or "tpu-top" in name or "tputop" in name:
            continue
        row = rows.get(pid)
        if row is None:
            row = rows[pid] = ProcessRow(
                pid=pid,
                name=name,
                user=p.get("user", "") or "",
                command=p.get("command", "") or "",
                cpu=p.get("cpu", 0.0) / cpu_count,
                memory=p.get("memory", 0),
            )
        device = str(p.get("device", "CPU"))
        if device.startswith("TPU"):
            try:
                chip = int(device.split()[1])
            except (IndexError, ValueError):
                continue
            if chip not in row.devices:
                row.devices.append(chip)
    for row in rows.values():
        row.devices.sort()
    return list(rows.values())


def sort_process_rows(rows: List[ProcessRow], mode: str = "device") -> List[ProcessRow]:
    """Sort rows. ``device`` lists TPU owners first (by chip), then CPU processes."""
    if mode == "cpu":
        return sorted(rows, key=lambda r: (-r.cpu, -r.memory, r.pid))
    if mode == "memory":
        return sorted(rows, key=lambda r: (-r.memory, r.pid))
    if mode == "pid":
        return sorted(rows, key=lambda r: r.pid)
    return sorted(
        rows,
        key=lambda r: (0, r.devices[0], -r.memory, r.pid) if r.devices else (1, 0, -r.memory, r.pid),
    )


# ---------------------------------------------------------------------------
# Hardware info sheet
# ---------------------------------------------------------------------------


def make_info_sheet(sections: List[Tuple[str, List[Tuple[str, str]]]], palette: Optional[Palette] = None) -> Table:
    """Two-column spec sheet with a heading per section."""
    p = palette or DEFAULT_PALETTE
    table = Table.grid(padding=(0, 3), expand=True)
    table.add_column(style=p.muted, no_wrap=True)
    table.add_column(style=f"bold {p.text}", ratio=1)
    for i, (title, rows) in enumerate(sections):
        if i:
            table.add_row("", "")
        table.add_row(Text(title, style=f"bold {p.primary}"), "")
        for key, value in rows:
            table.add_row(f"  {key}", value)
    return table
