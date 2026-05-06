from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich import box
from rich.text import Text
import psutil

console = Console()

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
    for h in range(height - 1, -1, -1): # From top to bottom
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

def make_timeline(width: int) -> str:
    """Generate a timeline string with markers."""
    res = ["-"] * width
    if width >= 3:
        res[-3] = " "
        res[-2] = "0"
        res[-1] = "s"
    if width >= 34:
        res[width - 34] = " "
        res[width - 33] = "1"
        res[width - 32] = "5"
        res[width - 31] = "s"
        res[width - 30] = " "
    if width >= 64:
        res[width - 64] = " "
        res[width - 63] = "3"
        res[width - 62] = "0"
        res[width - 61] = "s"
        res[width - 60] = " "
    if width >= 124:
        res[width - 124] = " "
        res[width - 123] = "6"
        res[width - 122] = "0"
        res[width - 121] = "s"
        res[width - 120] = " "
    if width >= 245:
        res[width - 245] = " "
        res[width - 244] = "1"
        res[width - 243] = "2"
        res[width - 242] = "0"
        res[width - 241] = "s"
        res[width - 240] = " "
    return "".join(res)

def make_device_table(devices: List[Dict[str, Any]], hlo_map: Dict[int, Dict[int, str]], devices_per_chip: int) -> Table:
    table = Table(box=box.ROUNDED, expand=True, show_lines=True)
    
    term_width = console.width
    avail_col_width = int((term_width - 25) * 0.20)
    bar_len = max(0, avail_col_width - 7)
    
    table.add_column("TPU", justify="center", style="cyan", width=7)
    table.add_column("Memory Usage", justify="left", style="bold #FBBC05", ratio=20, no_wrap=True, overflow="ellipsis")
    table.add_column("TC Utilization", justify="left", style="bold #34A853", ratio=20, no_wrap=True, overflow="ellipsis")
    table.add_column("Duty Cycle", justify="left", style="bold #E066FF", ratio=20, no_wrap=True, overflow="ellipsis")
    table.add_column("Current HLO Op / Core", justify="left", style="white", no_wrap=True, overflow="ellipsis", ratio=40)
    
    for d in devices:
        mem_gb = d["memory_usage"] / (1024**3)
        total_gb = d["total_memory"] / (1024**3)
        mem_pct = d["memory_usage"] / d["total_memory"] * 100 if d["total_memory"] > 0 else 0
        
        mem_bar_len = round(mem_pct / (100 / bar_len))
        mem_bar_len = max(0, min(bar_len, mem_bar_len))
        mem_bar = "█" * mem_bar_len + " " * (bar_len - mem_bar_len)
        
        util_bar_len = round(d["tensorcore_util"] / (100 / bar_len))
        util_bar_len = max(0, min(bar_len, util_bar_len))
        util_bar = "█" * util_bar_len + " " * (bar_len - util_bar_len)
        
        dc_bar_len = round(d["duty_cycle"] / (100 / bar_len))
        dc_bar_len = max(0, min(bar_len, dc_bar_len))
        dc_bar = "█" * dc_bar_len + " " * (bar_len - dc_bar_len)
        
        hlo_dict = hlo_map.get(d["id"], {})
        hlo_lines = []
        
        has_real_hlo = any(loc and loc != "N/A" for loc in hlo_dict.values())
        
        if not has_real_hlo and d["id"] != 0:
            if d["tensorcore_util"] > 10 and 0 in hlo_map:
                hlo_dict = hlo_map[0]
                
        for i in range(devices_per_chip):
            label = f"TC{i}"
            loc = hlo_dict.get(label, "N/A")
            hlo_lines.append(f"{label}: {loc}")
            
        for label in sorted(hlo_dict.keys()):
            if not label.startswith("TC"):
                loc = hlo_dict[label]
                if loc and loc != "N/A":
                    hlo_lines.append(f"{label}: {loc}")
                        
        hlo_op = "\n".join(hlo_lines) if hlo_lines else "N/A"

        if term_width < 110:
            mem_text = f"MEM: [{mem_bar}]\n{mem_pct:.0f}% ({mem_gb:.1f} GiB)"
            util_text = f"UTL: [{util_bar}]\n{d['tensorcore_util']:.0f}%"
            dc_text = f"DC:  [{dc_bar}]\n{d['duty_cycle']:.0f}%"
        else:
            mem_text = f"MEM: [{mem_bar}]\n{mem_pct:.1f}% ({mem_gb:.1f}/{total_gb:.1f} GiB)"
            util_text = f"UTL: [{util_bar}]\n{d['tensorcore_util']:.0f}%"
            dc_text = f"DC:  [{dc_bar}]\n{d['duty_cycle']:.0f}%"
            
        table.add_row(
            f"TPU {d['id']}",
            mem_text,
            util_text,
            dc_text,
            hlo_op
        )
    return table

def make_process_table(processes: List[Dict[str, Any]]) -> Table:
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("PID", justify="right", style="cyan", width=10)
    table.add_column("Process", justify="left", style="green", ratio=1)
    table.add_column("Device", justify="center", style="yellow", width=10)
    table.add_column("%CPU", justify="right", style="blue", width=7)
    table.add_column("RAM", justify="right", style="magenta", width=14)
    
    cpu_count = psutil.cpu_count() or 1
    
    for p in processes:
        mem_gb = p.get("memory", 0) / (1024**3)
        cpu_val = p.get("cpu", 0.0)
        scaled_cpu = cpu_val / cpu_count
        table.add_row(
            str(p["pid"]),
            p["name"],
            str(p["device"]),
            f"{scaled_cpu:.1f}%",
            f"{mem_gb:.1f} GiB" if mem_gb > 0 else "N/A"
        )
    return table

def make_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=4),
        Layout(name="devices", size=12),
        Layout(name="history", size=14),
        Layout(name="processes", ratio=1)
    )

    layout["history"].split_row(
        Layout(name="left_col", ratio=1),
        Layout(name="right_col", ratio=1)
    )
    layout["history"]["left_col"].split(
        Layout(name="cpu_graph"),
        Layout(name="ram_graph")
    )
    layout["history"]["right_col"].split(
        Layout(name="gpu_util_graph"),
        Layout(name="gpu_mem_graph")
    )

    return layout
