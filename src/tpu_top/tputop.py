import os
import sys
import time
import random
import select
import tty
import termios
from typing import List, Dict, Any

# Add site-packages to path if needed, but assuming running with correct python
# sys.path.append('/usr/local/google/home/hosseins/miniconda3/lib/python3.13/site-packages')

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich import box
from rich.text import Text
import grpc
import psutil


try:

    from tpu_info import metrics
    from tpu_info import device
    HAS_TPU_INFO = True
except ImportError:
    HAS_TPU_INFO = False

try:
    from libtpu import sdk as libtpu_sdk
except ImportError:
    libtpu_sdk = None


console = Console()

_prev_cpu_times = None

def get_cpu_usage() -> float:
    """Read CPU usage from /proc/stat."""
    global _prev_cpu_times
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
            if not line.startswith("cpu "):
                return 0.0
            parts = [int(x) for x in line.split()[1:]]
            idle = parts[3] + parts[4]
            total = sum(parts)
            
            if _prev_cpu_times is None:
                _prev_cpu_times = (idle, total)
                return 0.0
            
            prev_idle, prev_total = _prev_cpu_times
            idle_diff = idle - prev_idle
            total_diff = total - prev_total
            
            _prev_cpu_times = (idle, total)
            
            if total_diff == 0:
                return 0.0
            return (total_diff - idle_diff) / total_diff * 100
    except Exception:
        return 0.0

def get_ram_usage() -> Dict[str, float]:
    """Read RAM usage from /proc/meminfo."""
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].split()[0])
        
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        used = total - available
        return {
            "total": total * 1024,
            "used": used * 1024,
            "percent": (used / total * 100) if total > 0 else 0.0
        }
    except Exception:
        return {"total": 0, "used": 0, "percent": 0.0}


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

cpu_history = []
ram_history = []
tpu_util_history = []
tpu_mem_history = []



def get_process_name(pid: int) -> str:
    """Get process name from PID."""
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        if "python" in name.lower():
            cmdline = proc.cmdline()
            if len(cmdline) > 1:
                import os
                return os.path.basename(cmdline[1])
        return name
    except Exception:
        try:
            with open(f"/proc/{pid}/comm", "r") as f:
                return f.read().strip()
        except Exception:
            return "unknown"


def get_mock_metrics() -> List[Dict[str, Any]]:
    """Generate mock metrics for demonstration."""
    devices = []
    for i in range(4):
        devices.append({
            "id": i,
            "memory_usage": random.randint(10, 32) * 1024 * 1024 * 1024,
            "total_memory": 32 * 1024 * 1024 * 1024,
            "duty_cycle": random.random() * 100,
            "temperature": random.randint(30, 70)
        })
    return devices

def get_mock_processes() -> List[Dict[str, Any]]:
    """Generate mock processes."""
    procs = []
    names = ["train.py", "eval.py", "data_loader", "python"]
    for i in range(3):
        procs.append({
            "pid": random.randint(1000, 9999),
            "name": random.choice(names),
            "device": f"TPU {random.randint(0, 3)}",
            "cpu": random.random() * 100,
            "memory": random.randint(1, 8) * 1024 * 1024 * 1024
        })
    # Add a CPU process
    procs.append({
        "pid": random.randint(1000, 9999),
        "name": "top",
        "device": "CPU",
        "cpu": 15.0,
        "memory": 100 * 1024 * 1024
    })
    return procs


def make_device_table(devices: List[Dict[str, Any]], hlo_map: Dict[int, Dict[int, str]]) -> Table:
    table = Table(box=box.ROUNDED, expand=True, show_lines=True)
    
    term_width = console.width
    bar_len = 10 if term_width >= 120 else 5
    
    table.add_column("TPU", justify="center", style="cyan", ratio=10)
    table.add_column("Memory Usage", justify="left", style="magenta", ratio=25, no_wrap=True, overflow="ellipsis")
    table.add_column("Utilization", justify="left", style="green", ratio=25, no_wrap=True, overflow="ellipsis")
    table.add_column("Current HLO Op / Core", justify="left", style="white", no_wrap=True, overflow="ellipsis", ratio=40)
    
    for d in devices:
        mem_gb = d["memory_usage"] / (1024**3)
        total_gb = d["total_memory"] / (1024**3)
        mem_pct = d["memory_usage"] / d["total_memory"] * 100 if d["total_memory"] > 0 else 0
        
        mem_bar_len = int(mem_pct / (100 / bar_len))
        if mem_bar_len == 0 and mem_pct > 0:
            mem_bar_len = 1
        mem_bar = "█" * mem_bar_len + " " * (bar_len - mem_bar_len)
        
        util_bar_len = int(d["duty_cycle"] / (100 / bar_len))
        if util_bar_len == 0 and d["duty_cycle"] > 0:
            util_bar_len = 1
        util_bar = "█" * util_bar_len + " " * (bar_len - util_bar_len)
        
        hlo_dict = hlo_map.get(d["id"], {})
        hlo_lines = []
        for core_idx in sorted(hlo_dict.keys()):
            loc = hlo_dict[core_idx]
            if loc and loc != "N/A":
                hlo_lines.append(f"C{core_idx}: {loc}")
        
        if not hlo_lines and d["id"] != 0:
            # Fallback to TPU 0 if this TPU is utilized but has no HLO info
            if d["duty_cycle"] > 10 and 0 in hlo_map:
                hlo_dict_0 = hlo_map[0]
                for core_idx in sorted(hlo_dict_0.keys()):
                    loc = hlo_dict_0[core_idx]
                    if loc and loc != "N/A":
                        hlo_lines.append(f"C{core_idx}*: {loc}")
                        
        hlo_op = "\n".join(hlo_lines) if hlo_lines else "N/A"

        if term_width < 110:
            mem_text = f"MEM: [{mem_bar}]\n{mem_pct:.0f}% ({mem_gb:.1f} GiB)"
            util_text = f"UTL: [{util_bar}]\n{d['duty_cycle']:.0f}%"
        else:
            mem_text = f"MEM: [{mem_bar}]\n{mem_pct:.1f}% ({mem_gb:.1f}/{total_gb:.1f} GiB)"
            util_text = f"UTL: [{util_bar}]\n{d['duty_cycle']:.1f}%"
            
        table.add_row(
            f"TPU {d['id']}",
            mem_text,
            util_text,
            hlo_op
        )
    return table


def make_process_table(processes: List[Dict[str, Any]]) -> Table:
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("PID", justify="right", style="cyan")
    table.add_column("Process", justify="left", style="green")
    table.add_column("Device", justify="center", style="yellow")
    table.add_column("%CPU", justify="right", style="blue")
    table.add_column("Memory", justify="right", style="magenta")
    
    for p in processes:
        mem_gb = p.get("memory", 0) / (1024**3)
        cpu_val = p.get("cpu", 0.0)
        # Scale by CPU count to show percentage of total machine
        cpu_count = psutil.cpu_count() or 1
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
        Layout(name="history", size=15),
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




def main():
    # Redirect stderr to a file to prevent Go library from corrupting UI
    try:
        import os
        stderr_log = open("/tmp/tputop_stderr.log", "a")
        os.dup2(stderr_log.fileno(), 2)
    except Exception:
        pass
        
    layout = make_layout()

    
    use_mock = not HAS_TPU_INFO or os.environ.get("TPU_TOP_MOCK") == "1"
    
    try:
        import jax
        jax_version = jax.__version__
    except ImportError:
        jax_version = "--"
        
    try:
        import libtpu
        libtpu_version = libtpu.__version__
    except ImportError:
        libtpu_version = "--"
        
    tpu_info_version = "--"
    
    if use_mock:
        tpu_gen = "TPU Mock"
    else:
        try:
            from tpu_info import device
            chip_type, _ = device.get_local_chips()
            tpu_gen = f"TPU {chip_type.value.name}"
        except Exception:
            tpu_gen = "TPU Unknown"

    # Fill header
    header_text = Text(f"TPU-TOP - TPU Utilization Monitor ({tpu_gen})", justify="center", style="bold green")
    header_text.append(f"\nJAX: {jax_version} | libtpu: {libtpu_version} | tpu-info: {tpu_info_version}", style="dim")
    layout["header"].update(Panel(header_text, box=box.ROUNDED))

    
    iterations = int(os.environ.get("TPU_TOP_ITERATIONS", "0"))
    iter_count = 0
    
    with Live(layout, refresh_per_second=2, screen=True):
        while True:
            if iterations > 0 and iter_count >= iterations:
                break
            iter_count += 1
            
            cpu_usage = get_cpu_usage()
            ram_usage = get_ram_usage()
            
            cpu_history.append(cpu_usage)
            ram_history.append(ram_usage['percent'])

            
            hlo_map = {}
            if use_mock:
                print("Mock devices is enabled")
                devices = get_mock_metrics()
                processes = get_mock_processes()
            else:
                chip_type, count = device.get_local_chips()
                if count == 0 or not chip_type:
                    raise RuntimeError("No TPU chips found")
                
                usages = []
                try:
                    try:
                        usages = metrics.get_chip_usage(chip_type)
                    except AssertionError:
                        if hasattr(metrics, "get_chip_usage_new"):
                            usages = metrics.get_chip_usage_new(chip_type)
                except Exception:
                    usages = []


                
                # Get TensorCore util as fallback or override
                tensorcore_util_data = []
                if libtpu_sdk:
                    try:
                        monitoring_module = getattr(libtpu_sdk, "tpumonitoring", getattr(libtpu_sdk, "monitoring", None))
                        if monitoring_module:
                            tensorcore_util_data = monitoring_module.get_metric("tensorcore_util").data()
                    except Exception:
                        pass
                devices = []
                num_devices = count * chip_type.value.devices_per_chip if chip_type else count
                devices = []
                devices_per_chip = chip_type.value.devices_per_chip if chip_type else 1
                num_chips = count // devices_per_chip if chip_type and count >= devices_per_chip else count
                
                try:
                    core_states = metrics.get_tpuz_info(include_hlo_info=True)
                    for cs in core_states:
                        if cs.sequencer_states:
                            loc = cs.sequencer_states[0].hlo_location or "N/A"
                            if cs.chip_id not in hlo_map:
                                hlo_map[cs.chip_id] = {}
                            hlo_map[cs.chip_id][cs.core_on_chip_index] = loc
                except Exception:
                    pass
                
                for i in range(num_chips):
                    u = usages[i] if usages and i < len(usages) else None

                    
                    mem_usage = u.memory_usage if u else 0
                    total_mem = u.total_memory if u else 1
                    
                    # Aggregate TensorCore util for this chip
                    util = u.duty_cycle_pct if u else 0.0
                    if tensorcore_util_data:
                        core_utils = []
                        for c in range(devices_per_chip):
                            core_idx = i * devices_per_chip + c
                            if core_idx < len(tensorcore_util_data):
                                u_val = tensorcore_util_data[core_idx]
                                if isinstance(u_val, str):
                                    u_val = u_val.replace("%", "")
                                core_utils.append(float(u_val))
                        if core_utils:
                            util = sum(core_utils) / len(core_utils)
                    
                    devices.append({
                        "id": i,
                        "memory_usage": mem_usage,
                        "total_memory": total_mem,
                        "duty_cycle": util,
                        "temperature": 0
                    })
                
                owners = device.get_chip_owners()
                # Map dev_path to chip index based on sorted order
                sorted_paths = sorted(owners.keys())
                path_to_chip_idx = {path: idx for idx, path in enumerate(sorted_paths)}
                
                tpu_pids = set(owners.values())
                processes = []
                
                # Add TPU processes first
                for dev_path, pid in owners.items():
                    chip_idx = path_to_chip_idx.get(dev_path, -1)
                    try:
                        proc = psutil.Process(pid)
                        cpu_pct = proc.cpu_percent(interval=None)
                        mem_info = proc.memory_info()
                        mem_rss = mem_info.rss if mem_info else 0
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        cpu_pct = 0.0
                        mem_rss = 0
                        
                    processes.append({
                        "pid": pid,
                        "name": get_process_name(pid),
                        "device": f"TPU {chip_idx}",
                        "cpu": cpu_pct,
                        "memory": mem_rss
                    })
                    
                # Add top CPU processes
                try:
                    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                        try:
                            pid = proc.info['pid']
                            if pid in tpu_pids:
                                continue # Already added as TPU process
                                
                            cpu_pct = proc.info['cpu_percent']
                            if cpu_pct and cpu_pct > 1.0: # Only show if > 1% CPU
                                mem_info = proc.info['memory_info']
                                mem_rss = mem_info.rss if mem_info else 0
                                processes.append({
                                    "pid": pid,
                                    "name": get_process_name(pid),
                                    "device": "CPU",
                                    "cpu": cpu_pct,
                                    "memory": mem_rss
                                })

                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except Exception:
                    pass

            # Calculate averages for history
            avg_util = sum(d["duty_cycle"] for d in devices) / len(devices) if devices else 0
            avg_mem_pct = sum(d["memory_usage"] / d["total_memory"] * 100 for d in devices) / len(devices) if devices else 0
            
            tpu_util_history.append(avg_util)
            tpu_mem_history.append(avg_mem_pct)
            
            # Limit history to 240 items (120s at 0.5s interval)
            if len(tpu_util_history) > 240:
                tpu_util_history.pop(0)
            if len(tpu_mem_history) > 240:
                tpu_mem_history.pop(0)
            if len(cpu_history) > 240:
                cpu_history.pop(0)
            
            # Calculate required height for devices table
            dev_height = 3 # Header and borders
            for d in devices:
                hlo_dict = hlo_map.get(d["id"], {})
                hlo_lines_count = 0
                for core_idx in sorted(hlo_dict.keys()):
                    loc = hlo_dict[core_idx]
                    if loc and loc != "N/A":
                        hlo_lines_count += 1
                if hlo_lines_count == 0 and d["id"] != 0 and d["duty_cycle"] > 10 and 0 in hlo_map:
                    hlo_lines_count = len(hlo_map[0])
                
                row_height = max(1, hlo_lines_count)
                dev_height += row_height + 1 # +1 for separator line
            
            dev_height += 2 # Panel border
            
            # Cap it to leave space for other panels
            max_allowed = max(5, console.height - 25)
            layout["devices"].size = min(dev_height, max_allowed)

            # Update layout
            layout["devices"].update(Panel(make_device_table(devices, hlo_map), title="Devices", box=box.ROUNDED))
            
            # Update graphs with dynamic width and timeline
            # Google Corp Colors
            # Blue: #4285F4, Red: #EA4335, Yellow: #FBBC05, Green: #34A853
            
            col_width = console.width // 2
            graph_width = max(10, col_width - 6) # Subtract borders only!
            
            timeline_str = make_timeline(graph_width)
            
            # CPU (Blue)
            cpu_prefix = f"CPU: {cpu_usage:5.1f}%"
            cpu_bars = vertical_bar_chart(cpu_history, width=graph_width, height=3)
            cpu_text = Text(cpu_prefix, style="bold #4285F4") + Text("\n")
            for line in cpu_bars:
                cpu_text.append(line + "\n", style="#4285F4")
            cpu_text.append(timeline_str, style="dim")
            
            # RAM (Red)
            ram_prefix = f"RAM: {ram_usage['percent']:5.1f}%"
            ram_bars = vertical_bar_chart(ram_history, width=graph_width, height=3)
            ram_text = Text(ram_prefix, style="bold #EA4335") + Text("\n")
            for line in ram_bars:
                ram_text.append(line + "\n", style="#EA4335")
            ram_text.append(timeline_str, style="dim")
            
            # TPU HBM (Yellow)
            mem_prefix = f"HBM: {avg_mem_pct:5.1f}%"
            mem_bars = vertical_bar_chart(tpu_mem_history, width=graph_width, height=3)
            mem_text = Text(mem_prefix, style="bold #FBBC05") + Text("\n")
            for line in mem_bars:
                mem_text.append(line + "\n", style="#FBBC05")
            mem_text.append(timeline_str, style="dim")
            
            # TPU Util (Green)
            util_prefix = f"UTL: {avg_util:5.1f}%"
            util_bars = vertical_bar_chart(tpu_util_history, width=graph_width, height=3)
            util_text = Text(util_prefix, style="bold #34A853") + Text("\n")
            for line in util_bars:
                util_text.append(line + "\n", style="#34A853")
            util_text.append(timeline_str, style="dim")

            
            layout["history"]["left_col"]["cpu_graph"].update(Panel(cpu_text, title="CPU Activity", box=box.ROUNDED, border_style="#4285F4"))
            layout["history"]["left_col"]["ram_graph"].update(Panel(ram_text, title="RAM Activity", box=box.ROUNDED, border_style="#EA4335"))
            layout["history"]["right_col"]["gpu_util_graph"].update(Panel(util_text, title="AVG TPU UTL", box=box.ROUNDED, border_style="#34A853"))
            layout["history"]["right_col"]["gpu_mem_graph"].update(Panel(mem_text, title="AVG TPU MEM", box=box.ROUNDED, border_style="#FBBC05"))

            
            layout["processes"].update(Panel(make_process_table(processes), title="Processes", box=box.ROUNDED))
            
            time.sleep(0.5)


if __name__ == "__main__":
    main()


