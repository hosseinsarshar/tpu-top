import os
import sys
import time
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.layout import Layout
import psutil

from tpu_top.state import MetricsHistory
from tpu_top.metrics import MetricsCollector, HAS_TPU_INFO
from tpu_top.ui import console, make_layout, make_device_table, make_process_table, vertical_bar_chart, make_timeline

def main():
    # Redirect stderr to a file to prevent Go library from corrupting UI
    try:
        stderr_log = open("/tmp/tputop_stderr.log", "a")
        os.dup2(stderr_log.fileno(), 2)
    except Exception:
        pass
        
    layout = make_layout()

    use_mock = not HAS_TPU_INFO or os.environ.get("TPU_TOP_MOCK") == "1"
    
    try:
        import libtpu
        libtpu_version = libtpu.__version__
    except ImportError:
        libtpu_version = "--"
        
    try:
        import tpu_info
        tpu_info_version = getattr(tpu_info, "__version__", None)
        if not tpu_info_version:
            import importlib.metadata
            try:
                tpu_info_version = importlib.metadata.version("tpu-info")
            except importlib.metadata.PackageNotFoundError:
                tpu_info_version = "--"
    except ImportError:
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

    try:
        import importlib.metadata
        tpu_top_version = importlib.metadata.version("tpu-top")
    except (importlib.metadata.PackageNotFoundError, ImportError):
        tpu_top_version = "0.1.6"

    # Fill header
    header_text = Text(f"TPU-TOP - TPU Utilization Monitor ({tpu_gen})", justify="center", style="bold green")
    header_text.append(f"\ntpu-top: {tpu_top_version} | libtpu: {libtpu_version} | tpu-info: {tpu_info_version}", style="dim")
    layout["header"].update(Panel(header_text, box=box.ROUNDED))

    iterations = int(os.environ.get("TPU_TOP_ITERATIONS", "0"))
    iter_count = 0
    
    history = MetricsHistory()
    collector = MetricsCollector(use_mock=use_mock)

    target_interval = 0.5

    with Live(layout, refresh_per_second=2, screen=True):
        while True:
            start_time = time.time()

            if iterations > 0 and iter_count >= iterations:
                break
            iter_count += 1
            
            # Collect metrics
            try:
                metrics_data = collector.collect_metrics()
            except RuntimeError as e:
                console.print(f"[red]Error:[/red] {e}")
                break

            cpu_usage = metrics_data["cpu_usage"]
            ram_usage = metrics_data["ram_usage"]
            devices = metrics_data["devices"]
            processes = metrics_data["processes"]
            hlo_map = metrics_data["hlo_map"]
            devices_per_chip = metrics_data["devices_per_chip"]

            # Update history
            history.append_cpu(cpu_usage)
            history.append_ram(ram_usage['percent'])

            avg_util = sum(d["tensorcore_util"] for d in devices) / len(devices) if devices else 0
            avg_duty_cycle = sum(d["duty_cycle"] for d in devices) / len(devices) if devices else 0
            avg_mem_pct = sum(d["memory_usage"] / d["total_memory"] * 100 for d in devices) / len(devices) if devices else 0
            
            history.append_tpu_util(avg_util)
            history.append_tpu_mem(avg_mem_pct)
            history.append_tpu_duty_cycle(avg_duty_cycle)
            
            # Filter and sort processes
            current_pid = os.getpid()
            processes = [p for p in processes if p["pid"] != current_pid]
            processes = [p for p in processes if "tpu-top" not in p["name"] and "tputop" not in p["name"]]
            
            tpu_procs = [p for p in processes if "TPU" in p["device"]]
            cpu_procs = [p for p in processes if p["device"] == "CPU"]
            
            tpu_procs.sort(key=lambda x: (int(x["device"].split()[1]), -x["memory"]))
            cpu_procs.sort(key=lambda x: x["memory"], reverse=True)
            
            processes = tpu_procs + cpu_procs

            # Calculate required height for devices table
            dev_height = 3 # Header and borders
            for d in devices:
                if devices_per_chip <= 1:
                    row_height = 1
                else:
                    hlo_dict = hlo_map.get(d["id"], {})
                    hlo_lines_count = 0
                    for core_idx in sorted(hlo_dict.keys()):
                        loc = hlo_dict[core_idx]
                        if loc and loc != "N/A":
                            hlo_lines_count += 1
                    if hlo_lines_count == 0 and d["id"] != 0 and d["duty_cycle"] > 10 and 0 in hlo_map:
                        hlo_lines_count = len(hlo_map[0])
                    
                    row_height = max(2, hlo_lines_count)
                
                dev_height += row_height + 1
            
            dev_height += 2 # Panel border
            
            # Priority layout logic
            header_h = 4
            dev_target_h = dev_height
            
            if console.height >= header_h + 21 + 5 + dev_target_h:
                is_grid = True
                history_h = 21
                proc_h = console.height - header_h - 21 - dev_target_h
            elif console.height >= header_h + 7 + 5 + dev_target_h:
                is_grid = False
                history_h = 7
                proc_h = console.height - header_h - 7 - dev_target_h
            elif console.height >= header_h + 7 + 3 + dev_target_h:
                is_grid = False
                history_h = 7
                proc_h = 3
            else:
                is_grid = False
                history_h = 7
                proc_h = 3
                dev_target_h = max(5, console.height - header_h - history_h - proc_h)

            layout["history"].size = history_h
            layout["processes"].size = proc_h
            layout["devices"].size = dev_target_h

            if is_grid:
                layout["history"].split(
                    Layout(name="grid", ratio=2),
                    Layout(name="duty_cycle_graph", ratio=1)
                )
                layout["history"]["grid"].split_row(
                    Layout(name="left_col", ratio=1),
                    Layout(name="right_col", ratio=1)
                )
                layout["history"]["grid"]["left_col"].split(
                    Layout(name="cpu_graph"),
                    Layout(name="ram_graph")
                )
                layout["history"]["grid"]["right_col"].split(
                    Layout(name="gpu_util_graph"),
                    Layout(name="gpu_mem_graph")
                )
                col_width = console.width // 2
            else:
                layout["history"].split_row(
                    Layout(name="cpu_graph"),
                    Layout(name="ram_graph"),
                    Layout(name="gpu_util_graph"),
                    Layout(name="gpu_mem_graph"),
                    Layout(name="duty_cycle_graph")
                )
                col_width = console.width // 5

            layout["devices"].update(Panel(make_device_table(devices, hlo_map, devices_per_chip), title="Devices", box=box.ROUNDED))
            
            graph_width = max(10, col_width - 6)
            timeline_str = make_timeline(graph_width)
            
            # CPU
            cpu_prefix = f"CPU: {cpu_usage:5.1f}%"
            cpu_bars = vertical_bar_chart(history.cpu, width=graph_width, height=3)
            cpu_text = Text(cpu_prefix, style="bold #4285F4") + Text("\n")
            for line in cpu_bars:
                cpu_text.append(line + "\n", style="#4285F4")
            cpu_text.append(timeline_str, style="dim")
            
            # RAM
            ram_prefix = f"RAM: {ram_usage['percent']:5.1f}%"
            ram_bars = vertical_bar_chart(history.ram, width=graph_width, height=3)
            ram_text = Text(ram_prefix, style="bold #EA4335") + Text("\n")
            for line in ram_bars:
                ram_text.append(line + "\n", style="#EA4335")
            ram_text.append(timeline_str, style="dim")
            
            # TPU HBM
            mem_prefix = f"HBM: {avg_mem_pct:5.1f}%"
            mem_bars = vertical_bar_chart(history.tpu_mem, width=graph_width, height=3)
            mem_text = Text(mem_prefix, style="bold #FBBC05") + Text("\n")
            for line in mem_bars:
                mem_text.append(line + "\n", style="#FBBC05")
            mem_text.append(timeline_str, style="dim")
            
            # TPU Util
            util_prefix = f"UTL: {avg_util:5.1f}%"
            util_bars = vertical_bar_chart(history.tpu_util, width=graph_width, height=3)
            util_text = Text(util_prefix, style="bold #34A853") + Text("\n")
            for line in util_bars:
                util_text.append(line + "\n", style="#34A853")
            util_text.append(timeline_str, style="dim")

            if is_grid:
                dc_graph_width = max(10, console.width - 6)
                dc_timeline_str = make_timeline(dc_graph_width)
            else:
                dc_graph_width = graph_width
                dc_timeline_str = timeline_str
                
            # TPU Duty Cycle
            dc_prefix = f"DC: {avg_duty_cycle:5.1f}%"
            dc_bars = vertical_bar_chart(history.tpu_duty_cycle, width=dc_graph_width, height=3)
            dc_text = Text(dc_prefix, style="bold #E066FF") + Text("\n")
            for line in dc_bars:
                dc_text.append(line + "\n", style="#E066FF")
            dc_text.append(dc_timeline_str, style="dim")
            
            if is_grid:
                layout["history"]["grid"]["left_col"]["cpu_graph"].update(Panel(cpu_text, title="CPU Activity", box=box.ROUNDED, border_style="#4285F4"))
                layout["history"]["grid"]["left_col"]["ram_graph"].update(Panel(ram_text, title="RAM Activity", box=box.ROUNDED, border_style="#EA4335"))
                layout["history"]["grid"]["right_col"]["gpu_util_graph"].update(Panel(util_text, title="AVG TPU (TC) UTL", box=box.ROUNDED, border_style="#34A853"))
                layout["history"]["grid"]["right_col"]["gpu_mem_graph"].update(Panel(mem_text, title="AVG TPU Mem", box=box.ROUNDED, border_style="#FBBC05"))
                layout["history"]["duty_cycle_graph"].update(Panel(dc_text, title="AVG TPU DUTY CYCLE", box=box.ROUNDED, border_style="#E066FF"))
            else:
                layout["history"]["cpu_graph"].update(Panel(cpu_text, title="CPU Activity", box=box.ROUNDED, border_style="#4285F4"))
                layout["history"]["ram_graph"].update(Panel(ram_text, title="RAM Activity", box=box.ROUNDED, border_style="#EA4335"))
                layout["history"]["gpu_util_graph"].update(Panel(util_text, title="AVG TPU UTL", box=box.ROUNDED, border_style="#34A853"))
                layout["history"]["gpu_mem_graph"].update(Panel(mem_text, title="AVG TPU MEM", box=box.ROUNDED, border_style="#FBBC05"))
                layout["history"]["duty_cycle_graph"].update(Panel(dc_text, title="AVG TPU DUTY CYCLE", box=box.ROUNDED, border_style="#E066FF"))

            layout["processes"].update(Panel(make_process_table(processes), title="Processes", box=box.ROUNDED))
            
            # Dynamic sleep to maintain 2Hz rate
            processing_time = time.time() - start_time
            sleep_time = max(0, target_interval - processing_time)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
