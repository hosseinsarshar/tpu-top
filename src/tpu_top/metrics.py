import os
import time
import random
from typing import List, Dict, Any
import psutil
import logging

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

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._prev_cpu_times = None
        self.process_cache = {}

    def get_cpu_usage(self) -> float:
        """Read CPU usage from /proc/stat."""
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
                if not line.startswith("cpu "):
                    return 0.0
                parts = [int(x) for x in line.split()[1:]]
                idle = parts[3] + parts[4]
                total = sum(parts)
                
                if self._prev_cpu_times is None:
                    self._prev_cpu_times = (idle, total)
                    return 0.0
                
                prev_idle, prev_total = self._prev_cpu_times
                idle_diff = idle - prev_idle
                total_diff = total - prev_total
                
                self._prev_cpu_times = (idle, total)
                
                if total_diff == 0:
                    return 0.0
                return (total_diff - idle_diff) / total_diff * 100
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return 0.0

    def get_ram_usage(self) -> Dict[str, float]:
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
        except Exception as e:
            logger.error(f"Failed to get RAM usage: {e}")
            return {"total": 0, "used": 0, "percent": 0.0}

    def get_process_name(self, pid: int) -> str:
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
        except Exception as e:
            logger.debug(f"Failed to get process name for PID {pid} via psutil: {e}")
            try:
                with open(f"/proc/{pid}/comm", "r") as f:
                    return f.read().strip()
            except Exception as e2:
                logger.debug(f"Failed to get process name for PID {pid} via /proc: {e2}")
                return "unknown"

    def get_mock_metrics(self) -> List[Dict[str, Any]]:
        """Generate mock metrics for demonstration."""
        devices = []
        for i in range(4):
            devices.append({
                "id": i,
                "memory_usage": random.randint(10, 32) * 1024 * 1024 * 1024,
                "total_memory": 32 * 1024 * 1024 * 1024,
                "duty_cycle": random.random() * 100,
                "temperature": random.randint(30, 70),
                "tensorcore_util": random.random() * 100
            })
        return devices

    def get_mock_processes(self) -> List[Dict[str, Any]]:
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

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all metrics."""
        hlo_map = {}
        devices = []
        processes = []
        
        if self.use_mock:
            devices = self.get_mock_metrics()
            processes = self.get_mock_processes()
            cpu_usage = random.uniform(40.0, 80.0)
            ram_usage = {"percent": random.uniform(60.0, 90.0), "total": 1000 * 1024**3, "used": 700 * 1024**3}
            devices_per_chip = 1
            num_chips = len(devices)
        else:
            cpu_usage = self.get_cpu_usage()
            ram_usage = self.get_ram_usage()
            if not HAS_TPU_INFO:
                raise RuntimeError("tpu-info is not available")
                
            chip_type, count = device.get_local_chips()
            if count == 0 or not chip_type:
                raise RuntimeError("No TPU chips found")
            
            devices_per_chip = chip_type.value.devices_per_chip if chip_type else 1
            num_chips = count // devices_per_chip if chip_type and count >= devices_per_chip else count
            
            usages = []
            try:
                try:
                    usages = metrics.get_chip_usage(chip_type)
                except AssertionError:
                    if hasattr(metrics, "get_chip_usage_new"):
                        usages = metrics.get_chip_usage_new(chip_type)
            except Exception as e:
                logger.warning(f"Failed to get chip usage: {e}")
                usages = []

            tensorcore_util_data = []
            if libtpu_sdk:
                try:
                    monitoring_module = getattr(libtpu_sdk, "tpumonitoring", getattr(libtpu_sdk, "monitoring", None))
                    if monitoring_module:
                        tensorcore_util_data = monitoring_module.get_metric("tensorcore_util").data()
                except Exception as e:
                    logger.warning(f"Failed to get tensorcore util data: {e}")
                    pass
                    
            try:
                core_states = metrics.get_tpuz_info(include_hlo_info=True)
                for cs in core_states:
                    if cs.sequencer_states:
                        loc = cs.sequencer_states[0].hlo_location or "N/A"
                        if cs.chip_id not in hlo_map:
                            hlo_map[cs.chip_id] = {}
                        
                        core_type = cs.core_type
                        if 'TENSOR_CORE' in core_type:
                            core_label = f"TC{cs.core_on_chip_index}"
                        elif 'SPARSE_CORE' in core_type:
                            core_label = f"SC{cs.core_on_chip_index}"
                        else:
                            core_label = f"C{cs.core_on_chip_index}"
                            
                        hlo_map[cs.chip_id][core_label] = loc
            except Exception as e:
                logger.warning(f"Failed to get HLO info: {e}")
                pass
                
            for i in range(num_chips):
                u = usages[i] if usages and i < len(usages) else None
                
                mem_usage = u.memory_usage if u else 0
                total_mem = u.total_memory if u else 1
                
                raw_duty_cycle = u.duty_cycle_pct if u else 0.0
                util = raw_duty_cycle
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
                    "duty_cycle": raw_duty_cycle,
                    "tensorcore_util": util,
                    "temperature": 0
                })
                
            owners = device.get_chip_owners()
            sorted_paths = sorted(owners.keys())
            path_to_chip_idx = {path: idx for idx, path in enumerate(sorted_paths)}
            
            tpu_pids = set(owners.values())
            
            for dev_path, pid in owners.items():
                chip_idx = path_to_chip_idx.get(dev_path, -1)
                try:
                    if pid not in self.process_cache:
                        proc = psutil.Process(pid)
                        self.process_cache[pid] = {
                            'proc': proc,
                            'prev_time': time.time(),
                            'prev_cpu': sum(proc.cpu_times()[:2])
                        }
                        cpu_pct = 0.0
                    else:
                        cache_entry = self.process_cache[pid]
                        curr_time = time.time()
                        curr_cpu = sum(cache_entry['proc'].cpu_times()[:2])
                        
                        time_diff = curr_time - cache_entry['prev_time']
                        cpu_diff = curr_cpu - cache_entry['prev_cpu']
                        
                        if time_diff > 0.1:
                            cpu_pct = (cpu_diff / time_diff) * 100
                            cache_entry['prev_time'] = curr_time
                            cache_entry['prev_cpu'] = curr_cpu
                        else:
                            cpu_pct = 0.0
                            
                    mem_info = self.process_cache[pid]['proc'].memory_info()
                    mem_rss = mem_info.rss if mem_info else 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu_pct = 0.0
                    mem_rss = 0
                    
                processes.append({
                    "pid": pid,
                    "name": self.get_process_name(pid),
                    "device": f"TPU {chip_idx}",
                    "cpu": cpu_pct,
                    "memory": mem_rss
                })
                
            try:
                for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                    try:
                        pid = proc.info['pid']
                        if pid in tpu_pids:
                            continue
                            
                        if pid not in self.process_cache:
                            proc_obj = psutil.Process(pid)
                            self.process_cache[pid] = {
                                'proc': proc_obj,
                                'prev_time': time.time(),
                                'prev_cpu': sum(proc_obj.cpu_times()[:2])
                            }
                            cpu_pct = 0.0
                        else:
                            cache_entry = self.process_cache[pid]
                            curr_time = time.time()
                            curr_cpu = sum(cache_entry['proc'].cpu_times()[:2])
                            
                            time_diff = curr_time - cache_entry['prev_time']
                            cpu_diff = curr_cpu - cache_entry['prev_cpu']
                            
                            if time_diff > 0.1:
                                cpu_pct = (cpu_diff / time_diff) * 100
                                cache_entry['prev_time'] = curr_time
                                cache_entry['prev_cpu'] = curr_cpu
                            else:
                                cpu_pct = 0.0
                            
                        if cpu_pct and cpu_pct > 1.0:
                            mem_info = proc.info['memory_info']
                            mem_rss = mem_info.rss if mem_info else 0
                            processes.append({
                                "pid": pid,
                                "name": self.get_process_name(pid),
                                "device": "CPU",
                                "cpu": cpu_pct,
                                "memory": mem_rss
                            })

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                logger.error(f"Error iterating processes: {e}")
                pass

        return {
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "devices": devices,
            "processes": processes,
            "hlo_map": hlo_map,
            "devices_per_chip": devices_per_chip
        }
