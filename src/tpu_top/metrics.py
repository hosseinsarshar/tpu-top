import os
import time
import random
from typing import List, Dict, Any, Optional
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

# Python flags that take a value as the next argument.
_PYTHON_FLAGS_WITH_VALUE = {"-W", "-X", "--check-hash-based-pycs"}


def python_target(cmdline: List[str]) -> Optional[str]:
    """Return the script or module a python command line runs, if any."""
    args = cmdline[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-m" and i + 1 < len(args):
            return args[i + 1]
        if arg == "-c":
            return None
        if arg in _PYTHON_FLAGS_WITH_VALUE:
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        return os.path.basename(arg)
    return None


class _MockTPU:
    """Smooth, plausible fake metrics, used when no TPU is available."""

    NUM_CHIPS = 4
    CORES_PER_CHIP = 2
    HBM_PER_CHIP = 95 * 1024**3
    COMPUTATION = "main.204_spmd"
    OPS = (
        "fusion.1824",
        "convolution.311",
        "dot_general.92",
        "custom-call.18",
        "all-reduce.47",
        "reduce-scatter.12",
        "all-gather.33",
        "copy-start.6",
    )

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.tick = 0
        self.duty = [93.0] * self.NUM_CHIPS
        self.mxu = [0.55] * self.NUM_CHIPS
        self.hbm = [0.86 + 0.01 * i for i in range(self.NUM_CHIPS)]
        self.cpu = 18.0
        self.ram = 41.0
        self.stall = 0
        self.user = "root"

    def _walk(self, value: float, target: float, speed: float, noise: float, low: float, high: float) -> float:
        value += speed * (target - value) + self.rng.gauss(0.0, noise)
        return min(max(value, low), high)

    def step(self) -> None:
        self.tick += 1
        # Every so often the input pipeline stalls and the TPUs go quiet.
        if self.stall == 0 and self.rng.random() < 0.02:
            self.stall = self.rng.randint(4, 10)
        stalled = self.stall > 0
        if stalled:
            self.stall -= 1
        for i in range(self.NUM_CHIPS):
            self.duty[i] = self._walk(self.duty[i], 25.0 if stalled else 94.0, 0.35, 2.5, 0.0, 100.0)
            self.mxu[i] = self._walk(self.mxu[i], 0.58, 0.15, 0.04, 0.2, 0.9)
            self.hbm[i] = self._walk(self.hbm[i], 0.86 + 0.01 * i, 0.05, 0.002, 0.5, 0.99)
        self.cpu = self._walk(self.cpu, 45.0 if stalled else 16.0, 0.3, 2.0, 1.0, 100.0)
        self.ram = self._walk(self.ram, 41.0, 0.1, 0.15, 5.0, 95.0)

    def devices(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": i,
                "memory_usage": int(self.hbm[i] * self.HBM_PER_CHIP),
                "total_memory": self.HBM_PER_CHIP,
                "duty_cycle": self.duty[i],
                "temperature": 0,
                "tensorcore_util": self.duty[i] * self.mxu[i],
            }
            for i in range(self.NUM_CHIPS)
        ]

    def hlo_map(self) -> Dict[int, Dict[str, str]]:
        hlo_map: Dict[int, Dict[str, str]] = {}
        for chip in range(self.NUM_CHIPS):
            cores = {}
            for core in range(self.CORES_PER_CHIP):
                if self.duty[chip] < 30:
                    cores[f"TC{core}"] = "N/A"
                    continue
                op = self.OPS[(self.tick // 3 + core * 3 + chip // 2) % len(self.OPS)]
                cores[f"TC{core}"] = f"HLO: {op}; HLO computation: {self.COMPUTATION}"
            hlo_map[chip] = cores
        return hlo_map

    def processes(self) -> List[Dict[str, Any]]:
        train_cpu = 140.0 + self.rng.uniform(-20.0, 20.0)
        procs = [
            {
                "pid": 4242,
                "name": "train.py",
                "user": self.user,
                "command": "python3 train.py --model=llama-3-8b --global-batch-size=256",
                "device": f"TPU {chip}",
                "cpu": train_cpu,
                "memory": 26 * 1024**3 + 712 * 1024**2,
            }
            for chip in range(self.NUM_CHIPS)
        ]
        procs += [
            {
                "pid": 4317,
                "name": "input_pipeline",
                "user": self.user,
                "command": "python3 -m input_pipeline --workers=16 --prefetch=8",
                "device": "CPU",
                "cpu": max(0.0, 240.0 + self.rng.uniform(-40.0, 40.0)),
                "memory": 6 * 1024**3 + 205 * 1024**2,
            },
            {
                "pid": 4388,
                "name": "tensorboard.main",
                "user": self.user,
                "command": "python3 -m tensorboard.main --logdir=/tmp/logs --port=6006",
                "device": "CPU",
                "cpu": max(0.0, 6.0 + self.rng.uniform(-3.0, 3.0)),
                "memory": 1100 * 1024**2,
            },
            {
                "pid": 1206,
                "name": "gcsfuse",
                "user": self.user,
                "command": "gcsfuse --implicit-dirs my-training-data /gcs/data",
                "device": "CPU",
                "cpu": max(0.0, 12.0 + self.rng.uniform(-4.0, 4.0)),
                "memory": 310 * 1024**2,
            },
        ]
        return procs

    def ram_usage(self) -> Dict[str, float]:
        total = 1000 * 1024**3
        return {"percent": self.ram, "total": total, "used": total * self.ram / 100}


class MetricsCollector:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._prev_cpu_times = None
        self._use_new_chip_usage = False
        self.process_cache = {}
        self._mock = _MockTPU() if use_mock else None

    def get_cpu_usage(self) -> float:
        """Read CPU usage using psutil."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return 0.0

    def get_ram_usage(self) -> Dict[str, float]:
        """Read RAM usage using psutil."""
        try:
            mem = psutil.virtual_memory()
            return {
                "total": mem.total,
                "used": mem.used,
                "percent": mem.percent
            }
        except Exception as e:
            logger.error(f"Failed to get RAM usage: {e}")
            return {"total": 0.0, "used": 0.0, "percent": 0.0}

    def get_process_name(self, pid: int) -> str:
        """Get process name from PID."""
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            if "python" in name.lower():
                target = python_target(proc.cmdline())
                if target:
                    return target
            return name
        except Exception as e:
            logger.debug(f"Failed to get process name for PID {pid} via psutil: {e}")
            try:
                with open(f"/proc/{pid}/comm", "r") as f:
                    return f.read().strip()
            except Exception as e2:
                logger.debug(f"Failed to get process name for PID {pid} via /proc: {e2}")
                return "unknown"

    def get_process_details(self, pid: int) -> Dict[str, str]:
        """Get the owner and full command line of a process."""
        details = {"user": "", "command": ""}
        try:
            entry = self.process_cache.get(pid)
            proc = entry["proc"] if entry else psutil.Process(pid)
            details["command"] = " ".join(proc.cmdline()).strip()
            try:
                details["user"] = proc.username()
            except (KeyError, psutil.Error):
                details["user"] = str(proc.uids().real)
        except (psutil.Error, OSError) as e:
            logger.debug(f"Failed to get details for PID {pid}: {e}")
        return details

    def get_mock_metrics(self) -> List[Dict[str, Any]]:
        """Generate mock metrics for demonstration."""
        if self._mock is None:
            self._mock = _MockTPU()
        return self._mock.devices()

    def get_mock_processes(self) -> List[Dict[str, Any]]:
        """Generate mock processes."""
        if self._mock is None:
            self._mock = _MockTPU()
        return self._mock.processes()

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all metrics."""
        hlo_map = {}
        devices = []
        processes = []
        
        if self.use_mock:
            if self._mock is None:
                self._mock = _MockTPU()
            self._mock.step()
            devices = self.get_mock_metrics()
            processes = self.get_mock_processes()
            hlo_map = self._mock.hlo_map()
            cpu_usage = self._mock.cpu
            ram_usage = self._mock.ram_usage()
            devices_per_chip = self._mock.CORES_PER_CHIP
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
            
            usages = self._get_chip_usage(chip_type)
            tensorcore_util_data = self._get_tensorcore_util()
            hlo_map = self._get_hlo_map()
            
            devices = self._build_devices_list(num_chips, devices_per_chip, usages, tensorcore_util_data)
            processes = self._get_processes_metrics()

        return {
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "devices": devices,
            "processes": processes,
            "hlo_map": hlo_map,
            "devices_per_chip": devices_per_chip
        }

    def _get_chip_usage(self, chip_type) -> List[Any]:
        try:
            if self._use_new_chip_usage and hasattr(metrics, "get_chip_usage_new"):
                return metrics.get_chip_usage_new(chip_type)
            try:
                return metrics.get_chip_usage(chip_type)
            except AssertionError:
                if hasattr(metrics, "get_chip_usage_new"):
                    self._use_new_chip_usage = True
                    return metrics.get_chip_usage_new(chip_type)
        except Exception as e:
            logger.warning(f"Failed to get chip usage: {e}")
        return []

    def _get_tensorcore_util(self) -> List[Any]:
        tensorcore_util_data = []
        if libtpu_sdk:
            try:
                monitoring_module = getattr(libtpu_sdk, "tpumonitoring", getattr(libtpu_sdk, "monitoring", None))
                if monitoring_module:
                    tensorcore_util_data = monitoring_module.get_metric("tensorcore_util").data()
            except Exception as e:
                logger.warning(f"Failed to get tensorcore util data: {e}")
        return tensorcore_util_data

    def _get_hlo_map(self) -> Dict[int, Dict[str, str]]:
        hlo_map = {}
        try:
            core_states = metrics.get_tpuz_info(include_hlo_info=True)
            for cs in core_states:
                if cs.sequencer_states:
                    raw_loc = cs.sequencer_states[0].hlo_location
                    loc = raw_loc if raw_loc and raw_loc != "no HLO mapping" else "N/A"
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
            if hlo_map and 0 not in hlo_map:
                hlo_map = {idx: hlo_map[cid] for idx, cid in enumerate(sorted(hlo_map))}
        except Exception as e:
            logger.warning(f"Failed to get HLO info: {e}")
        return hlo_map

    def _build_devices_list(self, num_chips, devices_per_chip, usages, tensorcore_util_data) -> List[Dict[str, Any]]:
        devices = []
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
        return devices

    def _get_process_cpu_mem(self, pid: int) -> tuple[float, int]:
        try:
            if pid not in self.process_cache:
                proc = psutil.Process(pid)
                self.process_cache[pid] = {
                    'proc': proc,
                    'prev_time': time.time(),
                    'prev_cpu': sum(proc.cpu_times()[:2]),
                    'last_pct': 0.0
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
                    cache_entry['last_pct'] = cpu_pct
                else:
                    # Asked again within the same sample (e.g. one PID that
                    # owns several chips): reuse the last reading.
                    cpu_pct = cache_entry.get('last_pct', 0.0)
                    
            mem_info = self.process_cache[pid]['proc'].memory_info()
            mem_rss = mem_info.rss if mem_info else 0
            return cpu_pct, mem_rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0

    def _get_processes_metrics(self) -> List[Dict[str, Any]]:
        processes = []
        owners = device.get_chip_owners()
        sorted_paths = sorted(owners.keys())
        path_to_chip_idx = {path: idx for idx, path in enumerate(sorted_paths)}
        
        tpu_pids = set(owners.values())
        seen_pids = set(tpu_pids)
        details_cache: Dict[int, Dict[str, Any]] = {}
        
        for dev_path, pid in owners.items():
            chip_idx = path_to_chip_idx.get(dev_path, -1)
            cpu_pct, mem_rss = self._get_process_cpu_mem(pid)
            if pid not in details_cache:
                details_cache[pid] = {"name": self.get_process_name(pid), **self.get_process_details(pid)}
                
            processes.append({
                "pid": pid,
                **details_cache[pid],
                "device": f"TPU {chip_idx}",
                "cpu": cpu_pct,
                "memory": mem_rss
            })
            
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    pid = proc.info['pid']
                    seen_pids.add(pid)
                    if pid in tpu_pids:
                        continue
                        
                    cpu_pct, mem_rss = self._get_process_cpu_mem(pid)
                        
                    if cpu_pct and cpu_pct > 1.0:
                        processes.append({
                            "pid": pid,
                            "name": self.get_process_name(pid),
                            **self.get_process_details(pid),
                            "device": "CPU",
                            "cpu": cpu_pct,
                            "memory": mem_rss
                        })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error iterating processes: {e}")
        else:
            # Forget processes that have exited so the cache doesn't grow forever.
            for pid in list(self.process_cache):
                if pid not in seen_pids:
                    del self.process_cache[pid]
            
        return processes
