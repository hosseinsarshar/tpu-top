from typing import Dict, List, Optional


class MetricsHistory:
    def __init__(self, max_len: int = 720):
        self.max_len = max_len
        self.cpu: List[float] = []
        self.ram: List[float] = []
        self.tpu_util: List[float] = []
        self.tpu_mem: List[float] = []
        self.tpu_duty_cycle: List[float] = []
        # Per-chip series, keyed by chip id.
        self.chip_util: Dict[int, List[float]] = {}
        self.chip_duty_cycle: Dict[int, List[float]] = {}
        self.chip_mem: Dict[int, List[float]] = {}
        # Smoothed time between samples, in seconds.
        self.sample_period: Optional[float] = None
        self._last_tick: Optional[float] = None

    def _append(self, series: List[float], val: float):
        series.append(val)
        if len(series) > self.max_len:
            del series[0]

    def append_cpu(self, val: float):
        self._append(self.cpu, val)

    def append_ram(self, val: float):
        self._append(self.ram, val)

    def append_tpu_util(self, val: float):
        self._append(self.tpu_util, val)

    def append_tpu_mem(self, val: float):
        self._append(self.tpu_mem, val)

    def append_tpu_duty_cycle(self, val: float):
        self._append(self.tpu_duty_cycle, val)

    def append_chip(self, chip_id: int, util: float, duty_cycle: float, mem_pct: float):
        self._append(self.chip_util.setdefault(chip_id, []), util)
        self._append(self.chip_duty_cycle.setdefault(chip_id, []), duty_cycle)
        self._append(self.chip_mem.setdefault(chip_id, []), mem_pct)

    def tick(self, now: float):
        """Record the time of a new sample to track the real sampling period."""
        if self._last_tick is not None:
            delta = now - self._last_tick
            if delta > 0:
                if self.sample_period is None:
                    self.sample_period = delta
                else:
                    self.sample_period = 0.8 * self.sample_period + 0.2 * delta
        self._last_tick = now

    def reset_clock(self):
        """Forget the last sample time, e.g. after a pause."""
        self._last_tick = None
