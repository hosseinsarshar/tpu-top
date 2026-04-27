from typing import List

class MetricsHistory:
    def __init__(self, max_len: int = 240):
        self.max_len = max_len
        self.cpu: List[float] = []
        self.ram: List[float] = []
        self.tpu_util: List[float] = []
        self.tpu_mem: List[float] = []
        self.tpu_duty_cycle: List[float] = []

    def append_cpu(self, val: float):
        self.cpu.append(val)
        if len(self.cpu) > self.max_len:
            self.cpu.pop(0)

    def append_ram(self, val: float):
        self.ram.append(val)
        if len(self.ram) > self.max_len:
            self.ram.pop(0)

    def append_tpu_util(self, val: float):
        self.tpu_util.append(val)
        if len(self.tpu_util) > self.max_len:
            self.tpu_util.pop(0)

    def append_tpu_mem(self, val: float):
        self.tpu_mem.append(val)
        if len(self.tpu_mem) > self.max_len:
            self.tpu_mem.pop(0)

    def append_tpu_duty_cycle(self, val: float):
        self.tpu_duty_cycle.append(val)
        if len(self.tpu_duty_cycle) > self.max_len:
            self.tpu_duty_cycle.pop(0)
