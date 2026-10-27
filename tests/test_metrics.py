import os
import unittest
from types import SimpleNamespace
from unittest import mock

import tpu_top.metrics as metrics_module
from tpu_top.metrics import MetricsCollector, python_target

GIB = 1024**3


class TestCollectorWithFakeTpuInfo(unittest.TestCase):
    """Runs the real (non-mock) collection path against a fake tpu-info."""

    def test_collect_metrics(self):
        chip_type = SimpleNamespace(value=SimpleNamespace(devices_per_chip=2, name="7x"))
        me = os.getpid()
        fake_device = SimpleNamespace(
            get_local_chips=lambda: (chip_type, 8),
            get_chip_owners=lambda: {"/dev/vfio/1": me, "/dev/vfio/0": me},
        )
        usage = SimpleNamespace(memory_usage=10 * GIB, total_memory=96 * GIB, duty_cycle_pct=97.0)
        core = SimpleNamespace(
            chip_id=0,
            core_type="TENSOR_CORE",
            core_on_chip_index=1,
            sequencer_states=[SimpleNamespace(hlo_location="HLO: fusion.1; HLO computation: main")],
        )
        fake_metrics = SimpleNamespace(
            get_chip_usage=lambda _chip_type: [usage] * 4,
            get_tpuz_info=lambda include_hlo_info: [core],
        )
        with mock.patch.object(metrics_module, "HAS_TPU_INFO", True), \
                mock.patch.object(metrics_module, "device", fake_device, create=True), \
                mock.patch.object(metrics_module, "metrics", fake_metrics, create=True), \
                mock.patch.object(metrics_module, "libtpu_sdk", None):
            collector = MetricsCollector(use_mock=False)
            collector.collect_metrics()
            data = collector.collect_metrics()

        self.assertEqual(data["devices_per_chip"], 2)
        self.assertEqual(len(data["devices"]), 4)
        self.assertEqual(data["devices"][0]["duty_cycle"], 97.0)
        self.assertEqual(data["hlo_map"], {0: {"TC1": "HLO: fusion.1; HLO computation: main"}})

        tpu_procs = [p for p in data["processes"] if p["device"].startswith("TPU")]
        self.assertEqual(sorted(p["device"] for p in tpu_procs), ["TPU 0", "TPU 1"])
        for proc in tpu_procs:
            self.assertEqual(proc["pid"], me)
            self.assertTrue(proc["command"])
            self.assertTrue(proc["user"])
        # Both entries for the PID report the same CPU reading.
        self.assertEqual(tpu_procs[0]["cpu"], tpu_procs[1]["cpu"])
        # Only live processes stay cached.
        self.assertIn(me, collector.process_cache)

class TestMetricsCollector(unittest.TestCase):

    def test_collect_metrics_mock(self):
        collector = MetricsCollector(use_mock=True)
        metrics = collector.collect_metrics()
        
        self.assertIn("cpu_usage", metrics)
        self.assertIn("ram_usage", metrics)
        self.assertIn("devices", metrics)
        self.assertIn("processes", metrics)
        self.assertIn("hlo_map", metrics)
        
        self.assertEqual(len(metrics["devices"]), 4)

        # One process owns every chip; a few more are busy on the CPU.
        tpu_procs = [p for p in metrics["processes"] if p["device"].startswith("TPU")]
        cpu_procs = [p for p in metrics["processes"] if p["device"] == "CPU"]
        self.assertEqual(len(tpu_procs), 4)
        self.assertEqual(len({p["pid"] for p in tpu_procs}), 1)
        self.assertGreater(len(cpu_procs), 0)
        for proc in metrics["processes"]:
            self.assertIn("user", proc)
            self.assertIn("command", proc)
        
        # Check structure of a device
        dev = metrics["devices"][0]
        self.assertIn("id", dev)
        self.assertIn("tensorcore_util", dev)

    def test_mock_metrics_stay_in_range(self):
        collector = MetricsCollector(use_mock=True)
        for _ in range(200):
            metrics = collector.collect_metrics()
            self.assertTrue(0.0 <= metrics["cpu_usage"] <= 100.0)
            for dev in metrics["devices"]:
                self.assertTrue(0.0 <= dev["duty_cycle"] <= 100.0)
                self.assertTrue(0.0 <= dev["tensorcore_util"] <= dev["duty_cycle"] + 1e-9)
                self.assertLessEqual(dev["memory_usage"], dev["total_memory"])


class TestPythonTarget(unittest.TestCase):

    def test_script(self):
        self.assertEqual(python_target(["python3", "-u", "/srv/train.py", "--steps=10"]), "train.py")

    def test_module(self):
        self.assertEqual(python_target(["python", "-m", "vllm.entrypoints.openai.api_server"]),
                         "vllm.entrypoints.openai.api_server")

    def test_flag_with_value(self):
        self.assertEqual(python_target(["python", "-X", "dev", "main.py"]), "main.py")

    def test_no_target(self):
        self.assertIsNone(python_target(["python"]))
        self.assertIsNone(python_target(["python", "-c", "print(1)"]))

if __name__ == '__main__':
    unittest.main()
