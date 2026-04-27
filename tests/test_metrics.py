import unittest
from tpu_top.metrics import MetricsCollector

class TestMetricsCollector(unittest.TestCase):

    def test_collect_metrics_mock(self):
        collector = MetricsCollector(use_mock=True)
        metrics = collector.collect_metrics()
        
        self.assertIn("cpu_usage", metrics)
        self.assertIn("ram_usage", metrics)
        self.assertIn("devices", metrics)
        self.assertIn("processes", metrics)
        
        self.assertEqual(len(metrics["devices"]), 4)
        self.assertEqual(len(metrics["processes"]), 4)
        
        # Check structure of a device
        dev = metrics["devices"][0]
        self.assertIn("id", dev)
        self.assertIn("tensorcore_util", dev)

if __name__ == '__main__':
    unittest.main()
