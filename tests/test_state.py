import unittest
from tpu_top.state import MetricsHistory

class TestMetricsHistory(unittest.TestCase):

    def test_append_cpu(self):
        history = MetricsHistory(max_len=3)
        history.append_cpu(10.0)
        history.append_cpu(20.0)
        history.append_cpu(30.0)
        self.assertEqual(history.cpu, [10.0, 20.0, 30.0])
        
        history.append_cpu(40.0)
        self.assertEqual(history.cpu, [20.0, 30.0, 40.0])

    def test_append_ram(self):
        history = MetricsHistory(max_len=2)
        history.append_ram(10.0)
        history.append_ram(20.0)
        history.append_ram(30.0)
        self.assertEqual(history.ram, [20.0, 30.0])

if __name__ == '__main__':
    unittest.main()
