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

    def test_append_chip(self):
        history = MetricsHistory(max_len=2)
        for i in range(3):
            history.append_chip(1, util=i, duty_cycle=10 * i, mem_pct=100 * i)
        self.assertEqual(history.chip_util[1], [1, 2])
        self.assertEqual(history.chip_duty_cycle[1], [10, 20])
        self.assertEqual(history.chip_mem[1], [100, 200])
        self.assertNotIn(0, history.chip_util)

    def test_sample_period(self):
        history = MetricsHistory()
        self.assertIsNone(history.sample_period)
        history.tick(10.0)
        history.tick(10.5)
        self.assertAlmostEqual(history.sample_period, 0.5)
        history.tick(11.5)
        self.assertAlmostEqual(history.sample_period, 0.6)
        # A pause should not count as one long sample.
        history.reset_clock()
        history.tick(100.0)
        self.assertAlmostEqual(history.sample_period, 0.6)

if __name__ == '__main__':
    unittest.main()
