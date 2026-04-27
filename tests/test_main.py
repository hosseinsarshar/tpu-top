import unittest
from tpu_top.ui import sparkline, vertical_bar_chart, make_timeline


class TestTpuTop(unittest.TestCase):

    def test_sparkline_empty(self):
        self.assertEqual(sparkline([], width=10), " " * 10)

    def test_sparkline_full(self):
        res = sparkline([100.0] * 10, width=10)
        self.assertEqual(res, "█" * 10)

    def test_sparkline_half(self):
        res = sparkline([50.0] * 10, width=10)
        # 50% should be around middle block
        self.assertTrue(all(c in "▄▅" for c in res))

    def test_vertical_bar_chart_empty(self):
        res = vertical_bar_chart([], width=10, height=3)
        self.assertEqual(len(res), 3)
        for line in res:
            self.assertEqual(line, " " * 10)

    def test_vertical_bar_chart_full(self):
        res = vertical_bar_chart([100.0] * 10, width=10, height=3)
        self.assertEqual(len(res), 3)
        for line in res:
            self.assertEqual(line, "█" * 10)

    def test_make_timeline(self):
        res = make_timeline(10)
        self.assertEqual(len(res), 10)
        self.assertTrue(res.startswith("-"))
        self.assertTrue(res.endswith("0s"))

if __name__ == '__main__':
    unittest.main()
