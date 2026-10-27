import unittest

from tpu_top.tputop import TpuTopApp, parse_args
from tpu_top.widgets import HelpScreen, InfoScreen, MetricGraph, ProcessTable


class TestApp(unittest.IsolatedAsyncioTestCase):

    async def test_mock_dashboard(self):
        app = TpuTopApp(use_mock=True, interval=0.1)
        async with app.run_test(size=(160, 48)) as pilot:
            for _ in range(20):
                await pilot.pause(0.1)
                if len(app.history.cpu) >= 3:
                    break
            self.assertIsNotNone(app.last_data)
            self.assertGreaterEqual(len(app.history.cpu), 3)

            table = app.query_one(ProcessTable)
            self.assertEqual(table.row_count, 4)  # TPU owner merged into one row

            await pilot.press("s")
            self.assertEqual(table.sort_mode, "cpu")

            await pilot.press("g")
            self.assertEqual(app.query_one("#cpu-graph", MetricGraph).graph_style, "block")

            await pilot.press("p")
            self.assertTrue(app.paused)
            samples = len(app.history.cpu)
            await pilot.pause(0.4)
            self.assertEqual(len(app.history.cpu), samples)
            await pilot.press("p")
            self.assertFalse(app.paused)

            await pilot.press("question_mark")
            self.assertIsInstance(app.screen, HelpScreen)
            await pilot.press("escape")
            self.assertNotIsInstance(app.screen, HelpScreen)

            await pilot.press("i")
            for _ in range(100):
                await pilot.pause(0.1)
                if isinstance(app.screen, InfoScreen):
                    break
            self.assertIsInstance(app.screen, InfoScreen)
            await pilot.press("i")
            self.assertNotIsInstance(app.screen, InfoScreen)

    async def test_small_terminal(self):
        app = TpuTopApp(use_mock=True, interval=0.1)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.5)
            self.assertTrue(app.compact)
            self.assertTrue(app.query_one("#graphs").has_class("row"))
            self.assertLessEqual(app.screen.virtual_size.height, 24)

    async def test_layout_follows_resize(self):
        app = TpuTopApp(use_mock=True, interval=0.1)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.5)
            self.assertFalse(app.compact)
            self.assertFalse(app.query_one("#graphs").has_class("row"))
            await pilot.resize_terminal(80, 24)
            await pilot.pause(0.3)
            self.assertTrue(app.compact)
            self.assertTrue(app.query_one("#graphs").has_class("row"))


class TestArgs(unittest.TestCase):

    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.interval, 0.5)
        self.assertFalse(args.mock)

    def test_flags(self):
        args = parse_args(["--mock", "-n", "2"])
        self.assertTrue(args.mock)
        self.assertEqual(args.interval, 2.0)


if __name__ == "__main__":
    unittest.main()
