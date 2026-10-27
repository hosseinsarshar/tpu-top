import unittest
from rich.console import Console
from tpu_top.theme import DEFAULT_PALETTE, blend, build_palette
from tpu_top.ui import (
    Gauge,
    braille_graph,
    build_process_rows,
    format_bytes,
    format_device_list,
    format_duration,
    hlo_lines_for_device,
    make_device_table,
    make_timeline,
    parse_hlo_location,
    sort_process_rows,
    sparkline,
    vertical_bar_chart,
)


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
        self.assertTrue(res.startswith("─"))
        self.assertTrue(res.endswith("0s"))

    def test_make_timeline_labels(self):
        # One column per second: labels every 30s, newest on the right.
        res = make_timeline(100, seconds_per_col=1.0)
        self.assertEqual(len(res), 100)
        self.assertTrue(res.endswith(" 0s"))
        self.assertIn(" 30s ", res)
        self.assertIn(" 1m ", res)
        # The "30s" label is centered on the column 30 samples from the end.
        self.assertEqual(res.index(" 30s ") + 2, 100 - 1 - 30)

    def test_make_timeline_tiny(self):
        self.assertEqual(make_timeline(0), "")
        self.assertEqual(make_timeline(2), "0s")

    def test_format_duration(self):
        self.assertEqual(format_duration(45), "45s")
        self.assertEqual(format_duration(120), "2m")
        self.assertEqual(format_duration(90), "1m30s")
        self.assertEqual(format_duration(3600), "1h")


class TestBrailleGraph(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(braille_graph([], width=5, height=2), [" " * 5] * 2)

    def test_full(self):
        self.assertEqual(braille_graph([100.0] * 10, width=5, height=2), ["⣿" * 5] * 2)

    def test_half(self):
        top, bottom = braille_graph([50.0] * 10, width=5, height=2)
        self.assertEqual(top, " " * 5)
        self.assertEqual(bottom, "⣿" * 5)

    def test_two_samples_per_column(self):
        # Left half empty, right half full.
        (line,) = braille_graph([0.0, 100.0], width=1, height=1)
        self.assertEqual(line, "⢸")

    def test_small_values_keep_a_baseline(self):
        (line,) = braille_graph([1.0, 1.0], width=1, height=1)
        self.assertEqual(line, "⣀")

    def test_newest_sample_on_the_right(self):
        (line,) = braille_graph([100.0, 100.0], width=3, height=1)
        self.assertEqual(line, "  ⣿")


class TestGauge(unittest.TestCase):

    def render(self, renderable, width):
        console = Console(width=width, color_system=None, legacy_windows=False)
        with console.capture() as capture:
            console.print(renderable)
        return capture.get().rstrip("\n")

    def test_fills_the_width(self):
        gauge = Gauge(50.0, DEFAULT_PALETTE.hbm, DEFAULT_PALETTE.track)
        text = gauge.render(30)
        self.assertEqual(text.cell_len, 30)
        self.assertTrue(text.plain.endswith("50.0%"))

    def test_partial_cell(self):
        gauge = Gauge(55.0, DEFAULT_PALETTE.hbm, DEFAULT_PALETTE.track, label="")
        text = gauge.render(10)
        # 5.5 of 10 cells: the 6th cell is a half block.
        self.assertEqual(text.plain[5], "▌")

    def test_clamps(self):
        for value in (-5.0, 250.0):
            text = Gauge(value, DEFAULT_PALETTE.cpu, DEFAULT_PALETTE.track, label="x").render(12)
            self.assertEqual(text.cell_len, 12)

    def test_device_table_renders(self):
        devices = [
            {"id": 0, "memory_usage": 40 * 1024**3, "total_memory": 95 * 1024**3,
             "duty_cycle": 99.0, "tensorcore_util": 42.0},
            {"id": 1, "memory_usage": 0, "total_memory": 0, "duty_cycle": 0.0, "tensorcore_util": 0.0},
        ]
        hlo_map = {0: {"TC0": "HLO: fusion.1; HLO computation: main.1_spmd", "TC1": "N/A"}}
        for compact in (False, True):
            out = self.render(make_device_table(devices, hlo_map, 2, compact=compact), 140)
            self.assertIn("TPU 0", out)
            self.assertIn("fusion.1", out)
            if compact:
                self.assertIn("42%", out)
            else:
                self.assertIn("main.1_spmd", out)
                self.assertIn("40.0 / 95.0 GiB", out)


class TestHelpers(unittest.TestCase):

    def test_format_device_list(self):
        self.assertEqual(format_device_list([]), "")
        self.assertEqual(format_device_list([3, 1, 2, 0]), "0-3")
        self.assertEqual(format_device_list([0, 2, 3, 7]), "0,2-3,7")

    def test_format_bytes(self):
        self.assertEqual(format_bytes(0), "0 B")
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(310 * 1024**2), "310.0 MiB")
        self.assertEqual(format_bytes(26.7 * 1024**3), "26.7 GiB")

    def test_parse_hlo_location(self):
        self.assertEqual(
            parse_hlo_location("HLO: RPAm-p256-b1.1; HLO computation: main.579_spmd"),
            ("RPAm-p256-b1.1", "main.579_spmd"),
        )
        self.assertEqual(parse_hlo_location("N/A"), ("", ""))
        self.assertEqual(parse_hlo_location(None), ("", ""))
        self.assertEqual(parse_hlo_location("something else"), ("something else", ""))

    def test_hlo_lines_fall_back_to_chip_0(self):
        hlo_map = {0: {"TC0": "HLO: a; HLO computation: b", "SC0": "HLO: c; HLO computation: d"}}
        busy = {"id": 1, "tensorcore_util": 50.0}
        lines = hlo_lines_for_device(busy, hlo_map, devices_per_chip=2)
        self.assertEqual(lines[0], ("TC0", "HLO: a; HLO computation: b"))
        self.assertEqual(lines[1], ("TC1", "N/A"))
        self.assertEqual(lines[2], ("SC0", "HLO: c; HLO computation: d"))
        idle = {"id": 1, "tensorcore_util": 0.0}
        self.assertEqual(hlo_lines_for_device(idle, hlo_map, 1), [("TC0", "N/A")])

    def test_build_process_rows_merges_tpu_owners(self):
        processes = [
            {"pid": 10, "name": "train.py", "device": f"TPU {i}", "cpu": 200.0, "memory": 5}
            for i in (2, 0, 1, 3)
        ]
        processes += [
            {"pid": 20, "name": "loader", "device": "CPU", "cpu": 50.0, "memory": 9},
            {"pid": 30, "name": "tpu-top", "device": "CPU", "cpu": 5.0, "memory": 1},
            {"pid": 40, "name": "me", "device": "CPU", "cpu": 5.0, "memory": 1},
        ]
        rows = build_process_rows(processes, cpu_count=4, exclude_pids={40})
        self.assertEqual([r.pid for r in rows], [10, 20])
        train = rows[0]
        self.assertEqual(train.devices, [0, 1, 2, 3])
        self.assertEqual(train.device_label, "TPU 0-3")
        self.assertAlmostEqual(train.cpu, 50.0)
        self.assertEqual(rows[1].device_label, "CPU")

    def test_sort_process_rows(self):
        processes = [
            {"pid": 5, "name": "a", "device": "CPU", "cpu": 90.0, "memory": 1},
            {"pid": 3, "name": "b", "device": "TPU 1", "cpu": 10.0, "memory": 2},
            {"pid": 9, "name": "c", "device": "TPU 0", "cpu": 20.0, "memory": 3},
            {"pid": 1, "name": "d", "device": "CPU", "cpu": 30.0, "memory": 8},
        ]
        rows = build_process_rows(processes)
        pids = lambda mode: [r.pid for r in sort_process_rows(rows, mode)]
        self.assertEqual(pids("device"), [9, 3, 1, 5])
        self.assertEqual(pids("cpu"), [5, 1, 9, 3])
        self.assertEqual(pids("memory"), [1, 9, 3, 5])
        self.assertEqual(pids("pid"), [1, 3, 5, 9])


class TestTheme(unittest.TestCase):

    def test_blend(self):
        self.assertEqual(blend("#000000", "#FFFFFF", 0.0), "#000000")
        self.assertEqual(blend("#000000", "#FFFFFF", 1.0), "#FFFFFF")
        self.assertEqual(blend("#000000", "#FFFFFF", 0.5), "#808080")

    def test_build_palette_handles_odd_colors(self):
        palette = build_palette(background="ansi_default", foreground=None, dark=False)
        self.assertTrue(palette.track.startswith("#"))
        self.assertFalse(palette.dark)


if __name__ == '__main__':
    unittest.main()
