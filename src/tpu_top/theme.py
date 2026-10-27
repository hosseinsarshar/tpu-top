"""Colors used by tpu-top.

Each metric keeps one hue everywhere it shows up (gauges, graphs, borders),
based on Google's palette. Neutral colors (gauge tracks, muted text, rules)
are derived from the active Textual theme, so the dashboard stays readable
when the theme is switched from the command palette.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.theme import Theme

THEME_NAME = "tpu-top"

TPU_TOP_THEME = Theme(
    name=THEME_NAME,
    primary="#8AB4F8",
    secondary="#C58AF9",
    accent="#FDD663",
    warning="#FDD663",
    error="#F28B82",
    success="#81C995",
    foreground="#E3E8EF",
    background="#0E1117",
    surface="#151A22",
    panel="#1C232E",
    dark=True,
)

_FALLBACK_BACKGROUND = "#0E1117"
_FALLBACK_FOREGROUND = "#E3E8EF"

# Metric tones as (dark theme, light theme).
_METRIC_TONES = {
    "cpu": ("#8AB4F8", "#1967D2"),  # blue
    "ram": ("#F28B82", "#C5221F"),  # red
    "tensorcore": ("#81C995", "#188038"),  # green
    "hbm": ("#FDD663", "#B06000"),  # yellow
    "duty_cycle": ("#C58AF9", "#8430CE"),  # purple
}


def _parse_hex(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) not in (6, 8):
        raise ValueError(f"Not a hex color: {color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def blend(start: str, end: str, t: float) -> str:
    """Mix two ``#rrggbb`` colors. ``t=0`` gives ``start`` and ``t=1`` gives ``end``."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = _parse_hex(start)
    r2, g2, b2 = _parse_hex(end)
    return "#{:02X}{:02X}{:02X}".format(
        round(r1 + (r2 - r1) * t),
        round(g1 + (g2 - g1) * t),
        round(b1 + (b2 - b1) * t),
    )


@dataclass(frozen=True)
class Hue:
    """A metric color. ``fg`` is the strong tone, ``floor`` the faint one."""

    fg: str
    floor: str

    def at(self, t: float) -> str:
        """Return the color ``t`` (0..1) of the way from ``floor`` to ``fg``."""
        return blend(self.floor, self.fg, t)

    @property
    def border(self) -> str:
        """A softer tone of the hue, for panel borders."""
        return self.at(0.55)


@dataclass(frozen=True)
class Palette:
    """Every color the renderers need, resolved for one theme."""

    cpu: Hue
    ram: Hue
    tensorcore: Hue
    hbm: Hue
    duty_cycle: Hue
    background: str
    text: str
    muted: str
    faint: str
    track: str
    stripe: str
    primary: str
    good: str
    warn: str
    bad: str
    dark: bool


def _safe_hex(color: str | None, fallback: str) -> str:
    if not color:
        return fallback
    try:
        _parse_hex(color)
    except ValueError:
        return fallback
    return "#" + color.strip().lstrip("#")[:6]


def build_palette(
    background: str | None = None,
    foreground: str | None = None,
    primary: str | None = None,
    dark: bool = True,
) -> Palette:
    """Build a palette from a few theme colors (falls back to the default theme)."""
    bg = _safe_hex(background, _FALLBACK_BACKGROUND)
    fg = _safe_hex(foreground, _FALLBACK_FOREGROUND)
    tone_index = 0 if dark else 1

    def hue(name: str) -> Hue:
        tone = _METRIC_TONES[name][tone_index]
        return Hue(fg=tone, floor=blend(bg, tone, 0.30))

    tensorcore = hue("tensorcore")
    hbm = hue("hbm")
    ram = hue("ram")
    return Palette(
        cpu=hue("cpu"),
        ram=ram,
        tensorcore=tensorcore,
        hbm=hbm,
        duty_cycle=hue("duty_cycle"),
        background=bg,
        text=fg,
        muted=blend(bg, fg, 0.58),
        faint=blend(bg, fg, 0.28),
        track=blend(bg, fg, 0.10),
        stripe=blend(bg, fg, 0.035),
        primary=_safe_hex(primary, _METRIC_TONES["cpu"][tone_index]),
        good=tensorcore.fg,
        warn=hbm.fg,
        bad=ram.fg,
        dark=dark,
    )


DEFAULT_PALETTE = build_palette()
