from __future__ import annotations


COLOR_BG_MAIN = "#0A0F0D"
COLOR_BG_SIDEBAR = "#0D1411"
COLOR_PANEL = "#111814"
COLOR_BORDER = "#1B261F"
COLOR_PRIMARY = "#10B981"
COLOR_PRIMARY_HOVER = "#059669"
COLOR_DANGER = "#EF4444"
COLOR_WARNING = "#F59E0B"
COLOR_TEXT_MAIN = "#F3F4F6"
COLOR_TEXT_SUB = "#6B7280"

# BGR equivalent of COLOR_PRIMARY (#10B981) for cv2 primitives.
COLOR_PRIMARY_BGR: tuple[int, int, int] = (129, 185, 16)

BEHAVIOR_COLORS: dict[str, str] = {
    "Stand_or_walk": "#10B981",
    "Standing_feeding": "#3B82F6",
    "egestion": "#F59E0B",
    "Lie_down_and_rest": "#8B5CF6",
}
BEHAVIOR_COLOR_ALIASES: dict[str, str] = {
    "Stand/Walk": BEHAVIOR_COLORS["Stand_or_walk"],
    "Stand/Walk": BEHAVIOR_COLORS["Stand_or_walk"],
    "Feeding": BEHAVIOR_COLORS["Standing_feeding"],
    "Feeding": BEHAVIOR_COLORS["Standing_feeding"],
    "Egestion": BEHAVIOR_COLORS["egestion"],
    "Egestion": BEHAVIOR_COLORS["egestion"],
    "Lie Down": BEHAVIOR_COLORS["Lie_down_and_rest"],
    "Lie Down": BEHAVIOR_COLORS["Lie_down_and_rest"],
}
BEHAVIOR_FALLBACK_COLORS: tuple[str, ...] = (
    "#06B6D4",
    "#EF4444",
    "#EC4899",
    "#84CC16",
)


def behavior_color(action_key: str | None, display_name: str | None = None) -> str:
    for candidate in (action_key, display_name):
        if not candidate:
            continue
        if candidate in BEHAVIOR_COLORS:
            return BEHAVIOR_COLORS[candidate]
        if candidate in BEHAVIOR_COLOR_ALIASES:
            return BEHAVIOR_COLOR_ALIASES[candidate]
    basis = action_key or display_name or ""
    index = sum(ord(char) for char in basis) % len(BEHAVIOR_FALLBACK_COLORS)
    return BEHAVIOR_FALLBACK_COLORS[index]


APP_STYLESHEET = f"""
QMainWindow {{
    background: {COLOR_BG_MAIN};
}}
QWidget {{
    color: {COLOR_TEXT_MAIN};
    font-family: "Microsoft YaHei", "Segoe UI", Arial;
}}
QPushButton {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 10px 12px;
    background: transparent;
    color: {COLOR_TEXT_MAIN};
    text-align: left;
}}
QPushButton:hover {{
    background: {COLOR_PANEL};
}}
QPushButton[active="true"] {{
    background: {COLOR_PANEL};
    color: {COLOR_PRIMARY};
    font-weight: 700;
}}
QLabel[muted="true"] {{
    color: {COLOR_TEXT_SUB};
}}
QFrame[panel="true"] {{
    background: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
}}
"""
