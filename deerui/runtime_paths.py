from __future__ import annotations

import os
import sys
from pathlib import Path


APP_DATA_DIR_NAME = "DeerUI"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        return Path(bundled_root).resolve()
    return app_root()


def program_data_root() -> Path:
    override = os.getenv("DEERUI_DATA_ROOT", "").strip()
    if override:
        return Path(override)
    program_data = os.getenv("PROGRAMDATA", r"C:\ProgramData")
    return Path(program_data) / APP_DATA_DIR_NAME


def default_output_dir() -> Path:
    if is_frozen() or os.getenv("DEERUI_DEPLOYMENT_MODE", "").strip().lower() == "full":
        return program_data_root() / "UI_result"
    return app_root() / "UI_result"


def default_env_file() -> Path:
    return program_data_root() / "config" / "deerui.env"


def env_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.getenv("DEERUI_ENV_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            default_env_file(),
            app_root() / "deerui.env",
            app_root() / ".env",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def bundled_resource(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def default_benchmark_video_path() -> Path | None:
    for path in (
        bundled_resource("benchmark_assets", "053358f40cf7b5cfe5d8620a94d621b9.mp4"),
        bundled_resource(
            "packaging",
            "windows",
            "assets",
            "053358f40cf7b5cfe5d8620a94d621b9.mp4",
        ),
    ):
        if path.exists():
            return path
    return None


def packaged_default_database_url() -> str:
    password = os.getenv("DEERUI_LOCAL_POSTGRES_PASSWORD", "deerui_local_pw")
    return f"postgresql+psycopg://deerui_app:{password}@127.0.0.1:5432/deerui"
