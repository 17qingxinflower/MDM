from __future__ import annotations

import logging
import sys
import threading
import traceback
from pathlib import Path


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "deerui.log"

    root = logging.getLogger()
    # Reset handlers so re-configuration (e.g. tests) doesn't stack them.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    _install_exception_hooks(log_path)


def _install_exception_hooks(log_path: Path) -> None:
    """Route uncaught exceptions and thread crashes to the log file.

    Without these, a crash in a Qt slot, a `QThread` body, or a worker shows up
    as either a silent process death (segfault) or an unhandled-exception print
    to stderr that's easy to miss when the GUI window has focus. Logging them
    gives us a trail to debug from.
    """

    def _excepthook(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
        logging.getLogger("deerui.uncaught").error(
            "Uncaught exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _excepthook

    def _thread_excepthook(args) -> None:  # noqa: ANN001
        logging.getLogger("deerui.thread").error(
            "Uncaught exception in thread %s:\n%s",
            getattr(args, "thread", None) and args.thread.name,
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    # Python 3.8+ has threading.excepthook; on older interpreters we silently skip.
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook
