from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deerui.config import load_config, resolve_inference_device

YoloFactory = Callable[[str], Any]


def tensorrt_status() -> tuple[bool, str]:
    try:
        import tensorrt as trt  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, getattr(trt, "__version__", "unknown")


def build_export_kwargs(
    *,
    imgsz: int,
    batch: int,
    half: bool,
    dynamic: bool,
    workspace: int | None,
    device: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "format": "engine",
        "imgsz": max(int(imgsz), 1),
        "batch": max(int(batch), 1),
        "half": bool(half),
        "dynamic": bool(dynamic),
    }
    if workspace is not None:
        workspace_value = int(workspace)
        if workspace_value > 0:
            kwargs["workspace"] = workspace_value
    if device:
        kwargs["device"] = str(device)
    return kwargs


def expected_engine_path(model_path: Path, output_dir: Path | None = None) -> Path:
    if output_dir is not None:
        return output_dir / f"{model_path.stem}.engine"
    return model_path.with_suffix(".engine")


def export_engine(
    model_path: Path,
    export_kwargs: dict[str, Any],
    *,
    output_dir: Path | None = None,
    yolo_factory: YoloFactory | None = None,
) -> Path:
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    exported = Path(yolo_factory(str(model_path)).export(**export_kwargs))
    if output_dir is None:
        return exported

    output_dir.mkdir(parents=True, exist_ok=True)
    target = expected_engine_path(model_path, output_dir)
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
        return target
    return exported


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Export a YOLO model to TensorRT engine.")
    parser.add_argument("--model", type=Path, default=config.model_path)
    parser.add_argument("--imgsz", type=int, default=config.inference_imgsz)
    parser.add_argument("--batch", type=int, default=config.inference_batch_size)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=config.inference_half,
        help="Export FP16 engine when supported.",
    )
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export a dynamic-shape/dynamic-batch engine.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=None,
        help="TensorRT workspace size in GiB. Omit to use the Ultralytics default.",
    )
    parser.add_argument("--device", default=resolve_inference_device(config.inference_device))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print export settings without importing TensorRT or running YOLO export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    export_kwargs = build_export_kwargs(
        imgsz=args.imgsz,
        batch=args.batch,
        half=args.half,
        dynamic=args.dynamic,
        workspace=args.workspace,
        device=args.device,
    )
    expected_path = expected_engine_path(args.model, args.output_dir)
    print(f"dry_run={bool(args.dry_run)}")
    print(f"model={args.model}")
    print(f"expected_engine={expected_path}")
    print("export_kwargs=" + json.dumps(export_kwargs, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    available, detail = tensorrt_status()
    if not available:
        print("TensorRT runtime is not available.")
        print(detail)
        print("Install NVIDIA TensorRT runtime DLLs and ensure nvinfer_10.dll is on PATH.")
        return 2

    print(f"TensorRT={detail}")
    exported = export_engine(args.model, export_kwargs, output_dir=args.output_dir)
    print(f"engine={exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
