from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _as_positive_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _number_text(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _normalize_strategy(raw: str | None) -> str:
    strategy = (raw or "capacity").strip().lower().replace("-", "_")
    if strategy in {"smooth", "preview", "smooth_preview"}:
        return "smooth_preview"
    return "capacity"


def _recommendation_value(
    recommendation: dict[str, Any],
    metadata: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = recommendation.get(key)
    if value is None:
        value = metadata.get(key, default)
    return value


def _infer_strategy(
    *,
    metadata: dict[str, Any],
    recommendation: dict[str, Any],
    strategy: str | None,
) -> str:
    if strategy:
        return _normalize_strategy(strategy)
    metadata_strategy = metadata.get("inference_strategy")
    if metadata_strategy:
        return _normalize_strategy(str(metadata_strategy))
    target_fps = float(
        _recommendation_value(recommendation, metadata, "target_fps_per_stream", 10.0)
    )
    sample_interval = int(
        _recommendation_value(recommendation, metadata, "sample_every_n_frames", 2)
    )
    if target_fps >= 14.5 and sample_interval >= 3:
        return "smooth_preview"
    return "capacity"


def _validate_recommendation(recommendation: dict[str, Any]) -> int:
    if recommendation.get("usable") is not True:
        raise ValueError("benchmark recommendation is not usable")
    if recommendation.get("resource_stable") is not True:
        raise ValueError("benchmark recommendation is not resource stable")
    if recommendation.get("benchmark_status_stable") is not True:
        raise ValueError("benchmark recommendation is not status stable")
    if recommendation.get("fault_injection_enabled") is True:
        if recommendation.get("fault_reliable") is not True:
            raise ValueError("benchmark recommendation is not fault reliable")
        _as_positive_int(
            recommendation.get("candidate_faulted_run_count"),
            "recommendation.candidate_faulted_run_count",
        )
        _as_positive_int(
            recommendation.get("candidate_fault_events"),
            "recommendation.candidate_fault_events",
        )

    candidate_run_count = _as_positive_int(
        recommendation.get("candidate_run_count"),
        "recommendation.candidate_run_count",
    )
    candidate_reliable_run_count = _as_positive_int(
        recommendation.get("candidate_reliable_run_count"),
        "recommendation.candidate_reliable_run_count",
    )
    if candidate_reliable_run_count < candidate_run_count:
        raise ValueError("benchmark recommendation is not reliable in every candidate run")

    return _as_positive_int(
        recommendation.get("recommended_streams_per_worker"),
        "recommendation.recommended_streams_per_worker",
    )


def build_deployment_variables(
    payload: dict[str, Any],
    *,
    benchmark_path: Path,
    capacity_profile: str = "rtx5090",
    strategy: str | None = None,
    model_path: Path | str | None = None,
    device: str | None = None,
) -> dict[str, str]:
    metadata = _as_dict(payload.get("metadata"), "metadata")
    recommendation = _as_dict(payload.get("recommendation"), "recommendation")
    capacity = _validate_recommendation(recommendation)

    resolved_strategy = _infer_strategy(
        metadata=metadata,
        recommendation=recommendation,
        strategy=strategy,
    )
    model = str(model_path or metadata.get("model") or "")
    if not model:
        raise ValueError("metadata.model is required unless --model is provided")

    return {
        "DEERUI_MODEL_PATH": model,
        "DEERUI_INFERENCE_DEVICE": str(device or metadata.get("device") or "cuda:0"),
        "DEERUI_INFERENCE_STRATEGY": resolved_strategy,
        "DEERUI_INFERENCE_BATCH_SIZE": str(
            _as_positive_int(metadata.get("batch_size", 1), "metadata.batch_size")
        ),
        "DEERUI_INFERENCE_WORKER_COUNT": str(
            _as_positive_int(metadata.get("worker_count", 1), "metadata.worker_count")
        ),
        "DEERUI_INFERENCE_IMGSZ": str(
            _as_positive_int(metadata.get("imgsz", 640), "metadata.imgsz")
        ),
        "DEERUI_INFERENCE_HALF": _bool_text(metadata.get("half", True)),
        "DEERUI_ADAPTIVE_SAMPLING": _bool_text(metadata.get("adaptive_sampling", False)),
        "DEERUI_SAMPLE_EVERY_N_FRAMES": str(
            _as_positive_int(
                _recommendation_value(recommendation, metadata, "sample_every_n_frames", 2),
                "recommendation.sample_every_n_frames",
            )
        ),
        "DEERUI_STREAM_TARGET_FPS": _number_text(
            _recommendation_value(recommendation, metadata, "target_fps_per_stream", 10.0)
        ),
        "DEERUI_CAPACITY_PROFILE": capacity_profile.strip().lower() or "rtx5090",
        "DEERUI_CAPACITY_CALIBRATION_PATH": str(benchmark_path),
        "DEERUI_WORKER_STREAM_CAPACITY": str(capacity),
    }


def format_powershell_profile(variables: dict[str, str]) -> str:
    lines = []
    for key, value in variables.items():
        escaped = str(value).replace("'", "''")
        lines.append(f"$env:{key}='{escaped}'")
    return "\n".join(lines) + "\n"


def format_env_profile(variables: dict[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in variables.items())


def write_profile_file(output_path: Path, content: str, profile_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if profile_format == "powershell" else "utf-8"
    output_path.write_text(content, encoding=encoding)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read benchmark json: {path}") from exc
    return _as_dict(payload, "benchmark payload")


def _default_output_path(benchmark_path: Path, profile_format: str) -> Path:
    suffix = ".ps1" if profile_format == "powershell" else ".env"
    return benchmark_path.with_name(f"{benchmark_path.stem}_deployment{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a deployment profile from an accepted stream-capacity benchmark."
    )
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=["powershell", "env"], default="powershell")
    parser.add_argument("--capacity-profile", default="rtx5090")
    parser.add_argument("--strategy", choices=["capacity", "smooth-preview", "smooth_preview"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_path = args.benchmark_json.resolve()
    payload = _load_payload(benchmark_path)
    variables = build_deployment_variables(
        payload,
        benchmark_path=benchmark_path,
        capacity_profile=args.capacity_profile,
        strategy=args.strategy,
        model_path=args.model,
        device=args.device,
    )
    output_path = args.output or _default_output_path(benchmark_path, args.format)
    if args.format == "powershell":
        content = format_powershell_profile(variables)
    else:
        content = format_env_profile(variables)
    write_profile_file(output_path, content, args.format)
    print(f"profile={output_path}")
    print(
        "strategy="
        f"{variables['DEERUI_INFERENCE_STRATEGY']} "
        "capacity_per_worker="
        f"{variables['DEERUI_WORKER_STREAM_CAPACITY']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
