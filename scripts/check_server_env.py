from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_command(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {"command": command, "available": False, "returncode": None, "stdout": "", "stderr": "not found"}

    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def collect_torch_info(device: str) -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure is the signal
        return {"available": False, "error": repr(exc)}

    info: dict[str, object] = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "requested_device": device,
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        target = device if device != "auto" else "cuda"
        tensor = torch.tensor([1.0, 2.0, 3.0], device=target)
        info["cuda_smoke_test"] = {
            "device": str(tensor.device),
            "sum": float(tensor.sum().item()),
        }
    return info


def dataset_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return summary

    line_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_count, _ in enumerate(handle, start=1):
            pass
    summary["num_rows"] = line_count
    summary["size_bytes"] = path.stat().st_size
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check server environment, CUDA, and benchmark file availability.")
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="*",
        default=[
            Path("data/processed/pheme_forecast_ratio_05_train.jsonl"),
            Path("data/processed/pheme_forecast_ratio_05_val.jsonl"),
            Path("data/processed/pheme_forecast_ratio_05_test.jsonl"),
        ],
        help="Benchmark files that must exist on the server.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device used for the CUDA smoke test. Use auto to select cuda when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/server_env_report.json"),
        help="Path to write the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_device = args.device
    if target_device == "auto":
        target_device = "cuda"

    report = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "commands": {
            "whoami": run_command(["whoami"]),
            "uname": run_command(["uname", "-a"]),
            "python3": run_command(["python3", "--version"]) if shutil.which("python3") else {"available": False},
            "nvidia_smi": run_command(["nvidia-smi"]),
            "nvcc": run_command(["nvcc", "--version"]),
        },
        "torch": collect_torch_info(target_device),
        "datasets": [dataset_summary(path) for path in args.dataset],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"saved_report={args.output}")


if __name__ == "__main__":
    main()
