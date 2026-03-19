from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_reporting import ensure_experiment_paths, load_hypothesis_config, persist_manifest_outputs, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute figures and idea.md alignment analysis from an existing manifest.")
    parser.add_argument("--run_manifest", type=Path, required=True)
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--config_path", type=Path, default=Path("configs/research_hypotheses.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_experiment_paths(args.experiments_root)
    config = load_hypothesis_config(args.config_path)
    manifest = read_json(args.run_manifest)
    from src.experiment_reporting import analyze_hypotheses, detect_anomalies, recommendation_from_hypotheses

    manifest["hypothesis_analysis"] = analyze_hypotheses(manifest, config)
    manifest["anomalies"] = detect_anomalies(manifest, config)
    manifest["idea_followup"] = recommendation_from_hypotheses(manifest["hypothesis_analysis"], manifest["anomalies"], config)
    persist_manifest_outputs(paths, manifest)
    print(f"updated_manifest={manifest['manifest_path']}")
    print(f"updated_report={manifest['report_path']}")


if __name__ == "__main__":
    main()
