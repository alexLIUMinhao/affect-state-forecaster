from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    logs: Path
    records: Path
    figures: Path
    manifests: Path


def ensure_experiment_paths(root: str | Path = "experiments") -> ExperimentPaths:
    root_path = Path(root)
    paths = ExperimentPaths(
        root=root_path,
        logs=root_path / "logs",
        records=root_path / "records",
        figures=root_path / "figures",
        manifests=root_path / "manifests",
    )
    for path in (paths.root, paths.logs, paths.records, paths.figures, paths.manifests):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_hypothesis_config(path: str | Path = "configs/research_hypotheses.json") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in normalized.split("_") if part)


def default_run_id(tag: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{slugify(tag)}"


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _status(rank: str) -> int:
    order = {"不支持": 0, "部分支持": 1, "支持": 2, "证据不足": -1}
    return order.get(rank, -1)


def discover_run_models(run_root: Path) -> list[str]:
    if not run_root.exists():
        return []
    return sorted(path.name for path in run_root.iterdir() if path.is_dir())


def collect_run_results(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    models: list[dict[str, Any]] = []
    available_ratios: set[float] = set()
    for model_dir in discover_run_models(run_root):
        artifact_dir = run_root / model_dir / "artifacts"
        eval_dir = run_root / model_dir / "eval"
        summary_rows = read_csv_rows(eval_dir / "results_summary.csv")
        event_rows = read_csv_rows(eval_dir / "error_analysis_by_event.csv")
        history_payload = read_json(next(iter(sorted(artifact_dir.glob("*_summary.json"))), artifact_dir / "missing.json")) if list(artifact_dir.glob("*_summary.json")) else {"history": []}
        overall_row = next((row for row in summary_rows if row.get("group_name") == "overall"), None)
        ratio_rows = [row for row in summary_rows if row.get("group_name") == "observation_ratio"]
        for ratio_row in ratio_rows:
            ratio_value = _to_float(ratio_row.get("group_value"))
            if ratio_value is not None:
                available_ratios.add(ratio_value)
        models.append(
            {
                "model": model_dir,
                "status": "completed" if overall_row else "failed",
                "paths": {
                    "artifact_dir": str(artifact_dir),
                    "eval_dir": str(eval_dir),
                    "summary_csv": str(eval_dir / "results_summary.csv"),
                    "event_error_csv": str(eval_dir / "error_analysis_by_event.csv"),
                },
                "overall_metrics": {
                    "mae": _to_float(overall_row.get("mae")) if overall_row else None,
                    "rmse": _to_float(overall_row.get("rmse")) if overall_row else None,
                    "pearson": _to_float(overall_row.get("pearson")) if overall_row else None,
                    "spearman": _to_float(overall_row.get("spearman")) if overall_row else None,
                    "accuracy": _to_float(overall_row.get("majority_sentiment_accuracy")) if overall_row else None,
                    "macro_f1": _to_float(overall_row.get("majority_sentiment_macro_f1")) if overall_row else None,
                    "weighted_f1": _to_float(overall_row.get("majority_sentiment_weighted_f1")) if overall_row else None,
                    "loss": _to_float(overall_row.get("loss")) if overall_row else None,
                },
                "ratio_metrics": ratio_rows,
                "event_metrics": event_rows,
                "history": history_payload.get("history", []),
            }
        )

    return {
        "run_root": str(run_root),
        "models": models,
        "observation_ratios": sorted(available_ratios),
    }


def build_model_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["model"]: entry for entry in manifest.get("models", [])}


def best_model_by_metric(manifest: dict[str, Any], metric: str = "mae") -> dict[str, Any] | None:
    candidates = []
    for entry in manifest.get("models", []):
        value = entry.get("overall_metrics", {}).get(metric)
        if value is None:
            continue
        candidates.append((value, entry))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _make_hypothesis_result(hypothesis: dict[str, Any], verdict: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "id": hypothesis["id"],
        "title": hypothesis["title"],
        "verdict": verdict,
        "related_idea_sections": hypothesis.get("related_idea_sections", []),
        "evidence": evidence,
    }


def analyze_hypotheses(manifest: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = build_model_lookup(manifest)
    text = lookup.get("text_baseline")
    temporal = lookup.get("temporal_baseline")
    structure = lookup.get("structure_baseline")
    affect = lookup.get("affect_state_forecaster")
    best = best_model_by_metric(manifest, "mae")
    results: list[dict[str, Any]] = []

    for hypothesis in config.get("hypotheses", []):
        hypothesis_id = hypothesis["id"]
        criteria = hypothesis.get("criteria", {})
        evidence: list[str] = []
        verdict = "证据不足"

        if hypothesis_id == "H1":
            if not text or not best:
                verdict = "证据不足"
                evidence.append("缺少文本基线或总体最佳模型结果。")
            else:
                text_mae = text["overall_metrics"]["mae"]
                best_mae = best["overall_metrics"]["mae"]
                best_corr = max(best["overall_metrics"].get("pearson") or -1.0, best["overall_metrics"].get("spearman") or -1.0)
                mae_gain = (text_mae - best_mae) if text_mae is not None and best_mae is not None else None
                if mae_gain is None:
                    verdict = "证据不足"
                elif mae_gain >= criteria.get("min_mae_gain_vs_text", 0.0) and best_corr >= criteria.get("min_positive_correlation", 0.0):
                    verdict = "支持"
                elif mae_gain > 0.0 or best_corr >= 0.0:
                    verdict = "部分支持"
                else:
                    verdict = "不支持"
                evidence.append(f"最佳模型为 {best['model']}，相对文本基线 MAE 改进 {mae_gain:.4f}。" if mae_gain is not None else "无法计算相对文本基线改进。")
                evidence.append(f"最佳模型相关系数表现 Pearson={best['overall_metrics'].get('pearson')}, Spearman={best['overall_metrics'].get('spearman')}。")

        elif hypothesis_id == "H2":
            if not affect or not text:
                verdict = "证据不足"
                evidence.append("缺少 affect-state 或文本基线结果。")
            else:
                affect_mae = affect["overall_metrics"]["mae"]
                affect_rmse = affect["overall_metrics"]["rmse"]
                text_mae = text["overall_metrics"]["mae"]
                text_rmse = text["overall_metrics"]["rmse"]
                better_than_text = (
                    affect_mae is not None
                    and text_mae is not None
                    and affect_rmse is not None
                    and text_rmse is not None
                    and (text_mae - affect_mae) >= criteria.get("min_mae_gain_vs_text", 0.0)
                    and (text_rmse - affect_rmse) >= criteria.get("min_rmse_gain_vs_text", 0.0)
                )
                better_than_others = True
                for candidate in (temporal, structure):
                    if candidate and candidate["overall_metrics"]["mae"] is not None and affect_mae is not None:
                        if affect_mae > candidate["overall_metrics"]["mae"]:
                            better_than_others = False
                if better_than_text and better_than_others:
                    verdict = "支持"
                elif better_than_text:
                    verdict = "部分支持"
                else:
                    verdict = "不支持"
                evidence.append(f"Affect-State Forecaster: MAE={affect_mae}, RMSE={affect_rmse}。")
                evidence.append(f"Text baseline: MAE={text_mae}, RMSE={text_rmse}。")
                if structure:
                    evidence.append(f"Structure baseline: MAE={structure['overall_metrics']['mae']}, RMSE={structure['overall_metrics']['rmse']}。")

        elif hypothesis_id == "H3":
            if not text:
                verdict = "证据不足"
                evidence.append("缺少文本基线结果。")
            else:
                gains = []
                for name, candidate in (("temporal_baseline", temporal), ("structure_baseline", structure)):
                    if candidate and candidate["overall_metrics"]["mae"] is not None and text["overall_metrics"]["mae"] is not None:
                        gains.append((name, text["overall_metrics"]["mae"] - candidate["overall_metrics"]["mae"]))
                if not gains:
                    verdict = "证据不足"
                    evidence.append("缺少时间或结构基线结果。")
                else:
                    best_gain = max(gains, key=lambda item: item[1])
                    if best_gain[1] >= criteria.get("min_mae_gain_vs_text", 0.0):
                        verdict = "支持"
                    elif best_gain[1] > 0.0:
                        verdict = "部分支持"
                    else:
                        verdict = "不支持"
                    for name, gain in gains:
                        evidence.append(f"{name} 相对文本基线的 MAE 改进为 {gain:.4f}。")

        elif hypothesis_id == "H4":
            ratio_points = []
            for entry in manifest.get("models", []):
                for row in entry.get("ratio_metrics", []):
                    ratio_value = _to_float(row.get("group_value"))
                    mae_value = _to_float(row.get("mae"))
                    if ratio_value is not None and mae_value is not None:
                        ratio_points.append((entry["model"], ratio_value, mae_value))
            observed_ratios = sorted({point[1] for point in ratio_points})
            if len(observed_ratios) < criteria.get("min_required_ratios", 2):
                verdict = "证据不足"
                evidence.append(f"当前只有 {len(observed_ratios)} 个 observation ratio，尚不足以判断趋势。")
            else:
                best_by_ratio: dict[float, float] = {}
                for _, ratio_value, mae_value in ratio_points:
                    best_by_ratio[ratio_value] = min(best_by_ratio.get(ratio_value, float("inf")), mae_value)
                ordered = sorted(best_by_ratio.items(), key=lambda item: item[0])
                non_increasing = all(ordered[index][1] >= ordered[index + 1][1] for index in range(len(ordered) - 1))
                if non_increasing:
                    verdict = "支持"
                else:
                    verdict = "部分支持"
                evidence.append("best MAE by ratio: " + ", ".join(f"{ratio:.1f}->{mae:.4f}" for ratio, mae in ordered))

        results.append(_make_hypothesis_result(hypothesis, verdict, evidence))
    return results


def detect_anomalies(manifest: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
    thresholds = config.get("anomaly_thresholds", {})
    lookup = build_model_lookup(manifest)
    anomalies: list[dict[str, str]] = []
    text = lookup.get("text_baseline")
    temporal = lookup.get("temporal_baseline")
    structure = lookup.get("structure_baseline")
    affect = lookup.get("affect_state_forecaster")

    if affect and structure:
        affect_mae = affect["overall_metrics"].get("mae")
        structure_mae = structure["overall_metrics"].get("mae")
        if affect_mae is not None and structure_mae is not None and affect_mae - structure_mae >= thresholds.get("material_gap_for_model_comparison", 0.01):
            anomalies.append(
                {
                    "phenomenon": "Affect-State Forecaster 的 MAE 明显差于 structure_baseline。",
                    "possible_cause": "当前显式 affect-state 表达未带来比结构特征更稳定的增益，或模型容量不足。",
                    "idea_conflict": "与 `idea.md` 中“显式情绪状态建模优于直接预测”主线存在张力。",
                    "recommendation": "先不新建 `idea.md`；继续做 affect-state 消融与正则化实验。若多轮复现同结论，再考虑新增结构优先版研究计划。 ",
                }
            )

    if temporal and text:
        temporal_mae = temporal["overall_metrics"].get("mae")
        text_mae = text["overall_metrics"].get("mae")
        if temporal_mae is not None and text_mae is not None and temporal_mae >= text_mae:
            anomalies.append(
                {
                    "phenomenon": "temporal_baseline 未优于 text_baseline。",
                    "possible_cause": "当前时序编码没有有效利用 reply 顺序，或 observation window 信息密度不足。",
                    "idea_conflict": "削弱了 `idea.md` 中时间动态应带来增益的预期。",
                    "recommendation": "优先检查时间编码与序列长度处理；单次出现不建议新建 `idea.md`。",
                }
            )

    for entry in manifest.get("models", []):
        pearson = entry["overall_metrics"].get("pearson")
        spearman = entry["overall_metrics"].get("spearman")
        if pearson is None or spearman is None:
            continue
        if abs(pearson) <= thresholds.get("correlation_near_zero", 0.05) and abs(spearman) <= thresholds.get("correlation_near_zero", 0.05):
            anomalies.append(
                {
                    "phenomenon": f"{entry['model']} 的 Pearson/Spearman 接近 0。",
                    "possible_cause": "模型可能只学到了均值附近回归，未捕捉 thread 级差异。",
                    "idea_conflict": "削弱了“早期观测足以预测未来走势”的论点。",
                    "recommendation": "需要检查标签噪声、事件切分与模型表达；若多轮如此，再考虑修订研究主线。",
                }
            )
        accuracy = entry["overall_metrics"].get("accuracy")
        macro_f1 = entry["overall_metrics"].get("macro_f1")
        if accuracy is not None and macro_f1 is not None and (accuracy - macro_f1) >= thresholds.get("macro_f1_gap", 0.35):
            anomalies.append(
                {
                    "phenomenon": f"{entry['model']} 的 Accuracy 远高于 Macro-F1。",
                    "possible_cause": "多数类占优，分类表现可能被类别不平衡掩盖。",
                    "idea_conflict": "说明扩展任务分类结果暂时不适合作为强支撑证据。",
                    "recommendation": "在报告中弱化分类结论，优先以回归指标论证主任务。",
                }
            )
        event_maes = [_to_float(row.get("mae")) for row in entry.get("event_metrics", [])]
        clean_event_maes = [value for value in event_maes if value is not None]
        if clean_event_maes and (max(clean_event_maes) - min(clean_event_maes)) >= thresholds.get("event_mae_variation", 0.08):
            anomalies.append(
                {
                    "phenomenon": f"{entry['model']} 在不同事件上的 MAE 波动较大。",
                    "possible_cause": "跨事件泛化仍不稳定，数据或标签可能受事件主题影响。",
                    "idea_conflict": "提示当前证据对 event-level 泛化的支撑还有限。",
                    "recommendation": "后续优先补 cross-event 分析，不建议仅凭当前结果扩展主张。",
                }
            )

    ratio_scores: dict[float, float] = {}
    for entry in manifest.get("models", []):
        for row in entry.get("ratio_metrics", []):
            ratio_value = _to_float(row.get("group_value"))
            mae_value = _to_float(row.get("mae"))
            if ratio_value is None or mae_value is None:
                continue
            ratio_scores[ratio_value] = min(ratio_scores.get(ratio_value, float("inf")), mae_value)
    if len(ratio_scores) >= 2:
        ordered = sorted(ratio_scores.items(), key=lambda item: item[0])
        if any(ordered[index][1] < ordered[index + 1][1] for index in range(len(ordered) - 1)):
            anomalies.append(
                {
                    "phenomenon": "observation ratio 趋势不符合“观测越多越容易”的预期。",
                    "possible_cause": "标签噪声、样本量差异或模型不稳定性干扰了趋势。",
                    "idea_conflict": "与 `idea.md` 中 ratio 评测的可解释叙事存在偏差。",
                    "recommendation": "先补更多 ratio 实验；单轮现象不建议新建 `idea.md`。",
                }
            )
    return anomalies


def recommendation_from_hypotheses(hypotheses: list[dict[str, Any]], anomalies: list[dict[str, str]], config: dict[str, Any]) -> dict[str, Any]:
    conflicts = sum(1 for item in hypotheses if item["verdict"] == "不支持")
    partials = sum(1 for item in hypotheses if item["verdict"] == "部分支持")
    recommend_new = conflicts >= config.get("anomaly_thresholds", {}).get("repeat_conflict_runs_for_new_idea", 2)
    if recommend_new:
        summary = "建议准备补充版研究计划文件，例如 `idea_v2.md`，因为核心假设已有多项被当前证据削弱。"
    elif anomalies:
        summary = "当前更适合保留原始 `idea.md`，同时在报告中记录风险与后续验证动作。"
    else:
        summary = "当前结果整体沿着 `idea.md` 推进，不建议新增研究计划文件。"
    return {
        "suggest_new_idea": recommend_new,
        "summary": summary,
        "num_not_supported": conflicts,
        "num_partial": partials,
    }


def plot_model_metrics(manifest: dict[str, Any], figure_path: Path) -> None:
    labels = []
    maes = []
    rmses = []
    for entry in manifest.get("models", []):
        mae = entry["overall_metrics"].get("mae")
        rmse = entry["overall_metrics"].get("rmse")
        if mae is None or rmse is None:
            continue
        labels.append(entry["model"])
        maes.append(mae)
        rmses.append(rmse)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    positions = list(range(len(labels)))
    plt.bar([pos - 0.18 for pos in positions], maes, width=0.36, label="MAE")
    plt.bar([pos + 0.18 for pos in positions], rmses, width=0.36, label="RMSE")
    plt.xticks(positions, labels, rotation=15, ha="right")
    plt.ylabel("Error")
    plt.title("Model Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def plot_event_errors(manifest: dict[str, Any], figure_path: Path) -> None:
    event_names = sorted(
        {
            str(row.get("event_name"))
            for entry in manifest.get("models", [])
            for row in entry.get("event_metrics", [])
            if row.get("event_name")
        }
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    if event_names:
        x_positions = list(range(len(event_names)))
        num_models = max(len(manifest.get("models", [])), 1)
        width = 0.8 / num_models
        for index, entry in enumerate(manifest.get("models", [])):
            event_lookup = {str(row.get("event_name")): _to_float(row.get("mae")) for row in entry.get("event_metrics", [])}
            values = [event_lookup.get(name, float("nan")) for name in event_names]
            offsets = [pos - 0.4 + width / 2 + index * width for pos in x_positions]
            plt.bar(offsets, values, width=width, label=entry["model"])
        plt.xticks(x_positions, event_names, rotation=20, ha="right")
        plt.ylabel("MAE")
        plt.title("Event-level Error Comparison")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No event error data", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def plot_ratio_trends(manifest: dict[str, Any], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    has_points = False
    for entry in manifest.get("models", []):
        ratios = []
        maes = []
        for row in entry.get("ratio_metrics", []):
            ratio_value = _to_float(row.get("group_value"))
            mae_value = _to_float(row.get("mae"))
            if ratio_value is None or mae_value is None:
                continue
            ratios.append(ratio_value)
            maes.append(mae_value)
        if ratios:
            ordered = sorted(zip(ratios, maes), key=lambda item: item[0])
            plt.plot([item[0] for item in ordered], [item[1] for item in ordered], marker="o", label=entry["model"])
            has_points = True
    if has_points:
        plt.xlabel("Observation Ratio")
        plt.ylabel("MAE")
        plt.title("Observation Ratio Trends")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Insufficient ratio data", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def create_figures(manifest: dict[str, Any], figure_dir: Path, run_id: str) -> dict[str, str]:
    model_metrics = figure_dir / f"{run_id}_model_metrics.png"
    event_errors = figure_dir / f"{run_id}_event_errors.png"
    ratio_trends = figure_dir / f"{run_id}_ratio_trends.png"
    plot_model_metrics(manifest, model_metrics)
    plot_event_errors(manifest, event_errors)
    plot_ratio_trends(manifest, ratio_trends)
    return {
        "model_metrics": str(model_metrics),
        "event_errors": str(event_errors),
        "ratio_trends": str(ratio_trends),
    }


def build_report_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        f"# Experiment Report: {manifest['run_id']}",
        "",
        "## Experiment Configuration",
        f"- Created at: {manifest['created_at']}",
        f"- Run root: `{manifest['run_root']}`",
        f"- Models: {', '.join(entry['model'] for entry in manifest.get('models', [])) or 'none'}",
        f"- Observation ratios: {', '.join(str(ratio) for ratio in manifest.get('observation_ratios', [])) or 'unknown'}",
        "",
        "## Key Results",
    ]
    for entry in manifest.get("models", []):
        metrics = entry.get("overall_metrics", {})
        lines.append(
            f"- `{entry['model']}`: status={entry['status']}, MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}, "
            f"Pearson={metrics.get('pearson')}, Spearman={metrics.get('spearman')}"
        )

    lines.extend(
        [
            "",
            "## Figures",
            f"- Model metrics: `{manifest['figures']['model_metrics']}`",
            f"- Event errors: `{manifest['figures']['event_errors']}`",
            f"- Ratio trends: `{manifest['figures']['ratio_trends']}`",
            "",
            "## Alignment With idea.md",
        ]
    )
    for hypothesis in manifest.get("hypothesis_analysis", []):
        lines.append(f"### {hypothesis['id']} {hypothesis['title']}")
        lines.append(f"- Verdict: {hypothesis['verdict']}")
        for evidence in hypothesis.get("evidence", []):
            lines.append(f"- {evidence}")
    lines.extend(["", "## Mismatch Signals"])
    if manifest.get("anomalies"):
        for anomaly in manifest["anomalies"]:
            lines.append(f"### {anomaly['phenomenon']}")
            lines.append(f"- Possible cause: {anomaly['possible_cause']}")
            lines.append(f"- idea.md conflict: {anomaly['idea_conflict']}")
            lines.append(f"- Recommendation: {anomaly['recommendation']}")
    else:
        lines.append("- No material mismatch signals detected in this run.")
    lines.extend(
        [
            "",
            "## Next Step Recommendation",
            f"- {manifest['idea_followup']['summary']}",
        ]
    )
    return "\n".join(lines) + "\n"


def append_journal(paths: ExperimentPaths, manifest: dict[str, Any]) -> None:
    journal_path = paths.records / "experiment_journal.md"
    existing = journal_path.read_text("utf-8") if journal_path.exists() else "# Experiment Journal\n\n"
    anchor = f"## {manifest['run_id']}\n"
    if anchor in existing:
        return
    summary_lines = [anchor, f"- Created at: {manifest['created_at']}", f"- Run root: `{manifest['run_root']}`"]
    for entry in manifest.get("models", []):
        summary_lines.append(
            f"- {entry['model']}: MAE={entry['overall_metrics'].get('mae')}, RMSE={entry['overall_metrics'].get('rmse')}, status={entry['status']}"
        )
    summary_lines.append(f"- idea follow-up: {manifest['idea_followup']['summary']}")
    summary_lines.append("")
    journal_path.write_text(existing + "\n".join(summary_lines), encoding="utf-8")


def upsert_index(paths: ExperimentPaths, manifest: dict[str, Any]) -> None:
    index_path = paths.records / "experiment_index.csv"
    fieldnames = [
        "run_id",
        "created_at",
        "run_root",
        "best_model",
        "best_mae",
        "best_rmse",
        "hypothesis_summary",
        "suggest_new_idea",
        "report_path",
        "manifest_path",
    ]
    existing_rows = read_csv_rows(index_path)
    best = best_model_by_metric(manifest, "mae")
    new_row = {
        "run_id": manifest["run_id"],
        "created_at": manifest["created_at"],
        "run_root": manifest["run_root"],
        "best_model": best["model"] if best else "",
        "best_mae": best["overall_metrics"].get("mae") if best else "",
        "best_rmse": best["overall_metrics"].get("rmse") if best else "",
        "hypothesis_summary": "; ".join(f"{item['id']}={item['verdict']}" for item in manifest.get("hypothesis_analysis", [])),
        "suggest_new_idea": str(manifest["idea_followup"]["suggest_new_idea"]),
        "report_path": manifest["report_path"],
        "manifest_path": manifest["manifest_path"],
    }
    filtered = [row for row in existing_rows if row.get("run_id") != manifest["run_id"]]
    filtered.append(new_row)
    filtered.sort(key=lambda row: row["created_at"])
    write_csv_rows(index_path, filtered, fieldnames)


def build_manifest(run_id: str, run_root: str | Path, config: dict[str, Any], log_path: str | Path | None = None) -> dict[str, Any]:
    run_data = collect_run_results(run_root)
    manifest = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "run_root": str(Path(run_root)),
        "log_path": str(log_path) if log_path else "",
        "models": run_data["models"],
        "observation_ratios": run_data["observation_ratios"],
    }
    manifest["hypothesis_analysis"] = analyze_hypotheses(manifest, config)
    manifest["anomalies"] = detect_anomalies(manifest, config)
    manifest["idea_followup"] = recommendation_from_hypotheses(manifest["hypothesis_analysis"], manifest["anomalies"], config)
    return manifest


def persist_manifest_outputs(paths: ExperimentPaths, manifest: dict[str, Any]) -> dict[str, Any]:
    figures = create_figures(manifest, paths.figures, manifest["run_id"])
    manifest["figures"] = figures
    manifest_path = paths.manifests / f"{manifest['run_id']}.json"
    report_path = paths.records / f"{manifest['run_id']}.md"
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    write_json(manifest_path, manifest)
    report_path.write_text(build_report_markdown(manifest), encoding="utf-8")
    append_journal(paths, manifest)
    upsert_index(paths, manifest)
    return manifest


def run_command_with_logging(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return process.wait()
