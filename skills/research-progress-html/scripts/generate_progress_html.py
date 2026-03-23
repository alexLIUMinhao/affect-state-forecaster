from __future__ import annotations

import argparse
import csv
import html
import os
import re
from pathlib import Path
from string import Template
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_MODELS = [
    "text_baseline",
    "temporal_baseline",
    "structure_baseline",
    "affect_state_forecaster",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a paper-style research progress HTML page.")
    parser.add_argument("--idea-path", type=Path, default=PROJECT_ROOT / "idea.md")
    parser.add_argument("--experiments-root", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--output-html", type=Path, default=PROJECT_ROOT / "experiments/html/paper_progress.html")
    parser.add_argument("--mode", choices=("auto", "main", "ablation"), default="auto")
    parser.add_argument("--result-source", choices=("auto", "structured", "html"), default="auto")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "TODO"


def slug_sort_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, path.name)
    except OSError:
        return (0.0, path.name)


def extract_section(text: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else ""


def extract_first_meaningful_line(block: str) -> str:
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\*+|\*+$", "", line).strip()
        if line:
            return line
    return ""


def derive_idea_context(idea_text: str) -> dict[str, str]:
    title = extract_first_meaningful_line(extract_section(idea_text, "## 一、研究题目"))
    background = extract_section(idea_text, "## 二、研究背景与问题定位")
    task = extract_section(idea_text, "## 四、任务定义")
    data = extract_section(idea_text, "## 五、数据与 benchmark 构建方案")
    method = extract_section(idea_text, "## 六、方法方案")

    context = {
        "title": title or "面向公共事件的多模态情绪驱动演化预测研究",
        "dataset": "PHEME" if "PHEME" in idea_text else "TODO",
        "goal": "在事件早期观测阶段预测后续群体情绪走势",
        "task": "以 future_neg_ratio 为主任务，评估未来负面情绪比例预测能力",
        "method": "Affect-State Forecaster",
        "architecture": (
            "Affect-State Forecaster 采用“输入编码层 -> 时序聚合层 -> affect-state 中间层 -> 未来预测层”的四段式结构，"
            "用于把早期观测压缩为可解释的群体情绪状态，再预测后续走势。"
        ),
        "input": (
            "输入包括 source post、早期 replies、timestamps、reply tree / parent-child structure，"
            "第一阶段以语义、时间、结构三类模态为主。"
        ),
        "output": (
            "主输出为 future_neg_ratio 回归值；扩展输出可包括 future sentiment distribution "
            "([p_neg, p_neu, p_pos]) 或 majority sentiment 分析指标。"
        ),
    }
    if "future_neg_ratio" in task:
        context["task"] = "以 future_neg_ratio 为主任务，预测未来窗口中的负面情绪比例"
    if "Affect-State Forecaster" in method:
        context["method"] = "Affect-State Forecaster"
        context["architecture"] = (
            "Affect-State Forecaster 采用“输入编码层 -> 时序聚合层 -> affect-state 中间层 -> 未来预测层”的四段式结构，"
            "先估计当前群体 affect state，再进行未来情绪预测。"
        )
    if "事件线程" in background:
        context["goal"] = "在真实公共事件线程的早期观测阶段预测后续群体情绪走势"
    if "source text、reply texts、timestamps、reply tree / propagation structure" in task or "source text、reply texts" in task:
        context["input"] = (
            "输入包括 source text、reply texts、timestamps、reply tree / propagation structure，"
            "并允许后续扩展图像或外部事件上下文。"
        )
    if "future_neg_ratio" in task:
        context["output"] = (
            "主输出为 future_neg_ratio，即未来窗口中的负面情绪比例；扩展任务输出为未来情绪分布 "
            "([p_neg, p_neu, p_pos])。"
        )
    return context


def collect_structured_results(project_root: Path) -> dict[str, Any]:
    experiments_root = project_root / "experiments"
    records_root = experiments_root / "records"
    manifests_root = experiments_root / "manifests"
    figures_root = experiments_root / "figures"
    figure_candidates: list[str] = []
    if figures_root.exists():
        for image in sorted(figures_root.glob("*.png"), key=slug_sort_key, reverse=True):
            figure_candidates.append(str(image))
            if len(figure_candidates) == 3:
                break

    base_results: dict[str, Any] | None = None
    summary_candidates = sorted(
        (
            path
            for path in records_root.glob("*capacity_summary.csv")
            if "fusion_capacity" not in path.name
        ),
        key=slug_sort_key,
        reverse=True,
    )
    if summary_candidates:
        base_results = collect_capacity_summary_results(summary_candidates[0], figure_candidates)
    else:
        latest_main_summary = records_root / "latest_main_results.csv"
        if latest_main_summary.exists():
            base_results = collect_capacity_summary_results(latest_main_summary, figure_candidates)
        else:
            manifest_path = select_main_manifest(manifests_root)
            if manifest_path:
                base_results = collect_manifest_results(manifest_path, figure_candidates)

    fusion_candidates = sorted(records_root.glob("*fusion_diagnostic_summary.csv"), key=slug_sort_key, reverse=True)
    fusion_capacity_candidates = sorted(records_root.glob("*fusion_capacity_summary.csv"), key=slug_sort_key, reverse=True)
    source_gate_validation_candidates = sorted(records_root.glob("*source_gate_validation_summary.csv"), key=slug_sort_key, reverse=True)
    if base_results and fusion_candidates:
        base_results["fusion_diagnostic"] = collect_fusion_diagnostic_results(fusion_candidates[0])
        base_results["tables"].append(build_fusion_table(base_results["fusion_diagnostic"]))
        base_results["source_label"] = f"{base_results['source_label']} + {fusion_candidates[0].name}"
    if base_results and fusion_capacity_candidates:
        base_results["fusion_capacity_control"] = collect_fusion_capacity_results(fusion_capacity_candidates[0])
        base_results["tables"].append(build_fusion_capacity_table(base_results["fusion_capacity_control"]))
        base_results["source_label"] = f"{base_results['source_label']} + {fusion_capacity_candidates[0].name}"
    if base_results and source_gate_validation_candidates:
        base_results["source_gate_validation"] = collect_source_gate_validation_results(source_gate_validation_candidates[0])
        base_results["tables"].append(build_source_gate_validation_table(base_results["source_gate_validation"]))
        base_results["source_label"] = f"{base_results['source_label']} + {source_gate_validation_candidates[0].name}"
    if base_results:
        return base_results

    outputs_root = project_root / "outputs"
    models: dict[str, dict[str, Any]] = {}
    for model in DEFAULT_OUTPUT_MODELS:
        summary_path = outputs_root / model / "results_summary.csv"
        rows = read_csv_rows(summary_path)
        overall = next((row for row in rows if row.get("group_name") == "overall"), None)
        if not overall:
            continue
        models[model] = {
            "model": model,
            "dataset": "PHEME",
            "param_count": None,
            "capacity_group": "default",
            "event_metrics": [],
            "metrics": {
                "mae": to_float(overall.get("mae")),
                "rmse": to_float(overall.get("rmse")),
                "pearson": to_float(overall.get("pearson")),
                "spearman": to_float(overall.get("spearman")),
            },
        }
    return {
        "models": models,
        "tables": [{"key": "default", "title": "Table 1. 当前主实验（默认配置）", "models": models}],
        "current_group": "default",
        "figures": figure_candidates,
        "source_label": "outputs",
        "fusion_diagnostic": None,
        "fusion_capacity_control": None,
        "source_gate_validation": None,
    }


def select_main_manifest(manifests_root: Path) -> Path | None:
    preferred = manifests_root / "import_first_round_ratio_05.json"
    if preferred.exists():
        return preferred
    candidates = []
    for path in manifests_root.glob("*.json"):
        if any(token in path.name for token in ("labeler", "cross_event", "seed", "ablation")):
            continue
        try:
            import json

            payload = json.loads(read_text(path))
        except Exception:
            continue
        model_names = {entry.get("model") for entry in payload.get("models", [])}
        if model_names == set(DEFAULT_OUTPUT_MODELS):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=slug_sort_key, reverse=True)[0]


def collect_manifest_results(manifest_path: Path, figure_candidates: list[str]) -> dict[str, Any]:
    import json

    payload = json.loads(read_text(manifest_path))
    models: dict[str, dict[str, Any]] = {}
    for entry in payload.get("models", []):
        name = entry.get("model", "")
        models[name] = {
            "model": name,
            "dataset": "PHEME",
            "param_count": entry.get("param_count"),
            "capacity_group": entry.get("capacity_group", "default"),
            "event_metrics": entry.get("event_metrics", []),
            "metrics": {
                "mae": to_float(entry.get("overall_metrics", {}).get("mae")),
                "rmse": to_float(entry.get("overall_metrics", {}).get("rmse")),
                "pearson": to_float(entry.get("overall_metrics", {}).get("pearson")),
                "spearman": to_float(entry.get("overall_metrics", {}).get("spearman")),
            },
        }
    return {
        "models": models,
        "tables": [{"key": "default", "title": "Table 1. 当前主实验（默认配置）", "models": models}],
        "current_group": "default",
        "figures": figure_candidates,
        "source_label": manifest_path.name,
        "fusion_diagnostic": None,
        "fusion_capacity_control": None,
        "source_gate_validation": None,
    }


def collect_capacity_summary_results(summary_path: Path, figure_candidates: list[str]) -> dict[str, Any]:
    rows = read_csv_rows(summary_path)
    tables: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    run_ids = {row.get("run_id", "") for row in rows if row.get("run_id")}
    manifest_payloads: dict[str, Any] = {}
    manifests_root = summary_path.parents[1] / "manifests"
    for run_id in run_ids:
        manifest_path = manifests_root / f"{run_id}.json"
        if manifest_path.exists():
            import json

            manifest_payloads[run_id] = json.loads(read_text(manifest_path))
    for row in rows:
        capacity_group = row.get("capacity_group", "default") or "default"
        grouped.setdefault(capacity_group, {})
        model_name = row.get("model_name", "")
        event_metrics: list[dict[str, Any]] = []
        manifest_payload = manifest_payloads.get(row.get("run_id", ""))
        if manifest_payload:
            for entry in manifest_payload.get("models", []):
                if entry.get("model") == model_name:
                    event_metrics = entry.get("event_metrics", [])
                    break
        grouped[capacity_group][model_name] = {
            "model": model_name,
            "dataset": "PHEME",
            "param_count": int(float(row["param_count"])) if row.get("param_count") not in {"", None} else None,
            "capacity_group": capacity_group,
            "event_metrics": event_metrics,
            "metrics": {
                "mae": to_float(row.get("mae")),
                "rmse": to_float(row.get("rmse")),
                "pearson": to_float(row.get("pearson")),
                "spearman": to_float(row.get("spearman")),
            },
        }
    for key in ("default", "matched"):
        if key in grouped:
            title = "Table 1. 当前主实验（默认配置）" if key == "default" else "Table 2. 参数量对齐结果"
            tables.append({"key": key, "title": title, "models": grouped[key]})
    current_group = "matched" if "matched" in grouped else "default"
    return {
        "models": grouped.get(current_group, {}),
        "tables": tables,
        "current_group": current_group,
        "figures": figure_candidates,
        "source_label": summary_path.name,
        "fusion_diagnostic": None,
        "fusion_capacity_control": None,
        "source_gate_validation": None,
    }


def collect_fusion_diagnostic_results(summary_path: Path) -> dict[str, Any]:
    rows = read_csv_rows(summary_path)
    variants: list[dict[str, Any]] = []
    for row in rows:
        variants.append(
            {
                "model_name": row.get("model_name", ""),
                "fusion_variant": row.get("fusion_variant", ""),
                "param_count": int(float(row["param_count"])) if row.get("param_count") not in {"", None} else None,
                "metrics": {
                    "mae": to_float(row.get("mae")),
                    "rmse": to_float(row.get("rmse")),
                    "pearson": to_float(row.get("pearson")),
                    "spearman": to_float(row.get("spearman")),
                },
                "gate_means": {
                    "source": to_float(row.get("gate_source_mean")),
                    "temporal": to_float(row.get("gate_temporal_mean")),
                    "structure": to_float(row.get("gate_structure_mean")),
                },
            }
        )
    return {"summary_path": str(summary_path), "variants": variants}


def build_fusion_table(fusion_results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in fusion_results.get("variants", []):
        gate = item.get("gate_means", {})
        rows.append(
            {
                "variant": item.get("fusion_variant", ""),
                "model_name": item.get("model_name", ""),
                "param_count": item.get("param_count"),
                "metric_text": (
                    f"MAE={format_metric(item['metrics'].get('mae'))}, "
                    f"RMSE={format_metric(item['metrics'].get('rmse'))}, "
                    f"Pearson={format_metric(item['metrics'].get('pearson'))}, "
                    f"Spearman={format_metric(item['metrics'].get('spearman'))}"
                ),
                "gate_text": (
                    f"source={format_metric(gate.get('source'))}, "
                    f"temporal={format_metric(gate.get('temporal'))}, "
                    f"structure={format_metric(gate.get('structure'))}"
                ),
            }
        )
    return {
        "key": "fusion_diagnostic",
        "title": "Table 3. 融合筛选诊断结果",
        "headers": ["Variant", "Model", "Params", "Metrics", "Gate Means"],
        "rows": rows,
    }


def collect_fusion_capacity_results(summary_path: Path) -> dict[str, Any]:
    rows = read_csv_rows(summary_path)
    variants: list[dict[str, Any]] = []
    for row in rows:
        variants.append(
            {
                "capacity_group": row.get("capacity_group", ""),
                "model_name": row.get("model_name", ""),
                "fusion_variant": row.get("fusion_variant", ""),
                "param_count": int(float(row["param_count"])) if row.get("param_count") not in {"", None} else None,
                "metrics": {
                    "mae": to_float(row.get("mae")),
                    "rmse": to_float(row.get("rmse")),
                    "pearson": to_float(row.get("pearson")),
                    "spearman": to_float(row.get("spearman")),
                },
                "gate_means": {
                    "source": to_float(row.get("gate_source_mean")),
                    "temporal": to_float(row.get("gate_temporal_mean")),
                    "structure": to_float(row.get("gate_structure_mean")),
                },
            }
        )
    return {"summary_path": str(summary_path), "variants": variants}


def build_fusion_capacity_table(fusion_capacity_results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in fusion_capacity_results.get("variants", []):
        gate = item.get("gate_means", {})
        rows.append(
            {
                "variant": f"{item.get('capacity_group', '')}:{item.get('fusion_variant', '')}",
                "model_name": item.get("model_name", ""),
                "param_count": item.get("param_count"),
                "metric_text": (
                    f"MAE={format_metric(item['metrics'].get('mae'))}, "
                    f"RMSE={format_metric(item['metrics'].get('rmse'))}, "
                    f"Pearson={format_metric(item['metrics'].get('pearson'))}, "
                    f"Spearman={format_metric(item['metrics'].get('spearman'))}"
                ),
                "gate_text": (
                    f"source={format_metric(gate.get('source'))}, "
                    f"temporal={format_metric(gate.get('temporal'))}, "
                    f"structure={format_metric(gate.get('structure'))}"
                ),
            }
        )
    return {
        "key": "fusion_capacity_control",
        "title": "Table 4. 门控变体的公平容量控制结果",
        "headers": ["Variant", "Model", "Params", "Metrics", "Gate Means"],
        "rows": rows,
    }


def collect_source_gate_validation_results(summary_path: Path) -> dict[str, Any]:
    rows = read_csv_rows(summary_path)
    variants: list[dict[str, Any]] = []
    for row in rows:
        variants.append(
            {
                "capacity_group": row.get("capacity_group", ""),
                "model_name": row.get("model_name", ""),
                "fusion_variant": row.get("fusion_variant", ""),
                "seeds": row.get("seeds", ""),
                "param_count": int(float(row["param_count"])) if row.get("param_count") not in {"", None} else None,
                "param_count_std": to_float(row.get("param_count_std")),
                "metrics": {
                    "mae": to_float(row.get("mae")),
                    "mae_std": to_float(row.get("mae_std")),
                    "rmse": to_float(row.get("rmse")),
                    "rmse_std": to_float(row.get("rmse_std")),
                    "pearson": to_float(row.get("pearson")),
                    "pearson_std": to_float(row.get("pearson_std")),
                    "spearman": to_float(row.get("spearman")),
                    "spearman_std": to_float(row.get("spearman_std")),
                },
                "gate_means": {
                    "source": to_float(row.get("gate_source_mean")),
                    "source_std": to_float(row.get("gate_source_mean_std")),
                    "temporal": to_float(row.get("gate_temporal_mean")),
                    "temporal_std": to_float(row.get("gate_temporal_mean_std")),
                    "structure": to_float(row.get("gate_structure_mean")),
                    "structure_std": to_float(row.get("gate_structure_mean_std")),
                },
            }
        )
    return {"summary_path": str(summary_path), "variants": variants}


def build_source_gate_validation_table(validation_results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in validation_results.get("variants", []):
        gate = item.get("gate_means", {})
        metrics = item.get("metrics", {})
        rows.append(
            {
                "variant": f"{item.get('capacity_group', '')}:{item.get('fusion_variant', '')}",
                "model_name": item.get("model_name", ""),
                "param_count": item.get("param_count"),
                "metric_text": (
                    f"MAE={format_metric(metrics.get('mae'))}±{format_metric(metrics.get('mae_std'))}, "
                    f"RMSE={format_metric(metrics.get('rmse'))}±{format_metric(metrics.get('rmse_std'))}, "
                    f"Pearson={format_metric(metrics.get('pearson'))}±{format_metric(metrics.get('pearson_std'))}, "
                    f"Spearman={format_metric(metrics.get('spearman'))}±{format_metric(metrics.get('spearman_std'))}"
                ),
                "gate_text": (
                    f"source={format_metric(gate.get('source'))}, "
                    f"temporal={format_metric(gate.get('temporal'))}, "
                    f"structure={format_metric(gate.get('structure'))}; "
                    f"seeds={html.escape(item.get('seeds', ''))}"
                ),
            }
        )
    return {
        "key": "source_gate_validation",
        "title": "Table 5. source_gate_only 的多 seed 稳定性验证",
        "headers": ["Variant", "Model", "Params", "Metrics", "Gate Means / Seeds"],
        "rows": rows,
    }


def strip_tags(raw_html: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw_html)
    return html.unescape(" ".join(text.split()))


def parse_metrics_cell(cell: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"mae": None, "rmse": None, "pearson": None, "spearman": None}
    for metric in metrics:
        match = re.search(rf"{metric}\s*[=:]\s*(-?\d+(?:\.\d+)?)", cell, flags=re.I)
        if match:
            metrics[metric] = float(match.group(1))
    return metrics


def collect_html_results(experiments_root: Path) -> dict[str, Any]:
    html_dir = experiments_root / "html"
    models: dict[str, dict[str, Any]] = {}
    figures: list[str] = []
    if not html_dir.exists():
        return {"models": models, "figures": figures}

    candidates = [path for path in html_dir.glob("*paper_progress.html") if path.name != "index.html"]
    if not candidates:
        candidates = [path for path in html_dir.glob("*.html") if path.name not in {"index.html", "paper_progress.html"}]
    if not candidates:
        return {"models": models, "tables": [], "current_group": "default", "figures": figures, "source_label": "html"}
    latest = sorted(candidates, key=slug_sort_key, reverse=True)[0]
    page = read_text(latest)

    row_matches = re.findall(r"<tr><td>([^<]+)</td><td>([^<]+)</td><td>([^<]*)</td><td(?: class='mono')?>([^<]*)</td><td(?: class='mono')?>([^<]*)</td></tr>", page)
    for first_col, second_col, third_col, _params_or_metric, metrics_cell in row_matches:
        model_name = third_col if third_col and third_col != "—" else second_col
        parsed = parse_metrics_cell(metrics_cell)
        models[model_name] = {
            "model": model_name,
            "dataset": "PHEME",
            "param_count": None,
            "capacity_group": "default",
            "event_metrics": [],
            "metrics": {
                "mae": parsed.get("mae"),
                "rmse": parsed.get("rmse"),
                "pearson": parsed.get("pearson"),
                "spearman": parsed.get("spearman"),
            },
        }

    for src in re.findall(r"<img src='([^']+)'", page):
        figures.append(str((html_dir / src).resolve()))
    return {
        "models": models,
        "tables": [{"key": "default", "title": "Table 1. 当前主实验（默认配置）", "models": models}],
        "current_group": "default",
        "figures": figures,
        "source_label": latest.name,
        "fusion_diagnostic": None,
        "fusion_capacity_control": None,
        "source_gate_validation": None,
    }


def select_result_source(project_root: Path, experiments_root: Path, result_source: str) -> dict[str, Any]:
    if result_source == "structured":
        return collect_structured_results(project_root)
    if result_source == "html":
        return collect_html_results(experiments_root)

    structured = collect_structured_results(project_root)
    if structured["models"]:
        return structured
    return collect_html_results(experiments_root)


def compute_evidence(results: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = results.get("models", {})
    text_metrics = models.get("text_baseline", {}).get("metrics", {})
    ours_metrics = models.get("affect_state_forecaster", {}).get("metrics", {})
    structure_metrics = models.get("structure_baseline", {}).get("metrics", {})
    temporal_metrics = models.get("temporal_baseline", {}).get("metrics", {})

    best_model = "Pending"
    best_mae = None
    best_rmse_model = "Pending"
    best_rmse = None
    best_corr_model = "Pending"
    best_corr_score = None
    for name, payload in models.items():
        metrics = payload.get("metrics", {})
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        corr_score = max(value for value in (metrics.get("pearson"), metrics.get("spearman")) if value is not None) if any(
            value is not None for value in (metrics.get("pearson"), metrics.get("spearman"))
        ) else None
        if mae is not None and (best_mae is None or mae < best_mae):
            best_model = name
            best_mae = mae
        if rmse is not None and (best_rmse is None or rmse < best_rmse):
            best_rmse_model = name
            best_rmse = rmse
        if corr_score is not None and (best_corr_score is None or corr_score > best_corr_score):
            best_corr_model = name
            best_corr_score = corr_score

    text_mae = text_metrics.get("mae")
    ours_mae = ours_metrics.get("mae")
    structure_mae = structure_metrics.get("mae")
    temporal_mae = temporal_metrics.get("mae")

    mae_gain_vs_text = None
    if text_mae is not None and best_mae is not None:
        mae_gain_vs_text = text_mae - best_mae

    ours_vs_structure = None
    if ours_mae is not None and structure_mae is not None:
        ours_vs_structure = structure_mae - ours_mae

    ours_vs_temporal = None
    if ours_mae is not None and temporal_mae is not None:
        ours_vs_temporal = temporal_mae - ours_mae

    evidence_status = "insufficient"
    anomalies: list[str] = []
    if len(models) >= 2 and mae_gain_vs_text is not None and mae_gain_vs_text > 0:
        evidence_status = "task_supported"
        if best_model == "affect_state_forecaster":
            evidence_status = "ours_supported"
        elif best_model in {"structure_baseline", "temporal_baseline"}:
            evidence_status = "diagnostic_needed"
    if best_mae is None:
        anomalies.append("缺少可读的总体 MAE，当前只能做占位式总结。")
    if best_model != "Pending" and ours_mae is not None and best_model != "affect_state_forecaster":
        anomalies.append("当前最优模型不是 affect_state_forecaster，方法主线仍需诊断。")
    if best_model == "structure_baseline":
        anomalies.append("结构基线领先，说明结构信息有效，但 affect-state 叙事尚未闭合。")
    if best_model != "Pending" and best_rmse_model != "Pending" and best_model != best_rmse_model:
        anomalies.append("MAE 与 RMSE 的最优模型不一致，说明平均误差和极端误差控制可能分裂。")
    if any((metrics.get("pearson") or 0.0) <= 0.0 or (metrics.get("spearman") or 0.0) <= 0.0 for metrics in (text_metrics, temporal_metrics, structure_metrics, ours_metrics)):
        anomalies.append("部分模型的相关性接近 0 或为负，说明排序关系学习仍然偏弱。")

    return {
        "best_model": best_model,
        "best_mae": best_mae,
        "best_mae_model": best_model,
        "best_rmse_model": best_rmse_model,
        "best_rmse": best_rmse,
        "best_corr_model": best_corr_model,
        "best_corr_score": best_corr_score,
        "mae_gain_vs_text": mae_gain_vs_text,
        "ours_model": "affect_state_forecaster",
        "ours_vs_structure": ours_vs_structure,
        "ours_vs_temporal": ours_vs_temporal,
        "evidence_status": evidence_status,
        "anomalies": anomalies,
    }


def compute_reason_diagnosis(results: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, str]]:
    fusion = results.get("fusion_diagnostic") or {}
    variants = fusion.get("variants", [])
    fusion_capacity = results.get("fusion_capacity_control") or {}
    fusion_capacity_variants = fusion_capacity.get("variants", [])
    source_gate_validation = results.get("source_gate_validation") or {}
    source_gate_validation_variants = source_gate_validation.get("variants", [])
    asf_full = next((item for item in variants if item.get("fusion_variant") == "full"), None)
    best_gate = None
    for item in variants:
        if item.get("model_name") != "affect_state_forecaster" or item.get("fusion_variant") == "full":
            continue
        if best_gate is None or (item["metrics"].get("mae") or float("inf")) < (best_gate["metrics"].get("mae") or float("inf")):
            best_gate = item

    fusion_status = "待进一步验证"
    fusion_reason = "当前还没有门控变体结果，暂时只能依据已有消融猜测融合筛选问题。"
    if asf_full and best_gate:
        full_mae = asf_full["metrics"].get("mae")
        full_rmse = asf_full["metrics"].get("rmse")
        gate_mae = best_gate["metrics"].get("mae")
        gate_rmse = best_gate["metrics"].get("rmse")
        gate_corr = max(best_gate["metrics"].get("pearson") or -1.0, best_gate["metrics"].get("spearman") or -1.0)
        full_corr = max(asf_full["metrics"].get("pearson") or -1.0, asf_full["metrics"].get("spearman") or -1.0)
        if gate_mae is not None and full_mae is not None and gate_mae < full_mae and (full_rmse is None or gate_rmse is None or gate_rmse <= full_rmse + 0.01):
            fusion_status = "当前证据最强"
            fusion_reason = f"最佳门控变体 {best_gate['fusion_variant']} 优于 asf_full，说明问题更像是筛选机制缺失，而不是多模态本身无效。"
        elif gate_corr > full_corr:
            fusion_status = "部分成立"
            fusion_reason = f"最佳门控变体 {best_gate['fusion_variant']} 主要改善相关性，说明筛选机制可能更影响趋势保持而非点误差。"
        else:
            fusion_status = "待削弱"
            fusion_reason = "现有门控变体没有稳定优于 asf_full，融合筛选问题仍需谨慎判断。"
        if fusion_status in {"当前证据最强", "部分成立"} and fusion_capacity_variants:
            matched_full = next(
                (item for item in fusion_capacity_variants if item.get("capacity_group") == "matched" and item.get("fusion_variant") == "full"),
                None,
            )
            matched_best_gate = next(
                (
                    item
                    for item in fusion_capacity_variants
                    if item.get("capacity_group") == "matched" and item.get("fusion_variant") == best_gate.get("fusion_variant")
                ),
                None,
            )
            if matched_full and matched_best_gate:
                matched_full_mae = matched_full["metrics"].get("mae")
                matched_gate_mae = matched_best_gate["metrics"].get("mae")
                if matched_full_mae is not None and matched_gate_mae is not None:
                    if matched_gate_mae < matched_full_mae:
                        fusion_reason += f" 在公平容量下，{best_gate['fusion_variant']} 仍优于 matched full，说明筛选机制收益并不完全依赖更大容量。"
                    else:
                        fusion_reason += f" 但在公平容量下，{best_gate['fusion_variant']} 不再优于 matched full，说明筛选机制有效但仍依赖较大容量承载多模态特征。"
        matched_source_gate = next(
            (
                item
                for item in source_gate_validation_variants
                if item.get("capacity_group") == "matched" and item.get("fusion_variant") == "source_gate_only"
            ),
            None,
        )
        matched_source_full = next(
            (
                item
                for item in source_gate_validation_variants
                if item.get("capacity_group") == "matched" and item.get("fusion_variant") == "full"
            ),
            None,
        )
        if matched_source_gate and matched_source_full:
            matched_source_gate_mae = matched_source_gate["metrics"].get("mae")
            matched_source_full_mae = matched_source_full["metrics"].get("mae")
            if matched_source_gate_mae is not None and matched_source_full_mae is not None:
                if matched_source_gate_mae < matched_source_full_mae:
                    fusion_reason += " 进一步的多 seed 验证表明，source_gate_only 在公平容量下仍略优于 matched full，说明轻量筛选机制具备一定稳健性。"
                else:
                    fusion_reason += " 进一步的多 seed 验证没有支持 source_gate_only 在公平容量下稳定优于 matched full，轻量筛选收益仍需继续确认。"

    capacity_status = "部分成立"
    capacity_reason = "容量对齐后 ASF 明显变弱，说明多模态表征确实受模型容量影响，但这不足以单独解释全部现象。"
    other_status = "待进一步验证"
    other_reason = "若门控变体依然没有收益，下一步应转向目标建模、排序学习和弱标签噪声诊断。"

    return [
        {"title": "容量因素", "status": capacity_status, "reason": capacity_reason},
        {"title": "融合筛选问题", "status": fusion_status, "reason": fusion_reason},
        {"title": "其他问题", "status": other_status, "reason": other_reason},
    ]


def build_introduction(context: dict[str, str], evidence: dict[str, Any]) -> list[str]:
    dataset = context["dataset"]
    method = context["method"]
    paragraph_one = (
        f"本研究面向公共事件线程中的群体情绪演化预测，目标是在早期观测阶段利用文本、时间与传播结构线索，"
        f"对后续窗口中的群体情绪走势进行前瞻预测。当前主任务围绕 {context['task']} 展开，"
        f"并以 {dataset} 作为第一阶段 benchmark 重构与验证的主要数据基础。"
    )

    best_model = evidence["best_model"]
    mae_gain = evidence["mae_gain_vs_text"]
    ours_vs_structure = evidence["ours_vs_structure"]
    ours_vs_temporal = evidence["ours_vs_temporal"]
    status = evidence["evidence_status"]
    anomaly_count = len(evidence["anomalies"])
    best_rmse_model = evidence.get("best_rmse_model", best_model)
    best_corr_model = evidence.get("best_corr_model", best_model)

    if status == "ours_supported":
        paragraph_two = (
            f"从当前关键结果看，{best_model} 已经取得本轮最优 MAE={format_metric(evidence['best_mae'])}，"
            f"相对文本基线的 MAE 改进为 {format_metric(mae_gain)}。这说明 {method} 所依赖的显式 affect-state 建模"
            f"获得了初步支持；同时，方法相对结构基线和时序基线的比较分别为 {format_metric(ours_vs_structure)} 与 "
            f"{format_metric(ours_vs_temporal)}，因此现阶段可以把论文叙事重心放在“情绪状态作为中间驱动变量”的有效性上。"
        )
    elif status in {"diagnostic_needed", "task_supported"}:
        if evidence.get("fusion_diagnosis_status") == "当前证据最强":
            paragraph_two = (
                f"当前主任务已经证明多模态信息并非天然无效，但新增诊断更倾向于说明问题不在多模态本身，而在缺少信息筛选机制。"
                f"在保持任务与训练设置不变的前提下，门控筛选变体相对 asf_full 出现改进，说明当前直接融合可能把无效或冲突线索一并送入预测路径。"
            )
        elif evidence.get("fusion_diagnosis_status") == "待削弱":
            paragraph_two = (
                f"当前结果首先支持了任务本身的可做性，但门控筛选变体并没有稳定优于 asf_full。"
                f"这说明当前瓶颈更可能来自目标建模、标签噪声或优化方式，而不是简单的融合策略本身。"
            )
        elif best_model == "structure_baseline" and best_rmse_model == "affect_state_forecaster":
            paragraph_two = (
                f"当前结果首先支持了任务本身的可做性：最佳 MAE 模型为 {best_model}，其 MAE={format_metric(evidence['best_mae'])}，"
                f"相对文本基线的改进幅度为 {format_metric(mae_gain)}；但最佳 RMSE 模型是 {best_rmse_model}，"
                f"说明结构基线在点预测平均误差上更强，而 {method} 在极端误差控制上呈现局部优势。"
                f"再结合当前相关性最优模型为 {best_corr_model}，更稳妥的叙事应是“多模态 affect-state 路线仍值得在公平容量下继续诊断”，"
                f"而不是直接宣称方法已经稳定领先。"
            )
        else:
            paragraph_two = (
                f"当前结果首先支持了任务本身的可做性：最佳模型为 {best_model}，其 MAE={format_metric(evidence['best_mae'])}，"
                f"相对文本基线的改进幅度为 {format_metric(mae_gain)}。但 {method} 还没有稳定领先，"
                f"其相对结构基线与时序基线的 MAE 差分别为 {format_metric(ours_vs_structure)} 和 {format_metric(ours_vs_temporal)}；"
                f"再加上当前检测到 {anomaly_count} 个需要解释的风险信号，因此更稳妥的叙事应是“任务成立、结构/时序信息有效、"
                f"而 affect-state 主线仍需进一步诊断”。"
            )
    else:
        paragraph_two = (
            f"当前证据仍不充分，现阶段更适合将结果表述为 benchmark 与实验设定的可行性验证，而不是方法优势的定论。"
            f"目前可确认的信号包括：最佳模型为 {best_model}，相对文本基线的 MAE 改进为 {format_metric(mae_gain)}；"
            f"但结果覆盖仍不完整，因此 introduction 需要保持保守措辞。"
        )

    return [paragraph_one, paragraph_two]


def build_signal_items(evidence: dict[str, Any]) -> list[str]:
    items = [
        f"当前 MAE 最优模型：<span class='mono'>{html.escape(evidence['best_mae_model'])}</span>，MAE={format_metric(evidence['best_mae'])}。",
        f"当前 RMSE 最优模型：<span class='mono'>{html.escape(evidence['best_rmse_model'])}</span>，RMSE={format_metric(evidence['best_rmse'])}。",
        f"相对文本基线的 MAE 改进：{format_metric(evidence['mae_gain_vs_text'])}。",
        f"`affect_state_forecaster` 相对结构基线的 MAE 差：{format_metric(evidence['ours_vs_structure'])}。",
        f"`affect_state_forecaster` 相对时序基线的 MAE 差：{format_metric(evidence['ours_vs_temporal'])}。",
        f"当前相关性最优模型：<span class='mono'>{html.escape(evidence['best_corr_model'])}</span>。",
        f"当前证据状态：{html.escape(evidence['evidence_status'])}。",
    ]
    if evidence["anomalies"]:
        items.append(f"主要风险：{html.escape(evidence['anomalies'][0])}")
    return items


def format_param_count(value: int | None) -> str:
    if value is None:
        return "TODO"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def table_row(dataset: str, baseline: str, ours: str, metrics: dict[str, float | None], param_count: int | None) -> str:
    metric_text = (
        f"MAE={format_metric(metrics.get('mae'))}, "
        f"RMSE={format_metric(metrics.get('rmse'))}, "
        f"Pearson={format_metric(metrics.get('pearson'))}, "
        f"Spearman={format_metric(metrics.get('spearman'))}"
    )
    return (
        "<tr>"
        f"<td>{html.escape(dataset)}</td>"
        f"<td>{html.escape(baseline)}</td>"
        f"<td>{html.escape(ours)}</td>"
        f"<td class='mono'>{html.escape(format_param_count(param_count))}</td>"
        f"<td class='mono'>{html.escape(metric_text)}</td>"
        "</tr>"
    )


def render_table(context: dict[str, str], table: dict[str, Any]) -> str:
    models = table.get("models", {})
    dataset = context["dataset"]
    rows = []

    baseline_order = ["text_baseline", "temporal_baseline", "structure_baseline"]
    for baseline in baseline_order:
        payload = models.get(baseline)
        rows.append(
            table_row(
                dataset,
                baseline,
                "—",
                payload.get("metrics", {}) if payload else {},
                payload.get("param_count") if payload else None,
            )
        )

    ours_payload = models.get("affect_state_forecaster")
    rows.append(
        table_row(
            dataset,
            "—",
            "affect_state_forecaster",
            ours_payload.get("metrics", {}) if ours_payload else {},
            ours_payload.get("param_count") if ours_payload else None,
        )
    )

    return (
        f"<h3>{html.escape(table.get('title', 'Table 1'))}</h3>"
        "<table>"
        "<thead><tr><th>Dataset</th><th>Compared Baselines</th><th>Ours</th><th>Params</th><th>Metrics</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_custom_table(table: dict[str, Any]) -> str:
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    row_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('variant', '')))}</td>"
        f"<td>{html.escape(str(row.get('model_name', '')))}</td>"
        f"<td class='mono'>{html.escape(format_param_count(row.get('param_count')) if row.get('param_count') is not None else 'TODO')}</td>"
        f"<td class='mono'>{html.escape(str(row.get('metric_text', '')))}</td>"
        f"<td class='mono'>{html.escape(str(row.get('gate_text', '')))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        f"<h3>{html.escape(table.get('title', 'Table'))}</h3>"
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{row_html}</tbody>"
        "</table>"
    )


def build_table_html(context: dict[str, str], results: dict[str, Any]) -> tuple[str, str]:
    tables = results.get("tables", [])
    if not tables:
        return render_table(context, {"title": "Table 1. 当前主实验（默认配置）", "models": results.get("models", {})}), ""
    primary = render_custom_table(tables[0]) if tables[0].get("rows") else render_table(context, tables[0])
    secondary = "".join(render_custom_table(table) if table.get("rows") else render_table(context, table) for table in tables[1:])
    return primary, secondary


def build_progress_summary(evidence: dict[str, Any]) -> str:
    if evidence["evidence_status"] == "ours_supported":
        return (
            "当前主实验已经形成可用于论文原型叙事的正向证据：ours 暂时领先，且相对文本基线存在明确改进。"
        )
    if evidence["best_mae_model"] != evidence["best_rmse_model"]:
        return "当前主实验呈现指标分裂：structure 在平均误差上更优，而 ASF 在大误差惩罚上更稳，因此需要用容量对齐实验继续诊断。"
    if evidence["evidence_status"] in {"diagnostic_needed", "task_supported"}:
        return (
            "当前主实验首先证明了任务成立，但优势主要来自结构或时序线索，ours 仍需要通过诊断实验进一步澄清。"
        )
    return "当前主实验结果仍不完整，因此本页以可行性说明和占位式结果汇总为主。"


def build_table_note(evidence: dict[str, Any]) -> str:
    if evidence["best_mae_model"] != evidence["best_rmse_model"]:
        return (
            f"当前结果存在 MAE/RMSE 分裂：{evidence['best_mae_model']} 的平均误差最低，"
            f"而 {evidence['best_rmse_model']} 对大误差更稳。这通常意味着两类模型分别擅长“均值拟合”与“极端误差控制”，"
            "不能只靠单一指标下结论。"
        )
    if evidence["best_model"] == "affect_state_forecaster":
        return (
            "Table 1 显示 affect_state_forecaster 当前领先，说明显式情绪状态建模已具备进入多 ratio 与泛化验证的基础。"
        )
    if evidence["best_model"] in {"structure_baseline", "temporal_baseline"}:
        return (
            f"Table 1 显示 {evidence['best_model']} 当前领先，说明结构或时序信息对未来情绪预测有效；"
            "但 affect-state 方法仍需通过消融和正则化实验解释其尚未稳定领先的原因。"
        )
    return "Table 1 当前仍包含占位项，适合作为实验记录入口，但还不应承载过强论文结论。"


def build_metric_reading(evidence: dict[str, Any], results: dict[str, Any]) -> str:
    models = results.get("models", {})
    structure = models.get("structure_baseline", {}).get("metrics", {})
    ours = models.get("affect_state_forecaster", {}).get("metrics", {})
    warnings: list[str] = []
    if evidence["best_mae_model"] != evidence["best_rmse_model"]:
        warnings.append(
            f"{evidence['best_mae_model']} 的 MAE 更低，但 {evidence['best_rmse_model']} 的 RMSE 更低，说明平均误差与极端误差控制分裂。"
        )
    if (structure.get("pearson") or 0.0) <= 0.0 or (structure.get("spearman") or 0.0) <= 0.0:
        warnings.append("structure_baseline 的相关性偏弱，说明它虽然误差较低，但未必学到了稳定的排序关系。")
    if (ours.get("pearson") or 0.0) <= 0.0 or (ours.get("spearman") or 0.0) <= 0.0:
        warnings.append("affect_state_forecaster 的相关性仍不够稳，需要结合容量对齐和校准诊断一起解释。")
    if not warnings:
        warnings.append("当前误差和相关性方向基本一致，可以把主叙事建立在同一组指标上。")
    return " ".join(warnings)


def summarize_event_head_to_head(results: dict[str, Any]) -> str:
    models = results.get("models", {})
    structure_rows = models.get("structure_baseline", {}).get("event_metrics", [])
    ours_rows = models.get("affect_state_forecaster", {}).get("event_metrics", [])
    if not structure_rows or not ours_rows:
        return "当前结构化结果未包含事件级误差细节，暂时无法判断哪些事件更偏向结构建模、哪些事件更偏向 affect-state。"
    structure_map = {row.get("event_name"): row for row in structure_rows}
    ours_map = {row.get("event_name"): row for row in ours_rows}
    structure_better: list[str] = []
    ours_rmse_better: list[str] = []
    for event_name in sorted(set(structure_map) & set(ours_map)):
        structure_mae = to_float(structure_map[event_name].get("mae"))
        ours_mae = to_float(ours_map[event_name].get("mae"))
        structure_rmse = to_float(structure_map[event_name].get("rmse"))
        ours_rmse = to_float(ours_map[event_name].get("rmse"))
        if structure_mae is not None and ours_mae is not None and structure_mae < ours_mae:
            structure_better.append(event_name)
        if structure_rmse is not None and ours_rmse is not None and ours_rmse < structure_rmse:
            ours_rmse_better.append(event_name)
    structure_text = "、".join(structure_better) if structure_better else "暂无明显事件"
    ours_text = "、".join(ours_rmse_better) if ours_rmse_better else "暂无明显事件"
    return f"按当前事件级结果，structure_baseline 在 {structure_text} 上的平均误差更稳；affect_state_forecaster 在 {ours_text} 上更能控制极端误差。"


def build_metric_guide_html() -> str:
    items = [
        ("MAE", "平均绝对误差，反映平均偏差大小。越低越好。"),
        ("RMSE", "均方根误差，对大误差更敏感。越低越好。"),
        ("Pearson", "线性相关性，越接近 1 越说明预测值与真实值线性同步。越高越好。"),
        ("Spearman", "秩相关性，越接近 1 越说明排序趋势一致。越高越好。"),
    ]
    return "".join(
        f"<section class='metric-item'><strong>{html.escape(name)}</strong><span>{html.escape(desc)}</span></section>"
        for name, desc in items
    )


def build_next_plan(evidence: dict[str, Any]) -> list[str]:
    if evidence.get("fusion_diagnosis_status") == "当前证据最强":
        return [
            "保留表现最好的 1-2 个门控变体，并追加一轮对齐到 structure_baseline 的容量控制实验。",
            "记录门控均值与事件级误差，确认增益来自信息筛选而不是偶然拟合。",
            "若公平容量下收益仍成立，再把论文叙事升级为“问题在筛选机制而非多模态本身”。",
        ]
    if evidence.get("fusion_diagnosis_status") == "待削弱":
        return [
            "暂停继续设计更多融合结构，转向目标建模、标签噪声和排序学习诊断。",
            "补充预测值分布、残差分位和按事件相关性分析，确认是否存在均值化或排序失效问题。",
            "若这些分析给出清晰异常，再针对损失函数或标签质量设计下一轮实验。",
        ]
    if evidence["best_mae_model"] != evidence["best_rmse_model"]:
        return [
            "先做参数量对齐实验，以 structure_baseline 为容量锚点，检查 ASF 在公平容量下是否仍保留 RMSE 或趋势优势。",
            "补充残差分位统计和事件级 head-to-head，对“平均误差更优”与“极端误差更稳”的分裂现象做诊断。",
            "若对齐后 ASF 仍未在 MAE 上领先，则收缩论文叙事，避免把优势过早归因于 affect-state 建模。",
        ]
    if evidence["evidence_status"] == "ours_supported":
        return [
            "扩展到多 observation ratio，检查 affect-state 优势是否在 30/70、50/50、70/30 三种设置下保持稳定。",
            "补做 cross-event 或跨数据集验证，确认当前优势不是由少数事件模式驱动。",
            "增加稳健性分析，例如不同 sentiment labeler 或随机种子的重复实验。",
        ]
    if evidence["evidence_status"] in {"diagnostic_needed", "task_supported"}:
        return [
            "优先做 affect-state 诊断实验，明确中间状态层、正则项和结构融合模块各自的真实贡献。",
            "设计针对性的消融实验，验证去掉结构建模、去掉 affect-state 层、只保留时序建模时的性能变化。",
            "在不扩大任务叙事的前提下，补充正则化或结构感知融合改进，先解决“ours 未稳定领先”的核心问题。",
        ]
    return [
        "先补齐结构化结果文件，确保各模型的总体指标与关键图表可以被稳定读取。",
        "优先验证 benchmark 和评测协议，而不是继续扩充模型复杂度。",
        "待证据完整后，再决定 introduction 是否可以升级为方法导向叙事。",
    ]


def build_figures_html(output_html: Path, figures: list[str]) -> str:
    existing = [Path(item) for item in figures if item and Path(item).exists()]
    if not existing:
        return ""
    cards = []
    for image in existing[:3]:
        rel = os.path.relpath(image, start=output_html.parent)
        cards.append(f"<img src='{html.escape(rel)}' alt='{html.escape(image.stem)}' />")
    return "<div class='figure-grid'>" + "".join(cards) + "</div>"


def render_html(template_path: Path, payload: dict[str, str]) -> str:
    template = Template(read_text(template_path))
    return template.safe_substitute(payload)


def generate_html(args: argparse.Namespace) -> tuple[str, dict[str, Any], dict[str, Any]]:
    idea_text = read_text(args.idea_path)
    context = derive_idea_context(idea_text)
    results = select_result_source(PROJECT_ROOT, args.experiments_root, args.result_source)
    evidence = compute_evidence(results)
    reason_diagnosis = compute_reason_diagnosis(results, evidence)
    evidence["fusion_diagnosis_status"] = next(
        (item["status"] for item in reason_diagnosis if item["title"] == "融合筛选问题"),
        "待进一步验证",
    )

    introduction = build_introduction(context, evidence)
    signals = build_signal_items(evidence)
    plan_items = build_next_plan(evidence)
    table_html, secondary_tables_html = build_table_html(context, results)

    payload = {
        "page_title": f"{context['title']} - 研究进展",
        "hero_eyebrow": "Research Progress HTML",
        "hero_title": context["title"],
        "hero_subtitle": "根据研究目标与当前关键实验结果生成的论文式进展页，默认服务于主实验叙事。",
        "current_stage": "主实验优先",
        "experiment_type": "主实验" if args.mode in {"auto", "main"} else "消融实验",
        "best_model": evidence["best_model"],
        "introduction_html": "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in introduction),
        "signals_html": "".join(f"<li>{item}</li>" for item in signals),
        "architecture_text": html.escape(context["architecture"]),
        "input_text": html.escape(context["input"]),
        "output_text": html.escape(context["output"]),
        "progress_summary": html.escape(build_progress_summary(evidence)),
        "metric_guide_html": build_metric_guide_html(),
        "reason_diagnosis_html": "".join(
            "<section class='diagnosis-card'>"
            f"<h3>{html.escape(item['title'])}</h3>"
            f"<p>{html.escape(item['reason'])}</p>"
            f"<span class='diagnosis-status'>{html.escape(item['status'])}</span>"
            "</section>"
            for item in reason_diagnosis
        ),
        "table_html": table_html,
        "secondary_tables_html": secondary_tables_html,
        "table_note": html.escape(build_table_note(evidence)),
        "metric_reading": html.escape(build_metric_reading(evidence, results)),
        "event_summary": html.escape(summarize_event_head_to_head(results)),
        "figures_html": build_figures_html(args.output_html, results.get("figures", [])),
        "plan_html": "".join(f"<li>{html.escape(item)}</li>" for item in plan_items),
    }
    html_text = render_html(PROJECT_ROOT / "skills/research-progress-html/assets/report_template.html", payload)
    return html_text, context, evidence


def main() -> None:
    args = parse_args()
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    html_text, _context, evidence = generate_html(args)
    args.output_html.write_text(html_text, encoding="utf-8")
    print(f"output_html={args.output_html}")
    print(f"best_model={evidence['best_model']}")
    print(f"evidence_status={evidence['evidence_status']}")


if __name__ == "__main__":
    main()
