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
    }
    if "future_neg_ratio" in task:
        context["task"] = "以 future_neg_ratio 为主任务，预测未来窗口中的负面情绪比例"
    if "Affect-State Forecaster" in method:
        context["method"] = "Affect-State Forecaster"
    if "事件线程" in background:
        context["goal"] = "在真实公共事件线程的早期观测阶段预测后续群体情绪走势"
    return context


def collect_structured_results(project_root: Path) -> dict[str, Any]:
    outputs_root = project_root / "outputs"
    models: dict[str, dict[str, Any]] = {}
    figure_candidates: list[str] = []

    for model in DEFAULT_OUTPUT_MODELS:
        summary_path = outputs_root / model / "results_summary.csv"
        rows = read_csv_rows(summary_path)
        overall = next((row for row in rows if row.get("group_name") == "overall"), None)
        if not overall:
            continue
        models[model] = {
            "model": model,
            "dataset": "PHEME",
            "metrics": {
                "mae": to_float(overall.get("mae")),
                "rmse": to_float(overall.get("rmse")),
                "pearson": to_float(overall.get("pearson")),
                "spearman": to_float(overall.get("spearman")),
            },
        }

    figures_root = project_root / "experiments" / "figures"
    if figures_root.exists():
        for image in sorted(figures_root.glob("*.png"), key=slug_sort_key, reverse=True):
            figure_candidates.append(str(image))
            if len(figure_candidates) == 3:
                break

    return {"models": models, "figures": figure_candidates}


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

    candidates = [path for path in html_dir.glob("*.html") if path.name not in {"index.html", "paper_progress.html"}]
    if not candidates:
        return {"models": models, "figures": figures}
    latest = sorted(candidates, key=slug_sort_key, reverse=True)[0]
    page = read_text(latest)

    row_matches = re.findall(r"<tr><td>([^<]+)</td><td>([^<]+)</td><td>([^<]*)</td><td>([^<]*)</td><td>([^<]*)</td><td>([^<]*)</td></tr>", page)
    for model_name, _status, mae, rmse, pearson, spearman in row_matches:
        models[model_name] = {
            "model": model_name,
            "dataset": "PHEME",
            "metrics": {
                "mae": to_float(mae),
                "rmse": to_float(rmse),
                "pearson": to_float(pearson),
                "spearman": to_float(spearman),
            },
        }

    for src in re.findall(r"<img src='([^']+)'", page):
        figures.append(str((html_dir / src).resolve()))
    return {"models": models, "figures": figures}


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
    for name, payload in models.items():
        mae = payload.get("metrics", {}).get("mae")
        if mae is None:
            continue
        if best_mae is None or mae < best_mae:
            best_model = name
            best_mae = mae

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

    return {
        "best_model": best_model,
        "best_mae": best_mae,
        "mae_gain_vs_text": mae_gain_vs_text,
        "ours_model": "affect_state_forecaster",
        "ours_vs_structure": ours_vs_structure,
        "ours_vs_temporal": ours_vs_temporal,
        "evidence_status": evidence_status,
        "anomalies": anomalies,
    }


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

    if status == "ours_supported":
        paragraph_two = (
            f"从当前关键结果看，{best_model} 已经取得本轮最优 MAE={format_metric(evidence['best_mae'])}，"
            f"相对文本基线的 MAE 改进为 {format_metric(mae_gain)}。这说明 {method} 所依赖的显式 affect-state 建模"
            f"获得了初步支持；同时，方法相对结构基线和时序基线的比较分别为 {format_metric(ours_vs_structure)} 与 "
            f"{format_metric(ours_vs_temporal)}，因此现阶段可以把论文叙事重心放在“情绪状态作为中间驱动变量”的有效性上。"
        )
    elif status in {"diagnostic_needed", "task_supported"}:
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
        f"当前最优模型：<span class='mono'>{html.escape(evidence['best_model'])}</span>，MAE={format_metric(evidence['best_mae'])}。",
        f"相对文本基线的 MAE 改进：{format_metric(evidence['mae_gain_vs_text'])}。",
        f"`affect_state_forecaster` 相对结构基线的 MAE 差：{format_metric(evidence['ours_vs_structure'])}。",
        f"`affect_state_forecaster` 相对时序基线的 MAE 差：{format_metric(evidence['ours_vs_temporal'])}。",
        f"当前证据状态：{html.escape(evidence['evidence_status'])}。",
    ]
    if evidence["anomalies"]:
        items.append(f"主要风险：{html.escape(evidence['anomalies'][0])}")
    return items


def table_row(dataset: str, baseline: str, ours: str, metrics: dict[str, float | None]) -> str:
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
        f"<td class='mono'>{html.escape(metric_text)}</td>"
        "</tr>"
    )


def build_table_html(context: dict[str, str], results: dict[str, Any]) -> str:
    models = results.get("models", {})
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
            )
        )

    ours_payload = models.get("affect_state_forecaster")
    rows.append(
        table_row(
            dataset,
            "—",
            "affect_state_forecaster",
            ours_payload.get("metrics", {}) if ours_payload else {},
        )
    )

    return (
        "<table>"
        "<thead><tr><th>Dataset</th><th>Compared Baselines</th><th>Ours</th><th>Metrics</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def build_progress_summary(evidence: dict[str, Any]) -> str:
    if evidence["evidence_status"] == "ours_supported":
        return (
            "当前主实验已经形成可用于论文原型叙事的正向证据：ours 暂时领先，且相对文本基线存在明确改进。"
        )
    if evidence["evidence_status"] in {"diagnostic_needed", "task_supported"}:
        return (
            "当前主实验首先证明了任务成立，但优势主要来自结构或时序线索，ours 仍需要通过诊断实验进一步澄清。"
        )
    return "当前主实验结果仍不完整，因此本页以可行性说明和占位式结果汇总为主。"


def build_table_note(evidence: dict[str, Any]) -> str:
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


def build_next_plan(evidence: dict[str, Any]) -> list[str]:
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

    introduction = build_introduction(context, evidence)
    signals = build_signal_items(evidence)
    plan_items = build_next_plan(evidence)

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
        "progress_summary": html.escape(build_progress_summary(evidence)),
        "table_html": build_table_html(context, results),
        "table_note": html.escape(build_table_note(evidence)),
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
