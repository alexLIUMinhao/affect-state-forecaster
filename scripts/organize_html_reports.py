from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.experiment_reporting import (
    ensure_experiment_paths,
    html_output_path,
    rebuild_html_indexes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize experiment HTML reports into category folders.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def move_manifest_backed_reports(paths_root: Path) -> None:
    paths = ensure_experiment_paths(paths_root)
    index_path = paths.records / "experiment_index.csv"
    rows = read_csv_rows(index_path)
    if not rows:
        rebuild_html_indexes(paths, rows)
        return

    fieldnames = list(rows[0].keys())
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        desired = html_output_path(paths.html, run_id)
        desired.parent.mkdir(parents=True, exist_ok=True)

        candidates = []
        html_path = row.get("html_path", "")
        if html_path:
            candidates.append(Path(html_path))
        candidates.append(paths.html / f"{run_id}.html")
        candidates.append(desired)

        current = next((path for path in candidates if path.exists()), None)
        if current and current.resolve() != desired.resolve():
            shutil.move(str(current), str(desired))

        row["html_path"] = str(desired)
        manifest_path = paths.manifests / f"{run_id}.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["html_path"] = str(desired)
            manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv_rows(index_path, rows, fieldnames)
    rebuild_html_indexes(paths, rows)


def move_progress_pages(paths_root: Path) -> None:
    html_root = paths_root / "html"
    progress_dir = html_root / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for path in html_root.glob("*paper_progress.html"):
        shutil.move(str(path), str(progress_dir / path.name))
    if (html_root / "paper_progress.html").exists():
        shutil.move(str(html_root / "paper_progress.html"), str(progress_dir / "paper_progress.html"))
    pages = sorted(path for path in progress_dir.glob("*.html") if path.name != "index.html")
    cards = "".join(
        f"<article class='card'><h2>{path.name}</h2><p>论文进展总览页。适合先看当前最优模型、证据状态和下一步计划。</p><p><a href='{path.name}'>打开页面</a></p></article>"
        for path in pages
    ) or "<article class='card'><p>暂无 progress 页面。</p></article>"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>论文进展 HTML 总览</title>
  <style>
    body {{ margin: 0; font-family: Georgia, "Noto Serif SC", serif; background: #f4f1e8; color: #1e293b; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{ background: linear-gradient(135deg, #f8e7cf, #f7f5ef); border: 1px solid #d8d1c2; border-radius: 22px; padding: 28px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ background: #fffdf8; border: 1px solid #d8d1c2; border-radius: 18px; padding: 18px; box-shadow: 0 10px 24px rgba(0,0,0,0.04); }}
    a {{ color: #8c3b2f; text-decoration: none; font-weight: 700; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <p><a href="../index.html">返回总目录</a></p>
      <h1>论文进展</h1>
      <p>这类 HTML 是高层总结页，不是单轮 run 报告。适合先看当前阶段最优模型、证据强弱和下一步实验计划。</p>
    </section>
    <section class="grid">{cards}</section>
  </div>
</body>
</html>
"""
    (progress_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    move_manifest_backed_reports(args.experiments_root)
    move_progress_pages(args.experiments_root)
    rebuild_html_indexes(ensure_experiment_paths(args.experiments_root))


if __name__ == "__main__":
    main()
