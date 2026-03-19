---
name: research-progress-html
description: Generate a Chinese research-progress HTML page from idea.md and current experiment results. Use when the user wants a paper-style progress page with an introduction prototype, current main-experiment progress with Table 1, and next-step experiment planning driven by the latest evidence.
---

# Research Progress HTML

Use this skill when the user asks for a research-progress HTML page, a paper-style experiment progress page, or an HTML summary built from the research goal and current experiment results.

## Inputs

Read inputs in this order:

1. `idea.md` for research goal, task framing, dataset context, and method narrative.
2. Structured results under `outputs/*/results_summary.csv` if they exist.
3. Existing HTML pages under `experiments/html/*.html` as fallback when structured results are missing.

If fields are still missing after these steps, keep the page renderable and use placeholders such as `TODO` or `Pending`.

## Workflow

1. Read `idea.md`.
2. Run `scripts/generate_progress_html.py`.
3. Default to `main` experiment rendering unless the user explicitly asks for ablation mode or provides ablation inputs.
4. After updating the HTML, sync the changed HTML and related skill files to GitHub. At minimum, stage the updated files, commit with a task-specific message, and push the current branch unless the user explicitly says not to.
5. Ensure the output HTML contains:
   - Hero section
   - `Introduction Prototype`
   - `Current Progress`
   - `Next Experiment Plan`

## Result-driven writing rules

- The introduction must combine research intent and current evidence.
- If `affect_state_forecaster` is the best model, state that the affect-state hypothesis has preliminary support.
- If a structure or temporal baseline beats `affect_state_forecaster`, state that the task is supported but the core method narrative still needs diagnosis.
- If results are incomplete or unstable, frame the page as validation of benchmark feasibility rather than method superiority.
- The introduction must mention at least two signals from:
  - best model
  - MAE gain versus text baseline
  - whether `affect_state_forecaster` beats structure or temporal baselines
  - evidence sufficiency or anomaly status

## Commands

Primary command:

```bash
python skills/research-progress-html/scripts/generate_progress_html.py
```

Useful flags:

```bash
python skills/research-progress-html/scripts/generate_progress_html.py \
  --idea-path idea.md \
  --experiments-root experiments \
  --output-html experiments/html/paper_progress.html \
  --mode auto \
  --result-source auto
```

Git sync after each HTML update:

```bash
git add skills/research-progress-html experiments/html/paper_progress.html
git commit -m "Update research progress HTML"
git push
```

## References

- Writing rules and evidence mapping: `references/writing_rules.md`
- HTML template: `assets/report_template.html`
