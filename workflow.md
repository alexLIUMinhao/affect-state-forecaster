# 服务器优先实验工作流（workflow）

## 1. 目标与原则
- 所有实验与代码改动仅在服务器执行。
- 服务器工作目录固定为 `/home/alexmhliu/affect-state-forecaster`。
- 本地仓库不做开发修改，只用于同步 GitHub 最新代码。
- 本流程默认分支为 `main`。

## 2. 登录与环境
1. 登录服务器：`ssh root@36.138.18.243`
2. 切换用户：`su - alexmhliu`
3. 进入项目：`cd /home/alexmhliu/affect-state-forecaster`
4. 激活环境：`conda activate asf311`

## 3. 每轮实验标准流程
1. 同步远端最新：
   - `git fetch origin`
   - `git pull --ff-only origin main`
2. 检查工作区状态：`git status -sb`
3. 运行本轮实验脚本（根据实验目标选择）。
4. 更新实验产物（按需要）：
   - `experiments/manifests`
   - `experiments/records`
   - `experiments/html`
   - `experiments/figures`
5. 复核结果与关键指标，确认无明显异常后再提交。

## 4. 提交策略
- 每轮实验结束后立即提交并推送（小步快跑，便于回溯）。
- 允许提交范围：
  - 代码与脚本：`src`、`scripts`、`tests`、必要文档
  - 论文相关实验产物：`experiments/{manifests,records,html,figures}`
- 不要把不相关改动混入同一次提交。

## 5. 禁止提交项
- `runs/`
- 临时日志与缓存目录
- 压缩包（如 `*.tar.gz`）
- 系统噪声文件（如 `._*`）
- 服务器备份目录（如 `server_backups/`）

## 6. 推送命令模板
- 本次（仅提交 workflow）：
  - `git add workflow.md`
  - `git commit -m "docs: add server-first experiment workflow"`
  - `git push origin main`
- 常规实验轮次（白名单方式 add）：
  - `git add src scripts tests experiments/manifests experiments/records experiments/html experiments/figures`
  - `git commit -m "<run_id>: <实验目的与结论>"`
  - `git push origin main`

## 7. 本地同步流程（只同步，不开发）
在本地仓库执行：
- `git fetch origin`
- `git pull --ff-only origin main`

## 8. 异常处理
- 若工作区有脏改动：先分类为“本轮有效改动”和“噪声/遗留改动”，只提交有效部分。
- 若 `push` 被拒绝：先 `git fetch`，处理与远端差异后再推送。
- 若改动跨度过大：拆分成多个小提交，保证每个提交语义单一、可回滚。
