# run_agent_batch.py

Run `AgentZ` on a list of tasks (single execution), with per-task episode metrics saved to CSV.

## Quick Start

From the repository root:

```bash
python scripts/run_agent_batch.py
```

Typical run in your Conda env:

```bash
conda run --no-capture-output -n thesis-env python scripts/run_agent_batch.py \
  --conf agent/config_files/start_agent.yaml \
  --tasks data/tasks.json \
  --metrics-csv data/experiments.csv \
  --max-cycles 10
```

## What It Does

1. Loads an `ExperimentConfiguration` from `--conf`.
2. Loads tasks from `--tasks` (JSON/YAML list or single task object).
3. Runs tasks sequentially with `agent.run_task_bdi(...)`.
4. Appends one episode row per task to `--metrics-csv`.
5. Writes a batch summary CSV to `data/experiment_runs/`.

## Experiment Tracking in CSV

Each episode row in `--metrics-csv` includes:

- task + execution metrics
- `experiment_id`, `experiment_name`
- config metadata: `experiment_config_path`, `experiment_config_hash`, `experiment_config_json`
- agent metadata: `agent_name`, `agent_settings_hash`, `agent_settings_json`

This allows grouping multiple episodes under the same experiment and comparing different configs.

## CLI Parameters

| Argument | Default | Description |
|---|---|---|
| `--conf` | `agent/config_files/start_agent.yaml` | YAML config file for `ExperimentConfiguration`. |
| `--tasks` | `data/tasks.json` | Task file (`.json/.yaml/.yml`). |
| `--metrics-csv` | `data/experiments.csv` | Output CSV for episode metrics. |
| `--agent-id` | `batch-agent` | Base agent identifier. |
| `--experiment-id` | auto (`exp_YYYYMMDD_HHMMSS`) | Stable experiment ID for the whole batch. |
| `--experiment-name` | config filename stem | Human-readable experiment label. |
| `--max-cycles` | `0` | Max BDI cycles per task (`0` = use agent default). |
| `--start-index` | `0` | Start task index in loaded task list. |
| `--limit` | `0` | Max number of tasks to run (`0` = no limit). |
| `--task-ids` | `None` | Optional list of exact task IDs to execute. |
| `--stop-on-error` | `False` | Stop batch at first task exception. |
| `--no-reuse-agent` | `False` | Create a fresh agent for each task. |
| `--verbose-ui` | `False` | Enable UI visualization during execution. |
| `--skip-existing-task-ids` | `False` | Skip task IDs already present in metrics CSV for the same experiment context. |
| `--dry-run` | `False` | Validate config/tasks and exit without running tasks. |
| `--log-dir` | `data/logs/agent` | Directory for batch log file. |
| `--log-level` | `INFO` | Console logging level. |

## Useful Examples

Run only 3 tasks starting from index 2:

```bash
python scripts/run_agent_batch.py --start-index 2 --limit 3
```

Run specific task IDs:

```bash
python scripts/run_agent_batch.py --task-ids e0df059f-28a6-4169-924f-b9623e7184cc 94d95f96-9699-4208-98ba-3c3119edf9c2
```

Create a named experiment:

```bash
python scripts/run_agent_batch.py --experiment-name baseline-v2 --experiment-id exp_baseline_v2_01
```

Validate setup only:

```bash
python scripts/run_agent_batch.py --dry-run
```

Resume an interrupted experiment without re-running completed task IDs:

```bash
python scripts/run_agent_batch.py --skip-existing-task-ids
```
