# Desktop Agent on OSWorld

This repository contains an LLM-driven desktop agent that operates in the OSWorld environment.

Core components:

- agent runtime (`agent/`): perception, planning, execution, judging, memory
- OSWorld backend (`OSWorld/`, cloned separately)
- TCP bridge server (`OSWorld_server/`) between agent and environment
- automation scripts (`scripts/`)
- notebooks for debugging and analysis (`notebooks/`)

## How it works

1. The agent receives a task.
2. It observes screenshot, accessibility tree, and terminal state.
3. Perception fuses vision + accessibility into structured UI elements.
4. Planner generates the next action chunk.
5. Executor runs actions in OSWorld.
6. Judge evaluates outcome and updates memory.
7. The loop continues until completion or max cycles.

## Prerequisites

- Windows + PowerShell
- Conda/Miniforge available in shell
- VMware/OSWorld VM configured
- Azure OpenAI key in environment:
  - `OPENAI_AZURE_API_KEY`

## Clone OSWorld first

`OSWorld` is an external dependency and is not committed in this repository.
Clone it in the project root so you get `./OSWorld`:

```powershell
git clone https://github.com/xlang-ai/OSWorld.git OSWorld
```

## Environment setup

### Option A (recommended): full setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 -AgentEnvName agent-env -OSWorldEnvName osworld-env
```

This will:

- create/update the agent environment
- create the OSWorld environment
- start OSWorld server in a separate PowerShell window

### Option B: step-by-step

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_agent_env.ps1 -EnvName agent-env
powershell -ExecutionPolicy Bypass -File scripts/setup_osworld_env.ps1 -EnvName osworld-env
powershell -ExecutionPolicy Bypass -File scripts/run_osworld_server.ps1 -EnvName osworld-env
```

## OSWorld setup reference

Use OSWorld docs as source of truth:

- `OSWorld/README.md` (Quick Start section)
- `OSWorld/SETUP_GUIDELINE.md`

Quick validation:

```powershell
conda run -n osworld-env python OSWorld/quickstart.py
```

## Agent configuration

Main config files:

- `agent/config_files/start_agent.yaml` (CLI / batch)
- `agent/config_files/start_agent_nb.yaml` (notebooks)

Fields to verify before running:

- `osworld_settings.path_to_vm`
- `osworld_settings.snapshot_name`
- `osworld_settings.client_password`
- `memory_settings.memory_name`
- `memory_settings.initialize_memory`
- `perception_settings.use_vision`
- `gpt_client_settings.model`

## Batch run

Single batch execution:

```powershell
conda run --no-capture-output -n agent-env python scripts/run_agent_batch.py `
  --conf agent/config_files/start_agent.yaml `
  --tasks data/tasks.json `
  --metrics-csv data/experiments.csv `
  --max-cycles 10
```

More options and examples:

- `scripts/run_agent_batch.md`

## Notebooks

- `notebooks/agent_env.ipynb`: end-to-end runs and debugging
- `notebooks/osworld_env.ipynb`: OSWorld/server checks
- `notebooks/memory_env.ipynb`: memory inspection
- `notebooks/gpt_usage_analysis.ipynb`: model usage analysis
- `notebooks/experiments_results_analysis.ipynb`: experiment analysis

## Outputs

- Agent logs: `data/logs/agent/`
- LLM interaction logs: `data/gpt_interactions/`
- Batch summaries: `data/experiment_runs/`
- Experiment CSVs: `data/` and experiment folders
