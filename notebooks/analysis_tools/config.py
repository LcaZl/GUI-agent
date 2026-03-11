from pathlib import Path


EXP_ORDER = ["exp01", "exp02", "exp03", "exp04"]

PLOT_CONDITION_ORDER = [
    "Baseline | Vision ON",
    "Baseline | Vision OFF",
    "No STM | Vision ON",
    "No STM | Vision OFF",
    "Warm LTM | Vision ON",
    "Warm LTM | Vision OFF",
]

DEFAULT_RUN_SPECS = [
    {
        "run_id": "baseline_run1",
        "run_label": "Baseline and warm LTM",
        "csv_path": Path("../thesis_experiment/experiments_final_v1_20260227_160645.csv"),
    },
    {
        "run_id": "baseline_run2",
        "run_label": "Baseline and warm LTM",
        "csv_path": Path("../thesis_experiment/experiments_final_v1_20260228_104054.csv"),
    },
    {
        "run_id": "baseline_run3",
        "run_label": "Baseline and warm LTM",
        "csv_path": Path("../thesis_experiment/experiments_final_v1_20260228_164937.csv"),
    },
    {
        "run_id": "no_stm_run1",
        "run_label": "No STM",
        "csv_path": Path("../thesis_experiment/experiments_final_v2_stm_toggle_20260307_050000.csv"),
    },
    {
        "run_id": "no_stm_run2",
        "run_label": "No STM",
        "csv_path": Path("../thesis_experiment/experiments_final_v2_stm_toggle_20260307_101100.csv"),
    },
    {
        "run_id": "no_stm_run3",
        "run_label": "No STM",
        "csv_path": Path("../thesis_experiment/experiments_final_v2_stm_toggle_20260307_121428.csv"),
    },
]
