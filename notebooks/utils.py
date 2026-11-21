"""Load complete native ensembles and retain only reduced notebook outputs."""

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sari_t.models.config import PredictionPeriod
from sari_t.run.checkpoint import load_run_config


CONFIG = ROOT / "configs" / "training.example.yaml"
RUN_ROOT = ROOT / "runtime" / "training" / "experiments"
ANALYSIS_PERIOD = PredictionPeriod(date(2014, 10, 13), date(2025, 12, 28))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGE_LABELS = ["0–29", *(f"{age}–{age + 4}" for age in range(30, 95, 5)), "95+"]


def checkpoint_paths(setup):
    """Return the ten released weights for a named experiment."""
    config = load_run_config(CONFIG, setup=setup)
    run = RUN_ROOT / config.run_name
    paths = [run / "checkpoints" / f"final_{member:03d}.pt" for member in range(10)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing native weights in {run}: {', '.join(missing)}")

    return paths


def ensemble(model, paths, summarize, *, scenarios=None):
    """Evaluate weights in the supplied workspace and average compact summaries."""
    totals = {}
    for index, path in enumerate(paths):
        model.load_weights(path)
        result = model.evaluate(scenarios=scenarios)
        for name, value in summarize(model, result).items():
            value = value.detach().cpu()
            if index == 0:
                totals[name] = value.clone()
            else:
                totals[name].add_(value)
        del result
        model.clear_member()
    return {name: value / len(paths) for name, value in totals.items()}


def daily_index(period=ANALYSIS_PERIOD):
    return pd.date_range(period.start_date, period.end_date, freq="D", name="Date")


def weekly_sum(values):
    return values.reshape(-1, 7, *values.shape[1:]).sum(dim=1)


def national_daily(model, result):
    return {
        "Observed": model.labels.daily_land.sum(dim=1),
        "Predicted": result.predictions.total_deaths.sum(dim=(1, 2, 3)),
    }
