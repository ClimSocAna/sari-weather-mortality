import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from . import config
from .config import HeatMortalityConfig

SARI_AGE_BUCKETS = ("15-34", "35-59", "60-79", "80+")
SARI_AGE_REPEATS = (2, 5, 4, 4)
COVID_SPECIFIC_AGE_BUCKETS = ("60-79", "80+")


def _extract_age_series(df: pd.DataFrame, age: str) -> np.ndarray:
    return df[df.Altersgruppe == age].iloc[::-1, -1].to_numpy()


def _interpolate_weekly_series(values: np.ndarray) -> np.ndarray:
    interpolator = PchipInterpolator(np.arange(3, values.shape[0] * 7, 7), values)
    return interpolator(np.arange(values.shape[0] * 7)).clip(0)


def sari_preprocessing(model_config: HeatMortalityConfig) -> tuple[torch.Tensor, torch.Tensor]:
    sari_path = config.BASE_DIR / "data" / "sari" / model_config.sari_filename
    df_sari = pd.read_csv(sari_path, sep="\t")
    df_covid = df_sari[df_sari.SARI == "COVID-19"]
    df_total = df_sari[df_sari.SARI == "Gesamt"]

    covid_00 = _extract_age_series(df_covid, "00+")
    total_00 = _extract_age_series(df_total, "00+")
    len_diff = total_00.shape[0] - covid_00.shape[0]
    covid_00 = np.concatenate([np.zeros(len_diff), covid_00])
    ratio = covid_00 / total_00
    arr_sari_no_covid: list[np.ndarray] = []
    arr_covid: list[np.ndarray] = []
    for age in SARI_AGE_BUCKETS:
        total = _extract_age_series(df_total, age)
        if age not in COVID_SPECIFIC_AGE_BUCKETS:
            sari = total * (1 - ratio)
            covid = total * ratio
        else:
            covid = np.concatenate([np.zeros(len_diff), _extract_age_series(df_covid, age)])
            sari = total - covid

        arr_sari_no_covid.append(_interpolate_weekly_series(sari))
        arr_covid.append(_interpolate_weekly_series(covid))

    n_covid_sari_data = np.concatenate([np.stack([arr_sari_no_covid[i]] * repeat) for i, repeat in enumerate(SARI_AGE_REPEATS)])
    covid_data = np.concatenate([np.stack([arr_covid[i]] * repeat) for i, repeat in enumerate(SARI_AGE_REPEATS)])
    return torch.from_numpy(n_covid_sari_data).float(), torch.from_numpy(covid_data).float()
