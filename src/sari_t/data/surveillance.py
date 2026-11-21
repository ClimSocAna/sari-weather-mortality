"""Load daily SARI and COVID from the pipeline's prepared NPZ format."""

import numpy as np
import torch

from .config import DataConfig
from .interfaces import SurveillanceData


def load_surveillance(config: DataConfig) -> SurveillanceData:
    """Load paired (age group, day) incidence arrays without model scaling.

    The pipeline already separates COVID, interpolates daily incidence and
    expands source ages to the 15 model groups. The configured start date must
    match this archive's first represented day. Source selection is explicit.
    """
    with np.load(config.data_root / config.surveillance_path) as archive:
        sari = torch.from_numpy(archive["sari"].astype(np.float32, copy=False))
        covid = torch.from_numpy(archive["covid"].astype(np.float32, copy=False))
    return SurveillanceData(
        sari=sari,
        covid=covid,
        start_date=config.surveillance_start_date,
    )
