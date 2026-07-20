import math
from pathlib import Path
import re
from typing import Literal

import geopandas as gpd
import geojson
from shapely.geometry import Polygon, MultiPolygon
import torch

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / 'data'
GEO_DIR = BASE_DIR / 'data' / 'geodata'
STATE_DICT_DIR = BASE_DIR / 'statedict'

ExecutionDevice = Literal["auto", "cpu", "cuda"]


def resolve_device(execution_device: ExecutionDevice = "auto") -> torch.device:
    """Resolve an explicit or automatically detected notebook execution device."""
    if execution_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if execution_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if execution_device not in {"cpu", "cuda"}:
        raise ValueError("execution_device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(execution_device)


def _checkpoint_version(path: Path) -> int:
    match = re.match(r"v(\d+)-", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(match.group(1))


def get_checkpoint_paths(experiment_name: str, device: torch.device) -> list[Path]:
    """Return v0 for CPU execution or the complete ordered ensemble for CUDA."""
    experiment_dir = STATE_DICT_DIR / experiment_name
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    checkpoint_paths = sorted(experiment_dir.glob("v*-epoch=*.ckpt"), key=_checkpoint_version)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files found in: {experiment_dir}")

    if device.type == "cpu":
        checkpoint_paths = [path for path in checkpoint_paths if _checkpoint_version(path) == 0]
        if len(checkpoint_paths) != 1:
            raise RuntimeError(f"Expected exactly one v0 checkpoint in {experiment_dir}, found {len(checkpoint_paths)}")

    return checkpoint_paths


def load_experiment(experiment_name: str, device: torch.device):
    """Construct an evaluation model and select checkpoints for the execution device."""
    from model.base_model import HeatMortalityBase

    experiment_dir = STATE_DICT_DIR / experiment_name
    hparams_path = experiment_dir / "hparams.yaml"
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Hyperparameter file not found: {hparams_path}")

    model = HeatMortalityBase.from_lightning_hparams_yaml(hparams_path).to(device)
    model.eval()
    model.on_train_start()
    model._notebook_initial_basic_death_rate_range = model.basic_death_rate_range.detach().clone()
    checkpoint_paths = get_checkpoint_paths(experiment_name, device)
    print(
        f"{experiment_name}: device={device.type}, "
        f"checkpoints={len(checkpoint_paths)} ({', '.join(path.name.split('-', 1)[0] for path in checkpoint_paths)})"
    )
    return model, checkpoint_paths


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    """Load one trusted, repository-bundled Lightning checkpoint for evaluation."""
    initial_baseline = getattr(model, "_notebook_initial_basic_death_rate_range", None)
    if initial_baseline is None:
        raise RuntimeError("Model must be constructed with load_experiment before loading checkpoints")
    model.basic_death_rate_range.copy_(initial_baseline)
    model.training_mean_correction_factor = None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()


def dew_point_from_relative_humidity(temperature_c: float, relative_humidity: float) -> float:
    """Return dew point in Celsius for relative humidity expressed from 0 to 1."""
    if not 0 < relative_humidity <= 1:
        raise ValueError("relative_humidity must be in the interval (0, 1]")
    magnus_a = 17.625
    magnus_b = 243.04
    gamma = math.log(relative_humidity) + magnus_a * temperature_c / (magnus_b + temperature_c)
    return magnus_b * gamma / (magnus_a - gamma)


def _load_geojson_gdf(file_name: str, geometry_column: str, properties: list[str] | None = None) -> gpd.GeoDataFrame:
    with open(GEO_DIR / file_name) as file_handle:
        gj = geojson.load(file_handle)

    geometries: list[MultiPolygon] = []
    data = {key: [] for key in (properties or [])}
    for feature in gj["features"]:
        polygons: list[Polygon] = []
        process_coordinates(feature["geometry"]["coordinates"], polygons)
        geometries.append(MultiPolygon(polygons))
        for key in data:
            data[key].append(feature["properties"][key])

    data[geometry_column] = geometries
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def get_gdf_kreis() -> gpd.GeoDataFrame:
    gdf = _load_geojson_gdf("georef-germany-kreis.geojson", "geometry", ["krs_code"])
    gdf.iloc[118, gdf.columns.get_loc("geometry")] = gdf.iloc[118].geometry.union(gdf.iloc[82].geometry)
    gdf = gdf.drop(index=82).reset_index(drop=True)
    gdf = gdf.sort_values(by="krs_code").reset_index(drop=True)
    return gdf


def get_gdf_de() -> gpd.GeoDataFrame:
    return _load_geojson_gdf("georef-germany.geojson", "geometry")


def get_gdf_land() -> gpd.GeoDataFrame:
    return _load_geojson_gdf("georef-germany-land.geojson", "geometry").reset_index(drop=True)


def process_coordinates(coord: list, polygons: list[Polygon]) -> None:
    if isinstance(coord[0][0][0], float):
        p = Polygon(coord[0])
        for j in range(1, len(coord)):
            p = p.difference(Polygon(coord[j]))
        polygons.append(p)
    else:
        for c in coord:
            process_coordinates(c, polygons)
