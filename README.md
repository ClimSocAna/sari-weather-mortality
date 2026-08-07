# Unified all-season mortality attribution

This repository contains the model implementation, prepared input data, trained model ensembles, and analysis notebooks used for the manuscript:

> Junyu Wang et al. **“Integrating respiratory infection surveillance and temperature data improves all-season mortality reconstruction.”** Preprint manuscript, 07 Aug 2026.

The study reconstructs daily all-cause mortality in Germany with a component-based neural network that combines weather, severe acute respiratory infection (SARI) surveillance, demographic structure, and an adaptive mortality baseline. The primary validation target is reconstruction of observed all-cause mortality. Temperature-, non-COVID-SARI-, and COVID-19-associated mortality are model-based decomposition components, not directly observed cause-of-death counts.

## Model overview

For each day, district, age group, and sex, the model combines:

- population and an age- and sex-specific background mortality rate;
- population-weighted daily mean temperature and dew-point temperature;
- age-specific non-COVID SARI hospitalization incidence;
- age-specific COVID-19-associated SARI hospitalization incidence; and
- weekday and federal-state calibration terms.

In simplified form, predicted mortality is

```text
predicted deaths = population × corrected baseline
                 × (1 + temperature component
                      + non-COVID SARI component
                      + COVID-19 SARI component)
```

The weather branch uses a six-day temporal kernel. The two infection branches use non-negative ten-day convolution kernels. During training, the adaptive baseline is updated from smoothed residual mortality rather than optimized by gradient descent. See the manuscript for the complete model definition, counterfactual attribution procedure, validation design, and interpretation.

## Repository layout

```text
.
├── data/
│   ├── death/       # Destatis daily and weekly all-cause mortality arrays
│   ├── geodata/     # German national, state, and district boundaries
│   ├── population/  # District-, age-, and sex-specific population data
│   ├── sari/        # RKI SARI hospitalization incidence
│   └── weather/     # HOSTRADA weather tensors
├── model/           # Model architecture, preprocessing, loss, and metrics
├── notebooks/       # Ordered manuscript analysis and figure notebooks
├── statedict/       # Trained checkpoints and Lightning hyperparameters
├── training.py      # Command-line entry point for one training replicate
└── training.sh      # Portable shell wrapper for training.py
```

The main hybrid ensemble is stored in:

```text
statedict/HeatMortality_EXP_t_6_sari_10_covid_median_ewma_corr_on_pop_average_tas-tdew_stage1/
```

Its name records the main configuration: six-day weather kernel, ten-day SARI kernels, separate COVID-19 input, median-filter/EWMA adaptive baseline, calibration enabled, population-weighted mean temperature (`tas`) and dew point (`tdew`), and stage-1 training.

## Data represented in this snapshot

The analysis combines the following sources described in the manuscript:

- all-cause mortality from the Federal Statistical Office of Germany ([Destatis mortality reports](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Sterbefaelle-Lebenserwartung/Publikationen/Downloads-Sterbefaelle/statistischer-bericht-sterbefaelle-tage-wochen-monate-aktuell-5126109.html));
- district population counts from the Federal Statistical Office and the statistical offices of the federal states ([GENESIS table 12411-0018](https://www.regionalstatistik.de/genesis//online?operation=table&code=12411-09-01-4&bypass=true&levelindex=1&levelid=1783933967383#abreadcrumb));
- daily mean temperature and dew point from the German Weather Service ([DWD HOSTRADA](https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/)), with population weighting based on the [2022 German census](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Zensus2022/_inhalt.html#1403950); and
- total and COVID-19-associated SARI hospitalization incidence from the Robert Koch Institute ([SARI-Hospitalisierungsinzidenz, Zenodo DOI 10.5281/zenodo.17422294](https://doi.org/10.5281/zenodo.17422294)).

The manuscript analysis covers 13 October 2014 through 28 December 2025. The split is chronological: 75% of observations through 5 June 2022 are used for training, 6 June 2022 through 29 December 2024 for validation, and 2025 as an additional test period. Population values after 2024 are held at their 2024 levels.

The checked-in data are analysis-ready derivatives. Consult the manuscript and the original providers for definitions, provenance, access conditions, and any reuse restrictions.

## License

Source code is available under the [MIT License](LICENSE). Original research
content, notebook narrative and outputs, figures, and trained model checkpoints
are available under the [Creative Commons Attribution 4.0 International
License](https://creativecommons.org/licenses/by/4.0/). The files under
`data/` retain their providers' terms and are not relicensed by this
repository; see [third-party data and attribution](THIRD_PARTY_DATA.md) for
details.

## Environment

The repository includes a cross-platform Conda environment with the versions used to validate the notebooks. From the repository root, create the environment and register its Jupyter kernel:

```bash
conda env create --file environment.yml
conda activate sari-weather-mortality
python -m ipykernel install --user \
  --name sari-weather-mortality \
  --display-name "Python (sari-weather-mortality)"
```

Then start Jupyter from the repository root so that the local `model` and `notebooks` packages are importable:

```bash
jupyter lab
```

The environment is portable and does not force a CUDA runtime. On an NVIDIA system, install a CUDA-enabled PyTorch build compatible with the local driver if the resolved PyTorch package does not provide CUDA support.

Each model-driven notebook exposes `EXECUTION_DEVICE = "auto"` in its first code cell. Auto mode uses CUDA when PyTorch reports it as available and CPU otherwise. CPU mode evaluates only checkpoint `v0` for a faster exploratory run; CUDA mode evaluates all ten checkpoints used for manuscript ensemble summaries. Set the parameter explicitly to `"cpu"` or `"cuda"` to override automatic selection.

## Manuscript analyses

The notebooks are exploratory manuscript workflows rather than a single automated pipeline:

| Notebook | Manuscript output or analysis |
| --- | --- |
| `01_mortality_component_attribution.ipynb` | Manuscript Figures 1 and 4: seasonal component attribution and age-/sex-specific attributed mortality |
| `02_temperature_humidity_response.ipynb` | Manuscript Figure 2: temperature-mortality response and humidity modification |
| `03_adaptive_baseline_mortality.ipynb` | Manuscript Figure 3 and Supplementary Figure 9: registered deaths and adaptive baseline trajectories |
| `04_model_performance_sensitivity.ipynb` | Manuscript Figure 5 source metrics and Supplementary Figures 3-5: input, baseline, and humidity sensitivity |
| `05_daily_residual_diagnostics.ipynb` | Supplementary diagnostic comparisons of daily residuals, weather inputs, and SARI activity |

Each experiment directory contains one `hparams.yaml` file recording the complete model configuration and matching SARI filename, together with ten independently initialized checkpoints. The shared notebook loader resolves these repository-relative paths automatically. For example:

```python
from notebooks import load_experiment, resolve_device

device = resolve_device("auto")
model, checkpoint_paths = load_experiment(
    "HeatMortality_EXP_t_6_sari_10_covid_median_ewma_corr_on_"
    "pop_average_tas-tdew_stage1",
    device,
)
```

Each model configuration generally contains ten independently initialized replicates (`v0`–`v9`). Manuscript results are based on ensemble summaries rather than a single checkpoint.

## Model code

- `model/config.py` defines data windows, input combinations, kernel sizes, baseline choices, and training constants.
- `model/base_model.py` implements data alignment, training and validation logic, calibration, adaptive-baseline updates, and prediction aggregation.
- `model/exp_model.py` defines the exponential weather branch and non-negative SARI/COVID branches used in the manuscript.
- `model/baseline.py` implements the static and adaptive baseline estimators.
- `model/loss_function.py` and `model/metric.py` implement the fitting objective and reconstruction metrics.
- `model/weather_data.py`, `model/sari_data.py`, and `model/population_data.py` load and align the model inputs.

## Training

The supplied checkpoints are sufficient to reproduce the notebook analyses. `training.py` and `training.sh` are provided only as illustrative examples of the training workflow; successful execution and exact reproduction of the published checkpoints are not guaranteed across systems. Users may need to adapt the scripts to their hardware, CUDA installation, scheduler, and software environment.

To attempt training a new replicate, activate the repository environment and run:

```bash
./training.sh --accelerator gpu --version 0
```

Full model training should be performed on a CUDA-capable GPU. CPU execution is substantially slower and is intended only for short smoke tests, such as `./training.sh --accelerator cpu --max-epochs 1`. The launcher performs one run and forwards all arguments to `training.py`; use `./training.sh --help` to list the available model, input, device, and output options. Although Lightning can select an accelerator automatically, specifying `--accelerator gpu` is recommended for production training so that a missing CUDA setup does not silently start a long CPU run.

Training artifacts are written under `training_outputs/`, separated into checkpoint and CSV-log directories. Each replicate is deterministically seeded from `--version` unless `--seed` is supplied. Stage-2 training additionally requires `--stage 2 --checkpoint-path PATH`.

## Interpretation and limitations

The repository should be used with the interpretation given in the manuscript:

- Better reconstruction indicates that weather and infection indicators explain complementary variation in all-cause mortality.
- Component estimates are conditional on the fitted model, baseline definition, lag structure, smoothing, surveillance inputs, and counterfactual temperature definition.
- The components are not interchangeable with primary-cause death-certificate categories.
- National SARI incidence is assigned to all districts because finer-scale SARI surveillance is unavailable.
- The COVID-19 SARI-to-mortality relationship is modeled as time invariant, although severity changed with variants, immunity, vaccination, and treatment.
- Ensemble ranges and sensitivity analyses describe model stability but do not constitute conventional likelihood-based confidence intervals.

## Citation

Until a journal citation or DOI is available, cite the accompanying manuscript as:

```text
Wang, J., an der Heiden, M., Irrgang, C. (2026).
Integrating respiratory infection surveillance and temperature data improves all-season mortality reconstruction. Preprint manuscript, 07 Aug 2026.
```

