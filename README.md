# Unified all-season mortality attribution

This repository contains the model implementation, prepared input data and native training and analysis workflows for the manuscript:

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
│   ├── death/       # Prepared Destatis daily and weekly mortality archive
│   ├── geodata/     # German national, state, and district boundaries
│   ├── population/  # District-, age-, and sex-specific population data
│   ├── sari/        # RKI weekly incidence and prepared daily surveillance
│   └── weather/     # HOSTRADA weather tensors
├── src/sari_t/      # Native model, training, data readers, and inference
├── configs/         # Native experiment configuration
├── notebooks/       # Manuscript analyses using native ensembles
├── runtime/
│   └── training/    # Experiment configurations, weights, and timing reports
└── train.py         # Train or resume one ensemble member
```

## Data represented in this snapshot

The analysis combines the following sources described in the manuscript:

- all-cause mortality from the Federal Statistical Office of Germany ([Destatis mortality reports](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Sterbefaelle-Lebenserwartung/Publikationen/Downloads-Sterbefaelle/statistischer-bericht-sterbefaelle-tage-wochen-monate-aktuell-5126109.html));
- district population counts from the Federal Statistical Office and the statistical offices of the federal states ([GENESIS table 12411-0018](https://www.regionalstatistik.de/genesis//online?operation=table&code=12411-09-01-4&bypass=true&levelindex=1&levelid=1783933967383#abreadcrumb));
- daily mean temperature and dew point from the German Weather Service ([DWD HOSTRADA](https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/)), with population weighting based on the [2022 German census](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Zensus2022/_inhalt.html#1403950); and
- total and COVID-19-associated SARI hospitalization incidence from the Robert Koch Institute ([SARI-Hospitalisierungsinzidenz, Zenodo DOI 10.5281/zenodo.17422294](https://doi.org/10.5281/zenodo.17422294)).

The manuscript analysis covers 13 October 2014 through 28 December 2025. The split is chronological: 75% of observations through 5 June 2022 are used for training, with validation through 29 December 2024 and 2025 examined separately. The native configuration calculates its split over complete weeks. Adaptive-baseline inference uses observed mortality over the evaluation window, including 2025. Population values after 2024 are held at their 2024 levels.

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

## Manuscript analyses

The notebooks follow the analyses in the [original notebook
snapshot](https://github.com/ClimSocAna/sari-weather-mortality/tree/4402a80d3e490ad7964a3be12597043761bebda1/notebooks).
They load native `final_000.pt` through `final_009.pt` from the matching run
directories under `runtime/training/experiments/`. Run them on a compute node
with access to the checkout and weights. The notebooks select CUDA when
available. Their figures and tables are recomputed from the native ensembles.
The **Load models for inspection** cell in each notebook loads the selected
member's predictors on CPU before running an analysis. Inspect the hybrid
predictor as `model` or access any setup as `models[setup]`. Change
`inspection_member` to inspect another checkpoint.

| Notebook | Analysis | Training setups |
| --- | --- | --- |
| [01](notebooks/01_mortality_component_attribution.ipynb) | Seasonal and age/sex-specific attribution | `default` |
| [02](notebooks/02_temperature_humidity_response.ipynb) | Temperature and humidity response | `default`, `no_disease`, `tas_only` |
| [03](notebooks/03_adaptive_baseline_mortality.ipynb) | Registered and adaptive baseline mortality | `default` |
| [04](notebooks/04_model_performance_sensitivity.ipynb) | Model performance and sensitivity | `default`, `exclude_temperature`, `temperature_only`, `no_covid`, `baseline_static`, `baseline_trimmed_mean`, `baseline_double_sided`, `tas_only`, `tas_wet_bulb`, `no_mean_alignment` |
| [05](notebooks/05_daily_residual_diagnostics.ipynb) | Daily residual, SARI and weather diagnostics | `default`, `tas_only` |
| [06](notebooks/06_temperature_lag_sensitivity.ipynb) | Daily and weekly training- and validation-period losses across 1–30 weather input days | Hybrid and air-temperature-only lag sweeps |

Notebooks 01–05 reconstruct mortality through 28 December 2025. Notebook 06
uses each checkpoint's original training and validation periods. Full notebook
execution requires the native ensembles and a compute node.

## Native training and inference

Use the repository environment and run commands from the repository root.
Write new training runs to a separate directory to preserve the published ensembles.

```bash
python train.py --device auto --member-id 0 --run-directory runtime/training/new-runs
```

Training reads the bundled mortality archive at `data/death/mortality_20260714.npz`
and daily surveillance archive at `data/sari/sari_20260709.npz`, along with the
population and weather files. The weekly RKI TSV supplies the surveillance
panel in notebook 01. See [third-party data and attribution](THIRD_PARTY_DATA.md)
for source details.

The default [experiment configuration](configs/training.example.yaml) uses
mean temperature and dew point, separate non-COVID and COVID-19 SARI inputs,
and a fixed 16 °C, 50% relative humidity reference. Run `python train.py --list-setups`
or `python train.py --list-references` to see the available overrides. For a
CPU smoke test, copy the YAML to `configs/smoke.yaml`, set `period.end_date`
to `2015-01-11`, and set `max_epochs: 1` and `train_steps_per_epoch: 1` under
`training_config`. Then run:

```bash
python train.py --config configs/smoke.yaml --setup baseline_static --device cpu
```

For a full ensemble on Slurm, place the checkout on shared storage,
activate the environment, and submit one task per member from the repository
root. Adjust the resource requests and add any partition or account options
required by your cluster.

```bash
setup=default
sbatch --array=0-9 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=24:00:00 \
  --wrap="python train.py --device cuda --setup ${setup} --member-id \$SLURM_ARRAY_TASK_ID --run-directory runtime/training/new-runs"
```

Run full-period ensemble inference on a compute node. The example below uses
CPU for portability. Use `device="cuda"` on a GPU node with sufficient memory.

Each run directory contains its resolved `config.yaml`, member metrics,
`checkpoints/last_NNN.pt` for resuming, and `checkpoints/final_NNN.pt` for
inference after completion. Saved run configurations contain resolved data
paths. Resume in the same data location with
`python train.py --resume PATH_TO_LAST_CHECKPOINT`. Train ensemble members
with distinct `--member-id` values and the same experiment configuration.

To load the completed members using the default experiment configuration:

```bash
run_dir="$(python train.py --print-run-directory)"
PYTHONPATH=src python - "$run_dir" <<'PYTHON'
import sys
from sari_t.inference import evaluate_runs

result = evaluate_runs(sys.argv[1], device="cpu")
print(result.predictions.total_deaths.shape)
PYTHON
```

For the CPU smoke test, pass `--config configs/smoke.yaml --setup baseline_static`
to the `--print-run-directory` command as well.

## Public weights and run times

The release imports 70 native ensembles from the current `weather_sari_model`
experiment collection. Each experiment under
`runtime/training/experiments/<experiment>/` includes all ten final checkpoints
and the run configuration. The links in
`runtime/training/sets/` group related experiments. Input paths are relative to
the release checkout and point to the matching files under `data/`.

The source implementation and trained weights come from
[`weather_sari_model`](https://github.com/lanmao0803/weather_sari_model/tree/9c8f5ae53fbae6e4e907134ae2fa810b770a2875).
The source checkout also contains the experiment selection reflected here.
Measured Slurm wall-clock times were unavailable in that collection.

After a Slurm array completes, export its elapsed times alongside the
checkpoints. Set `job_id` to the array job ID and `run_dir` to the experiment
directory. If members were resumed in a separate job, export its accounting
records as well so the total time for each member can be recovered.

```bash
sacct --jobs "$job_id" --allocations --parsable2 \
  --format=JobID,State,ElapsedRaw,AllocCPUS,ReqMem,AllocTRES \
  > "$run_dir/slurm-${job_id}-runtimes.psv"
```

`ElapsedRaw` is the wall-clock time in seconds for each allocation. Record the
GPU model and software environment with the report, where available. Review
the export and run metadata for cluster-specific identifiers or paths before
publishing.

For inference from a downloaded run, pass the matching public data
configuration to `evaluate_runs` to resolve the inputs in your checkout.
For a run that used the bundled inputs:

```python
from pathlib import Path

from sari_t.inference import evaluate_runs
from sari_t.run.checkpoint import load_run_config

run = Path("runtime/training/experiments/<experiment>")
public_data = load_run_config(run / "config.yaml").data_config
result = evaluate_runs(run, data_config=public_data)
```

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

