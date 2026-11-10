# Silicon Subjects

In this experiment, we’re testing whether AI can replicate human behaviour in studies. We give a model the same kind of choice a participant faced and ask it to pick an option. Then we compare its choices with what people actually did.

Take a study where a disease is expected to kill 600 people. One group hears that a treatment will save 200 people for sure. Another hears that 400 people will die for sure. The outcome is the same, but the wording changes how often people take the safe option over a gamble. Does changing a few words shift the AI’s choices by about as much?

In a separate part of the experiment, we ask the model how it thinks people answered. It might guess that most people took the gamble and still pick the safe option when it has to choose for itself.

We use the framing and obedience studies to develop the experiment. For the main test, we use a moral dilemma and a survey question about allowing or forbidding public speeches.

The code and an offline demo are here. We haven’t reported results from the main experiment yet.

## Setup

Use Python 3.11 or later. In PowerShell, enable UTF-8 for each session that reads the study materials. To set up the project, run the following commands.

```powershell
$env:PYTHONUTF8 = "1"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, `make install` creates the environment and installs the same hash-pinned dependencies. Activate it with `source .venv/bin/activate`.

## Offline demo

```text
sim freeze -c configs/demo.yaml
sim run -c configs/demo.yaml --run-id demo
sim analyze demo -c configs/demo.yaml
```

Open `results/demo/report.md` for the report and `results/demo/analysis/` for the tables. The demo uses mock models and made-up data so you can try it offline. `make demo` runs the same pipeline in a POSIX shell.

An unchanged run ID resumes unfinished trials. After changing the source or inputs, use `sim freeze -c configs/demo.yaml --force` and a new run ID. Later confirmatory collections of the same frozen plan require `--replication`.

## Sources and study preparation

[The source registry](data/sources/sources.yaml) records citations, URLs, licenses, hashes and variable definitions. [The source notes](docs/sources.md) describe the verification procedures.

```text
sim sources fetch
sim sources extract
sim sources verify
sim plan
sim estimate
sim doctor
```

Raw sources stay in `data/raw/` and are ignored by Git. The hold-out benchmarks are also ignored; extraction rebuilds them only when their hashes match the registry. Earlier commits contain these outcomes, and the original studies are public, so sealing does not establish that the material is unfamiliar to a model.

The study runs three models: Claude Opus 5.5 and Claude Sonnet 5.5 (training cutoff June 2026) and OpenAI GPT-6.1 Sol (cutoff April 30, 2026). The baseline has 800 responses per condition for framing and 900 for each held-out study. Only the held-out studies make confirmatory claims, and a study counts as reproduced only if its effect and every condition level are equivalent (`analysis.primary_test: joint`) and the model shows an effect of its own in the human direction (`analysis.primary_requires_effect`). Burger's human sample cannot support equivalence at the current h margin of 0.20 under the implemented test, so obedience is descriptive: 50 role-play responses per condition plus population forecasts, because its 70 participants bound the precision of any comparison. [The methods](docs/methods.md) explain the analyses and their limits. [The GSS notes](data/survey/README.md) explain the respondent panel.

## Collection and analysis

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
$env:OPENAI_API_KEY = "your-key"
sim run -c configs/smoke.yaml --exploratory --run-id smoke-1 --max-cost 5
sim analyze smoke-1 -c configs/smoke.yaml
sim estimate --from-run smoke-1
```

The pilot (`configs/smoke.yaml`, 48 trials, about $1.85 to $3.25) runs one or two trials of every kind the study runs, except on the held-out studies. `sim estimate --from-run smoke-1` then replaces the assumed tokens per trial with the measured ones, including format repairs and retries, and reports a typical and a cautious cost for the full plan. Before measuring, the estimate is $473.74 (cautious $875.42).

For the main study, `sim freeze` records the configuration, data and source identity. `sim run --dry-run` shows the pending plan without collecting responses. `sim run --run-id study-1` collects or resumes it; `--max-cost` caps what the run has spent in total, earlier invocations included. The first confirmatory run is registered in `prereg.<study_id>.runs.jsonl`; archive that record with the lock. Later runs need `--replication`.

`sim analyze study-1` uses the saved inputs and checks the analysis code. Incomplete collections require `--partial` and produce provisional outputs. Changed inputs or code require `--reanalyze`, which writes a separate exploratory analysis. `sim compare-runs` reports variation between independent runs. `sim recover` repairs an interrupted results file without starting collection. `sim audit` exports a blinded coding sample and a separate automatic-coding key.

## Files

| Path | Contents |
|---|---|
| `configs/` | Study, pilot and demo settings |
| `data/stimuli/` | Task materials and prompt variants |
| `data/benchmarks/` | Benchmarks, import specifications and templates |
| `data/personas/` | Demographic and GSS profiles with provenance |
| `data/sources/` | Source registry, extract and audit manifest |
| `src/silicon_subjects/` | Collection, coding, import and analysis code |
| `tests/` | Statistical and workflow regression tests |
| `docs/` | Methods and source checks |
| `results/` | Local runs, reports and tables |

## Checks

```text
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pytest -q -n 4
```

`make check` runs lint, types and tests. `make verify` also checks the sources and runs an isolated demo; the source files must first be available locally. `make lock` regenerates the dependency lock after a dependency change and requires uv.
