# Silicon Subjects

How closely do language models reproduce human choices, and when could their predictions reduce the burden on research participants? Adham Mater examines these questions in Silicon Subjects and the Limits of Simulating Human Behavior.

The study compares model responses with three published experiments. Many Labs 1 provides a gain–loss framing benchmark. Burger’s obedience replication provides a descriptive comparison at the 150-volt decision point. The Many Labs 2 moral-judgment task (Hauser.4) serves as the hold-out. The software records the prompts, responses, coding decisions and analysis inputs for each run.

**Study status.** This release contains the research design, imported framing data, a fixed General Social Survey (GSS) panel and an offline demonstration. Confirmatory model results are pending. The [source record](docs/sources.md) identifies the wording, benchmark and sampling decisions that remain open.

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

The report is `results/demo/report.md`; tables are in `results/demo/analysis/`. The demo uses mock models and synthetic benchmarks. Its output is a software check, not a thesis finding. `make demo` runs the same pipeline in a POSIX shell.

An unchanged run ID resumes unfinished trials. After changing the source or inputs, use `sim freeze -c configs/demo.yaml --force` and a new run ID. Later confirmatory collections of the same frozen plan require `--replication`.

## Sources and study preparation

[The source registry](data/sources/sources.yaml) records citations, URLs, licenses, hashes, variables and unresolved discrepancies. [The source notes](docs/sources.md) describe the checks and decisions still open.

```text
sim sources fetch
sim sources extract
sim sources verify
sim plan
sim estimate
sim doctor
```

Raw sources stay in `data/raw/` and are ignored by Git. The hold-out benchmarks are also ignored; extraction rebuilds them only when their hashes match the registry. Earlier commits contain these outcomes, and the original studies are public, so sealing does not establish that the material is unfamiliar to a model.

Before freezing the main study:

1. Sign off the framing and hold-out import specifications after reviewing the counts and coding.
2. Resolve the framing wording and choose the hold-out analysis set: published results or the four-second exclusion rule.
3. Check Burger's prods, learner script and counts against the paper.
4. Verify the model identifiers and training cutoffs, demographic targets and GSS weight choice.
5. Review the primary family, sample sizes, margins, token limits, timeouts and prices. The current primary family has four baseline comparisons across framing, the hold-out and two models. Obedience and persona comparisons are secondary or descriptive.
6. Run the exploratory pilot, review refusals and truncation, then freeze and archive the final plan with an independent timestamp.

The current baseline has 800 responses per condition for framing and the hold-out. Burger's human sample cannot support equivalence at the current h margin of 0.20 under the implemented test; obedience is therefore descriptive. [The methods](docs/methods.md) explain the analyses and their limits. [The GSS notes](data/survey/README.md) explain the respondent panel.

## Collection and analysis

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
sim run -c configs/smoke.yaml --exploratory --run-id smoke-1 --max-cost 5
sim analyze smoke-1 -c configs/smoke.yaml
```

For the main study, `sim freeze` records the configuration, data and source identity. `sim run --dry-run` shows the pending plan without collecting responses. `sim run --run-id study-1` collects or resumes it. The first confirmatory run is registered in `prereg.<study_id>.runs.jsonl`; archive that record with the lock. Later runs need `--replication`.

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
