"""Run and analyze the Silicon Subjects experiments."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Annotated, Any

import typer

from . import analysis, costs, design, personas, prereg, probes, runstore
from .config import StudyConfig, load_config
from .hashing import stable_id
from .providers import build_provider
from .runner import RunIdentityError, Runner
from .tasks import Task, TrialSpec, build_tasks

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
sources_app = typer.Typer(
    no_args_is_help=True,
    help="External research sources: hash-pinned fetch, extracts, provenance audit, verify.",
)
app.add_typer(sources_app, name="sources")
RootOpt = Annotated[Path, typer.Option("--root", help="Repository root")]

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Study config YAML")]
DEFAULT_CONFIG = Path("configs/study.yaml")


def _setup(config: Path, *, inputs: bool = False) -> StudyConfig:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(config)
    if inputs and (missing := missing_inputs(cfg)):
        typer.echo("missing input files:", err=True)
        for line in missing:
            typer.echo(f"  {line}", err=True)
        raise typer.Exit(2)
    return cfg


def missing_inputs(cfg: StudyConfig) -> list[str]:
    split = {t.id: t.split for t in cfg.tasks}
    out = []
    for name, path in prereg.tracked_files(cfg).items():
        if path.is_file():
            continue
        task, _, part = name.partition(".")
        if name == "personas":
            how = "run `sim personas`"
        elif name == "survey_personas":
            how = "run `sim survey-personas` (see data/survey/README.md)"
        elif name == "calibration":
            how = "run `sim calibrate`, or unset analysis.calibration.file"
        elif part == "benchmark" and split.get(task) == "evaluation":
            how = (
                "sealed evaluation benchmarks are never committed: run `sim sources fetch`, "
                "then `sim sources extract`"
            )
        else:
            how = "check the path in the config"
        out.append(f"{name}: {path} ({how})")
    return out


def _results_dir(cfg: StudyConfig) -> Path:
    return cfg.root / "results"


@app.command("personas")
def personas_cmd(config: ConfigOpt = DEFAULT_CONFIG, force: bool = False) -> None:
    """Generate the persona file (refuses to overwrite unless --force)."""
    cfg = _setup(config)
    path = cfg.path(cfg.personas.file)
    if path.exists() and not force:
        raise typer.BadParameter(f"{path} exists; use --force to regenerate")
    ps = personas.generate(cfg.personas, cfg.seed)
    personas.save(ps, path)
    typer.echo(f"wrote {len(ps)} profiles to {path}")
    typer.echo(json.dumps(personas.marginals(ps, cfg.personas.attributes), indent=2))


@app.command("survey-personas")
def survey_personas_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    raw: Annotated[Path | None, typer.Option(help="Override survey_personas.raw_file")] = None,
    force: bool = False,
) -> None:
    """Draw profiles of real respondents from survey microdata (refuses to overwrite)."""
    from . import survey_personas

    cfg = _setup(config)
    spec = cfg.survey_personas
    if spec is None:
        raise typer.BadParameter("config has no survey_personas section")
    path = cfg.path(spec.file)
    if path.exists() and not force:
        raise typer.BadParameter(f"{path} exists; use --force to regenerate")
    raw_path = raw or cfg.path(spec.raw_file)
    if not raw_path.exists():
        raise typer.BadParameter(f"{raw_path} not found; see data/survey/README.md")
    ps, rep, provenance = survey_personas.build(spec, raw_path, cfg.seed)
    personas.save(ps, path)
    rep.to_csv(path.with_suffix(".representativeness.csv"), index=False)
    path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    typer.echo(f"wrote {len(ps)} respondent profiles to {path}")
    tvd = rep.groupby("attribute", sort=False)["attribute_tvd"].first()
    typer.echo("total variation distance, sample vs weighted population (0 = identical):")
    typer.echo(tvd.round(3).to_string())


@app.command()
def plan(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Print the design: trials per cell and totals."""
    cfg = _setup(config, inputs=True)
    tasks = build_tasks(cfg)
    trials = design.build_manifest(cfg, tasks, personas.load_sets(cfg))
    for (m, t, a, c, v), n in sorted(design.summarize(trials).items()):
        typer.echo(f"{m:12} {t:16} {a:12} {c:16} {v:10} n={n}")
    typer.echo(f"\nTOTAL trials: {len(trials)}")


@app.command()
def doctor(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Check readiness for a confirmatory run; exits 1 if anything FAILs. Calls no API."""
    from . import doctor as doc

    cfg = _setup(config)
    checks = doc.run(cfg)
    for c in sorted(checks, key=lambda c: ["FAIL", "WARN", "INFO", "PASS"].index(c.level)):
        typer.echo(f"{c.level:4}  {c.item}: {c.detail}")
    fails = sum(c.level == "FAIL" for c in checks)
    typer.echo(f"\n{fails} blocking problem(s)" if fails else "\nready for a confirmatory run")
    raise typer.Exit(1 if fails else 0)


@app.command()
def estimate(
    config: ConfigOpt = DEFAULT_CONFIG,
    from_run: Annotated[
        list[str] | None,
        typer.Option(
            "--from-run",
            help="Run id (under results/) or run directory whose measured per-trial tokens "
            "replace the assumptions for the same model, task and method; repeatable. The "
            "exploratory pilot (configs/smoke.yaml) is the intended source.",
        ),
    ] = None,
    thinking_multiplier: Annotated[
        float,
        typer.Option(help="Scale assumed output tokens (rows not measured by --from-run)"),
    ] = 1.0,
) -> None:
    """Estimate API cost of the full plan (nothing is called)."""
    cfg = _setup(config, inputs=True)
    tasks = build_tasks(cfg)
    trials = design.build_manifest(cfg, tasks, personas.load_sets(cfg))
    runs = [Path(r) if Path(r).is_dir() else _results_dir(cfg) / r for r in from_run or []]
    try:
        measured = costs.measure(runs)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    rows = costs.estimate(cfg, tasks, trials, measured, thinking_multiplier)
    for r in rows:
        who = "persona" if r.persona else "-"
        typer.echo(
            f"{r.model_id:10} {r.task_id:10} {r.method:20} {who:7} trials={r.trials:5}  "
            f"est=${r.expected_usd:8.2f}  cautious=${r.cautious_usd:8.2f}  {r.source}"
        )
    assumed = sum(r.trials for r in rows if r.source == "assumed")
    typer.echo(
        f"\nTOTAL {len(trials)} trials, estimated ${costs.total(rows):.2f} "
        f"(cautious ${costs.total(rows, 'cautious_usd'):.2f}: 90th-percentile tokens per trial "
        "where measured, doubled output where assumed)"
    )
    typer.echo(
        f"{assumed} trial(s) use assumed tokens (costs.ASSUMED_TOKENS); "
        + (
            "measure them with the pilot: `sim run -c configs/smoke.yaml --exploratory`, then "
            "`sim estimate --from-run <pilot run id>`"
            if assumed
            else "every row is measured or a measured proxy"
        )
        + ". Failed trials are not included."
    )


@app.command()
def calibrate(
    run_ids: list[str],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Fit effect calibration from DEVELOPMENT runs only and write the frozen parameter file."""
    from . import calibration

    cfg = _setup(config, inputs=True)
    spec = cfg.analysis.calibration
    effects: list[dict[str, object]] = []
    for rid in run_ids:
        out = analysis.analyze_run(_results_dir(cfg) / rid)
        effects += out.result.tables["method_effects"].to_dict("records")
    fits = {}
    for arm in sorted({str(e["arm_id"]) for e in effects}):
        points = calibration.development_points(cfg, effects, arm)
        try:
            fits[arm] = calibration.fit(points, spec.min_development_studies)
        except calibration.CalibrationError as e:
            typer.echo(f"{arm}: not calibrated ({e})")
    if not fits or spec.file is None:
        typer.echo("no calibration written (see docs/methods.md for the data needed)")
        raise typer.Exit(1)
    calibration.write(cfg.path(spec.file), fits, run_ids)
    typer.echo(f"wrote {spec.file}; set calibration.enabled and re-freeze before evaluation")


@app.command()
def freeze(config: ConfigOpt = DEFAULT_CONFIG, force: bool = False) -> None:
    """Write prereg.<study_id>.lock.json. Commit it and upload it to OSF before running."""
    cfg = _setup(config, inputs=True)
    path = prereg.freeze(cfg, force=force)
    typer.echo(f"frozen: {path}\n{path.read_text()}")


@app.command()
def verify(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Check that nothing changed since the preregistration was frozen."""
    cfg = _setup(config, inputs=True)
    s = prereg.verify(cfg)
    typer.echo(s.describe())
    raise typer.Exit(0 if s.matches else 1)


@app.command()
def run(
    config: ConfigOpt = DEFAULT_CONFIG,
    run_id: Annotated[str | None, typer.Option(help="Resume an existing run")] = None,
    limit: Annotated[
        int | None, typer.Option(help="Run at most N pending trials (smoke test)")
    ] = None,
    exploratory: Annotated[
        bool, typer.Option(help="Allow running without a matching prereg lock")
    ] = False,
    max_cost: Annotated[
        float | None,
        typer.Option(
            help="Stop starting new trials once the run's measured spend reaches this many USD, "
            "counting earlier invocations of the same run ID (in-flight trials finish). Raise "
            "the cap to continue a run that reached it."
        ),
    ] = None,
    recover_tail: Annotated[
        bool,
        typer.Option(
            help="Repair an incomplete final line in trials.jsonl/errors.jsonl left by a crash. "
            "The original file is kept as a .bak and the action is logged to recovery.jsonl."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would run (labels, pending trials, estimated cost); call no API and "
            "write nothing.",
        ),
    ] = False,
    replication: Annotated[
        bool,
        typer.Option(
            help="Start a new confirmatory run of a plan that already has its primary run. It is "
            "registered and reported as a replication, never as the primary result."
        ),
    ] = False,
) -> None:
    """Execute (or resume) the trial manifest."""
    cfg = _setup(config, inputs=True)
    status = prereg.verify(cfg)
    confirmatory = status.locked and status.matches
    if not confirmatory and not exploratory:
        typer.echo(
            f"Not a confirmatory run: {status.describe()}. Run `sim freeze`, or pass "
            "--exploratory to label this run as exploratory.",
            err=True,
        )
        raise typer.Exit(2)
    tasks = build_tasks(cfg)
    sets = personas.load_sets(cfg)
    trials = design.build_manifest(cfg, tasks, sets)
    if dry_run:
        _dry_run(cfg, tasks, trials, run_id, limit, confirmatory)
        return
    run_id = run_id or f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-{status.actual[:8]}"
    out = _results_dir(cfg) / run_id
    runner = Runner(
        cfg,
        tasks,
        personas.all_personas(sets),
        out,
        prereg_hash=status.expected if confirmatory else None,
        confirmatory=confirmatory,
        max_cost_usd=max_cost,
        replication=replication,
    )
    try:
        rep = asyncio.run(runner.run(trials, limit=limit, recover_tail=recover_tail))
    except (RunIdentityError, runstore.ResultsCorruptError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    typer.echo(f"{rep.summary()} -> {out}")
    if rep.failed:
        typer.echo(
            f"{len(rep.failed)} trial(s) failed (details in {out / 'errors.jsonl'}); "
            f"resume with --run-id {run_id} to retry them",
            err=True,
        )
        raise typer.Exit(1)
    if rep.skipped_for_budget:
        typer.echo(
            f"spending cap ${max_cost:.2f} reached: {rep.skipped_for_budget} selected trial(s) "
            f"not started; the cap counts everything this run has spent, so resume with "
            f"--run-id {run_id} and a higher --max-cost"
        )
    if not rep.study_complete:
        typer.echo(
            f"study incomplete: {rep.pending} trial(s) pending; resume with --run-id {run_id}"
        )


def _dry_run(
    cfg: StudyConfig,
    tasks: dict[str, Task],
    trials: list[TrialSpec],
    run_id: str | None,
    limit: int | None,
    confirmatory: bool,
) -> None:
    done: set[str] = set()
    if run_id and (path := _results_dir(cfg) / run_id / "trials.jsonl").is_file():
        done = {r["trial_id"] for r in runstore.read_records(path)}
    pending = [t for t in trials if t.trial_id not in done]
    selected = pending[:limit] if limit is not None else pending
    cost = costs.total(costs.estimate(cfg, tasks, selected))
    by_task: dict[str, int] = {}
    for t in selected:
        by_task[t.task_id] = by_task.get(t.task_id, 0) + 1
    label = "confirmatory" if confirmatory else "exploratory"
    typer.echo(f"DRY RUN: no API calls, nothing written ({label})")
    typer.echo(
        f"  planned trials: {len(trials)}; already done: {len(done)}; pending: {len(pending)}"
    )
    typer.echo(f"  would start now: {len(selected)} {by_task}")
    typer.echo(f"  estimated cost of those: ${cost:.2f} (see `sim estimate` for assumptions)")
    remote = {m.provider for m in cfg.models} - {"mock"}
    if "anthropic" in remote and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        typer.echo("  note: ANTHROPIC_API_KEY is not set; a real run would fail to authenticate")
    for env in sorted({m.key_env or "" for m in cfg.models if m.provider == "openai_compatible"}):
        if env and not os.environ.get(env):
            typer.echo(f"  note: {env} is not set; a real run would fail to authenticate")


@app.command("compare-runs")
def compare_runs(
    run_ids: Annotated[list[str], typer.Argument(help="Two or more run ids of the same plan")],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Between-run variability of every cell (AI outputs only; no human data is read)."""
    from . import evaluation

    if len(run_ids) < 2:
        raise typer.BadParameter("give at least two run ids")
    cfg = _setup(config, inputs=True)
    tasks = build_tasks(cfg)
    frames, identities = {}, set()
    for rid in run_ids:
        d = _results_dir(cfg) / rid
        if not (d / "trials.jsonl").is_file():
            typer.echo(f"no results for run {rid!r} in {d}", err=True)
            raise typer.Exit(2)
        manifest = (
            json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").is_file() else {}
        )
        identities.add(json.dumps(manifest.get("experiment_identity"), sort_keys=True))
        frames[rid] = analysis.load_trials(runstore.read_records(d / "trials.jsonl"), tasks)
    if len(identities) > 1:
        typer.echo(
            "warning: the runs have different experiment identities (model, data, code or "
            "design changed); differences mix run-to-run noise with those changes",
            err=True,
        )
    table = evaluation.between_run(
        frames, cfg, tasks, personas.all_personas(personas.load_sets(cfg))
    )
    out = _results_dir(cfg) / "compare" / "__".join(run_ids)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "between_run.csv", index=False)
    if table.empty:
        typer.echo("no cell appears in two or more runs")
        return
    typer.echo(f"wrote {out / 'between_run.csv'}: {len(table)} cells in {len(run_ids)} runs")
    typer.echo(f"  largest between-run range: {table['run_range'].max():.3f}")
    if "homogeneity_p" in table:
        typer.echo(
            f"  cells with homogeneity p < 0.05: {int((table['homogeneity_p'] < 0.05).sum())}"
        )


@app.command()
def analyze(
    run_id: str,
    config: ConfigOpt = DEFAULT_CONFIG,
    reanalyze: Annotated[
        bool,
        typer.Option(
            help="Exploratory reanalysis with the CURRENT config, data files and code. Writes to "
            "results/<run>/reanalysis/<timestamp>/ and never touches the run's own report."
        ),
    ] = False,
    partial: Annotated[
        bool,
        typer.Option(help="Allow an incomplete collection; outputs are labelled provisional."),
    ] = False,
) -> None:
    """Analyze a run with the inputs it was collected with, and write report.md."""
    cfg = _setup(config, inputs=reanalyze)
    run_dir = _results_dir(cfg) / run_id
    try:
        out = analysis.analyze_run(
            run_dir, reanalyze_with=cfg if reanalyze else None, partial=partial
        )
    except analysis.AnalysisProvenanceError as e:
        typer.echo(f"analysis refused: {e}", err=True)
        raise typer.Exit(2) from e
    typer.echo(f"analysis status: {out.provenance['analysis_status']}")
    typer.echo(f"report: {out.report}")


@app.command()
def probe(config: ConfigOpt = DEFAULT_CONFIG, n: int = 5) -> None:
    """Run contamination probes for every model x task."""
    cfg = _setup(config, inputs=True)
    tasks = build_tasks(cfg)
    providers = {m.id: build_provider(m, cfg.runner.request_timeout_s) for m in cfg.models}
    out = _results_dir(cfg) / "probes" / f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}"
    summary = asyncio.run(probes.run_probes(cfg, tasks, providers, n, out))
    typer.echo(summary.to_string(index=False))
    typer.echo(f"-> {out}")


@app.command("import-benchmark")
def import_benchmark_cmd(
    task: Annotated[str, typer.Option(help="Task id whose benchmark is being filled")],
    raw: Annotated[Path, typer.Option(help="Raw human data file (csv, tsv, dta, sav)")],
    spec: Annotated[Path, typer.Option(help="Import spec YAML (see importer.py)")],
    out: Annotated[
        Path | None,
        typer.Option(help="Benchmark YAML to write (default: the task's benchmark in the config)"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Compute and report; write nothing")
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Write nothing; exit 1 unless re-importing reproduces --out exactly"
        ),
    ] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Compute benchmark counts from raw human data, with full import provenance."""
    from . import importer

    cfg = _setup(config)
    t = next((x for x in cfg.tasks if x.id == task), None)
    if t is None:
        raise typer.BadParameter(f"unknown task {task!r} (tasks: {[x.id for x in cfg.tasks]})")
    for label, path in (("raw file", raw), ("spec", spec)):
        if not path.is_file():
            typer.echo(f"import refused: {label} not found: {path}", err=True)
            raise typer.Exit(2)
    target = out or cfg.path(t.benchmark)
    try:
        if check:
            same = importer.matches_existing(raw, spec, target, cfg.path(t.benchmark), cfg.root)
            typer.echo(f"{target}: {'reproduced exactly' if same else 'DIFFERS from a re-import'}")
            raise typer.Exit(0 if same else 1)
        bench = importer.import_benchmark(
            cfg.path(t.benchmark), raw, spec, target, write=not dry_run, root=cfg.root
        )
    except importer.ImportError_ as e:
        typer.echo(f"import refused: {e}", err=True)
        raise typer.Exit(2) from e
    prov = bench["import_provenance"]
    verb = "dry run (nothing written)" if dry_run else f"wrote {target}"
    typer.echo(f"{verb}: {prov['rows_used']} of {prov['rows_in_file']} rows used")
    for reason, n in prov["excluded"].items():
        typer.echo(f"  excluded ({reason}): {n}")
    for col, counts in prov["flagged_rows_kept"].items():
        typer.echo(f"  flagged but kept ({col}): {counts}")
    for cond, c in bench["conditions"].items():
        shown = c if t.split == "development" else {"n": c["n"]}
        typer.echo(f"  {cond}: {shown}")
    if t.split == "evaluation":
        typer.echo("  evaluation study: outcomes written but not shown (sealed)")
        _, text = importer.build(raw, spec, cfg.path(t.benchmark), cfg.root)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        typer.echo(
            f"  sha256 {digest}: pin it as the benchmark's `sha256` in data/sources/sources.yaml "
            "(the file itself stays uncommitted)"
        )
    typer.echo(f"  sites: {len(bench.get('sites', []))}")
    typer.echo(f"  verified: {bench['verified']} (sign-off is recorded in the import spec)")
    if not dry_run and target != cfg.path(t.benchmark):
        typer.echo(
            f"Note: {target} is not the configured benchmark of task {task}; to use it, point "
            f"tasks[{task}].benchmark at it and re-freeze."
        )


@sources_app.command("fetch")
def sources_fetch(
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Source id (repeatable); default: all")
    ] = None,
    root: RootOpt = Path("."),
) -> None:
    """Download missing sources into data/raw/; keep a file only if its SHA-256 matches."""
    from . import sources

    results = sources.fetch(sources.load_registry(root), root, only)
    for r in results:
        typer.echo(f"{r.status:14} {r.source}: {r.detail}")
    if any(r.status in ("hash_mismatch", "failed") for r in results):
        typer.echo(
            "Some sources could not be fetched. Download them manually to the path shown in "
            "data/sources/sources.yaml and run the command again.",
            err=True,
        )
        raise typer.Exit(1)


@sources_app.command("extract")
def sources_extract(root: RootOpt = Path(".")) -> None:
    """Write the committed extracts (e.g. the CC0 Many Labs 1 framing subset) from raw files."""
    from . import sources

    reg = sources.load_registry(root)
    done = sources.write_extracts(reg, root)
    for ext in reg.extracts:
        if ext.id in done:
            typer.echo(f"wrote {ext.path} (sha256 {done[ext.id]})")
        else:
            typer.echo(f"skipped {ext.id}: source {ext.from_} missing or not matching its pin")

    failed = False
    for b in sources.write_sealed_benchmarks(reg, root):
        message = {
            "written": "rebuilt (matches its committed sha256; outcomes not shown)",
            "unchanged": "present and matches its committed sha256",
            "source_missing": "skipped: source missing or not matching its pin "
            "(run `sim sources fetch`)",
            "pin_mismatch": "NOT written: the rebuild does not match the committed sha256",
        }[b.status]
        failed |= b.status == "pin_mismatch"
        typer.echo(f"sealed benchmark {b.out}: {message}")
    if failed:
        raise typer.Exit(1)


@sources_app.command("audit")
def sources_audit(root: RootOpt = Path(".")) -> None:
    """Write data/sources/provenance.json from the registry, local files and cross-checks."""
    from . import sources

    m = sources.write_manifest(root)
    sm = m["summary"]
    typer.echo(f"wrote {sources.MANIFEST}: {sm['sources']} sources")
    typer.echo(f"  verified:      {', '.join(sm['verified']) or '-'}")
    typer.echo(f"  missing:       {', '.join(sm['missing']) or '-'}")
    if sm["unpinned"] or sm["hash_mismatch"]:
        typer.echo(f"  unpinned:      {', '.join(sm['unpinned']) or '-'}")
        typer.echo(f"  hash mismatch: {', '.join(sm['hash_mismatch']) or '-'}")
    typer.echo(
        f"  checks: {sm['checks_passed']} passed, {len(sm['checks_failed'])} failed, "
        f"{len(sm['checks_skipped'])} skipped"
    )
    for c in m["checks"]:
        if c["passed"] is False:
            typer.echo(f"  FAIL {c['id']}: {c['detail']}")
    if sm["checks_failed"] or sm["hash_mismatch"]:
        raise typer.Exit(1)


@sources_app.command("verify")
def sources_verify(
    require_all: Annotated[
        bool, typer.Option("--require-all", help="Treat a missing source as a failure")
    ] = False,
    root: RootOpt = Path("."),
) -> None:
    """Recompute everything and compare with the committed manifest (exit 1 on any FAIL)."""
    from . import sources

    findings = sources.verify(root, require_all=require_all)
    for f in findings:
        typer.echo(f"{f.level:4}  {f.item}: {f.detail}")
    n_fail = sum(f.level == "FAIL" for f in findings)
    typer.echo(
        f"{sum(f.level == 'PASS' for f in findings)} passed, {n_fail} failed, "
        f"{sum(f.level == 'SKIP' for f in findings)} skipped"
    )
    if n_fail:
        raise typer.Exit(1)


@app.command()
def recover(run_id: str, config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Repair an incomplete final line left by a crash. Never runs trials or calls a model."""
    cfg = _setup(config)
    run_dir = _results_dir(cfg) / run_id
    try:
        for name in ("trials.jsonl", "errors.jsonl"):
            action = runstore.recover_tail(run_dir / name)
            if action:
                typer.echo(
                    f"{name}: removed {action['removed_bytes']} bytes of an incomplete record, "
                    f"kept {action['kept_records']} records; original saved to {action['backup']}"
                )
            else:
                typer.echo(f"{name}: nothing to recover")
        runstore.read_records(run_dir / "trials.jsonl")
        runstore.read_jsonl(run_dir / "errors.jsonl")
    except runstore.ResultsCorruptError as e:
        typer.echo(f"recover refused: {e}", err=True)
        raise typer.Exit(2) from e


def blinded_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": t["role"], "content": t["content"]} for t in transcript if t["role"] != "system"
    ]


@app.command()
def audit(run_id: str, config: ConfigOpt = DEFAULT_CONFIG, per_status: int = 25) -> None:
    """Export a blinded, stratified sample of transcripts for independent human coding."""
    cfg = _setup(config)
    run_dir = _results_dir(cfg) / run_id
    try:
        rows = runstore.read_records(run_dir / "trials.jsonl")
    except runstore.ResultsCorruptError as e:
        typer.echo(f"audit refused: {e}", err=True)
        raise typer.Exit(2) from e
    rng = random.Random(cfg.seed)
    by_status: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r)
    sample = [
        r for group in by_status.values() for r in rng.sample(group, min(per_status, len(group)))
    ]
    rng.shuffle(sample)
    out = run_dir / "audit"
    out.mkdir(exist_ok=True)
    with (out / "to_code.jsonl").open("w") as f, (out / "key.jsonl").open("w") as k:
        for r in sample:
            aid = stable_id(r["trial_id"], "audit")
            stratum = len(by_status[r["status"]])
            f.write(
                json.dumps(
                    {
                        "audit_id": aid,
                        "task_id": r["task_id"],
                        "transcript": blinded_transcript(r["transcript"]),
                        "human_status": None,
                        "human_value": None,
                    }
                )
                + "\n"
            )
            k.write(
                json.dumps(
                    {
                        "audit_id": aid,
                        "trial_id": r["trial_id"],
                        "status": r["status"],
                        "outcome": r["outcome"],
                        "stratum_size": stratum,
                        "sampling_weight": stratum / min(per_status, stratum),
                    }
                )
                + "\n"
            )
    typer.echo(f"wrote {len(sample)} items to {out}")


if __name__ == "__main__":
    app()
