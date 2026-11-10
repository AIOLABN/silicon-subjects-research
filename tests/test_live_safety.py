from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from silicon_subjects import design, personas, runner
from silicon_subjects.coding import Status
from silicon_subjects.config import StudyConfig
from silicon_subjects.providers import TransientProviderError
from silicon_subjects.providers.mock import MockProvider
from silicon_subjects.tasks import TaskResult, TrialSpec, build_tasks


def test_anthropic_rate_limit_carries_retry_after() -> None:
    anthropic = pytest.importorskip("anthropic")
    import httpx

    from silicon_subjects.config import ModelSpec
    from silicon_subjects.providers.anthropic_provider import AnthropicProvider

    class Messages:
        async def create(self, **kw: Any) -> Any:
            r = httpx.Response(
                429,
                headers={"retry-after": "7"},
                request=httpx.Request("POST", "https://api.anthropic.com"),
            )
            raise anthropic.RateLimitError("rate limited", response=r, body=None)

    class Client:
        messages = Messages()

    p = AnthropicProvider(ModelSpec(id="m", provider="anthropic", model="claude-opus-5-5"), 5)
    p._client = Client()
    with pytest.raises(TransientProviderError) as e:
        asyncio.run(p.complete("s", [], seed=1))
    assert e.value.retry_after == 7.0


class Flaky:
    def __init__(self, fails: int, retry_after: float | None) -> None:
        self.fails, self.retry_after = fails, retry_after

    async def run(self, trial: TrialSpec, persona: object, provider: object) -> TaskResult:
        if self.fails:
            self.fails -= 1
            raise TransientProviderError("429", retry_after=self.retry_after)
        return TaskResult(Status.VALID, {"y": 1}, [], input_tokens=1000, output_tokens=1000)


def _runner(cfg: StudyConfig, task: Any, **kw: Any) -> tuple[runner.Runner, list[TrialSpec]]:
    real = build_tasks(cfg)
    sets = personas.load_sets(cfg)
    r = runner.Runner(
        cfg,
        {k: task for k in real},
        personas.all_personas(sets),
        cfg.root / "results" / "run",
        prereg_hash=None,
        confirmatory=False,
        providers={m.id: MockProvider(m) for m in cfg.models},
        **kw,
    )
    return r, design.build_manifest(cfg, real, sets)


@pytest.mark.parametrize(("retry_after", "expected"), [(7.0, 7.0), (None, 0.0), (10_000.0, 300.0)])
def test_runner_waits_at_least_retry_after_but_capped(
    demo_cfg: StudyConfig,
    monkeypatch: pytest.MonkeyPatch,
    retry_after: float | None,
    expected: float,
) -> None:
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    r, trials = _runner(demo_cfg, Flaky(1, retry_after), retry_delay=lambda a: 0.0)
    rep = asyncio.run(r.run(trials, limit=1))
    assert rep.ok and slept == [expected]


def _priced(cfg: StudyConfig) -> StudyConfig:
    return cfg.model_copy(
        update={"models": [m.model_copy(update={"model": "claude-opus-5-5"}) for m in cfg.models]}
    )


def test_budget_stops_starting_new_trials(demo_cfg: StudyConfig) -> None:
    cfg = _priced(
        demo_cfg.model_copy(
            update={"runner": demo_cfg.runner.model_copy(update={"concurrency": 1})}
        )
    )

    r, trials = _runner(cfg, Flaky(0, None), max_cost_usd=0.05)
    rep = asyncio.run(r.run(trials, limit=10))
    assert rep.completed == 3
    assert rep.skipped_for_budget == 7 and rep.failed == []
    assert rep.spent_usd == pytest.approx(0.072)
    assert rep.pending == len(trials) - 3


def test_budget_counts_what_earlier_invocations_of_the_run_spent(demo_cfg: StudyConfig) -> None:
    cfg = _priced(
        demo_cfg.model_copy(
            update={"runner": demo_cfg.runner.model_copy(update={"concurrency": 1})}
        )
    )
    r, trials = _runner(cfg, Flaky(0, None), max_cost_usd=0.05)
    assert asyncio.run(r.run(trials, limit=10)).completed == 3

    # Resuming with the same cap starts nothing: the run has already spent more than the cap.
    r, _ = _runner(cfg, Flaky(0, None), max_cost_usd=0.05)
    rep = asyncio.run(r.run(trials, limit=10))
    assert rep.completed == 0 and rep.skipped_for_budget == 10
    assert rep.previously_completed == 3 and rep.spent_usd == pytest.approx(0.072)

    # A higher cap continues from the recorded spend, not from zero.
    r, _ = _runner(cfg, Flaky(0, None), max_cost_usd=0.10)
    rep = asyncio.run(r.run(trials, limit=10))
    assert rep.completed == 2 and rep.spent_usd == pytest.approx(0.12)

    # A failed attempt's tokens were paid for too, and are counted once.
    r, _ = _runner(cfg, Flaky(1, None), max_cost_usd=1.0, retry_delay=lambda a: 0.0)
    rep = asyncio.run(r.run(trials, limit=1))
    assert rep.completed == 1 and rep.spent_usd == pytest.approx(0.144)


def test_budget_refuses_a_run_whose_earlier_spend_is_unknown(demo_cfg: StudyConfig) -> None:
    cfg = _priced(demo_cfg)
    r, trials = _runner(cfg, Flaky(0, None))
    assert asyncio.run(r.run(trials, limit=1)).spent_usd == pytest.approx(0.024)
    # A record written before costs were recorded: tokens were used, their cost is unknown.
    path = r.paths.trials
    record = json.loads(path.read_text())
    del record["cost_usd"]
    path.write_text(json.dumps(record) + "\n")

    r, _ = _runner(cfg, Flaky(0, None))
    assert asyncio.run(r.run(trials, limit=0)).spent_usd is None
    r, _ = _runner(cfg, Flaky(0, None), max_cost_usd=1.0)
    with pytest.raises(ValueError, match="cannot be enforced"):
        asyncio.run(r.run(trials, limit=1))
    assert len(path.read_text().splitlines()) == 1


def test_budget_requires_known_prices(demo_cfg: StudyConfig) -> None:
    unpriced = demo_cfg.model_copy(
        update={
            "models": [
                m.model_copy(update={"model": f"local-model-{i}"})
                for i, m in enumerate(demo_cfg.models)
            ]
        }
    )
    r, trials = _runner(unpriced, Flaky(0, None), max_cost_usd=1.0)
    with pytest.raises(ValueError, match="no price"):
        asyncio.run(r.run(trials, limit=1))
    assert not (demo_cfg.root / "results" / "run" / "trials.jsonl").exists()


def test_spend_is_reported_without_a_cap(demo_cfg: StudyConfig) -> None:
    r, trials = _runner(_priced(demo_cfg), Flaky(0, None))
    rep = asyncio.run(r.run(trials, limit=2))
    assert rep.spent_usd == pytest.approx(0.048) and rep.skipped_for_budget == 0


def test_cli_max_cost(demo_cfg: StudyConfig) -> None:
    from typer.testing import CliRunner

    from silicon_subjects.cli import app

    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("concurrency: 8", "concurrency: 1"))
    res = CliRunner().invoke(
        app,
        [
            "run",
            "-c",
            str(path),
            "--exploratory",
            "--run-id",
            "b",
            "--limit",
            "20",
            "--max-cost",
            "0.0000001",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "spending cap" in res.output and "skipped (budget) 19" in res.output


def _doctor(cfg_path: Path) -> tuple[int, str]:
    from typer.testing import CliRunner

    from silicon_subjects.cli import app

    res = CliRunner().invoke(app, ["doctor", "-c", str(cfg_path)])
    return res.exit_code, res.output


def test_doctor_lists_blocking_problems(demo_cfg: StudyConfig) -> None:
    code, out = _doctor(demo_cfg.root / "configs" / "demo.yaml")
    assert code == 1
    assert "FAIL  preregistration: not frozen" in out
    assert "mock provider" in out
    assert "synthetic benchmark" in out
    assert "benchmark not marked verified" in out
    assert "WARN  primary power" in out and "WARN  level power" in out
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("    enabled: true\n", "    enabled: false\n"))
    _, out = _doctor(path)
    assert "no enabled evaluation study" in out


def test_doctor_detects_placeholders_and_missing_data(demo_cfg: StudyConfig) -> None:
    stim = demo_cfg.root / "data" / "stimuli" / "framing.yaml"
    stim.write_text(
        stim.read_text().replace(
            "Which of the two programs would you favor?", "TODO final question wording", 1
        )
    )
    bench = demo_cfg.root / "data" / "benchmarks" / "obedience_burger2009.yaml"
    bench.write_text(bench.read_text().replace("{k: 28, n: 40}", "{k: null, n: null}"))
    _, out = _doctor(demo_cfg.root / "configs" / "demo.yaml")
    assert "stimuli has TODO placeholders" in out
    assert "no human data for: base" in out


def test_doctor_blocks_stimuli_not_verified_against_their_source(demo_cfg: StudyConfig) -> None:
    _, out = _doctor(demo_cfg.root / "configs" / "demo.yaml")
    assert "FAIL  task framing: stimuli not marked verified against their source" in out
    stim = demo_cfg.root / "data" / "stimuli" / "framing.yaml"
    stim.write_text(stim.read_text().replace("verified: false", "verified: true", 1))
    _, out = _doctor(demo_cfg.root / "configs" / "demo.yaml")
    assert "task framing: stimuli not marked verified" not in out
    assert "FAIL  task obedience: stimuli not marked verified against their source" in out


def test_doctor_requires_credentials_for_live_models(
    demo_cfg: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(path.read_text().replace("provider: mock", "provider: anthropic"))
    _, out = _doctor(path)
    assert "FAIL  credentials: ANTHROPIC_API_KEY not set" in out
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _, out = _doctor(path)
    assert "PASS  credentials" in out


def test_doctor_never_reads_evaluation_outcomes(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    path.write_text(
        path.read_text().replace(
            "  - id: framing\n    split: development", "  - id: framing\n    split: evaluation", 1
        )
    )
    _, out = _doctor(path)
    assert "no enabled evaluation study" not in out

    assert "level power framing" not in out
    assert "primary power framing" in out
    assert "720" not in out


def test_doctor_reports_missing_inputs_instead_of_crashing(demo_cfg: StudyConfig) -> None:
    (demo_cfg.root / "data" / "personas" / "demo_profiles_v1.jsonl").unlink()
    code, out = _doctor(demo_cfg.root / "configs" / "demo.yaml")
    assert code == 1
    assert "FAIL  preregistration: cannot fingerprint the inputs" in out
    assert "FAIL  personas" in out


class HalfwayFlaky:
    def __init__(self) -> None:
        self.attempts = 0

    async def run(self, trial: TrialSpec, persona: object, provider: Any) -> TaskResult:
        self.attempts += 1
        res = TaskResult(Status.VALID, {"y": 1}, [])
        for _ in range(2):
            comp = await provider.complete("s", [], seed=1)
            res.input_tokens += comp.input_tokens or 0
            res.output_tokens += comp.output_tokens or 0
        if self.attempts == 1:
            raise TransientProviderError("529 overloaded")
        return res


class Billed(MockProvider):
    async def complete(self, *a: Any, **k: Any) -> Any:
        from silicon_subjects.providers.base import Completion

        return Completion("x", "end_turn", "claude-opus-5-5", input_tokens=1000, output_tokens=1000)


def test_spend_includes_discarded_attempts(demo_cfg: StudyConfig) -> None:
    cfg = _priced(demo_cfg)
    task = HalfwayFlaky()
    r, trials = _runner(cfg, task, retry_delay=lambda a: 0.0)
    r.providers = {m.id: Billed(m) for m in cfg.models}
    rep = asyncio.run(r.run(trials, limit=1))

    assert rep.ok and rep.spent_usd == pytest.approx(0.096)
    rec = runner.runstore.read_records(cfg.root / "results" / "run" / "trials.jsonl")[0]
    assert rec["input_tokens"] == 2000 and rec["discarded_input_tokens"] == 2000
    assert rec["cost_usd"] == pytest.approx(0.096)
    assert rec["attempt_errors"][0]["output_tokens"] == 2000


def test_spend_includes_failed_trials(demo_cfg: StudyConfig) -> None:
    class AlwaysHalfway(HalfwayFlaky):
        async def run(self, trial: TrialSpec, persona: object, provider: Any) -> TaskResult:
            self.attempts = 0
            return await super().run(trial, persona, provider)

    cfg = _priced(demo_cfg)
    cfg = cfg.model_copy(update={"runner": cfg.runner.model_copy(update={"max_attempts": 2})})
    r, trials = _runner(cfg, AlwaysHalfway(), retry_delay=lambda a: 0.0)
    r.providers = {m.id: Billed(m) for m in cfg.models}
    rep = asyncio.run(r.run(trials, limit=1))
    assert len(rep.failed) == 1 and rep.spent_usd == pytest.approx(0.096)
    assert rep.failed[0]["cost_usd"] == pytest.approx(0.096)


def test_anthropic_models_refuse_sampling_settings() -> None:
    from pydantic import ValidationError

    from silicon_subjects.config import ModelSpec

    with pytest.raises(ValidationError, match="does not accept sampling"):
        ModelSpec(
            id="m", provider="anthropic", model="claude-opus-5-5", sampling={"temperature": 1.0}
        )
    ModelSpec(id="o", provider="openai_compatible", model="x", sampling={"temperature": 1.0})


def test_doctor_fails_a_timeout_shorter_than_a_full_reply(demo_cfg: StudyConfig) -> None:
    path = demo_cfg.root / "configs" / "demo.yaml"
    text = path.read_text().replace("provider: mock", "provider: anthropic")
    path.write_text(text.replace("max_tokens: 2048", "max_tokens: 16000"))
    _, out = _doctor(path)
    assert "request_timeout_s" not in out
    path.write_text(path.read_text().replace("request_timeout_s: 600", "request_timeout_s: 180"))
    _, out = _doctor(path)
    assert "FAIL  model opus55: request_timeout_s 180 is shorter" in out


def test_missing_inputs_are_refused_with_instructions(demo_cfg: StudyConfig) -> None:
    from typer.testing import CliRunner

    from silicon_subjects.cli import app

    path = demo_cfg.root / "configs" / "demo.yaml"
    (demo_cfg.root / "data" / "benchmarks" / "demo" / "holdout_synthetic.yaml").unlink()
    (demo_cfg.root / "data" / "personas" / "demo_profiles_v1.jsonl").unlink()
    for cmd in (["plan"], ["estimate"], ["freeze"], ["verify"], ["run", "--exploratory"]):
        res = CliRunner().invoke(app, [*cmd, "-c", str(path)])
        assert res.exit_code == 2, (cmd, res.output)
        assert res.exception is None or isinstance(res.exception, SystemExit)
        assert "holdout.benchmark" in res.output and "sim sources extract" in res.output
        assert "personas" in res.output and "sim personas" in res.output


def test_dependency_lock_is_recorded_and_differences_reported(tmp_path: Path) -> None:
    import importlib.metadata

    from silicon_subjects import prereg

    assert prereg.dependency_lock(tmp_path)["file"] is None
    real = importlib.metadata.version("pydantic")
    (tmp_path / "requirements.lock").write_text(
        f"pydantic=={real} \\\n    --hash=sha256:00\npyyaml==0.0.1 ; python_version < '3.99'\n"
    )
    sub = tmp_path / "configs"
    sub.mkdir()
    lock = prereg.dependency_lock(sub)
    assert lock["file"] == "requirements.lock" and lock["pinned"] == 2
    assert set(lock["mismatches"]) == {"pyyaml"}
    assert lock["mismatches"]["pyyaml"]["locked"] == ["0.0.1"]


def test_repository_lock_pins_every_declared_dependency() -> None:
    import tomllib

    from silicon_subjects import prereg

    repo = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    declared = list(project["dependencies"]) + [
        d for extra in project["optional-dependencies"].values() for d in extra
    ]
    text = (repo / "requirements.lock").read_text()
    pinned = {prereg._norm_dist(m.group(1)) for m in map(prereg._PIN.match, text.splitlines()) if m}
    import re

    names = {prereg._norm_dist(re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0]) for d in declared}
    assert names <= pinned, names - pinned
    assert "--hash=sha256:" in text


def test_config_price_makes_an_unlisted_model_cappable(demo_cfg: StudyConfig) -> None:
    from silicon_subjects import pricing

    priced = demo_cfg.model_copy(
        update={
            "models": [
                m.model_copy(
                    update={"model": f"local-model-{i}", "price_usd_per_mtok": (4.0, 20.0)}
                )
                for i, m in enumerate(demo_cfg.models)
            ]
        }
    )
    assert pricing.price_for(priced.models[0]) == (4.0, 20.0)
    assert pricing.cost_for(priced.models[0], 1000, 1000) == pytest.approx(0.024)
    r, trials = _runner(
        priced.model_copy(update={"runner": priced.runner.model_copy(update={"concurrency": 1})}),
        Flaky(0, None),
        max_cost_usd=0.05,
    )
    rep = asyncio.run(r.run(trials, limit=10))
    assert rep.completed == 3 and rep.skipped_for_budget == 7


def test_records_say_whether_the_seed_reached_the_model(demo_cfg: StudyConfig) -> None:
    import json

    from silicon_subjects.config import ModelSpec
    from silicon_subjects.providers import seed_is_sent

    assert not seed_is_sent(ModelSpec(id="a", provider="anthropic", model="claude-opus-5-5"))
    assert seed_is_sent(ModelSpec(id="o", provider="openai_compatible", model="x", base_url="u"))
    real = build_tasks(demo_cfg)
    sets = personas.load_sets(demo_cfg)
    trials = design.build_manifest(demo_cfg, real, sets)
    r = runner.Runner(
        demo_cfg,
        real,
        personas.all_personas(sets),
        demo_cfg.root / "results" / "run",
        prereg_hash=None,
        confirmatory=False,
        providers={m.id: MockProvider(m) for m in demo_cfg.models},
    )
    asyncio.run(r.run(trials, limit=2))
    run = demo_cfg.root / "results" / "run"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["seed_sent_to_provider"] == {m.id: True for m in demo_cfg.models}
    rec = json.loads((run / "trials.jsonl").read_text().splitlines()[0])
    assert rec["generation"]["seed_sent"] is True


def test_doctor_flags_single_developer_model_sets(demo_cfg: StudyConfig) -> None:
    from silicon_subjects import doctor

    live = demo_cfg.model_copy(
        update={"models": [m.model_copy(update={"provider": "anthropic"}) for m in demo_cfg.models]}
    )
    checks = {c.item: c for c in doctor.run(live)}
    assert checks["model diversity"].level == "WARN"
    assert "model diversity" not in {c.item for c in doctor.run(demo_cfg)}


def _openai_spec(**kw: Any) -> Any:
    from silicon_subjects.config import ModelSpec

    return ModelSpec(
        id="gpt",
        provider="openai_compatible",
        base_url="https://api.example.test/v1",
        model="gpt-6.1-sol",
        api_key_env="OPENAI_API_KEY",
        max_tokens=16000,
        max_tokens_param="max_completion_tokens",
        **kw,
    )


def _fake_openai(spec: Any, reply: dict[str, Any], seen: list[Any]) -> Any:
    import httpx

    from silicon_subjects.providers.openai_compatible import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=reply)

    p = OpenAICompatibleProvider(spec, 5)
    headers = dict(p._client.headers)
    p._client = httpx.AsyncClient(
        base_url=spec.base_url, headers=headers, transport=httpx.MockTransport(handler)
    )
    return p


def test_openai_provider_uses_the_configured_key_and_token_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from silicon_subjects.providers.base import ChatMessage

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    seen: list[Any] = []
    reply = {
        "model": "gpt-6.1-sol-2026-09-01",
        "choices": [{"message": {"content": "CHOICE: A"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    p = _fake_openai(_openai_spec(), reply, seen)
    comp = asyncio.run(p.complete("sys", [ChatMessage("user", "hi")], seed=7))
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert body["max_completion_tokens"] == 16000 and "max_tokens" not in body
    assert body["seed"] == 7 and comp.text == "CHOICE: A" and not comp.platform_refusal


def test_openai_refusal_field_is_a_platform_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    from silicon_subjects.providers.base import ChatMessage

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    reply = {
        "model": "gpt-6.1-sol",
        "choices": [
            {
                "message": {"content": None, "refusal": "I can't help with that."},
                "finish_reason": "stop",
            }
        ],
    }
    p = _fake_openai(_openai_spec(), reply, [])
    comp = asyncio.run(p.complete("sys", [ChatMessage("user", "hi")], seed=1))
    assert comp.platform_refusal and comp.text == "I can't help with that."


def test_model_options_are_checked_against_the_provider() -> None:
    from pydantic import ValidationError

    from silicon_subjects.config import ModelSpec

    with pytest.raises(ValidationError, match="effort"):
        _openai_spec(effort="medium")
    with pytest.raises(ValidationError, match="openai_compatible models only"):
        ModelSpec(id="c", provider="anthropic", model="claude-opus-5-5", api_key_env="X")
    assert _openai_spec().key_env == "OPENAI_API_KEY"
    default = ModelSpec(id="o", provider="openai_compatible", model="m", base_url="u")
    assert default.key_env == "OPENAI_COMPATIBLE_API_KEY"


def test_doctor_requires_each_providers_key(
    demo_cfg: StudyConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from silicon_subjects import doctor

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = demo_cfg.model_copy(update={"models": [*demo_cfg.models, _openai_spec()]})
    creds = [c for c in doctor.run(cfg) if c.item == "credentials"]
    assert any(c.level == "FAIL" and "OPENAI_API_KEY not set for gpt" in c.detail for c in creds)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    creds = [c for c in doctor.run(cfg) if c.item == "credentials"]
    assert any(c.level == "PASS" and "OPENAI_API_KEY found" in c.detail for c in creds)
