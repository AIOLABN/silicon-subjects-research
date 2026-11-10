from __future__ import annotations

import asyncio
import difflib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import StudyConfig
from .providers import ChatMessage, Provider, TransientProviderError
from .tasks import Task

PROBE_SYSTEM = "You are a knowledgeable assistant."


GENERIC_KEYWORDS = frozenset(
    {
        "trolley",
        "double effect",
        "obedience",
        "teacher",
        "learner",
        "framing",
        "moral dilemma",
    }
)


def recognition_keywords(task: Task) -> list[str]:
    keywords = [str(k).lower() for k in task.benchmark.get("recognition_keywords", [])]
    if generic := sorted(set(keywords) & GENERIC_KEYWORDS):
        raise ValueError(
            f"{task.id}: recognition_keywords {generic} name the paradigm, not the study; a "
            "model that does not recognize the study would use them too"
        )
    if not keywords:
        raise ValueError(f"{task.id}: benchmark has no recognition_keywords")
    return keywords


async def _call(
    provider: Provider, prompt: str, seed: int, sem: asyncio.Semaphore, attempts: int
) -> str:
    from .runner import MAX_WAIT_S, default_retry_delay

    for attempt in range(1, attempts + 1):
        try:
            async with sem:
                comp = await provider.complete(
                    PROBE_SYSTEM, [ChatMessage("user", prompt)], seed=seed
                )
            return comp.text
        except TransientProviderError as e:
            if attempt == attempts:
                raise
            await asyncio.sleep(
                max(default_retry_delay(attempt), min(e.retry_after or 0.0, MAX_WAIT_S))
            )
    raise AssertionError("unreachable")


def stimulus_text(task: Task, variant: str = "original") -> str:
    v = task.stimuli["variants"][variant]
    if task.kind == "framing":
        cond = task.contrast[0]
        opts = v["conditions"][cond]
        return (
            f"{v['preamble']} {opts['certain'].format(label='A')} {opts['risky'].format(label='B')}"
        )
    if task.kind == "obedience":
        return str(v["intro"])
    if task.kind == "vignette":
        c = v["conditions"][task.contrast[0]]
        return f"{c.get('scenario') or ''} {c['question']}".strip()
    return " ".join(str(x) for x in v["conditions"].values())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


async def _recognition(task: Task, call: Any, seed: int) -> dict[str, Any]:
    text = stimulus_text(task)
    opening = text[: max(200, len(text) // 3)]
    prompt = (
        f'Here is the opening of a scenario used in a published experiment:\n\n"{opening}..."\n\n'
        "Do you recognize it? If so, name the study (authors and year) and describe what most "
        "human participants did. If you do not recognize it, say so."
    )
    reply = await call(prompt, seed)
    hits = [k for k in recognition_keywords(task) if k in reply.lower()]
    return {"probe": "recognition", "score": float(bool(hits)), "hits": hits, "response": reply}


async def _completion(task: Task, call: Any, variant: str, seed: int) -> dict[str, Any]:
    text = stimulus_text(task, variant)
    cut = len(text) // 2
    prefix, truth = text[:cut], text[cut:]
    prompt = (
        "Continue the following text exactly as it appears in its original source. Output only "
        f"the continuation.\n\n{prefix}"
    )
    reply = await call(prompt, seed)
    guess = reply[: len(truth) + 50]
    return {
        "probe": f"completion_{variant}",
        "score": similarity(guess, truth),
        "response": reply,
    }


async def run_probes(
    cfg: StudyConfig, tasks: dict[str, Task], providers: dict[str, Provider], n: int, out: Path
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for t in tasks.values():
        recognition_keywords(t)
    sem = asyncio.Semaphore(cfg.runner.concurrency)

    async def one(model_id: str, task: Task, i: int) -> None:
        p = providers[model_id]

        async def call(prompt: str, seed: int) -> str:
            return await _call(p, prompt, seed, sem, cfg.runner.max_attempts)

        probes = [("recognition", _recognition(task, call, seed=i))] + [
            (f"completion_{v}", _completion(task, call, v, seed=i)) for v in task.variants()
        ]
        for name, probe in probes:
            try:
                r = await probe
            except Exception as e:
                r = {"probe": name, "score": None, "error": repr(e)}
            records.append({"model_id": model_id, "task_id": task.id, "sample": i, **r})

    try:
        await asyncio.gather(
            *(one(m.id, t, i) for m in cfg.models for t in tasks.values() for i in range(n))
        )
    finally:
        for p in providers.values():
            await p.aclose()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "probes.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(records)
    summary = (
        df.groupby(["model_id", "task_id", "probe"])["score"]
        .agg(["mean", "count", lambda s: int(s.isna().sum())])
        .rename(columns={"<lambda_0>": "failed"})
        .reset_index()
    )
    cutoffs = []
    for m in cfg.models:
        for t in tasks.values():
            first = t.spec.materials_first_public
            cutoffs.append(
                {
                    "model_id": m.id,
                    "task_id": t.id,
                    "training_cutoff": m.training_cutoff,
                    "materials_first_public": first,
                    "predates_materials": (m.training_cutoff < first)
                    if (m.training_cutoff and first)
                    else None,
                }
            )
    pd.DataFrame(cutoffs).to_csv(out / "cutoff_check.csv", index=False)
    summary.to_csv(out / "probe_summary.csv", index=False)
    return summary
