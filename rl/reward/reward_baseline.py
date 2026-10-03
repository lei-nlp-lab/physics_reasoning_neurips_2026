"""
reward_baseline.py — 4 GRPO baselines that all use GPT-4o-mini as the judge.

Selected via env var BASELINE_REWARD_KIND in
  {rlvr, prm_only, prm_gt, prm_rubric}

  rlvr        Extract [ans]…[ans] from the response. Ask 4o-mini whether the
              extracted answer matches the ground-truth answer list. Reward is
              the fraction of GT items judged correct (0..1).
  prm_only    Pass only the response to 4o-mini. PRM-style: judge reasoning
              quality with no reference. 0..1.
  prm_gt      Pass response + ground-truth answer to 4o-mini. 0..1.
  prm_rubric  Pass response + the per-problem rubric loaded from
              BASELINE_RUBRIC_DIR/<prob_id>.json to 4o-mini. If a problem has
              no rubric file, fall back to prm_gt. 0..1.

verl reward interface:
  compute_score(data_source, solution_str, ground_truth, extra_info) -> float
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

try:
    import fcntl
except ImportError:
    fcntl = None

ROOT = Path(__file__).resolve().parent

ALLOWED_KINDS = {"rlvr", "prm_only", "prm_gt", "prm_rubric"}
KIND = os.environ.get("BASELINE_REWARD_KIND", "rlvr").strip().lower()
if KIND not in ALLOWED_KINDS:
    raise RuntimeError(
        f"BASELINE_REWARD_KIND must be one of {sorted(ALLOWED_KINDS)}, got {KIND!r}"
    )

RUBRIC_DIR = Path(os.environ.get(
    "BASELINE_RUBRIC_DIR",
    str(ROOT.parent.parent / "data" / "rubrics" / "phyreason"),
))
LOG_PATH = os.environ.get("BASELINE_LOG_PATH", "").strip()
MIN_REWARD = float(os.environ.get("BASELINE_MIN_REWARD", "0.0"))
MAX_RETRIES = int(os.environ.get("BASELINE_MAX_RETRIES", "3"))
TIMEOUT_S = float(os.environ.get("BASELINE_TIMEOUT_S", "60"))
API_CONCURRENCY = int(os.environ.get("BASELINE_API_CONCURRENCY", "512"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") \
    or "https://openrouter.ai/api/v1"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini-2024-07-18") \
    or "openai/gpt-4o-mini-2024-07-18"


_client_lock = threading.Lock()
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            # httpx default pool is 100 connections; bump to API_CONCURRENCY so
            # verl's parallel compute_score callers don't queue at the http layer.
            limits = httpx.Limits(
                max_connections=API_CONCURRENCY,
                max_keepalive_connections=API_CONCURRENCY,
            )
            http_client = httpx.Client(limits=limits, timeout=TIMEOUT_S)
            _client = OpenAI(
                api_key=OPENAI_API_KEY or "dummy",
                base_url=OPENAI_BASE_URL,
                timeout=TIMEOUT_S,
                max_retries=0,
                http_client=http_client,
            )
    return _client


def _log(payload: dict) -> None:
    if not LOG_PATH:
        return
    p = Path(LOG_PATH)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _call_api(messages: list[dict]) -> str | None:
    last_exc: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _get_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # network, rate limit, json mode unsupported, ...
            last_exc = exc
            time.sleep(min(0.5 * attempt, 2.0))
    print(f"[reward_baseline] api failed after {MAX_RETRIES}: {last_exc}",
          file=sys.stderr, flush=True)
    return None


def _parse_score(text: str | None) -> float | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r'(?<![\d.])([01](?:\.\d+)?)(?![\d.])', text)
        return float(m.group(1)) if m else None
    if isinstance(obj, dict):
        for k in ("score", "reward", "value", "result"):
            if k in obj:
                try:
                    return float(obj[k])
                except (TypeError, ValueError):
                    return None
    if isinstance(obj, (int, float)):
        return float(obj)
    return None


def _extract_ans_block(text: str) -> str:
    m = re.findall(r'\[ans\](.*?)\[ans\]', text, flags=re.DOTALL | re.IGNORECASE)
    return m[-1].strip() if m else ""


def _gt_answers(ground_truth: Any) -> list:
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else (ground_truth or {})
    except Exception:
        gt = {}
    return gt.get("answer", []) if isinstance(gt, dict) else []


def _gt_reasoning(ground_truth: Any) -> str:
    """Optional reference reasoning bundled into ground_truth
    (flattened from the dataset's reference explanation steps).
    Returns empty string if absent."""
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else (ground_truth or {})
    except Exception:
        gt = {}
    if not isinstance(gt, dict):
        return ""
    val = gt.get("reasoning") or ""
    return str(val)


def _msgs_rlvr(solution: str, ground_truth: Any, extra: dict) -> list[dict]:
    pred = _extract_ans_block(solution)
    gt = _gt_answers(ground_truth)
    user = (
        "You are an automatic grader for physics final-answer matching.\n"
        "There may be multiple ground-truth answers (one per sub-question).\n"
        "Compare the student's [ans]…[ans] block to the ground truth list. "
        "Allow equivalent algebraic forms, equivalent units, and small numeric "
        "rounding. Order does not strictly matter, but each GT answer must be "
        "matched at most once.\n\n"
        f"Ground-truth answer list (JSON): {json.dumps(gt, ensure_ascii=False)}\n"
        f"Student [ans] block:\n{pred}\n\n"
        'Return JSON exactly: {"score": x} where x is the fraction of GT '
        "answers correctly produced, in [0, 1]."
    )
    return [{"role": "user", "content": user}]


def _msgs_prm_only(solution: str, ground_truth: Any, extra: dict) -> list[dict]:
    user = (
        "You are a process-reward physics grader. You have NO access to the "
        "ground-truth answer. Score the student's solution based ONLY on the "
        "quality of reasoning, clarity, correct application of physical laws, "
        "and internal consistency.\n\n"
        f"Student solution:\n{solution}\n\n"
        'Return JSON exactly: {"score": x} where x is in [0, 1]. '
        "0 = nonsense / no physics; 1 = high-quality, internally consistent."
    )
    return [{"role": "user", "content": user}]


def _msgs_prm_gt(solution: str, ground_truth: Any, extra: dict) -> list[dict]:
    gt = _gt_answers(ground_truth)
    reasoning = _gt_reasoning(ground_truth)
    reasoning_block = (
        f"Reference reasoning steps:\n{reasoning}\n\n"
        if reasoning else ""
    )
    user = (
        "You are a physics grader. Use the reference solution (final answer "
        "AND reasoning steps when available) to assess BOTH the correctness "
        "of the student's final answer and the quality / soundness of their "
        "reasoning. Reward sound reasoning even if the student's path differs "
        "from the reference, as long as it reaches the right answer.\n\n"
        f"Reference (correct) answer(s): {json.dumps(gt, ensure_ascii=False)}\n\n"
        f"{reasoning_block}"
        f"Student solution:\n{solution}\n\n"
        'Return JSON exactly: {"score": x} where x is in [0, 1].'
    )
    return [{"role": "user", "content": user}]


_rubric_cache: dict[str, list[dict] | None] = {}
_rubric_cache_lock = threading.Lock()


def _load_rubric(prob_id: str) -> list[dict] | None:
    if not prob_id:
        return None
    with _rubric_cache_lock:
        if prob_id in _rubric_cache:
            return _rubric_cache[prob_id]
    p = RUBRIC_DIR / f"{prob_id}.json"
    data: list[dict] | None
    if not p.exists():
        data = None
    else:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            data = obj if isinstance(obj, list) else None
        except Exception:
            data = None
    with _rubric_cache_lock:
        _rubric_cache[prob_id] = data
    return data


def _msgs_prm_rubric(solution: str, ground_truth: Any, extra: dict) -> list[dict] | None:
    prob_id = str((extra or {}).get("prob_id") or "")
    rubric = _load_rubric(prob_id)
    if not rubric:
        # No rubric for this problem: fall back to the reference-answer /
        # reference-reasoning grader instead of returning a guaranteed zero.
        return _msgs_prm_gt(solution, ground_truth, extra)
    rubric_lines = []
    total_weight = 0
    for item in rubric:
        try:
            w = int(item.get("weight", 0))
        except (TypeError, ValueError):
            w = 0
        total_weight += max(0, w)
        rubric_lines.append(
            f"- (weight={item.get('weight', 0)}) {item.get('title', '').strip()}: "
            f"{item.get('description', '').strip()}"
        )
    rubric_text = "\n".join(rubric_lines)
    total_weight = total_weight or 1
    user = (
        "You are a physics grader. Grade the student's solution against the "
        "rubric below. For each criterion, decide how fully the student "
        "satisfies it (0–1). Then aggregate to a SINGLE weighted score in "
        "[0, 1] using the weights provided.\n\n"
        f"Rubric:\n{rubric_text}\n"
        f"(Total possible weight: {total_weight})\n\n"
        f"Student solution:\n{solution}\n\n"
        'Return JSON exactly: {"score": x} where x is the weighted total in [0, 1].'
    )
    return [{"role": "user", "content": user}]


_BUILDERS = {
    "rlvr": _msgs_rlvr,
    "prm_only": _msgs_prm_only,
    "prm_gt": _msgs_prm_gt,
    "prm_rubric": _msgs_prm_rubric,
}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    del data_source
    extra_info = extra_info or {}
    t0 = time.perf_counter()
    payload: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": KIND,
        "model": OPENAI_MODEL,
        "prob_id": str(extra_info.get("prob_id") or ""),
        "pid": os.getpid(),
    }
    try:
        msgs = _BUILDERS[KIND](solution_str, ground_truth, extra_info)
        if msgs is None:
            payload.update({"score": MIN_REWARD, "reason": "no_input_built"})
            _log(payload)
            return MIN_REWARD

        text = _call_api(msgs)
        score = _parse_score(text)
        if score is None:
            payload.update({"score": MIN_REWARD, "reason": "score_parse_failed",
                            "raw": (text or "")[:500]})
            _log(payload)
            return MIN_REWARD

        score = float(max(MIN_REWARD, min(1.0, score)))
        payload.update({
            "score": score,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        })
        _log(payload)
        return score
    except Exception as exc:
        payload.update({
            "score": MIN_REWARD, "reason": "exception", "error": str(exc),
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        })
        _log(payload)
        return MIN_REWARD
