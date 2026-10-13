"""
reward_pmf_diverge.py — PMF-based reward for the [diverge] section.

Scores the physical modelling quality of the model's diverge phase using
Physical Modelling Fidelity (PMF):
  - QEL (Quantity-Equation-Law) F1
  - VCL (Variable Coupling Link) F1
  - AEB (Assumption-Equation Binding) F1

GT motifs can be precomputed offline and stored in the parquet ground truth as
`gt_motifs`. When they are absent, this reward falls back to online extraction.
Fallback caching is keyed by `prob_id`, or by `sha256(gt_text)` when `prob_id`
is missing, so empty-string problem ids no longer alias unrelated problems.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

# Add project root to sys.path so PMF module can be imported
ROOT = Path(__file__).resolve().parent        # rl/reward/
PROJECT_ROOT = ROOT.parent                     # rl/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

from PMF.extractor import extract_motifs_async
from PMF.motifs import MotifSet
from PMF.scorer import compute_pmf_async

try:
    import fcntl
except ImportError:
    fcntl = None

# ============================================================
# Config (all overridable via env vars)
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_DEFAULT_PMF_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
_DEFAULT_PMF_MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini-2024-07-18")
PMF_BASE_URL = os.environ.get("PMF_BASE_URL", _DEFAULT_PMF_BASE_URL).strip() or _DEFAULT_PMF_BASE_URL
PMF_MODEL = os.environ.get("PMF_MODEL", _DEFAULT_PMF_MODEL).strip() or _DEFAULT_PMF_MODEL


PMF_API_KEY = os.environ.get("PMF_API_KEY", OPENAI_API_KEY).strip()
PMF_MAX_RETRIES = int(os.environ.get("PMF_MAX_RETRIES", "1"))
PMF_TIMEOUT_S = float(os.environ.get("PMF_TIMEOUT_S", "30"))
# Optional fallback: when PMF_FALLBACK_BASE_URL is set and the primary judge times out
# or errors, the reward client retries once against that endpoint (unset by default).
PMF_FALLBACK_BASE_URL = os.environ.get("PMF_FALLBACK_BASE_URL", "").strip()
PMF_FALLBACK_MODEL = os.environ.get("PMF_FALLBACK_MODEL", "").strip()
PMF_FALLBACK_API_KEY = os.environ.get("PMF_FALLBACK_API_KEY", "dummy-local").strip()
PMF_FALLBACK_TIMEOUT_S = float(os.environ.get("PMF_FALLBACK_TIMEOUT_S", "60"))
# Motif weights default to 1/3 each; renormalized over active motif types in compute_score.
PMF_QEL_WEIGHT = float(os.environ.get("PMF_QEL_WEIGHT", str(1 / 3)))
PMF_VCL_WEIGHT = float(os.environ.get("PMF_VCL_WEIGHT", str(1 / 3)))
PMF_AEB_WEIGHT = float(os.environ.get("PMF_AEB_WEIGHT", str(1 / 3)))
PMF_MIN_REWARD = float(os.environ.get("PMF_MIN_REWARD", "0.0"))
DEBUG = os.environ.get("PMF_DEBUG", "0") == "1"
LOG_PATH = os.environ.get("PMF_LOG_PATH", "").strip()
PMF_STATS_INTERVAL = int(os.environ.get("PMF_STATS_INTERVAL", "50"))
PMF_USE_SEMANTIC_JUDGE = os.environ.get("PMF_USE_SEMANTIC_JUDGE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _resolve_pmf_api_key() -> str:
    if PMF_API_KEY:
        return PMF_API_KEY
    if PMF_BASE_URL.rstrip("/") != _DEFAULT_PMF_BASE_URL:
        return "pmf-local"
    raise ValueError("PMF_API_KEY/OPENAI_API_KEY is required for PMF reward.")


def _make_pmf_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_resolve_pmf_api_key(),
        base_url=PMF_BASE_URL,
        timeout=PMF_TIMEOUT_S,
        max_retries=PMF_MAX_RETRIES,
    )


def _make_pmf_fallback_client() -> AsyncOpenAI | None:
    """Build the fallback judge client, or None if no fallback is configured."""
    if not PMF_FALLBACK_BASE_URL:
        return None
    return AsyncOpenAI(
        api_key=PMF_FALLBACK_API_KEY or "dummy-local",
        base_url=PMF_FALLBACK_BASE_URL,
        timeout=PMF_FALLBACK_TIMEOUT_S,
        max_retries=1,
    )


def _is_fallback_error(exc: BaseException) -> bool:
    """Errors worth retrying against the local fallback judge."""
    import asyncio as _asyncio

    import openai as _openai
    return isinstance(exc, (
        _openai.APITimeoutError,
        _openai.APIConnectionError,
        _openai.InternalServerError,
        _openai.RateLimitError,
        _asyncio.TimeoutError,
    ))


# ============================================================
# Singleton PMFEvaluator
# ============================================================
class PMFConfiguredEvaluator:
    def __init__(
        self,
        *,
        model: str,
        use_semantic_judge: bool,
    ) -> None:
        self.model = model
        self.use_semantic_judge = use_semantic_judge

    def evaluate_motifs(self, pred: MotifSet, gt: MotifSet) -> dict:
        return asyncio.run(self._evaluate_motifs_async(pred, gt))

    async def _evaluate_motifs_async(self, pred: MotifSet, gt: MotifSet) -> dict:
        async def _run(client: AsyncOpenAI, model: str) -> dict:
            judge_client = client if self.use_semantic_judge else None
            return await compute_pmf_async(
                pred=pred,
                gt=gt,
                judge_client=judge_client,
                judge_model=model,
            )

        client = _make_pmf_client()
        try:
            try:
                return await _run(client, self.model)
            except Exception as exc:
                if not _is_fallback_error(exc):
                    raise
                fb = _make_pmf_fallback_client()
                if fb is None:
                    raise
                _debug(
                    f"primary evaluate failed ({type(exc).__name__}: {exc}); "
                    f"falling back to local judge at {PMF_FALLBACK_BASE_URL}"
                )
                try:
                    return await _run(fb, PMF_FALLBACK_MODEL or self.model)
                finally:
                    await fb.close()
        finally:
            await client.close()


def _extract_motifs_with_config(text: str):
    async def _extract(client: AsyncOpenAI, model: str):
        return await extract_motifs_async(
            client,
            text,
            model=model,
            max_retries=PMF_MAX_RETRIES,
        )

    async def _run():
        client = _make_pmf_client()
        try:
            try:
                return await _extract(client, PMF_MODEL)
            except Exception as exc:
                if not _is_fallback_error(exc):
                    raise
                fb = _make_pmf_fallback_client()
                if fb is None:
                    raise
                _debug(
                    f"primary extract failed ({type(exc).__name__}: {exc}); "
                    f"falling back to local judge at {PMF_FALLBACK_BASE_URL}"
                )
                try:
                    return await _extract(fb, PMF_FALLBACK_MODEL or PMF_MODEL)
                finally:
                    await fb.close()
        finally:
            await client.close()

    return asyncio.run(_run())


_evaluator: "PMFConfiguredEvaluator | None" = None
_evaluator_lock = threading.Lock()

# Parallelizing GT + Pred extraction: each call to _extract_motifs_with_config
# does its own asyncio.run, so we can run them concurrently via threads.
_extraction_executor: "concurrent.futures.ThreadPoolExecutor | None" = None
_extraction_executor_lock = threading.Lock()


def _get_extraction_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _extraction_executor
    if _extraction_executor is not None:
        return _extraction_executor
    with _extraction_executor_lock:
        if _extraction_executor is None:
            _extraction_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=int(os.environ.get("PMF_EXTRACT_WORKERS", "2")),
                thread_name_prefix="pmf_extract",
            )
    return _extraction_executor


def _get_evaluator() -> PMFConfiguredEvaluator:
    global _evaluator
    if _evaluator is not None:
        return _evaluator
    with _evaluator_lock:
        if _evaluator is None:
            _evaluator = PMFConfiguredEvaluator(
                model=PMF_MODEL,
                use_semantic_judge=PMF_USE_SEMANTIC_JUDGE,
            )
    return _evaluator


# ============================================================
# GT motif cache
# ============================================================
_gt_motif_cache: dict[str, MotifSet] = {}
_gt_cache_lock = threading.Lock()

_stats_lock = threading.Lock()
_stats: dict[str, Any] = {
    "total_calls": 0,
    "reward_exceptions": 0,
    "empty_gt_text": 0,
    "empty_diverge": 0,
    "gt_precomputed": 0,
    "gt_cache_hits": 0,
    "gt_api_calls": 0,
    "pred_api_calls": 0,
    "gt_status_counts": {},
    "pred_status_counts": {},
    "pmf_eval_status_counts": {},
}


def _bump(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def _bump_stat(key: str, amount: int = 1) -> None:
    with _stats_lock:
        _stats[key] = int(_stats.get(key, 0)) + amount


def _record_gt_meta(meta: dict[str, Any]) -> None:
    with _stats_lock:
        source = meta.get("source", "unknown")
        if source == "precomputed":
            _stats["gt_precomputed"] += 1
        elif source == "cache":
            _stats["gt_cache_hits"] += 1
        elif source == "api":
            _stats["gt_api_calls"] += 1
        _bump(_stats["gt_status_counts"], str(meta.get("status", "unknown")))


def _record_pred_meta(meta: dict[str, Any]) -> None:
    with _stats_lock:
        _stats["pred_api_calls"] += 1
        _bump(_stats["pred_status_counts"], str(meta.get("status", "unknown")))


def _record_pmf_eval_meta(meta: dict[str, Any]) -> None:
    with _stats_lock:
        _bump(_stats["pmf_eval_status_counts"], str(meta.get("status", "unknown")))


def _build_gt_cache_key(prob_id: str, gt_text: str) -> str:
    if prob_id:
        return f"prob_id:{prob_id}"
    digest = hashlib.sha256(gt_text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _classify_runtime_error(exc: Exception) -> tuple[str, str]:
    exc_name = type(exc).__name__
    lowered = f"{exc_name}: {exc}".lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout", exc_name
    return "error", exc_name


def _parse_precomputed_gt_motifs(raw_gt_motifs: Any) -> MotifSet | None:
    if raw_gt_motifs is None:
        return None
    parsed = raw_gt_motifs
    if isinstance(raw_gt_motifs, str):
        if not raw_gt_motifs.strip():
            return None
        parsed = json.loads(raw_gt_motifs)
    if not isinstance(parsed, dict):
        raise ValueError("gt_motifs must be a JSON object")
    return MotifSet.from_dict(parsed)


def _get_gt_motifs(
    prob_id: str,
    gt_text: str,
    raw_gt_motifs: Any = None,
) -> tuple[MotifSet, dict[str, Any]]:
    """
    Resolve GT motifs from parquet-precomputed data, cache, or online extraction.

    Returns:
        (motifs, meta)
    """
    cache_key = _build_gt_cache_key(prob_id, gt_text)
    precomputed_error: str | None = None

    try:
        precomputed = _parse_precomputed_gt_motifs(raw_gt_motifs)
    except Exception as exc:
        precomputed = None
        precomputed_error = str(exc)

    if precomputed is not None:
        with _gt_cache_lock:
            cache_hit = cache_key in _gt_motif_cache
            if not cache_hit:
                _gt_motif_cache[cache_key] = precomputed
            motifs = _gt_motif_cache[cache_key]
        return motifs, {
            "status": "ok",
            "source": "precomputed",
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "elapsed_ms": 0,
            "attempts": 0,
            "error": None,
            "error_type": None,
        }

    with _gt_cache_lock:
        if cache_key in _gt_motif_cache:
            return _gt_motif_cache[cache_key], {
                "status": "ok",
                "source": "cache",
                "cache_key": cache_key,
                "cache_hit": True,
                "elapsed_ms": 0,
                "attempts": 0,
                "error": precomputed_error,
                "error_type": "invalid_precomputed" if precomputed_error else None,
            }

    if not gt_text.strip():
        return MotifSet.empty(), {
            "status": "missing_gt_text",
            "source": "none",
            "cache_key": cache_key,
            "cache_hit": False,
            "elapsed_ms": 0,
            "attempts": 0,
            "error": precomputed_error or "GT text is empty and gt_motifs is unavailable.",
            "error_type": "invalid_precomputed" if precomputed_error else "missing_gt_text",
        }

    start_time = time.perf_counter()
    _debug(f"Extracting GT motifs for cache_key={cache_key!r}")
    result = _extract_motifs_with_config(gt_text)
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    if result.status == "ok":
        with _gt_cache_lock:
            _gt_motif_cache[cache_key] = result.motifs

    return result.motifs, {
        "status": result.status,
        "source": "api",
        "cache_key": cache_key,
        "cache_hit": False,
        "elapsed_ms": elapsed_ms,
        "attempts": result.attempts,
        "error": result.error or precomputed_error,
        "error_type": result.error_type or ("invalid_precomputed" if precomputed_error else None),
    }


# ============================================================
# Text helpers
# ============================================================

def extract_diverge_section(solution_str: str) -> str:
    """Extract the physics modeling section.

    Supports both the legacy [diverge]…[converge] format and the new
    [modeling]…[sloving] format used by the current training prompts.
    """
    m = re.search(
        r"\[(?:diverge|modeling)\](.*?)(?=\[(?:converge|sloving|pseudocode|ans)\]|</thinking>|\Z)",
        solution_str,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def build_gt_text(physics_mapping: str, key_constraints: str) -> str:
    """Combine ground-truth fields into a single PMF-ready reference text."""
    parts = []
    if physics_mapping and physics_mapping.strip():
        parts.append(f"Physics Mapping:\n{physics_mapping.strip()}")
    if key_constraints and key_constraints.strip():
        parts.append(f"Key Physics Constraints:\n{key_constraints.strip()}")
    return "\n\n".join(parts)


def build_fallback_gt_text(gt: dict[str, Any]) -> tuple[str, str]:
    """Build PMF reference text for older parquet files missing mapping fields."""
    reference_modeling = str(gt.get("reference_modeling") or "").strip()
    reference_sloving = str(gt.get("reference_sloving") or "").strip()
    reference_response = str(gt.get("reference_response") or "").strip()
    reference_pseudocode = str(gt.get("reference_pseudocode") or "").strip()
    # data/rl/phyreason_baseline only carries `reasoning` (flattened reference
    # explanation steps); use it as last resort
    # so PMF doesn't degenerate to empty_gt_text on the no-SFT ablations.
    reasoning = str(gt.get("reasoning") or "").strip()

    if reference_modeling:
        return f"Reference Modeling:\n{reference_modeling}", "reference_modeling"
    if reference_sloving:
        return f"Reference Solving Trace:\n{reference_sloving}", "reference_sloving"
    if reference_pseudocode:
        return f"Reference Pseudocode:\n{reference_pseudocode}", "reference_pseudocode"
    if reference_response:
        return f"Reference Response:\n{reference_response}", "reference_response"
    if reasoning:
        return f"Reference Reasoning:\n{reasoning}", "reasoning"
    return "", ""


# ============================================================
# Logging
# ============================================================

def _debug(msg: str) -> None:
    if DEBUG:
        print(f"[pmf_reward] {msg}", flush=True)


def _append_log(payload: dict) -> None:
    if not LOG_PATH:
        return
    path = Path(LOG_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _maybe_emit_stats_summary(force: bool = False) -> None:
    if not LOG_PATH:
        return
    if PMF_STATS_INTERVAL <= 0 and not force:
        return

    with _stats_lock:
        total_calls = int(_stats.get("total_calls", 0))
        if not force and total_calls <= 0:
            return
        if not force and total_calls % PMF_STATS_INTERVAL != 0:
            return
        snapshot = copy.deepcopy(_stats)

    gt_api_calls = int(snapshot.get("gt_api_calls", 0))
    pred_api_calls = int(snapshot.get("pred_api_calls", 0))
    pmf_eval_status_counts = snapshot.get("pmf_eval_status_counts", {})
    pmf_eval_total = sum(int(v) for v in pmf_eval_status_counts.values())

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "pmf_metrics_summary",
        "pid": os.getpid(),
        **snapshot,
        "gt_api_success_rate": (
            snapshot.get("gt_status_counts", {}).get("ok", 0) / gt_api_calls
            if gt_api_calls
            else None
        ),
        "pred_success_rate": (
            snapshot.get("pred_status_counts", {}).get("ok", 0) / pred_api_calls
            if pred_api_calls
            else None
        ),
        "pmf_eval_success_rate": (
            pmf_eval_status_counts.get("ok", 0) / pmf_eval_total
            if pmf_eval_total
            else None
        ),
    }
    _append_log(summary)


def _log_reward_event(payload: dict[str, Any]) -> None:
    if LOG_PATH:
        _append_log(payload)
    _maybe_emit_stats_summary()


# ============================================================
# Main entry point (verl reward interface)
# ============================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """
    Compute PMF reward for the [diverge] section of the model's response.

    Args:
        data_source:   Dataset identifier (unused, kept for interface compatibility)
        solution_str:  Full model response including <thinking>...</thinking>
        ground_truth:  JSON string from parquet; must contain
                       'physics_mapping', 'key_constraints', 'prob_id', and
                       optionally precomputed 'gt_motifs'
        extra_info:    Optional dict with 'prob_id' fallback

    Returns:
        Float reward in [PMF_MIN_REWARD, 1.0]
    """
    extra_info = extra_info or {}

    call_start = time.perf_counter()
    with _stats_lock:
        _stats["total_calls"] += 1

    try:
        gt = (
            json.loads(ground_truth)
            if isinstance(ground_truth, str)
            else (ground_truth or {})
        )
    except Exception:
        gt = {}

    prob_id = str(
        gt.get("prob_id") or extra_info.get("prob_id") or ""
    ).strip()

    # ---- Build GT text / load offline GT motifs ----
    physics_mapping  = gt.get("physics_mapping", "")
    key_constraints  = gt.get("key_constraints", "")
    gt_text, gt_text_source = build_fallback_gt_text(gt)
    if not gt_text.strip():
        gt_text = build_gt_text(physics_mapping, key_constraints)
        gt_text_source = "physics_mapping_key_constraints" if gt_text.strip() else ""
    raw_gt_motifs = gt.get("gt_motifs")

    log_payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "pmf_reward",
        "pid": os.getpid(),
        "prob_id": prob_id,
        "gt_text_source": gt_text_source,
    }

    if not gt_text.strip() and raw_gt_motifs is None:
        _bump_stat("empty_gt_text")
        _debug(f"prob_id={prob_id!r}: empty GT text → min reward")
        log_payload.update(
            {
                "score": PMF_MIN_REWARD,
                "reason": "empty_gt_text",
                "call_elapsed_ms": int((time.perf_counter() - call_start) * 1000),
            }
        )
        _log_reward_event(log_payload)
        return PMF_MIN_REWARD

    # ---- Extract [diverge] section from model output ----
    pred_text = extract_diverge_section(solution_str)
    if not pred_text.strip():
        _bump_stat("empty_diverge")
        _debug(f"prob_id={prob_id!r}: empty [diverge] section → min reward")
        log_payload.update(
            {
                "score": PMF_MIN_REWARD,
                "reason": "empty_diverge",
                "call_elapsed_ms": int((time.perf_counter() - call_start) * 1000),
            }
        )
        _log_reward_event(log_payload)
        return PMF_MIN_REWARD

    try:
        # Kick off pred extraction in a background thread so it runs concurrently
        # with GT resolution (which may itself need an API call on cache miss).
        # When GT is cached/precomputed, this costs only a thread hop; when GT
        # also needs an API call, the two run in parallel and total wall time
        # is roughly max(gt_api, pred_api) instead of their sum.
        pred_extract_start = time.perf_counter()
        pred_future = _get_extraction_executor().submit(
            _extract_motifs_with_config, pred_text
        )

        # GT motifs: prefer offline-precomputed motifs, then cache, then online fallback.
        gt_motifs, gt_meta = _get_gt_motifs(prob_id, gt_text, raw_gt_motifs)
        _record_gt_meta(gt_meta)
        log_payload.update(
            {
                "gt_source": gt_meta.get("source"),
                "gt_status": gt_meta.get("status"),
                "gt_error": gt_meta.get("error"),
                "gt_error_type": gt_meta.get("error_type"),
                "gt_attempts": gt_meta.get("attempts"),
                "gt_elapsed_ms": gt_meta.get("elapsed_ms"),
                "gt_cache_key": gt_meta.get("cache_key"),
                "gt_cache_hit": gt_meta.get("cache_hit"),
                "gt_qel_count": len(gt_motifs.qel),
                "gt_vcl_count": len(gt_motifs.vcl),
                "gt_aeb_count": len(gt_motifs.aeb),
            }
        )

        if gt_meta.get("status") != "ok":
            _debug(
                f"prob_id={prob_id!r}: GT motif resolution failed "
                f"status={gt_meta.get('status')} error={gt_meta.get('error')}"
            )
            # GT failed; drain pred_future so the worker thread doesn't leak.
            # Any exception it raises is irrelevant to the min-reward return.
            try:
                pred_future.result()
            except Exception:
                pass
            log_payload.update(
                {
                    "score": PMF_MIN_REWARD,
                    "reason": "gt_motif_resolution_failed",
                    "call_elapsed_ms": int((time.perf_counter() - call_start) * 1000),
                }
            )
            _log_reward_event(log_payload)
            return PMF_MIN_REWARD

        # Pred motifs: already submitted above, now wait for the result.
        # pred_extract_elapsed_ms is wall-clock time from submit to ready, which
        # equals the max(pending, api_time) of the overlap with GT.
        pred_result = pred_future.result()
        pred_extract_elapsed_ms = int((time.perf_counter() - pred_extract_start) * 1000)
        pred_motifs = pred_result.motifs
        pred_meta = {
            "status": pred_result.status,
            "error": pred_result.error,
            "error_type": pred_result.error_type,
            "attempts": pred_result.attempts,
            "elapsed_ms": pred_extract_elapsed_ms,
        }
        _record_pred_meta(pred_meta)
        log_payload.update(
            {
                "pred_status": pred_result.status,
                "pred_error": pred_result.error,
                "pred_error_type": pred_result.error_type,
                "pred_attempts": pred_result.attempts,
                "pred_elapsed_ms": pred_extract_elapsed_ms,
                "pred_qel_count": len(pred_motifs.qel),
                "pred_vcl_count": len(pred_motifs.vcl),
                "pred_aeb_count": len(pred_motifs.aeb),
            }
        )

        _debug(
            f"prob_id={prob_id!r}: pred "
            f"qel={len(pred_motifs.qel)} vcl={len(pred_motifs.vcl)} "
            f"aeb={len(pred_motifs.aeb)} status={pred_result.status}"
        )

        # Score via PMFEvaluator (handles semantic judge + exact matching)
        evaluator = _get_evaluator()
        pmf_eval_start = time.perf_counter()
        try:
            result = evaluator.evaluate_motifs(pred_motifs, gt_motifs)
            pmf_eval_meta = {
                "status": "ok",
                "error": None,
                "error_type": None,
                "elapsed_ms": int((time.perf_counter() - pmf_eval_start) * 1000),
            }
        except Exception as exc:
            pmf_status, pmf_error_type = _classify_runtime_error(exc)
            pmf_eval_meta = {
                "status": pmf_status,
                "error": str(exc),
                "error_type": pmf_error_type,
                "elapsed_ms": int((time.perf_counter() - pmf_eval_start) * 1000),
            }
            _record_pmf_eval_meta(pmf_eval_meta)
            log_payload.update(
                {
                    "pmf_eval_status": pmf_eval_meta["status"],
                    "pmf_eval_error": pmf_eval_meta["error"],
                    "pmf_eval_error_type": pmf_eval_meta["error_type"],
                    "pmf_eval_elapsed_ms": pmf_eval_meta["elapsed_ms"],
                    "score": PMF_MIN_REWARD,
                    "reason": "pmf_eval_failed",
                    "call_elapsed_ms": int((time.perf_counter() - call_start) * 1000),
                }
            )
            _log_reward_event(log_payload)
            return PMF_MIN_REWARD
        _record_pmf_eval_meta(pmf_eval_meta)

        qel_f1 = float(result["qel"]["f1"])
        vcl_f1 = float(result["vcl"]["f1"])
        aeb_f1 = float(result["aeb"]["f1"])
        active = result.get("active_motifs", [])

        # Weighted sum over active motif types (re-normalized if some are absent)
        w_qel = PMF_QEL_WEIGHT if "qel" in active else 0.0
        w_vcl = PMF_VCL_WEIGHT if "vcl" in active else 0.0
        w_aeb = PMF_AEB_WEIGHT if "aeb" in active else 0.0
        w_total = w_qel + w_vcl + w_aeb

        if w_total > 0:
            score = (w_qel * qel_f1 + w_vcl * vcl_f1 + w_aeb * aeb_f1) / w_total
        else:
            # Fallback: use PMFEvaluator's own adaptive-weight score
            score = float(result["pmf"])

        score = float(max(PMF_MIN_REWARD, min(1.0, score)))

        _debug(
            f"prob_id={prob_id!r}: qel_f1={qel_f1:.3f} vcl_f1={vcl_f1:.3f} "
            f"aeb_f1={aeb_f1:.3f} active={active} → score={score:.3f}"
        )

        if LOG_PATH:
            log_payload.update(
                {
                    "qel_f1": qel_f1,
                    "vcl_f1": vcl_f1,
                    "aeb_f1": aeb_f1,
                    "pmf": float(result["pmf"]),
                    "score": score,
                    "active_motifs": active,
                    "pmf_eval_status": pmf_eval_meta["status"],
                    "pmf_eval_error": pmf_eval_meta["error"],
                    "pmf_eval_error_type": pmf_eval_meta["error_type"],
                    "pmf_eval_elapsed_ms": pmf_eval_meta["elapsed_ms"],
                    "call_elapsed_ms": int((time.perf_counter() - call_start) * 1000),
                }
            )
            _log_reward_event(log_payload)

        return score

    except Exception as exc:
        _bump_stat("reward_exceptions")
        _debug(f"prob_id={prob_id!r}: exception → min reward: {exc}")
        log_payload.update(
            {
                "score": PMF_MIN_REWARD,
                "reason": "reward_exception",
                "reward_error": str(exc),
                "call_elapsed_ms": int((time.perf_counter() - call_start) * 1000),
            }
        )
        _log_reward_event(log_payload)
        return PMF_MIN_REWARD
