"""
reward_joint_pmf_checker.py — Joint reward: PMF (diverge) + Checker (converge).

  JOINT_PMF_WEIGHT     * pmf_score      (diverge quality via PMF motif F1)
+ JOINT_CHECKER_WEIGHT * checker_score  (converge execution via checker)
+ JOINT_ANSWER_WEIGHT  * answer_ratio   ([ans] vs GT match ratio)

All per-component env vars (PMF_*, CHECKER_REWARD_*) are still respected.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

import reward_pmf_diverge
import reward_checker_only
import reward_report_utils

# ============================================================
# Joint weight config
# ============================================================
JOINT_PMF_WEIGHT     = float(os.environ.get("JOINT_PMF_WEIGHT", "0.4"))
JOINT_CHECKER_WEIGHT = float(os.environ.get("JOINT_CHECKER_WEIGHT", "0.4"))
JOINT_ANSWER_WEIGHT  = float(os.environ.get("JOINT_ANSWER_WEIGHT", "0.2"))
JOINT_MIN_REWARD     = float(os.environ.get("JOINT_MIN_REWARD", "0.0"))
DEBUG                = os.environ.get("PMF_DEBUG", "0") == "1"


def _debug(msg: str) -> None:
    if DEBUG:
        print(f"[joint_reward] {msg}", flush=True)


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
    Compute joint PMF + checker reward.

    Args:
        data_source:   Dataset identifier
        solution_str:  Full model response
        ground_truth:  JSON string containing prob_id, answer,
                       physics_mapping, key_constraints
        extra_info:    Optional auxiliary info dict

    Returns:
        Float in [JOINT_MIN_REWARD, 1.0]:
            JOINT_PMF_WEIGHT * pmf_score + JOINT_CHECKER_WEIGHT * checker_score
    """
    import json
    extra_info = extra_info or {}

    pmf_score = reward_pmf_diverge.compute_score(
        data_source, solution_str, ground_truth, extra_info
    )

    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else (ground_truth or {})
    except Exception:
        gt = {}

    # Skip the sandbox checker entirely when its weight is zero (pmf-only mode).
    if JOINT_CHECKER_WEIGHT > 0.0:
        prob_id = reward_checker_only.resolve_prob_id(extra_info, gt)
        breakdown = reward_checker_only.compute_reward_breakdown(
            prob_id, solution_str, gt.get("answer", [])
        )
        checker_score = float(breakdown["checker_score"])
        answer_score = float(breakdown["answer_ratio"])
        translation_failed = bool(breakdown["checker_meta"].get("translation_failed", False))

        # Mirror reward_checker_only.compute_score's logging so the joint path
        # still produces the sandbox checker JSONL (previously missing).
        if reward_checker_only.SANDBOX_LOG_ALL or checker_score < 1.0:
            meta = breakdown["checker_meta"]
            log_payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prob_id": prob_id,
                "checker_passed": bool(meta["passed"]),
                "translation_failed": bool(meta.get("translation_failed", False)),
                "checker_score": checker_score,
                "execution_score": float(breakdown["execution_score"]),
                "coverage_score": float(breakdown["coverage_score"]),
                "answer_ratio": answer_score,
                "expected_var_count": len(meta["checker_vars"]),
                "emitted_var_count": len(breakdown["emitted_vars"]),
                # pmf_score is logged alongside so the joint reward's density can
                # be attributed to the checker vs the PMF term.
                "pmf_score": float(pmf_score),
                "report": breakdown["checker_report"],
            }
            log_payload.update(
                reward_report_utils.report_diagnostics(
                    breakdown["checker_report"],
                    bool(meta.get("translation_failed", False)),
                )
            )
            if meta.get("pseudocode"):
                log_payload["pseudocode"] = meta["pseudocode"][:500]
            if meta.get("translated_code"):
                log_payload["translated_code"] = meta["translated_code"][:1000]
            reward_checker_only._append_sandbox_log(log_payload)
    else:
        checker_score = 0.0
        translation_failed = False
        if JOINT_ANSWER_WEIGHT > 0.0:
            answer_score = float(reward_checker_only.answer_match_score_from_items(
                reward_checker_only.extract_ans_items(solution_str),
                gt.get("answer", []),
            ))
        else:
            answer_score = 0.0

    # Translator failure: the checker can't run, so its weight is zeroed and
    # the remaining mass is renormalised onto PMF + answer to keep the signal
    # well-calibrated instead of slapping a deterministic deficit penalty on
    # the trajectory.
    weights = {
        "pmf":     JOINT_PMF_WEIGHT,
        "checker": 0.0 if translation_failed else JOINT_CHECKER_WEIGHT,
        "answer":  JOINT_ANSWER_WEIGHT,
    }
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    total = (
        weights["pmf"] * pmf_score
        + weights["checker"] * checker_score
        + weights["answer"] * answer_score
    )
    total = float(max(JOINT_MIN_REWARD, min(1.0, total)))

    _debug(
        f"pmf={pmf_score:.3f} (w={weights['pmf']:.3f}) "
        f"checker={checker_score:.3f} (w={weights['checker']:.3f}) "
        f"answer={answer_score:.3f} (w={weights['answer']:.3f}) "
        f"translator_failed={translation_failed} "
        f"→ joint={total:.3f}"
    )

    return total
