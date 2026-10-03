"""
pmf.py — Top-level PMFEvaluator class.

Usage:
    from PMF import PMFEvaluator

    ev = PMFEvaluator()
    result = ev.evaluate(gt_text="...", pred_text="...")
    print(result["pmf"])          # overall PMF score 0-1
    print(result["qel"]["f1"])    # QEL F1
    print(result["qel"]["missing"])  # GT tuples the model missed
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict

from openai import AsyncOpenAI

from openai_responses_utils import (
    DEFAULT_OPENAI_MODEL,
    build_client_kwargs,
    resolve_api_key,
)

from .extractor import ExtractionResult, extract_motifs_async
from .motifs import MotifSet
from .scorer import compute_pmf_async


class PMFEvaluator:
    """
    Evaluates the Physical Modelling Fidelity of an LLM's physics reasoning.

    Args:
        api_key:           OpenAI API key (falls back to OPENAI_API_KEY env var)
        model:             Model for motif extraction and semantic judging
        max_retries:       Max retries for extraction API calls
        use_semantic_judge: If True, use LLM to judge semantic equivalence of tuples
        weights:           Per-motif base weights (default: equal 1/3 each)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        max_retries: int = 3,
        use_semantic_judge: bool = True,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key, env_names=("OPENAI_API_KEY",))
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )
        self.model = model
        self.max_retries = max_retries
        self.use_semantic_judge = use_semantic_judge
        self.weights = weights

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate(self, gt_text: str, pred_text: str) -> dict:
        """
        Evaluate PMF for a single problem.

        Args:
            gt_text:   Expert ground-truth reasoning / solution text
            pred_text: LLM-generated reasoning text

        Returns dict:
            {
              "pmf": float,                    # overall score [0, 1]
              "qel": {f1, precision, recall, missing, extra, ...},
              "vcl": {...},
              "aeb": {...},
              "active_motifs": [...],          # motif types included in PMF
              "gt_extraction": {status, attempts, elapsed_ms, ...},
              "pred_extraction": {status, ...},
            }
        """
        return asyncio.run(self._evaluate_async(gt_text, pred_text))

    async def _evaluate_async(self, gt_text: str, pred_text: str) -> dict:
        client = AsyncOpenAI(**build_client_kwargs(api_key=self._api_key, base_url=os.environ.get("OPENAI_BASE_URL", "").strip()))
        semaphore = asyncio.Semaphore(512)

        try:
            # Extract motifs from both texts in parallel
            gt_task = asyncio.create_task(
                extract_motifs_async(client, gt_text, model=self.model,
                                     max_retries=self.max_retries)
            )
            pred_task = asyncio.create_task(
                extract_motifs_async(client, pred_text, model=self.model,
                                     max_retries=self.max_retries)
            )
            gt_result: ExtractionResult = await gt_task
            pred_result: ExtractionResult = await pred_task

            # Compute PMF score
            judge_client = client if self.use_semantic_judge else None
            pmf_result = await compute_pmf_async(
                pred=pred_result.motifs,
                gt=gt_result.motifs,
                judge_client=judge_client,
                judge_model=self.model,
                weights=self.weights,
                semaphore=semaphore,
            )

            return {
                **pmf_result,
                "gt_extraction": _extraction_meta(gt_result),
                "pred_extraction": _extraction_meta(pred_result),
            }
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        items: list[dict],
        max_concurrent: int = 512,
        max_judge_concurrent: int = 512,
    ) -> list[dict]:
        """
        Batch evaluate PMF for multiple problems.

        Args:
            items: List of dicts, each with keys:
                   - "problem_id": str
                   - "gt_text": str
                   - "pred_text": str
            max_concurrent: Max concurrent API workers
            max_judge_concurrent: Global cap for semantic-judge calls across the batch

        Returns list of dicts, each containing problem_id + full PMF result.
        """
        return asyncio.run(self._evaluate_batch_async(items, max_concurrent, max_judge_concurrent))

    async def _evaluate_batch_async(
        self,
        items: list[dict],
        max_concurrent: int,
        max_judge_concurrent: int,
    ) -> list[dict]:
        client = AsyncOpenAI(**build_client_kwargs(api_key=self._api_key, base_url=os.environ.get("OPENAI_BASE_URL", "").strip()))
        semaphore = asyncio.Semaphore(max_concurrent)
        judge_semaphore = asyncio.Semaphore(max_judge_concurrent)

        async def _process_one(item: dict) -> dict:
            async with semaphore:
                problem_id = item.get("problem_id", "unknown")
                gt_text = item["gt_text"]
                pred_text = item["pred_text"]

                # Extract both in parallel (sharing the same client)
                gt_task = asyncio.create_task(
                    extract_motifs_async(client, gt_text, model=self.model,
                                         max_retries=self.max_retries)
                )
                pred_task = asyncio.create_task(
                    extract_motifs_async(client, pred_text, model=self.model,
                                         max_retries=self.max_retries)
                )
                gt_result: ExtractionResult = await gt_task
                pred_result: ExtractionResult = await pred_task

                judge_client = client if self.use_semantic_judge else None
                pmf_result = await compute_pmf_async(
                    pred=pred_result.motifs,
                    gt=gt_result.motifs,
                    judge_client=judge_client,
                    judge_model=self.model,
                    weights=self.weights,
                    semaphore=judge_semaphore,
                )

                return {
                    "problem_id": problem_id,
                    **pmf_result,
                    "gt_extraction": _extraction_meta(gt_result),
                    "pred_extraction": _extraction_meta(pred_result),
                }

        try:
            tasks = [asyncio.create_task(_process_one(item)) for item in items]
            return await asyncio.gather(*tasks)
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Pre-extracted motif evaluation (skip extraction step)
    # ------------------------------------------------------------------

    def evaluate_motifs(self, pred: MotifSet, gt: MotifSet) -> dict:
        """
        Compute PMF directly from pre-extracted MotifSets.
        Useful when GT motifs are pre-annotated offline.
        """
        return asyncio.run(self._evaluate_motifs_async(pred, gt))

    async def _evaluate_motifs_async(self, pred: MotifSet, gt: MotifSet) -> dict:
        client = AsyncOpenAI(**build_client_kwargs(api_key=self._api_key, base_url=os.environ.get("OPENAI_BASE_URL", "").strip()))
        try:
            judge_client = client if self.use_semantic_judge else None
            return await compute_pmf_async(
                pred=pred,
                gt=gt,
                judge_client=judge_client,
                judge_model=self.model,
                weights=self.weights,
            )
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extraction_meta(result: ExtractionResult) -> dict:
    return {
        "status": result.status,
        "error": result.error,
        "error_type": result.error_type,
        "attempts": result.attempts,
        "elapsed_ms": result.elapsed_ms,
        "qel_count": len(result.motifs.qel),
        "vcl_count": len(result.motifs.vcl),
        "aeb_count": len(result.motifs.aeb),
        "motifs": result.motifs.to_dict(),
    }
