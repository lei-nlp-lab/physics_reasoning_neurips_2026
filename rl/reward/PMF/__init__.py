"""
PMF — Physical Modelling Fidelity metric.

Measures how closely an LLM's physics modelling process matches an expert's,
decomposed into three motif types: QEL, VCL, AEB.

Quick start:
    from PMF import PMFEvaluator

    ev = PMFEvaluator()
    result = ev.evaluate(
        gt_text="Expert solution text...",
        pred_text="LLM reasoning text...",
    )
    print(f"PMF score: {result['pmf']:.3f}")
    print(f"QEL F1:   {result['qel']['f1']:.3f}")
    print(f"Missing:  {result['qel']['missing']}")
"""

from .motifs import AEB, QEL, VCL, MotifSet
from .extractor import ExtractionResult, extract_motifs, extract_motifs_async
from .scorer import compute_pmf, compute_pmf_async
from .pmf import PMFEvaluator

__all__ = [
    "PMFEvaluator",
    "QEL",
    "VCL",
    "AEB",
    "MotifSet",
    "ExtractionResult",
    "extract_motifs",
    "extract_motifs_async",
    "compute_pmf",
    "compute_pmf_async",
]
