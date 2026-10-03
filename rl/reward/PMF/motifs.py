"""
motifs.py — PMF motif dataclass definitions.

Three motif types capture different dimensions of physics modelling:
  QEL: Quantity-Equation-Law  (quantity, law, context)
  VCL: Variable Coupling Link  ((law_a, ctx_a), shared_var, (law_b, ctx_b))
  AEB: Assumption-Equation Binding  (assumption, law, context)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QEL:
    """A physical quantity used in an equation, under a specific law in a given context.

    Example: (quantity="m", law="energy_conservation", context="incline_descent")
    """
    quantity: str
    law: str
    context: str

    def to_dict(self) -> dict:
        return {"quantity": self.quantity, "law": self.law, "context": self.context}

    def __str__(self) -> str:
        return f"QEL({self.quantity}, {self.law}, {self.context})"


@dataclass(frozen=True)
class VCL:
    """A variable coupling link between two equations.

    Equation A (law_a applied in ctx_a) outputs shared_var,
    which is consumed as input by Equation B (law_b applied in ctx_b).
    """
    law_a: str
    ctx_a: str
    shared_var: str
    law_b: str
    ctx_b: str

    def to_dict(self) -> dict:
        return {
            "law_a": self.law_a,
            "ctx_a": self.ctx_a,
            "shared_var": self.shared_var,
            "law_b": self.law_b,
            "ctx_b": self.ctx_b,
        }

    def __str__(self) -> str:
        return (
            f"VCL(({self.law_a},{self.ctx_a}) --{self.shared_var}--> "
            f"({self.law_b},{self.ctx_b}))"
        )


@dataclass(frozen=True)
class AEB:
    """A simplifying assumption that constrains a specific equation.

    Example: (assumption="frictionless_incline", law="energy_conservation",
              context="incline_descent")
    """
    assumption: str
    law: str
    context: str

    def to_dict(self) -> dict:
        return {"assumption": self.assumption, "law": self.law, "context": self.context}

    def __str__(self) -> str:
        return f"AEB({self.assumption}, {self.law}, {self.context})"


@dataclass(frozen=True)
class MotifSet:
    """The complete set of modelling motifs extracted from a physics reasoning text."""
    qel: frozenset[QEL]
    vcl: frozenset[VCL]
    aeb: frozenset[AEB]

    @classmethod
    def empty(cls) -> "MotifSet":
        return cls(qel=frozenset(), vcl=frozenset(), aeb=frozenset())

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MotifSet":
        raw = raw or {}

        qel_items: set[QEL] = set()
        for item in raw.get("qel", []):
            if not isinstance(item, dict):
                continue
            quantity = str(item.get("quantity", "")).strip()
            law = str(item.get("law", "")).strip()
            context = str(item.get("context", "")).strip()
            if quantity and law and context:
                qel_items.add(QEL(quantity=quantity, law=law, context=context))

        vcl_items: set[VCL] = set()
        for item in raw.get("vcl", []):
            if not isinstance(item, dict):
                continue
            law_a = str(item.get("law_a", "")).strip()
            ctx_a = str(item.get("ctx_a", "")).strip()
            shared_var = str(item.get("shared_var", "")).strip()
            law_b = str(item.get("law_b", "")).strip()
            ctx_b = str(item.get("ctx_b", "")).strip()
            if law_a and ctx_a and shared_var and law_b and ctx_b:
                vcl_items.add(
                    VCL(
                        law_a=law_a,
                        ctx_a=ctx_a,
                        shared_var=shared_var,
                        law_b=law_b,
                        ctx_b=ctx_b,
                    )
                )

        aeb_items: set[AEB] = set()
        for item in raw.get("aeb", []):
            if not isinstance(item, dict):
                continue
            assumption = str(item.get("assumption", "")).strip()
            law = str(item.get("law", "")).strip()
            context = str(item.get("context", "")).strip()
            if assumption and law and context:
                aeb_items.add(AEB(assumption=assumption, law=law, context=context))

        return cls(
            qel=frozenset(qel_items),
            vcl=frozenset(vcl_items),
            aeb=frozenset(aeb_items),
        )

    def is_empty(self) -> bool:
        return not self.qel and not self.vcl and not self.aeb

    def to_dict(self) -> dict:
        return {
            "qel": [t.to_dict() for t in self.qel],
            "vcl": [t.to_dict() for t in self.vcl],
            "aeb": [t.to_dict() for t in self.aeb],
        }
