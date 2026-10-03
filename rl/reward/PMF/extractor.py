"""
extractor.py — LLM-based motif extraction from physics reasoning text.

Uses OpenAI GPT API to extract QEL, VCL, AEB
motif tuples from free-form physics reasoning text.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from openai_responses_utils import (
    DEFAULT_OPENAI_MODEL,
    build_client_kwargs,
    build_responses_kwargs,
    extract_response_text,
    resolve_api_key,
)

from .motifs import AEB, QEL, VCL, MotifSet


# ---------------------------------------------------------------------------
# Extraction result (includes status for error handling)
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    motifs: MotifSet
    status: str           # "ok" | "parse_error" | "timeout" | "api_error"
    error: str | None
    raw_response: str | None
    attempts: int = 0
    elapsed_ms: int | None = None
    error_type: str | None = None


class MotifExtractionParseError(ValueError):
    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a physics modelling analyst. Your task is to extract modelling facts \
from a physics reasoning text.

Extract three types of modelling tuples:

1. **QEL** (Quantity-Equation-Law): A physical quantity that appears in an \
equation, where that equation is based on a specific physics law applied in \
a specific physical context/stage/object.

2. **VCL** (Variable Coupling Link): A shared variable that connects two \
equations — one equation outputs this variable, and another equation takes \
it as input. Only extract VCL when there is a clear multi-equation coupling.

3. **AEB** (Assumption-Equation Binding): A simplifying assumption that \
constrains the validity of a specific equation.

## Normalization rules
- All string values must use lowercase with underscores (e.g. \
"newton_second_law", "block_a", "incline_descent")
- `law` must be a named physics principle (e.g. "energy_conservation", \
"newton_second_law", "work_energy_theorem", "momentum_conservation", \
"coulombs_law", "ohms_law", "kinematics")
- `context` = the physical stage, object, or region where the equation applies \
(e.g. "incline_descent", "horizontal_surface", "block_a", "block_b", \
"collision", "free_fall")
- `quantity` / `shared_var` = symbolic name of the physical quantity \
(e.g. "v", "m", "g", "mu", "h", "theta", "t")
- `assumption` = a named simplifying condition \
(e.g. "frictionless_incline", "lossless_transition", "negligible_air_resistance", \
"point_mass", "uniform_field")

## Output format
You MUST output valid JSON with exactly these three keys. \
Empty arrays are allowed when no tuples of that type are present.

```json
{
  "qel": [
    {"quantity": "m", "law": "energy_conservation", "context": "incline_descent"}
  ],
  "vcl": [
    {"law_a": "energy_conservation", "ctx_a": "incline_descent",
     "shared_var": "v", "law_b": "work_energy_theorem", "ctx_b": "horizontal_surface"}
  ],
  "aeb": [
    {"assumption": "frictionless_incline", "law": "energy_conservation",
     "context": "incline_descent"}
  ]
}
```

Only extract motifs that are **explicitly stated or clearly implied** by the \
text. Do not invent motifs that are absent. If a tuple type has no instances, \
return an empty array for that key.
"""


def _build_user_prompt(text: str) -> str:
    return (
        "Extract all QEL, VCL, and AEB modelling tuples from the following "
        "physics reasoning text.\n\n"
        f"--- BEGIN TEXT ---\n{text}\n--- END TEXT ---"
    )


# ---------------------------------------------------------------------------
# Field-level normalization
# ---------------------------------------------------------------------------

_GREEK_MAP = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "θ": "theta", "λ": "lambda", "μ": "mu",
    "ν": "nu", "π": "pi", "ρ": "rho", "σ": "sigma",
    "τ": "tau", "φ": "phi", "ω": "omega",
}


def _normalize_symbol(s: str) -> str:
    """Normalize a physical quantity / shared_var symbol."""
    s = s.strip()
    for greek, name in _GREEK_MAP.items():
        s = s.replace(greek, name)
    # Remove spaces around subscripts/superscripts
    s = re.sub(r"\s+", "_", s)
    return s.lower()


def _normalize_identifier(s: str) -> str:
    """Normalize a law, context, or assumption identifier to snake_case."""
    s = s.strip().lower()
    # Replace spaces and hyphens with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    # Remove characters that aren't alphanumeric or underscore
    s = re.sub(r"[^\w]", "", s)
    return s


# ---------------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> str:
    """Try to extract a JSON object from text that may contain markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Find the outermost { ... }
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unmatched braces in response")


# ---------------------------------------------------------------------------
# Parse raw JSON → MotifSet
# ---------------------------------------------------------------------------

def _parse_motif_set(raw: dict) -> MotifSet:
    qel_set: set[QEL] = set()
    vcl_set: set[VCL] = set()
    aeb_set: set[AEB] = set()

    for item in raw.get("qel", []):
        try:
            qel_set.add(QEL(
                quantity=_normalize_symbol(str(item["quantity"])),
                law=_normalize_identifier(str(item["law"])),
                context=_normalize_identifier(str(item["context"])),
            ))
        except (KeyError, TypeError):
            continue

    for item in raw.get("vcl", []):
        try:
            vcl_set.add(VCL(
                law_a=_normalize_identifier(str(item["law_a"])),
                ctx_a=_normalize_identifier(str(item["ctx_a"])),
                shared_var=_normalize_symbol(str(item["shared_var"])),
                law_b=_normalize_identifier(str(item["law_b"])),
                ctx_b=_normalize_identifier(str(item["ctx_b"])),
            ))
        except (KeyError, TypeError):
            continue

    for item in raw.get("aeb", []):
        try:
            aeb_set.add(AEB(
                assumption=_normalize_identifier(str(item["assumption"])),
                law=_normalize_identifier(str(item["law"])),
                context=_normalize_identifier(str(item["context"])),
            ))
        except (KeyError, TypeError):
            continue

    return MotifSet(
        qel=frozenset(qel_set),
        vcl=frozenset(vcl_set),
        aeb=frozenset(aeb_set),
    )


# ---------------------------------------------------------------------------
# Core async extraction
# ---------------------------------------------------------------------------

async def _extract_once(
    client: AsyncOpenAI,
    text: str,
    model: str,
    temperature: float,
) -> tuple[MotifSet, str]:
    """Single extraction attempt. Returns (MotifSet, raw_response)."""
    response = await client.chat.completions.create(
        **build_responses_kwargs(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(text)},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_output_tokens=2048,
        )
    )
    raw_text = extract_response_text(response)
    try:
        json_str = _extract_json_from_text(raw_text)
        parsed = json.loads(json_str)
        motifs = _parse_motif_set(parsed)
    except Exception as exc:
        raise MotifExtractionParseError(str(exc), raw_response=raw_text) from exc
    return motifs, raw_text


def _classify_extraction_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, MotifExtractionParseError):
        return "parse_error", type(exc).__name__
    if isinstance(exc, json.JSONDecodeError):
        return "parse_error", type(exc).__name__

    exc_name = type(exc).__name__
    lowered = f"{exc_name}: {exc}".lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout", exc_name
    if "json" in lowered or "brace" in lowered:
        return "parse_error", exc_name
    return "api_error", exc_name


async def extract_motifs_async(
    client: AsyncOpenAI,
    text: str,
    model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = 3,
    base_temperature: float = 0.0,
) -> ExtractionResult:
    """Extract motifs from text. Retries on parse failure at fixed temperature (determinism)."""
    start_time = time.perf_counter()
    last_error: str = ""
    last_raw: str = ""
    last_status = "api_error"
    last_error_type: str | None = None

    for attempt in range(max_retries):
        temperature = base_temperature
        try:
            motifs, raw_text = await _extract_once(client, text, model, temperature)
            return ExtractionResult(
                motifs=motifs,
                status="ok",
                error=None,
                raw_response=raw_text,
                attempts=attempt + 1,
                elapsed_ms=int((time.perf_counter() - start_time) * 1000),
                error_type=None,
            )
        except Exception as exc:
            last_error = str(exc)
            last_status, last_error_type = _classify_extraction_error(exc)
            last_raw = getattr(exc, "raw_response", "") or ""
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            continue

    return ExtractionResult(
        motifs=MotifSet.empty(),
        status=last_status,
        error=last_error,
        raw_response=last_raw or None,
        attempts=max_retries,
        elapsed_ms=int((time.perf_counter() - start_time) * 1000),
        error_type=last_error_type,
    )


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------

def extract_motifs(
    text: str,
    api_key: str | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = 3,
) -> ExtractionResult:
    """Synchronous motif extraction. Builds its own event loop."""
    _api_key = resolve_api_key(api_key, env_names=("OPENAI_API_KEY",))
    if not _api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

    client = AsyncOpenAI(**build_client_kwargs(api_key=_api_key, base_url=os.environ.get("OPENAI_BASE_URL", "").strip()))

    async def _run():
        result = await extract_motifs_async(client, text, model=model, max_retries=max_retries)
        await client.close()
        return result

    return asyncio.run(_run())
