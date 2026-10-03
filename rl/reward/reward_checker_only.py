from __future__ import annotations

import ast
import builtins
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import itertools
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    class _TorchStub:
        bfloat16 = "bfloat16"

        @staticmethod
        def inference_mode():
            def _decorator(fn):
                return fn

            return _decorator

    torch = _TorchStub()

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CHECKER_DIR = Path(
    os.environ.get(
        "CHECKER_DIR",
        str(ROOT.parent.parent / "data" / "checkers" / "phyreason"),
    )
)

sys.path.insert(0, str(ROOT))
from reward_report_utils import (
    compute_coverage_score,
    dense_checker_score,
    execution_score,
    report_diagnostics,
    extract_checker_init_contract,
    extract_checker_expected_vars,
    extract_computed_results,
    extract_result_keys_from_code,
    normalize_text,
)
from sand_box import run_physics_sandbox


CHECKER_WEIGHT = float(os.environ.get("CHECKER_REWARD_CHECKER_WEIGHT", "0.4"))
ANSWER_WEIGHT = float(os.environ.get("CHECKER_REWARD_ANSWER_WEIGHT", "0.5"))
EXECUTION_WEIGHT = float(os.environ.get("CHECKER_REWARD_EXECUTION_WEIGHT", "0.0"))
COVERAGE_WEIGHT = float(os.environ.get("CHECKER_REWARD_COVERAGE_WEIGHT", "0.0"))
ANSWER_MIN_EXECUTION_SCORE = float(
    os.environ.get("CHECKER_REWARD_ANSWER_MIN_EXECUTION_SCORE", "0.0")
)
ANSWER_REQUIRES_COMPUTED_RESULTS = (
    os.environ.get("CHECKER_REWARD_ANSWER_REQUIRES_COMPUTED_RESULTS", "0") == "1"
)
COVERAGE_MIN_EXECUTION_SCORE = float(
    os.environ.get("CHECKER_REWARD_COVERAGE_MIN_EXECUTION_SCORE", "0.5")
)
RUNTIME_COVERAGE_ONLY = os.environ.get(
    "CHECKER_REWARD_RUNTIME_COVERAGE_ONLY", "1"
) == "1"
RENORMALIZE_AVAILABLE_COMPONENTS = os.environ.get(
    "CHECKER_REWARD_RENORMALIZE_AVAILABLE_COMPONENTS", "1"
) == "1"
# Dynamic checker availability: mark checker as unavailable (and redistribute its weight)
# when code translation failed OR execution_score is below threshold.
# Combined with RENORMALIZE=1, checker's weight flows to answer when checker cannot run.
DYNAMIC_CHECKER_AVAILABILITY = os.environ.get(
    "CHECKER_REWARD_DYNAMIC_CHECKER_AVAILABILITY", "0"
) == "1"
CHECKER_MIN_EXECUTION_SCORE = float(
    os.environ.get("CHECKER_REWARD_CHECKER_MIN_EXECUTION_SCORE", "0.5")
)
# Bonus mode: checker is NOT a weighted component but an additive bonus on top of the answer.
# total = min(1.0, answer_ratio + CHECKER_BONUS_WEIGHT * checker_score)
# This ensures correct answers always receive full credit regardless of checker success.
CHECKER_BONUS_MODE = os.environ.get("CHECKER_REWARD_BONUS_MODE", "0") == "1"
CHECKER_BONUS_WEIGHT = float(os.environ.get("CHECKER_REWARD_BONUS_WEIGHT", "0.3"))
MIN_REWARD = float(os.environ.get("CHECKER_REWARD_MIN_REWARD", "0.0"))
DEBUG = os.environ.get("CHECKER_REWARD_DEBUG", "0") == "1"
FIRST_BATCH_DEBUG = os.environ.get("VERL_FIRST_BATCH_DEBUG", "0") == "1"
SANDBOX_LOG_PATH = os.environ.get("CHECKER_REWARD_LOG_PATH", "").strip()
SANDBOX_LOG_ALL = os.environ.get("CHECKER_REWARD_LOG_ALL", "0") == "1"
ERROR_LOG_PATH = os.environ.get("CHECKER_REWARD_ERROR_LOG_PATH", "log/error.log").strip()

CODE_MODEL_PATH = os.environ.get("CODE_MODEL_PATH") or "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CODE_MODEL_DEVICE = os.environ.get("CODE_MODEL_DEVICE", "cuda:0")
CODE_MODEL_MAX_TOKENS = int(os.environ.get("CODE_MODEL_MAX_TOKENS", "2048"))
CODE_TRANSLATOR_BACKEND = os.environ.get("CODE_TRANSLATOR_BACKEND", "local").strip().lower()
CODE_TRANSLATOR_ACTOR_NAME = os.environ.get("CODE_TRANSLATOR_ACTOR_NAME", "code_model_translator").strip() or "code_model_translator"
CODE_TRANSLATOR_ACTOR_NUM_GPUS = float(os.environ.get("CODE_TRANSLATOR_ACTOR_NUM_GPUS", "1.0"))
CODE_TRANSLATOR_POOL_SIZE = max(1, int(os.environ.get("CODE_TRANSLATOR_POOL_SIZE", "1")))
CODE_TRANSLATOR_ACTOR_MAX_CONCURRENCY = int(os.environ.get("CODE_TRANSLATOR_ACTOR_MAX_CONCURRENCY", "1"))
CODE_TRANSLATOR_RAY_FALLBACK_LOCAL = os.environ.get("CODE_TRANSLATOR_RAY_FALLBACK_LOCAL", "1") == "1"
CODE_TRANSLATOR_VLLM_GPU_UTILIZATION = float(
    os.environ.get("CODE_TRANSLATOR_VLLM_GPU_UTILIZATION", "0.85")
)
CODE_TRANSLATOR_VLLM_MAX_MODEL_LEN = int(
    os.environ.get("CODE_TRANSLATOR_VLLM_MAX_MODEL_LEN", "4096")
)
CODE_TRANSLATOR_VLLM_BATCH_SIZE = max(
    1, int(os.environ.get("CODE_TRANSLATOR_VLLM_BATCH_SIZE", "16"))
)
CODE_TRANSLATOR_VLLM_BATCH_TIMEOUT_MS = max(
    1, int(os.environ.get("CODE_TRANSLATOR_VLLM_BATCH_TIMEOUT_MS", "20"))
)
CODE_TRANSLATOR_VLLM_TENSOR_PARALLEL_SIZE = max(
    1, int(os.environ.get("CODE_TRANSLATOR_VLLM_TENSOR_PARALLEL_SIZE", "1"))
)
TRANSLATION_FAILED_TOKEN = "TRANSLATION_FAILED"

_PSEUDOCODE_RESULT_KEY_RE = re.compile(r"""results\[['"]([^'"]+)['"]\]""")
_PSEUDOCODE_VAR_LINE_RE = re.compile(r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\b")
_PSEUDOCODE_EMIT_ASSIGNMENT_RE = re.compile(
    r"""^\s*(?:-\s*)?results\[['"](?P<key>[^'"]+)['"]\]\s*=\s*(?P<expr>.+?)\s*$"""
)
_SIMPLE_ASSIGNMENT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$"
)
_RESULT_ASSIGN_FLOAT_RE = re.compile(
    r"""^(?P<indent>\s*)results\s*\[\s*(?P<quote>['"])(?P<key>[^'"]+)(?P=quote)\s*\]\s*=\s*float\((?P<expr>.+)\)\s*$"""
)
_RESULT_ASSIGN_RE = re.compile(
    r"""^(?P<indent>\s*)results\s*\[\s*(?P<quote>['"])(?P<key>[^'"]+)(?P=quote)\s*\]\s*=\s*(?P<expr>.+)\s*$"""
)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECTION_HEADER_RE = re.compile(r"^\s*\[([A-Za-z0-9_]+)\]\s*$")
_ANGLE_FUNC_TOKENS = (
    "asin",
    "acos",
    "atan",
    "math.asin",
    "math.acos",
    "math.atan",
    "sp.asin",
    "sp.acos",
    "sp.atan",
    "_rad",
)
_STATIC_SANDBOX_NAMES = set(dir(builtins)) | {
    "sp",
    "sympy",
    "math",
    "results",
    "trace",
    "g",
    "__builtins__",
    "Eq",
    "Symbol",
    "symbols",
    "simplify",
    "solve",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sqrt",
    "log",
    "ln",
    "exp",
    "pi",
    "rad",
    "degrees",
    "radians",
}

_RESULT_KEY_SUBSCRIPT_MAP = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)
_RESULT_KEY_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)
_RESULT_KEY_TEXT_REPLACEMENTS = {
    "Δ": "delta_",
    "∆": "delta_",
    "δ": "delta_",
    "θ": "theta",
    "μ": "mu",
    "ω": "omega",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "η": "eta",
    "τ": "tau",
    "φ": "phi",
    "Φ": "phi",
    "ρ": "rho",
    "λ": "lambda",
    "′": "_prime_",
    "'": "_prime_",
    "\\": "",
}


def _extract_emit_keys_from_pseudocode(pseudocode: str) -> list[str]:
    seen = set()
    result = []
    for key in _PSEUDOCODE_RESULT_KEY_RE.findall(pseudocode or ""):
        cleaned = str(key).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _extract_emit_assignments_from_pseudocode(pseudocode: str) -> list[tuple[str, str]]:
    seen = set()
    result: list[tuple[str, str]] = []
    current_section = ""

    for raw_line in (pseudocode or "").splitlines():
        section_match = _SECTION_HEADER_RE.match(raw_line)
        if section_match:
            current_section = section_match.group(1).strip().lower()
            continue
        if current_section != "emit":
            continue
        emit_match = _PSEUDOCODE_EMIT_ASSIGNMENT_RE.match(raw_line)
        if not emit_match:
            continue
        key = emit_match.group("key").strip()
        expr = re.sub(r"\s+#.*$", "", emit_match.group("expr")).strip()
        if key and expr and key not in seen:
            seen.add(key)
            result.append((key, expr))

    if result:
        return result

    for raw_line in (pseudocode or "").splitlines():
        emit_match = _PSEUDOCODE_EMIT_ASSIGNMENT_RE.match(raw_line)
        if not emit_match:
            continue
        key = emit_match.group("key").strip()
        expr = re.sub(r"\s+#.*$", "", emit_match.group("expr")).strip()
        if key and expr and key not in seen:
            seen.add(key)
            result.append((key, expr))
    return result


def _extract_degree_vars_from_pseudocode(pseudocode: str) -> list[str]:
    seen = set()
    result = []
    current_section = ""
    for raw_line in (pseudocode or "").splitlines():
        section_match = re.match(r"^\s*\[([A-Za-z0-9_]+)\]\s*$", raw_line)
        if section_match:
            current_section = section_match.group(1).strip().lower()
            continue
        if current_section not in {"givens", "shared_variables", "variables"}:
            continue
        lowered = raw_line.lower()
        if "[deg]" not in lowered and "degree" not in lowered and "degrees" not in lowered and "°" not in raw_line:
            continue
        match = _PSEUDOCODE_VAR_LINE_RE.match(raw_line)
        if not match:
            continue
        key = match.group(1)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _extract_pseudocode_variable_glossary(
    pseudocode: str,
) -> list[tuple[str, str, str]]:
    seen = set()
    result: list[tuple[str, str, str]] = []
    current_section = ""
    allowed_sections = {"givens", "shared_variables", "variables"}

    for raw_line in (pseudocode or "").splitlines():
        section_match = re.match(r"^\s*\[([A-Za-z0-9_]+)\]\s*$", raw_line)
        if section_match:
            current_section = section_match.group(1).strip().lower()
            continue
        if current_section not in allowed_sections:
            continue
        match = _PSEUDOCODE_VAR_LINE_RE.match(raw_line)
        if not match:
            continue

        name = match.group(1).strip()
        if not name or name in seen:
            continue

        remainder = raw_line[match.end() :].strip()
        description = ""
        paren_matches = re.findall(r"\(([^()]*)\)", remainder)
        if paren_matches:
            description = paren_matches[-1].strip()
        elif remainder.startswith(":"):
            description = remainder[1:].strip()
        elif remainder.startswith("="):
            description = remainder[1:].strip()
        elif remainder:
            description = remainder.strip(" -")

        description = re.sub(r"\s+", " ", description).strip(" .")
        seen.add(name)
        result.append((name, current_section, description))

    return result


def _canonicalize_result_key(key: str) -> str:
    text = str(key).strip()
    text = text.translate(_RESULT_KEY_SUBSCRIPT_MAP)
    text = text.translate(_RESULT_KEY_SUPERSCRIPT_MAP)
    for source, target in _RESULT_KEY_TEXT_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _checker_contract_result_keys(checker_init_contract: list[dict[str, str]]) -> list[str]:
    seen = set()
    result = []
    for item in checker_init_contract:
        key = str(item.get("result_key", "")).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _checker_contract_lines(checker_init_contract: list[dict[str, str]]) -> list[str]:
    lines = []
    for item in checker_init_contract:
        checker_name = str(item.get("checker_name", "")).strip()
        result_key = str(item.get("result_key", "")).strip()
        if checker_name and result_key:
            lines.append(f"{checker_name} <- results[{result_key!r}]")
    return lines


def _build_result_alias_pairs(
    source_keys: list[str],
    checker_result_keys: list[str],
    preferred_source_keys: list[str] | None = None,
) -> list[tuple[str, str]]:
    canonical_sources: dict[str, list[str]] = {}
    ordered_sources = list(dict.fromkeys(source_keys))
    preferred_order = list(dict.fromkeys(preferred_source_keys or []))
    for source in ordered_sources:
        canonical = _canonicalize_result_key(source)
        if not canonical:
            continue
        canonical_sources.setdefault(canonical, []).append(source)

    alias_pairs: list[tuple[str, str]] = []
    existing_sources = set(ordered_sources)
    for target in checker_result_keys:
        if target in existing_sources:
            continue
        canonical = _canonicalize_result_key(target)
        if not canonical:
            continue
        candidates = canonical_sources.get(canonical, [])
        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) == 1 and unique_candidates[0] != target:
            alias_pairs.append((unique_candidates[0], target))
            continue
        preferred_candidates = [key for key in preferred_order if key in unique_candidates]
        if len(preferred_candidates) == 1 and preferred_candidates[0] != target:
            alias_pairs.append((preferred_candidates[0], target))
    return alias_pairs


def build_translation_messages(
    pseudocode: str, checker_init_contract: list[dict[str, str]]
) -> list[dict]:
    checker_result_keys = _checker_contract_result_keys(checker_init_contract)
    vars_str = ", ".join(f'"{v}"' for v in checker_result_keys) if checker_result_keys else "(none)"
    contract_lines = _checker_contract_lines(checker_init_contract)
    contract_text = "\n".join(f"  - {line}" for line in contract_lines) if contract_lines else "  - (none)"
    emit_keys = _extract_emit_keys_from_pseudocode(pseudocode)
    emit_keys_str = ", ".join(f'"{key}"' for key in emit_keys) if emit_keys else "(none)"
    emit_assignments = _extract_emit_assignments_from_pseudocode(pseudocode)
    emit_assignment_text = (
        "\n".join(f"  - results[{key!r}] = {expr}" for key, expr in emit_assignments)
        if emit_assignments
        else "  - (none)"
    )
    degree_vars = _extract_degree_vars_from_pseudocode(pseudocode)
    degree_vars_str = ", ".join(f'"{key}"' for key in degree_vars) if degree_vars else "(none)"
    pseudocode_glossary = _extract_pseudocode_variable_glossary(pseudocode)
    glossary_text = (
        "\n".join(
            f"  - {name} [{section}]: {description or '(no description provided)'}"
            for name, section, description in pseudocode_glossary
        )
        if pseudocode_glossary
        else "  - (none)"
    )
    system = (
        "You are a Python code translator. Convert the physics pseudocode below "
        "into a self-contained, executable Python script.\n\n"
        "Rules:\n"
        "- The pseudocode is complete. Treat it as the full specification.\n"
        "- Only translate the provided pseudocode into Python. Do not solve the problem again.\n"
        "- First understand the pseudocode's physical quantities and their meanings; only after that may you align names to the checker contract.\n"
        "- Use [equations], [constraints], [input_mapping], and [emit] as the primary executable specification.\n"
        "- Treat [steps] as a concise guide to solve order, not as a requirement to reproduce every arithmetic manipulation line by line.\n"
        "- If an emitted quantity can be obtained directly from the provided equations, compute it in code using sympy or math instead of expanding every manual arithmetic step from [steps].\n"
        "- Prefer concise library-based solving such as sympy symbols, Eq, solve, simplify, and substitution when they match the provided pseudocode.\n"
        "- The pseudocode may contain short local calculation steps, substitutions, or rearrangements; preserve their meaning, but do not generate verbose calculator-style code just to mirror them.\n"
        "- Do not invent equations, constants, variables, fallback logic, or assumptions that are not explicitly present.\n"
        "- If the pseudocode is missing information or is ambiguous, output exactly TRANSLATION_FAILED.\n"
        "- Only use sympy and math (no numpy, scipy, etc.).\n"
        "- Use ASCII-friendly Python identifiers for local Python variables. If the pseudocode refers to common symbols such as dt, dx, theta, mu, pi, or delta_P, keep them as ASCII variable names in code.\n"
        "- Initialize results = {} at the top.\n"
        "- Store every computed quantity into results[<variable_name>].\n"
        "- Each [emit] entry in the pseudocode corresponds to exactly one update to the results dictionary.\n"
        "- The PRIMARY output keys are the [emit] keys from the pseudocode: "
        f"{emit_keys_str}\n"
        "- The exact [emit] contract is:\n"
        f"{emit_assignment_text}\n"
        "- Pseudocode variable glossary (semantic source of truth):\n"
        f"{glossary_text}\n"
        "- The ONLY checker context available to you is this checker-init contract. It describes how checker __init__ reads results keys, and must be used only for variable-name alignment, not for inferring physics logic:\n"
        f"{contract_text}\n"
        "- results[...] keys are exact string literals and may contain Unicode or punctuation. Preserve the exact [emit] / checker key text even when the local Python variable name is ASCII.\n"
        "- If an emitted value is numeric and concrete, write results[<variable_name>] = float(value) after solving it.\n"
        "- If an emitted value still contains free sympy symbols, DO NOT call float(...). Keep it as a symbolic sympy expression.\n"
        "- Treat the pseudocode glossary and [emit] block as the semantic source of truth. The checker contract may rename a quantity, but it must not change what the quantity means.\n"
        "- If a pseudocode quantity is the same physical variable as one listed in the checker-init contract, align the final results[...] key to the contract key.\n"
        "- When aligning to checker keys, map by physical meaning from the pseudocode, not by superficial name similarity alone.\n"
        "- If a checker key is ambiguous or conflicts with the pseudocode meaning, keep the pseudocode-defined quantity and only mirror an alias when the two quantities are clearly identical.\n"
        "- If an [emit] key reuses a short name that also appears in givens or intermediates, the [emit] definition wins for the final results[...] output of that key.\n"
        "- The exact results keys required by checker-init are: "
        f"{vars_str}\n"
        "- If one contract key is only a trivial alias of an emitted quantity (for example t vs t_meet, v vs v_avg, overlinev vs v), you may mirror the same value into that contract key. Do not invent new physics quantities.\n"
        "- If the pseudocode marks a variable with [deg] or says degrees, store degrees in results, not radians. Degree-marked variables detected here: "
        f"{degree_vars_str}\n"
        "- Never leave a variable name undefined at runtime. If a symbol such as m, theta, lambda, R, or S is used symbolically and not given numerically, declare it with sympy symbols(...) before using it.\n"
        "- Do not emit extra bookkeeping keys unless they are direct aliases needed for checker compatibility.\n"
        "- Use g = 10.0 for gravitational acceleration unless specified otherwise.\n"
        "- The pseudocode may be organized with tags such as [givens], [shared_variables], [global_equations], [sub1], [input_mapping], [variables], [equations], [constraints], [steps], [emit], and [subans]; preserve their meaning faithfully.\n"
        "- [subans] provides the final answer annotation for a sub-question; it is not a results key and should not be translated into code.\n"
        "- Common failure modes to avoid:\n"
        "  * storing asin/acos/atan outputs in radians when the pseudocode expects degrees\n"
        "  * using float(...) on symbolic expressions with free symbols\n"
        "  * referencing names like m or theta before declaring them\n"
        "- Output ONLY the Python code inside a ```python ... ``` block, no explanation."
    )
    user = (
        f"CHECKER INIT CONTRACT:\n{contract_text}\n\n"
        f"CONTRACT RESULT KEYS: {vars_str}\n\n"
        f"PSEUDOCODE:\n{pseudocode}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_PYTHON_FENCE_PATTERNS = (
    r"```python\n(.*?)\n```",
    r"```\n(.*?)\n```",
)
_LIKELY_CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"from\s+\S+\s+import\s+|"
    r"import\s+\S+|"
    r"def\s+\w+\s*\(|"
    r"class\s+\w+\s*[:\(]|"
    r"if\s+.+:|"
    r"elif\s+.+:|"
    r"else:|"
    r"for\s+.+:|"
    r"while\s+.+:|"
    r"try:|"
    r"except\b.+:|"
    r"with\s+.+:|"
    r"return\b|"
    r"raise\b|"
    r"pass\b|"
    r"break\b|"
    r"continue\b|"
    r"results\s*=|"
    r"results\[|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*="
    r")"
)


def _parseable_python(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return ""
    try:
        ast.parse(candidate)
    except SyntaxError:
        return ""
    return candidate


def _extract_python_from_response_text(response: str) -> str:
    text = (response or "").strip()
    if not text:
        return ""

    for pattern in _PYTHON_FENCE_PATTERNS:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            fenced = match.group(1).strip()
            if fenced:
                return fenced

    parsed_full = _parseable_python(text)
    if parsed_full:
        return parsed_full

    lines = text.splitlines()
    start_idx = None
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or _LIKELY_CODE_LINE_RE.match(raw_line):
            start_idx = idx
            break

    if start_idx is not None:
        candidate_lines = lines[start_idx:]
        while candidate_lines and not candidate_lines[-1].strip():
            candidate_lines.pop()
        while candidate_lines:
            candidate = "\n".join(candidate_lines).strip()
            parsed_candidate = _parseable_python(candidate)
            if parsed_candidate:
                return parsed_candidate
            candidate_lines.pop()

    return text


class CodeModelTranslator:
    """Lazily loads the code model used for pseudocode -> Python translation."""

    def __init__(self):
        self._loaded = False
        self._load_lock = threading.Lock()
        self._last_prompt_messages = []
        self._last_prompt_text = ""
        self._last_raw_response = ""
        self._last_extracted_python = ""

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Using `TRANSFORMERS_CACHE` is deprecated.*",
                    category=FutureWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="`torch_dtype` is deprecated! Use `dtype` instead!.*",
                    category=FutureWarning,
                )
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()
                self._tokenizer = AutoTokenizer.from_pretrained(
                    CODE_MODEL_PATH, trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    CODE_MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to(CODE_MODEL_DEVICE).eval()
            self._loaded = True
            _debug(f"Loaded {CODE_MODEL_PATH} on {CODE_MODEL_DEVICE}")

    def _build_prompt(
        self, pseudocode: str, checker_init_contract: list[dict[str, str]]
    ) -> list[dict]:
        return build_translation_messages(pseudocode, checker_init_contract)

    def _extract_python_from_response(self, response: str) -> str:
        return _extract_python_from_response_text(response)

    @torch.inference_mode()
    def translate(self, pseudocode: str, checker_init_contract: list[dict[str, str]]) -> str:
        messages = self._build_prompt(pseudocode, checker_init_contract)
        self._last_prompt_messages = messages
        self._last_prompt_text = ""
        self._last_raw_response = ""
        self._last_extracted_python = ""
        self._ensure_loaded()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self._last_prompt_text = text
        inputs = self._tokenizer(text, return_tensors="pt").to(CODE_MODEL_DEVICE)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=CODE_MODEL_MAX_TOKENS,
            temperature=0.0,
            do_sample=False,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)
        extracted = self._extract_python_from_response(response)
        self._last_raw_response = response
        self._last_extracted_python = extracted
        return extracted


class VllmCodeTranslator:
    """Single vLLM engine translator that can serve batched translation requests."""

    def __init__(self):
        self._loaded = False
        self._load_lock = threading.Lock()
        self._last_prompt_messages = []
        self._last_prompt_text = ""
        self._last_raw_response = ""
        self._last_extracted_python = ""

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Using `TRANSFORMERS_CACHE` is deprecated.*",
                        category=FutureWarning,
                    )
                    from transformers import AutoTokenizer
                    from transformers.utils import logging as hf_logging
                    from vllm import LLM, SamplingParams

                    hf_logging.set_verbosity_error()
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        CODE_MODEL_PATH, trust_remote_code=True
                    )
                    self._sampling_params = SamplingParams(
                        max_tokens=CODE_MODEL_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    self._llm = LLM(
                        model=CODE_MODEL_PATH,
                        tensor_parallel_size=CODE_TRANSLATOR_VLLM_TENSOR_PARALLEL_SIZE,
                        dtype="bfloat16",
                        gpu_memory_utilization=CODE_TRANSLATOR_VLLM_GPU_UTILIZATION,
                        trust_remote_code=True,
                        max_model_len=CODE_TRANSLATOR_VLLM_MAX_MODEL_LEN,
                        enable_prefix_caching=True,
                    )
            except Exception as exc:
                has_hf_token = bool(
                    os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                )
                raise RuntimeError(
                    "Failed to initialize vLLM translator. "
                    f"model={CODE_MODEL_PATH}, "
                    f"gpu_memory_utilization={CODE_TRANSLATOR_VLLM_GPU_UTILIZATION}, "
                    f"max_model_len={CODE_TRANSLATOR_VLLM_MAX_MODEL_LEN}, "
                    f"tensor_parallel_size={CODE_TRANSLATOR_VLLM_TENSOR_PARALLEL_SIZE}, "
                    f"has_hf_token={has_hf_token}. "
                    "Common causes: Hugging Face auth/rate limit issues, invalid model path, "
                    "or insufficient free GPU memory on the translator GPU."
                ) from exc
            self._loaded = True
            _debug(
                "Loaded vLLM translator "
                f"{CODE_MODEL_PATH} with gpu_mem={CODE_TRANSLATOR_VLLM_GPU_UTILIZATION:.2f}, "
                f"max_model_len={CODE_TRANSLATOR_VLLM_MAX_MODEL_LEN}"
            )

    def _build_prompt(
        self, pseudocode: str, checker_init_contract: list[dict[str, str]]
    ) -> list[dict]:
        return build_translation_messages(pseudocode, checker_init_contract)

    def _extract_python_from_response(self, response: str) -> str:
        return _extract_python_from_response_text(response)

    @torch.inference_mode()
    def translate_batch(
        self,
        pseudocode_batch: list[str],
        checker_init_contract_batch: list[list[dict[str, str]]],
    ) -> list[str]:
        self._ensure_loaded()
        prompts = []
        last_prompt_messages = []
        last_prompt_text = ""

        for pseudocode, checker_init_contract in zip(
            pseudocode_batch, checker_init_contract_batch
        ):
            messages = self._build_prompt(pseudocode, checker_init_contract)
            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            last_prompt_messages = messages
            last_prompt_text = prompt_text

        outputs = self._llm.generate(prompts, sampling_params=self._sampling_params)
        translated = []
        last_raw_response = ""
        last_extracted_python = ""
        for output in outputs:
            response = output.outputs[0].text if output.outputs else ""
            extracted = self._extract_python_from_response(response)
            translated.append(extracted)
            last_raw_response = response
            last_extracted_python = extracted

        self._last_prompt_messages = last_prompt_messages
        self._last_prompt_text = last_prompt_text
        self._last_raw_response = last_raw_response
        self._last_extracted_python = last_extracted_python
        return translated

    def translate(self, pseudocode: str, checker_init_contract: list[dict[str, str]]) -> str:
        return self.translate_batch([pseudocode], [checker_init_contract])[0]


_local_translator = None
_local_translator_lock = threading.Lock()


def _get_translator():
    global _local_translator
    if _local_translator is None:
        with _local_translator_lock:
            if _local_translator is None:
                if CODE_TRANSLATOR_BACKEND in {"local_vllm", "vllm_local"}:
                    _local_translator = VllmCodeTranslator()
                else:
                    _local_translator = CodeModelTranslator()
    return _local_translator


class _RayCodeTranslatorActor:
    """Frozen inference-only translator service hosted as a Ray actor."""

    def __init__(self):
        self._translator = CodeModelTranslator()

    def translate(self, pseudocode: str, checker_init_contract: list[dict[str, str]]) -> str:
        return self._translator.translate(pseudocode, checker_init_contract)


class _QueuedTranslationRequest:
    def __init__(self, pseudocode: str, checker_init_contract: list[dict[str, str]]):
        self.pseudocode = pseudocode
        self.checker_init_contract = checker_init_contract
        self.done = threading.Event()
        self.result = None
        self.error = None


class _RayBatchedVllmTranslatorActor:
    """Single vLLM engine with a lightweight batching queue inside one Ray actor."""

    def __init__(self):
        self._translator = VllmCodeTranslator()
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker.start()

    def _resolve_batch(self, requests: list[_QueuedTranslationRequest]) -> None:
        outputs = self._translator.translate_batch(
            [request.pseudocode for request in requests],
            [request.checker_init_contract for request in requests],
        )
        for request, output in zip(requests, outputs):
            request.result = output
            request.done.set()

    def _fail_batch(self, requests: list[_QueuedTranslationRequest], exc: Exception) -> None:
        for request in requests:
            request.error = exc
            request.done.set()

    def _batch_loop(self) -> None:
        timeout_s = CODE_TRANSLATOR_VLLM_BATCH_TIMEOUT_MS / 1000.0
        while True:
            first = self._queue.get()
            requests = [first]
            deadline = time.monotonic() + timeout_s
            while len(requests) < CODE_TRANSLATOR_VLLM_BATCH_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    requests.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                self._resolve_batch(requests)
            except Exception as exc:  # pragma: no cover - depends on runtime backend
                self._fail_batch(requests, exc)

    def translate(self, pseudocode: str, checker_init_contract: list[dict[str, str]]) -> str:
        request = _QueuedTranslationRequest(pseudocode, checker_init_contract)
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result


_translator_actors = None
_translator_actor_lock = threading.Lock()
_translator_rr_index = 0
_translator_rr_lock = threading.Lock()

def _translator_actor_name(index: int) -> str:
    if CODE_TRANSLATOR_POOL_SIZE <= 1:
        return CODE_TRANSLATOR_ACTOR_NAME
    return f"{CODE_TRANSLATOR_ACTOR_NAME}_{index}"


def _get_or_create_ray_actor(ray, actor_cls, actor_name: str):
    try:
        return ray.get_actor(actor_name)
    except ValueError:
        pass

    try:
        return actor_cls.options(
            name=actor_name,
            lifetime="detached",
        ).remote()
    except Exception as exc:
        try:
            return ray.get_actor(actor_name)
        except ValueError:
            raise exc


def _get_ray_translator_actors():
    global _translator_actors
    if _translator_actors is not None:
        return _translator_actors

    with _translator_actor_lock:
        if _translator_actors is not None:
            return _translator_actors

        import ray

        actor_impl = _RayCodeTranslatorActor
        if CODE_TRANSLATOR_BACKEND in {"ray_vllm", "vllm", "vllm_actor"}:
            actor_impl = _RayBatchedVllmTranslatorActor

        actor_cls = ray.remote(
            num_gpus=CODE_TRANSLATOR_ACTOR_NUM_GPUS,
            max_concurrency=CODE_TRANSLATOR_ACTOR_MAX_CONCURRENCY,
        )(actor_impl)

        _translator_actors = [
            _get_or_create_ray_actor(ray, actor_cls, _translator_actor_name(idx))
            for idx in range(CODE_TRANSLATOR_POOL_SIZE)
        ]
        return _translator_actors


def _select_ray_translator_actor():
    actors = _get_ray_translator_actors()
    if len(actors) == 1:
        return actors[0]

    global _translator_rr_index
    with _translator_rr_lock:
        actor = actors[_translator_rr_index % len(actors)]
        _translator_rr_index += 1
        return actor


_api_translator_client = None
_api_translator_lock = threading.Lock()


def _translate_via_api(
    pseudocode: str, checker_init_contract: list[dict[str, str]]
) -> str:
    global _api_translator_client
    from openai_responses_utils import (
        build_client_kwargs,
        build_responses_kwargs,
        extract_response_text,
        resolve_api_key,
    )
    messages = build_translation_messages(pseudocode, checker_init_contract)
    # Allow translator to be routed to a separate endpoint (e.g. a local vLLM
    # on GPU 3) without colliding with the judge/pmf endpoint on OPENAI_*.
    api_key = resolve_api_key(env_names=("CODE_TRANSLATOR_API_KEY", "OPENAI_API_KEY"))
    base_url = os.environ.get(
        "CODE_TRANSLATOR_BASE_URL",
        os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    model = os.environ.get(
        "CODE_TRANSLATOR_MODEL",
        os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini-2024-07-18"),
    )
    if not api_key:
        raise RuntimeError(
            "CODE_TRANSLATOR_API_KEY or OPENAI_API_KEY is required for translator backend=api"
        )
    with _api_translator_lock:
        if _api_translator_client is None:
            from openai import OpenAI
            _api_translator_client = OpenAI(
                **build_client_kwargs(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=float(os.environ.get("CODE_TRANSLATOR_TIMEOUT_S", "120")),
                )
            )
    resp = _api_translator_client.chat.completions.create(
        **build_responses_kwargs(
            model=model,
            messages=messages,
            temperature=0.0,
            max_output_tokens=int(os.environ.get("CODE_TRANSLATOR_MAX_TOKENS", "4096")),
        )
    )
    return _extract_python_from_response_text(extract_response_text(resp))


def _translate_pseudocode(
    pseudocode: str, checker_init_contract: list[dict[str, str]]
) -> str:
    backend = CODE_TRANSLATOR_BACKEND
    checker_result_keys = _checker_contract_result_keys(checker_init_contract)
    if backend in {"api", "openai", "openai_api"}:
        _first_batch_trace(
            f"translate START backend={backend} checker_contract={len(checker_init_contract)} "
            f"checker_keys={len(checker_result_keys)} pseudocode_chars={len(pseudocode)}"
        )
        translated = _translate_via_api(pseudocode, checker_init_contract)
        _first_batch_trace(f"translate DONE backend={backend} translated_chars={len(translated)}")
        return translated

    if backend in {"local_vllm", "vllm_local"}:
        _first_batch_trace(
            f"translate START backend={backend} checker_contract={len(checker_init_contract)} "
            f"checker_keys={len(checker_result_keys)} pseudocode_chars={len(pseudocode)}"
        )
        translated = _get_translator().translate(pseudocode, checker_init_contract)
        _first_batch_trace(f"translate DONE backend={backend} translated_chars={len(translated)}")
        return translated

    if backend in {"ray", "ray_actor", "actor", "ray_vllm", "vllm", "vllm_actor"}:
        try:
            import ray

            _first_batch_trace(
                f"translate START backend={backend} checker_contract={len(checker_init_contract)} "
                f"checker_keys={len(checker_result_keys)} pseudocode_chars={len(pseudocode)}"
            )
            actor = _select_ray_translator_actor()
            translated = ray.get(actor.translate.remote(pseudocode, checker_init_contract))
            _first_batch_trace(f"translate DONE backend={backend} translated_chars={len(translated)}")
            return translated
        except Exception as exc:
            _debug(f"Ray translator actor failed; backend={backend}: {exc}")
            if not CODE_TRANSLATOR_RAY_FALLBACK_LOCAL:
                raise RuntimeError(
                    "Translator actor is required but unavailable. "
                    "This GRPO run is configured so that only the policy model is trained, "
                    "while the code model must remain a separate frozen inference service."
                ) from exc

    _first_batch_trace(
        f"translate START backend=fallback_local checker_contract={len(checker_init_contract)} "
        f"checker_keys={len(checker_result_keys)} pseudocode_chars={len(pseudocode)}"
    )
    translated = _get_translator().translate(pseudocode, checker_init_contract)
    _first_batch_trace(f"translate DONE backend=fallback_local translated_chars={len(translated)}")
    return translated


def _apply_pseudocode_ascii_replacements(text: str) -> str:
    replacements = [
        ("Δt", "dt"),
        ("Δx", "dx"),
        ("Δv", "dv"),
        ("ΔP", "delta_P"),
        ("∆t", "dt"),
        ("∆x", "dx"),
        ("∆v", "dv"),
        ("∆P", "delta_P"),
        ("Δ", "delta_"),
        ("∆", "delta_"),
        ("θ", "theta"),
        ("μ", "mu"),
        ("ω", "omega"),
        ("α", "alpha"),
        ("φ", "phi"),
        ("π", "pi"),
        ("²", "**2"),
        ("³", "**3"),
        ("≤", "<="),
        ("≥", ">="),
        ("−", "-"),
        ("×", "*"),
        ("·", "*"),
        ("⋅", "*"),
    ]
    normalized = text
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


def _normalize_pseudocode_ascii(text: str) -> str:
    normalized_lines = []
    for raw_line in (text or "").splitlines():
        emit_match = _PSEUDOCODE_EMIT_ASSIGNMENT_RE.match(raw_line)
        if emit_match:
            prefix = raw_line[: emit_match.start("expr")]
            expr = emit_match.group("expr")
            normalized_lines.append(f"{prefix}{_apply_pseudocode_ascii_replacements(expr)}")
            continue
        normalized_lines.append(_apply_pseudocode_ascii_replacements(raw_line))
    return "\n".join(normalized_lines)


def _collect_assignment_sources(code: str) -> dict[str, str]:
    assignment_sources: dict[str, str] = {}
    for raw_line in code.splitlines():
        match = _SIMPLE_ASSIGNMENT_RE.match(raw_line)
        if not match:
            continue
        name = match.group(1)
        expr = match.group(2).strip()
        if name and expr:
            assignment_sources[name] = expr
    return assignment_sources


def _find_undefined_symbol_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    defined = set(_STATIC_SANDBOX_NAMES)
    loaded = set()

    class _NameCollector(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            defined.add(node.name)
            for arg in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            ):
                defined.add(arg.arg)
            if node.args.vararg is not None:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                defined.add(node.args.kwarg.arg)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            defined.add(node.name)
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Param)):
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            self.generic_visit(node)

    _NameCollector().visit(tree)

    undefined = []
    for name in sorted(loaded - defined):
        if _SAFE_IDENTIFIER_RE.fullmatch(name) is None:
            continue
        if name in _STATIC_SANDBOX_NAMES:
            continue
        undefined.append(name)
    return undefined


def _should_convert_degrees(
    key: str,
    expr: str,
    degree_vars: set[str],
    assignment_sources: dict[str, str],
) -> bool:
    if key not in degree_vars:
        return False

    expr_lower = expr.lower()
    if any(token in expr_lower for token in _ANGLE_FUNC_TOKENS):
        return True

    if "solve(" in expr_lower:
        return True

    for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr):
        source = assignment_sources.get(name)
        if not source:
            continue
        source_lower = source.lower()
        if "solve(" in source_lower or any(token in source_lower for token in _ANGLE_FUNC_TOKENS):
            return True

    return False


def _rewrite_result_assignments(code: str, degree_vars: set[str]) -> str:
    assignment_sources = _collect_assignment_sources(code)
    rewritten_lines = []

    for raw_line in code.splitlines():
        float_match = _RESULT_ASSIGN_FLOAT_RE.match(raw_line)
        if float_match:
            indent = float_match.group("indent")
            quote = float_match.group("quote")
            key = float_match.group("key")
            expr = re.sub(r"\s+#.*$", "", float_match.group("expr")).strip()
            convert_degrees = _should_convert_degrees(key, expr, degree_vars, assignment_sources)
            rewritten_lines.append(
                f"{indent}_store_result({quote}{key}{quote}, {expr}, convert_degrees={str(convert_degrees)})"
            )
            continue

        assign_match = _RESULT_ASSIGN_RE.match(raw_line)
        if assign_match:
            indent = assign_match.group("indent")
            quote = assign_match.group("quote")
            key = assign_match.group("key")
            expr = re.sub(r"\s+#.*$", "", assign_match.group("expr")).strip()
            convert_degrees = _should_convert_degrees(key, expr, degree_vars, assignment_sources)
            rewritten_lines.append(
                f"{indent}_store_result({quote}{key}{quote}, {expr}, convert_degrees={str(convert_degrees)})"
            )
            continue

        rewritten_lines.append(raw_line)

    return "\n".join(rewritten_lines)


def _build_translation_prelude(
    degree_vars: set[str], undefined_names: list[str]
) -> str:
    degree_keys_literal = repr(sorted(degree_vars))
    symbol_lines = [f"{name} = sp.symbols('{name}')" for name in undefined_names]
    symbol_block = "\n".join(symbol_lines)
    if symbol_block:
        symbol_block += "\n"
    return (
        f"_AUTO_DEGREE_RESULT_KEYS = set({degree_keys_literal})\n"
        f"{symbol_block}"
        "def _coerce_result_value(value):\n"
        "    try:\n"
        "        if isinstance(value, sp.Basic):\n"
        "            if hasattr(value, 'free_symbols') and value.free_symbols:\n"
        "                return value\n"
        "            if value.is_number:\n"
        "                return float(value.evalf())\n"
        "    except:\n"
        "        pass\n"
        "    try:\n"
        "        return float(value)\n"
        "    except:\n"
        "        return value\n"
        "\n"
        "def _store_result(key, value, convert_degrees=False):\n"
        "    coerced = _coerce_result_value(value)\n"
        "    if convert_degrees and isinstance(coerced, (int, float)):\n"
        "        coerced = math.degrees(float(coerced))\n"
        "    results[key] = coerced\n"
        "    return coerced\n"
        "\n"
        "def _store_emit_expr(key, expr_text, convert_degrees=False):\n"
        "    env = dict(_store_emit_expr.__globals__)\n"
        "    env.update(results)\n"
        "    env['results'] = results\n"
        "    try:\n"
        "        value = sp.sympify(expr_text, locals=env)\n"
        "    except:\n"
        "        try:\n"
        "            value = eval(expr_text, {'__builtins__': {}}, env)\n"
        "        except:\n"
        "            if key in results:\n"
        "                return results[key]\n"
        "            raise\n"
        "    return _store_result(key, value, convert_degrees=convert_degrees)\n"
    )


def _build_emit_injection_block(
    pseudocode: str,
    existing_result_keys: list[str],
    degree_vars: set[str],
) -> str:
    emit_assignments = _extract_emit_assignments_from_pseudocode(pseudocode)
    if not emit_assignments:
        return ""

    lines = [
        "",
        "# Materialize exact [emit] keys from pseudocode.",
        "# Run these writes after translated code so the [emit] contract stays authoritative.",
    ]
    for key, expr in emit_assignments:
        lines.append(
            f"_store_emit_expr({key!r}, {expr!r}, convert_degrees={str(key in degree_vars)})"
        )
    return "\n".join(lines)


def _build_alias_injection_block(
    source_keys: list[str],
    checker_init_contract: list[dict[str, str]],
    preferred_source_keys: list[str] | None = None,
) -> str:
    checker_result_keys = _checker_contract_result_keys(checker_init_contract)
    alias_pairs = _build_result_alias_pairs(
        source_keys,
        checker_result_keys,
        preferred_source_keys=preferred_source_keys,
    )
    if not alias_pairs:
        return ""

    lines = ["", "# Add deterministic aliases for checker compatibility."]
    for source_key, target_key in alias_pairs:
        lines.append(
            f"if {source_key!r} in results and {target_key!r} not in results:"
        )
        lines.append(
            f"    _store_result({target_key!r}, results[{source_key!r}], convert_degrees=False)"
        )
    return "\n".join(lines)


def _sanitize_translated_code(
    pseudocode: str,
    translated_code: str,
    checker_init_contract: list[dict[str, str]],
) -> str:
    code = translated_code.strip()
    if not code:
        return code

    degree_vars = set(_extract_degree_vars_from_pseudocode(pseudocode))
    emit_keys = _extract_emit_keys_from_pseudocode(pseudocode)
    undefined_names = _find_undefined_symbol_names(code)
    rewritten_code = _rewrite_result_assignments(code, degree_vars)
    static_result_keys = extract_result_keys_from_code(rewritten_code)
    emit_block = _build_emit_injection_block(pseudocode, static_result_keys, degree_vars)
    source_keys = list(
        dict.fromkeys(static_result_keys + emit_keys)
    )
    alias_block = _build_alias_injection_block(
        source_keys,
        checker_init_contract,
        preferred_source_keys=emit_keys,
    )
    prelude = _build_translation_prelude(degree_vars, undefined_names)
    return f"{prelude}\n{rewritten_code}{emit_block}{alias_block}".strip()


def _is_translation_failed(translated_code: str) -> bool:
    stripped = translated_code.strip()
    if not stripped or stripped == TRANSLATION_FAILED_TOKEN:
        return True
    try:
        ast.parse(stripped)
    except SyntaxError:
        return True
    return False


def _normalized_weights():
    weights = {
        "checker": CHECKER_WEIGHT,
        "answer": ANSWER_WEIGHT,
        "execution": EXECUTION_WEIGHT,
        "coverage": COVERAGE_WEIGHT,
    }
    total = sum(weights.values())
    if total <= 0:
        return {
            "checker": 0.5,
            "answer": 0.5,
            "execution": 0.0,
            "coverage": 0.0,
        }
    return {name: value / total for name, value in weights.items()}


def _effective_component_weights(
    base_weights: dict[str, float],
    component_available: dict[str, bool],
) -> dict[str, float]:
    masked = {
        name: (value if component_available.get(name, True) else 0.0)
        for name, value in base_weights.items()
    }
    if not RENORMALIZE_AVAILABLE_COMPONENTS:
        return masked

    total = sum(masked.values())
    if total > 0:
        return {name: value / total for name, value in masked.items()}

    fallback = {name: 0.0 for name in base_weights}
    for name in ("checker", "execution", "answer", "coverage"):
        if component_available.get(name, False):
            fallback[name] = 1.0
            break
    return fallback


def _debug(message: str):
    if DEBUG:
        print(f"[checker_reward] {message}")


def _first_batch_trace(message: str):
    if FIRST_BATCH_DEBUG:
        print(f"[FIRSTBATCH][checker_reward][pid={os.getpid()}] {message}", flush=True)


def _resolved_log_path() -> Path | None:
    if not SANDBOX_LOG_PATH:
        return None
    path = Path(SANDBOX_LOG_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _resolved_error_log_path() -> Path | None:
    if not ERROR_LOG_PATH:
        return None
    path = Path(ERROR_LOG_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _append_jsonl_log(log_path: Path | None, payload: dict[str, Any]):
    if log_path is None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with open(log_path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


_seq_counter = itertools.count()
_seq_lock = threading.Lock()


def _append_sandbox_log(payload: dict[str, Any]):
    # (pid, seq) recovers emission order per worker. GRPO scores the n rollouts
    # of one prompt in the same batch, so ordering + prob_id is what lets the
    # analysis rebuild groups and measure within-group reward variance -- verl
    # passes no step index through extra_info.
    with _seq_lock:
        payload.setdefault("seq", next(_seq_counter))
    payload.setdefault("pid", os.getpid())
    _append_jsonl_log(_resolved_log_path(), payload)


def _append_error_log(payload: dict[str, Any]):
    _append_jsonl_log(_resolved_error_log_path(), payload)


def _serialize_exception_chain(exc: BaseException) -> list[dict[str, str]]:
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": str(current),
                "repr": repr(current),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _capture_translator_debug_payload(
    prob_id: str,
    reason: str,
    pseudocode: str,
    checker_init_contract: list[dict[str, str]],
    translated_code: str,
    extracted_code: str,
    direct_code: str,
    translation_failed: bool,
    translation_error: str,
    translation_error_traceback: str = "",
    translation_error_chain: list[dict[str, str]] | None = None,
) -> None:
    translator = _local_translator
    raw_response = ""
    prompt_messages = build_translation_messages(pseudocode, checker_init_contract) if pseudocode else []
    prompt_text = ""
    if translator is not None:
        raw_response = getattr(translator, "_last_raw_response", "") or ""
        prompt_messages = getattr(translator, "_last_prompt_messages", prompt_messages) or prompt_messages
        prompt_text = getattr(translator, "_last_prompt_text", "") or ""

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prob_id": prob_id,
        "reason": reason,
        "translator_backend": CODE_TRANSLATOR_BACKEND,
        "translation_failed": bool(translation_failed),
        "translation_error": translation_error,
        "translation_error_traceback": translation_error_traceback,
        "translation_error_chain": translation_error_chain or [],
        "checker_init_contract": checker_init_contract,
        "checker_vars": _checker_contract_result_keys(checker_init_contract),
        "translator_input_messages": prompt_messages,
        "translator_input_text": prompt_text,
        "translator_output_raw": raw_response,
        "translator_output_extracted": translated_code,
        "extracted_code": extracted_code,
        "direct_code_block": direct_code,
        "pseudocode": pseudocode,
    }
    _append_error_log(payload)


def resolve_prob_id(extra_info: dict[str, Any], gt: dict[str, Any]) -> str:
    for source in (extra_info or {}, gt or {}):
        prob_id = str(source.get("prob_id", "")).strip()
        if prob_id:
            return prob_id
    return ""


def extract_code_block(solution_str: str) -> str:
    explicit_patterns = [
        r"\[code\]\s*```python\n(.*?)\n```",
        r"\[code\]\s*```\n(.*?)\n```",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, solution_str, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Bare ```python blocks are treated as code only when they are not the
    # fenced payload of a [pseudocode] section.
    for match in re.finditer(r"```python\n(.*?)\n```", solution_str, re.DOTALL | re.IGNORECASE):
        prefix = solution_str[:match.start()].rstrip().lower()
        if prefix.endswith("[pseudocode]"):
            continue
        return match.group(1).strip()
    return ""


def extract_pseudocode_block(solution_str: str) -> str:
    """Extract pseudocode from [pseudocode] tags in model output."""
    # Pattern 1: [pseudocode] ```...``` block
    patterns = [
        r"\[pseudocode\]\s*```(?:text|python)?\n(.*?)\n```",
        r"\[pseudocode\]\s*```(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Pattern 2: [pseudocode] raw text until [ans] or end.
    # [subans] may appear inside each sub-question and must not truncate the pseudocode.
    match = re.search(
        r"\[pseudocode\]\s*(.*?)(?=\[ans\]|\Z)",
        solution_str,
        re.DOTALL,
    )
    if match:
        text = match.group(1).strip()
        if text:
            return text

    return ""


def extract_ans_items(solution_str: str) -> list[str]:
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return _parse_answer_block(answer_match.group(1))

    paired_block_pattern = r"\[(ans|answer)\]\s*(.*?)\s*\[\1\]"
    paired_matches = re.findall(paired_block_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if paired_matches:
        return _parse_answer_block(paired_matches[-1][1])

    inline_pattern = r"\[(?:answer|ans)\]\s*(.*?)(?:\n\n|\[[^\]]+\]|$)"
    matches = re.findall(inline_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if not matches:
        fallback_pattern = r"\[(?:answer|ans)\]\s*(.*)"
        matches = re.findall(fallback_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if not matches:
        return []

    return _parse_answer_block(matches[-1])


def _parse_answer_block(raw_text: str) -> list[str]:
    answer_block = raw_text.strip()
    if "," in answer_block and "\n" not in answer_block:
        lines = [part.strip() for part in answer_block.split(",") if part.strip()]
    else:
        lines = [line.strip() for line in answer_block.splitlines() if line.strip()]
    if not lines:
        return []

    numbered = []
    plain = []
    for line in lines:
        stripped = re.sub(r"^\d+\s*[\.\)]\s+", "", line).strip()
        plain.append(stripped)
        if re.match(r"^\d+\s*[\.\)]\s+", line):
            numbered.append(stripped)
    return numbered if numbered else plain


def normalize_answer_text(text: str) -> str:
    return normalize_text(text)


def parse_simple_numeric_answer(text: str):
    normalized = normalize_answer_text(text)
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)", normalized)
    if not match:
        return None

    number_str, unit = match.groups()
    try:
        number = Decimal(number_str)
    except InvalidOperation:
        return None

    return number, number_str, unit


def decimal_places(number_str: str) -> int:
    mantissa = number_str.lower().split("e", 1)[0]
    if "." not in mantissa:
        return 0
    return len(mantissa.split(".", 1)[1])


def round_decimal(number: Decimal, decimals: int) -> Decimal:
    quant = Decimal("1").scaleb(-decimals)
    return number.quantize(quant, rounding=ROUND_HALF_UP)


def _find_unused_exact_answer_match(
    target_text: str,
    pred_items: list[str],
    used_indices: set[int],
) -> int | None:
    target_norm = normalize_answer_text(target_text)
    for idx, pred in enumerate(pred_items):
        if idx in used_indices:
            continue
        if normalize_answer_text(pred) == target_norm:
            return idx
    return None


def _find_unused_relaxed_numeric_match(
    gt_answer: str,
    pred_items: list[str],
    used_indices: set[int],
) -> int | None:
    gt_parsed = parse_simple_numeric_answer(gt_answer)
    if gt_parsed is None:
        return None

    gt_number, gt_number_str, gt_unit = gt_parsed
    gt_decimals = decimal_places(gt_number_str)
    gt_unit_norm = normalize_answer_text(gt_unit)

    for idx, pred in enumerate(pred_items):
        if idx in used_indices:
            continue

        pred_parsed = parse_simple_numeric_answer(pred)
        if pred_parsed is None:
            continue

        pred_number, _, pred_unit = pred_parsed
        if normalize_answer_text(pred_unit) != gt_unit_norm:
            continue

        if gt_decimals == 0:
            if pred_number == gt_number:
                return idx
            continue

        if round_decimal(pred_number, gt_decimals) == round_decimal(gt_number, gt_decimals):
            return idx

    return None


def answer_match_score_from_items(pred_items: list[str], gt_answers: Any) -> float:
    if isinstance(gt_answers, list):
        gold_items = [str(item) for item in gt_answers]
    elif gt_answers is None:
        gold_items = []
    else:
        gold_items = [str(gt_answers)]

    if not gold_items:
        return 0.0

    if not pred_items:
        return 0.0

    matches = 0
    used_indices: set[int] = set()
    for gold in gold_items:
        matched_idx = _find_unused_exact_answer_match(gold, pred_items, used_indices)
        if matched_idx is None:
            matched_idx = _find_unused_relaxed_numeric_match(gold, pred_items, used_indices)
        if matched_idx is not None:
            used_indices.add(matched_idx)
            matches += 1

    return matches / len(gold_items)


@lru_cache(maxsize=None)
def load_checker_code(prob_id: str) -> str | None:
    path = CHECKER_DIR / f"checker_{prob_id}.py"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def checker_pass_score(prob_id: str, solution_str: str) -> tuple[float, str, dict[str, Any]]:
    empty_meta = {
        "passed": False,
        "checker_init_contract": [],
        "checker_vars": [],
        "result_keys": [],
        "computed_results": {},
        "pseudocode": "",
        "translated_code": "",
        "translation_failed": False,
        "translation_error": "",
        "translation_error_traceback": "",
        "translation_error_chain": [],
    }
    if not prob_id:
        return 0.0, "missing_prob_id", empty_meta

    checker_code = load_checker_code(prob_id)
    if not checker_code:
        return 0.0, "missing_checker", empty_meta

    checker_init_contract = extract_checker_init_contract(checker_code)
    checker_vars = _checker_contract_result_keys(checker_init_contract)
    if not checker_vars:
        checker_vars = extract_checker_expected_vars(checker_code)
        checker_init_contract = [
            {"checker_name": key, "result_key": key}
            for key in checker_vars
        ]

    # Try pseudocode path first, fall back to direct code extraction
    pseudocode = _normalize_pseudocode_ascii(extract_pseudocode_block(solution_str))
    direct_code = extract_code_block(solution_str)
    translated_code = ""
    translation_failed = False
    translation_error = ""
    translation_error_traceback = ""
    translation_error_chain: list[dict[str, str]] = []
    if pseudocode:
        try:
            _first_batch_trace(f"checker_pass_score translate START prob_id={prob_id}")
            translated_code = _translate_pseudocode(pseudocode, checker_init_contract)
            _first_batch_trace(f"checker_pass_score translate DONE prob_id={prob_id}")
            if _is_translation_failed(translated_code):
                translation_failed = True
                translated_code = ""
            else:
                translated_code = _sanitize_translated_code(
                    pseudocode,
                    translated_code,
                    checker_init_contract,
                )
                if _is_translation_failed(translated_code):
                    translation_failed = True
                    translated_code = ""
        except Exception as e:
            _debug(f"Code model translation failed for {prob_id}: {e}")
            translation_error = f"{type(e).__name__}: {e}"
            translation_error_traceback = traceback.format_exc()
            translation_error_chain = _serialize_exception_chain(e)
            translated_code = ""

    # Use translated code if available, otherwise fall back to [code] block
    extracted_code = translated_code if translated_code else direct_code
    if extracted_code and not translated_code:
        extracted_code = _sanitize_translated_code(
            pseudocode,
            extracted_code,
            checker_init_contract,
        )
    if not extracted_code:
        empty_meta["checker_init_contract"] = checker_init_contract
        empty_meta["checker_vars"] = checker_vars
        empty_meta["pseudocode"] = pseudocode
        empty_meta["translated_code"] = translated_code
        empty_meta["translation_failed"] = translation_failed
        empty_meta["translation_error"] = translation_error
        empty_meta["translation_error_traceback"] = translation_error_traceback
        empty_meta["translation_error_chain"] = translation_error_chain
        _capture_translator_debug_payload(
            prob_id=prob_id,
            reason="translation_failed" if translation_failed else "missing_code",
            pseudocode=pseudocode,
            checker_init_contract=checker_init_contract,
            translated_code=translated_code,
            extracted_code=extracted_code,
            direct_code=direct_code,
            translation_failed=translation_failed,
            translation_error=translation_error,
            translation_error_traceback=translation_error_traceback,
            translation_error_chain=translation_error_chain,
        )
        if translation_failed:
            return 0.0, "translation_failed", empty_meta
        return 0.0, "missing_code", empty_meta

    result_keys = extract_result_keys_from_code(extracted_code)
    _first_batch_trace(f"checker_pass_score sandbox START prob_id={prob_id} result_keys={len(result_keys)}")
    passed, report = run_physics_sandbox(extracted_code, checker_code)
    _first_batch_trace(f"checker_pass_score sandbox DONE prob_id={prob_id} passed={passed}")
    computed_results = extract_computed_results(report)
    meta = {
        "passed": passed,
        "checker_init_contract": checker_init_contract,
        "checker_vars": checker_vars,
        "result_keys": result_keys,
        "computed_results": computed_results,
        "pseudocode": pseudocode,
        "translated_code": translated_code,
        "translation_failed": translation_failed,
        "translation_error": translation_error,
        "translation_error_traceback": translation_error_traceback,
        "translation_error_chain": translation_error_chain,
    }
    return dense_checker_score(passed, report), report, meta


def compute_reward_breakdown(
    prob_id: str,
    solution_str: str,
    gt_answers: Any,
) -> dict[str, Any]:
    pred_items = extract_ans_items(solution_str)
    answer_ratio = answer_match_score_from_items(pred_items, gt_answers)
    checker_score, checker_report, checker_meta = checker_pass_score(prob_id, solution_str)

    runtime_emitted_vars = list(
        dict.fromkeys(str(key) for key in checker_meta["computed_results"].keys())
    )
    if RUNTIME_COVERAGE_ONLY:
        emitted_vars = runtime_emitted_vars
    else:
        emitted_vars = list(
            dict.fromkeys(checker_meta["result_keys"] + runtime_emitted_vars)
        )

    execution_stage_score = execution_score(checker_meta["passed"], checker_report)
    coverage_score = compute_coverage_score(checker_meta["checker_vars"], emitted_vars)

    if isinstance(gt_answers, list):
        has_gold_answers = bool(gt_answers)
    else:
        has_gold_answers = gt_answers is not None and str(gt_answers).strip() != ""

    answer_available = bool(pred_items) and has_gold_answers
    if answer_available and execution_stage_score < ANSWER_MIN_EXECUTION_SCORE:
        answer_available = False
    if answer_available and ANSWER_REQUIRES_COMPUTED_RESULTS:
        answer_available = bool(checker_meta["computed_results"])

    coverage_available = bool(checker_meta["checker_vars"]) and bool(emitted_vars)
    if coverage_available and execution_stage_score < COVERAGE_MIN_EXECUTION_SCORE:
        coverage_available = False

    if DYNAMIC_CHECKER_AVAILABILITY:
        checker_available = (
            not checker_meta.get("translation_failed", False)
            and execution_stage_score >= CHECKER_MIN_EXECUTION_SCORE
        )
    else:
        checker_available = True

    component_available = {
        "checker": checker_available,
        "answer": answer_available,
        "execution": True,
        "coverage": coverage_available,
    }
    raw_scores = {
        "checker": float(checker_score),
        "answer": float(answer_ratio),
        "execution": float(execution_stage_score),
        "coverage": float(coverage_score),
    }
    base_weights = _normalized_weights()
    effective_weights = _effective_component_weights(
        base_weights, component_available
    )

    if CHECKER_BONUS_MODE:
        # Checker is a pure additive bonus; answer always gets full credit.
        # total = min(1.0, answer_ratio + CHECKER_BONUS_WEIGHT * checker_score)
        total_score = min(1.0, float(answer_ratio) + CHECKER_BONUS_WEIGHT * float(checker_score))
        total_score = max(MIN_REWARD, total_score)
    else:
        total_score = sum(
            effective_weights[name] * raw_scores[name] for name in raw_scores
        )
        total_score = max(MIN_REWARD, min(1.0, total_score))

    return {
        "prob_id": prob_id,
        "pred_items": pred_items,
        "checker_report": checker_report,
        "checker_meta": checker_meta,
        "checker_score": raw_scores["checker"],
        "answer_ratio": raw_scores["answer"],
        "execution_score": raw_scores["execution"],
        "coverage_score": raw_scores["coverage"],
        "raw_scores": raw_scores,
        "component_available": component_available,
        "base_weights": base_weights,
        "effective_weights": effective_weights,
        "emitted_vars": emitted_vars,
        "runtime_emitted_vars": runtime_emitted_vars,
        "total_score": float(total_score),
    }


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    del data_source
    extra_info = extra_info or {}

    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else (ground_truth or {})
    except Exception:
        gt = {}

    prob_id = resolve_prob_id(extra_info, gt)
    _first_batch_trace(f"compute_score START prob_id={prob_id}")
    breakdown = compute_reward_breakdown(prob_id, solution_str, gt.get("answer", []))
    checker_report = breakdown["checker_report"]
    checker_meta = breakdown["checker_meta"]
    total_score = breakdown["total_score"]

    if SANDBOX_LOG_ALL or breakdown["checker_score"] < 1.0:
        log_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prob_id": prob_id,
            "checker_passed": bool(checker_meta["passed"]),
            "translation_failed": bool(checker_meta.get("translation_failed", False)),
            "checker_score": float(breakdown["checker_score"]),
            "execution_score": float(breakdown["execution_score"]),
            "coverage_score": float(breakdown["coverage_score"]),
            "answer_ratio": float(breakdown["answer_ratio"]),
            "expected_var_count": len(checker_meta["checker_vars"]),
            "static_result_key_count": len(checker_meta["result_keys"]),
            "runtime_emitted_var_count": len(breakdown["runtime_emitted_vars"]),
            "emitted_var_count": len(breakdown["emitted_vars"]),
            "component_available": breakdown["component_available"],
            "base_weights": breakdown["base_weights"],
            "effective_weights": breakdown["effective_weights"],
            "total_score": float(total_score),
            "report": checker_report,
        }
        log_payload.update(
            report_diagnostics(
                checker_report,
                bool(checker_meta.get("translation_failed", False)),
            )
        )
        if checker_meta.get("pseudocode"):
            log_payload["pseudocode"] = checker_meta["pseudocode"][:500]
        if checker_meta.get("translated_code"):
            log_payload["translated_code"] = checker_meta["translated_code"][:1000]
        _append_sandbox_log(log_payload)

    _debug(
        f"prob_id={prob_id} checker={breakdown['checker_score']:.3f} "
        f"execution={breakdown['execution_score']:.3f} "
        f"coverage={breakdown['coverage_score']:.3f} "
        f"answer_ratio={breakdown['answer_ratio']:.3f} "
        f"weights={breakdown['effective_weights']} total={total_score:.3f} "
        f"report={checker_report[:200] if isinstance(checker_report, str) else checker_report}"
    )
    _first_batch_trace(f"compute_score DONE prob_id={prob_id} total_score={total_score:.3f}")

    return float(total_score)
