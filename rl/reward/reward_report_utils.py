from __future__ import annotations

import ast
import keyword
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


_FATAL_REPORT_MARKERS = (
    "missing_prob_id",
    "missing_checker",
    "missing_code",
    "[code execution failed",
    "[checker execution error]",
    "[checker timeout]",
    "[sandbox timeout]",
    "[sandbox error]",
)

_NEUTRAL_PREFIXES = (
    "===",
    "---",
    ">>> computed results:",
    "[alignment:",
    "traceback",
    "diagnosis:",
    "computed results:",
)

_SUMMARY_PREFIXES = (
    "overall",
    "summary",
    "result:",
    "total tests",
    "passed:",
    "failed:",
    "some tests failed",
    "some tests passed",
    "monotonicity test completed",
)

_SECTION_WEIGHTS = {
    "invariants": 1.0,
    "trace": 0.7,
    "monotonicity": 0.4,
    "default": 0.7,
}

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_BAD_CONTRACT_KEYS = {
    "results",
    "result",
    "inputs",
    "input",
    "steps",
    "step",
    "self",
    "trace",
    "params",
    "solution",
    "solution_instance",
    "none",
    "null",
    "n/a",
    "na",
}

_COMPUTED_RESULTS_FLOOR = float(
    os.environ.get("CHECKER_REWARD_CHECKER_COMPUTED_RESULTS_FLOOR", "0.0")
)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    # Strip list numbering like "1. answer" or "2) answer", but preserve
    # decimal values such as "1.6J".
    text = re.sub(r"^\d+\s*[\.\)]\s+", "", text)
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("−", "-")
    text = re.sub(r"\s+", "", text)
    text = text.strip(".,;")
    return text


def _detect_section(line: str) -> Optional[str]:
    lowered = line.lower()
    if "monotonicity" in lowered:
        return "monotonicity"
    if "trace consistency" in lowered:
        return "trace"
    if "invariant" in lowered or "physics invariant" in lowered:
        return "invariants"
    return None


def _parse_structured_outcome(line: str) -> Optional[bool]:
    try:
        obj = ast.literal_eval(line)
    except Exception:
        return None

    if isinstance(obj, dict) and isinstance(obj.get("success"), bool):
        return obj["success"]
    if isinstance(obj, tuple) and len(obj) >= 2 and isinstance(obj[1], bool):
        return obj[1]
    return None


def _classify_outcome_line(line: str) -> Optional[bool]:
    stripped = line.strip()
    if not stripped:
        return None

    structured = _parse_structured_outcome(stripped)
    if structured is not None:
        return structured

    lowered = stripped.lower()
    if any(marker in lowered for marker in _FATAL_REPORT_MARKERS):
        return None
    if any(lowered.startswith(prefix) for prefix in _NEUTRAL_PREFIXES):
        return None
    if any(lowered.startswith(prefix) for prefix in _SUMMARY_PREFIXES):
        return None
    if "skipped" in lowered:
        return None

    if lowered.startswith(("[pass]", "pass:", "pass ", "passed:", "✓")):
        return True
    if lowered.startswith(("[fail]", "fail:", "fail ", "failed:", "✗")):
        return False

    if re.search(r"\bpassed\b", lowered):
        return True
    if re.search(r"\bfailed\b", lowered):
        return False
    return None


def _extract_string_literal(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _is_results_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "results"
    )


def _is_results_getattr_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
        return False
    if len(node.args) < 2:
        return False
    return _extract_string_literal(node.args[1]) == "results"


def _is_result_store_expr(node: ast.AST, aliases: set[str]) -> bool:
    return (
        (isinstance(node, ast.Name) and node.id in aliases)
        or _is_results_attr(node)
        or _is_results_getattr_call(node)
    )


def _add_result_aliases_from_target(target: ast.AST, aliases: set[str]) -> bool:
    if isinstance(target, ast.Name) and target.id not in aliases:
        aliases.add(target.id)
        return True
    return False


def _collect_result_aliases(nodes: List[ast.AST], initial_aliases: Optional[set[str]] = None) -> set[str]:
    aliases = set(initial_aliases or set())
    aliases.add("results")
    module = ast.Module(body=nodes, type_ignores=[])

    changed = True
    while changed:
        changed = False
        for node in ast.walk(module):
            if isinstance(node, ast.Assign) and _is_result_store_expr(node.value, aliases):
                for target in node.targets:
                    changed |= _add_result_aliases_from_target(target, aliases)
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and _is_result_store_expr(node.value, aliases)
            ):
                changed |= _add_result_aliases_from_target(node.target, aliases)

    return aliases


def _node_reads_results(node: ast.AST, result_aliases: set[str]) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and _is_result_store_expr(func.value, result_aliases)
        ):
            return True
    if isinstance(node, ast.Subscript) and _is_result_store_expr(node.value, result_aliases):
        return True
    return False


def _collect_result_helper_names(tree: ast.AST, result_aliases: set[str]) -> set[str]:
    helper_names: set[str] = set()
    function_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

    for func in function_nodes:
        local_result_aliases = _collect_result_aliases(list(func.body), initial_aliases=result_aliases)
        if any(_node_reads_results(node, local_result_aliases) for node in ast.walk(func)):
            helper_names.add(func.name)

    return helper_names


def _extract_ast_checker_keys(checker_code: str) -> List[str]:
    try:
        tree = ast.parse(checker_code)
    except SyntaxError:
        return []

    result_aliases = _collect_result_aliases(list(tree.body))
    helper_names = _collect_result_helper_names(tree, result_aliases)

    seen = set()
    extracted: List[str] = []

    def _add_key(key: Optional[str]) -> None:
        if not key:
            return
        cleaned = str(key).strip()
        if not _is_contract_friendly_result_key(cleaned):
            return
        if cleaned not in seen:
            seen.add(cleaned)
            extracted.append(cleaned)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func

            if isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2:
                key = _extract_string_literal(node.args[1])
                if key not in {"inputs", "results"}:
                    _add_key(key)

            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and _is_result_store_expr(func.value, result_aliases)
                and node.args
            ):
                _add_key(_extract_string_literal(node.args[0]))

            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "self" and func.attr in helper_names and node.args:
                    _add_key(_extract_string_literal(node.args[0]))

            if isinstance(func, ast.Name) and func.id in helper_names and node.args:
                _add_key(_extract_string_literal(node.args[0]))

        elif isinstance(node, ast.Subscript) and _is_result_store_expr(node.value, result_aliases):
            slice_node = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            _add_key(_extract_string_literal(slice_node))

    return extracted


def _extract_result_keys_from_expr(
    expr: ast.AST,
    *,
    result_aliases: set[str],
    helper_names: set[str],
    local_sources: Dict[str, List[str]],
) -> List[str]:
    if expr is None:
        return []

    if isinstance(expr, ast.Name):
        return list(local_sources.get(expr.id, []))

    if isinstance(expr, ast.Subscript) and _is_result_store_expr(expr.value, result_aliases):
        slice_node = expr.slice.value if isinstance(expr.slice, ast.Index) else expr.slice
        key = _extract_string_literal(slice_node)
        return [key] if key else []

    if isinstance(expr, ast.Call):
        func = expr.func

        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and _is_result_store_expr(func.value, result_aliases)
            and expr.args
        ):
            key = _extract_string_literal(expr.args[0])
            return [key] if key else []

        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "self" and func.attr in helper_names:
                if expr.args:
                    key = _extract_string_literal(expr.args[0])
                    if key:
                        return [key]
                for kw in expr.keywords:
                    if kw.arg == "key":
                        key = _extract_string_literal(kw.value)
                        if key:
                            return [key]

        if isinstance(func, ast.Name) and func.id in helper_names:
            if expr.args:
                key = _extract_string_literal(expr.args[0])
                if key:
                    return [key]
            for kw in expr.keywords:
                if kw.arg == "key":
                    key = _extract_string_literal(kw.value)
                    if key:
                        return [key]

        keys: List[str] = []
        for child in list(expr.args) + [kw.value for kw in expr.keywords]:
            keys.extend(
                _extract_result_keys_from_expr(
                    child,
                    result_aliases=result_aliases,
                    helper_names=helper_names,
                    local_sources=local_sources,
                )
            )
        return list(dict.fromkeys(keys))

    keys: List[str] = []
    for child in ast.iter_child_nodes(expr):
        keys.extend(
            _extract_result_keys_from_expr(
                child,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=local_sources,
            )
        )
    return list(dict.fromkeys(keys))


def _merge_local_key_sources(*source_maps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for source_map in source_maps:
        for name, keys in source_map.items():
            bucket = merged.setdefault(name, [])
            for key in keys:
                if key not in bucket:
                    bucket.append(key)
    return merged


def _populate_local_result_sources(
    statements: List[ast.stmt],
    *,
    result_aliases: set[str],
    helper_names: set[str],
    local_sources: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    current = {name: list(keys) for name, keys in local_sources.items()}

    for stmt in statements:
        if isinstance(stmt, ast.Assign):
            keys = _extract_result_keys_from_expr(
                stmt.value,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=current,
            )
            if keys:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        current[target.id] = list(dict.fromkeys(keys))
        elif isinstance(stmt, ast.AnnAssign):
            keys = _extract_result_keys_from_expr(
                stmt.value,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=current,
            )
            if keys and isinstance(stmt.target, ast.Name):
                current[stmt.target.id] = list(dict.fromkeys(keys))
        elif isinstance(stmt, ast.If):
            body_sources = _populate_local_result_sources(
                stmt.body,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=current,
            )
            orelse_sources = _populate_local_result_sources(
                stmt.orelse,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=current,
            )
            current = _merge_local_key_sources(current, body_sources, orelse_sources)
        elif isinstance(stmt, ast.Try):
            body_sources = _populate_local_result_sources(
                stmt.body,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=current,
            )
            handler_sources = [
                _populate_local_result_sources(
                    handler.body,
                    result_aliases=result_aliases,
                    helper_names=helper_names,
                    local_sources=current,
                )
                for handler in stmt.handlers
            ]
            orelse_sources = _populate_local_result_sources(
                stmt.orelse,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=current,
            )
            finalbody_sources = _populate_local_result_sources(
                stmt.finalbody,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=current,
            )
            current = _merge_local_key_sources(
                current,
                body_sources,
                orelse_sources,
                finalbody_sources,
                *handler_sources,
            )

    return current


def extract_checker_init_contract(checker_code: str) -> List[Dict[str, str]]:
    try:
        tree = ast.parse(checker_code)
    except SyntaxError:
        return []

    physics_checker = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PhysicsChecker":
            physics_checker = node
            break
    if physics_checker is None:
        return []

    init_func = None
    for node in physics_checker.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_func = node
            break
    if init_func is None:
        return []

    result_aliases = _collect_result_aliases(list(init_func.body))
    helper_names = _collect_result_helper_names(tree, result_aliases)
    local_sources: Dict[str, List[str]] = {}
    previous_local_sources: Optional[Dict[str, List[str]]] = None

    while previous_local_sources != local_sources:
        previous_local_sources = {name: list(keys) for name, keys in local_sources.items()}
        local_sources = _populate_local_result_sources(
            list(init_func.body),
            result_aliases=result_aliases,
            helper_names=helper_names,
            local_sources=local_sources,
        )

    seen_pairs = set()
    contract: List[Dict[str, str]] = []

    def _add_pair(checker_name: str, result_key: str) -> None:
        cleaned_checker = str(checker_name).strip()
        cleaned_key = str(result_key).strip()
        if not cleaned_checker or not cleaned_key:
            return
        if not _is_contract_friendly_checker_name(cleaned_checker):
            return
        if not _is_contract_friendly_result_key(cleaned_key):
            return
        pair = (cleaned_checker, cleaned_key)
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        contract.append({"checker_name": cleaned_checker, "result_key": cleaned_key})

    for node in ast.walk(init_func):
        if isinstance(node, ast.Assign):
            keys = _extract_result_keys_from_expr(
                node.value,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=local_sources,
            )
            if not keys:
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    for key in keys:
                        _add_pair(target.attr, key)
        elif isinstance(node, ast.AnnAssign):
            keys = _extract_result_keys_from_expr(
                node.value,
                result_aliases=result_aliases,
                helper_names=helper_names,
                local_sources=local_sources,
            )
            if not keys:
                continue
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                for key in keys:
                    _add_pair(target.attr, key)

    return contract


def _extract_outcome_records(report_text: str) -> List[Tuple[bool, float]]:
    records = []
    current_section = None
    for raw_line in report_text.splitlines():
        section = _detect_section(raw_line)
        if section is not None:
            current_section = section

        outcome = _classify_outcome_line(raw_line)
        if outcome is not None:
            line_section = section or current_section or "default"
            weight = _SECTION_WEIGHTS.get(line_section, _SECTION_WEIGHTS["default"])
            records.append((outcome, weight))
    return records


def extract_computed_results(report: Any) -> Dict[str, Any]:
    report_text = "" if report is None else str(report)
    for raw_line in report_text.splitlines():
        if ">>> Computed results:" not in raw_line:
            continue
        payload = raw_line.split(">>> Computed results:", 1)[1].strip()
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return None
        return number if math.isfinite(number) else None

    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except Exception:
        match = _NUMBER_RE.search(text)
        if match is None:
            return None
        try:
            number = float(match.group(0))
        except Exception:
            return None
        return number if math.isfinite(number) else None


_ERROR_TYPE_MARKERS = (
    "[sandbox timeout]", "[checker timeout]", "[sandbox error]",
    "[checker execution error]", "[code execution failed",
    "syntaxerror", "nameerror", "typeerror", "attributeerror",
    "indexerror", "keyerror", "valueerror", "zerodivisionerror",
    "overflowerror", "recursionerror", "importerror", "assertionerror",
)


def report_diagnostics(report: Any, translation_failed: bool = False) -> Dict[str, Any]:
    """Per-sample checker diagnostics for supervision-density analysis.

    Exposes what `dense_checker_score` already computes internally but discards:
    how many individual test outcomes the checker actually emitted, how many
    passed, and — when the score is 0 — whether that came from a real test
    failure or from the checker never running at all. Reported per sample so
    execution-failure rate and reward sparsity can be measured over a run
    rather than inferred from the scalar reward.
    """
    report_text = "" if report is None else str(report)
    lowered = report_text.lower().strip()
    records = _extract_outcome_records(report_text)
    n_tests = len(records)
    n_passed = sum(1 for outcome, _ in records if outcome)
    weight_total = sum(weight for _, weight in records)
    weight_passed = sum(weight for outcome, weight in records if outcome)

    error_type = ""
    for marker in _ERROR_TYPE_MARKERS:
        if marker in lowered:
            error_type = marker.strip("[]").strip()
            break

    if translation_failed or lowered == "translation_failed":
        stage = "translation_failed"
    elif lowered in ("missing_prob_id", "missing_checker", "missing_code"):
        stage = lowered
    elif any(marker in lowered for marker in _FATAL_REPORT_MARKERS):
        stage = "sandbox_error"
    elif n_tests == 0:
        stage = "no_tests_emitted"
    elif n_passed == n_tests:
        stage = "ok"
    else:
        stage = "tests_failed"

    # "Ran" means the checker produced at least one real test outcome; anything
    # else is an execution failure, not evidence about the solution.
    return {
        "n_tests": n_tests,
        "n_tests_passed": n_passed,
        "test_weight_total": round(weight_total, 6),
        "test_weight_passed": round(weight_passed, 6),
        "failure_stage": stage,
        "error_type": error_type,
        "checker_ran": n_tests > 0,
    }


def execution_score(passed: bool, report: Any) -> float:
    if passed:
        return 1.0

    report_text = "" if report is None else str(report)
    lowered = report_text.lower().strip()
    if not lowered:
        return 0.0
    if any(marker == lowered for marker in ("missing_prob_id", "missing_checker", "missing_code")):
        return 0.0

    has_results = bool(extract_computed_results(report_text))
    has_outcomes = bool(_extract_outcome_records(report_text))
    if has_results and has_outcomes:
        return 0.8
    if has_results:
        return 0.6
    if has_outcomes:
        return 0.5

    if "syntaxerror" in lowered:
        return 0.05
    if "nameerror" in lowered:
        return 0.1
    if "indexerror" in lowered:
        return 0.2
    if "attributeerror" in lowered or "typeerror" in lowered:
        return 0.15
    if "zerodivisionerror" in lowered:
        return 0.25
    if "[sandbox timeout]" in lowered or "[checker timeout]" in lowered:
        return 0.0
    if (
        "[checker execution error]" in lowered
        or "[checker runtime error]" in lowered
        or "[interface match failed]" in lowered
        or "[format/symbolic value error]" in lowered
        or "[checker zerodivisionerror]" in lowered
    ):
        return 0.45
    if "[code execution failed" in lowered:
        return 0.08
    return 0.0


def dense_checker_score(passed: bool, report: Any) -> float:
    if passed:
        return 1.0

    report_text = "" if report is None else str(report)
    lowered = report_text.lower().strip()
    if not lowered:
        return 0.0
    if any(marker in lowered for marker in _FATAL_REPORT_MARKERS):
        return 0.0

    outcome_records = _extract_outcome_records(report_text)
    if outcome_records:
        total_weight = sum(weight for _, weight in outcome_records)
        if total_weight > 0:
            return sum(weight for outcome, weight in outcome_records if outcome) / total_weight

    if _COMPUTED_RESULTS_FLOOR > 0.0 and (
        extract_computed_results(report_text) or "[alignment:" in lowered
    ):
        return _COMPUTED_RESULTS_FLOOR
    return 0.0


def extract_checker_expected_vars(checker_code: str) -> List[str]:
    contract = extract_checker_init_contract(checker_code)
    if contract:
        seen = set()
        result = []
        for item in contract:
            key = str(item.get("result_key", "")).strip()
            if not _is_contract_friendly_result_key(key):
                continue
            if key not in seen:
                seen.add(key)
                result.append(key)
        if result:
            return result

    patterns = (
        r"""safe_get\(\s*['"]([^'"]+)['"]""",
        r"""_safe_get\(\s*['"]([^'"]+)['"]""",
        r"""_safe_get_result\(\s*['"]([^'"]+)['"]""",
        r"""_get_result\(\s*['"]([^'"]+)['"]""",
        r"""_get_value\(\s*['"]([^'"]+)['"]""",
        r"""_safe_extract\(\s*['"]([^'"]+)['"]""",
        r"""_get_numeric\(\s*['"]([^'"]+)['"]""",
        r"""_safe_get_numeric\(\s*['"]([^'"]+)['"]""",
        r"""_safe_get_float\(\s*['"]([^'"]+)['"]""",
        r"""getattr\(\s*solution_instance\s*,\s*['"]([^'"]+)['"]""",
        r"""[A-Za-z_][A-Za-z0-9_]*\.get\(\s*['"]([^'"]+)['"]""",
        r"""results\.get\(\s*['"]([^'"]+)['"]""",
        r"""self\.results\.get\(\s*['"]([^'"]+)['"]""",
        r"""results\s*\[\s*['"]([^'"]+)['"]\s*\]""",
    )
    seen = set()
    result = []
    for match in _extract_ast_checker_keys(checker_code):
        if match not in seen:
            seen.add(match)
            result.append(match)
    for pattern in patterns:
        for match in re.findall(pattern, checker_code):
            cleaned = str(match).strip()
            if not _is_contract_friendly_result_key(cleaned):
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
    return result


def _is_contract_friendly_text(key: str, *, allow_keywords: bool) -> bool:
    if not key:
        return False

    cleaned = unicodedata.normalize("NFKC", key).strip()
    lowered = cleaned.lower()
    if lowered in _BAD_CONTRACT_KEYS:
        return False
    if not allow_keywords and keyword.iskeyword(cleaned):
        return False
    if any(ch in cleaned for ch in ("\n", "\r", "\t")):
        return False
    if len(cleaned) > 80:
        return False
    if not any(ch.isalnum() for ch in cleaned):
        return False

    return True


def _is_contract_friendly_result_key(key: str) -> bool:
    return _is_contract_friendly_text(key, allow_keywords=True)


def _is_contract_friendly_checker_name(key: str) -> bool:
    return _is_contract_friendly_text(key, allow_keywords=False)


def extract_result_keys_from_code(generated_code: str) -> List[str]:
    seen = set()
    result = []
    patterns = (
        r"""results\s*\[\s*['"]([^'"]+)['"]\s*\]""",
        r"""_store_result\(\s*['"]([^'"]+)['"]\s*,""",
    )
    for pattern in patterns:
        for match in re.findall(pattern, generated_code):
            if match not in seen:
                seen.add(match)
                result.append(match)
    return result


def _fuzzy_key_match(expected: str, emitted_set: set) -> bool:
    """Check if expected key matches any emitted key, allowing near-matches."""
    if expected in emitted_set:
        return True
    # Try substring containment (e.g. "velocity_final" matches "v_final" style)
    for emitted in emitted_set:
        # One is a substring of the other
        if (expected in emitted or emitted in expected) and min(len(expected), len(emitted)) >= 3:
            return True
        # Underscore/case-insensitive normalization: compare without underscores
        if expected.replace("_", "") == emitted.replace("_", ""):
            return True
    return False


def compute_coverage_score(expected_vars: List[str], emitted_vars: List[str]) -> float:
    normalized_expected = []
    seen_expected = set()
    for key in expected_vars:
        normalized = normalize_text(str(key))
        if normalized and normalized not in seen_expected:
            seen_expected.add(normalized)
            normalized_expected.append(normalized)

    if not normalized_expected:
        return 0.0

    normalized_emitted = {normalize_text(str(key)) for key in emitted_vars if normalize_text(str(key))}
    hits = sum(1 for key in normalized_expected if _fuzzy_key_match(key, normalized_emitted))
    return hits / len(normalized_expected)


def _add_result_candidates(key: str, value: Any, text_candidates: set, numeric_candidates: List[float]) -> None:
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            nested_key = f"{key}.{sub_key}" if key else str(sub_key)
            _add_result_candidates(nested_key, sub_value, text_candidates, numeric_candidates)
        return

    if isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value):
            nested_key = f"{key}[{idx}]"
            _add_result_candidates(nested_key, item, text_candidates, numeric_candidates)
        return

    key_text = str(key)
    value_text = str(value)
    for candidate in (value_text, f"{key_text}={value_text}", f"{key_text}: {value_text}"):
        normalized = normalize_text(candidate)
        if normalized:
            text_candidates.add(normalized)

    numeric_value = _coerce_float(value)
    if numeric_value is not None:
        numeric_candidates.append(numeric_value)
        pretty = f"{numeric_value:g}"
        for candidate in (pretty, f"{key_text}={pretty}", f"{key_text}: {pretty}"):
            normalized = normalize_text(candidate)
            if normalized:
                text_candidates.add(normalized)
        rounded = round(numeric_value)
        if math.isclose(numeric_value, rounded, rel_tol=0.0, abs_tol=1e-9):
            normalized = normalize_text(str(int(rounded)))
            if normalized:
                text_candidates.add(normalized)


def answer_code_consistency_score(pred_items: List[str], computed_results: Dict[str, Any]) -> float:
    if not pred_items or not computed_results:
        return 0.0

    text_candidates = set()
    numeric_candidates: List[float] = []
    for key, value in computed_results.items():
        _add_result_candidates(str(key), value, text_candidates, numeric_candidates)

    if not text_candidates and not numeric_candidates:
        return 0.0

    matches = 0
    for pred in pred_items:
        pred_norm = normalize_text(pred)
        if not pred_norm:
            continue
        if pred_norm in text_candidates:
            matches += 1
            continue
        pred_numeric = _coerce_float(pred)
        if pred_numeric is not None and any(
            math.isclose(pred_numeric, candidate, rel_tol=1e-2, abs_tol=1e-6) for candidate in numeric_candidates
        ):
            matches += 1

    return matches / len(pred_items)
