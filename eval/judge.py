"""Answer extraction and LLM-judge scoring (identical to the paper's evaluation).

A response is scored by the fraction of ground-truth sub-answers that the judge matches
inside its final [ans] block; a deterministic numeric pass re-accepts answers that agree
with the ground truth after rounding to its precision. Judge calls that fail 3 times are
marked as skipped and excluded from the averages.
"""

import json
import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from openai import OpenAI

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    repair_json = None


@dataclass
class JudgeConfig:
    model: str = "deepseek/deepseek-v4-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""

    def extra_body(self) -> dict:
        # The default judge runs in non-thinking mode; OpenRouter enables reasoning
        # for this model unless it is switched off explicitly.
        if "openrouter.ai" in self.base_url:
            return {"reasoning": {"enabled": False}}
        return {}


JUDGE = JudgeConfig()
_JUDGE_CLIENT_LOCAL = threading.local()


def configure_judge(model: str, base_url: str, api_key: str) -> None:
    JUDGE.model, JUDGE.base_url, JUDGE.api_key = model, base_url, api_key


def get_thread_local_judge_client() -> OpenAI:
    """One judge client per worker thread for concurrent judging."""
    client = getattr(_JUDGE_CLIENT_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=JUDGE.api_key, base_url=JUDGE.base_url)
        _JUDGE_CLIENT_LOCAL.client = client
    return client


def split_answer_lines(raw_text: str) -> list:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def normalize_answer_text(text: str) -> str:
    """Normalize light LaTeX formatting so simple numeric answers can be compared."""
    normalized = text.strip().strip("$")
    prev = None
    while prev != normalized:
        prev = normalized
        normalized = re.sub(
            r'\\(?:mathrm|mathsf|mathbf|text)\s*{([^{}]*)}',
            r'\1',
            normalized,
        )
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace(" ", "")
    return normalized


def parse_simple_numeric_answer(text: str):
    """Parse answers like 1.6J, 20m/s, 30N into (Decimal, original_number, unit)."""
    normalized = normalize_answer_text(text)
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)', normalized)
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


def find_pred_line_index(target_text: str, pred_lines: list, used_indices: set):
    target_norm = normalize_answer_text(target_text)
    for idx, line in enumerate(pred_lines):
        if idx in used_indices:
            continue
        if normalize_answer_text(line) == target_norm:
            return idx
    return None


def find_relaxed_numeric_match(gt_answer: str, pred_lines: list, used_indices: set):
    """Allow numeric matches that agree after rounding to the GT decimal precision."""
    gt_parsed = parse_simple_numeric_answer(gt_answer)
    if gt_parsed is None:
        return None

    gt_number, gt_number_str, gt_unit = gt_parsed
    gt_decimals = decimal_places(gt_number_str)
    gt_unit_norm = normalize_answer_text(gt_unit)

    for idx, line in enumerate(pred_lines):
        if idx in used_indices:
            continue

        pred_parsed = parse_simple_numeric_answer(line)
        if pred_parsed is None:
            continue

        pred_number, _, pred_unit = pred_parsed
        if normalize_answer_text(pred_unit) != gt_unit_norm:
            continue

        if gt_decimals == 0:
            if pred_number == gt_number:
                return idx, line, (
                    f"Accepted by local numeric match: {line} equals {gt_answer} "
                    f"after normalization."
                )
            continue

        if round_decimal(pred_number, gt_decimals) == round_decimal(gt_number, gt_decimals):
            return idx, line, (
                f"Accepted by local numeric match: {line} rounds to the same value as "
                f"{gt_answer} at {gt_decimals} decimal place(s)."
            )

    return None


def extract_answers_from_response(response: str) -> tuple:
    """
    Extract answers from model response.

    Returns:
        (raw_answer_text, answer_list)
        - raw_answer_text: the raw string inside the answer tags (for judge matching)
        - answer_list: best-effort list of individual answers (for logging only)

    Supports these formats:
      - GRPO format: <answer>...</answer> (one answer per line)
      - Baseline format: [answer] ans1, ans2, ...
      - Pseudocode/reflect format: [ans] ... [ans] (one answer per line)
    """
    # Try <answer> tag first (GRPO format)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        raw_text = answer_match.group(1).strip()
        return raw_text, split_answer_lines(raw_text)

    # Try paired [ans]... [ans] or [answer]... [answer] blocks.
    paired_block_pattern = r'\[(ans|answer)\]\s*(.*?)\s*\[\1\]'
    paired_matches = re.findall(paired_block_pattern, response, re.DOTALL | re.IGNORECASE)
    if paired_matches:
        raw_text = paired_matches[-1][1].strip()
        return raw_text, split_answer_lines(raw_text)

    # Try single [answer] / [ans] tag followed by inline content.
    inline_pattern = r'\[(?:answer|ans)\]\s*(.*?)(?:\n\n|\[[^\]]+\]|$)'
    matches = re.findall(inline_pattern, response, re.DOTALL | re.IGNORECASE)
    if not matches:
        fallback_pattern = r'\[(?:answer|ans)\]\s*(.*)'
        matches = re.findall(fallback_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        raw_text = matches[-1].strip()
        if "," in raw_text and "\n" not in raw_text:
            lines = [a.strip() for a in raw_text.split(',') if a.strip()]
        else:
            lines = split_answer_lines(raw_text)
        return raw_text, lines

    return "", []


def parse_judge_json(result_text: str) -> dict:
    """Parse judge JSON with a repair fallback for malformed escapes in LLM output."""
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        if repair_json is None:
            raise
        repaired = repair_json(result_text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        return json.loads(repaired)


def judge_with_api(client: OpenAI, gt_answers: list, pred_raw_text: str,
                   max_retries: int = 3) -> dict:
    """
    Use the judge API to decide whether each ground-truth sub-answer can be
    found somewhere in the predicted answer text.

    Instead of positional 1:1 matching, we send the full GT list and the raw
    predicted text to the judge, asking it to intelligently match each GT
    answer against the predicted text. This handles skipped or reordered lines.

    After the judge API responds, a deterministic Python pass re-tries every
    rejected GT against any still-unused PRED line for simple numeric matches
    that agree after rounding to the GT decimal precision.

    Args:
        client: OpenAI-compatible judge client.
        gt_answers: List of ground truth answers (one per sub-question).
        pred_raw_text: Raw predicted text extracted from the [ans] block.
        max_retries: Number of API retries.

    Returns:
        {
            "sub_results": [
                {"gt": "...", "matched_pred": "...", "correct": bool, "reason": "..."},
                ...
            ],
            "correct_count": int,
            "total": int,
            "score": float
        }
    """
    n_gt = len(gt_answers)

    if n_gt == 0:
        return {
            "sub_results": [],
            "correct_count": 0,
            "total": 0,
            "score": 0.0,
            "skipped": False,
            "skip_reason": "",
        }

    if not pred_raw_text.strip():
        return {
            "sub_results": [{"gt": str(gt), "matched_pred": "", "correct": False,
                             "reason": "No predicted answer"}
                            for gt in gt_answers],
            "correct_count": 0,
            "total": n_gt,
            "score": 0.0,
            "skipped": False,
            "skip_reason": "",
        }

    # Build the GT list for the prompt
    gt_list_str = "\n".join([f"  {i+1}. {gt}" for i, gt in enumerate(gt_answers)])

    system_prompt = """You are a strict physics answer grading assistant.

You will be given:
1. A list of ground truth (GT) answers for sub-questions (numbered 1, 2, 3, ...).
2. The model's predicted answer text, which may contain multiple answers (one per line, or in other formats). The predicted answers may be fewer, more, or in a different order than the GT.

Your task:
For EACH ground truth answer, search through the entire predicted text and determine if there is a predicted answer that matches it.

Matching rules:
1. The predicted answer must convey the SAME physical quantity and value as the ground truth.
2. Different but equivalent units are acceptable (e.g. "28000W" and "28kW" are the same).
3. Minor formatting differences are acceptable (e.g. "10 m/s" vs "10m/s").
4. LaTeX and plain text representations of the same value are equivalent (e.g. "$5\\mathrm{m/s^2}$" and "5 m/s²").
5. Equivalent mathematical expressions are acceptable (e.g. "√3/2" and "$\\frac{\\sqrt{3}}{2}$").
6. The numerical value must match, but respect the precision shown in the ground truth. If the GT is rounded to N decimal places, accept predictions that round to the same value at N decimal places (e.g. "1.625J" matches "1.6J" to 1 decimal place).
7. If no predicted answer matches the ground truth, mark it as incorrect.
8. Each predicted answer can only match ONE ground truth answer (no double counting).

Output a JSON object with the following structure:
{
    "sub_results": [
        {
            "gt_index": 1,
            "matched_pred": "the predicted text that matches (or empty string if no match)",
            "correct": true or false,
            "reason": "brief explanation"
        },
        ...
    ]
}

You MUST have exactly one entry per GT answer. You MUST output valid JSON only, no other text."""

    user_prompt = f"""Ground truth answers ({n_gt} sub-questions):
{gt_list_str}

Predicted answer text:
\"\"\"
{pred_raw_text}
\"\"\""""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1024,
                extra_body=JUDGE.extra_body(),
            )

            result_text = response.choices[0].message.content
            result = parse_judge_json(result_text)

            sub_results = result.get("sub_results", [])

            full_sub_results = []
            for i, gt in enumerate(gt_answers):
                judge_entry = None
                for sr in sub_results:
                    if sr.get("gt_index") == i + 1:
                        judge_entry = sr
                        break

                is_correct = judge_entry.get("correct", False) if judge_entry else False
                matched_pred = judge_entry.get("matched_pred", "") if judge_entry else ""
                reason = judge_entry.get("reason", "No judge result") if judge_entry else "No judge result"

                full_sub_results.append({
                    "gt": str(gt),
                    "matched_pred": str(matched_pred),
                    "correct": is_correct,
                    "reason": reason,
                })

            # Track which pred lines the judge already claimed, so the local
            # numeric fallback never double-uses a line (rule 8).
            pred_lines = split_answer_lines(pred_raw_text)
            used_indices = set()
            for sr in full_sub_results:
                if not sr["matched_pred"]:
                    continue
                matched_idx = find_pred_line_index(sr["matched_pred"], pred_lines, used_indices)
                if matched_idx is not None:
                    used_indices.add(matched_idx)

            # Python fallback for simple numeric answers using v1's rounding rule.
            for sr in full_sub_results:
                if sr["correct"]:
                    continue
                relaxed_match = find_relaxed_numeric_match(sr["gt"], pred_lines, used_indices)
                if relaxed_match is None:
                    continue
                matched_idx, matched_pred, reason = relaxed_match
                used_indices.add(matched_idx)
                sr["matched_pred"] = matched_pred
                sr["correct"] = True
                sr["reason"] = reason

            correct_count = sum(1 for sr in full_sub_results if sr["correct"])
            score = correct_count / n_gt

            return {
                "sub_results": full_sub_results,
                "correct_count": correct_count,
                "total": n_gt,
                "score": score,
                "skipped": False,
                "skip_reason": "",
            }

        except Exception as e:
            print(f"  Judge API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # All retries failed: mark as skipped so transient judge issues do not hurt scores.
    skip_reason = f"Judge API/JSON parsing failed after {max_retries} attempts"
    print(f"  WARNING: {skip_reason}, skipping this sample.")
    return {
        "sub_results": [{"gt": str(gt), "matched_pred": "", "correct": False,
                         "reason": skip_reason}
                        for gt in gt_answers],
        "correct_count": None,
        "total": n_gt,
        "score": None,
        "skipped": True,
        "skip_reason": skip_reason,
    }


def evaluate_sample_response(gt_answers: list, response: str) -> dict:
    """Extract answers and run the judge API for a single model response."""
    pred_raw_text, pred_list = extract_answers_from_response(response)

    try:
        client = get_thread_local_judge_client()
        eval_result = judge_with_api(client, gt_answers, pred_raw_text)
    except Exception as e:  # pragma: no cover - defensive fallback
        skip_reason = f"Unexpected judge worker error: {e}"
        print(f"  WARNING: {skip_reason}")
        eval_result = {
            "sub_results": [{"gt": str(gt), "matched_pred": "", "correct": False,
                             "reason": skip_reason}
                            for gt in gt_answers],
            "correct_count": None,
            "total": len(gt_answers),
            "score": None,
            "skipped": True,
            "skip_reason": skip_reason,
        }

    return {
        "response": response,
        "predicted_raw_text": pred_raw_text,
        "predicted_answers": pred_list,
        "has_answer_tag": bool(pred_raw_text),
        "evaluation": eval_result,
    }

