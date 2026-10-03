import math
import traceback
from typing import Any, Dict, List, Tuple, Optional
import sympy as sp

class PhysicsChecker:
    """
    Auditor for the WKB validity condition and perturbation theory
    for the ground state energy of an atom in an electric field.
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction from solution_instance
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = []

        self.givens = getattr(solution_instance, 'givens', {})

        # Extract expected result keys with None defaults
        self.validity_parameter = self.results.get('validity_parameter', None)
        self.E0_second = self.results.get('E0_second', None)

        # Extract optional numeric givens for monotonicity (may be None)
        self.hbar = self.givens.get('hbar', None)
        self.m = self.givens.get('m', None)
        self.E = self.givens.get('E', None)
        self.Vx = self.givens.get('Vx', None)   # V(x) at x

    # ------------------------------------------------------------------
    # Interface tests – required keys exist and are of acceptable type
    # ------------------------------------------------------------------
    def test_interface_validity_parameter_exists(self) -> Tuple[bool, str]:
        if self.validity_parameter is None:
            return (False, "SKIP: 'validity_parameter' not found in results")
        if not isinstance(self.validity_parameter, (str, int, float, complex)):
            return (False, f"Invalid type for validity_parameter: {type(self.validity_parameter)}")
        return (True, "validity_parameter exists and has valid type")

    def test_interface_E0_second_exists(self) -> Tuple[bool, str]:
        if self.E0_second is None:
            return (False, "SKIP: 'E0_second' not found in results")
        if not isinstance(self.E0_second, (str, int, float, complex)):
            return (False, f"Invalid type for E0_second: {type(self.E0_second)}")
        return (True, "E0_second exists and has valid type")

    def test_interface_explanation_steps_present(self) -> Tuple[bool, str]:
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps is empty or missing")
        return (True, "explanation_steps is present and non-empty")

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_validity_condition_expression_form(self) -> Tuple[bool, str]:
        """Check that the validity parameter expression contains hbar, p, and dp/dx or m and dV/dx."""
        if self.validity_parameter is None:
            return (False, "SKIP: validity_parameter not available")

        expr_str = str(self.validity_parameter)
        # Look for essential symbols
        has_hbar = 'ħ' in expr_str or 'hbar' in expr_str or 'h' in expr_str
        has_p = 'p' in expr_str
        has_dp_dx = 'dp/dx' in expr_str or 'dp/dx' in expr_str or 'dpdx' in expr_str
        # Accept alternative: m hbar / p^3 dV/dx
        has_m = 'm' in expr_str
        has_dV_dx = 'dV/dx' in expr_str or 'dV/dx' in expr_str

        # At least one of the two forms must be present
        condition1 = has_hbar and has_p and has_dp_dx
        condition2 = has_hbar and has_m and has_dV_dx

        if condition1 or condition2:
            return (True, "Validity parameter expression contains expected physical quantities")
        else:
            return (False, f"Validity parameter expression '{expr_str}' lacks required components (hbar, p, dp/dx or m, dV/dx)")

    def test_invariant_turning_point_failure(self) -> Tuple[bool, str]:
        """Check that explanation acknowledges failure at turning point (p=0)."""
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps missing")

        combined = "\n".join(str(step) for step in self.explanation_steps)
        keywords = ['turning point', 'p=0', 'p → 0', 'p->0', 'p→0',
                     'diverg', 'singular', 'fail', 'invalid', 'not valid',
                     'denominator zero', 'p^2 denominator']
        found = any(kw in combined.lower() for kw in keywords)
        if found:
            return (True, "Explanation mentions failure at turning point")
        else:
            return (False, "Explanation does not mention turning-point failure")

    def test_invariant_second_order_negative(self) -> Tuple[bool, str]:
        """Check that E0_second is reported as negative (<0)."""
        if self.E0_second is None:
            return (False, "SKIP: E0_second not available")

        # If numeric, check sign
        if isinstance(self.E0_second, (int, float)):
            if self.E0_second < 0:
                return (True, f"E0_second = {self.E0_second} < 0")
            else:
                return (False, f"E0_second must be negative, got {self.E0_second}")

        # If string, look for negative indicator
        expr_str = str(self.E0_second)
        if '<0' in expr_str or '< 0' in expr_str or 'negative' in expr_str.lower() or '-' in expr_str:
            if 'negative' in expr_str.lower() or '<0' in expr_str:
                return (True, f"E0_second stated as negative: '{expr_str}'")
            if expr_str.strip().startswith('-') and len(expr_str.strip()) > 1:
                return (True, f"E0_second given with explicit minus sign: '{expr_str}'")
            return (True, f"E0_second indicates negativity: '{expr_str}'")
        else:
            return (False, f"E0_second does not indicate negative sign: '{expr_str}'")

    # ------------------------------------------------------------------
    # Modeling/trace tests – check derivation steps
    # ------------------------------------------------------------------
    def test_modeling_dp_dx_derivation(self) -> Tuple[bool, str]:
        """Check that explanation contains the chain rule for dp/dx."""
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps missing")

        steps_text = "\n".join(str(s) for s in self.explanation_steps)
        needle1 = "dp/dx" in steps_text or "dp/dx" in steps_text
        needle2 = "-m/p dV/dx" in steps_text or "- (m/p) dV/dx" in steps_text
        needle3 = "−(m/p) dV/dx" in steps_text
        if needle1 and (needle2 or needle3):
            return (True, "Derivation of dp/dx is present")
        if "chain rule" in steps_text.lower() and "dp/dx" in steps_text:
            return (True, "dp/dx derivation via chain rule mentioned")
        return (False, "dp/dx derivation not found in explanation steps")

    def test_modeling_parity_argument(self) -> Tuple[bool, str]:
        """Check that explanation uses parity to claim first order vanishes."""
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps missing")

        steps_text = "\n".join(str(s) for s in self.explanation_steps)
        key_phrases = ['odd', 'parity', 'diagonal element', 'E0_first = 0',
                       'first order', '<0|H\'|0>', '<0|H′|0>']
        count = sum(phrase in steps_text.lower() or phrase in steps_text for phrase in key_phrases)
        if count >= 2:
            return (True, "Parity argument for vanishing first order present")
        else:
            return (False, f"Insufficient parity argument in explanation steps (found {count} indicators)")

    def test_modeling_second_order_sum_rationale(self) -> Tuple[bool, str]:
        """Check that the reason for E0.second < 0 is given (denominator negative)."""
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps missing")

        steps_text = "\n".join(str(s) for s in self.explanation_steps)
        keywords = ['E_0 - E_n < 0', 'E0 < En', 'denominator negative', 'negative denominator',
                    'each term negative', 'sum negative', 'second order negative',
                    '<0 for all n', 'E0^(0) < En^(0)']
        found = any(kw in steps_text for kw in keywords)
        if found:
            return (True, "Second-order shift negativity justified by negative denominator")
        else:
            return (False, "Explanation does not justify the negativity of second-order shift")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb an input and compare output
    # ------------------------------------------------------------------
    def test_monotonicity_validity_parameter_decreases_with_energy(self) -> Tuple[bool, str]:
        """
        Perturb the energy E and check that the validity parameter decreases
        (since p increases, λ decreases). Requires numeric givens; otherwise skip.
        """
        # We need numeric values for m, V(x), E, hbar to compute p and the derivative.
        if any(v is None for v in [self.hbar, self.m, self.E, self.Vx]):
            return (False, "SKIP: numeric givens (hbar, m, E, Vx) not available for monotonicity test")

        # We also need a numeric form of validity_parameter; if it's symbolic, try to evaluate
        if isinstance(self.validity_parameter, str):
            # Attempt to sympify and substitute numeric values
            try:
                expr = sp.sympify(self.validity_parameter)
                subs_dict = {'hbar': self.hbar, 'm': self.m, 'E': self.E, 'Vx': self.Vx}
                # Assuming p = sqrt(2*m*(E - Vx))
                p_val = math.sqrt(2 * self.m * (self.E - self.Vx))
                # Need dp/dx; we may not have dV/dx. If not, skip.
                return (False, "SKIP: cannot compute dp/dx (no dV/dx given); monotonicity test requires derivative")
            except Exception:
                return (False, "SKIP: validity_parameter expression could not be parsed numerically")
        else:
            # numeric validity_parameter already
            pass

        # If we have a numeric validity_parameter and can compute p, we can compare for two energies
        try:
            # Baseline: current energy E
            p_baseline = math.sqrt(2 * self.m * (self.E - self.Vx))
            # For perturbation, use E + delta (positive delta increases kinetic energy)
            delta = 0.05 * self.E
            E_perturbed = self.E + delta
            p_perturbed = math.sqrt(2 * self.m * (E_perturbed - self.Vx))
            # dp/dx is proportional to -dV/dx; we need dV/dx. Without it, we cannot compute validity_parameter.
            # Instead, we note that validity_parameter ∝ 1/p^3 * |dV/dx|. So if energy increases, p increases,
            # so validity_parameter decreases. This is a qualitative monotonicity.
            # To make it quantitative, we would need dV/dx. Since it's not available, skip.
            return (False, "SKIP: cannot compute exact perturbed validity_parameter without dV/dx")
        except Exception:
            return (False, "SKIP: numerical error in monotonicity computation")

    # ------------------------------------------------------------------
    # Answer sanity (weak) – at most one method
    # ------------------------------------------------------------------
    def test_answer_sanity_basic_checks(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that validity_parameter and E0_second are non-empty
        and (if numeric) finite. No hardcoded target values.
        """
        messages = []
        ok = True

        if self.validity_parameter is None:
            messages.append("validity_parameter missing")
            ok = False
        else:
            if isinstance(self.validity_parameter, (int, float)):
                if not math.isfinite(self.validity_parameter):
                    messages.append("validity_parameter is infinite/NaN")
                    ok = False
                else:
                    messages.append(f"validity_parameter finite: {self.validity_parameter}")
            else:
                expr_str = str(self.validity_parameter)
                if not expr_str.strip():
                    messages.append("validity_parameter empty string")
                    ok = False
                else:
                    messages.append("validity_parameter expression non-empty")

        if self.E0_second is None:
            messages.append("E0_second missing")
            ok = False
        else:
            if isinstance(self.E0_second, (int, float)):
                if not math.isfinite(self.E0_second):
                    messages.append("E0_second is infinite/NaN")
                    ok = False
                else:
                    messages.append(f"E0_second finite: {self.E0_second}")
            else:
                expr_str = str(self.E0_second)
                if not expr_str.strip():
                    messages.append("E0_second empty string")
                    ok = False
                else:
                    messages.append("E0_second expression non-empty")

        if ok:
            return (True, "; ".join(messages))
        else:
            return (False, "; ".join(messages))

    # ------------------------------------------------------------------
    # Master test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods, execute them with try/except,
        collect results, return (all_pass, logs).
        """
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)