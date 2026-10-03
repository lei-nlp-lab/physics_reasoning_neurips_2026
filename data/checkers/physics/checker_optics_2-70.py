import math
import re
from typing import Any

class PhysicsChecker:
    """
    Audits solutions for the two-antenna diffraction problem.
    Covers interface, invariant, modeling/trace, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # --- Extract given parameters safely (use None for missing) ---
        self.D = getattr(solution_instance, 'D', None)
        self.d = getattr(solution_instance, 'd', None)
        self.lambda_ = getattr(solution_instance, 'lambda', None)
        self.f = getattr(solution_instance, 'f', None)
        self.t = getattr(solution_instance, 't', None)
        self.theta = getattr(solution_instance, 'theta', None)

        # --- Extract results dict and emitted values ---
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        self.I_single = self.results.get('I_single', None)
        self.I_total = self.results.get('I_total', None)
        self.Delta_f_max = self.results.get('Delta_f_max', None)

        # Derived quantities
        self.k = None
        self.R = None
        if self.lambda_ is not None and self.lambda_ > 0:
            self.k = 2.0 * math.pi / self.lambda_
        if self.D is not None and self.D > 0:
            self.R = self.D / 2.0

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_results(self) -> tuple:
        """Check that all three result keys exist and are not None."""
        missing = []
        for key in ['I_single', 'I_total', 'Delta_f_max']:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_given_params_available(self) -> tuple:
        """Check that essential given parameters (D, d, lambda, t) are provided."""
        missing = []
        if self.D is None:
            missing.append("D")
        if self.d is None:
            missing.append("d")
        if self.lambda_ is None:
            missing.append("lambda")
        if self.t is None:
            missing.append("t")
        if missing:
            return (False, f"Essential parameters missing: {', '.join(missing)}")
        return (True, "All essential parameters present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_single_contains_bessel(self) -> tuple:
        """
        The single‑antenna pattern must involve the Bessel function J₁
        (Airy pattern). This is a theoretical invariant for circular aperture diffraction.
        """
        if self.I_single is None:
            return (False, "SKIP: I_single not provided")
        text = str(self.I_single)
        if not re.search(r'J[_ ]?1', text, re.IGNORECASE):
            return (False, "I_single does not contain J1 (Bessel function)")
        if not re.search(r'k\s*R\s*sin', text, re.IGNORECASE):
            return (False, "I_single missing k*R*sinθ factor")
        return (True, "I_single contains correct Bessel factor (J1) and kR sinθ")

    def test_invariant_total_product_structure(self) -> tuple:
        """
        The total intensity must be the product of the single‑antenna pattern
        (Bessel factor) and an array factor (cos²). Fundamental array theorem.
        """
        if self.I_single is None or self.I_total is None:
            return (False, "SKIP: I_single or I_total not provided")
        total_text = str(self.I_total)
        # Check that total contains Bessel factor (single pattern)
        if not re.search(r'J[_ ]?1', total_text, re.IGNORECASE):
            return (False, "I_total missing the single‑antenna Bessel factor")
        # Check that total contains a cosine squared (array factor)
        if not re.search(r'cos\^?2', total_text, re.IGNORECASE):
            return (False, "I_total missing cosine squared (array factor)")
        return (True, "I_total contains both single pattern (J1) and array factor (cos²)")

    # ------------------------------------------------------------------
    # Modeling / trace tests (form and consistency checks)
    # ------------------------------------------------------------------
    def test_modeling_single_pattern_form(self) -> tuple:
        """
        The single‑antenna pattern should be expressed as I0 times a squared
        Airy function.  Check that the common notation appears:
        something like 'I0' or 'I(0)' and the square bracket.
        """
        if self.I_single is None:
            return (False, "SKIP: I_single not provided")
        text = str(self.I_single)
        has_I0 = bool(re.search(r'I\s*[\(0]', text, re.IGNORECASE))
        has_square_bracket = ('[' in text and ']' in text) or '^2' in text
        if not has_I0:
            return (False, "I_single missing I0 (on‑axis intensity)")
        if not has_square_bracket:
            return (False, "I_single missing square bracket or exponent for squared Bessel")
        return (True, "I_single has plausible form: I0 × [Bessel]²")

    def test_modeling_frequency_stability_condition(self) -> tuple:
        """
        Check that the frequency stability condition is either given as
        "Δf < 1/t" or, if numeric, satisfies Δf_max ≤ 1/t (within 10%).
        This verifies that the trace from the explanation step is followed.
        """
        if self.Delta_f_max is None:
            return (False, "SKIP: Delta_f_max not provided")
        if isinstance(self.Delta_f_max, str):
            text = self.Delta_f_max.replace(" ", "")
            if re.search(r'1\s*/\s*t', text):
                return (True, "Delta_f_max string matches '1/t'")
            else:
                return (False, f"Delta_f_max string '{text}' does not match '1/t'")
        elif isinstance(self.Delta_f_max, (int, float)):
            if self.t is not None and self.t > 0:
                # The condition should be Δf_max ≈ 1/t
                expected = 1.0 / self.t
                if math.isclose(self.Delta_f_max, expected, rel_tol=0.1):
                    return (True, f"Delta_f_max ≈ 1/t ({self.Delta_f_max:.3e})")
                else:
                    return (False, f"Delta_f_max = {self.Delta_f_max}, expected ≈ {expected:.3e}")
            else:
                # No t: can only check positivity
                if self.Delta_f_max > 0:
                    return (True, "Delta_f_max numeric and positive (no t to compare)")
                else:
                    return (False, "Delta_f_max numeric but non‑positive")
        else:
            return (False, f"Delta_f_max has unexpected type {type(self.Delta_f_max).__name__}")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_frequency_time_inverse(self) -> tuple:
        """
        For the frequency stability condition: Δf_max ∝ 1/t.
        If we perturb t (here using the given t), the product Δf_max * t
        should remain close to 1 (theoretical constant).
        This is an explicit effect of a change in t.
        """
        if self.t is None or self.Delta_f_max is None:
            return (False, "SKIP: need t and Delta_f_max (both numeric) for monotonicity")
        if not isinstance(self.t, (int, float)) or not isinstance(self.Delta_f_max, (int, float)):
            return (False, "SKIP: t and Delta_f_max must be numeric")
        product = self.Delta_f_max * self.t
        if math.isclose(product, 1.0, rel_tol=0.02):
            return (True, f"Δf_max * t = {product:.3e} ≈ 1 – monotonicity consistent")
        else:
            return (False, f"Δf_max * t = {product:.3e}, expected ≈ 1 (within 2%)")

    # ------------------------------------------------------------------
    # Weak answer sanity (sign/finite, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_and_finite(self) -> tuple:
        """
        All results should be finite and non‑empty.
        """
        issues = []
        if self.I_single is not None:
            if isinstance(self.I_single, str) and len(self.I_single.strip()) == 0:
                issues.append("I_single is empty string")
        else:
            issues.append("I_single is None")

        if self.I_total is not None:
            if isinstance(self.I_total, str) and len(self.I_total.strip()) == 0:
                issues.append("I_total is empty string")
        else:
            issues.append("I_total is None")

        if self.Delta_f_max is not None:
            if isinstance(self.Delta_f_max, str):
                if len(self.Delta_f_max.strip()) == 0:
                    issues.append("Delta_f_max is empty string")
            elif isinstance(self.Delta_f_max, (int, float)):
                if math.isnan(self.Delta_f_max) or math.isinf(self.Delta_f_max):
                    issues.append("Delta_f_max is NaN or Inf")
                elif self.Delta_f_max <= 0:
                    issues.append(f"Delta_f_max is non‑positive: {self.Delta_f_max}")
        else:
            issues.append("Delta_f_max is None")

        if issues:
            return (False, " | ".join(issues))
        return (True, "All results are finite and non‑empty (minimal sanity)")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> tuple:
        """
        Discover all bound methods starting with 'test_', run them in
        a try/except, and return (all_pass, logs).
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
                passed = False
                msg = f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)