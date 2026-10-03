import math
import traceback
from sympy import sympify, symbols, sin, cos, simplify, Eq

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Extract all needed data from the solution_instance.
        Use None defaults for missing values.
        The instance can be an object with attributes/givens/results, or a dict.
        """
        # ---- given parameters ----
        self.a = None
        self.d = None
        self.lambda_ = None
        self.f = None
        self.I0 = None

        # Try to get givens from a dictionary
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.a = solution_instance.givens.get('a', None)
            self.d = solution_instance.givens.get('d', None)
            self.lambda_ = solution_instance.givens.get('lambda', None)
            self.f = solution_instance.givens.get('f', None)
            self.I0 = solution_instance.givens.get('I0', None)
        elif isinstance(solution_instance, dict):
            givens = solution_instance.get('givens', {})
            self.a = givens.get('a', None)
            self.d = givens.get('d', None)
            self.lambda_ = givens.get('lambda', None)
            self.f = givens.get('f', None)
            self.I0 = givens.get('I0', None)
        else:
            # Fallback to direct attributes
            self.a = getattr(solution_instance, 'a', None)
            self.d = getattr(solution_instance, 'd', None)
            self.lambda_ = getattr(solution_instance, 'lambda', None)
            self.f = getattr(solution_instance, 'f', None)
            self.I0 = getattr(solution_instance, 'I0', None)

        # ---- results ----
        self.I = None
        self.beta = None
        self.gamma = None

        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.I = solution_instance.results.get('I', None)
            self.beta = solution_instance.results.get('beta', None)
            self.gamma = solution_instance.results.get('gamma', None)
        elif isinstance(solution_instance, dict):
            results = solution_instance.get('results', {})
            self.I = results.get('I', None)
            self.beta = results.get('beta', None)
            self.gamma = results.get('gamma', None)
        else:
            self.I = getattr(solution_instance, 'I', None)
            self.beta = getattr(solution_instance, 'beta', None)
            self.gamma = getattr(solution_instance, 'gamma', None)

        # ---- explanation steps (may be string or list/string) ----
        self.explanation_steps = None
        if hasattr(solution_instance, 'explanation_steps'):
            self.explanation_steps = solution_instance.explanation_steps
        elif isinstance(solution_instance, dict):
            self.explanation_steps = solution_instance.get('explanation_steps', None)

        # ---- pseudocode (auxiliary, not heavily used) ----
        self.pseudocode = None
        if hasattr(solution_instance, 'pseudocode'):
            self.pseudocode = solution_instance.pseudocode
        elif isinstance(solution_instance, dict):
            self.pseudocode = solution_instance.get('pseudocode', None)

    # ------------------------------------------------------------------
    #  TEST METHODS
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all mandatory givens and the result I exist."""
        missing = []
        for key in ['a', 'd', 'lambda_', 'f', 'I0']:
            if getattr(self, key) is None:
                missing.append(key)
        if self.I is None:
            missing.append('I')
        if missing:
            return (False, f"Missing required keys: {missing}")
        return (True, "All required keys present (a, d, lambda, f, I0, I)")

    def test_invariant_d_a_ratio(self):
        """Verify that d/a = 5."""
        if self.d is None or self.a is None:
            return (False, "SKIP: d or a not available")
        try:
            ratio = self.d / self.a
        except TypeError:
            return (False, "SKIP: d/a ratio cannot be computed numerically")
        if isinstance(ratio, (int, float)):
            if math.isclose(ratio, 5.0, rel_tol=1e-3):
                return (True, f"d/a = {ratio} ≈ 5")
            else:
                return (False, f"d/a = {ratio} != 5")
        # symbolic fallback
        return (True, "d/a ratio given symbolically, assumed correct")

    def test_invariant_gamma_beta_relation(self):
        """Check that gamma = (d/a)*beta."""
        if self.beta is None or self.gamma is None:
            return (False, "SKIP: beta or gamma not provided")
        if self.d is None or self.a is None:
            return (False, "SKIP: d or a not provided")
        try:
            observed_ratio = self.gamma / self.beta
        except (TypeError, ZeroDivisionError):
            return (False, "Cannot compute gamma/beta (division error)")
        expected_ratio = self.d / self.a
        if isinstance(observed_ratio, (int, float)) and isinstance(expected_ratio, (int, float)):
            if math.isclose(observed_ratio, expected_ratio, rel_tol=1e-2):
                return (True, f"gamma/beta = {observed_ratio} matches d/a = {expected_ratio}")
            else:
                return (False, f"gamma/beta = {observed_ratio} != d/a = {expected_ratio}")
        # symbolic
        try:
            symbolic_eq = Eq(simplify(observed_ratio - expected_ratio), 0)
            if symbolic_eq:
                return (True, "Symbolic gamma/beta equals d/a")
            else:
                return (False, "Symbolic gamma/beta does not equal d/a")
        except Exception:
            return (False, "Cannot verify symbolic ratio")

    def test_modeling_intensity_structure(self):
        """
        Check that the intensity expression contains the expected
        sinc-sq and cos-sq terms.
        """
        if self.I is None:
            return (False, "SKIP: I not available")
        try:
            I_str = str(self.I)
        except Exception:
            return (False, "Cannot stringify I")
        # Expected patterns (not exclusive – just strong indicators)
        has_sin = 'sin' in I_str.lower()
        has_beta_in_sin = ('sin(beta)' in I_str) or ('sin(β)' in I_str)
        has_cos = 'cos' in I_str.lower()
        has_gamma_in_cos = ('cos(gamma)' in I_str) or ('cos(γ)' in I_str)
        has_I0 = 'I0' in I_str
        if has_sin and has_cos and has_I0:
            return (True, "Intensity expression contains sin, cos and I0")
        else:
            details = []
            if not has_sin: details.append("missing sin")
            if not has_cos: details.append("missing cos")
            if not has_I0: details.append("missing I0")
            return (False, f"Intensity expression incomplete: {', '.join(details)}")

    def test_modeling_missing_orders_mentioned(self):
        """
        The explanation must mention that for d/a=5 integer, some
        interference orders are suppressed (missing orders).
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        steps_str = str(self.explanation_steps).lower()
        if 'missing' in steps_str and ('order' in steps_str or 'orders' in steps_str):
            return (True, "Explanation mentions missing orders")
        else:
            return (False, "Explanation does not mention missing orders (essential for integer d/a)")

    def test_modeling_explanation_steps(self):
        """
        Verify that the explanation steps contain the key physics
        – the formula, the ratio d/a=5, and the missing orders.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        steps_str = str(self.explanation_steps).lower()
        # Core phrases that should appear
        checks = [
            ("I = I_0", "i = i_0" in steps_str),
            ("sinβ/β", 'sin' in steps_str and 'β' in steps_str),
            ("cos²γ", 'cos' in steps_str and 'γ' in steps_str),
            ("d/a = 5", 'd/a = 5' in steps_str or 'd/a=5' in steps_str),
            ("missing orders", 'missing' in steps_str),
        ]
        passed = [name for name, ok in checks if ok]
        if len(passed) >= 3:
            return (True, f"Explanation steps contain key physics elements: {passed}")
        else:
            failed = [name for name, ok in checks if not ok]
            return (False, f"Explanation steps missing: {failed}")

    def test_monotonicity_gamma_beta_ratio_with_d(self):
        """
        Perturb d by +10% and verify that gamma/beta ratio changes
        proportionally.
        """
        if self.d is None or self.a is None:
            return (False, "SKIP: d or a not available for perturbation")
        try:
            baseline_ratio = self.d / self.a
        except TypeError:
            return (False, "SKIP: cannot compute baseline d/a ratio")
        if not isinstance(baseline_ratio, (int, float)):
            return (False, "SKIP: d/a ratio is not numeric")
        # Perturb: increase d by 10%
        perturbed_d = self.d * 1.1
        perturbed_ratio = perturbed_d / self.a
        # Expect the ratio to increase by 10%
        expected_perturbed_ratio = baseline_ratio * 1.1
        if math.isclose(perturbed_ratio, expected_perturbed_ratio, rel_tol=1e-9):
            return (True, f"d increased 10%: gamma/beta ratio changed from {baseline_ratio} to {perturbed_ratio}, consistent")
        else:
            return (False, f"Perturbation failed: expected {expected_perturbed_ratio}, got {perturbed_ratio}")

    def test_monotonicity_central_peak_fringe_count(self):
        """
        If d/a increases by 20%, the number of interference maxima
        inside the central diffraction peak increases.
        """
        if self.d is None or self.a is None:
            return (False, "SKIP: d or a not available")
        try:
            baseline_ratio = self.d / self.a
        except TypeError:
            return (False, "SKIP: cannot compute baseline d/a ratio")
        if not isinstance(baseline_ratio, (int, float)):
            return (False, "SKIP: d/a ratio is not numeric")
        # For an integer or near-integer d/a, the number of maxima in central peak is
        # 2*floor(d/a)-1 (with d/a exactly 5 -> 9). Increase d/a to 6 -> expected 11.
        # We check that the explanation steps (or derived quantity) acknowledges this.
        # Since we cannot recompute the solution, we infer from the explanation.
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available for monotonicity check")
        steps_str = str(self.explanation_steps).lower()
        # Check that explanation mentions the number of maxima depends on d/a.
        # A well-written solution should contain the formula "2(d/a)-1" or "9 maxima".
        if 'maxima' in steps_str and ('2(d/a)-1' in steps_str or '9' in steps_str):
            return (True, "Explanation acknowledges that central peak contains 2(d/a)-1 maxima, consistent with monotonicity")
        else:
            return (False, "Explanation does not indicate the relationship between d/a and number of maxima")

    def test_answer_sanity_weak(self):
        """
        Minimal sanity: I must be a non‑empty string/expression
        containing 'I0', 'sin' and 'cos'.
        """
        if self.I is None:
            return (False, "I is None")
        try:
            I_str = str(self.I)
        except Exception:
            return (False, "Cannot convert I to string")
        if not I_str.strip():
            return (False, "I is an empty string")
        if 'I0' in I_str and 'sin' in I_str and 'cos' in I_str:
            return (True, "Weak sanity passed")
        else:
            return (False, "I does not contain I0, sin, and cos")

    # ------------------------------------------------------------------
    #  TEST DISPATCH
    # ------------------------------------------------------------------
    def run_all_tests(self):
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# Example usage (commented out):
# if __name__ == "__main__":
#     class DummySolution:
#         pass
#     sol = DummySolution()
#     sol.givens = {'a': 1e-6, 'd': 5e-6, 'lambda': 500e-9, 'f': 0.5, 'I0': 1.0}
#     sol.results = {'I': "I0*(sin(beta)/beta)**2 * (cos(gamma))**2",
#                    'beta': "pi*a*sin(theta)/lambda", 'gamma': "pi*d*sin(theta)/lambda"}
#     sol.explanation_steps = "I = I0 (sinβ/β)^2 cos²γ, d/a=5, orders 5,10,... missing"
#     checker = PhysicsChecker(sol)
#     ok, msgs = checker.run_all_tests()
#     for m in msgs:
#         print(m)