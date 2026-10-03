import sympy
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        self.pseudocode = getattr(solution_instance, 'pseudocode', [])

        # Result quantities
        self.lambda_ = self.results.get('lambda', None)
        self.radiation_emitted = self.results.get('radiation_emitted', None)

        # Given physical parameters
        self.v = self.givens.get('v', None)
        self.d = self.givens.get('d', None)
        self.c = self.givens.get('c', None)
        self.theta = self.givens.get('theta', None)
        self.m = self.givens.get('m', None)

        # Derived quantities
        self.beta = None
        if self.v is not None and self.c is not None and self.c != 0:
            self.beta = self.v / self.c

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _is_symbolic(self, expr):
        return hasattr(expr, 'free_symbols')

    def _compute_expected_lambda(self, theta_val=None, v_val=None, d_val=None, c_val=None, m_val=None):
        """Compute lambda from Smith-Purcell formula using given or default parameters."""
        v = v_val if v_val is not None else self.v
        d = d_val if d_val is not None else self.d
        c = c_val if c_val is not None else self.c
        m = m_val if m_val is not None else self.m
        theta_ = theta_val if theta_val is not None else self.theta

        if any(x is None for x in [v, d, c, m, theta_]):
            return None
        if not all(isinstance(x, (int, float)) for x in [v, d, c, m, theta_]):
            return None
        beta = v / c
        return d / m * (c / v - math.cos(theta_))

    # ------------------------------------------------------------------
    # Interface tests – check required keys
    # ------------------------------------------------------------------
    def test_interface_lambda_key(self):
        if 'lambda' not in self.results:
            return (False, "FAIL: 'lambda' key missing in results")
        return (True, "PASS: 'lambda' key present")

    def test_interface_radiation_emitted_key(self):
        if 'radiation_emitted' not in self.results:
            return (False, "FAIL: 'radiation_emitted' key missing in results")
        return (True, "PASS: 'radiation_emitted' key present")

    def test_interface_lambda_type(self):
        if self.lambda_ is None:
            return (True, "SKIP: lambda is None, cannot check type")
        if not (isinstance(self.lambda_, (int, float)) or self._is_symbolic(self.lambda_)):
            return (False, f"FAIL: lambda type is {type(self.lambda_)}, expected numeric or sympy expression")
        return (True, "PASS: lambda type is valid")

    # ------------------------------------------------------------------
    # Invariant tests – based on physical laws
    # ------------------------------------------------------------------
    def test_invariant_smith_purcell(self):
        # Check that d*(1/beta - cosθ) = m*λ
        if self.lambda_ is None:
            return (True, "SKIP: lambda not available")
        if not isinstance(self.theta, (int, float)):
            return (True, "SKIP: theta is symbolic, cannot perform numeric comparison")
        if any(x is None for x in [self.d, self.v, self.c, self.m, self.theta]):
            return (True, "SKIP: missing givens for Smith-Purcell check")
        expected = self._compute_expected_lambda()
        if expected is None:
            return (True, "SKIP: could not compute expected lambda")

        if self._is_symbolic(self.lambda_):
            # Try numeric substitution
            try:
                numeric_lambda = float(self.lambda_.subs({sympy.Symbol('theta'): self.theta,
                                                          sympy.Symbol('v'): self.v,
                                                          sympy.Symbol('d'): self.d,
                                                          sympy.Symbol('c'): self.c,
                                                          sympy.Symbol('m'): self.m}))
            except:
                # Fallback: symbolic simplification
                try:
                    expr = self.d * (1/self.beta - sympy.cos(self.theta)) - self.m * self.lambda_
                    if sympy.simplify(expr) == 0:
                        return (True, "PASS: Smith-Purcell condition satisfied symbolically")
                except:
                    return (False, "FAIL: Could not verify Smith-Purcell condition")
            if math.isclose(numeric_lambda, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"PASS: lambda numeric matches expected: {numeric_lambda:.4g} vs {expected:.4g}")
            else:
                return (False, f"FAIL: lambda {numeric_lambda:.4g} does not match expected {expected:.4g} within tolerance")
        else:
            # Numeric lambda
            if math.isclose(self.lambda_, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"PASS: lambda matches expected: {self.lambda_:.4g} vs {expected:.4g}")
            else:
                return (False, f"FAIL: lambda {self.lambda_:.4g} does not match expected {expected:.4g} within tolerance")

    def test_invariant_beta_less_than_c(self):
        if self.beta is None:
            return (True, "SKIP: beta not available")
        if self.beta >= 1:
            return (False, f"FAIL: beta = {self.beta} >= 1, v must be less than c")
        return (True, f"PASS: beta = {self.beta} < 1")

    def test_invariant_lambda_positive(self):
        if self.lambda_ is None:
            return (True, "SKIP: lambda not available")
        if isinstance(self.lambda_, (int, float)):
            if self.lambda_ > 0:
                return (True, f"PASS: lambda = {self.lambda_} > 0")
            else:
                return (False, f"FAIL: lambda = {self.lambda_} <= 0")
        else:
            return (True, "SKIP: lambda is symbolic, positivity not checked numerically")

    def test_invariant_radiation_emitted_false(self):
        if self.radiation_emitted is None:
            return (True, "SKIP: radiation_emitted not available")
        if self.radiation_emitted is False:
            return (True, "PASS: radiation_emitted is False as expected")
        else:
            return (False, f"FAIL: radiation_emitted is {self.radiation_emitted}, expected False")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check the quality of the derivation
    # ------------------------------------------------------------------
    def test_modeling_constructive_interference_mention(self):
        # Look for key phrases in explanation_steps or pseudocode
        combined_text = " ".join(self.explanation_steps).lower() if self.explanation_steps else ""
        combined_text += " " + " ".join([str(p) for p in self.pseudocode]).lower()
        keywords = ["constructive interference", "smith-purcell", "interference", "path difference"]
        if any(kw in combined_text for kw in keywords):
            return (True, "PASS: Explanation mentions constructive interference or Smith-Purcell effect")
        return (True, "WARN: No explicit mention of constructive interference or Smith-Purcell in explanation")

    def test_trace_formula_match(self):
        # Compare the reported lambda with the formula d*(c/v - cosθ)/m
        if self.lambda_ is None:
            return (True, "SKIP: lambda not available")
        if not isinstance(self.theta, (int, float)):
            return (True, "SKIP: theta is symbolic, cannot perform numeric check")
        expected = self._compute_expected_lambda()
        if expected is None:
            return (True, "SKIP: could not compute expected lambda from givens")
        if self._is_symbolic(self.lambda_):
            try:
                numeric_lambda = float(self.lambda_.subs({sympy.Symbol('theta'): self.theta}))
            except:
                return (True, "SKIP: Could not evaluate symbolic lambda numerically")
            if math.isclose(numeric_lambda, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"PASS: lambda matches expected formula: {numeric_lambda:.4g} vs {expected:.4g}")
            else:
                return (False, f"FAIL: lambda {numeric_lambda:.4g} does not match expected {expected:.4g}")
        else:
            if math.isclose(self.lambda_, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"PASS: lambda matches expected formula: {self.lambda_:.4g} vs {expected:.4g}")
            else:
                return (False, f"FAIL: lambda {self.lambda_:.4g} does not match expected {expected:.4g}")

    # ------------------------------------------------------------------
    # Monotonicity tests – use perturbation and compare baseline vs perturbed output
    # ------------------------------------------------------------------
    def test_monotonicity_lambda_increases_with_theta(self):
        # λ = d*(c/v - cosθ)/m; as θ increases (0→π), cosθ decreases, so λ increases monotonically.
        if self.theta is None or not isinstance(self.theta, (int, float)):
            return (True, "SKIP: theta not numeric, cannot perform monotonicity test")
        if any(x is None for x in [self.d, self.v, self.c, self.m]):
            return (True, "SKIP: missing givens for monotonicity test")
        # Use the stored lambda as baseline (if available and numeric)
        if self.lambda_ is not None and isinstance(self.lambda_, (int, float)):
            baseline_lambda = self.lambda_
        else:
            # compute from formula using current givens
            baseline_lambda = self._compute_expected_lambda()
            if baseline_lambda is None:
                return (True, "SKIP: cannot compute baseline lambda")
        theta0 = float(self.theta)
        # Perturb theta: increase by 10% (minimum 0.1 rad, cap at π)
        delta = max(0.1 * theta0, 0.1) if theta0 > 0 else 0.1
        theta1 = min(theta0 + delta, math.pi)
        if theta1 <= theta0:
            return (True, "SKIP: theta already at maximum, cannot increase")
        # Compute perturbed lambda using the same formula
        perturbed_lambda = self._compute_expected_lambda(theta_val=theta1)
        if perturbed_lambda is None:
            return (True, "SKIP: could not compute perturbed lambda")
        # Check that perturbed lambda > baseline lambda
        if perturbed_lambda > baseline_lambda:
            return (True, f"PASS: lambda increases with theta (baseline {baseline_lambda:.4g} -> perturbed {perturbed_lambda:.4g})")
        else:
            return (False, f"FAIL: lambda does not increase with theta (baseline {baseline_lambda:.4g} >= perturbed {perturbed_lambda:.4g})")

    # ------------------------------------------------------------------
    # Answer sanity tests – weak checks only, no re-derivation of final answer
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        # Combined weak sanity: check lambda is finite (if numeric) and radiation_emitted is bool.
        all_ok = True
        messages = []
        if self.lambda_ is not None and isinstance(self.lambda_, (int, float)):
            if not math.isfinite(self.lambda_):
                all_ok = False
                messages.append("lambda is not finite")
        if self.radiation_emitted is not None:
            if not isinstance(self.radiation_emitted, bool):
                all_ok = False
                messages.append("radiation_emitted is not bool")
        if all_ok:
            return (True, "PASS: Weak sanity checks passed")
        else:
            return (False, "FAIL: " + "; ".join(messages))

    # ------------------------------------------------------------------
    # Main runner
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