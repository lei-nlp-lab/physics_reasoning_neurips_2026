import math
import sympy as sp
import traceback
from collections import defaultdict

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.P1 = self.results.get('P1', None)
        self.P4 = self.results.get('P4', None)
        # Extract explanation steps (list of strings, nullable)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = []
        elif not isinstance(self.explanation_steps, list):
            self.explanation_steps = []

        # Pre-parse symbolic expressions if possible
        self.sym_P1 = None
        self.sym_P4 = None
        self.x = sp.symbols('x', real=True, nonnegative=True)
        self._prepare_symbolic_expressions()

    def _prepare_symbolic_expressions(self):
        """Convert P1 and P4 to sympy expressions, if possible."""
        for attr, name in [(self.P1, 'P1'), (self.P4, 'P4')]:
            if attr is not None:
                try:
                    if isinstance(attr, str):
                        # Replace common math notation with Python style
                        expr_str = attr.replace('^', '**')
                        expr = sp.sympify(expr_str)
                    else:
                        # Assume it's already a sympy expression
                        expr = sp.sympify(attr)
                    setattr(self, f'sym_{name}', expr)
                except Exception:
                    setattr(self, f'sym_{name}', None)
            else:
                setattr(self, f'sym_{name}', None)

    def _evaluate_numeric(self, expr, x_val):
        """Evaluate a sympy expression at a given numeric x value."""
        if expr is None:
            return None
        try:
            val = expr.subs(self.x, x_val).evalf()
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_P1_exists(self):
        if self.P1 is None:
            return (False, "P1 key missing in results")
        return (True, "P1 key present")

    def test_interface_P4_exists(self):
        if self.P4 is None:
            return (False, "P4 key missing in results")
        return (True, "P4 key present")

    def test_interface_P1_type(self):
        if self.P1 is not None and not isinstance(self.P1, (str, sp.Basic)):
            return (False, f"P1 has unsupported type {type(self.P1).__name__}")
        return (True, "P1 type valid")

    def test_interface_P4_type(self):
        if self.P4 is not None and not isinstance(self.P4, (str, sp.Basic)):
            return (False, f"P4 has unsupported type {type(self.P4).__name__}")
        return (True, "P4 type valid")

    # ------------------------------------------------------------
    # Invariant / theorem-based tests (use math.isclose for numbers)
    # ------------------------------------------------------------
    def test_invariant_P1_non_negative(self):
        """Check that P1 evaluated at several x values is >= 0."""
        if self.sym_P1 is None:
            return (False, "SKIP: cannot parse P1")
        for x_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            val = self._evaluate_numeric(self.sym_P1, x_val)
            if val is None or val < 0 - 1e-12:
                return (False, f"P1 negative or NaN at x={x_val}")
        return (True, "P1 non-negative for tested x values")

    def test_invariant_P4_non_negative(self):
        if self.sym_P4 is None:
            return (False, "SKIP: cannot parse P4")
        for x_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            val = self._evaluate_numeric(self.sym_P4, x_val)
            if val is None or val < 0 - 1e-12:
                return (False, f"P4 negative or NaN at x={x_val}")
        return (True, "P4 non-negative for tested x values")

    def test_invariant_P1_less_than_one(self):
        if self.sym_P1 is None:
            return (False, "SKIP: cannot parse P1")
        for x_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            val = self._evaluate_numeric(self.sym_P1, x_val)
            if val is None or val > 1.0 + 1e-12:
                return (False, f"P1 exceeds 1 at x={x_val}")
        return (True, "P1 <= 1 for tested x values")

    def test_invariant_P4_less_than_one(self):
        if self.sym_P4 is None:
            return (False, "SKIP: cannot parse P4")
        for x_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            val = self._evaluate_numeric(self.sym_P4, x_val)
            if val is None or val > 1.0 + 1e-12:
                return (False, f"P4 exceeds 1 at x={x_val}")
        return (True, "P4 <= 1 for tested x values")

    def test_invariant_P1_plus_P4_less_than_one(self):
        if self.sym_P1 is None or self.sym_P4 is None:
            return (False, "SKIP: cannot parse both P1 and P4")
        for x_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            v1 = self._evaluate_numeric(self.sym_P1, x_val)
            v4 = self._evaluate_numeric(self.sym_P4, x_val)
            if v1 is None or v4 is None:
                continue
            if v1 + v4 > 1.0 + 1e-12:
                return (False, f"P1+P4 > 1 at x={x_val} (sum={v1+v4})")
        return (True, "P1+P4 <= 1 for tested x values (consistent with total probability constraint)")

    # ------------------------------------------------------------
    # Monotonicity tests (explicit perturbation from baseline)
    # ------------------------------------------------------------
    def test_monotonicity_P1_increases_at_small_x(self):
        """
        For x < 1/3, P1 is increasing. Baseline x=0.1, perturb to x=0.2.
        Expect P1(0.2) > P1(0.1).
        """
        if self.sym_P1 is None:
            return (False, "SKIP: cannot parse P1")
        baseline_x = 0.1
        perturbed_x = 0.2  # perturb by +0.1
        v_base = self._evaluate_numeric(self.sym_P1, baseline_x)
        v_pert = self._evaluate_numeric(self.sym_P1, perturbed_x)
        if v_base is None or v_pert is None:
            return (False, "Could not evaluate P1 at baseline or perturbed x")
        if v_pert > v_base:
            return (True, f"P1 increases from {v_base:.4f} at x={baseline_x} to {v_pert:.4f} at x={perturbed_x}")
        else:
            return (False, f"P1 did not increase: base={v_base:.4f}, perturbed={v_pert:.4f}")

    def test_monotonicity_P1_decreases_at_large_x(self):
        """
        For x > 1/3, P1 is decreasing. Baseline x=10, perturb to x=20.
        Expect P1(20) < P1(10).
        """
        if self.sym_P1 is None:
            return (False, "SKIP: cannot parse P1")
        baseline_x = 10.0
        perturbed_x = 20.0  # perturb by +10
        v_base = self._evaluate_numeric(self.sym_P1, baseline_x)
        v_pert = self._evaluate_numeric(self.sym_P1, perturbed_x)
        if v_base is None or v_pert is None:
            return (False, "Could not evaluate P1 at baseline or perturbed x")
        if v_pert < v_base:
            return (True, f"P1 decreases from {v_base:.6f} at x={baseline_x} to {v_pert:.6f} at x={perturbed_x}")
        else:
            return (False, f"P1 did not decrease: base={v_base:.6f}, perturbed={v_pert:.6f}")

    def test_monotonicity_P4_increases_with_x(self):
        """
        P4 = (x/(1+x))^4 is strictly increasing for x>0.
        Baseline x=0.1, perturb to x=5.0 (50× increase). Expect P4(5) > P4(0.1).
        """
        if self.sym_P4 is None:
            return (False, "SKIP: cannot parse P4")
        baseline_x = 0.1
        perturbed_x = 5.0
        v_base = self._evaluate_numeric(self.sym_P4, baseline_x)
        v_pert = self._evaluate_numeric(self.sym_P4, perturbed_x)
        if v_base is None or v_pert is None:
            return (False, "Could not evaluate P4 at baseline or perturbed x")
        if v_pert > v_base:
            return (True, f"P4 increases from {v_base:.6f} at x={baseline_x} to {v_pert:.6f} at x={perturbed_x}")
        else:
            return (False, f"P4 did not increase: base={v_base:.6f}, perturbed={v_pert:.6f}")

    # ------------------------------------------------------------
    # Modeling quality tests (based on explanation_steps)
    # ------------------------------------------------------------
    def test_modeling_explanation_mentions_grand_canonical(self):
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        text = ' '.join(self.explanation_steps).lower()
        if 'grand canonical' in text or 'partition function' in text:
            return (True, "Explanation mentions grand canonical or partition function")
        else:
            return (False, "Explanation does not mention grand canonical ensemble")

    def test_modeling_explanation_mentions_site_independence(self):
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        text = ' '.join(self.explanation_steps).lower()
        if 'independent' in text or 'non-interacting' in text or 'separate site' in text:
            return (True, "Explanation references site independence")
        else:
            return (False, "Explanation does not mention independence of sites")

    def test_modeling_explanation_includes_binomial_coefficient(self):
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        text = ' '.join(self.explanation_steps).lower()
        if 'binom' in text or 'choose' in text or '4 choose' in text:
            return (True, "Explanation includes binomial coefficient (or 'choose')")
        else:
            return (False, "Explanation missing binomial coefficient for site combinatorics")

    # ------------------------------------------------------------
    # Weak answer sanity (only one test, no derivations)
    # ------------------------------------------------------------
    def test_answer_sanity_not_trivially_wrong(self):
        """
        Quick checks: P1 and P4 should be non‑None, evaluate to finite values,
        and P1 should be positive (except at x=0 where it's zero).
        """
        if self.sym_P1 is None or self.sym_P4 is None:
            return (False, "Could not parse P1 or P4")
        # Check at x=1 (a convenient test point)
        v1 = self._evaluate_numeric(self.sym_P1, 1.0)
        v4 = self._evaluate_numeric(self.sym_P4, 1.0)
        if v1 is None or v4 is None:
            return (False, "Could not evaluate at x=1")
        if math.isnan(v1) or math.isinf(v1):
            return (False, "P1 is NaN or infinite at x=1")
        if math.isnan(v4) or math.isinf(v4):
            return (False, "P4 is NaN or infinite at x=1")
        return (True, "P1 and P4 are finite and evaluable at x=1")

    # ------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)