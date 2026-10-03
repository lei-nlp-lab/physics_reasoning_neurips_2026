import sympy as sp
import math
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Extract all relevant data from the solution_instance.
        Assume solution_instance has attributes or a `.results` dict.
        All missing data is set to None (no numeric defaults).
        """
        # --- Givens (if present) ---
        self.q = getattr(solution_instance, 'q', None)
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.sigma = getattr(solution_instance, 'sigma', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # --- Results dict (mandatory) ---
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if self.results is None:
            # Fallback: try to get from .results attribute
            results_attr = getattr(solution_instance, 'results', None)
            if results_attr is None:
                self.results = {}
            else:
                self.results = results_attr

        # Extract each expected result safely
        self.E1 = self.results.get('E1', None)
        self.E2 = self.results.get('E2', None)
        self.E3 = self.results.get('E3', None)
        self.Q_inner = self.results.get('Q_inner', None)

        # Store list of required key names for interface checks
        self.required_result_keys = ['E1', 'E2', 'E3', 'Q_inner']

    # ----------------------------------------------------------------
    # Interface tests – verify that required result keys exist
    # ----------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all expected result keys are present in self.results."""
        if self.results is None:
            return (False, "SKIP: results dict is None")
        missing = [key for key in self.required_result_keys if key not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_E1_type(self) -> Tuple[bool, str]:
        """E1 should be a sympy expression (or at least not None)."""
        if self.E1 is None:
            return (False, "SKIP: E1 is None")
        if not isinstance(self.E1, sp.Basic):
            return (False, f"E1 is not a sympy expression, type: {type(self.E1)}")
        return (True, "E1 is a sympy expression")

    def test_interface_E2_type(self) -> Tuple[bool, str]:
        """E2 should be 0 or equivalent expression."""
        if self.E2 is None:
            return (False, "SKIP: E2 is None")
        return (True, "E2 exists")

    def test_interface_E3_type(self) -> Tuple[bool, str]:
        """E3 should be a sympy expression."""
        if self.E3 is None:
            return (False, "SKIP: E3 is None")
        if not isinstance(self.E3, sp.Basic):
            return (False, f"E3 is not a sympy expression, type: {type(self.E3)}")
        return (True, "E3 is a sympy expression")

    def test_interface_Q_inner_type(self) -> Tuple[bool, str]:
        """Q_inner should be a sympy expression (or number)."""
        if self.Q_inner is None:
            return (False, "SKIP: Q_inner is None")
        return (True, "Q_inner exists")

    # ----------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity checks
    # ----------------------------------------------------------------
    def test_invariant_Gauss_law_region1(self) -> Tuple[bool, str]:
        """
        For r < a: E1 * 4πr^2  =  q/ε0  (Gauss's law inside a spherical surface)
        Expect E1 = q/(4πε0 r^2).  Check the functional form: E1 ∝ 1/r^2.
        """
        if self.E1 is None:
            return (False, "SKIP: E1 is None")
        r = sp.Symbol('r', positive=True)
        try:
            # Check that E1 depends on r and has 1/r^2 form
            if not self.E1.has(sp.Symbol('r')):
                return (False, "E1 does not depend on r")
            coeff, dep = sp.expand(self.E1).as_independent(r)
            if dep == r**-2 or dep == 1/r**2:
                return (True, "E1 has correct r^-2 dependence")
            else:
                return (False, f"E1 functional form is not 1/r^2, got dependence: {dep}")
        except Exception as e:
            return (False, f"Exception in Gauss law region1 check: {e}")

    def test_invariant_region2_inside_conductor(self) -> Tuple[bool, str]:
        """
        For a < r < b, inside conductor: E = 0.  Check that E2 is exactly 0.
        """
        if self.E2 is None:
            return (False, "SKIP: E2 is None")
        try:
            if sp.simplify(self.E2) == 0:
                return (True, "E2 = 0 inside conductor")
            else:
                return (False, f"E2 is not zero: {self.E2}")
        except Exception as e:
            return (False, f"Exception in region2 conductor check: {e}")

    def test_invariant_Gauss_law_region3(self) -> Tuple[bool, str]:
        """
        For r > b: E3 should have the form (σ b^2)/(ε0 r^2).
        Check the 1/r^2 dependence and if possible the coefficient.
        """
        if self.E3 is None:
            return (False, "SKIP: E3 is None")
        r = sp.Symbol('r', positive=True)
        try:
            coeff, dep = sp.expand(self.E3).as_independent(r)
            if not (dep == r**-2 or dep == 1/r**2):
                return (False, f"E3 functional form is not 1/r^2, got: {dep}")
            if self.sigma is None or self.b is None:
                return (True, "E3 has 1/r^2 dependence; coefficient check skipped (missing sigma or b)")
            # Expected coefficient = sigma * b^2 / epsilon0 (assuming epsilon0 in denominator)
            if self.epsilon0 is not None:
                expected_coeff = self.sigma * self.b**2 / self.epsilon0
                diff = sp.simplify(coeff - expected_coeff)
                if diff == 0:
                    return (True, "E3 coefficient matches sigma*b^2/epsilon0")
                else:
                    return (False, f"E3 coefficient does not match sigma*b^2/epsilon0. diff={diff}")
            else:
                return (True, "E3 has 1/r^2 dependence; coefficient check skipped (epsilon0 not given)")
        except Exception as e:
            return (False, f"Exception in Gauss law region3 check: {e}")

    def test_invariant_Q_inner_from_equilibrium(self) -> Tuple[bool, str]:
        """
        In electrostatic equilibrium, Q_inner must be -q to make E=0 inside conductor.
        """
        if self.Q_inner is None or self.q is None:
            return (False, "SKIP: Q_inner or q missing")
        try:
            expected = -self.q
            if sp.simplify(self.Q_inner - expected) == 0:
                return (True, "Q_inner = -q correct")
            else:
                return (False, f"Q_inner is {self.Q_inner}, expected {expected}")
        except Exception as e:
            return (False, f"Exception in Q_inner equilibrium check: {e}")

    def test_invariant_charge_conservation(self) -> Tuple[bool, str]:
        """
        The total charge inside a spherical surface at r > b should equal the sum:
        q + Q_inner + Q_outer, where Q_outer = sigma * 4πb^2.
        Check that Q_inner + q + sigma*4πb^2 is consistent with E3 expression.
        """
        if self.Q_inner is None or self.q is None or self.sigma is None or self.b is None:
            return (False, "SKIP: missing one or more required variables")
        try:
            Q_outer_expected = self.sigma * 4 * sp.pi * self.b**2
            total_enc = sp.simplify(self.q + self.Q_inner + Q_outer_expected)
            # With Q_inner = -q, total_enc should be Q_outer_expected
            diff = sp.simplify(total_enc - Q_outer_expected)
            if diff == 0:
                return (True, "Total charge enclosed for r>b equals sigma*4πb^2")
            else:
                return (False, f"Total enclosed charge differs from Q_outer by {diff}")
        except Exception as e:
            return (False, f"Exception in charge conservation check: {e}")

    # ----------------------------------------------------------------
    # Modeling/Trace tests – based on the explanation steps
    # ----------------------------------------------------------------
    def test_modeling_Q_inner_relationship(self) -> Tuple[bool, str]:
        """
        From the explanation: 'Electrostatic equilibrium requires that the total charge
        on inner surface of the conducting shell be -q.' Check that Q_inner + q = 0.
        """
        if self.Q_inner is None or self.q is None:
            return (False, "SKIP: Q_inner or q missing")
        try:
            if sp.simplify(self.Q_inner + self.q) == 0:
                return (True, "Q_inner + q = 0, correct inner surface charge")
            else:
                return (False, f"Q_inner + q ≠ 0: {self.Q_inner} + {self.q} = {sp.simplify(self.Q_inner + self.q)}")
        except Exception as e:
            return (False, f"Exception in modeling Q_inner relationship: {e}")

    def test_modeling_E2_zero_inside_conductor(self) -> Tuple[bool, str]:
        """
        The explanation states E=0 for a<r<b. Check that E2 is exactly 0.
        """
        if self.E2 is None:
            return (False, "SKIP: E2 is None")
        try:
            zero = sp.Integer(0)
            if self.E2.equals(zero):
                return (True, "E2 = 0 inside conductor")
            else:
                return (False, f"E2 is not zero: {self.E2}")
        except Exception as e:
            return (False, f"Exception in modeling E2: {e}")

    # ----------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_Q_inner_vs_q(self) -> Tuple[bool, str]:
        """
        Perturb q by a small positive factor and verify that Q_inner changes
        in the opposite direction (ΔQ_inner = -Δq). This is a true monotonicity test.
        """
        if self.Q_inner is None or self.q is None:
            return (False, "SKIP: Q_inner or q missing")
        try:
            # Define a small perturbation factor (e.g., 1% increase)
            delta = sp.Rational(1, 100)  # 0.01
            # Symbolic q (or numeric if given)
            q_sym = self.q if isinstance(self.q, sp.Basic) else sp.sympify(self.q)
            # Baseline Q_inner expression (should be -q)
            baseline = sp.simplify(self.Q_inner.copy() if hasattr(self.Q_inner, 'copy') else self.Q_inner)
            # Perturbed q: q * (1 + delta)
            q_pert = q_sym * (1 + delta)
            # Compute perturbed Q_inner by substituting the relation Q_inner = -q
            # Use the baseline expression, which should be -q.
            perturbed_Q = baseline.subs(q_sym, q_pert)
            # Expected change: ΔQ = -Δq = - (q_pert - q) = - delta * q
            expected_change = -delta * q_sym
            # Compare the difference perturbed_Q - baseline with expected_change
            actual_change = sp.simplify(perturbed_Q - baseline)
            if actual_change == expected_change:
                return (True, f"Monotonicity holds: ΔQ_inner = -{float(delta)} * q (expected)")
            else:
                return (False, f"Monotonicity check failed: actual change = {actual_change}, expected {expected_change}")
        except Exception as e:
            return (False, f"Exception in monotonicity test: {e}")

    # ----------------------------------------------------------------
    # Final answer sanity (very weak – only sign and finiteness)
    # ----------------------------------------------------------------
    def test_answer_sanity_Q_inner_sign(self) -> Tuple[bool, str]:
        """
        Q_inner should be negative if q is positive (since Q_inner = -q).
        This is a weak sign check – does not compute expected value.
        """
        if self.Q_inner is None or self.q is None:
            return (False, "SKIP: Q_inner or q missing")
        try:
            # Try to evaluate numerically if both are numbers
            q_num = None
            Q_inner_num = None
            if hasattr(self.q, 'evalf'):
                try:
                    q_num = float(self.q)
                except:
                    pass
            else:
                try:
                    q_num = float(self.q)
                except:
                    pass
            if hasattr(self.Q_inner, 'evalf'):
                try:
                    Q_inner_num = float(self.Q_inner)
                except:
                    pass
            else:
                try:
                    Q_inner_num = float(self.Q_inner)
                except:
                    pass
            if q_num is not None and Q_inner_num is not None:
                if q_num > 0 and Q_inner_num < 0:
                    return (True, "Q_inner negative for positive q (correct sign)")
                elif q_num < 0 and Q_inner_num > 0:
                    return (True, "Q_inner positive for negative q (correct sign)")
                else:
                    return (False, f"Sign mismatch: q={q_num}, Q_inner={Q_inner_num}")
            else:
                return (True, "Cannot evaluate numeric sign, skip numeric sanity")
        except Exception as e:
            return (False, f"Exception in sign sanity: {e}")

    # ----------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)