import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

class PhysicsChecker:
    """
    Robust physics checker for central force circular orbit stability problem.
    Audits solution instances for correctness based on theorem constraints,
    modeling quality, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract raw data from solution_instance safely, defaulting to None.
        Accepts objects with .results dict or plain dict.
        """
        # Initialize all attributes as None
        self.mass: Optional[float] = None
        self.radius: Optional[float] = None
        self.J_sq: Optional[float] = None
        self.stability_LHS: Optional[float] = None
        self.n_max: Optional[Union[int, float]] = None
        self.f_expr: Any = None          # sympy expression for f(r)
        self.f_prime_expr: Any = None    # sympy expression for f'(r)
        self.K: Optional[float] = None
        self.n: Optional[float] = None

        # ----- Extract results dict -----
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results: Dict[str, Any] = solution_instance.results
        elif isinstance(solution_instance, dict):
            self.results = solution_instance
        else:
            self.results = {}

        # ----- Extract givens from solution_instance attributes or results -----
        # Mass (m)
        self.mass = getattr(solution_instance, 'm', None)
        if self.mass is None:
            self.mass = self.results.get('m', None)

        # Radius (r0)
        self.radius = getattr(solution_instance, 'r0', None)
        if self.radius is None:
            self.radius = self.results.get('r0', None)

        # Force expression and derivative
        self.f_expr = getattr(solution_instance, 'f_expr', None)
        if self.f_expr is None:
            self.f_expr = self.results.get('f_expr', None)
        self.f_prime_expr = getattr(solution_instance, 'f_prime_expr', None)
        if self.f_prime_expr is None:
            self.f_prime_expr = self.results.get('f_prime_expr', None)

        # Power-law parameters
        self.K = getattr(solution_instance, 'K', None)
        if self.K is None:
            self.K = self.results.get('K', None)
        self.n = getattr(solution_instance, 'n', None)
        if self.n is None:
            self.n = self.results.get('n', None)

        # Emitted results (mandatory keys)
        self.J_sq = self.results.get('J_sq', None)
        self.stability_LHS = self.results.get('stability_LHS', None)
        self.n_max = self.results.get('n_max', None)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _has_required_numerics(self, *names: str) -> bool:
        """Check that all named attributes are not None."""
        return all(getattr(self, name, None) is not None for name in names)

    def _get_symbolic_or_skip(self, var_names: List[str]) -> Optional[Dict[str, sp.Symbol]]:
        """
        Return a dict of sympy symbols if we have symbolic expressions,
        otherwise return None (test will skip).
        """
        if self.f_expr is None and self.f_prime_expr is None:
            return None
        syms = {}
        for name in var_names:
            syms[name] = sp.Symbol(name, real=True, positive=True)
        return syms

    def _sympy_equal_to_zero(self, expr1: Any, expr2: Any) -> bool:
        """Check if sympy simplifies expr1 - expr2 to zero (symbolically)."""
        try:
            diff = sp.simplify(expr1 - expr2)
            return diff == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all mandatory result keys exist and are not None."""
        missing = []
        for key in ['J_sq', 'stability_LHS', 'n_max']:
            val = self.results.get(key, None)
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required keys (J_sq, stability_LHS, n_max) present")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that at least some givens are provided for deeper checks."""
        # We don't fail if missing, just report availability for skipping later.
        return (True, "Givens extraction attempted; deeper tests may skip if missing")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_circular_orbit_condition(self) -> Tuple[bool, str]:
        """
        Check that J_sq satisfies the circular orbit condition:
            J_sq = -m * r^3 * f(r)
        If symbolic expressions are available, verify symbolically.
        If numeric values exist, verify with math.isclose.
        """
        if self.J_sq is None:
            return (False, "SKIP: J_sq is None")

        # ---- Symbolic check ----
        syms = self._get_symbolic_or_skip(['m', 'r'])
        if syms and self.f_expr is not None:
            m_sym, r_sym = syms['m'], syms['r']
            f_expr = self.f_expr
            expected_J_sq = -m_sym * r_sym**3 * f_expr
            if self._sympy_equal_to_zero(self.J_sq, expected_J_sq):
                return (True, "Symbolic J_sq matches -m*r^3*f(r)")
            else:
                return (False, f"Symbolic mismatch: J_sq = {self.J_sq}, expected {expected_J_sq}")

        # ---- Numeric check (fallback) ----
        if not self._has_required_numerics('mass', 'radius'):
            return (False, "SKIP: numeric givens (mass, radius) missing")
        f_val = getattr(self, 'f_val', None)
        if f_val is None and self.f_expr is not None:
            try:
                f_val = float(self.f_expr.subs({'r': self.radius, 'm': self.mass}))
            except Exception:
                pass
        if f_val is None:
            return (False, "SKIP: numeric f(r) not available")
        expected_J_sq_numeric = -self.mass * self.radius**3 * f_val
        if math.isclose(float(self.J_sq), expected_J_sq_numeric, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Numeric J_sq matches expected {expected_J_sq_numeric:.3f}")
        else:
            return (False, f"Numeric mismatch: J_sq={self.J_sq}, expected {expected_J_sq_numeric:.3f}")

    def test_invariant_stability_condition(self) -> Tuple[bool, str]:
        """
        Check that stability_LHS > 0 (if numeric) and that its expression matches
        the derived formula: 3*J_sq/(m*r^4) - f'(r).
        """
        if self.stability_LHS is None:
            return (False, "SKIP: stability_LHS is None")

        # ---- Numeric positivity check ----
        try:
            lhs_val = float(self.stability_LHS)
            if lhs_val > 0:
                pass
            else:
                return (False, f"stability_LHS = {lhs_val} is not positive (unstable orbit)")
        except (TypeError, ValueError):
            pass

        # ---- Expression consistency check ----
        syms = self._get_symbolic_or_skip(['m', 'r'])
        if syms and self.J_sq is not None and self.f_prime_expr is not None:
            m_sym, r_sym = syms['m'], syms['r']
            try:
                J_expr = sp.sympify(self.J_sq) if not isinstance(self.J_sq, sp.Basic) else self.J_sq
                expected_LHS = 3 * J_expr / (m_sym * r_sym**4) - self.f_prime_expr
                if self._sympy_equal_to_zero(self.stability_LHS, expected_LHS):
                    return (True, "Symbolic stability_LHS matches 3J²/(m r⁴) - f'(r)")
                else:
                    return (False, f"Symbolic mismatch: LHS={self.stability_LHS}, expected {expected_LHS}")
            except Exception:
                pass

        # ---- Numeric fallback ----
        if self._has_required_numerics('mass', 'radius', 'J_sq'):
            f_prime_val = getattr(self, 'f_prime_val', None)
            if f_prime_val is None and self.f_prime_expr is not None:
                try:
                    f_prime_val = float(self.f_prime_expr.subs({'r': self.radius, 'm': self.mass}))
                except Exception:
                    pass
            if f_prime_val is not None:
                expected_LHS_num = 3 * float(self.J_sq) / (self.mass * self.radius**4) - f_prime_val
                try:
                    actual_LHS_num = float(self.stability_LHS)
                    if math.isclose(actual_LHS_num, expected_LHS_num, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Numeric stability_LHS matches expected {expected_LHS_num:.3f}")
                    else:
                        return (False, f"Numeric mismatch: actual {actual_LHS_num}, expected {expected_LHS_num}")
                except (TypeError, ValueError):
                    pass

        return (False, "SKIP: insufficient data to check stability condition expression")

    def test_invariant_n_max_derivation(self) -> Tuple[bool, str]:
        """
        Check that n_max is exactly 3 (or symbolically the condition n<3).
        """
        if self.n_max is None:
            return (False, "SKIP: n_max is None")
        try:
            nmax_val = float(self.n_max)
        except (TypeError, ValueError):
            if hasattr(self.n_max, 'evalf'):
                try:
                    nmax_val = float(self.n_max.evalf())
                except Exception:
                    return (False, f"Cannot evaluate n_max symbolically: {self.n_max}")
            else:
                return (False, f"n_max is not numeric: {self.n_max}")
        if math.isclose(nmax_val, 3.0, abs_tol=1e-2):
            return (True, "n_max = 3 (correct for inverse-power law)")
        else:
            return (False, f"n_max = {nmax_val}, expected 3")

    # ------------------------------------------------------------------
    # Modeling quality tests
    # ------------------------------------------------------------------
    def test_modeling_consistency_J_sq_from_pseudocode(self) -> Tuple[bool, str]:
        """
        Verify that J_sq, if numeric, satisfies the pseudocode equation:
            J_sq = -m * r^3 * f(r)
        """
        if not self._has_required_numerics('J_sq', 'mass', 'radius'):
            return (False, "SKIP: numeric values not available")
        f_val = getattr(self, 'f_val', None)
        if f_val is None:
            return (False, "SKIP: f(r) not available")
        expected = -self.mass * self.radius**3 * f_val
        if math.isclose(float(self.J_sq), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"J_sq = {self.J_sq} is consistent with pseudocode")
        else:
            return (False, f"J_sq = {self.J_sq}, but pseudocode gives {expected}")

    def test_modeling_stability_LHS_expression(self) -> Tuple[bool, str]:
        """
        Modeling quality: check that stability_LHS is positive (if numeric)
        and matches the theoretical expression from explanation steps.
        """
        if self.stability_LHS is None:
            return (False, "SKIP: stability_LHS is None")
        try:
            lhs = float(self.stability_LHS)
            if lhs <= 0:
                return (False, f"stability_LHS = {lhs} is not positive; orbit may be unstable")
            else:
                return (True, f"stability_LHS = {lhs} > 0 (stable condition holds)")
        except (TypeError, ValueError):
            pass
        return (False, "SKIP: cannot evaluate stability_LHS numerically")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturbation-based)
    # ------------------------------------------------------------------
    def test_monotonicity_stability_with_n(self) -> Tuple[bool, str]:
        """
        Perturb the force exponent n by +0.2 and verify that the stability
        condition LHS changes in the correct direction.
        Uses the analytical expression LHS = K * r^{-n-1} * (3 - n).
        If numeric values are available, compute baseline LHS, perturb n,
        compute perturbed LHS, and check monotonic trend.
        """
        # Need K, radius, n to compute LHS analytically.
        if not self._has_required_numerics('K', 'radius', 'n'):
            return (False, "SKIP: missing K, radius, or n")
        try:
            K = float(self.K)
            r = float(self.radius)
            n_base = float(self.n)
            # Perturbation: increase n by 0.2 (small relative, but large enough to see sign)
            delta = 0.2
            n_pert = n_base + delta

            # Compute baseline LHS using the analytical formula
            # (re-derived from theorem constraints, not copied from answer)
            LHS_base = K * (r ** (-n_base - 1)) * (3 - n_base)
            LHS_pert = K * (r ** (-n_pert - 1)) * (3 - n_pert)

            # Expected monotonic behavior:
            # For n < 3, increasing n makes LHS smaller (more negative).
            # For n > 3, LHS is negative and becoming more negative is also monotonic.
            # The derivative dLHS/dn = -K*r^{-n-1}*( (3-n)*ln(r) + 1 )
            # For physically plausible r>1, ln(r) > 0 so derivative is negative.
            # For 0<r<1, ln(r) negative, derivative could be positive.
            # We'll test a simpler case: assume r >= 1 (typical physics).
            if r < 1:
                # If radius is less than 1 (unusual), skip because monotonicity may reverse.
                return (False, "SKIP: radius < 1, monotonicity may be non-trivial")
            # For r >= 1, LHS should decrease when n increases.
            if LHS_pert < LHS_base:
                return (True, f"Perturbation test passed: n increased from {n_base} to {n_pert}, "
                              f"LHS changed from {LHS_base:.3f} to {LHS_pert:.3f} (decrease, consistent)")
            else:
                return (False, f"Perturbation test failed: n increased from {n_base} to {n_pert}, "
                              f"LHS changed from {LHS_base:.3f} to {LHS_pert:.3f} (expected decrease, got increase)")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    def test_monotonicity_stability_with_radius(self) -> Tuple[bool, str]:
        """
        Perturb the orbit radius by +5% and verify that the stability condition
        LHS changes in the expected direction.
        For f(r) = -K/r^n, the condition LHS = K * r^{-n-1} * (3-n) has derivative
        dLHS/dr = K * (-n-1) * r^{-n-2} * (3-n). For n < 3, LHS > 0 and
        the sign depends on n. For n > 1, -n-1 < 0 so LHS decreases with r.
        For n < 1, LHS increases with r. We'll check a simple case:
        if n > 1, increasing r should decrease LHS; if n < 1, increasing r increases LHS.
        We'll require n to be given for this test.
        """
        if not self._has_required_numerics('K', 'radius', 'n'):
            return (False, "SKIP: missing K, radius, or n")
        try:
            K = float(self.K)
            r_base = float(self.radius)
            n_val = float(self.n)
            # Only test if n is known and r_base > 0
            if r_base <= 0:
                return (False, "Radius must be positive")
            delta = 0.05 * r_base  # 5% increase
            r_pert = r_base + delta

            LHS_base = K * (r_base ** (-n_val - 1)) * (3 - n_val)
            LHS_pert = K * (r_pert ** (-n_val - 1)) * (3 - n_val)

            # Determine expected monotonic direction:
            # Derivative: dLHS/dr = K * (-n-1) * r^{-n-2} * (3-n)
            # For n > 1: (-n-1) < 0, so LHS decreases with r.
            # For n < 1: (-n-1) < 0, so LHS increases with r? Wait, (-n-1) is always negative if n > -1.
            # Actually (-n-1) is negative for n > -1, so dLHS/dr < 0 for n < 3? 
            # Let's check: LHS = K * (3-n) * r^{-n-1}. For n < 3, (3-n) > 0, so LHS positive.
            # dLHS/dr = K * (3-n) * (-n-1) * r^{-n-2}. Since (3-n)>0 and r^{-n-2}>0,
            # sign is sign of (-n-1). For n > -1, (-n-1) < 0 -> LHS decreases with r.
            # For n < -1, (-n-1) > 0 -> LHS increases with r, but typical central forces have n > -2.
            # We'll accept any monotonicity that matches the sign of (-n-1).
            # For simplicity, if n > -1, expect LHS_pert < LHS_base; else expect LHS_pert > LHS_base.
            if n_val > -1:
                expected_decrease = True
                condition = LHS_pert < LHS_base
                direction = "decrease"
            else:
                expected_decrease = False
                condition = LHS_pert > LHS_base
                direction = "increase"

            if condition:
                return (True, f"Perturbation test passed: radius increased from {r_base:.3f} to {r_pert:.3f}, "
                              f"LHS changed from {LHS_base:.3f} to {LHS_pert:.3f} ({direction}, consistent)")
            else:
                return (False, f"Perturbation test failed: radius increased, LHS changed from {LHS_base:.3f} to {LHS_pert:.3f}, "
                              f"expected {direction} but did not occur")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_n_max(self) -> Tuple[bool, str]:
        """
        Weak sanity: n_max should be a finite positive number, and likely between 0 and 5.
        Do NOT use math.isclose or re-derive.
        """
        if self.n_max is None:
            return (False, "SKIP: n_max is None")
        try:
            n_max_float = float(self.n_max)
        except (TypeError, ValueError):
            return (False, "n_max is not numeric")
        if 0 < n_max_float < 5:
            return (True, f"n_max = {n_max_float} is in reasonable range (0,5)")
        else:
            return (False, f"n_max = {n_max_float} is outside expected range")

    # ------------------------------------------------------------------
    # Main runner – discovers all test methods and executes them
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them safely,
        and return (all_pass, logs_list).
        """
        logs: List[str] = []
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
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)


# ===== Example usage (commented out) =====
# class SampleSolution:
#     def __init__(self):
#         self.results = {
#             'J_sq': sp.symbols('J_sq'),
#             'stability_LHS': sp.symbols('stability_LHS'),
#             'n_max': 3,
#             'f_expr': -sp.Symbol('K') / sp.Symbol('r')**sp.Symbol('n'),
#             'f_prime_expr': sp.Symbol('n') * sp.Symbol('K') / sp.Symbol('r')**(sp.Symbol('n')+1),
#         }
#         self.m = 1.0
#         self.r0 = 1.0
#         self.K = 1.0
#         self.n = 2.0
#         self.f_val = -1.0
#         self.f_prime_val = 2.0
#
# if __name__ == "__main__":
#     sol = SampleSolution()
#     checker = PhysicsChecker(sol)
#     ok, logs = checker.run_all_tests()
#     for l in logs:
#         print(l)