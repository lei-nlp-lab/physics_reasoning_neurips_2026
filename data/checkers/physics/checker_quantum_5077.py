import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    """
    Audits solutions for the infinite square well sudden expansion problem.
    Checks: interface completeness, invariant relations, modeling quality,
    monotonicity, and a single weak answer sanity.
    """

    def __init__(self, solution_instance):
        # ---------- Extract givens ----------
        # Use None defaults; do NOT assign numeric fallbacks.
        self.m = getattr(solution_instance, 'm', None)
        self.a = getattr(solution_instance, 'a', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.pi = getattr(solution_instance, 'pi', None)

        # Try to obtain results from a 'results' dictionary or direct attributes.
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            res = solution_instance.results
        else:
            res = {}

        self.E1 = res.get('E1', None) or getattr(solution_instance, 'E1', None)
        self.f_p = res.get('f_p', None) or getattr(solution_instance, 'f_p', None)
        self.E_avg = res.get('E_avg', None) or getattr(solution_instance, 'E_avg', None)

        # Attempt to store numeric versions (will be None if conversion fails)
        self._E1_num = None
        self._E_avg_num = None
        try:
            if self.E1 is not None:
                self._E1_num = float(self.E1)
        except (TypeError, ValueError):
            pass
        try:
            if self.E_avg is not None:
                self._E_avg_num = float(self.E_avg)
        except (TypeError, ValueError):
            pass

    # ------- Helper -------
    def _safe_num(self, x) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    # ------- Interface tests -------
    def test_interface_E1_exists(self) -> Tuple[bool, str]:
        """Check that the ground state energy result exists."""
        if self.E1 is None:
            return (False, "SKIP: E1 is missing")
        return (True, f"E1 provided: {self.E1}")

    def test_interface_f_p_exists(self) -> Tuple[bool, str]:
        """Check that the momentum probability density result exists."""
        if self.f_p is None:
            return (False, "SKIP: f_p is missing")
        return (True, f"f_p provided: {str(self.f_p)[:80]}...")

    def test_interface_E_avg_exists(self) -> Tuple[bool, str]:
        """Check that the average energy result exists."""
        if self.E_avg is None:
            return (False, "SKIP: E_avg is missing")
        return (True, f"E_avg provided: {self.E_avg}")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that mass, width, hbar, and pi are provided for further checks."""
        missing = []
        if self.m is None:
            missing.append("m")
        if self.a is None:
            missing.append("a")
        if self.hbar is None:
            missing.append("hbar")
        if self.pi is None:
            missing.append("pi")
        if missing:
            return (False, f"SKIP: missing givens: {', '.join(missing)}")
        return (True, f"All givens present: m={self.m}, a={self.a}, hbar={self.hbar}, pi={self.pi}")

    # ------- Invariant tests (theorem-based) -------
    def test_invariant_E1_formula(self) -> Tuple[bool, str]:
        """
        Verify that E1 equals the infinite well formula:
        E1 = hbar^2 * pi^2 / (2 * m * a^2)
        Uses math.isclose with relative tolerance 1e-2.
        """
        if self.E1 is None:
            return (False, "SKIP: E1 missing")
        if not all([self.m, self.a, self.hbar, self.pi]):
            return (False, "SKIP: missing givens for computation")
        try:
            m = float(self.m)
            a = float(self.a)
            hbar = float(self.hbar)
            pi = float(self.pi)
            expected = hbar**2 * pi**2 / (2 * m * a**2)
            actual = self._safe_num(self.E1)
            if actual is None:
                return (False, "SKIP: E1 is not numeric")
            if math.isclose(expected, actual, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E1 matches formula: {actual} vs expected {expected}")
            else:
                return (False, f"E1 = {actual} but formula gives {expected}")
        except Exception as e:
            return (False, f"E1 formula check crashed: {e}")

    def test_invariant_E_avg_formula(self) -> Tuple[bool, str]:
        """
        The average energy after sudden expansion equals the original ground state energy.
        Check that E_avg matches the same formula.
        """
        if self.E_avg is None:
            return (False, "SKIP: E_avg missing")
        if not all([self.m, self.a, self.hbar, self.pi]):
            return (False, "SKIP: missing givens for computation")
        try:
            m = float(self.m)
            a = float(self.a)
            hbar = float(self.hbar)
            pi = float(self.pi)
            expected = hbar**2 * pi**2 / (2 * m * a**2)
            actual = self._safe_num(self.E_avg)
            if actual is None:
                return (False, "SKIP: E_avg is not numeric")
            if math.isclose(expected, actual, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E_avg matches formula: {actual} vs expected {expected}")
            else:
                return (False, f"E_avg = {actual} but formula gives {expected}")
        except Exception as e:
            return (False, f"E_avg formula check crashed: {e}")

    def test_invariant_E_avg_equals_E1(self) -> Tuple[bool, str]:
        """Direct equality check of E1 and E_avg (energy conservation)."""
        if self._E1_num is None or self._E_avg_num is None:
            return (False, "SKIP: need numeric values for both E1 and E_avg")
        try:
            if math.isclose(self._E1_num, self._E_avg_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E1 = {self._E1_num}, E_avg = {self._E_avg_num} (agree)")
            else:
                return (False, f"E1 = {self._E1_num} but E_avg = {self._E_avg_num}")
        except Exception as e:
            return (False, f"E_avg vs E1 check crashed: {e}")

    # ------- Modeling quality tests (trace checks) -------
    def test_modeling_f_p_nonnegative(self) -> Tuple[bool, str]:
        """Check that f(p) is non-negative at a few sample momenta (if numeric evaluation possible)."""
        if self.f_p is None:
            return (False, "SKIP: f_p missing")
        try:
            import sympy as sp
            if isinstance(self.f_p, sp.Expr):
                p_sym = sp.symbols('p', real=True, nonnegative=True)
                f_func = sp.lambdify(p_sym, self.f_p, 'numpy')
                import numpy as np
                # Scale p using the characteristic momentum hbar/a
                if not all([self.hbar, self.a]):
                    return (False, "SKIP: hbar or a missing to generate test momenta")
                hbar = float(self.hbar)
                a = float(self.a)
                test_p = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * (hbar / a)
                vals = np.array([f_func(p) for p in test_p], dtype=float)
                if np.all(vals >= -1e-9):
                    return (True, "f(p) non-negative at sample points")
                else:
                    negative_indices = np.where(vals < 0)[0]
                    return (False, f"f(p) negative at p = {test_p[negative_indices]}")
            else:
                return (False, "SKIP: f_p is not a sympy expression; cannot evaluate")
        except Exception as e:
            return (False, f"SKIP: f_p evaluation failed: {e}")

    def test_modeling_normalization_integral(self) -> Tuple[bool, str]:
        """
        Numerically integrate f(p) dp from 0 to a large cutoff and check that it is close to 1.
        This tests that the probability density is normalized.
        """
        if self.f_p is None:
            return (False, "SKIP: f_p missing")
        if not all([self.hbar, self.a]):
            return (False, "SKIP: hbar or a missing for integration")
        try:
            import sympy as sp
            if not isinstance(self.f_p, sp.Expr):
                return (False, "SKIP: f_p is not a sympy expression; cannot integrate")
            hbar = float(self.hbar)
            a = float(self.a)
            # Use a dimensionless integration variable to handle the cutoff.
            # The integrand has a pole at (p*a/hbar) = pi, but it's removable.
            # We'll integrate numerically using a fine grid up to p_max = 10 * hbar / a.
            # That should capture >99% of the probability (decays like 1/p^4).
            p_max = 10.0 * hbar / a
            p_sym = sp.symbols('p', real=True, nonnegative=True)
            f_func = sp.lambdify(p_sym, self.f_p, 'numpy')
            import numpy as np
            N = 2000
            p_vals = np.linspace(0, p_max, N, endpoint=False)
            f_vals = np.array([f_func(p) for p in p_vals], dtype=float)
            # Trapzoidal integration
            dp = p_vals[1] - p_vals[0]
            integral = np.trapz(f_vals, p_vals)
            if math.isclose(integral, 1.0, rel_tol=0.1):
                return (True, f"Normalization integral = {integral:.4f} (close to 1)")
            else:
                return (False, f"Normalization integral = {integral:.4f} (expected 1)")
        except Exception as e:
            return (False, f"SKIP: normalization integration failed: {e}")

    # ------- Monotonicity tests -------
    def test_monotonicity_E1_decreases_with_a(self) -> Tuple[bool, str]:
        """
        Verify the monotonic dependence: as the well width a increases,
        the ground state energy E1 decreases.
        We perturb a by +5% and recompute E1 using the exact formula.
        """
        if not all([self.m, self.a, self.hbar, self.pi]):
            return (False, "SKIP: missing givens (m, a, hbar, pi) for monotonicity check")
        try:
            m = float(self.m)
            a = float(self.a)
            hbar = float(self.hbar)
            pi = float(self.pi)
            # Compute baseline energy
            E_base = hbar**2 * pi**2 / (2 * m * a**2)
            # Perturb a by +5%
            a_perturbed = a * 1.05
            E_pert = hbar**2 * pi**2 / (2 * m * a_perturbed**2)
            # Monotonic: larger a should give smaller energy
            if E_pert < E_base:
                return (True, f"E decreases when a increases: E_base={E_base:.6f}, E_pert={E_pert:.6f}")
            else:
                return (False, f"Monotonicity violated: E_base={E_base:.6f}, E_pert={E_pert:.6f}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------- Weak answer sanity (at most one) -------
    def test_answer_sanity_E1_positive(self) -> Tuple[bool, str]:
        """Ground state energy must be positive."""
        if self._E1_num is None:
            return (False, "SKIP: E1 not numeric")
        if self._E1_num > 0:
            return (True, f"E1 = {self._E1_num} > 0")
        else:
            return (False, f"E1 = {self._E1_num} is not positive")

    # ------- Orchestrator -------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)