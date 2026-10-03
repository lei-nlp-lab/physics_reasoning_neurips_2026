import math
import sympy as sp
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits a solution for a system of N fixed non-interacting magnetic moments
    in a uniform external magnetic field B, at thermal equilibrium temperature T.
    Checks partition function, specific heat, average magnetic moment, susceptibility,
    and high-temperature Curie law.
    """

    def __init__(self, solution_instance: Any):
        # Extract input parameters with safe defaults (None if missing)
        self.N = getattr(solution_instance, 'N', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.B = getattr(solution_instance, 'B', None)
        self.T = getattr(solution_instance, 'T', None)
        self.k = getattr(solution_instance, 'k', None)

        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Result quantities (use None if missing)
        self.Z = self.results.get('Z', None)
        self.C = self.results.get('C', None)
        self.M_bar = self.results.get('M_bar', None)
        self.chi = self.results.get('chi', None)
        self.chi_highT = self.results.get('chi_highT', None)

        # Derived quantity x = beta*mu0*B (only if all required givens exist)
        self.x = None
        if all(v is not None for v in [self.mu0, self.B, self.k, self.T]):
            try:
                self.x = self.mu0 * self.B / (self.k * self.T)
            except ZeroDivisionError:
                pass  # T=0 leads to division error, leave None

    # -------------------------------------------------------------------------
    # Interface tests – ensure required keys and valid types
    # -------------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all five result keys are present and not None."""
        required = ['Z', 'C', 'M_bar', 'chi', 'chi_highT']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, "All required result keys present."

    def test_interface_givens_positive(self) -> Tuple[bool, str]:
        """Check that givens N, mu0, B, T, k are positive finite numbers."""
        for name, val in [('N', self.N), ('mu0', self.mu0), ('B', self.B),
                          ('T', self.T), ('k', self.k)]:
            if val is None:
                return False, f"{name} is None."
            if not isinstance(val, (int, float)):
                return False, f"{name} is not numeric: {type(val)}."
            if val <= 0:
                return False, f"{name} is not positive: {val}."
        return True, "All givens present and positive."

    def test_interface_x_defined(self) -> Tuple[bool, str]:
        """Check that x = mu0*B/(k*T) can be computed."""
        if self.x is None:
            return False, "x = mu0*B/(k*T) could not be computed. Check givens."
        if not math.isfinite(self.x):
            return False, f"x is not finite: {self.x}."
        return True, f"x = {self.x} computed."

    # -------------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity checks (use math.isclose)
    # -------------------------------------------------------------------------
    def test_invariant_partition_function_single(self) -> Tuple[bool, str]:
        """Check that Z^(1/N) equals exp(x)+exp(-x) = 2*cosh(x)."""
        if None in (self.Z, self.N, self.x):
            return False, "SKIP: missing Z, N, or x."
        try:
            root = self.Z ** (1.0 / self.N)
        except (ValueError, ZeroDivisionError) as e:
            return False, f"Could not compute Z^(1/N): {e}"
        expected = 2.0 * math.cosh(self.x)
        if math.isclose(root, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Z^(1/N) ≈ {root} close to 2cosh(x) ≈ {expected}."
        return False, f"Z^(1/N) = {root} differs from 2cosh(x) = {expected}."

    def test_invariant_partition_function_alt(self) -> Tuple[bool, str]:
        """Check that Z = (exp(x)+exp(-x))^N."""
        if None in (self.Z, self.N, self.x):
            return False, "SKIP: missing Z, N, or x."
        expected = (math.exp(self.x) + math.exp(-self.x)) ** self.N
        if math.isclose(self.Z, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Z matches (exp(x)+exp(-x))^N."
        return False, f"Z = {self.Z}, expected ≈ {expected}."

    def test_invariant_specific_heat(self) -> Tuple[bool, str]:
        """Check that C = N*k*x^2 / cosh^2(x)."""
        if None in (self.C, self.N, self.k, self.x):
            return False, "SKIP: missing C, N, k, or x."
        if self.k <= 0:
            return False, "k is not positive."
        expected = self.N * self.k * (self.x ** 2) / (math.cosh(self.x) ** 2)
        if math.isclose(self.C, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"C matches N*k*x^2/cosh^2(x) ≈ {expected}."
        return False, f"C = {self.C} differs from expected = {expected}."

    def test_invariant_magnetic_moment(self) -> Tuple[bool, str]:
        """Check that M_bar = N*mu0*tanh(x)."""
        if None in (self.M_bar, self.N, self.mu0, self.x):
            return False, "SKIP: missing M_bar, N, mu0, or x."
        expected = self.N * self.mu0 * math.tanh(self.x)
        if math.isclose(self.M_bar, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"M_bar matches N*mu0*tanh(x) ≈ {expected}."
        return False, f"M_bar = {self.M_bar} differs from expected = {expected}."

    def test_invariant_susceptibility(self) -> Tuple[bool, str]:
        """Check that chi = (N*mu0^2)/(k*T) * sech^2(x)."""
        if None in (self.chi, self.N, self.mu0, self.k, self.T, self.x):
            return False, "SKIP: missing chi, N, mu0, k, T, or x."
        expected = (self.N * self.mu0 ** 2) / (self.k * self.T) * (1.0 / math.cosh(self.x) ** 2)
        if math.isclose(self.chi, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"chi matches formula ≈ {expected}."
        return False, f"chi = {self.chi} differs from expected = {expected}."

    def test_invariant_highT_susceptibility(self) -> Tuple[bool, str]:
        """Check that chi_highT ≈ N*mu0^2/(k*T) (Curie law)."""
        if None in (self.chi_highT, self.N, self.mu0, self.k, self.T):
            return False, "SKIP: missing chi_highT, N, mu0, k, T."
        expected = (self.N * self.mu0 ** 2) / (self.k * self.T)
        # Use a wider tolerance since high-T limit is approximate
        if math.isclose(self.chi_highT, expected, rel_tol=5e-2, abs_tol=1e-2):
            return True, f"chi_highT ≈ {expected} (Curie law)."
        return False, f"chi_highT = {self.chi_highT} differs from {expected}."

    # -------------------------------------------------------------------------
    # Modeling/trace tests – verify logical consistency with explanation steps
    # -------------------------------------------------------------------------
    def test_trace_partition_function_method(self) -> Tuple[bool, str]:
        """Soft check that Z is consistent with single-spin partition function raised to N."""
        if None in (self.Z, self.N, self.x):
            return False, "SKIP: missing Z, N, or x."
        try:
            root = self.Z ** (1.0 / self.N)
        except:
            return False, "Could not compute Z^(1/N)."
        expected = 2.0 * math.cosh(self.x)
        if math.isclose(root, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, "Modeling: Z consistent with (2cosh(x))^N."
        return False, f"Modeling fail: Z^(1/N) = {root}, 2cosh(x) = {expected}."

    def test_trace_specific_heat_positivity(self) -> Tuple[bool, str]:
        """Check that C is non-negative (physical requirement)."""
        if self.C is None:
            return False, "SKIP: C not available."
        if self.C >= 0:
            return True, f"C = {self.C} is non-negative."
        return False, f"C = {self.C} is negative."

    # -------------------------------------------------------------------------
    # Monotonicity tests – explicit input perturbation and comparison
    # -------------------------------------------------------------------------
    def test_monotonicity_moment_increases_with_B(self) -> Tuple[bool, str]:
        """
        Verify that M_bar increases when B is increased by 1%.
        Since we cannot change the actual input, we use the theorem formula
        to compute M_bar at perturbed B and compare it to the given M_bar.
        """
        if None in (self.M_bar, self.N, self.mu0, self.B, self.T, self.k):
            return False, "SKIP: missing required givens or M_bar."
        if self.B <= 0 or self.T <= 0 or self.k <= 0:
            return False, "SKIP: non-positive input."
        delta = 0.01 * self.B  # 1% increase
        B_plus = self.B + delta
        x_plus = self.mu0 * B_plus / (self.k * self.T)
        M_bar_plus = self.N * self.mu0 * math.tanh(x_plus)
        if self.M_bar < M_bar_plus:
            return True, (f"M_bar = {self.M_bar:.6g} < M_bar_plus = {M_bar_plus:.6g} "
                          f"(B increased by 1% to {B_plus:.6g})")
        else:
            return False, (f"M_bar = {self.M_bar:.6g} not less than perturbed "
                           f"{M_bar_plus:.6g} (B increased).")

    # -------------------------------------------------------------------------
    # Answer sanity test (weak, at most one)
    # -------------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity checks on the results:
        - Z > 0
        - C >= 0
        - M_bar between -N*mu0 and N*mu0
        - chi >= 0
        - chi_highT > 0
        No re-derivation, no isclose, no expected values.
        """
        issues = []
        # Z positivity
        if self.Z is not None:
            if self.Z <= 0:
                issues.append(f"Z = {self.Z} is not positive.")
        # C non-negativity
        if self.C is not None:
            if self.C < 0:
                issues.append(f"C = {self.C} is negative.")
        # M_bar range
        if self.M_bar is not None and self.N is not None and self.mu0 is not None:
            max_mag = self.N * self.mu0
            if abs(self.M_bar) > max_mag + 1e-9:
                issues.append(f"|M_bar| = {abs(self.M_bar):g} exceeds N*mu0 = {max_mag:g}.")
        # chi non-negativity
        if self.chi is not None:
            if self.chi < 0:
                issues.append(f"chi = {self.chi} is negative.")
        # chi_highT positivity
        if self.chi_highT is not None:
            if self.chi_highT <= 0:
                issues.append(f"chi_highT = {self.chi_highT} is not positive.")
        if issues:
            return False, "; ".join(issues)
        return True, "All weak sanity checks pass."

    # -------------------------------------------------------------------------
    # Run all tests
    # -------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods and run them, collecting results."""
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
        return all_pass, logs