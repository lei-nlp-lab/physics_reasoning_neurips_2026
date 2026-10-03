import math
import traceback
from typing import Any, Tuple

class PhysicsChecker:
    """
    Physics auditor for rotational specific heat and entropy of a dilute diatomic gas.
    Operates on a solution_instance that must contain a 'results' dict with keys:
      'C_rot_high', 'S_rot_high', 'C_rot_low', 'S_rot_low'
    Optionally, the instance may also have attributes: I, T, k_B, hbar, N_A.
    """
    
    def __init__(self, solution_instance: Any):
        # Safely extract results dict
        self.results = getattr(solution_instance, 'results', None) or {}
        
        # Extract physical constants (use None if missing)
        self.I = getattr(solution_instance, 'I', None)                  # moment of inertia
        self.T = getattr(solution_instance, 'T', None)                  # temperature
        self.k_B = getattr(solution_instance, 'k_B', None)              # Boltzmann constant
        self.hbar = getattr(solution_instance, 'hbar', None)            # reduced Planck constant
        self.N_A = getattr(solution_instance, 'N_A', None)              # Avogadro's number
        
        # Derived quantities
        self.R = self._compute_R()
        self.theta_rot = self._compute_theta_rot()
    
    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _to_float(self, value: Any) -> float | None:
        """Convert value to float if possible, else None."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _compute_R(self) -> float | None:
        if self.N_A is not None and self.k_B is not None:
            try:
                return float(self.N_A * self.k_B)
            except (ValueError, TypeError):
                return None
        return None
    
    def _compute_theta_rot(self) -> float | None:
        """Rotational temperature theta_rot = hbar^2/(2 I k_B)."""
        if self.hbar is not None and self.I is not None and self.k_B is not None:
            try:
                return float(self.hbar**2 / (2 * self.I * self.k_B))
            except (ValueError, TypeError, ZeroDivisionError):
                return None
        return None
    
    def _has_required(self, *names: str) -> bool:
        """Check that all named attributes are not None."""
        return all(getattr(self, n, None) is not None for n in names)
    
    # ----------------------------------------------------------------------
    # Interface Tests
    # ----------------------------------------------------------------------
    def test_interface_results_keys(self) -> Tuple[bool, str]:
        required = ['C_rot_high', 'S_rot_high', 'C_rot_low', 'S_rot_low']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")
    
    def test_interface_result_types_numeric(self) -> Tuple[bool, str]:
        """Each result value should be numeric (int, float, or sympy numeric)."""
        for key, val in self.results.items():
            if not isinstance(val, (int, float)) and not hasattr(val, '__float__'):
                return (False, f"Result '{key}' has non-numeric type {type(val).__name__}")
            if self._to_float(val) is None:
                return (False, f"Result '{key}' cannot be converted to float")
        return (True, "All result values are numeric")
    
    # ----------------------------------------------------------------------
    # Invariant / Theorem-based Sanity Checks
    # ----------------------------------------------------------------------
    def test_invariant_high_C_rot_classical(self) -> Tuple[bool, str]:
        """High‑T limit: C_rot should equal R (classical equipartition)."""
        C_high = self._to_float(self.results.get('C_rot_high'))
        if C_high is None or self.R is None:
            return (False, "SKIP: missing C_rot_high or R")
        if not math.isclose(C_high, self.R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_rot_high = {C_high} is not close to R = {self.R}")
        return (True, f"C_rot_high ≈ R ({self.R})")
    
    def test_invariant_low_C_rot_positive(self) -> Tuple[bool, str]:
        """Low‑T specific heat must be positive (exponential small but positive)."""
        C_low = self._to_float(self.results.get('C_rot_low'))
        if C_low is None:
            return (False, "SKIP: missing C_rot_low")
        if C_low <= 0:
            return (False, f"C_rot_low = {C_low} is not positive")
        return (True, "C_rot_low > 0")
    
    def test_invariant_low_S_rot_positive(self) -> Tuple[bool, str]:
        S_low = self._to_float(self.results.get('S_rot_low'))
        if S_low is None:
            return (False, "SKIP: missing S_rot_low")
        if S_low <= 0:
            return (False, f"S_rot_low = {S_low} is not positive")
        return (True, "S_rot_low > 0")
    
    def test_invariant_high_S_rot_positive(self) -> Tuple[bool, str]:
        S_high = self._to_float(self.results.get('S_rot_high'))
        if S_high is None:
            return (False, "SKIP: missing S_rot_high")
        if S_high <= 0:
            return (False, f"S_rot_high = {S_high} is not positive")
        return (True, "S_rot_high > 0")
    
    # ----------------------------------------------------------------------
    # Modeling / Trace Consistency Checks
    # ----------------------------------------------------------------------
    def test_modeling_low_C_rot_form(self) -> Tuple[bool, str]:
        """Low‑T C_rot should follow 3R*(hbar^2/(I k_B T))^2 * exp(-hbar^2/(I k_B T))."""
        C_low = self._to_float(self.results.get('C_rot_low'))
        if C_low is None or not self._has_required('I', 'T', 'k_B', 'hbar', 'R'):
            return (False, "SKIP: missing data for low‑T C_rot form")
        try:
            prefactor = 3 * self.R * (self.hbar**2 / (self.I * self.k_B * self.T))**2
            exponent = -self.hbar**2 / (self.I * self.k_B * self.T)
            expected = prefactor * math.exp(exponent)
        except (ValueError, ZeroDivisionError):
            return (False, "Error computing expected low‑T C_rot")
        if not math.isclose(C_low, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_rot_low = {C_low} does not match expected {expected}")
        return (True, "C_rot_low matches predicted exponential form")
    
    def test_modeling_low_S_rot_form(self) -> Tuple[bool, str]:
        """Low‑T S_rot should follow 3R*(1 + hbar^2/(I k_B T)) * exp(-hbar^2/(I k_B T))."""
        S_low = self._to_float(self.results.get('S_rot_low'))
        if S_low is None or not self._has_required('I', 'T', 'k_B', 'hbar', 'R'):
            return (False, "SKIP: missing data for low‑T S_rot form")
        try:
            factor = 1 + self.hbar**2 / (self.I * self.k_B * self.T)
            exponent = -self.hbar**2 / (self.I * self.k_B * self.T)
            expected = 3 * self.R * factor * math.exp(exponent)
        except (ValueError, ZeroDivisionError):
            return (False, "Error computing expected low‑T S_rot")
        if not math.isclose(S_low, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S_rot_low = {S_low} does not match expected {expected}")
        return (True, "S_rot_low matches predicted exponential form")
    
    def test_modeling_high_S_rot_form(self) -> Tuple[bool, str]:
        """High‑T S_rot should be R*(1 + ln(T/theta_rot))."""
        S_high = self._to_float(self.results.get('S_rot_high'))
        if S_high is None or self.theta_rot is None or self.R is None:
            return (False, "SKIP: missing data for high‑T S_rot form")
        try:
            expected = self.R * (1 + math.log(self.T / self.theta_rot))
        except (ValueError, ZeroDivisionError):
            return (False, "Error computing expected high‑T S_rot (invalid theta_rot or T)")
        if not math.isclose(S_high, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S_rot_high = {S_high} does not match expected {expected}")
        return (True, "S_rot_high matches classical partition function form")
    
    # ----------------------------------------------------------------------
    # Monotonicity Checks (perturb input T and compare to baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_high_S_rot_increases_with_T(self) -> Tuple[bool, str]:
        """
        High‑T entropy should increase with temperature.
        Perturb T by +10% and verify that the computed S (using the solution's functional form) 
        is larger than the solution's S_rot_high value.
        (Assumes the modeling test has verified the functional form for S_rot_high.)
        """
        S_high = self._to_float(self.results.get('S_rot_high'))
        if S_high is None or self.theta_rot is None or self.R is None or self.T is None:
            return (False, "SKIP: missing S_rot_high, theta_rot, R, or T")
        try:
            T_perturbed = 1.1 * self.T  # 10% increase
            S_perturbed = self.R * (1 + math.log(T_perturbed / self.theta_rot))
            if not S_perturbed > S_high:
                return (False, f"Computed S_perturbed({T_perturbed:.3g}) = {S_perturbed:.6g} is not greater than "
                               f"solution's S_rot_high = {S_high:.6g} (should increase with T)")
        except (ValueError, ZeroDivisionError):
            return (False, "Error in monotonicity computation")
        return (True, "High‑T S_rot increases with T (monotonic)")
    
    def test_monotonicity_low_C_rot_increases_with_T(self) -> Tuple[bool, str]:
        """
        Low‑T specific heat should increase with temperature (positive slope).
        Perturb T by +10% and compute C using the same exponential form; 
        the perturbed value should exceed the solution's C_rot_low.
        """
        C_low = self._to_float(self.results.get('C_rot_low'))
        if C_low is None or not self._has_required('I', 'T', 'k_B', 'hbar', 'R'):
            return (False, "SKIP: missing data for low‑T C_rot monotonicity")
        try:
            T_perturbed = 1.1 * self.T
            def compute_C(T_val):
                pref = 3 * self.R * (self.hbar**2 / (self.I * self.k_B * T_val))**2
                exp = math.exp(-self.hbar**2 / (self.I * self.k_B * T_val))
                return pref * exp
            C_perturbed = compute_C(T_perturbed)
            if not C_perturbed > C_low:
                return (False, f"Computed C_perturbed(T={T_perturbed:.3g}) = {C_perturbed:.6g} is not greater than "
                               f"solution's C_rot_low = {C_low:.6g} (should increase with T)")
        except Exception:
            return (False, "Error in low‑T C_rot monotonicity computation")
        return (True, "Low‑T C_rot increases with T (monotonic)")
    
    def test_monotonicity_low_S_rot_increases_with_T(self) -> Tuple[bool, str]:
        """
        Low‑T entropy should increase with temperature.
        Perturb T by +10% and compute S using the low‑T formula; 
        the perturbed value should exceed the solution's S_rot_low.
        """
        S_low = self._to_float(self.results.get('S_rot_low'))
        if S_low is None or not self._has_required('I', 'T', 'k_B', 'hbar', 'R'):
            return (False, "SKIP: missing data for low‑T S_rot monotonicity")
        try:
            T_perturbed = 1.1 * self.T
            def compute_S(T_val):
                factor = 1 + self.hbar**2 / (self.I * self.k_B * T_val)
                exp = math.exp(-self.hbar**2 / (self.I * self.k_B * T_val))
                return 3 * self.R * factor * exp
            S_perturbed = compute_S(T_perturbed)
            if not S_perturbed > S_low:
                return (False, f"Computed S_perturbed(T={T_perturbed:.3g}) = {S_perturbed:.6g} is not greater than "
                               f"solution's S_rot_low = {S_low:.6g} (should increase with T)")
        except Exception:
            return (False, "Error in low‑T S_rot monotonicity computation")
        return (True, "Low‑T S_rot increases with T (monotonic)")
    
    # ----------------------------------------------------------------------
    # Weak Answer Sanity (single test, no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_ranges(self) -> Tuple[bool, str]:
        """Check that all results are finite, non‑negative, and have plausible magnitudes."""
        checks = []
        keys = ['C_rot_high', 'S_rot_high', 'C_rot_low', 'S_rot_low']
        for key in keys:
            val = self._to_float(self.results.get(key))
            if val is None:
                checks.append(f"{key}: missing or non‑numeric")
                continue
            if not math.isfinite(val):
                checks.append(f"{key}: not finite ({val})")
            if val < 0:
                checks.append(f"{key}: negative ({val})")
        # Only weak magnitude bounds: high‑T C should not exceed ~10R, etc.
        C_high = self._to_float(self.results.get('C_rot_high'))
        if C_high is not None and self.R is not None and C_high > 10 * self.R:
            checks.append(f"C_rot_high = {C_high} seems too large (R ≈ {self.R})")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All results pass basic sanity (finite, non‑negative, reasonable magnitude)")
    
    # ----------------------------------------------------------------------
    # Main Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
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
                passed, msg = False, f"CRASHED: {traceback.format_exc()[:500]}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)