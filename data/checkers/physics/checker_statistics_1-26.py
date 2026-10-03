import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    """
    Physics checker for the radiation shield problem.
    Checks interface, invariants, modeling consistency, monotonicity, and answer sanity.
    """
    
    def __init__(self, solution_instance):
        # Extract solution data safely (None defaults)
        self.r = getattr(solution_instance, 'r', None)
        self.R = getattr(solution_instance, 'R', None)
        self.T = getattr(solution_instance, 'T', None)
        self.sigma = getattr(solution_instance, 'sigma', None)  # may not be given
        
        # Expected results from the solution (use .results dict or direct attribute)
        results = getattr(solution_instance, 'results', {})
        self.T_s4 = results.get('T_s4', None)
        self.f = results.get('f', None)
        self.a = results.get('a', None)
        self.b = results.get('b', None)
        
        # Also try direct attributes (some implementations may store as attributes)
        if self.T_s4 is None:
            self.T_s4 = getattr(solution_instance, 'T_s4', None)
        if self.f is None:
            self.f = getattr(solution_instance, 'f', None)
        if self.a is None:
            self.a = getattr(solution_instance, 'a', None)
        if self.b is None:
            self.b = getattr(solution_instance, 'b', None)
    
    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_keys_present(self) -> Tuple[bool, str]:
        """Check that required result keys exist (not None)."""
        missing = []
        for key, val in [('T_s4', self.T_s4), ('f', self.f), ('a', self.a), ('b', self.b)]:
            if val is None:
                missing.append(key)
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present"
    
    def test_interface_types(self) -> Tuple[bool, str]:
        """Check that numeric values are float or int (not string, list, etc.)."""
        for name, val in [('T_s4', self.T_s4), ('f', self.f), ('a', self.a), ('b', self.b)]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return False, f"{name} is not numeric (type {type(val).__name__})"
        return True, "All types are numeric"
    
    # ------------------------------------------------------------------
    # Invariant / theorem-based sanity tests (use math.isclose here)
    # ------------------------------------------------------------------
    def test_invariant_f_range(self) -> Tuple[bool, str]:
        """Check that f lies in (0,1)."""
        if self.f is None:
            return False, "SKIP: f not available"
        if not (0 < self.f < 1):
            return False, f"f = {self.f} not in (0,1)"
        return True, f"f = {self.f} in (0,1)"
    
    def test_invariant_a_b_positive(self) -> Tuple[bool, str]:
        """Check that a and b are positive."""
        for name, val in [('a', self.a), ('b', self.b)]:
            if val is None:
                return False, f"SKIP: {name} not available"
            if val <= 0:
                return False, f"{name} = {val} is not positive"
        return True, "a and b are positive"
    
    def test_invariant_T_s4_vs_T4(self) -> Tuple[bool, str]:
        """If T is given, check that T_s4 is less than T^4."""
        if self.T is None or self.T_s4 is None:
            return False, "SKIP: T or T_s4 missing"
        T4 = self.T ** 4
        if self.T_s4 >= T4:
            return False, f"T_s^4 = {self.T_s4} is not less than T^4 = {T4}"
        return True, f"T_s^4 < T^4 satisfied ({self.T_s4} < {T4})"
    
    def test_invariant_energy_balance(self) -> Tuple[bool, str]:
        """
        If r and R are given, derive expected T_s4 and compare with given T_s4.
        Use: T_s4 = (r^2/(r^2+R^2)) * T^4.
        """
        if None in (self.r, self.R, self.T, self.T_s4):
            return False, "SKIP: r, R, T, T_s4 required"
        expected_T_s4 = (self.r**2 / (self.r**2 + self.R**2)) * (self.T**4)
        if not math.isclose(self.T_s4, expected_T_s4, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"T_s^4 {self.T_s4} differs from expected {expected_T_s4}"
        return True, "Shield temperature relation holds"
    
    def test_invariant_reduction_factor(self) -> Tuple[bool, str]:
        """
        Check that f = 1 - (T_s/T)^4 (if both T and T_s4 are given)
        OR f = R^2/(R^2+r^2) (if r,R given).
        """
        if self.f is None:
            return False, "SKIP: f not available"
        
        # Try using T_s4 and T
        if None not in (self.T, self.T_s4):
            computed_f = 1 - self.T_s4 / (self.T**4)
            if not math.isclose(computed_f, self.f, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"f from T_s4 gives {computed_f}, but result gives {self.f}"
            return True, "f matches 1 - (T_s/T)^4"
        
        # Otherwise use r,R
        if None not in (self.r, self.R):
            computed_f = self.R**2 / (self.R**2 + self.r**2)
            if not math.isclose(computed_f, self.f, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"f from r,R gives {computed_f}, but result gives {self.f}"
            return True, "f matches R^2/(R^2+r^2)"
        
        return False, "SKIP: insufficient data to compute expected f"
    
    def test_invariant_a_b_form(self) -> Tuple[bool, str]:
        """Check that a and b satisfy the given form: a R^2/(R^2 + b r^2) = f."""
        if None in (self.a, self.b, self.f, self.r, self.R):
            return False, "SKIP: missing a,b,f,r,R"
        expected_f = self.a * self.R**2 / (self.R**2 + self.b * self.r**2)
        if not math.isclose(expected_f, self.f, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"With a={self.a}, b={self.b}, f should be {expected_f}, got {self.f}"
        return True, "a and b consistent with f"
    
    # ------------------------------------------------------------------
    # Modeling / trace tests (also use math.isclose)
    # ------------------------------------------------------------------
    def test_modeling_shield_temperature_derivation(self) -> Tuple[bool, str]:
        """
        Re-derive shield temperature from energy balance and compare with given T_s4.
        This is a modeling test that verifies the key step.
        """
        if None in (self.r, self.R, self.T, self.T_s4):
            return False, "SKIP: missing data"
        # Using energy balance: r^2 (T^4 - T_s^4) = R^2 T_s^4
        # => r^2 T^4 = (r^2+R^2) T_s^4
        lhs = self.r**2 * (self.T**4 - self.T_s4)
        rhs = self.R**2 * self.T_s4
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Energy balance fails: lhs={lhs}, rhs={rhs}"
        return True, "Energy balance satisfied"
    
    def test_modeling_reduction_factor_derivation(self) -> Tuple[bool, str]:
        """
        Check that f = Q'/Q = (unshielded - shielded)/unshielded.
        """
        if None in (self.f, self.r, self.R):
            return False, "SKIP: missing f,r,R"
        # f = R^2/(R^2+r^2)
        expected_f = self.R**2 / (self.R**2 + self.r**2)
        if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Derived f = {expected_f} but result gives {self.f}"
        return True, "Reduction factor form correct"
    
    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_f_increases_with_R(self) -> Tuple[bool, str]:
        """
        Perturb R upward by a small percentage and check that f increases.
        Requires r, R, f. We'll compute expected f for perturbed R and compare.
        """
        if None in (self.r, self.R, self.f):
            return False, "SKIP: missing r,R,f"
        # Baseline f from formula using given r,R
        base_f = self.R**2 / (self.R**2 + self.r**2)
        if not math.isclose(self.f, base_f, rel_tol=1e-2, abs_tol=1e-2):
            # If baseline f is inconsistent, we cannot test monotonicity
            return False, f"Baseline f does not match expected {base_f}"
        # Perturb R by +5%
        R_perturbed = self.R * 1.05
        perturbed_f = R_perturbed**2 / (R_perturbed**2 + self.r**2)
        if perturbed_f <= base_f:
            return False, f"Perturbed f = {perturbed_f} not greater than base f = {base_f}"
        return True, "f increases with R as expected"
    
    def test_monotonicity_T_s4_decreases_with_R(self) -> Tuple[bool, str]:
        """
        Perturb R upward and check that T_s4 decreases.
        Requires r, R, T, T_s4.
        """
        if None in (self.r, self.R, self.T, self.T_s4):
            return False, "SKIP: missing data"
        # Baseline T_s4 from formula
        base_ts4 = (self.r**2 / (self.r**2 + self.R**2)) * (self.T**4)
        if not math.isclose(self.T_s4, base_ts4, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Baseline T_s4 does not match expected {base_ts4}"
        # Perturb R by +5%
        R_pert = self.R * 1.05
        perturbed_ts4 = (self.r**2 / (self.r**2 + R_pert**2)) * (self.T**4)
        if perturbed_ts4 >= base_ts4:
            return False, f"Perturbed T_s4 = {perturbed_ts4} not less than base {base_ts4}"
        return True, "T_s4 decreases with R as expected"
    
    # ------------------------------------------------------------------
    # Weak answer sanity (ONLY one method, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_coefficients(self) -> Tuple[bool, str]:
        """
        Weak check: a and b should be positive finite numbers, and not obviously wrong (e.g., not 0).
        """
        if None in (self.a, self.b):
            return False, "SKIP: a or b missing"
        # Check that a and b are not zero and are finite
        for name, val in [('a', self.a), ('b', self.b)]:
            if val <= 0:
                return False, f"{name} = {val} is non-positive"
            if math.isinf(val) or math.isnan(val):
                return False, f"{name} is infinite or NaN"
        # Also check that they are not too large (say >1000)
        if self.a > 1000 or self.b > 1000:
            return False, f"a={self.a}, b={self.b} seem unreasonably large"
        return True, "a and b are positive finite numbers"
    
    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)