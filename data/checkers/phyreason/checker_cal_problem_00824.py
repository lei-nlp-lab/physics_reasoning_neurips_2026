import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get() method
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all values with safe defaults
        self.T = self._safe_extract('T')
        self.t = self._safe_extract('t')
        self.a = self._safe_extract('a')
        self.v_A = self._safe_extract('v_A')
        self.s_AB = self._safe_extract('s_AB')
        self.v_O = self._safe_extract('v_O')
        
        # Constants from problem statement
        self.OA = 0.015  # 1.5 cm in meters
        self.BC = 0.035  # 3.5 cm in meters
        self.f = 5.0     # Hz
        self.m = 2.0     # kg
        
        # For monotonicity test
        self._original_OA = self.OA
        
    def _safe_extract(self, key):
        """Safely extract value from solution instance."""
        # Try results dict first, then solution attributes
        value = self.results.get(key)
        if value is None:
            value = getattr(self.solution, key, None)
        return float(value) if value is not None else None
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants."""
        logs = []
        success = True
        
        if None in [self.T, self.t, self.a, self.v_A, self.s_AB, self.v_O]:
            logs.append("Missing one or more required results")
            return False, logs
        
        # 1. Check frequency-period relationship (step 2)
        expected_T = 1.0 / (2.0 * self.f)  # T = 1/(2f)
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"T={self.T} doesn't match expected 1/(2f)={expected_T}")
            success = False
        
        # 2. Check acceleration formula (step 3)
        # a = (BC - OA) / (2*T^2)
        expected_a = (self.BC - self.OA) / (2.0 * self.T**2)
        if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"a={self.a} doesn't match (BC-OA)/(2T^2)={expected_a}")
            success = False
        
        # 3. Check s_AB using kinematic relation (step 4)
        # s_AB = OA + a*T^2
        expected_s_AB = self.OA + self.a * self.T**2
        if not math.isclose(self.s_AB, expected_s_AB, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"s_AB={self.s_AB} doesn't match OA + aT^2={expected_s_AB}")
            success = False
        
        # 4. Check v_A using average velocity (step 5)
        # v_A = (OA + s_AB) / (2*T)
        expected_v_A = (self.OA + self.s_AB) / (2.0 * self.T)
        if not math.isclose(self.v_A, expected_v_A, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v_A={self.v_A} doesn't match (OA+s_AB)/(2T)={expected_v_A}")
            success = False
        
        # 5. Check v_O using v = v0 + at (step 6, 8)
        # v_O = v_A - a*T
        expected_v_O = self.v_A - self.a * self.T
        if not math.isclose(self.v_O, expected_v_O, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v_O={self.v_O} doesn't match v_A - aT={expected_v_O}")
            success = False
        
        # 6. Check time from rest (step 7)
        # t = v_O / a (since starting from rest)
        expected_t = self.v_O / self.a if self.a != 0 else 0
        if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"t={self.t} doesn't match v_O/a={expected_t}")
            success = False
        
        # 7. Check energy conservation (if velocity values are reasonable)
        # For uniformly accelerated motion from rest: v_O^2 = 2*a*s
        # where s is distance from rest to point O
        s_from_rest = self.v_O**2 / (2 * self.a) if self.a != 0 else 0
        # This should equal 0.5*a*t^2
        expected_s = 0.5 * self.a * self.t**2
        if self.a != 0 and not math.isclose(s_from_rest, expected_s, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Energy consistency check failed: s_from_rest={s_from_rest}, expected={expected_s}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        if None in [self.v_O, self.t]:
            logs.append("Missing v_O or t for trace consistency check")
            return False, logs
        
        # Check against target answers: ["0.1s", "0.1m/s"]
        if not math.isclose(self.t, 0.1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"t={self.t} doesn't match target answer 0.1s")
            success = False
        
        if not math.isclose(self.v_O, 0.1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v_O={self.v_O} doesn't match target answer 0.1m/s")
            success = False
        
        # Verify that v_O and t are consistent with starting from rest
        if not math.isclose(self.v_O, self.a * self.t, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v_O={self.v_O} not consistent with a*t={self.a * self.t}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test that results change monotonically with input perturbation."""
        logs = []
        success = True
        
        if self.a is None or self.v_O is None or self.t is None:
            logs.append("Missing required values for monotonicity test")
            return False, logs
        
        # Perturb OA by +10%
        perturbed_OA = self.OA * 1.1
        
        # Recalculate using formulas from steps analysis
        T = 1.0 / (2.0 * self.f)  # Unchanged
        a_perturbed = (self.BC - perturbed_OA) / (2.0 * T**2)
        s_AB_perturbed = perturbed_OA + a_perturbed * T**2
        v_A_perturbed = (perturbed_OA + s_AB_perturbed) / (2.0 * T)
        v_O_perturbed = v_A_perturbed - a_perturbed * T
        t_perturbed = v_O_perturbed / a_perturbed
        
        # Check monotonic trends:
        # 1. Larger OA should lead to smaller acceleration (since BC - OA is smaller)
        if not (a_perturbed < self.a):
            logs.append(f"Monotonicity failed: a doesn't decrease when OA increases")
            logs.append(f"  Original a={self.a}, Perturbed a={a_perturbed}")
            success = False
        
        # 2. Check if v_O changes in expected direction
        # v_O = (5*OA - BC)/(4T) from derived formula
        # Derivative wrt OA is positive: dv_O/dOA = 5/(4T) > 0
        # So v_O should increase when OA increases
        if not (v_O_perturbed > self.v_O):
            logs.append(f"Monotonicity failed: v_O doesn't increase when OA increases")
            logs.append(f"  Original v_O={self.v_O}, Perturbed v_O={v_O_perturbed}")
            success = False
        
        # 3. Check time t
        # t = v_O/a, both numerator and denominator change
        # From derived formula: t = (5*OA - BC)/(2*(BC-OA)) * T
        # This is more complex, but we can check consistency
        # Recalculate using formula to verify
        t_from_formula = (5*perturbed_OA - self.BC) / (2*(self.BC - perturbed_OA)) * T
        if not math.isclose(t_perturbed, t_from_formula, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Monotonicity consistency check failed: t_perturbed={t_perturbed}, t_from_formula={t_from_formula}")
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = []
        
        # Run each test
        inv_success, inv_logs = self.test_invariants()
        trace_success, trace_logs = self.test_trace_consistency()
        mono_success, mono_logs = self.test_monotonicity()
        
        # Collect all logs
        all_logs.extend(inv_logs)
        all_logs.extend(trace_logs)
        all_logs.extend(mono_logs)
        
        # Determine overall success
        overall_success = inv_success and trace_success and mono_success
        
        # Add summary
        if overall_success:
            all_logs.append("All physics checks passed!")
        else:
            all_logs.append(f"Physics check summary: Invariants={inv_success}, "
                          f"Trace={trace_success}, Monotonicity={mono_success}")
        
        return overall_success, all_logs

# Example usage:
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()