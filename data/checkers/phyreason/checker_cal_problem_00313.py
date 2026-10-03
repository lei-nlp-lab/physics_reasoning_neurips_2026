import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Extract inputs with defaults from problem context
        self.S_AB = float(self.inputs.get('S_AB', 14.0))
        self.R = float(self.inputs.get('R', 24.0))
        self.v_A = float(self.inputs.get('v_A', 16.0))
        self.mu = float(self.inputs.get('mu', 0.6))
        self.g = float(self.inputs.get('g', 10.0))
        
        # Extract results with safe fallbacks
        self.a_min = float(getattr(solution_instance, 'a_min', self.results.get('a_min', 0.0)))
        self.f_max = float(getattr(solution_instance, 'f_max', self.results.get('f_max', 0.0)))
        self.v_max = float(getattr(solution_instance, 'v_max', self.results.get('v_max', 0.0)))
        self.a_c = float(getattr(solution_instance, 'a_c', self.results.get('a_c', 0.0)))
        
        self.logs = []
        
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
        
    def test_invariants(self) -> bool:
        """Test physical invariants from KB theorems."""
        success = True
        
        # 1. Check centripetal force relation: f_max = μ * m * g
        # Since mass m cancels out in ratio, check a_c = μ * g
        expected_a_c = self.mu * self.g
        if not math.isclose(self.a_c, expected_a_c, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Centripetal acceleration invariant: a_c={self.a_c:.3f} ≠ μ*g={expected_a_c:.3f}",
                False
            )
        else:
            success = self._log(
                f"Centripetal acceleration invariant holds: a_c ≈ μ*g = {expected_a_c:.3f}",
                True
            ) and success
            
        # 2. Check kinematic relation: v_max² = v_A² + 2*a_min*S_AB
        # Rearranged: a_min = (v_max² - v_A²) / (2*S_AB)
        expected_a_min = (self.v_max**2 - self.v_A**2) / (2 * self.S_AB)
        if not math.isclose(self.a_min, expected_a_min, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Kinematic invariant: a_min={self.a_min:.3f} ≠ (v_max²-v_A²)/(2S_AB)={expected_a_min:.3f}",
                False
            )
        else:
            success = self._log(
                f"Kinematic invariant holds: a_min ≈ (v_max²-v_A²)/(2S_AB) = {expected_a_min:.3f}",
                True
            ) and success
            
        # 3. Check centripetal acceleration definition: a_c = v_max² / R
        expected_v_max = math.sqrt(self.a_c * self.R)
        if not math.isclose(self.v_max, expected_v_max, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Centripetal definition: v_max={self.v_max:.3f} ≠ sqrt(a_c*R)={expected_v_max:.3f}",
                False
            )
        else:
            success = self._log(
                f"Centripetal definition holds: v_max ≈ sqrt(a_c*R) = {expected_v_max:.3f}",
                True
            ) and success
            
        return success
        
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check Step 2: v_max = sqrt(μ*g*R)
        expected_v_max = math.sqrt(self.mu * self.g * self.R)
        if not math.isclose(self.v_max, expected_v_max, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Step 2 inconsistency: v_max={self.v_max:.3f} ≠ sqrt(μ*g*R)={expected_v_max:.3f}",
                False
            )
        else:
            success = self._log(
                f"Step 2 consistent: v_max ≈ sqrt(μ*g*R) = {expected_v_max:.3f}",
                True
            ) and success
            
        # Check Step 5: a_min = (v_max² - v_A²) / (2*S_AB)  (from kinematic equation)
        expected_a_min = (self.v_max**2 - self.v_A**2) / (2 * self.S_AB)
        if not math.isclose(self.a_min, expected_a_min, rel_tol=0.02, abs_tol=0.02):
            success = self._log(
                f"Step 5 inconsistency: a_min={self.a_min:.3f} ≠ (v_max²-v_A²)/(2S_AB)={expected_a_min:.3f}",
                False
            )
        else:
            success = self._log(
                f"Step 5 consistent: a_min ≈ (v_max²-v_A²)/(2S_AB) = {expected_a_min:.3f}",
                True
            ) and success
            
        # Check that a_min is negative (deceleration)
        if self.a_min >= 0:
            success = self._log(
                f"Sign inconsistency: a_min={self.a_min:.3f} should be negative (deceleration)",
                False
            )
        else:
            success = self._log(
                f"Sign consistent: a_min={self.a_min:.3f} is negative as expected",
                True
            ) and success
            
        return success
        
    def test_monotonicity(self) -> bool:
        """Perturb inputs and verify physical trends."""
        success = True
        
        # Save original state
        orig_mu = self.mu
        orig_v_A = self.v_A
        orig_S_AB = self.S_AB
        
        # Test 1: Increase μ → v_max should increase
        self.mu = orig_mu * 1.1  # +10%
        v_max_inc = math.sqrt(self.mu * self.g * self.R)
        if not (v_max_inc > self.v_max):
            success = self._log(
                f"Monotonicity fail: Increasing μ did not increase v_max (orig={self.v_max:.3f}, new={v_max_inc:.3f})",
                False
            )
        else:
            success = self._log(
                f"Monotonicity pass: v_max increases with μ (orig={self.v_max:.3f}, +10%μ→{v_max_inc:.3f})",
                True
            ) and success
        self.mu = orig_mu  # Restore
        
        # Test 2: Increase v_A → |a_min| should increase (more deceleration needed)
        self.v_A = orig_v_A * 1.1  # +10%
        a_min_inc = (self.v_max**2 - self.v_A**2) / (2 * self.S_AB)
        if not (abs(a_min_inc) > abs(self.a_min)):
            success = self._log(
                f"Monotonicity fail: Increasing v_A did not increase |a_min| (orig={self.a_min:.3f}, new={a_min_inc:.3f})",
                False
            )
        else:
            success = self._log(
                f"Monotonicity pass: |a_min| increases with v_A (orig={self.a_min:.3f}, +10%v_A→{a_min_inc:.3f})",
                True
            ) and success
        self.v_A = orig_v_A  # Restore
        
        # Test 3: Increase S_AB → |a_min| should decrease (longer distance for deceleration)
        self.S_AB = orig_S_AB * 1.1  # +10%
        a_min_dec = (self.v_max**2 - self.v_A**2) / (2 * self.S_AB)
        if not (abs(a_min_dec) < abs(self.a_min)):
            success = self._log(
                f"Monotonicity fail: Increasing S_AB did not decrease |a_min| (orig={self.a_min:.3f}, new={a_min_dec:.3f})",
                False
            )
        else:
            success = self._log(
                f"Monotonicity pass: |a_min| decreases with S_AB (orig={self.a_min:.3f}, +10%S_AB→{a_min_dec:.3f})",
                True
            ) and success
        self.S_AB = orig_S_AB  # Restore
        
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} Test ---")
            if not test_func():
                overall_success = False
                
        # Final summary
        self.logs.append(f"\n{'='*40}")
        self.logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs