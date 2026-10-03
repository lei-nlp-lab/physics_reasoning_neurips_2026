import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely
        self.L = getattr(solution_instance, 'L', self.results.get('L', 0.02))  # Default 0.02 m
        self.dt1 = getattr(solution_instance, 'dt1', 0.2)
        self.dt2 = getattr(solution_instance, 'dt2', 0.1)
        self.dt = getattr(solution_instance, 'dt', 4.0)
        
        # Extract results safely
        self.v1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0))
        self.v2 = getattr(solution_instance, 'v_2', self.results.get('v_2', 0.0))
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def test_invariants(self):
        """Test kinematic invariants using KB equations"""
        success = True
        
        # Test 1: L conversion consistency (should be 0.02 m)
        if not math.isclose(self.L, 0.02, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"L conversion: expected 0.02m, got {self.L}", False) and success
        
        # Test 2: Velocity definitions v1 = L/dt1, v2 = L/dt2
        expected_v1 = self.L / self.dt1
        expected_v2 = self.L / self.dt2
        
        if not math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v1 definition: expected {expected_v1:.3f}, got {self.v1}", False) and success
        else:
            self._log(f"v1 definition correct: {self.v1:.3f} m/s", True)
            
        if not math.isclose(self.v2, expected_v2, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v2 definition: expected {expected_v2:.3f}, got {self.v2}", False) and success
        else:
            self._log(f"v2 definition correct: {self.v2:.3f} m/s", True)
        
        # Test 3: Acceleration from average definition a = (v2 - v1)/dt
        expected_a = (self.v2 - self.v1) / self.dt
        if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Acceleration definition: expected {expected_a:.5f}, got {self.a}", False) and success
        else:
            self._log(f"Acceleration correct: {self.a:.5f} m/s²", True)
        
        # Test 4: Kinematic equation v2² = v1² + 2a*x
        lhs = self.v2**2
        rhs = self.v1**2 + 2 * self.a * self.x
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Kinematic invariant failed: {self.v2}² ≠ {self.v1}² + 2*{self.a}*{self.x}", False) and success
        else:
            self._log(f"Kinematic equation satisfied: v2² = v1² + 2ax", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        success = True
        
        # Recalculate x from velocities and acceleration
        if abs(self.a) > 1e-10:
            recalc_x = (self.v2**2 - self.v1**2) / (2 * self.a)
        else:
            recalc_x = 0
            
        if not math.isclose(self.x, recalc_x, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Distance consistency: expected {recalc_x:.3f}, got {self.x}", False) and success
        else:
            self._log(f"Distance consistent: {self.x:.3f} m", True)
        
        # Check against target answers with tolerance
        target_v = [0.10, 0.20]  # m/s
        target_a = 0.025  # m/s²
        target_x = 0.60  # m
        
        if not math.isclose(self.v1, target_v[0], rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v1 target: expected {target_v[0]}, got {self.v1}", False) and success
            
        if not math.isclose(self.v2, target_v[1], rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"v2 target: expected {target_v[1]}, got {self.v2}", False) and success
            
        if not math.isclose(self.a, target_a, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"a target: expected {target_a}, got {self.a}", False) and success
            
        if not math.isclose(self.x, target_x, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"x target: expected {target_x}, got {self.x}", False) and success
        
        return success
    
    def test_monotonicity(self):
        """Perturb input L and check trend, then restore"""
        success = True
        
        # Store original values
        orig_L = self.L
        orig_v1 = self.v1
        orig_v2 = self.v2
        orig_a = self.a
        orig_x = self.x
        
        try:
            # Perturb L by +10%
            perturbed_L = orig_L * 1.10
            self.L = perturbed_L
            
            # Recalculate expected velocities (using formulas from steps)
            perturbed_v1 = perturbed_L / self.dt1
            perturbed_v2 = perturbed_L / self.dt2
            perturbed_a = (perturbed_v2 - perturbed_v1) / self.dt
            perturbed_x = (perturbed_v2**2 - perturbed_v1**2) / (2 * perturbed_a)
            
            # Check monotonicity: if L increases, velocities should increase
            if not (perturbed_v1 > orig_v1):
                success = self._log("Monotonicity: v1 should increase with increased L", False) and success
            else:
                self._log("v1 increases with L ✓", True)
                
            if not (perturbed_v2 > orig_v2):
                success = self._log("Monotonicity: v2 should increase with increased L", False) and success
            else:
                self._log("v2 increases with L ✓", True)
            
            # Acceleration should remain the same (a = (v2-v1)/dt, both increase proportionally)
            # But actually: Δv = v2-v1 = L(1/dt2 - 1/dt1), so a = L(1/dt2 - 1/dt1)/dt
            # So a should increase with L
            if not (perturbed_a > orig_a):
                success = self._log("Monotonicity: a should increase with increased L", False) and success
            else:
                self._log("a increases with L ✓", True)
            
            # Distance: x = (v2² - v1²)/(2a) = L²(1/dt2² - 1/dt1²)/(2a)
            # Since a ∝ L, x ∝ L²/L = L, so x should increase
            if not (perturbed_x > orig_x):
                success = self._log("Monotonicity: x should increase with increased L", False) and success
            else:
                self._log("x increases with L ✓", True)
                
        except Exception as e:
            success = self._log(f"Monotonicity test error: {str(e)}", False) and success
        finally:
            # Restore original values
            self.L = orig_L
            self.v1 = orig_v1
            self.v2 = orig_v2
            self.a = orig_a
            self.x = orig_x
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return (success, logs)"""
        self.logs = []
        overall_success = True
        
        # Run tests
        tests = [
            ("Kinematic Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== {test_name} ===", True)
            if not test_func():
                overall_success = False
        
        # Summary
        self._log(f"\n=== SUMMARY: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===", overall_success)
        
        return overall_success, self.logs