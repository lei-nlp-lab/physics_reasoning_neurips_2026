import math
import sympy as sp
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Safely extract all inputs and results without direct attribute access."""
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with fallback to problem defaults
        self.M = getattr(solution_instance, 'M', 1.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.mu1 = getattr(solution_instance, 'mu1', 0.1)
        self.mu2 = getattr(solution_instance, 'mu2', 0.3)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract force cases if available
        self.F3 = getattr(solution_instance, 'F3', 3.0)  # 3N case
        self.F5 = getattr(solution_instance, 'F5', 5.0)  # 5N case
        
        # Safely extract all result quantities with fallback to 0.0
        self.f = self.results.get('f', 0.0)
        self.a_B = self.results.get('a_B', 0.0)
        self.a_1 = self.results.get('a_1', 0.0)
        self.F_1 = self.results.get('F_1', 0.0)
        self.t = self.results.get('t', 0.0)
        self.a = self.results.get('a', 0.0)
        self.a_A = self.results.get('a_A', 0.0)
        self.f_2 = self.results.get('f_2', 0.0)
        self.f_1 = self.results.get('f_1', 0.0)
        self.F_2 = self.results.get('F_2', 0.0)
        
        # Store logs for reporting
        self.logs: List[str] = []
        
    def _log(self, message: str):
        """Helper to add to logs."""
        self.logs.append(message)
    
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics laws based on KB theorems."""
        success = True
        
        # 1. Check friction force calculations
        expected_f1 = self.mu1 * (self.M + self.m) * self.g  # μ1(M+m)g
        if not self._is_close(self.f_1, expected_f1):
            self._log(f"FAIL: f_1 = {self.f_1}, expected μ1(M+m)g = {expected_f1}")
            success = False
        else:
            self._log(f"PASS: f_1 = μ1(M+m)g = {expected_f1}")
        
        expected_f2 = self.mu2 * self.m * self.g  # μ2·m·g
        if not self._is_close(self.f_2, expected_f2):
            self._log(f"FAIL: f_2 = {self.f_2}, expected μ2·m·g = {expected_f2}")
            success = False
        else:
            self._log(f"PASS: f_2 = μ2·m·g = {expected_f2}")
        
        # 2. Check equilibrium condition (F1 = f1)
        if not self._is_close(self.F_1, expected_f1):
            self._log(f"FAIL: F_1 = {self.F_1}, expected f_1 = {expected_f1}")
            success = False
        else:
            self._log(f"PASS: F_1 = f_1 = {expected_f1}")
        
        # 3. Check Newton's 2nd Law for F=3N case
        # For block B: F - f = m·a_1
        # For board A: f - f_1 = M·a_1
        if not self._is_close(self.F3 - self.f, self.m * self.a_1):
            self._log(f"FAIL: For F=3N, B: F - f = {self.F3 - self.f}, m·a_1 = {self.m * self.a_1}")
            success = False
        else:
            self._log(f"PASS: For F=3N, B satisfies F - f = m·a_1")
        
        if not self._is_close(self.f - self.f_1, self.M * self.a_1):
            self._log(f"FAIL: For F=3N, A: f - f_1 = {self.f - self.f_1}, M·a_1 = {self.M * self.a_1}")
            success = False
        else:
            self._log(f"PASS: For F=3N, A satisfies f - f_1 = M·a_1")
        
        # 4. Check Newton's 2nd Law for F=5N case (relative sliding)
        # For board A: f_2 - f_1 = M·a_A
        if not self._is_close(self.f_2 - self.f_1, self.M * self.a_A):
            self._log(f"FAIL: For F=5N, A: f_2 - f_1 = {self.f_2 - self.f_1}, M·a_A = {self.M * self.a_A}")
            success = False
        else:
            self._log(f"PASS: For F=5N, A satisfies f_2 - f_1 = M·a_A")
        
        # For block B: F - f_2 = m·a_B
        if not self._is_close(self.F5 - self.f_2, self.m * self.a_B):
            self._log(f"FAIL: For F=5N, B: F - f_2 = {self.F5 - self.f_2}, m·a_B = {self.m * self.a_B}")
            success = False
        else:
            self._log(f"PASS: For F=5N, B satisfies F - f_2 = m·a_B")
        
        # 5. Check the no-slip threshold case (F2 = 4N)
        # When A and B move together: F2 - f_1 = (M+m)·a
        if not self._is_close(self.F_2 - self.f_1, (self.M + self.m) * self.a):
            self._log(f"FAIL: Threshold F2: F2 - f_1 = {self.F_2 - self.f_1}, (M+m)·a = {(self.M + self.m) * self.a}")
            success = False
        else:
            self._log(f"PASS: Threshold F2 satisfies F2 - f_1 = (M+m)·a")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check that kinematic equations yield correct time for F=5N
        # Relative acceleration: a_rel = a_B - a_A
        a_rel = self.a_B - self.a_A
        # From L = 0.5 * a_rel * t^2
        expected_t = math.sqrt(2 * self.L / a_rel) if a_rel > 0 else 0
        
        if not self._is_close(self.t, expected_t):
            self._log(f"FAIL: t = {self.t}, expected from kinematics = {expected_t}")
            success = False
        else:
            self._log(f"PASS: t = √(2L/(a_B-a_A)) = {expected_t}")
        
        # 2. Check that a_rel is positive (block B should overtake board)
        if a_rel <= 0:
            self._log(f"FAIL: Relative acceleration a_B - a_A = {a_rel} should be positive")
            success = False
        else:
            self._log(f"PASS: Relative acceleration a_B - a_A = {a_rel} > 0")
        
        # 3. Verify friction force bounds
        if self.f > self.f_2:
            self._log(f"FAIL: f = {self.f} exceeds maximum static friction f_2 = {self.f_2}")
            success = False
        else:
            self._log(f"PASS: f ≤ f_2 ({self.f} ≤ {self.f_2})")
        
        # 4. Check force ranges match answers
        # Answer 1: 2N < F ≤ 4N
        if not (self._is_close(self.F_1, 2.0) and self._is_close(self.F_2, 4.0)):
            self._log(f"FAIL: Force bounds F1={self.F_1}, F2={self.F_2} don't match 2N and 4N")
            success = False
        else:
            self._log(f"PASS: Force bounds F1=2N, F2=4N correct")
        
        # Answer 2: a_1=0.5, f=2.5 for F=3N
        if not (self._is_close(self.a_1, 0.5) and self._is_close(self.f, 2.5)):
            self._log(f"FAIL: F=3N case: a_1={self.a_1}, f={self.f} vs expected 0.5, 2.5")
            success = False
        else:
            self._log(f"PASS: F=3N case: a_1=0.5 m/s², f=2.5 N")
        
        # Answer 3: t = √2 ≈ 1.414
        if not self._is_close(self.t, math.sqrt(2)):
            self._log(f"FAIL: t = {self.t}, expected √2 = {math.sqrt(2)}")
            success = False
        else:
            self._log(f"PASS: t = √2 ≈ 1.414 s")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb mass m and check if trends match physical intuition."""
        success = True
        original_m = self.m
        
        # Test 1: Increase mass m by 10%
        perturbed_m = original_m * 1.1
        self.m = perturbed_m
        
        # Recompute expected values with perturbed m
        new_f1 = self.mu1 * (self.M + self.m) * self.g
        new_f2 = self.mu2 * self.m * self.g
        
        # For F=5N case with increased m:
        # Board A: new_f2 - new_f1 = M * new_a_A
        new_a_A = (new_f2 - new_f1) / self.M if self.M > 0 else 0
        
        # Block B: F5 - new_f2 = m * new_a_B
        new_a_B = (self.F5 - new_f2) / self.m if self.m > 0 else 0
        
        new_a_rel = new_a_B - new_a_A
        new_t = math.sqrt(2 * self.L / new_a_rel) if new_a_rel > 0 else 0
        
        # Check monotonicity: increasing m should increase time t (slower relative motion)
        if not (new_t > self.t or self._is_close(new_t, self.t)):
            self._log(f"FAIL: Increasing m from {original_m} to {perturbed_m} did not increase t")
            self._log(f"  Original t={self.t}, New t={new_t}")
            success = False
        else:
            self._log(f"PASS: Increasing m increases time t (as expected)")
        
        # Test 2: Decrease mass m by 10%
        self.m = original_m * 0.9
        new_f1 = self.mu1 * (self.M + self.m) * self.g
        new_f2 = self.mu2 * self.m * self.g
        new_a_A = (new_f2 - new_f1) / self.M if self.M > 0 else 0
        new_a_B = (self.F5 - new_f2) / self.m if self.m > 0 else 0
        new_a_rel = new_a_B - new_a_A
        new_t = math.sqrt(2 * self.L / new_a_rel) if new_a_rel > 0 else 0
        
        # Decreasing m should decrease time t (faster relative motion)
        if not (new_t < self.t or self._is_close(new_t, self.t)):
            self._log(f"FAIL: Decreasing m from {original_m} to {self.m} did not decrease t")
            self._log(f"  Original t={self.t}, New t={new_t}")
            success = False
        else:
            self._log(f"PASS: Decreasing m decreases time t (as expected)")
        
        # Restore original mass
        self.m = original_m
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success flag with logs."""
        self.logs.clear()
        self._log("=== PhysicsChecker Starting ===")
        
        # Run tests
        test1 = self.test_invariants()
        self._log(f"Invariants test: {'PASS' if test1 else 'FAIL'}")
        
        test2 = self.test_trace_consistency()
        self._log(f"Trace consistency test: {'PASS' if test2 else 'FAIL'}")
        
        test3 = self.test_monotonicity()
        self._log(f"Monotonicity test: {'PASS' if test3 else 'FAIL'}")
        
        overall = test1 and test2 and test3
        self._log(f"=== Overall: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'} ===")
        
        return overall, self.logs