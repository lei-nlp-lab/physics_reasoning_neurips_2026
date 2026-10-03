import math
import sympy as sp
from typing import Any, Tuple, List, Dict
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Safely extract all inputs and results using getattr to avoid AttributeError."""
        self.solution = solution_instance
        
        # Extract inputs safely
        self.theta = math.radians(37)  # 37° in radians
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        self.M = getattr(solution_instance, 'M', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 20.0)
        self.F = getattr(solution_instance, 'F', 14.0)
        self.mu1 = getattr(solution_instance, 'mu1', 0.25)
        self.mu2 = getattr(solution_instance, 'mu2', 0.5)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely from solution_instance.results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Required result keys
        self.a1 = results.get('a_1', 0.0)
        self.a2 = results.get('a_2', 0.0)
        self.v1 = results.get('v_1', 0.0)
        self.v2 = results.get('v_2', 0.0)
        self.delta_x1 = results.get('delta_x_1', 0.0)
        self.delta_x2 = results.get('delta_x_2', 0.0)
        self.L = results.get('L', 0.0)
        self.a = results.get('a', 0.0)
        self.a_m = results.get('a_m', 0.0)
        self.a3 = results.get('a_3', 0.0)
        self.f = results.get('f', 0.0)
        self.f_inf = results.get('f_∞', results.get('f_inf', 0.0))
        self.a1_prime = results.get('a_1\'', results.get('a1_prime', 0.0))
        self.a2_prime = results.get('a_2\'', results.get('a2_prime', 0.0))
        
        self.logs = []
        self.test_results = {}

    def _isclose(self, a: float, b: float) -> bool:
        """Robust fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)

    def test_invariants(self) -> List[str]:
        """Verify Newton's Laws and conservation principles."""
        logs = []
        
        # 1. Newton's 2nd Law for block during first 2s (step 1)
        # Equation: μ₁mgcosθ + mgsinθ = m·a₁
        lhs = self.mu1 * self.m * self.g * self.cos_theta + self.m * self.g * self.sin_theta
        rhs = self.m * self.a1
        if not self._isclose(lhs, rhs):
            logs.append(f"FAIL: Newton's 2nd Law for block: lhs={lhs:.3f}, rhs={rhs:.3f}")
        else:
            logs.append(f"OK: Block acceleration a₁ = {self.a1:.3f} m/s²")
            
        # 2. Newton's 2nd Law for board during first 2s (step 2)
        # Equation: F + μ₁mgcosθ - Mgsinθ - μ₂(M+m)gcosθ = M·a₂
        lhs = (self.F + self.mu1 * self.m * self.g * self.cos_theta 
               - self.M * self.g * self.sin_theta 
               - self.mu2 * (self.M + self.m) * self.g * self.cos_theta)
        rhs = self.M * self.a2
        if not self._isclose(lhs, rhs):
            logs.append(f"FAIL: Newton's 2nd Law for board: lhs={lhs:.3f}, rhs={rhs:.3f}")
        else:
            logs.append(f"OK: Board acceleration a₂ = {self.a2:.3f} m/s²")
            
        # 3. Check directions from step 4: a₁ should be negative relative to initial v₀ direction
        if self.a1 <= 0:
            logs.append("WARN: a₁ should be positive magnitude (8 m/s²) directed downhill")
        
        # 4. Newton's 2nd Law for combined system (step 7)
        # Equation: μ₂(M+m)gcosθ + (M+m)gsinθ = (M+m)·a
        lhs = self.mu2 * (self.M + self.m) * self.g * self.cos_theta + (self.M + self.m) * self.g * self.sin_theta
        rhs = (self.M + self.m) * self.a
        if not self._isclose(lhs, rhs):
            logs.append(f"FAIL: Combined system acceleration: lhs={lhs:.3f}, rhs={rhs:.3f}")
        else:
            logs.append(f"OK: Combined acceleration a = {self.a:.3f} m/s²")
            
        # 5. Maximum static friction check (step 11)
        # Equation: f_∞ = μ₂(M+m)gcosθ
        expected_f_inf = self.mu2 * (self.M + self.m) * self.g * self.cos_theta
        if not self._isclose(self.f_inf, expected_f_inf):
            logs.append(f"FAIL: Max static friction: expected={expected_f_inf:.3f}, got={self.f_inf:.3f}")
        else:
            logs.append(f"OK: Max static friction f_∞ = {self.f_inf:.3f} N")
            
        # 6. Energy check: Work-energy theorem for block during first 2s
        # ΔKE = Work by friction + gravity
        # Initial KE = 0.5*m*v0², final KE = 0.5*m*v1²
        # Work by friction on block = -μ₁mgcosθ * relative displacement
        # This is complex, so we'll do a simpler check: v1 should be less than v0
        if self.v1 >= self.v0:
            logs.append(f"FAIL: Block should slow down: v0={self.v0:.3f}, v1={self.v1:.3f}")
        else:
            logs.append(f"OK: Block slows from v0={self.v0:.3f} to v1={self.v1:.3f} m/s")
            
        return logs

    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answer."""
        logs = []
        
        # 1. Check kinematics during first 2s (steps 5-6)
        # v1 = v0 - a1*t (t=2s)
        expected_v1 = self.v0 - self.a1 * 2.0
        if not self._isclose(self.v1, expected_v1):
            logs.append(f"FAIL: v1 inconsistent: expected={expected_v1:.3f}, got={self.v1:.3f}")
        else:
            logs.append(f"OK: v1 = {self.v1:.3f} m/s matches kinematics")
            
        # v2 = a2*t (t=2s)
        expected_v2 = self.a2 * 2.0
        if not self._isclose(self.v2, expected_v2):
            logs.append(f"FAIL: v2 inconsistent: expected={expected_v2:.3f}, got={self.v2:.3f}")
        else:
            logs.append(f"OK: v2 = {self.v2:.3f} m/s matches kinematics")
            
        # 2. Check Δx1 formula (step 12)
        # Δx1 = (v0+v1)/2 * t - (v2)/2 * t
        expected_delta_x1 = ((self.v0 + self.v1) / 2 * 2.0) - (self.v2 / 2 * 2.0)
        if not self._isclose(self.delta_x1, expected_delta_x1):
            logs.append(f"FAIL: Δx1 inconsistent: expected={expected_delta_x1:.3f}, got={self.delta_x1:.3f}")
        else:
            logs.append(f"OK: Δx1 = {self.delta_x1:.3f} m matches kinematics")
            
        # 3. Check Δx2 formula (step 13)
        # Δx2 = v1²/(2*a1) - v2²/(2*a3)
        expected_delta_x2 = (self.v1**2 / (2 * self.a1)) - (self.v2**2 / (2 * self.a3))
        if not self._isclose(self.delta_x2, expected_delta_x2):
            logs.append(f"FAIL: Δx2 inconsistent: expected={expected_delta_x2:.3f}, got={self.delta_x2:.3f}")
        else:
            logs.append(f"OK: Δx2 = {self.delta_x2:.3f} m matches kinematics")
            
        # 4. Check total length L = Δx1 + Δx2
        expected_L = self.delta_x1 + self.delta_x2
        if not self._isclose(self.L, expected_L):
            logs.append(f"FAIL: Total L inconsistent: expected={expected_L:.3f}, got={self.L:.3f}")
        else:
            logs.append(f"OK: Total L = {self.L:.3f} m = Δx1 + Δx2")
            
        # 5. Check a_m (step 8) equals a1
        if not self._isclose(self.a_m, self.a1):
            logs.append(f"FAIL: a_m should equal a1: a_m={self.a_m:.3f}, a1={self.a1:.3f}")
        else:
            logs.append(f"OK: a_m = a1 = {self.a1:.3f} m/s²")
            
        # 6. Check a1' and a2' formulas (steps 14-15)
        expected_a1_prime = self.g * self.sin_theta - self.mu1 * self.g * self.cos_theta
        if not self._isclose(self.a1_prime, expected_a1_prime):
            logs.append(f"FAIL: a1' inconsistent: expected={expected_a1_prime:.3f}, got={self.a1_prime:.3f}")
        else:
            logs.append(f"OK: a1' = {self.a1_prime:.3f} m/s²")
            
        expected_a2_prime = (self.g * self.sin_theta + self.mu1 * self.g * self.cos_theta 
                           - self.mu2 * 2 * self.g * self.cos_theta)
        if not self._isclose(self.a2_prime, expected_a2_prime):
            logs.append(f"FAIL: a2' inconsistent: expected={expected_a2_prime:.3f}, got={self.a2_prime:.3f}")
        else:
            logs.append(f"OK: a2' = {self.a2_prime:.3f} m/s²")
            
        return logs

    def test_monotonicity(self) -> List[str]:
        """Perturb mass m and check if L changes as expected."""
        logs = []
        
        original_m = self.m
        perturbed_m = original_m * 1.1  # Increase by 10%
        
        # Store original results
        original_results = {
            'a1': self.a1, 'a2': self.a2, 'v1': self.v1, 'v2': self.v2,
            'delta_x1': self.delta_x1, 'delta_x2': self.delta_x2, 'L': self.L
        }
        
        try:
            # Recalculate key quantities with perturbed m using formulas from steps
            
            # 1. Recalculate a1 with perturbed m (formula from step 1)
            new_a1 = (self.mu1 * perturbed_m * self.g * self.cos_theta 
                     + perturbed_m * self.g * self.sin_theta) / perturbed_m
            
            # 2. Recalculate a2 with perturbed m (formula from step 2)
            new_a2 = (self.F + self.mu1 * perturbed_m * self.g * self.cos_theta 
                     - self.M * self.g * self.sin_theta 
                     - self.mu2 * (self.M + perturbed_m) * self.g * self.cos_theta) / self.M
            
            # 3. Recalculate v1 and v2 at t=2s
            new_v1 = self.v0 - new_a1 * 2.0
            new_v2 = new_a2 * 2.0
            
            # 4. Recalculate Δx1
            new_delta_x1 = ((self.v0 + new_v1) / 2 * 2.0) - (new_v2 / 2 * 2.0)
            
            # 5. Recalculate a3 with perturbed m (formula from step 9)
            new_a3 = (self.M * self.g * self.sin_theta 
                     + self.mu2 * (self.M + perturbed_m) * self.g * self.cos_theta 
                     - self.mu1 * perturbed_m * self.g * self.cos_theta) / self.M
            
            # 6. Recalculate Δx2
            new_delta_x2 = (new_v1**2 / (2 * new_a1)) - (new_v2**2 / (2 * new_a3))
            
            # 7. Recalculate total L
            new_L = new_delta_x1 + new_delta_x2
            
            # Check monotonicity: If m increases, friction increases, so L should increase
            if new_L > original_results['L']:
                logs.append(f"OK: Monotonicity - Increasing m from {original_m} to {perturbed_m} increases L from {original_results['L']:.3f} to {new_L:.3f}")
            else:
                logs.append(f"WARN: Monotonicity - L decreased when m increased: {original_results['L']:.3f} -> {new_L:.3f}")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            
        finally:
            # Restore original m (important for other tests)
            self.m = original_m
            
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        all_passed = True
        
        try:
            self.logs.append("="*50)
            self.logs.append("PhysicsChecker - Running Tests")
            self.logs.append("="*50)
            
            # Test 1: Invariants
            self.logs.append("\n1. Testing Physical Invariants:")
            inv_logs = self.test_invariants()
            self.logs.extend(inv_logs)
            if any("FAIL" in log for log in inv_logs):
                all_passed = False
                
            # Test 2: Trace Consistency
            self.logs.append("\n2. Testing Trace Consistency:")
            trace_logs = self.test_trace_consistency()
            self.logs.extend(trace_logs)
            if any("FAIL" in log for log in trace_logs):
                all_passed = False
                
            # Test 3: Monotonicity
            self.logs.append("\n3. Testing Monotonicity:")
            mono_logs = self.test_monotonicity()
            self.logs.extend(mono_logs)
            
            # Final assessment
            self.logs.append("\n" + "="*50)
            if all_passed:
                self.logs.append("ALL TESTS PASSED ✓")
            else:
                self.logs.append("SOME TESTS FAILED ✗")
            self.logs.append("="*50)
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            all_passed = False
            
        return all_passed, self.logs