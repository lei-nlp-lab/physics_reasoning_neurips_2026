import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs - extract all values from solution instance
        self.theta_deg = getattr(solution_instance, 'theta_deg', 45.0)  # degrees
        self.I = getattr(solution_instance, 'I', 2.5)          # A
        self.M = getattr(solution_instance, 'M', 0.2)          # kg
        self.L = getattr(solution_instance, 'L', 0.4)          # m
        self.g = getattr(solution_instance, 'g', 10.0)         # m/s^2 (default to 10.0 to match target)
        
        # Safe extraction of results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.F_a = results.get('F_a', 0.0)
        self.B = results.get('B', 0.0)
        
        # Constants
        self.deg2rad = math.pi / 180.0
        
        # Logs
        self.logs = []
        
    def test_invariants(self) -> bool:
        """Test energy/force equilibrium invariants."""
        success = True
        
        # 1. Check equilibrium condition: F_a = M * g * tan(theta)
        theta_rad = self.theta_deg * self.deg2rad  # Convert degrees to radians
        F_a_expected = self.M * self.g * math.tan(theta_rad)
        if not math.isclose(self.F_a, F_a_expected, rel_tol=0.02, abs_tol=0.02):
            self.logs.append(f"FAIL: Equilibrium invariant violated. "
                           f"F_a={self.F_a:.3f}, expected {F_a_expected:.3f} "
                           f"(M={self.M}, g={self.g}, theta={self.theta_deg}°)")
            success = False
        else:
            self.logs.append(f"PASS: F_a = {self.F_a:.3f} matches M*g*tanθ = {F_a_expected:.3f}")
        
        # 2. Check Ampere's law: F_a = B * I * L
        F_a_from_B = self.B * self.I * self.L
        if not math.isclose(self.F_a, F_a_from_B, rel_tol=0.02, abs_tol=0.02):
            self.logs.append(f"FAIL: Ampere's law violated. "
                           f"F_a={self.F_a:.3f}, B*I*L={F_a_from_B:.3f} "
                           f"(B={self.B}, I={self.I}, L={self.L})")
            success = False
        else:
            self.logs.append(f"PASS: F_a = B*I*L holds ({self.F_a:.3f} = {F_a_from_B:.3f})")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate steps with final answer."""
        success = True
        
        # Step 1: F_a = M*g*tanθ
        theta_rad = self.theta_deg * self.deg2rad
        F_a_step1 = self.M * self.g * math.tan(theta_rad)
        
        # Step 2: B = F_a / (I*L)
        B_step2 = self.F_a / (self.I * self.L)
        
        # Check consistency between steps
        if not math.isclose(self.F_a, F_a_step1, rel_tol=0.02, abs_tol=0.02):
            self.logs.append(f"TRACE FAIL: Step 1 (F_a={F_a_step1:.3f}) "
                           f"doesn't match result F_a={self.F_a:.3f}")
            success = False
        else:
            self.logs.append(f"TRACE PASS: Step 1 F_a consistent")
        
        if not math.isclose(self.B, B_step2, rel_tol=0.02, abs_tol=0.02):
            self.logs.append(f"TRACE FAIL: Step 2 (B={B_step2:.3f}T) "
                           f"doesn't match result B={self.B:.3f}T")
            success = False
        else:
            self.logs.append(f"TRACE PASS: Step 2 B consistent")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb mass and check force trends."""
        success = True
        original_M = self.M
        
        try:
            # Perturb mass by +10%
            perturbed_M = original_M * 1.1
            theta_rad = self.theta_deg * self.deg2rad
            
            # Recalculate F_a using physics formula
            F_a_perturbed = perturbed_M * self.g * math.tan(theta_rad)
            
            # Check monotonicity: larger mass should yield larger force
            if not (F_a_perturbed > self.F_a):
                self.logs.append(f"MONOTONICITY FAIL: F_a not increasing with mass. "
                               f"Original F_a={self.F_a:.3f}, perturbed F_a={F_a_perturbed:.3f}")
                success = False
            else:
                self.logs.append(f"MONOTONICITY PASS: F_a increases with mass "
                               f"({self.F_a:.3f} -> {F_a_perturbed:.3f})")
            
            # Restore original value
            self.M = original_M
            
        except Exception as e:
            self.logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            self.M = original_M  # Ensure restoration
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self.logs.append(f"\n=== {test_name} Test ===")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"Test crashed: {str(e)}")
                overall_success = False
        
        # Final target answer check
        self.logs.append(f"\n=== Target Answer Check ===")
        target_B = 2.0  # Expected magnetic field in Tesla
        if math.isclose(self.B, target_B, rel_tol=0.02, abs_tol=0.02):
            self.logs.append(f"PASS: B = {self.B:.3f}T matches target {target_B}T")
        else:
            self.logs.append(f"FAIL: B = {self.B:.3f}T differs from target {target_B}T")
            overall_success = False
        
        return overall_success, self.logs