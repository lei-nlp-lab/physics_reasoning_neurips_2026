import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.AB = 12.0
        self.AC = 32.0
        self.T = 2.0
        self.BC = self.AC - self.AB
        
        # Extract results safely
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', solution_instance.results.get('v_B', 0.0))
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', 0.0))
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Check kinematic invariants from KB theorems using numeric comparisons."""
        logs = []
        success = True
        
        # 1. Check velocity-displacement formula: v_B^2 = 2*a*(x + AB)
        try:
            lhs = self.v_B**2
            rhs = 2 * self.a * (self.x + self.AB)
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Velocity-displacement invariant violated. LHS={lhs:.3f}, RHS={rhs:.3f}")
                success = False
            else:
                logs.append("PASS: Velocity-displacement formula holds")
        except Exception as e:
            logs.append(f"ERROR in invariant1: {e}")
            success = False
        
        # 2. Check displacement formula for AB segment: AB = v_A*T + 0.5*a*T^2
        # First compute v_A from v_B = v_A + a*T
        try:
            v_A = self.v_B - self.a * self.T
            left_AB = self.AB
            right_AB = v_A * self.T + 0.5 * self.a * self.T**2
            if not math.isclose(left_AB, right_AB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: AB displacement formula violated. Computed={right_AB:.3f}, Expected={left_AB:.3f}")
                success = False
            else:
                logs.append("PASS: AB displacement formula holds")
        except Exception as e:
            logs.append(f"ERROR in AB displacement check: {e}")
            success = False
        
        # 3. Check average velocity theorem for interval AC
        try:
            avg_vel_AC = self.AC / (2 * self.T)  # Average velocity over AC
            if not math.isclose(avg_vel_AC, self.v_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Average velocity theorem violated. v_B={self.v_B:.3f}, v_avg_AC={avg_vel_AC:.3f}")
                success = False
            else:
                logs.append("PASS: Average velocity theorem holds")
        except Exception as e:
            logs.append(f"ERROR in average velocity check: {e}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check step 1: a = (BC - AB) / T^2
        try:
            expected_a = (self.BC - self.AB) / (self.T**2)
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration mismatch. Computed a={self.a:.3f}, Expected a={expected_a:.3f}")
                success = False
            else:
                logs.append(f"PASS: Acceleration a={self.a:.3f} matches kinematic derivation")
        except Exception as e:
            logs.append(f"ERROR in acceleration check: {e}")
            success = False
        
        # Check step 2: v_B = AC / (2*T)
        try:
            expected_v_B = self.AC / (2 * self.T)
            if not math.isclose(self.v_B, expected_v_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Velocity at B mismatch. Computed v_B={self.v_B:.3f}, Expected v_B={expected_v_B:.3f}")
                success = False
            else:
                logs.append(f"PASS: Velocity v_B={self.v_B:.3f} matches average velocity theorem")
        except Exception as e:
            logs.append(f"ERROR in velocity check: {e}")
            success = False
        
        # Check step 3: x = (v_B^2)/(2*a) - AB
        try:
            expected_x = (self.v_B**2) / (2 * self.a) - self.AB
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Distance x mismatch. Computed x={self.x:.3f}, Expected x={expected_x:.3f}")
                success = False
            else:
                logs.append(f"PASS: Distance x={self.x:.3f} matches velocity-displacement formula")
        except Exception as e:
            logs.append(f"ERROR in distance check: {e}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb an input parameter and verify expected trends using numeric calculations."""
        logs = []
        success = True
        
        try:
            # Store original values
            original_AB = self.AB
            original_AC = self.AC
            original_T = self.T
            
            # Perturb AB by +1% (increase)
            perturb_factor = 1.01
            perturbed_AB = original_AB * perturb_factor
            
            # Recompute derived quantities using kinematic formulas
            perturbed_BC = original_AC - perturbed_AB
            perturbed_a = (perturbed_BC - perturbed_AB) / (original_T**2)
            perturbed_v_B = original_AC / (2 * original_T)  # v_B independent of AB
            perturbed_x = (perturbed_v_B**2) / (2 * perturbed_a) - perturbed_AB
            
            # Check expected trends:
            # 1. If AB increases (AC fixed), BC decreases => acceleration decreases
            if not (perturbed_a < self.a):
                logs.append(f"FAIL: Acceleration monotonicity violated. Original a={self.a:.3f}, Perturbed a={perturbed_a:.3f}")
                success = False
            
            # 2. v_B should remain unchanged (depends only on AC and T)
            if not math.isclose(perturbed_v_B, self.v_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B changed unexpectedly. Original v_B={self.v_B:.3f}, Perturbed v_B={perturbed_v_B:.3f}")
                success = False
            
            # 3. x should increase when AB increases (qualitative check)
            if not (perturbed_x > self.x):
                logs.append(f"FAIL: x monotonicity violated. Original x={self.x:.3f}, Perturbed x={perturbed_x:.3f}")
                success = False
            
            if success:
                logs.append("PASS: All monotonicity checks passed")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return aggregated results."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.extend([f"Invariants: {log}" for log in logs1])
        overall_success = overall_success and success1
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.extend([f"Trace: {log}" for log in logs2])
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.extend([f"Monotonicity: {log}" for log in logs3])
        overall_success = overall_success and success3
        
        return overall_success, all_logs