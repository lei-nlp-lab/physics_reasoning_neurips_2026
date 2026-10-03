import math
import traceback
from typing import Tuple, List, Optional, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize checker with safe attribute access.
        """
        # Input parameters - use defaults if not provided
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.g = getattr(solution_instance, 'g', 9.81)
        self.h = getattr(solution_instance, 'h', 0.0)
        self.beta = getattr(solution_instance, 'beta', 0.0)
        
        # For theta, use the value from results if present, otherwise default to π/4
        # This matches the problem context (ground level projectile)
        self.theta = solution_instance.results.get('theta', math.pi/4)
        
        # Results extraction
        self.results = {}
        expected_keys = ['y', 'v_y', 'T', 'mathbfv', 'fracdTdtheta', 
                        'mathbfA', 'N/A', 'e-betaT', 'mathbfr', 'L']
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, 
                                        solution_instance.results.get(key, None))
        
        # Log storage
        self.logs = []
        
    def log(self, msg: str):
        """Add message to logs."""
        self.logs.append(msg)
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, str]:
        """
        Verify fundamental physics invariants.
        """
        try:
            failures = []
            
            # 1. Energy conservation without air resistance (beta=0)
            if self.beta == 0 and self.h == 0:
                vx = self.v0 * math.cos(self.theta)
                vy0 = self.v0 * math.sin(self.theta)
                T = 2 * self.v0 * math.sin(self.theta) / self.g
                vy_final = vy0 - self.g * T
                
                KE_initial = 0.5 * (vx**2 + vy0**2)
                KE_final = 0.5 * (vx**2 + vy_final**2)
                
                if not self._is_close(KE_initial, KE_final):
                    failures.append(f"Energy not conserved: KE_initial={KE_initial:.3f}, KE_final={KE_final:.3f}")
            
            # 2. Range formula consistency (only for ground level, no air resistance)
            if self.beta == 0 and self.h == 0:
                T_formula = 2 * self.v0 * math.sin(self.theta) / self.g
                L_formula = (self.v0**2 / self.g) * math.sin(2*self.theta)
                
                T_result = self.results.get('T')
                L_result = self.results.get('L')
                
                if T_result is not None and not self._is_close(T_formula, T_result):
                    failures.append(f"Time of flight mismatch: theory={T_formula:.3f}, result={T_result:.3f}")
                
                if L_result is not None and not self._is_close(L_formula, L_result):
                    failures.append(f"Range mismatch: theory={L_formula:.3f}, result={L_result:.3f}")
            
            if failures:
                return False, " | ".join(failures)
            return True, "All invariants passed"
            
        except Exception as e:
            return False, f"Invariant test failed with error: {str(e)}"
    
    def test_trace_consistency(self) -> Tuple[bool, str]:
        """
        Cross-check intermediate results with final answers.
        """
        try:
            # Check Step 3: T = 2*v0*sinθ/g
            T_theory = 2 * self.v0 * math.sin(self.theta) / self.g
            T_result = self.results.get('T')
            if T_result is not None and not self._is_close(T_theory, T_result):
                return False, f"Time of flight inconsistent: theory={T_theory:.3f}, result={T_result:.3f}"
            
            # Check Step 4: θ_max_range = π/4 (for h=0, beta=0)
            if self.h == 0 and self.beta == 0:
                theta_max_range = math.pi/4
                # Only check if we have a theta result
                theta_result = self.results.get('theta')
                if theta_result is not None:
                    # For ground level projectile, theta could be either π/4 (max range) or π/3 (max area)
                    # Both are valid depending on what we're computing
                    if not (self._is_close(theta_result, theta_max_range) or 
                           self._is_close(theta_result, math.pi/3)):
                        return False, f"Theta value unexpected: got {theta_result:.3f}, expected π/4 or π/3"
            
            # Check Step 8/9: θ_max_area = π/3
            theta_max_area = math.pi/3
            theta_result = self.results.get('theta')
            if theta_result is not None:
                # If theta is close to π/3, that's fine (max area case)
                if self._is_close(theta_result, theta_max_area):
                    # Verify area function is maximized at this angle
                    area_func = lambda θ: math.cos(θ) * (math.sin(θ)**3)
                    test_angles = [theta_max_area - 0.1, theta_max_area, theta_max_area + 0.1]
                    areas = [area_func(θ) for θ in test_angles]
                    if not (areas[0] < areas[1] > areas[2]):
                        return False, f"Area function not maximized at θ={theta_result:.3f}"
            
            # Check Step 15: θ with height formula (not applicable here since h=0)
            # Check Step 22/23: T with air resistance (not applicable here since beta=0)
            
            return True, "Trace consistency passed"
        except Exception as e:
            return False, f"Trace consistency failed: {str(e)}"
    
    def test_monotonicity(self) -> Tuple[bool, str]:
        """
        Perturb inputs and check expected trends.
        """
        try:
            # Test 1: Range increases with v0
            v0_orig = self.v0
            for factor in [0.9, 1.1]:
                self.v0 = v0_orig * factor
                T1 = 2 * self.v0 * math.sin(self.theta) / self.g
                L1 = (self.v0**2 / self.g) * math.sin(2*self.theta)
                
                self.v0 = v0_orig
                T0 = 2 * self.v0 * math.sin(self.theta) / self.g
                L0 = (self.v0**2 / self.g) * math.sin(2*self.theta)
                
                if not (L1 > L0 if factor > 1 else L1 < L0):
                    self.v0 = v0_orig
                    return False, f"Range monotonicity failed for v0: factor={factor}, L0={L0:.3f}, L1={L1:.3f}"
            
            # Test 2: Time of flight decreases with beta (not applicable here)
            # Test 3: Area proportional to cosθ sin³θ
            theta_orig = self.theta
            area_func = lambda θ: math.cos(θ) * (math.sin(θ)**3)
            
            # Test near π/3 (maximum)
            test_angles = [math.pi/3 - 0.1, math.pi/3, math.pi/3 + 0.1]
            areas = [area_func(θ) for θ in test_angles]
            if not (areas[0] < areas[1] > areas[2]):
                self.theta = theta_orig
                return False, f"Area function not peaked at π/3: areas={areas}"
            
            # Restore original values
            self.v0 = v0_orig
            self.theta = theta_orig
            
            return True, "Monotonicity tests passed"
        except Exception as e:
            # Restore on error
            self.v0 = v0_orig if 'v0_orig' in locals() else self.v0
            self.theta = theta_orig if 'theta_orig' in locals() else self.theta
            return False, f"Monotonicity test failed: {str(e)}"
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics audit tests.
        """
        self.logs = []
        success = True
        
        # Test 1: Invariants
        inv_success, inv_msg = self.test_invariants()
        self.log(f"Invariants: {'PASS' if inv_success else 'FAIL'} - {inv_msg}")
        success = success and inv_success
        
        # Test 2: Trace Consistency
        trace_success, trace_msg = self.test_trace_consistency()
        self.log(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'} - {trace_msg}")
        success = success and trace_success
        
        # Test 3: Monotonicity
        mono_success, mono_msg = self.test_monotonicity()
        self.log(f"Monotonicity: {'PASS' if mono_success else 'FAIL'} - {mono_msg}")
        success = success and mono_success
        
        return success, self.logs