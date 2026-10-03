import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        
        # Extract input parameters (known from problem statement)
        self.v0 = 8.0  # m/s
        self.a = -2.0  # m/s² (negative for deceleration)
        self.t_step1 = 2.0  # s (time for first calculation)
        self.t_step4 = 3.0  # s (time for displacement after 3s)
        
        # Extract results safely - use getattr with fallback to results dict
        self.v_1 = getattr(solution_instance, 'v_1', 
                          getattr(solution_instance, 'results', {}).get('v_1', None))
        self.t = getattr(solution_instance, 't',
                        getattr(solution_instance, 'results', {}).get('t', None))
        self.x_1 = getattr(solution_instance, 'x_1',
                          getattr(solution_instance, 'results', {}).get('x_1', None))
        self.x_2 = getattr(solution_instance, 'x_2',
                          getattr(solution_instance, 'results', {}).get('x_2', None))
        
        # For symbolic computations
        self.symbols = self._setup_symbols()
        
        # Test results storage
        self.test_logs = []
    
    def _setup_symbols(self):
        """Initialize sympy symbols for kinematic equations"""
        v0, v1, a, t, x = sp.symbols('v0 v1 a t x')
        return {'v0': v0, 'v1': v1, 'a': a, 't': t, 'x': x}
    
    def _is_close(self, val1: float, val2: float, 
                  rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerance"""
        if val1 is None or val2 is None:
            return False
        return math.isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental kinematic invariants"""
        logs = []
        all_pass = True
        
        try:
            # 1. Check velocity after 2s using v1 = v0 + a*t
            if self.v_1 is not None:
                expected_v1 = self.v0 + self.a * self.t_step1
                if not self._is_close(self.v_1, expected_v1):
                    logs.append(f"FAIL: v_1 = {self.v_1}, expected {expected_v1}")
                    all_pass = False
                else:
                    logs.append(f"PASS: v_1 = v0 + a*t holds")
            
            # 2. Check stopping time using t = (0 - v0)/a
            if self.t is not None:
                expected_t = (0 - self.v0) / self.a
                if not self._is_close(self.t, expected_t):
                    logs.append(f"FAIL: t = {self.t}, expected {expected_t}")
                    all_pass = False
                else:
                    logs.append(f"PASS: t = (0 - v0)/a holds")
            
            # 3. Check stopping distance using x = v0*t + 0.5*a*t²
            if self.x_1 is not None and self.t is not None:
                expected_x1 = self.v0 * self.t + 0.5 * self.a * self.t**2
                if not self._is_close(self.x_1, expected_x1):
                    logs.append(f"FAIL: x_1 = {self.x_1}, expected {expected_x1}")
                    all_pass = False
                else:
                    logs.append(f"PASS: x_1 = v0*t + 0.5*a*t² holds")
            
            # 4. Check displacement after 3s using x = v0*t + 0.5*a*t²
            if self.x_2 is not None:
                expected_x2 = self.v0 * self.t_step4 + 0.5 * self.a * self.t_step4**2
                if not self._is_close(self.x_2, expected_x2):
                    logs.append(f"FAIL: x_2 = {self.x_2}, expected {expected_x2}")
                    all_pass = False
                else:
                    logs.append(f"PASS: x_2 = v0*t + 0.5*a*t² holds")
            
            # 5. Check that final velocity is 0 for stopping distance calculation
            if self.t is not None:
                v_final = self.v0 + self.a * self.t
                if not self._is_close(v_final, 0.0):
                    logs.append(f"FAIL: Final velocity at stopping time = {v_final}, expected 0")
                    all_pass = False
                else:
                    logs.append(f"PASS: Car comes to complete stop at t = {self.t}s")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            all_pass = False
        
        return all_pass, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers"""
        logs = []
        all_pass = True
        
        try:
            # Target answers from problem statement
            target_v1 = 4.0  # m/s
            target_x1 = 16.0  # m
            target_x2 = 15.0  # m
            
            # Check against extracted values
            if self.v_1 is not None:
                if not self._is_close(self.v_1, target_v1):
                    logs.append(f"FAIL: v_1 = {self.v_1} doesn't match target {target_v1}")
                    all_pass = False
                else:
                    logs.append(f"PASS: v_1 matches target answer")
            
            if self.x_1 is not None:
                if not self._is_close(self.x_1, target_x1):
                    logs.append(f"FAIL: x_1 = {self.x_1} doesn't match target {target_x1}")
                    all_pass = False
                else:
                    logs.append(f"PASS: x_1 matches target answer")
            
            if self.x_2 is not None:
                if not self._is_close(self.x_2, target_x2):
                    logs.append(f"FAIL: x_2 = {self.x_2} doesn't match target {target_x2}")
                    all_pass = False
                else:
                    logs.append(f"PASS: x_2 matches target answer")
            
            # Consistency check: x_2 should be less than x_1 (displacement after 3s < stopping distance)
            if self.x_1 is not None and self.x_2 is not None:
                if self.x_2 >= self.x_1:
                    logs.append(f"FAIL: x_2 ({self.x_2}) should be < x_1 ({self.x_1})")
                    all_pass = False
                else:
                    logs.append(f"PASS: x_2 < x_1 as expected")
            
            # Consistency check: v_1 should be positive but less than v0
            if self.v_1 is not None:
                if not (0 < self.v_1 < self.v0):
                    logs.append(f"FAIL: v_1 ({self.v_1}) should be between 0 and v0 ({self.v0})")
                    all_pass = False
                else:
                    logs.append(f"PASS: v_1 is between 0 and v0")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            all_pass = False
        
        return all_pass, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test that increasing initial speed increases results"""
        logs = []
        all_pass = True
        
        try:
            # Store original values
            original_v0 = self.v0
            
            # Perturb initial velocity by +10%
            perturbed_v0 = original_v0 * 1.10
            
            # Calculate expected values with perturbed v0
            t_step1 = self.t_step1
            t_step4 = self.t_step4
            
            # Using same formulas from KB
            perturbed_v1 = perturbed_v0 + self.a * t_step1
            perturbed_t = (0 - perturbed_v0) / self.a
            perturbed_x1 = perturbed_v0 * perturbed_t + 0.5 * self.a * perturbed_t**2
            perturbed_x2 = perturbed_v0 * t_step4 + 0.5 * self.a * t_step4**2
            
            # Restore original v0
            self.v0 = original_v0
            
            # Check monotonicity: all perturbed values should be greater than original
            if self.v_1 is not None and not (perturbed_v1 > self.v_1):
                logs.append(f"FAIL: v_1 not monotonic with v0")
                all_pass = False
            else:
                logs.append(f"PASS: v_1 increases with v0")
            
            if self.t is not None and not (perturbed_t > self.t):
                logs.append(f"FAIL: t not monotonic with v0")
                all_pass = False
            else:
                logs.append(f"PASS: t increases with v0")
            
            if self.x_1 is not None and not (perturbed_x1 > self.x_1):
                logs.append(f"FAIL: x_1 not monotonic with v0")
                all_pass = False
            else:
                logs.append(f"PASS: x_1 increases with v0")
            
            if self.x_2 is not None and not (perturbed_x2 > self.x_2):
                logs.append(f"FAIL: x_2 not monotonic with v0")
                all_pass = False
            else:
                logs.append(f"PASS: x_2 increases with v0")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            all_pass = False
        
        return all_pass, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall result"""
        all_logs = []
        overall_pass = True
        
        # Run each test
        tests = [
            ("Kinematic Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                passed, logs = test_func()
                overall_pass = overall_pass and passed
                all_logs.extend(logs)
                status = "PASSED" if passed else "FAILED"
                all_logs.append(f"--- {test_name}: {status} ---")
            except Exception as e:
                all_logs.append(f"ERROR during {test_name}: {str(e)}")
                overall_pass = False
        
        # Final summary
        all_logs.append(f"\n=== OVERALL RESULT ===")
        all_logs.append(f"All tests passed: {overall_pass}")
        
        return overall_pass, all_logs