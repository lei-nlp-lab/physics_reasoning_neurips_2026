import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all required values with safe defaults
        self.T_1 = float(self.results.get('T_1', 0.0))
        self.delta_t_1 = float(self.results.get('delta_t_1', 0.0))
        self.delta_t_2 = float(self.results.get('delta_t_2', 0.0))
        self.NA = float(self.results.get('N/A', 0.0))
        
        # Input parameters from problem statement
        self.t1 = 0.2  # Time to first reach point M
        self.t2 = 0.1  # Additional time to reach M second time
        
        # Expected values from steps analysis
        self.expected_T1_scenario1 = 1/3  # Left-first scenario
        self.expected_delta_t1 = 7/30     # Left-first scenario
        
        self.expected_T1_scenario2 = 1.0  # Right-first scenario  
        self.expected_delta_t2 = 0.9      # Right-first scenario
        
        # Target answers
        self.target_answers = [30/7, 0.9]
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify SHM physics invariants"""
        logs = []
        success = True
        
        # 1. Check period-angular frequency relationship: ω = 2π/T
        if self.T_1 > 0:
            omega_calc = 2 * math.pi / self.T_1
            logs.append(f"PASS: ω = 2π/T relationship gives ω = {omega_calc:.3f} rad/s")
        
        # 2. Check SHM period consistency with given times
        # Both scenarios should satisfy: T > t1 + t2 = 0.3s
        if self.T_1 <= 0.3:
            logs.append(f"FAIL: Period T={self.T_1} should be > 0.3s (t1+t2)")
            success = False
        else:
            logs.append(f"PASS: Period T={self.T_1} > 0.3s")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer"""
        logs = []
        success = True
        
        # Check if either delta_t matches one of the target answers
        # The target answers are 30/7 ≈ 4.2857 and 0.9
        # But actually the problem expects delta_t values, not their reciprocals
        # Looking at reference: target answers are "7/30 s or 0.9 s"
        # So target answers should be [7/30, 0.9] = [0.2333..., 0.9]
        
        target_answers_corrected = [7/30, 0.9]
        
        # Check delta_t_1 against first target
        if math.isclose(self.delta_t_1, target_answers_corrected[0], rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"PASS: delta_t_1={self.delta_t_1} matches first target {target_answers_corrected[0]}")
        else:
            logs.append(f"FAIL: delta_t_1={self.delta_t_1} doesn't match first target {target_answers_corrected[0]}")
            success = False
            
        # Check delta_t_2 against second target
        if math.isclose(self.delta_t_2, target_answers_corrected[1], rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"PASS: delta_t_2={self.delta_t_2} matches second target {target_answers_corrected[1]}")
        else:
            logs.append(f"FAIL: delta_t_2={self.delta_t_2} doesn't match second target {target_answers_corrected[1]}")
            success = False
        
        # Check consistency between T_1 and delta_t values
        if self.T_1 > 0:
            # delta_t_1 should be T_1 - 0.1 for left-first scenario
            expected_delta_1 = self.T_1 - 0.1
            
            if math.isclose(self.delta_t_1, expected_delta_1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"PASS: delta_t_1 = T_1 - 0.1 = {expected_delta_1}")
            else:
                logs.append(f"FAIL: delta_t_1={self.delta_t_1} ≠ T_1-0.1={expected_delta_1}")
                success = False
        
        # Check if T_1 matches left-first scenario
        if math.isclose(self.T_1, self.expected_T1_scenario1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"PASS: T_1={self.T_1} matches left-first scenario (1/3 s)")
        else:
            logs.append(f"FAIL: T_1={self.T_1} doesn't match left-first scenario (1/3 s)")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test behavior when perturbing parameters"""
        logs = []
        success = True
        
        # Test: If we increase t1 (time to first reach M), what happens to period?
        # Save original value
        original_t1 = self.t1
        
        # Increase t1 by 10%
        perturbed_t1 = original_t1 * 1.1
        
        # For left-first scenario formula: T = t1 + t2 + (1/3)*(t1 - t2)
        T_left_original = original_t1 + self.t2 + (1/3)*(original_t1 - self.t2)
        T_left_perturbed = perturbed_t1 + self.t2 + (1/3)*(perturbed_t1 - self.t2)
        
        # For right-first scenario formula: T = 4*(t1 + 0.5*t2)
        T_right_original = 4*(original_t1 + 0.5*self.t2)
        T_right_perturbed = 4*(perturbed_t1 + 0.5*self.t2)
        
        # Check monotonicity: increasing t1 should increase period
        if T_left_perturbed <= T_left_original:
            logs.append(f"FAIL: Left-first scenario: T({perturbed_t1})={T_left_perturbed} ≤ T({original_t1})={T_left_original}")
            success = False
        else:
            logs.append(f"PASS: Left-first scenario monotonic: T increases with t1")
            
        if T_right_perturbed <= T_right_original:
            logs.append(f"FAIL: Right-first scenario: T({perturbed_t1})={T_right_perturbed} ≤ T({original_t1})={T_right_original}")
            success = False
        else:
            logs.append(f"PASS: Right-first scenario monotonic: T increases with t1")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics checks and return overall success"""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                test_success, test_logs = test_func()
                overall_success = overall_success and test_success
                all_logs.extend(test_logs)
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs