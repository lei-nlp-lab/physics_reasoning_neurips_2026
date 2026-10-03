import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        self.solution = solution_instance
        
        # Extract input parameters with safe defaults
        self.V1 = 220.0  # Primary voltage (V)
        self.I1 = 0.5    # Primary current (A)
        self.n2 = 25.0   # Secondary coil I turns
        self.V2 = 5.0    # Secondary coil I voltage (V)
        self.R1 = 5.0    # Resistor on secondary I (Ω)
        
        # Extract results with safe fallbacks
        self.n_1 = getattr(solution_instance, 'n_1', 
                          getattr(solution_instance, 'results', {}).get('n_1', 0.0))
        self.P_1 = getattr(solution_instance, 'P_1', 
                          getattr(solution_instance, 'results', {}).get('P_1', 0.0))
        self.P = getattr(solution_instance, 'P', 
                        getattr(solution_instance, 'results', {}).get('P', 0.0))
        self.P_2 = getattr(solution_instance, 'P_2', 
                          getattr(solution_instance, 'results', {}).get('P_2', 0.0))
        
        # Target values from steps analysis
        self.target_n1 = 1100.0
        self.target_P1 = 5.0
        self.target_P = 110.0
        self.target_P2 = 105.0
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants.
        """
        logs = []
        success = True
        
        # 1. Transformer voltage ratio: V1/V2 = n1/n2
        if self.n_1 > 0 and self.n2 > 0:
            expected_ratio = self.V1 / self.V2
            actual_ratio = self.n_1 / self.n2
            if not math.isclose(actual_ratio, expected_ratio, 
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"FAIL: Voltage ratio mismatch. Expected V1/V2 = n1/n2: "
                           f"{expected_ratio:.3f} != {actual_ratio:.3f}")
                success = False
            else:
                logs.append(f"PASS: Voltage ratio correct: V1/V2 = {expected_ratio:.3f}, "
                           f"n1/n2 = {actual_ratio:.3f}")
        
        # 2. Ideal transformer power conservation: P_input = P1 + P2
        if self.P > 0 and self.P_1 >= 0 and self.P_2 >= 0:
            total_output = self.P_1 + self.P_2
            if not math.isclose(self.P, total_output, 
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"FAIL: Power conservation violated. Input P = {self.P:.3f}W, "
                           f"Output P1+P2 = {total_output:.3f}W")
                success = False
            else:
                logs.append(f"PASS: Power conservation holds: P_input = {self.P:.3f}W = "
                           f"P1+P2 = {total_output:.3f}W")
        
        # 3. Electrical power formulas consistency
        # Primary power: P = V1 * I1
        primary_power = self.V1 * self.I1
        if not math.isclose(self.P, primary_power, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: Primary power mismatch. V1*I1 = {primary_power:.3f}W, "
                       f"reported P = {self.P:.3f}W")
            success = False
        else:
            logs.append(f"PASS: Primary power formula correct: V1*I1 = P = {self.P:.3f}W")
        
        # Secondary I power: P1 = V2² / R1
        secondary_power = (self.V2 ** 2) / self.R1
        if not math.isclose(self.P_1, secondary_power, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: Secondary I power mismatch. V2²/R1 = {secondary_power:.3f}W, "
                       f"reported P1 = {self.P_1:.3f}W")
            success = False
        else:
            logs.append(f"PASS: Secondary I power formula correct: V2²/R1 = P1 = {self.P_1:.3f}W")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Verify intermediate calculations match final results.
        """
        logs = []
        success = True
        
        # Check n1 calculation: n1 = (V1/V2) * n2
        expected_n1 = (self.V1 / self.V2) * self.n2
        if not math.isclose(self.n_1, expected_n1, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: n1 calculation incorrect. Expected (V1/V2)*n2 = {expected_n1:.3f}, "
                       f"got {self.n_1:.3f}")
            success = False
        else:
            logs.append(f"PASS: n1 calculation correct: (V1/V2)*n2 = {expected_n1:.3f}")
        
        # Check P1 calculation: P1 = V2² / R1
        expected_P1 = (self.V2 ** 2) / self.R1
        if not math.isclose(self.P_1, expected_P1, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: P1 calculation incorrect. Expected V2²/R1 = {expected_P1:.3f}W, "
                       f"got {self.P_1:.3f}W")
            success = False
        else:
            logs.append(f"PASS: P1 calculation correct: V2²/R1 = {expected_P1:.3f}W")
        
        # Check P calculation: P = V1 * I1
        expected_P = self.V1 * self.I1
        if not math.isclose(self.P, expected_P, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: P calculation incorrect. Expected V1*I1 = {expected_P:.3f}W, "
                       f"got {self.P:.3f}W")
            success = False
        else:
            logs.append(f"PASS: P calculation correct: V1*I1 = {expected_P:.3f}W")
        
        # Check P2 calculation: P2 = P - P1
        expected_P2 = self.P - self.P_1
        if not math.isclose(self.P_2, expected_P2, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: P2 calculation incorrect. Expected P - P1 = {expected_P2:.3f}W, "
                       f"got {self.P_2:.3f}W")
            success = False
        else:
            logs.append(f"PASS: P2 calculation correct: P - P1 = {expected_P2:.3f}W")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb inputs and verify expected trends.
        """
        logs = []
        success = True
        
        # Store original values
        original_R1 = self.R1
        original_n2 = self.n2
        
        # Test 1: Perturb R1 (increase by 50%)
        perturb_R1 = original_R1 * 1.5
        # Recalculate P1 with perturbed R1
        new_P1 = (self.V2 ** 2) / perturb_R1
        # P2 should increase since P is fixed and P1 decreased
        new_P2 = self.P - new_P1
        
        # Check monotonic trend: If R1 increases, P1 should decrease
        if not (new_P1 < self.P_1):
            logs.append(f"FAIL: Monotonicity violated for R1. When R1 increased from {original_R1}Ω to {perturb_R1}Ω, "
                       f"P1 did not decrease. Original P1={self.P_1:.3f}W, new P1={new_P1:.3f}W")
            success = False
        else:
            logs.append(f"PASS: R1↑ causes P1↓: P1 decreased from {self.P_1:.3f}W to {new_P1:.3f}W")
        
        # Check that P2 increases when P1 decreases (since P is constant)
        if not (new_P2 > self.P_2):
            logs.append(f"FAIL: Monotonicity violated for P2. When P1 decreased, P2 did not increase. "
                       f"Original P2={self.P_2:.3f}W, new P2={new_P2:.3f}W")
            success = False
        else:
            logs.append(f"PASS: P1↓ causes P2↑: P2 increased from {self.P_2:.3f}W to {new_P2:.3f}W")
        
        # Test 2: Perturb n2 (increase by 20%)
        perturb_n2 = original_n2 * 1.2
        # Recalculate n1 with perturbed n2 (using transformer ratio)
        new_n1 = (self.V1 / self.V2) * perturb_n2
        
        # Check monotonic trend: If n2 increases, n1 should increase proportionally
        expected_ratio_increase = 1.2  # Since n2 increased by 20%
        actual_ratio = new_n1 / self.n_1
        if not math.isclose(actual_ratio, expected_ratio_increase, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: Monotonicity violated for turns ratio. When n2 increased by 20%, "
                       f"n1 did not increase proportionally. n1 ratio: {actual_ratio:.3f} vs expected {expected_ratio_increase:.3f}")
            success = False
        else:
            logs.append(f"PASS: n2↑ causes proportional n1↑: n1 increased from {self.n_1:.3f} to {new_n1:.3f}")
        
        # Restore original values
        self.R1 = original_R1
        self.n2 = original_n2
        
        return success, logs
    
    def test_target_matching(self) -> Tuple[bool, List[str]]:
        """
        Check if results match the target answers.
        """
        logs = []
        success = True
        
        # Check n1 against target
        if not math.isclose(self.n_1, self.target_n1, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: n1 does not match target. Got {self.n_1:.3f}, expected {self.target_n1:.3f}")
            success = False
        else:
            logs.append(f"PASS: n1 matches target: {self.n_1:.3f}")
        
        # Check P2 against target
        if not math.isclose(self.P_2, self.target_P2, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: P2 does not match target. Got {self.P_2:.3f}W, expected {self.target_P2:.3f}W")
            success = False
        else:
            logs.append(f"PASS: P2 matches target: {self.P_2:.3f}W")
        
        # Optional: Check intermediate targets
        if not math.isclose(self.P_1, self.target_P1, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"NOTE: P1 differs from expected intermediate. Got {self.P_1:.3f}W, expected {self.target_P1:.3f}W")
        
        if not math.isclose(self.P, self.target_P, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"NOTE: P differs from expected intermediate. Got {self.P:.3f}W, expected {self.target_P:.3f}W")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and log messages.
        """
        all_logs = ["=" * 50, "Physics Checker Report", "=" * 50]
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity),
            ("Target Matching Test", self.test_target_matching)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{test_name}:")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"RESULT: {test_name} FAILED")
                else:
                    all_logs.append(f"RESULT: {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
                all_logs.append(f"RESULT: {test_name} FAILED due to exception")
        
        # Summary
        all_logs.append("\n" + "=" * 50)
        if overall_success:
            all_logs.append("OVERALL: All physics checks PASSED")
        else:
            all_logs.append("OVERALL: Some physics checks FAILED")
        all_logs.append("=" * 50)
        
        return overall_success, all_logs

# Example usage (commented out for clarity):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'n_1': 1100.0,
#                 'P_1': 5.0,
#                 'P': 110.0,
#                 'P_2': 105.0
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)