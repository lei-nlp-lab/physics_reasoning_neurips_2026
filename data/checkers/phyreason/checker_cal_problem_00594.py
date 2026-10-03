import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        solution_instance is expected to have:
            - inputs: dict with keys 'm1', 'm2', 'g', 'alpha' (alpha in degrees)
            - results: dict with keys 'N', 'f', 'T', 'F' (if computed)
        """
        # Safe extraction of inputs
        self.inputs = {}
        input_keys = ['m1', 'm2', 'g', 'alpha']
        for key in input_keys:
            self.inputs[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'inputs', {}).get(key, None))
        
        # Safe extraction of results
        self.results = {}
        result_keys = ['N', 'f', 'T', 'F']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, None))
        
        # Convert alpha to radians for internal use
        self.alpha_rad = math.radians(self.inputs.get('alpha', 60))
        
        # Define symbols for symbolic checks
        self.m1, self.m2, self.g, self.alpha = sp.symbols('m1 m2 g alpha', positive=True)
        self.N, self.f, self.T, self.F = sp.symbols('N f T F', real=True)
        
        # Cache for monotonicity test
        self._original_results = self.results.copy()
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify physics invariants: 
        1. Newton's 1st law for m1 (vertical equilibrium)
        2. Newton's 1st law for m2 (horizontal and vertical equilibrium)
        3. Force parallelogram for pulley tension
        """
        logs = []
        all_ok = True
        
        # Extract values
        m1_val = self.inputs.get('m1', 2.0)
        m2_val = self.inputs.get('m2', 5.0)
        g_val = self.inputs.get('g', 10.0)
        F_val = self.results.get('F')
        N_val = self.results.get('N')
        f_val = self.results.get('f')
        T_val = self.results.get('T')
        
        # 1. Check tension F = m1 * g (equilibrium of m1)
        if F_val is not None:
            expected_F = m1_val * g_val
            if not math.isclose(F_val, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Tension F = {F_val}, expected m1*g = {expected_F}")
                all_ok = False
            else:
                logs.append(f"PASS: Tension F = m1*g = {expected_F} N")
        else:
            logs.append("WARN: Tension F not found in results")
            all_ok = False
        
        # 2. Check vertical equilibrium of m2: N + F*cos(α) = m2*g
        if all(v is not None for v in [N_val, F_val]):
            lhs = N_val + F_val * math.cos(self.alpha_rad)
            rhs = m2_val * g_val
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical equilibrium: N + F*cos(α) = {lhs}, m2*g = {rhs}")
                all_ok = False
            else:
                logs.append(f"PASS: Vertical equilibrium N + F*cos(α) = m2*g = {rhs} N")
        else:
            logs.append("WARN: Missing N or F for vertical equilibrium check")
        
        # 3. Check horizontal equilibrium of m2: f = F*sin(α)
        if all(v is not None for v in [f_val, F_val]):
            expected_f = F_val * math.sin(self.alpha_rad)
            if not math.isclose(f_val, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Horizontal equilibrium: f = {f_val}, F*sin(α) = {expected_f}")
                all_ok = False
            else:
                logs.append(f"PASS: Horizontal equilibrium f = F*sin(α) = {expected_f:.3f} N")
        else:
            logs.append("WARN: Missing f or F for horizontal equilibrium check")
        
        # 4. Check pulley tension using parallelogram law: T = 2*F*cos(30°)
        # Note: The two rope segments make 60° with each other, so resultant is 2F*cos(30°)
        if all(v is not None for v in [T_val, F_val]):
            expected_T = 2 * F_val * math.cos(math.radians(30))
            if not math.isclose(T_val, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Pulley tension T = {T_val}, expected 2F*cos(30°) = {expected_T}")
                all_ok = False
            else:
                logs.append(f"PASS: Pulley tension T = 2F*cos(30°) = {expected_T:.3f} N")
        else:
            logs.append("WARN: Missing T or F for pulley tension check")
        
        return all_ok, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate steps with final results.
        Uses the STEPS ANALYSIS to verify consistency.
        """
        logs = []
        all_ok = True
        
        # Step 1: F = m1*g
        m1_val = self.inputs.get('m1', 2.0)
        g_val = self.inputs.get('g', 10.0)
        F_val = self.results.get('F')
        if F_val is not None:
            step1_F = m1_val * g_val
            if not math.isclose(F_val, step1_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 1 mismatch: F = {F_val}, but m1*g = {step1_F}")
                all_ok = False
        
        # Step 3 & 5: N = m2*g - F*cos(α)
        m2_val = self.inputs.get('m2', 5.0)
        N_val = self.results.get('N')
        if N_val is not None and F_val is not None:
            step3_N = m2_val * g_val - F_val * math.cos(self.alpha_rad)
            if not math.isclose(N_val, step3_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 3/5 mismatch: N = {N_val}, but m2*g - F*cos(α) = {step3_N}")
                all_ok = False
        
        # Step 4 & 5: f = F*sin(α)
        f_val = self.results.get('f')
        if f_val is not None and F_val is not None:
            step4_f = F_val * math.sin(self.alpha_rad)
            if not math.isclose(f_val, step4_f, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 4/5 mismatch: f = {f_val}, but F*sin(α) = {step4_f}")
                all_ok = False
        
        # Step 6: T = 2*F*cos(30°)
        T_val = self.results.get('T')
        if T_val is not None and F_val is not None:
            step6_T = 2 * F_val * math.cos(math.radians(30))
            if not math.isclose(T_val, step6_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 6 mismatch: T = {T_val}, but 2F*cos(30°) = {step6_T}")
                all_ok = False
        
        if all_ok:
            logs.append("PASS: All intermediate steps consistent with final results")
        
        return all_ok, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb input m1, recalculate using physics formulas, check trend.
        Restores original values after test.
        """
        logs = []
        
        # Store original
        original_m1 = self.inputs.get('m1', 2.0)
        F_val = self.results.get('F')
        N_val = self.results.get('N')
        f_val = self.results.get('f')
        T_val = self.results.get('T')
        
        if None in [F_val, N_val, f_val, T_val]:
            logs.append("WARN: Missing some results for monotonicity test")
            return False, logs
        
        # Perturb m1 by +10%
        perturbed_m1 = original_m1 * 1.10
        g_val = self.inputs.get('g', 10.0)
        m2_val = self.inputs.get('m2', 5.0)
        
        # Recalculate using physics formulas
        new_F = perturbed_m1 * g_val
        new_N = m2_val * g_val - new_F * math.cos(self.alpha_rad)
        new_f = new_F * math.sin(self.alpha_rad)
        new_T = 2 * new_F * math.cos(math.radians(30))
        
        # Check trends
        trends_ok = True
        
        # F should increase with m1
        if not new_F > F_val:
            logs.append(f"FAIL: F does not increase with m1 (from {F_val} to {new_F})")
            trends_ok = False
        
        # N should decrease (since N = m2*g - F*cos(α) and F increased)
        if not new_N < N_val:
            logs.append(f"FAIL: N does not decrease with increased m1 (from {N_val} to {new_N})")
            trends_ok = False
        
        # f should increase (since f = F*sin(α) and F increased)
        if not new_f > f_val:
            logs.append(f"FAIL: f does not increase with m1 (from {f_val} to {new_f})")
            trends_ok = False
        
        # T should increase (since T = 2F*cos(30°) and F increased)
        if not new_T > T_val:
            logs.append(f"FAIL: T does not increase with m1 (from {T_val} to {new_T})")
            trends_ok = False
        
        if trends_ok:
            logs.append("PASS: All monotonicity trends correct for +10% m1 perturbation")
        
        # Restore original values
        self.inputs['m1'] = original_m1
        
        return trends_ok, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== Running {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)