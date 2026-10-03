import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        try:
            # Input parameters
            self.m = getattr(solution_instance, 'm', 
                           getattr(solution_instance, 'inputs', {}).get('m', 1.0))
            self.g = getattr(solution_instance, 'g', 
                           getattr(solution_instance, 'inputs', {}).get('g', 9.8))
            self.mu = getattr(solution_instance, 'mu', 
                            getattr(solution_instance, 'inputs', {}).get('mu', 0.5))
            
            # Extract results with safe fallbacks
            self.results = getattr(solution_instance, 'results', {})
            
            self.F1 = getattr(solution_instance, 'F_1', 
                            self.results.get('F_1', 0.0))
            self.F2 = getattr(solution_instance, 'F_2', 
                            self.results.get('F_2', 0.0))
            self.f1 = getattr(solution_instance, 'f_1', 
                            self.results.get('f_1', 0.0))
            self.f2 = getattr(solution_instance, 'f_2', 
                            self.results.get('f_2', 0.0))
            self.f_total = getattr(solution_instance, 'f', 
                                 self.results.get('f', 0.0))
            
            # Store original for perturbation tests
            self.original_m = self.m
            
            # Define symbolic variables for verification
            self.m_sym = sp.symbols('m', positive=True)
            self.g_sym = sp.symbols('g', positive=True)
            self.mu_sym = sp.symbols('mu', positive=True)
            
            # Expected symbolic expressions
            self.F1_expected = self.m_sym * self.g_sym * sp.sin(sp.rad(60))
            self.F2_expected = self.m_sym * self.g_sym * sp.sin(sp.rad(30))
            self.f1_expected = self.mu_sym * self.F1_expected
            self.f2_expected = self.mu_sym * self.F2_expected
            self.f_total_expected = self.f1_expected + self.f2_expected
            
            # Numerical simplification of expected total
            self.f_total_expected_num = sp.N(self.f_total_expected.subs({
                sp.sin(sp.rad(60)): math.sin(math.radians(60)),
                sp.sin(sp.rad(30)): math.sin(math.radians(30))
            }))
            
        except Exception as e:
            self.init_error = f"Initialization error: {str(e)}"
            print(f"Warning: {self.init_error}")
    
    def _is_close(self, a: float, b: float, 
                  rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerance."""
        if hasattr(a, 'evalf'):
            a = float(a.evalf())
        if hasattr(b, 'evalf'):
            b = float(b.evalf())
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _numerical_substitute(self, expr, m_val=None, g_val=None, mu_val=None):
        """Safely substitute numerical values into symbolic expression."""
        m_val = m_val if m_val is not None else self.m
        g_val = g_val if g_val is not None else self.g
        mu_val = mu_val if mu_val is not None else self.mu
        
        if hasattr(expr, 'subs'):
            return expr.subs({
                self.m_sym: m_val,
                self.g_sym: g_val,
                self.mu_sym: mu_val
            }).evalf()
        return expr
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physical invariants and laws."""
        logs = []
        success = True
        
        try:
            # 1. Check gravitational force components
            F1_expected_num = self._numerical_substitute(self.F1_expected)
            if not self._is_close(self.F1, F1_expected_num):
                logs.append(f"FAIL: F1 mismatch. Got {self.F1}, expected ~{F1_expected_num}")
                success = False
            else:
                logs.append(f"PASS: F1 = mg·sin60° = {self.F1:.4f}")
            
            F2_expected_num = self._numerical_substitute(self.F2_expected)
            if not self._is_close(self.F2, F2_expected_num):
                logs.append(f"FAIL: F2 mismatch. Got {self.F2}, expected ~{F2_expected_num}")
                success = False
            else:
                logs.append(f"PASS: F2 = mg·sin30° = {self.F2:.4f}")
            
            # 2. Check friction formulas
            f1_expected_num = self._numerical_substitute(self.f1_expected)
            if not self._is_close(self.f1, f1_expected_num):
                logs.append(f"FAIL: f1 mismatch. Got {self.f1}, expected μ·F1 = {f1_expected_num}")
                success = False
            else:
                logs.append(f"PASS: f1 = μ·F1 = {self.f1:.4f}")
            
            f2_expected_num = self._numerical_substitute(self.f2_expected)
            if not self._is_close(self.f2, f2_expected_num):
                logs.append(f"FAIL: f2 mismatch. Got {self.f2}, expected μ·F2 = {f2_expected_num}")
                success = False
            else:
                logs.append(f"PASS: f2 = μ·F2 = {self.f2:.4f}")
            
            # 3. Check superposition (additive friction)
            if not self._is_close(self.f_total, self.f1 + self.f2):
                logs.append(f"FAIL: f_total ≠ f1 + f2. Got {self.f_total}, sum = {self.f1 + self.f2}")
                success = False
            else:
                logs.append(f"PASS: Superposition holds: f = f1 + f2 = {self.f_total:.4f}")
            
            # 4. Check expected symbolic form
            f_total_expected_val = self._numerical_substitute(self.f_total_expected)
            if not self._is_close(self.f_total, f_total_expected_val):
                logs.append(f"FAIL: Total friction mismatch symbolic form. Got {self.f_total}, expected {f_total_expected_val}")
                success = False
            else:
                logs.append(f"PASS: Total friction matches symbolic form (1+√3)/2·μmg")
            
            # 5. Check ratio invariance
            ratio_expected = (1 + math.sqrt(3)) / 2  # ~1.366
            ratio_actual = self.f_total / (self.mu * self.m * self.g)
            if not self._is_close(ratio_actual, ratio_expected):
                logs.append(f"FAIL: Dimensionless ratio mismatch. Got {ratio_actual}, expected {ratio_expected}")
                success = False
            else:
                logs.append(f"PASS: Dimensionless ratio f/(μmg) = {ratio_actual:.4f} ≈ (1+√3)/2")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            # Recompute f1 from F1 using μ
            f1_from_F1 = self.mu * self.F1
            if not self._is_close(self.f1, f1_from_F1):
                logs.append(f"TRACE FAIL: f1 ≠ μ·F1. {self.f1} vs {f1_from_F1}")
                success = False
            
            # Recompute f2 from F2 using μ
            f2_from_F2 = self.mu * self.F2
            if not self._is_close(self.f2, f2_from_F2):
                logs.append(f"TRACE FAIL: f2 ≠ μ·F2. {self.f2} vs {f2_from_F2}")
                success = False
            
            # Recompute total from components
            f_from_components = self.f1 + self.f2
            if not self._is_close(self.f_total, f_from_components):
                logs.append(f"TRACE FAIL: f_total ≠ f1 + f2. {self.f_total} vs {f_from_components}")
                success = False
            
            # Verify F1/F2 ratio matches sin60/sin30 = √3
            if self.F2 != 0:
                ratio_F = self.F1 / self.F2
                expected_ratio = math.sin(math.radians(60)) / math.sin(math.radians(30))  # √3 ≈ 1.732
                if not self._is_close(ratio_F, expected_ratio):
                    logs.append(f"TRACE FAIL: F1/F2 ratio mismatch. Got {ratio_F}, expected {expected_ratio}")
                    success = False
            
            if success:
                logs.append("TRACE PASS: All intermediate steps consistent with final result")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb mass and check friction trend, then restore."""
        logs = []
        success = True
        
        try:
            # Store original values
            original_f = self.f_total
            
            # Perturb mass by +10%
            perturbed_m = self.m * 1.1
            
            # Recompute using physics formulas (not solution's code)
            F1_pert = perturbed_m * self.g * math.sin(math.radians(60))
            F2_pert = perturbed_m * self.g * math.sin(math.radians(30))
            f1_pert = self.mu * F1_pert
            f2_pert = self.mu * F2_pert
            f_pert = f1_pert + f2_pert
            
            # Check linear proportionality (should increase by 10%)
            expected_f_pert = original_f * 1.1
            if not self._is_close(f_pert, expected_f_pert, rel_tol=5e-2):  # 5% tolerance for perturbation
                logs.append(f"MONOTONICITY FAIL: f(m+10%) = {f_pert}, expected ~{expected_f_pert}")
                success = False
            else:
                logs.append(f"MONOTONICITY PASS: f ∝ m (linear dependence)")
            
            # Test opposite perturbation (-10%)
            perturbed_m2 = self.m * 0.9
            F1_pert2 = perturbed_m2 * self.g * math.sin(math.radians(60))
            F2_pert2 = perturbed_m2 * self.g * math.sin(math.radians(30))
            f_pert2 = self.mu * (F1_pert2 + F2_pert2)
            
            expected_f_pert2 = original_f * 0.9
            if not self._is_close(f_pert2, expected_f_pert2, rel_tol=5e-2):
                logs.append(f"MONOTONICITY FAIL: f(m-10%) = {f_pert2}, expected ~{expected_f_pert2}")
                success = False
            
            # Test μ dependence (if we had access to perturb μ)
            logs.append("Note: μ perturbation test skipped (requires solution recomputation)")
            
            # Restore original mass
            self.m = self.original_m
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            # Attempt to restore anyway
            try:
                self.m = self.original_m
            except:
                pass
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Initialize test results
        test_results = []
        
        # Run each test
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                test_success, test_logs = test_func()
                overall_success = overall_success and test_success
                all_logs.extend(test_logs)
                status = "PASS" if test_success else "FAIL"
                test_results.append(f"{test_name}: {status}")
            except Exception as e:
                all_logs.append(f"TEST ERROR in {test_name}: {str(e)}")
                overall_success = False
                test_results.append(f"{test_name}: ERROR")
        
        # Summary
        all_logs.append("\n=== SUMMARY ===")
        for result in test_results:
            all_logs.append(result)
        all_logs.append(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs