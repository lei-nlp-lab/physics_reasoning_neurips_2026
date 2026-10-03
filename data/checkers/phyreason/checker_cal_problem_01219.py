import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe attribute extraction using getattr with fallback to results dictionary.
        Prevents AttributeError in sandboxed environments.
        """
        self.solution = solution_instance
        
        # Extract known results safely
        self.v_m = getattr(solution_instance, 'v_m', 
                          solution_instance.results.get('v_m', 0.0))
        self.frac_a1_a2 = getattr(solution_instance, 'fraca_1a_2', 
                                 solution_instance.results.get('fraca_1a_2', 0.0))
        self.t = getattr(solution_instance, 't', 
                        solution_instance.results.get('t', 0.0))
        self.s = getattr(solution_instance, 's', 
                        solution_instance.results.get('s', 0.0))
        self.neg5 = getattr(solution_instance, '-5', 
                           solution_instance.results.get('-5', 0.0))
        
        # Define symbolic variables for physics laws
        self.v_m_sym = sp.Symbol('v_m', positive=True)
        self.a1_sym = sp.Symbol('a1', real=True)
        self.a2_sym = sp.Symbol('a2', real=True)
        self.t_sym = sp.Symbol('t', positive=True)
        self.s_sym = sp.Symbol('s', positive=True)
        
        # Target answers from problem statement
        self.target_v_m = 2.5
        self.target_ratio = 1/3
    
    def test_invariants(self):
        """
        Test fundamental kinematic invariants from the KB.
        1. From step 2: v_m^2 = 2*a1*s
        2. From step 3: 5^2 - v_m^2 = 2*a2*s
        3. From step 5: v_m = a1*t
        4. From step 6: -5 = v_m - a2*t
        """
        logs = []
        
        # Test 1: Check if v_m^2 = 2*a1*s holds numerically
        # We need to derive a1 from other equations first
        # From step 5: a1 = v_m/t
        if self.t > 0:
            a1_calc = self.v_m / self.t
            lhs = self.v_m**2
            rhs = 2 * a1_calc * self.s
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_m^2 = {lhs}, but 2*a1*s = {rhs} (a1={a1_calc})")
            else:
                logs.append(f"PASS: v_m^2 = 2*a1*s within tolerance")
        
        # Test 2: Check 25 - v_m^2 = 2*a2*s numerically
        # From step 6: a2 = (v_m + 5)/t (since -5 = v_m - a2*t)
        if self.t > 0:
            a2_calc = (self.v_m + 5) / self.t
            lhs = 25 - self.v_m**2
            rhs = 2 * a2_calc * self.s
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: 25 - v_m^2 = {lhs}, but 2*a2*s = {rhs} (a2={a2_calc})")
            else:
                logs.append(f"PASS: 25 - v_m^2 = 2*a2*s within tolerance")
        
        # Test 3: Check consistency of two ratio expressions
        # From step 4: ratio1 = v_m^2/(25 - v_m^2)
        # From step 7: ratio2 = v_m/(5 + v_m)
        ratio1 = self.v_m**2 / (25 - self.v_m**2) if 25 - self.v_m**2 != 0 else float('inf')
        ratio2 = self.v_m / (5 + self.v_m)
        
        if not math.isclose(ratio1, ratio2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Two ratio expressions disagree: {ratio1} vs {ratio2}")
        else:
            logs.append(f"PASS: Ratio expressions consistent: {ratio1} ≈ {ratio2}")
        
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        
        # Check v_m matches target 2.5 m/s
        if not math.isclose(self.v_m, self.target_v_m, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v_m = {self.v_m}, target = {self.target_v_m}")
        else:
            logs.append(f"PASS: v_m matches target value")
        
        # Check ratio matches target 1/3
        if not math.isclose(self.frac_a1_a2, self.target_ratio, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: a1/a2 = {self.frac_a1_a2}, target = {self.target_ratio}")
        else:
            logs.append(f"PASS: a1/a2 matches target ratio")
        
        # Check that -5 is indeed -5 (final velocity)
        if not math.isclose(self.neg5, -5.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"WARN: -5 = {self.neg5}, expected -5")
        else:
            logs.append(f"PASS: Final velocity is -5 m/s")
        
        # Check derived t and s are positive
        if self.t <= 0:
            logs.append(f"FAIL: Time t = {self.t} should be positive")
        else:
            logs.append(f"PASS: Time t = {self.t} > 0")
        
        if self.s <= 0:
            logs.append(f"FAIL: Displacement s = {self.s} should be positive")
        else:
            logs.append(f"PASS: Displacement s = {self.s} > 0")
        
        # Verify that using v_m=2.5 gives ratio=1/3 exactly
        test_ratio = 2.5**2 / (25 - 2.5**2)
        if not math.isclose(test_ratio, 1/3, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: With v_m=2.5, expected ratio=1/3, got {test_ratio}")
        else:
            logs.append(f"PASS: v_m=2.5 indeed yields ratio=1/3")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb v_m and check that the ratio changes in expected direction.
        From KB: ratio = v_m^2/(25 - v_m^2) is increasing in v_m for v_m in (0,5).
        """
        logs = []
        
        original_v_m = self.v_m
        original_ratio = self.frac_a1_a2
        
        # Test 1: Increase v_m by 10%
        perturbed_v_m = original_v_m * 1.1
        if perturbed_v_m < 5:  # Must be less than 5 for valid ratio
            perturbed_ratio = perturbed_v_m**2 / (25 - perturbed_v_m**2)
            if perturbed_ratio <= original_ratio:
                logs.append(f"FAIL: Ratio didn't increase when v_m increased from {original_v_m} to {perturbed_v_m}")
            else:
                logs.append(f"PASS: Ratio increases with v_m (monotonicity holds)")
        
        # Test 2: Decrease v_m by 10%
        perturbed_v_m = original_v_m * 0.9
        perturbed_ratio = perturbed_v_m**2 / (25 - perturbed_v_m**2)
        if perturbed_ratio >= original_ratio:
            logs.append(f"FAIL: Ratio didn't decrease when v_m decreased from {original_v_m} to {perturbed_v_m}")
        else:
            logs.append(f"PASS: Ratio decreases with v_m (monotonicity holds)")
        
        # Restore original values (no mutation needed)
        logs.append("NOTE: Original values preserved after perturbation test")
        
        return logs
    
    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        """
        all_logs = []
        
        # Run each test suite
        all_logs.append("=== Testing Invariants ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== Testing Trace Consistency ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== Testing Monotonicity ===")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        error_keywords = ['FAIL:', 'ERROR:', 'WARN:']  # WARN doesn't fail overall
        
        for log in all_logs:
            if log.startswith('FAIL:'):
                success = False
                break
        
        return success, all_logs