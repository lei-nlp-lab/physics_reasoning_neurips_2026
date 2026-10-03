import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results."""
        self.solution = solution_instance
        
        # Extract inputs with safe fallbacks
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 9.8))
        self.k = getattr(solution_instance, 'k', getattr(solution_instance, 'inputs', {}).get('k', 100.0))
        
        # Extract results with safe fallbacks (use .get() to avoid KeyError)
        results = getattr(solution_instance, 'results', {})
        self.F_N = results.get('F_N', 0.0)
        self.h_A = results.get('h_A', 0.0)
        self.x_2 = results.get('x_2', 0.0)
        self.F_T = results.get('F_T', 0.0)
        self.F_k = results.get('F_k', 0.0)
        self.x_1 = results.get('x_1', 0.0)
        
        # Expected symbolic expressions for comparison
        self.mg = self.m * self.g
        self.target_F_N = 4 * self.mg
        self.target_h_A = 4 * self.mg / self.k if self.k != 0 else 0.0
        
        self.logs = []
    
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify static equilibrium and Hooke's Law relationships."""
        try:
            # Test 1: Normal force from ground (static equilibrium for A+B system)
            # According to steps: F_N = 5mg - mg = 4mg
            expected_F_N = 4 * self.mg
            if not math.isclose(self.F_N, expected_F_N, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: F_N = {self.F_N} ≠ 4mg = {expected_F_N}")
                return False
            self.log(f"PASS: F_N = 4mg = {expected_F_N}")
            
            # Test 2: Tension in rope when C has mass 5m (static equilibrium for C)
            # According to steps: F_T = 5mg
            expected_F_T = 5 * self.mg
            if not math.isclose(self.F_T, expected_F_T, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: F_T = {self.F_T} ≠ 5mg = {expected_F_T}")
                return False
            self.log(f"PASS: F_T = 5mg = {expected_F_T}")
            
            # Test 3: Spring force on A (static equilibrium for A)
            # According to steps: F_k = F_T - 2mg = 3mg
            expected_F_k = 3 * self.mg
            if not math.isclose(self.F_k, expected_F_k, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: F_k = {self.F_k} ≠ 3mg = {expected_F_k}")
                return False
            self.log(f"PASS: F_k = 3mg = {expected_F_k}")
            
            # Test 4: Hooke's Law for elongation x_1: k*x_1 = 3mg
            expected_x_1 = 3 * self.mg / self.k if self.k != 0 else 0.0
            if not math.isclose(self.x_1, expected_x_1, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: x_1 = {self.x_1} ≠ 3mg/k = {expected_x_1}")
                return False
            self.log(f"PASS: x_1 = 3mg/k = {expected_x_1}")
            
            # Test 5: Hooke's Law for initial compression x_2: k*x_2 = mg
            expected_x_2 = self.mg / self.k if self.k != 0 else 0.0
            if not math.isclose(self.x_2, expected_x_2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: x_2 = {self.x_2} ≠ mg/k = {expected_x_2}")
                return False
            self.log(f"PASS: x_2 = mg/k = {expected_x_2}")
            
            # Test 6: Geometric sum h_A = x_1 + x_2
            expected_sum = self.x_1 + self.x_2
            if not math.isclose(self.h_A, expected_sum, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: h_A = {self.h_A} ≠ x_1 + x_2 = {expected_sum}")
                return False
            self.log(f"PASS: h_A = x_1 + x_2 = {expected_sum}")
            
            # Test 7: Final h_A matches target 4mg/k
            expected_h_A = 4 * self.mg / self.k if self.k != 0 else 0.0
            if not math.isclose(self.h_A, expected_h_A, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: h_A = {self.h_A} ≠ 4mg/k = {expected_h_A}")
                return False
            self.log(f"PASS: h_A = 4mg/k = {expected_h_A}")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results lead to final answers."""
        try:
            # Check that using steps leads to target answers
            # Step 1: F_N = 5mg - mg = 4mg
            step1_F_N = 5 * self.mg - self.mg  # 4mg
            if not math.isclose(step1_F_N, self.target_F_N, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: Step1 F_N {step1_F_N} ≠ target {self.target_F_N}")
                return False
            
            # Steps 2-6 should yield h_A = 4mg/k
            step2_F_T = 5 * self.mg
            step3_F_k = step2_F_T - 2 * self.mg  # 3mg
            step4_x_1 = step3_F_k / self.k if self.k != 0 else 0.0  # 3mg/k
            step5_x_2 = self.mg / self.k if self.k != 0 else 0.0  # mg/k
            step6_h_A = step4_x_1 + step5_x_2  # 4mg/k
            
            if not math.isclose(step6_h_A, self.target_h_A, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: Step6 h_A {step6_h_A} ≠ target {self.target_h_A}")
                return False
            
            # Verify intermediate results match extracted ones
            checks = [
                (self.F_T, step2_F_T, "F_T"),
                (self.F_k, step3_F_k, "F_k"),
                (self.x_1, step4_x_1, "x_1"),
                (self.x_2, step5_x_2, "x_2"),
                (self.h_A, step6_h_A, "h_A")
            ]
            for val, expected, name in checks:
                if not math.isclose(val, expected, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: {name} = {val} ≠ step-derived {expected}")
                    return False
            
            self.log("PASS: All steps consistent with targets")
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and check expected trends."""
        original_m = self.m
        try:
            # Increase mass by 10%
            perturbed_m = original_m * 1.1
            perturbed_mg = perturbed_m * self.g
            
            # Expected: F_N ∝ m, h_A ∝ m
            expected_F_N_ratio = (4 * perturbed_mg) / self.target_F_N
            expected_h_A_ratio = (4 * perturbed_mg / self.k) / self.target_h_A if self.k != 0 else 1.0
            
            # Check ratios are close to mass ratio
            mass_ratio = perturbed_m / original_m
            if not (math.isclose(expected_F_N_ratio, mass_ratio, rel_tol=0.05) and 
                    math.isclose(expected_h_A_ratio, mass_ratio, rel_tol=0.05)):
                self.log(f"FAIL: Linearity broken. Mass ratio {mass_ratio}, F_N ratio {expected_F_N_ratio}, h_A ratio {expected_h_A_ratio}")
                return False
            
            self.log(f"PASS: Monotonicity holds (linear in m)")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
        finally:
            # Restore original mass
            self.m = original_m
    
    def run_all_tests(self):
        """Execute all checks and return success status with logs."""
        self.logs.clear()
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.log(f"\n=== Running {name} Test ===")
            if not test_func():
                success = False
                self.log(f"*** {name} test FAILED ***")
            else:
                self.log(f"*** {name} test PASSED ***")
        
        return success, self.logs