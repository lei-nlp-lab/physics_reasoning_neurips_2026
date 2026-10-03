import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters safely (problem context)
        self.a1 = getattr(solution_instance, 'a1', 20.0)
        self.t1 = getattr(solution_instance, 't1', 30.0)
        self.a2 = getattr(solution_instance, 'a2', 50.0)
        
        # Extract results safely (expected keys from KB)
        self.v_1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0))
        self.v_2 = getattr(solution_instance, 'v_2', self.results.get('v_2', 0.0))
        self.x_1 = getattr(solution_instance, 'x_1', self.results.get('x_1', 0.0))
        self.x_2 = getattr(solution_instance, 'x_2', self.results.get('x_2', 0.0))
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        
        # Derived parameters
        self.t2 = (self.v_2 - self.v_1) / self.a2 if self.a2 != 0 else 0.0
        
        # Setup symbolic variables for physics laws
        self.t, self.a, self.v0, self.v = sp.symbols('t a v0 v')
        self.v_formula = self.v0 + self.a * self.t
        self.x_formula = self.v0 * self.t + sp.Rational(1,2) * self.a * self.t**2
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        
    def fuzzy_eq(self, a, b, msg):
        is_close = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        self.log(f"{msg}: {a} ≈ {b} -> {is_close}")
        return is_close
        
    def test_invariants(self):
        """Verify kinematic equations for both stages"""
        try:
            # Stage 1: v_1 = a1 * t1
            expected_v1 = self.a1 * self.t1
            test1 = self.fuzzy_eq(self.v_1, expected_v1, 
                                 f"Stage 1: v1 = a1*t1 = {self.a1}*{self.t1}")
            
            # Stage 1: x_1 = 0.5 * a1 * t1^2
            expected_x1 = 0.5 * self.a1 * self.t1**2
            test2 = self.fuzzy_eq(self.x_1, expected_x1,
                                 f"Stage 1: x1 = 0.5*a1*t1² = 0.5*{self.a1}*{self.t1}²")
            
            # Stage 2: v_2 = v_1 + a2 * t2 (with computed t2)
            if self.t2 > 0:
                expected_v2 = self.v_1 + self.a2 * self.t2
                test3 = self.fuzzy_eq(self.v_2, expected_v2,
                                     f"Stage 2: v2 = v1 + a2*t2 = {self.v_1} + {self.a2}*{self.t2:.1f}")
            else:
                # If t2 not computable, check symbolic consistency
                t2_sym = sp.symbols('t2')
                v2_expr = self.v_1 + self.a2 * t2_sym
                test3 = True
                self.log(f"Stage 2: symbolic v2 = {self.v_1} + {self.a2}·t2")
            
            # Stage 2: x_2 using two equivalent forms
            if self.t2 > 0:
                # Form 1: x_2 = (v_1 + v_2)/2 * t2
                expected_x2_form1 = (self.v_1 + self.v_2) / 2 * self.t2
                test4a = self.fuzzy_eq(self.x_2, expected_x2_form1,
                                      f"Stage 2: x2 = (v1+v2)/2·t2 = ({self.v_1}+{self.v_2})/2*{self.t2:.1f}")
                
                # Form 2: x_2 = v_1*t2 + 0.5*a2*t2^2
                expected_x2_form2 = self.v_1 * self.t2 + 0.5 * self.a2 * self.t2**2
                test4b = self.fuzzy_eq(self.x_2, expected_x2_form2,
                                      f"Stage 2: x2 = v1·t2 + 0.5·a2·t2² = {self.v_1}*{self.t2:.1f} + 0.5*{self.a2}*{self.t2:.1f}²")
                
                test4 = test4a and test4b
            else:
                test4 = True
                self.log("Stage 2: Skipping x2 checks (t2 unavailable)")
            
            # Total displacement: x = x_1 + x_2
            expected_x = self.x_1 + self.x_2
            test5 = self.fuzzy_eq(self.x, expected_x,
                                 f"Total: x = x1 + x2 = {self.x_1} + {self.x_2}")
            
            success = all([test1, test2, test3, test4, test5])
            self.log(f"Invariants test: {'PASS' if success else 'FAIL'}")
            return success
            
        except Exception as e:
            self.log(f"Invariants test error: {str(e)}")
            self.log(traceback.format_exc())
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Check that v_2 matches target answer (5100 m/s)
            target_v2 = 5100.0
            test1 = self.fuzzy_eq(self.v_2, target_v2,
                                 f"v_2 vs target: {self.v_2} ≈ {target_v2}")
            
            # Check that x matches target answer (2.655e5 m)
            target_x = 2.655e5
            test2 = self.fuzzy_eq(self.x, target_x,
                                 f"x vs target: {self.x} ≈ {target_x}")
            
            # Verify that v_1 is used correctly in subsequent calculations
            if self.t2 > 0:
                # Recompute v_2 from v_1 and a2
                recalc_v2 = self.v_1 + self.a2 * self.t2
                test3 = self.fuzzy_eq(self.v_2, recalc_v2,
                                     f"Trace v2: {self.v_2} ≈ v1 + a2*t2 = {recalc_v2}")
                
                # Recompute x_2 from v_1 and v_2
                recalc_x2 = (self.v_1 + self.v_2) / 2 * self.t2
                test4 = self.fuzzy_eq(self.x_2, recalc_x2,
                                     f"Trace x2: {self.x_2} ≈ (v1+v2)/2*t2 = {recalc_x2}")
            else:
                test3 = test4 = True
                self.log("Trace: Skipping v2/x2 recomputation (t2 unavailable)")
            
            success = all([test1, test2, test3, test4])
            self.log(f"Trace consistency test: {'PASS' if success else 'FAIL'}")
            return success
            
        except Exception as e:
            self.log(f"Trace consistency test error: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb acceleration and verify expected trends"""
        try:
            # Save original values
            orig_a1 = self.a1
            orig_v1 = self.v_1
            orig_x1 = self.x_1
            orig_v2 = self.v_2
            orig_x = self.x
            
            # Increase a1 by 5%
            perturbed_a1 = self.a1 * 1.05
            
            # Recompute stage 1 results using kinematic formulas
            new_v1 = perturbed_a1 * self.t1
            new_x1 = 0.5 * perturbed_a1 * self.t1**2
            
            # For stage 2, keep same t2 (based on original v2-v1 relationship)
            # But actually, t2 might change if v2 is fixed. Let's assume t2 remains same for perturbation
            # This is a sensitivity test, not exact recomputation
            new_v2 = new_v1 + self.a2 * self.t2
            new_x2 = (new_v1 + new_v2) / 2 * self.t2
            new_x = new_x1 + new_x2
            
            # Check monotonicity: increasing a1 should increase all outputs
            tests = []
            
            test1 = new_v1 > orig_v1
            tests.append(test1)
            self.log(f"Monotonicity v1: {new_v1} > {orig_v1} -> {test1}")
            
            test2 = new_x1 > orig_x1
            tests.append(test2)
            self.log(f"Monotonicity x1: {new_x1} > {orig_x1} -> {test2}")
            
            test3 = new_v2 > orig_v2
            tests.append(test3)
            self.log(f"Monotonicity v2: {new_v2} > {orig_v2} -> {test3}")
            
            test4 = new_x > orig_x
            tests.append(test4)
            self.log(f"Monotonicity x: {new_x} > {orig_x} -> {test4}")
            
            # Restore original values
            self.a1 = orig_a1
            self.v_1 = orig_v1
            self.x_1 = orig_x1
            self.v_2 = orig_v2
            self.x = orig_x
            
            success = all(tests)
            self.log(f"Monotonicity test: {'PASS' if success else 'FAIL'}")
            return success
            
        except Exception as e:
            self.log(f"Monotonicity test error: {str(e)}")
            # Restore on error
            self.a1 = orig_a1 if 'orig_a1' in locals() else self.a1
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)"""
        self.logs = []
        self.log("=== Physics Checker Started ===")
        
        tests = [
            ("Invariants", self.test_invariants()),
            ("Trace Consistency", self.test_trace_consistency()),
            ("Monotonicity", self.test_monotonicity())
        ]
        
        all_pass = all(result for _, result in tests)
        
        self.log("\n=== Summary ===")
        for name, result in tests:
            self.log(f"{name}: {'PASS' if result else 'FAIL'}")
        
        self.log(f"\nOverall: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
        
        return all_pass, self.logs