import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from the solution instance."""
        try:
            # Inputs
            self.M = getattr(solution_instance, 'M', getattr(solution_instance, 'inputs', {}).get('M', 8.0))
            self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 2.0))
            self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'inputs', {}).get('F', 8.0))
            self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'inputs', {}).get('v0', 1.5))
            self.mu = getattr(solution_instance, 'mu', getattr(solution_instance, 'inputs', {}).get('mu', 0.2))
            self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
            
            # Results - using safe access with fallback to 0.0
            results = getattr(solution_instance, 'results', {})
            self.a1 = results.get('a₁', results.get('a1', 0.0))
            self.a2 = results.get('a₂', results.get('a2', 0.0))
            self.a3 = results.get('a₃', results.get('a3', 0.0))
            self.t1 = results.get('t₁', results.get('t1', 0.0))
            self.v_common = results.get('v', 0.0)  # Common velocity after t1
            self.v_final = results.get("v''", results.get('v_final', 0.0))  # Final velocity at t=1.5s
            self.f_fric = results.get('f', 0.0)  # Frictional force during combined motion
            self.f_max = results.get('f_max', 0.0)  # Maximum static friction
            
            # Expected values from steps analysis
            self.expected_a1 = 2.0
            self.expected_a2 = 0.5
            self.expected_t1 = 1.0
            self.expected_v_common = 2.0
            self.expected_a3 = 0.8
            self.expected_f_fric = 1.6
            self.expected_f_max = 4.0
            self.expected_v_final = 2.4
            
        except Exception as e:
            self.error_log = f"Initialization error: {str(e)}"
            raise
    
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics invariants using symbolic and numerical methods."""
        logs = []
        
        try:
            # Symbolic variables
            M, m, F, v0, mu, g, a1, a2, a3, t1, v, v_final, f, f_max = sp.symbols(
                'M m F v0 mu g a1 a2 a3 t1 v v_final f f_max'
            )
            
            # 1. Newton's 2nd Law for block: μmg = m*a1 → a1 = μ*g
            lhs1 = mu * m * g
            rhs1 = m * a1
            expr1 = sp.Eq(lhs1, rhs1)
            # Substitute numerical values
            num_val1 = self.mu * self.m * self.g
            num_val2 = self.m * self.a1
            if not self.fuzzy_equal(num_val1, num_val2):
                logs.append(f"FAIL: Newton's 2nd Law for block: μmg = {num_val1}, m*a1 = {num_val2}")
            else:
                logs.append(f"PASS: Newton's 2nd Law for block (μmg = m*a1): {num_val1} ≈ {num_val2}")
            
            # 2. Newton's 2nd Law for cart: F - μmg = M*a2
            lhs2 = F - mu * m * g
            rhs2 = M * a2
            expr2 = sp.Eq(lhs2, rhs2)
            num_val3 = self.F - self.mu * self.m * self.g
            num_val4 = self.M * self.a2
            if not self.fuzzy_equal(num_val3, num_val4):
                logs.append(f"FAIL: Newton's 2nd Law for cart: F-μmg = {num_val3}, M*a2 = {num_val4}")
            else:
                logs.append(f"PASS: Newton's 2nd Law for cart (F-μmg = M*a2): {num_val3} ≈ {num_val4}")
            
            # 3. Newton's 2nd Law for combined system: F = (M+m)*a3
            lhs3 = F
            rhs3 = (M + m) * a3
            expr3 = sp.Eq(lhs3, rhs3)
            num_val5 = self.F
            num_val6 = (self.M + self.m) * self.a3
            if not self.fuzzy_equal(num_val5, num_val6):
                logs.append(f"FAIL: Newton's 2nd Law for combined: F = {num_val5}, (M+m)*a3 = {num_val6}")
            else:
                logs.append(f"PASS: Newton's 2nd Law for combined (F = (M+m)*a3): {num_val5} ≈ {num_val6}")
            
            # 4. Friction during combined motion: f = m*a3
            lhs4 = f
            rhs4 = m * a3
            expr4 = sp.Eq(lhs4, rhs4)
            num_val7 = self.f_fric
            num_val8 = self.m * self.a3
            if not self.fuzzy_equal(num_val7, num_val8):
                logs.append(f"FAIL: Friction during combined motion: f = {num_val7}, m*a3 = {num_val8}")
            else:
                logs.append(f"PASS: Friction during combined motion (f = m*a3): {num_val7} ≈ {num_val8}")
            
            # 5. Maximum static friction: f_max = μ*m*g
            lhs5 = f_max
            rhs5 = mu * m * g
            expr5 = sp.Eq(lhs5, rhs5)
            num_val9 = self.f_max
            num_val10 = self.mu * self.m * self.g
            if not self.fuzzy_equal(num_val9, num_val10):
                logs.append(f"FAIL: Maximum static friction: f_max = {num_val9}, μmg = {num_val10}")
            else:
                logs.append(f"PASS: Maximum static friction (f_max = μmg): {num_val9} ≈ {num_val10}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate calculations and final results."""
        logs = []
        
        try:
            # 1. Check kinematics for reaching common velocity
            # For block: v = a1*t1
            v_from_block = self.a1 * self.t1
            # For cart: v = v0 + a2*t1
            v_from_cart = self.v0 + self.a2 * self.t1
            
            if not (self.fuzzy_equal(v_from_block, self.v_common) and 
                    self.fuzzy_equal(v_from_cart, self.v_common)):
                logs.append(f"FAIL: Kinematics mismatch: v_block={v_from_block}, v_cart={v_from_cart}, v_common={self.v_common}")
            else:
                logs.append(f"PASS: Kinematics consistent: v_block={v_from_block}, v_cart={v_from_cart}, v_common={self.v_common}")
            
            # 2. Check time calculation from kinematics equations
            # From v = a1*t1 and v = v0 + a2*t1, we get t1 = v0/(a1 - a2)
            if self.a1 - self.a2 != 0:
                t1_calc = self.v0 / (self.a1 - self.a2)
                if not self.fuzzy_equal(t1_calc, self.t1):
                    logs.append(f"FAIL: Time t1 mismatch: calculated={t1_calc}, result={self.t1}")
                else:
                    logs.append(f"PASS: Time t1 consistent: calculated={t1_calc}, result={self.t1}")
            
            # 3. Check final velocity calculation
            # v_final = v_common + a3*(1.5 - t1)
            if self.t1 <= 1.5:
                v_final_calc = self.v_common + self.a3 * (1.5 - self.t1)
                if not self.fuzzy_equal(v_final_calc, self.v_final):
                    logs.append(f"FAIL: Final velocity mismatch: calculated={v_final_calc}, result={self.v_final}")
                else:
                    logs.append(f"PASS: Final velocity consistent: calculated={v_final_calc}, result={self.v_final}")
            else:
                logs.append("WARNING: t1 > 1.5s, final velocity calculation not applicable")
            
            # 4. Check that f_fric <= f_max (block doesn't slip during combined motion)
            if self.f_fric > self.f_max * 1.01:  # Allow 1% tolerance for rounding
                logs.append(f"FAIL: Friction force {self.f_fric} exceeds max static friction {self.f_max}")
            else:
                logs.append(f"PASS: Friction force {self.f_fric} ≤ max static friction {self.f_max}")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        
        return logs
    
    def test_monotonicity(self):
        """Test how results change when perturbing an input parameter."""
        logs = []
        
        try:
            # Save original values
            original_m = self.m
            perturbed_m = original_m * 1.1  # Increase by 10%
            
            # Recalculate key quantities using formulas from steps
            # 1. a1 = μ*g (independent of m)
            a1_new = self.mu * self.g
            
            # 2. a2 = (F - μ*m*g)/M
            a2_new = (self.F - self.mu * perturbed_m * self.g) / self.M
            
            # 3. t1 = v0/(a1 - a2)
            t1_new = self.v0 / (a1_new - a2_new) if (a1_new - a2_new) != 0 else float('inf')
            
            # 4. v_common = a1 * t1
            v_common_new = a1_new * t1_new
            
            # Check monotonicity trends
            # As m increases, a2 should decrease (more friction opposing cart)
            if not (a2_new < self.a2):
                logs.append(f"MONOTONICITY WARNING: a2 didn't decrease as m increased: {a2_new} vs {self.a2}")
            else:
                logs.append(f"PASS: a2 decreases as m increases: {a2_new} < {self.a2}")
            
            # As m increases, t1 should increase (takes longer to reach common velocity)
            if t1_new != float('inf') and not (t1_new > self.t1):
                logs.append(f"MONOTONICITY WARNING: t1 didn't increase as m increased: {t1_new} vs {self.t1}")
            elif t1_new != float('inf'):
                logs.append(f"PASS: t1 increases as m increases: {t1_new} > {self.t1}")
            
            # Restore original m (conceptually)
            logs.append(f"Monotonicity test complete, original m={original_m} restored")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
        
        return logs
    
    def test_numerical_accuracy(self):
        """Compare calculated results with expected values from steps analysis."""
        logs = []
        
        tests = [
            ("a₁ (block acceleration)", self.a1, self.expected_a1),
            ("a₂ (cart acceleration)", self.a2, self.expected_a2),
            ("t₁ (time to common velocity)", self.t1, self.expected_t1),
            ("v (common velocity)", self.v_common, self.expected_v_common),
            ("a₃ (combined acceleration)", self.a3, self.expected_a3),
            ("f (friction during combined)", self.f_fric, self.expected_f_fric),
            ("f_max (max static friction)", self.f_max, self.expected_f_max),
            ("v' (final velocity at t=1.5s)", self.v_final, self.expected_v_final)
        ]
        
        for name, calculated, expected in tests:
            if self.fuzzy_equal(calculated, expected):
                logs.append(f"PASS: {name}: {calculated} ≈ {expected}")
            else:
                logs.append(f"FAIL: {name}: {calculated} ≠ {expected} (expected {expected})")
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            all_logs.append("=== PhysicsChecker Test Report ===")
            all_logs.append(f"Inputs: M={self.M}kg, m={self.m}kg, F={self.F}N, v0={self.v0}m/s, μ={self.mu}, g={self.g}m/s²")
            all_logs.append(f"Results: a1={self.a1}, a2={self.a2}, t1={self.t1}, v={self.v_common}, a3={self.a3}, f={self.f_fric}, f_max={self.f_max}, v'={self.v_final}")
            
            # Test 1: Numerical accuracy against expected values
            all_logs.append("\n1. Numerical Accuracy Test:")
            accuracy_logs = self.test_numerical_accuracy()
            all_logs.extend(accuracy_logs)
            if any("FAIL" in log for log in accuracy_logs):
                success = False
            
            # Test 2: Physics invariants
            all_logs.append("\n2. Physics Invariants Test:")
            invariant_logs = self.test_invariants()
            all_logs.extend(invariant_logs)
            if any("FAIL" in log for log in invariant_logs):
                success = False
            
            # Test 3: Trace consistency
            all_logs.append("\n3. Trace Consistency Test:")
            trace_logs = self.test_trace_consistency()
            all_logs.extend(trace_logs)
            if any("FAIL" in log for log in trace_logs):
                success = False
            
            # Test 4: Monotonicity
            all_logs.append("\n4. Monotonicity Test:")
            mono_logs = self.test_monotonicity()
            all_logs.extend(mono_logs)
            # Monotonicity warnings don't necessarily mean failure
            
            # Final verdict
            all_logs.append("\n=== Summary ===")
            if success:
                all_logs.append("SUCCESS: All critical tests passed.")
            else:
                all_logs.append("FAILURE: One or more critical tests failed.")
            
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs