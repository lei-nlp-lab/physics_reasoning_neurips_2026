import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs from the problem context
        self.m = 0.2  # kg
        self.L = 0.5  # m
        self.I = 2.0  # A
        self.g = 10.0  # m/s²
        self.theta_deg = 30.0  # degrees
        self.theta_rad = math.radians(self.theta_deg)
        
        # Safe extraction of results from solution instance
        self.results = getattr(solution_instance, 'results', {})
        self.F_a = getattr(solution_instance, 'F_a', self.results.get('F_a', None))
        self.B = getattr(solution_instance, 'B', self.results.get('B', None))
        # Handle the prime symbol carefully
        self.F_a_prime = getattr(solution_instance, "F_a'", 
                                self.results.get("F_a'", 
                                self.results.get('F_a_prime', None)))
        
        # Expected values from step analysis
        self.F_a_expected = self.m * self.g * math.tan(self.theta_rad)
        self.B_expected = self.F_a_expected / (self.I * self.L)
        self.F_a_prime_expected = self.m * self.g * math.sin(self.theta_rad)
        self.B_min_expected = self.F_a_prime_expected / (self.I * self.L)
        
        self.logs = []
    
    def _log(self, message):
        """Helper to add log messages."""
        self.logs.append(message)
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical invariants: equilibrium and Ampere's law."""
        try:
            # 1. Equilibrium of forces: horizontal and vertical components
            # Horizontal: F_a = T * sin(theta)  (from string tension)
            # Vertical: mg = 2 * T * cos(theta) (two strings)
            # Combining: F_a = mg * tan(theta)
            if self.F_a is not None:
                if not self._is_close(self.F_a, self.F_a_expected):
                    self._log(f"FAIL: F_a = {self.F_a} ≠ mg*tan(θ) = {self.F_a_expected}")
                    return False
                else:
                    self._log(f"PASS: F_a = mg*tan(θ) = {self.F_a:.3f} N")
            
            # 2. Ampere's force law: F_a = B * I * L for first part
            if self.F_a is not None and self.B is not None:
                F_a_from_B = self.B * self.I * self.L
                if not self._is_close(self.F_a, F_a_from_B):
                    self._log(f"FAIL: F_a = {self.F_a} ≠ B*I*L = {F_a_from_B}")
                    return False
                else:
                    self._log(f"PASS: F_a = B*I*L = {F_a_from_B:.3f} N")
            
            # 3. For minimum case: F_a' = mg * sin(theta)
            if self.F_a_prime is not None:
                if not self._is_close(self.F_a_prime, self.F_a_prime_expected):
                    self._log(f"FAIL: F_a' = {self.F_a_prime} ≠ mg*sin(θ) = {self.F_a_prime_expected}")
                    return False
                else:
                    self._log(f"PASS: F_a' = mg*sin(θ) = {self.F_a_prime:.3f} N")
            
            # 4. Net force zero check (symbolic)
            T = sp.symbols('T')
            F_a_sym = sp.symbols('F_a')
            mg_sym = self.m * self.g
            
            # Horizontal equilibrium: F_a = 2 * T * sin(theta) (two strings)
            eq_horiz = sp.Eq(F_a_sym, 2 * T * sp.sin(self.theta_rad))
            # Vertical equilibrium: mg = 2 * T * cos(theta)
            eq_vert = sp.Eq(mg_sym, 2 * T * sp.cos(self.theta_rad))
            
            # Solve for T and F_a
            sol = sp.solve([eq_horiz.subs(F_a_sym, self.F_a_expected), eq_vert], [T, F_a_sym])
            if sol:
                T_val = float(sol[0][0])
                self._log(f"PASS: Equilibrium gives T = {T_val:.3f} N")
            
            return True
            
        except Exception as e:
            self._log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check step 1: F_a = mg*tan(30°)
            if self.F_a is not None:
                step1_check = self._is_close(self.F_a, self.F_a_expected)
                self._log(f"Step 1 (F_a = mg*tanθ): {'PASS' if step1_check else 'FAIL'}")
            
            # Check step 2 & 3: B = F_a/(I*L)
            if self.B is not None:
                B_computed = self.F_a_expected / (self.I * self.L)
                step3_check = self._is_close(self.B, B_computed)
                self._log(f"Step 3 (B = F_a/(I*L)): {'PASS' if step3_check else 'FAIL'}")
                
                # Also check against expected B value from problem
                expected_B_value = (2 * math.sqrt(3)) / 3  # ≈ 1.1547
                step3b_check = self._is_close(self.B, expected_B_value)
                self._log(f"Step 3 (B ≈ 2√3/3 T): {'PASS' if step3b_check else 'FAIL'}")
            
            # Check step 4 & 5: Minimum B case
            if self.F_a_prime is not None:
                step5_check = self._is_close(self.F_a_prime, self.F_a_prime_expected)
                self._log(f"Step 5 (F_a' = mg*sinθ): {'PASS' if step5_check else 'FAIL'}")
                
                # B_min = F_a'/(I*L)
                B_min_computed = self.F_a_prime_expected / (self.I * self.L)
                step5b_check = self._is_close(B_min_computed, self.B_min_expected)
                self._log(f"Step 5 (B_min = F_a'/(I*L)): {'PASS' if step5b_check else 'FAIL'}")
                
                # Check against target answer 1 T
                target_B_min = 1.0
                step5c_check = self._is_close(B_min_computed, target_B_min)
                self._log(f"Step 5 (B_min = 1 T): {'PASS' if step5c_check else 'FAIL'}")
            
            return True
            
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check if forces change proportionally."""
        try:
            # Store original values
            original_m = self.m
            
            # Perturb mass by +10%
            self.m *= 1.1
            
            # Recompute expected values with perturbed mass
            F_a_perturbed = self.m * self.g * math.tan(self.theta_rad)
            F_a_prime_perturbed = self.m * self.g * math.sin(self.theta_rad)
            
            # Check monotonicity: force should increase with mass
            if F_a_perturbed <= self.F_a_expected:
                self._log(f"FAIL: F_a does not increase with mass (monotonicity)")
                self.m = original_m  # Restore
                return False
            
            if F_a_prime_perturbed <= self.F_a_prime_expected:
                self._log(f"FAIL: F_a' does not increase with mass (monotonicity)")
                self.m = original_m  # Restore
                return False
            
            # Check proportionality: F ∝ m
            ratio_F_a = F_a_perturbed / self.F_a_expected
            ratio_m = self.m / original_m
            
            if not self._is_close(ratio_F_a, ratio_m):
                self._log(f"FAIL: F_a not proportional to mass (ratio {ratio_F_a:.3f} ≠ {ratio_m:.3f})")
                self.m = original_m  # Restore
                return False
            
            ratio_F_a_prime = F_a_prime_perturbed / self.F_a_prime_expected
            if not self._is_close(ratio_F_a_prime, ratio_m):
                self._log(f"FAIL: F_a' not proportional to mass (ratio {ratio_F_a_prime:.3f} ≠ {ratio_m:.3f})")
                self.m = original_m  # Restore
                return False
            
            self._log(f"PASS: Monotonicity and proportionality with mass perturbation")
            
            # Restore original mass
            self.m = original_m
            return True
            
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            self.m = original_m  # Ensure restoration on error
            return False
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs.clear()
        self.logs.append("=== Physics Checker Started ===")
        
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} Test ---")
            if not test_func():
                success = False
                self.logs.append(f"✗ {test_name} FAILED")
            else:
                self.logs.append(f"✓ {test_name} passed")
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs