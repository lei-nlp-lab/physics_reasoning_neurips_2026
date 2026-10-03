import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Safe attribute access with fallbacks
        self.P1 = 1.0e5  # Given atmospheric pressure
        self.T1 = getattr(solution_instance, 'T_1', solution_instance.results.get('T_1', 300.0))
        self.T2 = getattr(solution_instance, 'T_2', solution_instance.results.get('T_2', 500.0))
        self.T3 = getattr(solution_instance, 'T_3', solution_instance.results.get('T_3', 800.0))
        self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', 600.0))  # From step_2
        
        self.V1 = getattr(solution_instance, 'V_1', solution_instance.results.get('V_1', None))
        self.V = getattr(solution_instance, 'V', solution_instance.results.get('V', None))  # Step_2 volume
        self.V2 = getattr(solution_instance, 'V_2', solution_instance.results.get('V_2', None))
        self.V3 = getattr(solution_instance, 'V_3', solution_instance.results.get('V_3', None))
        
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', None))
        self.p3 = getattr(solution_instance, 'p_3', solution_instance.results.get('p_3', None))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 327.0))
        
        # Use numerical values for L and S (from solution if available, otherwise defaults)
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 1.0))
        self.S = getattr(solution_instance, 'S', solution_instance.results.get('S', 1.0))
        
        self.logs = []
        
    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((message, success))
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching for floats."""
        try:
            a_num = float(a)
            b_num = float(b)
            return math.isclose(a_num, b_num, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """
        Verify conservation laws and thermodynamic consistency.
        """
        success = True
        
        # 1. Check isobaric process until piston reaches left end (T1 -> T)
        if self.V1 is not None and self.V is not None:
            # Charles's Law: V1/T1 = V/T
            lhs = self.V1 / self.T1
            rhs = self.V / self.T
            if not self._isclose(lhs, rhs):
                self._log(f"Charles's Law violation: V1/T1={lhs}, V/T={rhs}", False)
                success = False
            else:
                self._log("Charles's Law holds for first isobaric expansion", True)
        
        # 2. Check second isobaric process (T1 -> T2)
        if self.V1 is not None and self.V2 is not None:
            lhs = self.V1 / self.T1
            rhs = self.V2 / self.T2
            if not self._isclose(lhs, rhs):
                self._log(f"Charles's Law violation for second heating: V1/T1={lhs}, V2/T2={rhs}", False)
                success = False
            else:
                self._log("Charles's Law holds for second isobaric expansion", True)
        
        # 3. Check combined gas law for final isochoric process
        if self.V1 is not None and self.V3 is not None and self.p3 is not None:
            # P1*V1/T1 = P3*V3/T3
            lhs = self.P1 * self.V1 / self.T1
            rhs = self.p3 * self.V3 / self.T3
            if not self._isclose(lhs, rhs):
                self._log(f"Combined gas law violation: P1*V1/T1={lhs}, p3*V3/T3={rhs}", False)
                success = False
            else:
                self._log("Combined gas law holds for final isochoric process", True)
        
        # 4. Verify volume constraints
        # V should be 2*V1 (since cylinder length is 2L)
        if self.V1 is not None and self.V is not None:
            if not self._isclose(self.V, 2*self.V1):
                self._log(f"Volume constraint violation: V={self.V} != 2*V1={2*self.V1}", False)
                success = False
            else:
                self._log("Volume constraint V=2*V1 satisfied", True)
        
        # V3 should equal V (both 2L*S)
        if self.V is not None and self.V3 is not None:
            if not self._isclose(self.V3, self.V):
                self._log(f"Final volume mismatch: V3={self.V3} != V={self.V}", False)
                success = False
            else:
                self._log("Final volume V3=V constraint satisfied", True)
        
        return success
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results and final answers.
        """
        success = True
        
        # 1. Check temperature conversions
        # T1 should be 300K
        if not self._isclose(self.T1, 300.0):
            self._log(f"T1 mismatch: {self.T1} != 300K", False)
            success = False
        
        # T2 should be 500K
        if not self._isclose(self.T2, 500.0):
            self._log(f"T2 mismatch: {self.T2} != 500K", False)
            success = False
        
        # T3 should be 800K
        if not self._isclose(self.T3, 800.0):
            self._log(f"T3 mismatch: {self.T3} != 800K", False)
            success = False
        
        # T should be 600K (from isobaric expansion to left end)
        if not self._isclose(self.T, 600.0):
            self._log(f"T (piston at left end) mismatch: {self.T} != 600K", False)
            success = False
        
        # t should be 327°C (allow small rounding difference)
        if not self._isclose(self.t, 327.0):
            self._log(f"t mismatch: {self.t} != 327°C", False)
            success = False
        
        # 2. Check x = (5/3)L
        if self.x is not None and self.V1 is not None and self.S is not None:
            # V1 = L*S, so L = V1/S
            L_calc = self.V1 / self.S
            x_over_L = self.x / L_calc
            if not self._isclose(x_over_L, 5/3):
                self._log(f"x/L mismatch: {x_over_L} != 5/3", False)
                success = False
            else:
                self._log("x = (5/3)L confirmed", True)
        
        # 3. Check p3 ≈ 1.33e5 Pa
        if self.p3 is not None and self.V1 is not None and self.V3 is not None:
            expected_p3 = self.P1 * (self.T3 / self.T1) * (self.V1 / self.V3)
            if not self._isclose(self.p3, expected_p3):
                self._log(f"p3 mismatch: {self.p3} != expected {expected_p3}", False)
                success = False
            else:
                # Numerical check against target answer 1.33e5
                if not math.isclose(float(self.p3), 1.33e5, rel_tol=1e-2):
                    self._log(f"p3={self.p3} not close to 1.33e5", False)
                    success = False
                else:
                    self._log(f"p3={self.p3} ≈ 1.33e5 confirmed", True)
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb temperatures and verify monotonic trends.
        """
        success = True
        
        try:
            # Perturb T2 (increase by 10%)
            T2_perturbed = self.T2 * 1.1
            
            # For isobaric process: V2_perturbed = V1 * (T2_perturbed / T1)
            V2_perturbed = self.V1 * (T2_perturbed / self.T1)
            
            # Check monotonicity: higher temperature -> larger volume
            if float(V2_perturbed) <= float(self.V2):
                self._log(f"Monotonicity violation: T2↑ but V2 not ↑", False)
                success = False
            else:
                self._log("Volume increases with temperature (isobaric)", True)
            
            # Perturb T3 (increase by 10%)
            T3_perturbed = self.T3 * 1.1
            
            # For isochoric process: p3_perturbed = P1 * (T3_perturbed / T1) * (self.V1 / self.V3)
            p3_perturbed = self.P1 * (T3_perturbed / self.T1) * (self.V1 / self.V3)
            
            # Check monotonicity: higher temperature -> higher pressure
            if float(p3_perturbed) <= float(self.p3):
                self._log(f"Monotonicity violation: T3↑ but p3 not ↑", False)
                success = False
            else:
                self._log("Pressure increases with temperature (isochoric)", True)
                
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """
        Execute all tests and return results.
        """
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success = test_func()
                if not test_success:
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} test crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs