import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Use getattr to avoid AttributeErrors in the sandbox.
        """
        # Extract inputs
        self.m = getattr(solution_instance, 'm', None)
        self.S = getattr(solution_instance, 'S', None)
        self.p0 = getattr(solution_instance, 'p0', None)
        self.T1 = getattr(solution_instance, 'T1', None)
        self.V1 = getattr(solution_instance, 'V1', None)
        self.V2 = getattr(solution_instance, 'V2', None)
        self.Q1 = getattr(solution_instance, 'Q1', None)
        self.g = getattr(solution_instance, 'g', None)
        
        # Extract results safely - use .results dict if available
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        else:
            self.results = {}
        
        # Safe extraction of result quantities with fallbacks
        self.T2_result = getattr(solution_instance, 'T_2', self.results.get('T_2', None))
        self.W_result = getattr(solution_instance, 'W', self.results.get('W', None))
        self.DeltaU_result = getattr(solution_instance, 'DeltaU', 
                                     self.results.get('DeltaU', self.results.get('ΔU', None)))
        
        # Store logs for debugging
        self.logs = []
    
    def _log(self, msg):
        """Helper to append log messages."""
        self.logs.append(msg)
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison for physics data."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        tests_passed = True
        
        try:
            # 1. Test ideal gas law for isobaric process (Charles's Law)
            if None not in (self.T1, self.V1, self.V2, self.T2_result):
                expected_T2 = (self.V2 / self.V1) * self.T1
                if not self._is_close(self.T2_result, expected_T2):
                    self._log(f"FAIL: T2 invariant failed. Expected {expected_T2}, got {self.T2_result}")
                    tests_passed = False
                else:
                    self._log(f"PASS: T2 = (V2/V1)*T1 invariant holds")
            
            # 2. Test work formula from step_3
            if None not in (self.p0, self.m, self.g, self.S, self.V1, self.V2, self.W_result):
                P_ext = self.p0 + (self.m * self.g) / self.S
                expected_W = -P_ext * (self.V2 - self.V1)
                if not self._is_close(self.W_result, expected_W):
                    self._log(f"FAIL: Work formula mismatch. Expected {expected_W}, got {self.W_result}")
                    tests_passed = False
                else:
                    self._log(f"PASS: Work formula W = -(p0+mg/S)(V2-V1) holds")
            
            # 3. Test first law of thermodynamics: ΔU = Q + W (note sign convention in problem)
            if None not in (self.Q1, self.W_result, self.DeltaU_result):
                expected_DeltaU = self.Q1 + self.W_result
                if not self._is_close(self.DeltaU_result, expected_DeltaU):
                    self._log(f"FAIL: First law ΔU = Q + W mismatch. Expected {expected_DeltaU}, got {self.DeltaU_result}")
                    tests_passed = False
                else:
                    self._log(f"PASS: First law ΔU = Q + W holds")
            
            # 4. Test combined expression from step_4
            if None not in (self.Q1, self.p0, self.m, self.g, self.S, self.V1, self.V2, self.DeltaU_result):
                P_ext = self.p0 + (self.m * self.g) / self.S
                expected_DeltaU_combined = self.Q1 - P_ext * (self.V2 - self.V1)
                if not self._is_close(self.DeltaU_result, expected_DeltaU_combined):
                    self._log(f"FAIL: Combined ΔU expression mismatch. Expected {expected_DeltaU_combined}, got {self.DeltaU_result}")
                    tests_passed = False
                else:
                    self._log(f"PASS: Combined ΔU = Q1 - (p0+mg/S)(V2-V1) holds")
                    
        except Exception as e:
            self._log(f"ERROR in test_invariants: {traceback.format_exc()}")
            tests_passed = False
        
        return tests_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        try:
            # Check T2 against target answer
            if None not in (self.T1, self.V1, self.V2, self.T2_result):
                expected_T2 = (self.V2 / self.V1) * self.T1
                if not self._is_close(self.T2_result, expected_T2):
                    self._log(f"FAIL: T2 trace inconsistency. Expected {expected_T2}, got {self.T2_result}")
                    tests_passed = False
                else:
                    self._log(f"PASS: T2 trace consistent with step_1")
            
            # Check ΔU against combined expression (which uses step_3 and step_4)
            if None not in (self.Q1, self.p0, self.m, self.g, self.S, self.V1, self.V2, self.DeltaU_result):
                P_ext = self.p0 + (self.m * self.g) / self.S
                expected_DeltaU = self.Q1 - P_ext * (self.V2 - self.V1)
                if not self._is_close(self.DeltaU_result, expected_DeltaU):
                    self._log(f"FAIL: ΔU trace inconsistency. Expected {expected_DeltaU}, got {self.DeltaU_result}")
                    tests_passed = False
                else:
                    self._log(f"PASS: ΔU trace consistent with step_4")
            
            # Additional check: verify W from step_3 matches what would be computed
            if None not in (self.p0, self.m, self.g, self.S, self.V1, self.V2, self.W_result):
                P_ext = self.p0 + (self.m * self.g) / self.S
                expected_W = -P_ext * (self.V2 - self.V1)
                if not self._is_close(self.W_result, expected_W):
                    self._log(f"FAIL: W trace inconsistency. Expected {expected_W}, got {self.W_result}")
                    tests_passed = False
                else:
                    self._log(f"PASS: W trace consistent with step_3")
                    
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            tests_passed = False
        
        return tests_passed
    
    def test_monotonicity(self):
        """Test monotonic trends by perturbing inputs."""
        tests_passed = True
        
        try:
            # Test 1: Perturb Q1 and check ΔU response
            if None not in (self.Q1, self.p0, self.m, self.g, self.S, self.V1, self.V2):
                P_ext = self.p0 + (self.m * self.g) / self.S
                original_DeltaU = self.Q1 - P_ext * (self.V2 - self.V1)
                
                # Increase Q1 by 1%
                perturbed_Q1 = self.Q1 * 1.01
                perturbed_DeltaU = perturbed_Q1 - P_ext * (self.V2 - self.V1)
                
                # ΔU should increase when Q1 increases (work term constant)
                if perturbed_DeltaU <= original_DeltaU:
                    self._log(f"FAIL: ΔU does not increase when Q1 increases")
                    tests_passed = False
                else:
                    self._log(f"PASS: ΔU increases monotonically with Q1")
            
            # Test 2: Perturb V2 and check W response
            if None not in (self.p0, self.m, self.g, self.S, self.V1, self.V2):
                P_ext = self.p0 + (self.m * self.g) / self.S
                original_W = -P_ext * (self.V2 - self.V1)
                
                # Increase V2 by 1% (more expansion)
                perturbed_V2 = self.V2 * 1.01
                perturbed_W = -P_ext * (perturbed_V2 - self.V1)
                
                # For expansion (V2 > V1), W should become more negative as volume increases
                if perturbed_W >= original_W:  # More negative means smaller (more negative)
                    self._log(f"FAIL: W does not become more negative with increased expansion")
                    tests_passed = False
                else:
                    self._log(f"PASS: W becomes more negative with increased expansion")
            
            # Test 3: Perturb T1 and check T2 response
            if None not in (self.T1, self.V1, self.V2):
                original_T2 = (self.V2 / self.V1) * self.T1
                perturbed_T1 = self.T1 * 1.01
                perturbed_T2 = (self.V2 / self.V1) * perturbed_T1
                
                if perturbed_T2 <= original_T2:
                    self._log(f"FAIL: T2 does not increase with T1")
                    tests_passed = False
                else:
                    self._log(f"PASS: T2 increases monotonically with T1")
                    
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            tests_passed = False
        
        return tests_passed
    
    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)."""
        self.logs = []  # Reset logs
        
        # Run tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        all_ok = invariants_ok and trace_ok and monotonicity_ok
        
        # Summary
        self._log(f"\n=== SUMMARY ===")
        self._log(f"Invariants: {'PASS' if invariants_ok else 'FAIL'}")
        self._log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self._log(f"Monotonicity: {'PASS' if monotonicity_ok else 'FAIL'}")
        self._log(f"Overall: {'PASS' if all_ok else 'FAIL'}")
        
        return all_ok, self.logs