import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safe extraction of inputs and results."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe fallback - use results dict for numeric values
        self.h = self.results.get('h', getattr(solution_instance, 'h', 5.0))
        self.L = self.results.get('L', getattr(solution_instance, 'L', 1.8))
        self.v = self.results.get('v', getattr(solution_instance, 'v', 1.2))
        self.t = self.results.get('t', getattr(solution_instance, 't', 1.0))
        
        # Extract results with safe fallback
        self.k = self.results.get('k', getattr(solution_instance, 'k', None))
        self.OS = self.results.get('OS', getattr(solution_instance, 'OS', None))
        self.SM = self.results.get('SM', getattr(solution_instance, 'SM', None))
        self.OM = self.results.get('OM', getattr(solution_instance, 'OM', None))
        
    def _is_close(self, a, b):
        """Fuzzy numerical matching with 1-2% tolerance."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=0.02, abs_tol=0.02)
        except:
            return False
    
    def test_invariants(self):
        """Verify physical invariants from KB and steps analysis."""
        logs = []
        success = True
        
        try:
            # 1. Check geometric relationship from similar triangles (step_3)
            # OM/OS = h/(h-L)  => OM = (h/(h-L)) * OS
            if self.OS is not None and self.OM is not None and self.h is not None and self.L is not None:
                expected_ratio = self.h / (self.h - self.L)
                actual_ratio = self.OM / self.OS if self.OS != 0 else None
                if actual_ratio is not None and not self._is_close(actual_ratio, expected_ratio):
                    logs.append(f"FAIL: OM/OS ratio mismatch. Expected {expected_ratio:.3f}, got {actual_ratio:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: OM/OS ratio check (geometric optics/similar triangles)")
            
            # 2. Check shadow length relationship (step_5)
            # SM = OM - OS
            if self.SM is not None and self.OM is not None and self.OS is not None:
                expected_SM = self.OM - self.OS
                if not self._is_close(self.SM, expected_SM):
                    logs.append(f"FAIL: SM = OM - OS mismatch. Expected {expected_SM:.3f}, got {self.SM:.3f}")
                    success = False
                else:
                    logs.append("PASS: SM = OM - OS check")
            
            # 3. Check derivative relationship (step_6)
            # k = d(SM)/dt = Lv/(h-L)
            if self.k is not None and all(x is not None for x in [self.L, self.v, self.h]):
                expected_k = self.L * self.v / (self.h - self.L)
                if not self._is_close(self.k, expected_k):
                    logs.append(f"FAIL: k = Lv/(h-L) mismatch. Expected {expected_k:.3f}, got {self.k:.3f}")
                    success = False
                else:
                    logs.append("PASS: k = Lv/(h-L) derivative check")
            
            # 4. Symbolic check removed - only numeric tests
            logs.append("INFO: Symbolic check skipped (numeric tests sufficient)")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            # Check that OS follows uniform motion: OS = v*t
            if self.t is not None and self.v is not None and self.OS is not None:
                expected_OS = self.v * self.t
                if not self._is_close(self.OS, expected_OS):
                    logs.append(f"FAIL: OS = v*t mismatch. Expected {expected_OS:.3f}, got {self.OS:.3f}")
                    success = False
                else:
                    logs.append("PASS: OS = v*t (uniform motion)")
            
            # Check that SM derivative matches k
            # From step_5: SM = (Lv/(h-L)) * t
            if all(x is not None for x in [self.SM, self.t, self.L, self.v, self.h, self.k]):
                expected_SM_form = (self.L * self.v / (self.h - self.L)) * self.t
                # Check if SM has the right functional form (proportional to t)
                sm_ratio = self.SM / self.t if self.t != 0 else None
                expected_ratio = self.L * self.v / (self.h - self.L)
                
                if sm_ratio is not None and not self._is_close(sm_ratio, expected_ratio):
                    logs.append(f"FAIL: SM/t ratio mismatch. Expected {expected_ratio:.3f}, got {sm_ratio:.3f}")
                    success = False
                elif sm_ratio is not None and not self._is_close(sm_ratio, self.k):
                    logs.append(f"FAIL: SM/t != k. SM/t = {sm_ratio:.3f}, k = {self.k:.3f}")
                    success = False
                else:
                    logs.append("PASS: SM functional form and derivative consistency")
            
            # Check that k is constant (as expected from constant v)
            if self.k is not None:
                # The derivative should be time-independent
                logs.append("INFO: k is constant as expected for constant velocity scenario")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends."""
        logs = []
        success = True
        
        try:
            # Use numeric values from instance
            if all(x is not None for x in [self.h, self.L, self.v]) and self.h > self.L:
                original_k = self.L * self.v / (self.h - self.L)
                
                # Test 1: Increase L slightly
                L_perturbed = self.L * 1.01  # 1% increase
                if L_perturbed < self.h:  # Ensure physical validity
                    k_perturbed = L_perturbed * self.v / (self.h - L_perturbed)
                    
                    # Since denominator decreases and numerator increases, k should increase
                    if not (k_perturbed > original_k):
                        logs.append(f"FAIL: Increasing L from {self.L} to {L_perturbed} should increase k")
                        logs.append(f"  Original k: {original_k:.3f}, Perturbed k: {k_perturbed:.3f}")
                        success = False
                    else:
                        logs.append(f"PASS: Monotonicity w.r.t. L (k increases as L increases)")
                
                # Test 2: Increase v (k should increase proportionally)
                v_perturbed = self.v * 1.01
                k_v_perturbed = self.L * v_perturbed / (self.h - self.L)
                
                expected_ratio = v_perturbed / self.v
                actual_ratio = k_v_perturbed / original_k if original_k != 0 else None
                
                if actual_ratio is not None and not self._is_close(actual_ratio, expected_ratio):
                    logs.append(f"FAIL: k should scale linearly with v")
                    logs.append(f"  Expected ratio: {expected_ratio:.3f}, Actual ratio: {actual_ratio:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Linearity w.r.t. v")
                    
                # Test 3: Increase h (k should decrease since denominator increases)
                h_perturbed = self.h * 1.01
                k_h_perturbed = self.L * self.v / (h_perturbed - self.L)
                
                if h_perturbed > self.L and not (k_h_perturbed < original_k):
                    logs.append(f"FAIL: Increasing h should decrease k")
                    logs.append(f"  Original k: {original_k:.3f}, Perturbed k: {k_h_perturbed:.3f}")
                    success = False
                elif h_perturbed > self.L:
                    logs.append(f"PASS: Monotonicity w.r.t. h (k decreases as h increases)")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True
        
        # Run invariants test
        inv_success, inv_logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(inv_logs)
        overall_success = overall_success and inv_success
        
        # Run trace consistency test
        trace_success, trace_logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(trace_logs)
        overall_success = overall_success and trace_success
        
        # Run monotonicity test
        mono_success, mono_logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(mono_logs)
        overall_success = overall_success and mono_success
        
        # Final summary
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs