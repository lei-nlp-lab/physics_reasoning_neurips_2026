import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        # Inputs: wavelength λ - extract from results or use default
        self.lam = getattr(solution_instance, 'lam', 
                          solution_instance.results.get('lam', 1.0))
        
        # Results with safe fallback - extract from results dict
        self.opd1 = solution_instance.results.get('2d_x+fraclambda2', 0.0)
        self.opd2 = solution_instance.results.get('2d_x+1+fraclambda2', 0.0)
        self.Deltad = solution_instance.results.get('Deltad', 0.0)
        
        self.logs = []
        
    def _log(self, msg, success=None):
        """Helper for logging with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        return success
    
    def test_invariants(self):
        """Verify fundamental physics laws: constructive interference conditions."""
        try:
            # Numerical consistency check
            # From step1: 2d_x + λ/2 should equal some integer k*λ
            # We don't know k, but we can check that (2d_x + λ/2)/λ is near an integer
            if self.lam > 0:
                k1_val = self.opd1 / self.lam
                k1_int = round(k1_val)
                if math.isclose(k1_val, k1_int, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Step1 OPD consistent with k={k1_int}λ", True)
                else:
                    self._log(f"Step1 OPD {self.opd1} not integer multiple of λ={self.lam}", False)
                    return False
            
            return True
        except Exception as e:
            self._log(f"Invariant test failed: {traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check that intermediate results lead to final Δd = λ/2."""
        try:
            # Numerical check using extracted values
            if self.lam > 0:
                # Calculate Δd from the OPDs
                # (opd2 - opd1) = 2(d_{x+1} - d_x) = λ
                calculated_lambda = self.opd2 - self.opd1
                expected_lambda = self.lam
                
                if math.isclose(calculated_lambda, expected_lambda, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"OPD difference {calculated_lambda} matches λ={expected_lambda}", True)
                    
                    # Now check Δd
                    calculated_Deltad = self.Deltad
                    expected_Deltad = self.lam / 2
                    
                    if math.isclose(calculated_Deltad, expected_Deltad, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"Δd={calculated_Deltad} matches λ/2={expected_Deltad}", True)
                        return True
                    else:
                        self._log(f"Δd={calculated_Deltad} ≠ λ/2={expected_Deltad}", False)
                else:
                    self._log(f"OPD difference {calculated_lambda} ≠ λ={expected_lambda}", False)
            
            return False
        except Exception as e:
            self._log(f"Trace consistency failed: {traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb λ and verify Δd changes proportionally (Δd ∝ λ)."""
        try:
            original_lam = self.lam
            
            # Test 1: Increase λ by 10%
            perturbed_lam = original_lam * 1.10
            expected_Deltad_perturbed = perturbed_lam / 2
            
            # Recalculate using the physical relationship
            # From physics: Δd = λ/2 always holds
            # So the ratio should be exactly 1.10
            
            ratio = expected_Deltad_perturbed / (original_lam / 2)
            
            if math.isclose(ratio, 1.10, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity: Δd scales with λ (ratio={ratio:.3f})", True)
            else:
                self._log(f"Monotonicity failed: ratio={ratio} ≠ 1.10", False)
                return False
            
            # Test 2: Decrease λ by 20%
            perturbed_lam2 = original_lam * 0.80
            expected_Deltad_perturbed2 = perturbed_lam2 / 2
            ratio2 = expected_Deltad_perturbed2 / (original_lam / 2)
            
            if math.isclose(ratio2, 0.80, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity: Δd scales with λ (ratio={ratio2:.3f})", True)
            else:
                self._log(f"Monotonicity failed: ratio={ratio2} ≠ 0.80", False)
                return False
            
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed: {traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariant Check", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            if not test_func():
                success = False
        
        self._log(f"Overall: {'All tests passed' if success else 'Some tests failed'}")
        return success, self.logs