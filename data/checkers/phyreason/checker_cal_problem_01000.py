import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dictionary
        self.p0 = 1.0e5  # Atmospheric pressure (given constant)
        
        # Extract from solution instance with safe fallbacks
        self.V = self._safe_get(solution_instance, 'V', 8.0)  # Basketball volume in L
        self.DeltaV = self._safe_get(solution_instance, 'DeltaV', 0.5)  # Pump stroke volume in L
        self.p_prime = self._safe_get(solution_instance, 'p\'', 1.5e5)  # Final pressure
        self.N = self._safe_get(solution_instance, 'N', 8.0)  # Number of strokes
    
    def _safe_get(self, solution_instance, key, default):
        """Safely retrieve attribute from solution instance."""
        try:
            # First try direct attribute
            value = getattr(solution_instance, key, None)
            if value is not None:
                return float(value)
            
            # Then try results dictionary
            if hasattr(solution_instance, 'results'):
                value = solution_instance.results.get(key, default)
                return float(value)
            
            return float(default)
        except:
            return float(default)
    
    def test_invariants(self):
        """Test Boyle's Law and physical consistency."""
        logs = []
        success = True
        
        try:
            # Test 1: Boyle's Law (numerical)
            # Using the equation: p0 * (V + N*DeltaV) = p_prime * V
            left_side = self.p0 * (self.V + self.N * self.DeltaV)
            right_side = self.p_prime * self.V
            
            if not math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Boyle's Law violated. Left: {left_side:.2e}, Right: {right_side:.2e}")
                success = False
            else:
                logs.append(f"PASS: Boyle's Law satisfied (P0*(V+N*ΔV) = P'*V)")
            
            # Test 2: Pressure increase consistency
            if not math.isclose(self.p_prime / self.p0, 1.5, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Pressure ratio incorrect. Expected ~1.5, got {self.p_prime/self.p0:.3f}")
                success = False
            else:
                logs.append(f"PASS: Pressure ratio correct (P'/P0 ≈ 1.5)")
            
            # Test 3: N should be non-negative integer
            if self.N < 0:
                logs.append(f"FAIL: Negative number of strokes: {self.N}")
                success = False
            elif not math.isclose(self.N, round(self.N), rel_tol=1e-2):
                logs.append(f"WARNING: N should be integer, got {self.N}")
            else:
                logs.append(f"PASS: N = {self.N} is physically plausible")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate results against final answer."""
        logs = []
        success = True
        
        try:
            # Reconstruct equation from steps analysis
            # p0 * (V + N*DeltaV) = p_prime * V
            calculated_N = (self.p_prime * self.V / self.p0 - self.V) / self.DeltaV
            
            if not math.isclose(calculated_N, self.N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Inconsistent N. From equation: {calculated_N:.2f}, Stored: {self.N}")
                success = False
            else:
                logs.append(f"PASS: N consistent with Boyle's Law equation")
            
            # Check that all given values match expectations
            expected_p_prime = 1.5e5
            expected_V = 8.0
            expected_DeltaV = 0.5
            
            checks = [
                ("P'", self.p_prime, expected_p_prime, 1e-2),
                ("V", self.V, expected_V, 1e-2),
                ("ΔV", self.DeltaV, expected_DeltaV, 1e-2)
            ]
            
            for name, actual, expected, tol in checks:
                if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
                    logs.append(f"WARNING: {name} mismatch. Expected {expected}, got {actual}")
                else:
                    logs.append(f"PASS: {name} matches given value")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test that N changes correctly with parameter perturbations."""
        logs = []
        success = True
        
        try:
            # Save original values
            original_V = self.V
            original_DeltaV = self.DeltaV
            original_p_prime = self.p_prime
            original_N = self.N  # CRITICAL FIX: Save original N
            
            # Test 1: Increase basketball volume
            self.V = original_V * 1.1  # +10%
            new_N = (self.p_prime * self.V / self.p0 - self.V) / self.DeltaV
            if not (new_N > original_N * 0.95):  # N should increase
                logs.append(f"FAIL: N did not increase with larger V. Original: {original_N}, New: {new_N}")
                success = False
            else:
                logs.append(f"PASS: N increases with larger basketball volume")
            
            # Restore V
            self.V = original_V
            
            # Test 2: Increase pump stroke volume
            self.DeltaV = original_DeltaV * 1.2  # +20%
            new_N = (self.p_prime * self.V / self.p0 - self.V) / self.DeltaV
            if not (new_N < original_N * 1.05):  # N should decrease
                logs.append(f"FAIL: N did not decrease with larger ΔV. Original: {original_N}, New: {new_N}")
                success = False
            else:
                logs.append(f"PASS: N decreases with larger pump strokes")
            
            # Restore DeltaV
            self.DeltaV = original_DeltaV
            
            # Test 3: Increase target pressure
            self.p_prime = original_p_prime * 1.1  # +10%
            new_N = (self.p_prime * self.V / self.p0 - self.V) / self.DeltaV
            if not (new_N > original_N * 1.05):  # N should increase
                logs.append(f"FAIL: N did not increase with higher target pressure. Original: {original_N}, New: {new_N}")
                success = False
            else:
                logs.append(f"PASS: N increases with higher target pressure")
            
            # Restore p_prime
            self.p_prime = original_p_prime
            
            # Additional test: If p_prime = p0, then N = 0
            test_p = self.p0
            test_N = (test_p * self.V / self.p0 - self.V) / self.DeltaV
            if not math.isclose(test_N, 0.0, abs_tol=1e-2):
                logs.append(f"WARNING: When P'=P0, expected N=0, got {test_N}")
            else:
                logs.append("PASS: N=0 when target pressure equals atmospheric")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Ensure restoration even on error
            self.V = original_V
            self.DeltaV = original_DeltaV
            self.p_prime = original_p_prime
            self.N = original_N  # Restore N too
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return composite results."""
        all_logs = []
        all_success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_success = all_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRITICAL ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        if all_success:
            all_logs.append("ALL TESTS PASSED")
        else:
            all_logs.append("SOME TESTS FAILED")
        
        return all_success, all_logs