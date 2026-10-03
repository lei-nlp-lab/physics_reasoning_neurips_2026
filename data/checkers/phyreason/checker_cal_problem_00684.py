import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (from problem context)
        self.R1 = 2.0
        self.U1 = 4.0
        self.R2 = 5.0
        self.U2 = 5.0
        # R3 should be 1.0 Ω to match KB reference (U3=3V)
        self.R3 = 1.0
        
        # Extracted results with safe defaults
        self.E = self.results.get('E', 0.0)
        self.r = self.results.get('r', 0.0)
        self.U3 = self.results.get('U_3', 0.0)
        
        # For monotonicity perturbation
        self.original_E = self.E
        self.original_r = self.r
        self.original_U3 = self.U3
        
    def _is_close(self, a, b):
        """Fuzzy numerical matching with 1-2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify Ohm's Law and circuit invariants."""
        logs = []
        success = True
        
        try:
            # 1. Check that E and r satisfy both given measurements
            # Using U = E * R / (R + r)
            U1_calc = self.E * self.R1 / (self.R1 + self.r)
            if not self._is_close(U1_calc, self.U1):
                logs.append(f"FAIL: Ohm's Law check 1: Calculated U1={U1_calc:.3f} ≠ given {self.U1}")
                success = False
            else:
                logs.append(f"OK: Ohm's Law check 1: U1 matches (E={self.E:.3f}, r={self.r:.3f})")
            
            U2_calc = self.E * self.R2 / (self.R2 + self.r)
            if not self._is_close(U2_calc, self.U2):
                logs.append(f"FAIL: Ohm's Law check 2: Calculated U2={U2_calc:.3f} ≠ given {self.U2}")
                success = False
            else:
                logs.append(f"OK: Ohm's Law check 2: U2 matches (E={self.E:.3f}, r={self.r:.3f})")
            
            # 2. Check internal consistency: E should be > max(U1, U2) since r > 0
            if self.E <= max(self.U1, self.U2):
                logs.append(f"WARN: E={self.E:.3f} should be > max(U1,U2)={max(self.U1, self.U2)} for positive r")
            
            # 3. Check power trends: P = U^2/R
            P1 = self.U1**2 / self.R1
            P2 = self.U2**2 / self.R2
            if P2 > P1:  # U2 > U1 but R2 > R1
                logs.append("OK: Power trend consistent (U2>U1, R2>R1)")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            # From KB reference: E=6V, r=1Ω, U3=3V (with R3=1Ω)
            expected_E = 6.0
            expected_r = 1.0
            expected_U3 = 3.0
            
            # Check against expected values from KB
            if not self._is_close(self.E, expected_E):
                logs.append(f"FAIL: E={self.E:.3f} ≠ expected {expected_E}")
                success = False
            else:
                logs.append(f"OK: E={self.E:.3f} matches KB reference")
            
            if not self._is_close(self.r, expected_r):
                logs.append(f"FAIL: r={self.r:.3f} ≠ expected {expected_r}")
                success = False
            else:
                logs.append(f"OK: r={self.r:.3f} matches KB reference")
            
            if not self._is_close(self.U3, expected_U3):
                logs.append(f"FAIL: U3={self.U3:.3f} ≠ expected {expected_U3}")
                success = False
            else:
                logs.append(f"OK: U3={self.U3:.3f} matches KB reference")
            
            # Verify U3 calculation using E and r with R3=1Ω
            U3_calc = self.E * self.R3 / (self.R3 + self.r)
            if not self._is_close(U3_calc, self.U3):
                logs.append(f"FAIL: U3 calculation mismatch: {U3_calc:.3f} ≠ {self.U3}")
                success = False
            else:
                logs.append(f"OK: U3 calculation consistent with E and r")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input and check trend."""
        logs = []
        success = True
        
        try:
            # Perturb R3 by +10%
            perturbed_R3 = self.R3 * 1.1
            
            # Recalculate U3 using the same E and r
            perturbed_U3 = self.E * perturbed_R3 / (perturbed_R3 + self.r)
            
            # For fixed E and r, U3 should increase with R3 (since dU/dR = E*r/(R+r)^2 > 0)
            if perturbed_U3 > self.U3:
                logs.append(f"OK: Monotonicity: U3 increases from {self.U3:.3f} to {perturbed_U3:.3f} when R3 increases")
            else:
                logs.append(f"FAIL: Monotonicity violated: U3={perturbed_U3:.3f} ≤ original {self.U3:.3f}")
                success = False
            
            # Additional check: perturb R1 and see if recalculated E and r are consistent
            perturbed_R1 = self.R1 * 1.05
            # Solve for E and r using perturbed R1 with original U1, and original R2, U2
            # Using the formula: U = E * R / (R + r)
            # Rearranged: E = U * (R + r) / R
            # From two equations: U1*(R1+r)/R1 = U2*(R2+r)/R2
            # Solve for r numerically
            U1, R1, U2, R2 = self.U1, perturbed_R1, self.U2, self.R2
            
            # Solve for r: U1*(R1+r)/R1 = U2*(R2+r)/R2
            # Cross multiply: U1*R2*(R1+r) = U2*R1*(R2+r)
            # U1*R2*R1 + U1*R2*r = U2*R1*R2 + U2*R1*r
            # r*(U1*R2 - U2*R1) = U2*R1*R2 - U1*R2*R1
            # r = (U2*R1*R2 - U1*R2*R1) / (U1*R2 - U2*R1)
            denominator = U1 * R2 - U2 * R1
            if abs(denominator) > 1e-10:
                perturbed_r = (U2 * R1 * R2 - U1 * R2 * R1) / denominator
                perturbed_E = U1 * (R1 + perturbed_r) / R1
                
                # Check that perturbed E and r are positive
                if perturbed_E > 0 and perturbed_r > 0:
                    logs.append(f"OK: Perturbation yields valid E={perturbed_E:.3f}, r={perturbed_r:.3f}")
                    
                    # Recalculate U3 with perturbed parameters
                    perturbed_U3_2 = perturbed_E * self.R3 / (self.R3 + perturbed_r)
                    
                    # The change should be relatively small (small input perturbation)
                    rel_change = abs(perturbed_U3_2 - self.U3) / self.U3
                    if rel_change < 0.1:  # Less than 10% change
                        logs.append(f"OK: U3 change {rel_change:.1%} reasonable for 5% R1 change")
                    else:
                        logs.append(f"WARN: Large U3 change {rel_change:.1%} for small R1 perturbation")
                else:
                    logs.append("WARN: Perturbation yields non-positive E or r")
            else:
                logs.append("WARN: Denominator zero in perturbation calculation")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Final summary
        if overall_success:
            all_logs.append("\n=== ALL TESTS PASSED ===")
        else:
            all_logs.append("\n=== SOME TESTS FAILED ===")
        
        return overall_success, all_logs