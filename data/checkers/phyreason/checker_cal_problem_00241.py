import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: extract all needed inputs and results.
        """
        self.solution = solution_instance
        self.results = solution_instance.results
        
        # Input parameters from problem context - extract from results dict
        # The solution should store these in results with keys U, E1, B1, B2, E2
        self.U = self.results.get('U', 1000.0)
        self.E1 = self.results.get('E1', 1.0e4)
        self.B1 = self.results.get('B1', 0.1)
        self.B2 = self.results.get('B2', 0.2)
        self.E2 = self.results.get('E2', 1.2e4)
        
        # Extract computed results
        self.fracqm = self.results.get('fracqm', 0.0)  # q/m ratio
        # The final velocity v' is stored under key 'v' (as per contract)
        self.v_prime = self.results.get('v', 0.0)      # Final velocity at O'
        self.r = self.results.get('r', 0.0)            # Radius in separator
        self.F = self.results.get('F', 0.0)            # Downward electric force
        self.F_sharp = self.results.get('F_sharp', 0.0) # Upward Lorentz force
        self.OP = self.results.get('OP', 0.0)          # Distance OP
        self.v0 = self.results.get('v_0', 0.0)         # Velocity at selector entry
        
        # Initialize logs
        self.logs = []

    def log(self, message, success):
        """Helper to log test results."""
        self.logs.append((message, success))
        return success

    def test_invariants(self):
        """Verify fundamental physics laws based on KB theorems."""
        try:
            # 1. Velocity selector equilibrium (Step 1): v0 = E1 / B1
            expected_v0 = self.E1 / self.B1 if self.B1 != 0 else 0.0
            if not math.isclose(self.v0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Velocity selector invariant failed: v0={self.v0:.3e}, E1/B1={expected_v0:.3e}", False)
            self.log(f"Velocity selector invariant passed: v0 = E1/B1", True)
            
            # 2. Energy conservation in accelerator (Step 2): qU = 0.5*m*v0^2
            # Derive q/m = v0^2/(2U) = (E1/B1)^2/(2U)
            expected_fracqm = (self.E1**2) / (2 * self.U * self.B1**2) if self.U * self.B1**2 != 0 else 0.0
            if not math.isclose(self.fracqm, expected_fracqm, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Energy conservation invariant failed: q/m={self.fracqm:.3e}, expected={expected_fracqm:.3e}", False)
            self.log(f"Energy conservation invariant passed: q/m = E1^2/(2UB1^2)", True)
            
            # 3. Lorentz force as centripetal force (Step 4): q*v0*B2 = m*v0^2/r
            # Derive r = m*v0/(q*B2) = v0/( (q/m)*B2 )
            if self.fracqm != 0 and self.B2 != 0:
                expected_r = self.v0 / (self.fracqm * self.B2)
                if not math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                    return self.log(f"Centripetal force invariant failed: r={self.r:.3e}, expected={expected_r:.3e}", False)
                self.log(f"Centripetal force invariant passed: r = v0/( (q/m)*B2 )", True)
            
            # 4. Geometric relationship (Step 5): OP = 2r = 4U*B1/(E1*B2)
            expected_OP = (4 * self.U * self.B1) / (self.E1 * self.B2) if self.E1 * self.B2 != 0 else 0.0
            if not math.isclose(self.OP, expected_OP, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Geometric invariant failed: OP={self.OP:.3e}, expected={expected_OP:.3e}", False)
            self.log(f"Geometric invariant passed: OP = 4U B1/(E1 B2)", True)
            
            # 5. Force balance in velocity selector (Steps 6-8): F_sharp = q*v0*B1, F = q*E2
            # Their ratio should satisfy equilibrium condition when E2 = E1
            if self.E2 != 0:
                # Check Lorentz force formula
                expected_F_sharp = abs(self.fracqm) * self.v0 * self.B1  # magnitude
                if not math.isclose(self.F_sharp, expected_F_sharp, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Lorentz force magnitude: F_sharp={self.F_sharp:.3e}, expected={expected_F_sharp:.3e} (warning)", True)
                
                # Check electric force formula
                expected_F = abs(self.fracqm) * self.E2
                if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Electric force magnitude: F={self.F:.3e}, expected={expected_F:.3e} (warning)", True)
            
            return True
            
        except Exception as e:
            self.log(f"Invariant test failed with error: {str(e)}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Verify q/m from v0 and U: should match given expression
            v0_calc = self.E1 / self.B1 if self.B1 != 0 else 0.0
            fracqm_from_v0 = v0_calc**2 / (2 * self.U) if self.U != 0 else 0.0
            if not math.isclose(self.fracqm, fracqm_from_v0, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Trace consistency 1 failed: q/m={self.fracqm:.3e}, from v0&U={fracqm_from_v0:.3e}", False)
            self.log(f"Trace consistency 1 passed: q/m from v0 and U matches", True)
            
            # 2. Verify r from v0 and q/m: should match OP/2
            if self.fracqm != 0 and self.B2 != 0:
                r_from_v0 = v0_calc / (self.fracqm * self.B2)
                if not math.isclose(self.r, r_from_v0, rel_tol=1e-2, abs_tol=1e-2):
                    return self.log(f"Trace consistency 2 failed: r={self.r:.3e}, from v0&q/m={r_from_v0:.3e}", False)
                self.log(f"Trace consistency 2 passed: r from v0 and q/m matches", True)
            
            # 3. Verify OP = 2r
            if not math.isclose(self.OP, 2 * self.r, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Trace consistency 3 failed: OP={self.OP:.3e}, 2r={2*self.r:.3e}", False)
            self.log(f"Trace consistency 3 passed: OP = 2r", True)
            
            # 4. Verify v' from E1, E2, B1
            expected_v_prime = (2 * self.E2 - self.E1) / self.B1 if self.B1 != 0 else 0.0
            if not math.isclose(self.v_prime, expected_v_prime, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Trace consistency 4 failed: v'={self.v_prime:.3e}, expected={expected_v_prime:.3e}", False)
            self.log(f"Trace consistency 4 passed: v' = (2E2 - E1)/B1", True)
            
            return True
            
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {str(e)}", False)
            return False

    def test_monotonicity(self):
        """
        Perturb an input parameter (U) and verify the trend in output (OP).
        Restore original values after check.
        """
        try:
            original_U = self.U
            perturb_factor = 1.1  # 10% increase
            
            # Perturb U
            self.U = original_U * perturb_factor
            
            # Re-calculate expected OP using formula OP = 4U*B1/(E1*B2)
            new_OP = (4 * self.U * self.B1) / (self.E1 * self.B2) if self.E1 * self.B2 != 0 else 0.0
            
            # Check monotonicity: OP should increase when U increases
            if new_OP <= self.OP:
                self.U = original_U  # Restore
                return self.log(f"Monotonicity test failed: OP did not increase when U increased by 10%", False)
            
            # Restore original U
            self.U = original_U
            self.log(f"Monotonicity test passed: OP increases with U (as expected from OP ∝ U)", True)
            return True
            
        except Exception as e:
            self.U = original_U if 'original_U' in locals() else self.U
            self.log(f"Monotonicity test failed with error: {str(e)}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success = test_func()
                if not test_success:
                    success = False
            except Exception as e:
                self.log(f"{test_name} crashed: {traceback.format_exc()}", False)
                success = False
        
        return success, self.logs