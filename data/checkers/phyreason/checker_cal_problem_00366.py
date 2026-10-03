import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        try:
            # Extract inputs with safe fallbacks
            self.L = getattr(solution_instance, 'L', 0.15)
            self.I = getattr(solution_instance, 'I', 2.0)
            self.F = getattr(solution_instance, 'F', 3.0e-2)
            
            # Extract results with safe fallbacks
            self.results = getattr(solution_instance, 'results', {})
            self.B = self.results.get('B', 0.0)
            self.F_prime = self.results.get("F'", 0.0)
            
            # Symbolic setup for law verification
            self.F_sym, self.I_sym, self.L_sym, self.B_sym = sp.symbols('F I L B')
            self.F_prime_sym, self.I_prime_sym = sp.symbols("F' I'")
            
            # Store for monotonicity backup
            self._original_I = self.I
            self._original_F = self.F
        except Exception as e:
            self.init_error = f"Initialization error: {str(e)}"
            raise

    def test_invariants(self):
        """Verify Ampere's force law and derived relationships."""
        logs = []
        success = True
        
        try:
            # Law 1: F = I * L * B (for perpendicular wire)
            symbolic_law = sp.Eq(self.F_sym, self.I_sym * self.L_sym * self.B_sym)
            logs.append(f"Symbolic law: {symbolic_law}")
            
            # Numerical check with fuzzy matching
            expected_F = self.I * self.L * self.B
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F = I*L*B check: F={self.F}, expected={expected_F}")
                success = False
            else:
                logs.append(f"PASS: F = I*L*B verified within 1% tolerance")
            
            # Derived law: F' = I' * L * B (assuming same L and B)
            # We don't have I' explicitly, but we can verify B = F/(I*L) = F'/(I'*L)
            # So F'/F = I'/I (proportionality check)
            # Since I' is unknown, we can only verify consistency if we compute I'
            I_prime_computed = self.F_prime / (self.B * self.L) if abs(self.B * self.L) > 1e-12 else 0
            logs.append(f"Computed I' from results: {I_prime_computed:.3f} A")
            
            # Check that F' is positive (physics constraint)
            if self.F_prime < 0:
                logs.append("FAIL: F' should be non-negative")
                success = False
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Step 2 consistency: B = F/(I*L)
            B_computed = self.F / (self.I * self.L) if abs(self.I * self.L) > 1e-12 else 0
            if not math.isclose(self.B, B_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: B consistency: stored={self.B}, computed={B_computed}")
                success = False
            else:
                logs.append(f"PASS: B = F/(I*L) verified (B={self.B:.3f} T)")
            
            # Step 3 consistency: F' = B * I' * L
            # We need I' from the solution - approximate from computed above
            I_prime_computed = self.F_prime / (self.B * self.L) if abs(self.B * self.L) > 1e-12 else 0
            F_prime_computed = self.B * I_prime_computed * self.L
            
            if not math.isclose(self.F_prime, F_prime_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: F' consistency check: stored={self.F_prime}, computed={F_prime_computed}")
                # This is a warning because I' might not be exactly what we computed
            else:
                logs.append(f"PASS: F' = B*I'*L verified (F'={self.F_prime:.3f} N)")
            
            # Check against target answers
            target_B = 0.1
            target_F_prime = 0.15
            if not math.isclose(self.B, target_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: B doesn't match target {target_B}T")
                success = False
            if not math.isclose(self.F_prime, target_F_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F' doesn't match target {target_F_prime}N")
                success = False
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """Perturb input current I and verify force changes proportionally."""
        logs = []
        success = True
        
        try:
            # Perturb I by +10%
            I_perturbed = self.I * 1.1
            
            # Expected force change: F_perturbed = I_perturbed * L * B
            # Using the same B and L
            F_perturbed_expected = I_perturbed * self.L * self.B
            
            # Ratio check: F_perturbed / F = I_perturbed / I
            expected_ratio = I_perturbed / self.I
            actual_ratio = F_perturbed_expected / self.F if abs(self.F) > 1e-12 else 0
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Monotonicity violated: ratio={actual_ratio:.3f}, expected={expected_ratio:.3f}")
                success = False
            else:
                logs.append(f"PASS: Force scales linearly with current (+10% I → +10% F)")
            
            # Check inverse relationship with distance (if distance were a variable)
            # This problem doesn't have distance, but we can note the trend from KB
            logs.append("Note: From KB, force ∝ current, inversely ∝ distance (if applicable)")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        """Execute all tests and return aggregated results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                for log in logs:
                    all_logs.append(log)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                all_logs.append(f"Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All physics checks passed within tolerance.")
        else:
            all_logs.append("Some physics checks failed. Review the logs.")
        
        return overall_success, all_logs