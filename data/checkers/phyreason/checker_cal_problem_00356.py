import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all needed inputs and results from the solution instance.
        Use getattr with fallback to results dictionary to avoid AttributeError.
        """
        # Input parameters (provided in problem)
        self.E = 3.0          # EMF (V)
        self.r = 0.5          # internal resistance (Ω)
        self.L = 0.4          # distance between rails (m)
        self.B = 0.5          # magnetic field (T)
        self.m = 0.04         # mass of rod (kg)
        self.g = 10.0         # gravity (m/s²)
        self.sin_theta = 0.6  # sin(37°)
        self.cos_theta = 0.8  # cos(37°)
        self.R = 1.0          # rod resistance (Ω)
        
        # Get results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract numeric results with defaults - handle potential strings gracefully
        def safe_float(val, default=0.0):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        
        self.I = safe_float(results_dict.get('I', 0.0))
        self.F = safe_float(results_dict.get('F', 0.0))
        self.f = safe_float(results_dict.get('f', 0.0))
        self.F_N = safe_float(results_dict.get('F_N', 0.0))
        self.mu = safe_float(results_dict.get('mu', 0.0))
        
        # For direction string - handle 'N/A' key specially
        # The value might be a string or a numeric code
        direction_val = results_dict.get('N/A', '')
        if isinstance(direction_val, str):
            self.direction_str = direction_val
        else:
            # If it's numeric, convert to string
            self.direction_str = str(direction_val)
        
        # Also check 'direction' key as fallback
        if not self.direction_str:
            direction_val2 = results_dict.get('direction', '')
            if isinstance(direction_val2, str):
                self.direction_str = direction_val2
            else:
                self.direction_str = str(direction_val2)
        
        # Expected target answers
        self.target_F_str = "0.4N, direction parallel to the inclined plane upwards"
        self.target_mu = 0.5
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs storage
        self.logs = []

    def _log(self, msg, success=None):
        """Helper to log messages and success status."""
        entry = {'message': msg}
        if success is not None:
            entry['success'] = success
        self.logs.append(entry)
        
    def test_invariants(self):
        """Verify fundamental physical laws using symbolic and numerical checks."""
        try:
            # 1. Ohm's Law: I = E / (R + r)
            I_expected = self.E / (self.R + self.r)
            if not math.isclose(self.I, I_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Ohm's Law violation: I = {self.I}, expected {I_expected}", False)
                return False
            self._log(f"Ohm's Law satisfied: I = {self.I} A", True)
            
            # 2. Ampere's Force: F = I * L * B
            F_expected = self.I * self.L * self.B
            if not math.isclose(self.F, F_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Ampere's Force violation: F = {self.F}, expected {F_expected}", False)
                return False
            self._log(f"Ampere's Force satisfied: F = {self.F} N", True)
            
            # 3. Left-hand rule direction check (qualitative)
            # The force should be along the incline. We check if the string contains key phrases.
            # BUT: The direction string might not be in results (per contract), so we skip this check
            # if the direction string is empty or doesn't contain required phrases.
            required_phrases = ["parallel", "inclined", "upwards", "upward"]
            direction_lower = self.direction_str.lower()
            
            # Only perform direction check if we actually have a non-empty direction string
            if self.direction_str and self.direction_str != '' and self.direction_str != '0.0':
                if not all(phrase in direction_lower for phrase in required_phrases):
                    self._log(f"Direction check failed: got '{self.direction_str}'", False)
                    return False
                self._log(f"Direction correct: {self.direction_str}", True)
            else:
                # Direction not in results (as per contract), skip the check
                self._log("Direction check skipped (not in results per contract)", True)
            
            # 4. Equilibrium condition (Newton's 1st law): F = m*g*sinθ + f
            gravity_component = self.m * self.g * self.sin_theta
            force_balance = self.F - (gravity_component + self.f)
            if not math.isclose(force_balance, 0.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Force balance violation: F ({self.F}) != mg sinθ + f ({gravity_component + self.f})", False)
                return False
            self._log(f"Force balance satisfied: F = mg sinθ + f", True)
            
            # 5. Normal force: F_N = m*g*cosθ
            F_N_expected = self.m * self.g * self.cos_theta
            if not math.isclose(self.F_N, F_N_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Normal force violation: F_N = {self.F_N}, expected {F_N_expected}", False)
                return False
            self._log(f"Normal force correct: F_N = {self.F_N} N", True)
            
            # 6. Friction coefficient: μ = f / F_N
            if self.F_N != 0:
                mu_expected = self.f / self.F_N
                if not math.isclose(self.mu, mu_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self._log(f"Friction coefficient violation: μ = {self.mu}, expected {mu_expected}", False)
                    return False
            else:
                self._log("F_N is zero, cannot compute μ", False)
                return False
            self._log(f"Friction coefficient correct: μ = {self.mu}", True)
            
            return True
        except Exception as e:
            self._log(f"Exception in test_invariants: {traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check current from Ohm's Law
            I_expected = self.E / (self.R + self.r)
            if not math.isclose(self.I, I_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Trace consistency failed: I mismatch", False)
                return False
            
            # 2. Check Ampere force F from I, L, B
            F_expected = self.I * self.L * self.B
            if not math.isclose(self.F, F_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Trace consistency failed: F mismatch", False)
                return False
            
            # 3. Check friction force f from equilibrium: f = F - m*g*sinθ
            f_expected = self.F - self.m * self.g * self.sin_theta
            if not math.isclose(self.f, f_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Trace consistency failed: f mismatch", False)
                return False
            
            # 4. Check normal force F_N
            F_N_expected = self.m * self.g * self.cos_theta
            if not math.isclose(self.F_N, F_N_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Trace consistency failed: F_N mismatch", False)
                return False
            
            # 5. Check μ = f / F_N
            if self.F_N != 0:
                mu_expected = self.f / self.F_N
                if not math.isclose(self.mu, mu_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self._log(f"Trace consistency failed: μ mismatch", False)
                    return False
            else:
                self._log("F_N is zero, cannot compute μ", False)
                return False
            
            # 6. Check final answers against targets
            # For F: compare magnitude
            if not math.isclose(self.F, 0.4, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Final answer F magnitude mismatch", False)
                return False
            
            # For direction: check if string contains required phrases
            # Skip if direction string is not in results (per contract)
            required_phrases = ["parallel", "inclined", "upwards", "upward"]
            direction_lower = self.direction_str.lower()
            
            # Only check direction if we have a non-empty string
            if self.direction_str and self.direction_str != '' and self.direction_str != '0.0':
                if not all(phrase in direction_lower for phrase in required_phrases):
                    self._log(f"Final answer F direction mismatch", False)
                    return False
                self._log(f"Direction matches target: {self.direction_str}", True)
            else:
                # Direction not in results, skip the check
                self._log("Direction check skipped (not in results per contract)", True)
            
            # For μ
            if not math.isclose(self.mu, self.target_mu, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Final answer μ mismatch", False)
                return False
            
            self._log("All intermediate results consistent with final answers", True)
            return True
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """
        Perturb an input (EMF E) and verify expected monotonic changes.
        We restore the original value after testing.
        """
        original_E = self.E
        original_I = self.I
        original_F = self.F
        
        try:
            # Increase E by 10%
            self.E *= 1.1
            # Recompute expected I and F using formulas
            new_I = self.E / (self.R + self.r)
            new_F = new_I * self.L * self.B
            
            # Check monotonicity: I and F should increase when E increases
            if not (new_I > original_I and new_F > original_F):
                self._log(f"Monotonicity violation: E increased but I or F didn't increase", False)
                # Restore original values
                self.E = original_E
                return False
            
            # Decrease E by 20% from original
            self.E = original_E * 0.8
            new_I = self.E / (self.R + self.r)
            new_F = new_I * self.L * self.B
            
            # Check monotonicity: I and F should decrease
            if not (new_I < original_I and new_F < original_F):
                self._log(f"Monotonicity violation: E decreased but I or F didn't decrease", False)
                # Restore original values
                self.E = original_E
                return False
            
            # Restore original E
            self.E = original_E
            
            self._log("Monotonicity test passed: I and F respond correctly to E changes", True)
            return True
        except Exception as e:
            # Restore original E on exception
            self.E = original_E
            self._log(f"Exception in test_monotonicity: {traceback.format_exc()}", False)
            return False

    def run_all_tests(self):
        """Run all tests and return (success_bool, logs)."""
        self.logs = []  # Reset logs
        
        # Run tests in sequence
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            success = test_func()
            all_passed = all_passed and success
            if not success:
                self._log(f"{test_name} failed", False)
                # Continue with other tests for comprehensive feedback
        
        # Final summary
        if all_passed:
            self._log("All physics checks passed!", True)
        else:
            self._log("Some physics checks failed", False)
        
        return all_passed, self.logs