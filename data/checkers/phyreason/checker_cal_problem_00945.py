import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs (using defaults from problem statement)
        self.U = getattr(solution_instance, 'U', 200.0)  # generator voltage in V
        self.P_out = getattr(solution_instance, 'P_out', 100000.0)  # output power in W
        self.R = getattr(solution_instance, 'R', 100.0)  # line resistance in Ω
        
        # Safe extraction of results with fallbacks
        results_dict = getattr(solution_instance, 'results', {})
        self.DeltaP = float(results_dict.get('DeltaP', 0.0))
        self.I_2 = float(results_dict.get('I_2', 0.0))
        self.I_1 = float(results_dict.get('I_1', 0.0))
        
        # Handle turns ratio - it could be a string "1:50" or a float
        ratio_value = results_dict.get('n_1:n_2', 0.0)
        
        # Initialize ratio values
        self.n1 = 0.0
        self.n2 = 0.0
        self.turns_ratio = 0.0
        self.stored_ratio_value = ratio_value
        
        try:
            # Case 1: ratio_value is a string like "1:50"
            if isinstance(ratio_value, str):
                n1_str, n2_str = str(ratio_value).replace(' ', '').split(':')
                self.n1 = float(n1_str)
                self.n2 = float(n2_str)
                if self.n1 != 0:
                    self.turns_ratio = self.n2 / self.n1
            # Case 2: ratio_value is a float
            elif isinstance(ratio_value, (int, float)):
                ratio_float = float(ratio_value)
                self.turns_ratio = ratio_float
                # Try to interpret as n1/n2 or n2/n1
                if ratio_float != 0:
                    from fractions import Fraction
                    frac = Fraction(ratio_float).limit_denominator()
                    self.n1 = frac.numerator
                    self.n2 = frac.denominator
        except (ValueError, AttributeError, TypeError, ZeroDivisionError):
            self.n1, self.n2, self.turns_ratio = 0.0, 0.0, 0.0
        
        # Initialize logs
        self.logs = []
        
    def _log(self, msg, status):
        self.logs.append(f"{status}: {msg}")
        
    def _check_close(self, a, b, name=""):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        
    def test_invariants(self):
        """Verify conservation laws and physical relationships from KB."""
        try:
            # 1. Check power loss percentage (10% of output)
            expected_DeltaP = 0.1 * self.P_out
            if not self._check_close(self.DeltaP, expected_DeltaP):
                self._log(f"Power loss mismatch: ΔP={self.DeltaP}, expected {expected_DeltaP}", "FAIL")
                return False
            self._log(f"Power loss check passed: ΔP={self.DeltaP}", "PASS")
            
            # 2. Check Joule's Law: ΔP = I₂²R
            expected_I2_squared = self.DeltaP / self.R
            actual_I2_squared = self.I_2 ** 2
            if not self._check_close(actual_I2_squared, expected_I2_squared):
                self._log(f"Joule's Law mismatch: I₂²R={actual_I2_squared*self.R}, ΔP={self.DeltaP}", "FAIL")
                return False
            self._log(f"Joule's Law check passed: I₂={self.I_2}A", "PASS")
            
            # 3. Check generator power: P_out = U·I₁
            expected_I1 = self.P_out / self.U
            if not self._check_close(self.I_1, expected_I1):
                self._log(f"Generator power mismatch: I₁={self.I_1}A, expected {expected_I1}A", "FAIL")
                return False
            self._log(f"Generator power check passed: I₁={self.I_1}A", "PASS")
            
            # 4. Check transformer current ratio
            if self.I_1 != 0:
                # The stored ratio could be n1/n2 or n2/n1
                # We need to check both possibilities
                n1_over_n2 = self.I_2 / self.I_1  # Should be 0.02
                n2_over_n1 = self.I_1 / self.I_2  # Should be 50
                
                # Check if stored ratio matches either n1/n2 or n2/n1
                matches_n1_over_n2 = self._check_close(self.turns_ratio, n1_over_n2)
                matches_n2_over_n1 = self._check_close(self.turns_ratio, n2_over_n1)
                
                if not (matches_n1_over_n2 or matches_n2_over_n1):
                    self._log(f"Transformer ratio mismatch: stored={self.turns_ratio}, n1/n2={n1_over_n2}, n2/n1={n2_over_n1}", "FAIL")
                    return False
                    
                # Determine which interpretation is correct
                if matches_n1_over_n2:
                    self._log(f"Transformer ratio check passed: n₁/n₂={self.turns_ratio}", "PASS")
                else:
                    self._log(f"Transformer ratio check passed: n₂/n₁={self.turns_ratio}", "PASS")
            else:
                self._log("Transformer ratio check skipped (I₁=0)", "INFO")
            
            return True
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", "ERROR")
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Recompute steps from the analysis
            # Step 1: ΔP = 10% of P_out
            step1_DeltaP = 0.1 * self.P_out
            if not self._check_close(self.DeltaP, step1_DeltaP):
                self._log(f"Trace mismatch step1: ΔP={self.DeltaP} ≠ {step1_DeltaP}", "FAIL")
                return False
                
            # Step 2: I₂ = sqrt(ΔP/R)
            step2_I2 = math.sqrt(self.DeltaP / self.R)
            if not self._check_close(self.I_2, step2_I2):
                self._log(f"Trace mismatch step2: I₂={self.I_2} ≠ {step2_I2}", "FAIL")
                return False
                
            # Step 3: I₁ = P_out / U
            step3_I1 = self.P_out / self.U
            if not self._check_close(self.I_1, step3_I1):
                self._log(f"Trace mismatch step3: I₁={self.I_1} ≠ {step3_I1}", "FAIL")
                return False
                
            # Step 4: Check transformer ratio consistency
            # The stored ratio should be consistent with either I₂/I₁ or I₁/I₂
            if self.I_1 != 0:
                n1_over_n2 = self.I_2 / self.I_1  # Should be 0.02
                n2_over_n1 = self.I_1 / self.I_2  # Should be 50
                
                # Check if stored ratio matches either
                matches_n1_over_n2 = self._check_close(self.turns_ratio, n1_over_n2)
                matches_n2_over_n1 = self._check_close(self.turns_ratio, n2_over_n1)
                
                if not (matches_n1_over_n2 or matches_n2_over_n1):
                    self._log(f"Trace mismatch step4: stored ratio={self.turns_ratio} doesn't match I₂/I₁={n1_over_n2} or I₁/I₂={n2_over_n1}", "FAIL")
                    return False
                    
            self._log("All trace steps consistent", "PASS")
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", "ERROR")
            return False
            
    def test_monotonicity(self):
        """Perturb input power and verify expected trends."""
        original_P_out = self.P_out
        try:
            # Perturbation: increase P_out by 10%
            perturbed_P_out = original_P_out * 1.1
            
            # Recalculate using physical formulas (mimicking student's method)
            perturbed_DeltaP = 0.1 * perturbed_P_out
            perturbed_I2 = math.sqrt(perturbed_DeltaP / self.R)
            perturbed_I1 = perturbed_P_out / self.U
            
            # Expected trend: When P_out increases, both I₁ and I₂ increase
            # I₁ increases linearly, I₂ increases as sqrt(P_out)
            # Therefore ratio I₂/I₁ should DECREASE
            if perturbed_I1 <= self.I_1:
                self._log(f"Monotonicity violated: I₁ should increase with P_out", "FAIL")
                return False
                
            if perturbed_I2 <= self.I_2:
                self._log(f"Monotonicity violated: I₂ should increase with P_out", "FAIL")
                return False
                
            if self.I_1 != 0:
                original_ratio = self.I_2 / self.I_1
                perturbed_ratio = perturbed_I2 / perturbed_I1
                if perturbed_ratio >= original_ratio:
                    self._log(f"Monotonicity violated: I₂/I₁ ratio should decrease", "FAIL")
                    return False
                    
            self._log("Monotonicity check passed: trends match physical expectations", "PASS")
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", "ERROR")
            return False
        finally:
            # Restore original value
            self.P_out = original_P_out
            
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...", "INFO")
            if not test_func():
                success = False
                
        return success, self.logs