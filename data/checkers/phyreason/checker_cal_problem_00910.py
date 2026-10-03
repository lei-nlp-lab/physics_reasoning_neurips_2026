import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Input parameters
        self.P_elec = 20000.0  # Output power in watts (20 kW)
        self.R = 6.0           # Transmission line resistance in ohms
        
        # Safely extract results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        self.P_electric = getattr(solution_instance, 'P_electric', 
                                 self.results.get('P_electric', 20000.0))
        self.U_3 = getattr(solution_instance, 'U_3', 
                          self.results.get('U_3', 0.0))
        self.delta_U = getattr(solution_instance, 'delta_U', 
                              self.results.get('delta_U', 0.0))
        self.P_loss = getattr(solution_instance, 'P_loss', 
                             self.results.get('P_loss', 0.0))
        self.U_4 = getattr(solution_instance, 'U_4', 
                          self.results.get('U_4', 0.0))
        
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def test_invariants(self):
        """Verify fundamental physical laws."""
        try:
            # 1. Correct Joule's Law test: P_loss = I²R
            # First compute I from ΔU = I×R (correct physics)
            if self.R != 0 and self.delta_U >= 0:
                I_calculated = self.delta_U / self.R
                P_loss_joule = I_calculated**2 * self.R
                
                if not math.isclose(self.P_loss, P_loss_joule, rel_tol=0.02, abs_tol=1.0):
                    self.log(f"FAIL: Joule's Law mismatch. P_loss={self.P_loss:.1f}, I²R={P_loss_joule:.1f}")
                    self.log(f"      I from ΔU/R = {I_calculated:.2f} A")
                else:
                    self.log(f"PASS: Joule's Law (P_loss = I²R) holds within 2%")
            else:
                self.log("SKIP: Joule's Law test (R=0 or ΔU invalid)")
            
            # 2. Transformer voltage ratio check
            if self.U_4 > 0:
                ratio = self.U_3 / self.U_4
                # Both step-up (ratio < 1) and step-down (ratio > 1) are physically possible
                # depending on transformer configuration
                if ratio > 0:
                    self.log(f"INFO: Transformer ratio U_3/U_4 = {ratio:.2f}")
                else:
                    self.log(f"WARN: Invalid transformer ratio {ratio:.2f}")
            
            # 3. Energy conservation: For ideal transformers, power is conserved
            # At transmission line: P_elec = P_loss + P_3, where P_3 = U_3 × I
            # At user end (after step-down transformer): P_4 = U_4 × I_4 = P_3 (ideal transformer)
            if self.R != 0 and self.delta_U >= 0:
                I = self.delta_U / self.R  # Transmission line current
                P_3 = self.U_3 * I  # Power at point 3 (before user transformer)
                total_power_at_3 = self.P_loss + P_3
                
                if not math.isclose(self.P_elec, total_power_at_3, rel_tol=0.05, abs_tol=100.0):
                    self.log(f"FAIL: Energy conservation at transmission: P_elec={self.P_elec:.1f} ≠ P_loss+P_3={total_power_at_3:.1f}")
                else:
                    self.log(f"PASS: Energy conservation at transmission within 5%")
                
                # Check transformer power conservation (ideal)
                # P_3 should equal P_4 = U_4 × I_4, but we don't know I_4
                # Instead, we can check if the solution is self-consistent
                self.log(f"INFO: Power at U_3: P_3 = {P_3:.1f} W")
            else:
                self.log("SKIP: Energy conservation test (R=0 or ΔU invalid)")
                
        except ZeroDivisionError:
            self.log("ERROR: Division by zero in test_invariants")
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")

    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        try:
            # Check step 4: P_loss = (P_elec/U_2)² * R
            # Derive U_2 from step 7: U_2 = U_3 + delta_U
            U_2 = self.U_3 + self.delta_U
            if U_2 > 0:
                P_loss_calc = (self.P_elec / U_2)**2 * self.R
                if math.isclose(self.P_loss, P_loss_calc, rel_tol=0.02, abs_tol=1.0):
                    self.log(f"PASS: Step 4 consistency (P_loss calculation) within 2%")
                else:
                    self.log(f"FAIL: P_loss mismatch. Calculated={P_loss_calc:.1f}, Reported={self.P_loss:.1f}")
                    self.log(f"      U_2 = {U_2:.1f} V")
            else:
                self.log("SKIP: Step 4 consistency (U_2 <= 0)")
            
            # Check step 7: U_3 = U_2 - delta_U (should hold exactly)
            if math.isclose(self.U_3, U_2 - self.delta_U, rel_tol=1e-4, abs_tol=0.1):
                self.log(f"PASS: Step 7 consistency (U_3 = U_2 - ΔU)")
            else:
                self.log(f"FAIL: U_3 mismatch. Should be {U_2 - self.delta_U:.1f}, is {self.U_3:.1f}")
            
            # Check target answers
            target_P_loss_kW = 16.6
            target_U_4 = 226.2
            
            P_loss_kW = self.P_loss / 1000.0
            if math.isclose(P_loss_kW, target_P_loss_kW, rel_tol=0.02, abs_tol=0.1):
                self.log(f"PASS: Power loss matches target {target_P_loss_kW}kW within 2%")
            else:
                self.log(f"FAIL: Power loss {P_loss_kW:.1f}kW ≠ target {target_P_loss_kW}kW")
            
            if math.isclose(self.U_4, target_U_4, rel_tol=0.02, abs_tol=1.0):
                self.log(f"PASS: User voltage matches target {target_U_4}V within 2%")
            else:
                self.log(f"FAIL: User voltage {self.U_4:.1f}V ≠ target {target_U_4}V")
                
        except ZeroDivisionError:
            self.log("ERROR: Division by zero in test_trace_consistency")
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")

    def test_monotonicity(self):
        """Perturb resistance and check physical trends."""
        original_R = self.R
        
        try:
            # Test 1: Increase resistance by 10%
            self.R *= 1.1
            
            # With increased R, if current I remains the same (U_2 fixed), 
            # both ΔU and P_loss should increase
            # Compute original current from original ΔU and original R
            original_I = self.delta_U / original_R if original_R > 0 else 0
            
            # For fixed U_2, I = P_elec / U_2 would remain constant
            # So with increased R:
            new_delta_U = original_I * self.R if original_I > 0 else 0
            new_P_loss = original_I**2 * self.R if original_I > 0 else 0
            
            # Check trends
            if new_delta_U > self.delta_U * 0.95:  # Allow small tolerance
                self.log(f"PASS: ΔU increases with R (from {self.delta_U:.1f} to {new_delta_U:.1f})")
            else:
                self.log(f"FAIL: ΔU trend wrong. Expected increase, got {new_delta_U:.1f}")
            
            if new_P_loss > self.P_loss * 0.95:
                self.log(f"PASS: P_loss increases with R (from {self.P_loss:.1f} to {new_P_loss:.1f})")
            else:
                self.log(f"FAIL: P_loss trend wrong. Expected increase, got {new_P_loss:.1f}")
            
            # Restore original R
            self.R = original_R
            
        except ZeroDivisionError:
            self.log("ERROR: Division by zero in test_monotonicity")
            self.R = original_R
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            self.R = original_R

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        
        try:
            self.log("=== Physics Checker Started ===")
            self.log(f"Inputs: P_elec={self.P_elec}W, R={self.R}Ω")
            self.log(f"Results: P_loss={self.P_loss}W, U_3={self.U_3}V, ΔU={self.delta_U}V, U_4={self.U_4}V")
            
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
            
            # Determine overall success
            # Only count actual FAILs, not INFO or WARN
            fail_count = sum(1 for msg in self.logs if msg.startswith("FAIL"))
            error_count = sum(1 for msg in self.logs if msg.startswith("ERROR"))
            
            # Consider it a success if we match target answers and have no critical failures
            # The target answers are the primary requirement
            target_P_loss_kW = 16.6
            target_U_4 = 226.2
            P_loss_kW = self.P_loss / 1000.0
            
            targets_match = (math.isclose(P_loss_kW, target_P_loss_kW, rel_tol=0.02, abs_tol=0.1) and
                           math.isclose(self.U_4, target_U_4, rel_tol=0.02, abs_tol=1.0))
            
            success = (fail_count == 0 and error_count == 0 and targets_match)
            
            self.log(f"\n=== Summary ===")
            self.log(f"Targets match: {'YES' if targets_match else 'NO'}")
            self.log(f"Tests passed: {len([m for m in self.logs if m.startswith('PASS')])}")
            self.log(f"Tests failed: {fail_count}")
            self.log(f"Errors: {error_count}")
            self.log(f"Overall: {'SUCCESS' if success else 'FAILURE'}")
            
        except Exception as e:
            self.log(f"CRITICAL ERROR: {str(e)}")
            success = False
            
        return success, self.logs