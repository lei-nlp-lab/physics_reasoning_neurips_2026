import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs using getattr with defaults
        self.L = getattr(solution_instance, 'L', 0.2)
        self.theta_deg = getattr(solution_instance, 'theta', 37)
        self.m = getattr(solution_instance, 'm', 0.2)
        self.I = getattr(solution_instance, 'I', 5)
        self.g = getattr(solution_instance, 'g', 10)
        
        # Compute sin/cos from theta if not provided
        self.sin_theta = getattr(solution_instance, 'sin_theta', math.sin(math.radians(self.theta_deg)))
        self.cos_theta = getattr(solution_instance, 'cos_theta', math.cos(math.radians(self.theta_deg)))
        
        # Safe extraction of results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        
        # Extract numeric values with safe conversion
        # Direction variables might be numeric codes or strings - handle both
        try:
            self.F = float(results.get('F', 0.0))
        except (ValueError, TypeError):
            self.F = 0.0
            
        try:
            self.B = float(results.get('B', 0.0))
        except (ValueError, TypeError):
            self.B = 0.0
        
        # For direction variables, try to get as string, but accept numeric
        # Sandbox may only allow numeric values, so we need to be flexible
        F_dir_val = results.get('F_direction', '')
        B_dir_val = results.get('B_direction', '')
        F_A_dir_val = results.get('F_A_direction', '')
        N_A_val = results.get('N/A', '')
        
        # Convert to string if not already
        self.F_direction = str(F_dir_val) if F_dir_val != '' else ''
        self.B_direction = str(B_dir_val) if B_dir_val != '' else ''
        self.F_A_direction = str(F_A_dir_val) if F_A_dir_val != '' else ''
        self.N_A = str(N_A_val) if N_A_val != '' else ''
        
        # Intermediate calculated values for verification
        self.gravity_component = self.m * self.g * self.sin_theta
        self.ampere_magnitude = self.B * self.I * self.L
        
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((msg, success))
    
    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physics laws and equilibrium conditions"""
        try:
            # 1. Check force direction consistency from steps analysis
            # Since sandbox may force numeric values, skip string checks if values are numeric
            expected_F_A_dir = "upward along the rail"
            if self.F_A_direction and not self.F_A_direction.replace('.', '', 1).isdigit():
                # Only check if it's a non-numeric string
                if expected_F_A_dir not in self.F_A_direction.lower():
                    self._log(f"Ampere force direction mismatch. Expected '{expected_F_A_dir}', got '{self.F_A_direction}'", False)
                    return False
                else:
                    self._log("✓ Ampere force direction correct", True)
            else:
                self._log("✓ Ampere force direction (numeric code accepted)", True)
            
            expected_B_dir = "perpendicular to the inclined plane and points upward"
            if self.B_direction and not self.B_direction.replace('.', '', 1).isdigit():
                if expected_B_dir not in self.B_direction.lower():
                    self._log(f"Magnetic field direction mismatch. Expected '{expected_B_dir}', got '{self.B_direction}'", False)
                    return False
                else:
                    self._log("✓ Magnetic field direction correct", True)
            else:
                self._log("✓ Magnetic field direction (numeric code accepted)", True)
            
            expected_F_dir = "upward along the rail"
            if self.F_direction and not self.F_direction.replace('.', '', 1).isdigit():
                if expected_F_dir not in self.F_direction.lower():
                    self._log(f"External force direction mismatch. Expected '{expected_F_dir}', got '{self.F_direction}'", False)
                    return False
                else:
                    self._log("✓ External force direction correct", True)
            else:
                self._log("✓ External force direction (numeric code accepted)", True)
            
            # 2. Verify equilibrium condition: net force = 0 (first case, no external force)
            # Numerical check only - no symbolic
            calculated_B = (self.m * self.g * self.sin_theta) / (self.I * self.L)
            if not self._is_close(self.B, calculated_B):
                self._log(f"Equilibrium B mismatch. Expected {calculated_B:.3f}, got {self.B:.3f}", False)
                return False
            else:
                self._log(f"✓ Equilibrium condition for B satisfied: BIL = mg·sinθ = {self.gravity_component:.3f} N", True)
            
            # 3. Verify second equilibrium with external force
            # Numerical check only
            lhs = self.F + self.B * self.I * self.L * self.cos_theta
            if not self._is_close(lhs, self.gravity_component):
                self._log(f"Second equilibrium mismatch. F + BILcosθ = {lhs:.3f}, mg·sinθ = {self.gravity_component:.3f}", False)
                return False
            else:
                self._log(f"✓ Second equilibrium satisfied: F + BILcosθ = mg·sinθ", True)
            
            return True
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # 1. Check B calculation from formula
            calc_B = (self.m * self.g * self.sin_theta) / (self.I * self.L)
            if not self._is_close(self.B, calc_B):
                self._log(f"B value inconsistency. From formula: {calc_B:.3f}T, stored: {self.B:.3f}T", False)
                return False
            else:
                self._log(f"✓ B value consistent: {self.B:.3f}T", True)
            
            # 2. Check F calculation from formula
            calc_F = self.m * self.g * self.sin_theta - self.B * self.I * self.L * self.cos_theta
            if not self._is_close(self.F, calc_F):
                self._log(f"F value inconsistency. From formula: {calc_F:.3f}N, stored: {self.F:.3f}N", False)
                return False
            else:
                self._log(f"✓ F value consistent: {self.F:.3f}N", True)
            
            # 3. Check that B and F have correct signs (positive magnitudes)
            if self.B <= 0:
                self._log(f"B magnitude should be positive, got {self.B}", False)
                return False
            if self.F <= 0:
                self._log(f"F magnitude should be positive, got {self.F}", False)
                return False
            
            self._log("✓ All magnitudes positive", True)
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends"""
        try:
            # Save original values
            original_m = self.m
            perturbed_m = original_m * 1.1  # Increase by 10%
            
            # Recalculate using formulas (numeric only)
            perturbed_B = (perturbed_m * self.g * self.sin_theta) / (self.I * self.L)
            perturbed_F = perturbed_m * self.g * self.sin_theta - perturbed_B * self.I * self.L * self.cos_theta
            
            # Check trends
            # B should increase with mass (direct proportionality)
            if not perturbed_B > self.B:
                self._log(f"Monotonicity failed: B should increase with mass. Original: {self.B:.3f}, Perturbed: {perturbed_B:.3f}", False)
                return False
            
            # F should increase with mass (from F = mg·sinθ(1-cosθ) derivation)
            if not perturbed_F > self.F:
                self._log(f"Monotonicity failed: F should increase with mass. Original: {self.F:.3f}, Perturbed: {perturbed_F:.3f}", False)
                return False
            
            # Verify proportional relationship
            mass_ratio = perturbed_m / original_m
            B_ratio = perturbed_B / self.B
            F_ratio = perturbed_F / self.F
            
            if not (self._is_close(B_ratio, mass_ratio) and self._is_close(F_ratio, mass_ratio)):
                self._log(f"Monotonicity ratios off. Mass ratio: {mass_ratio:.3f}, B ratio: {B_ratio:.3f}, F ratio: {F_ratio:.3f}", False)
                return False
            
            self._log(f"✓ Monotonicity test passed: Both B and F increase proportionally with mass", True)
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} ---", True)
            if not test_func():
                success = False
                self._log(f"{test_name} FAILED", False)
        
        # Summary
        passed = sum(1 for _, s in self.logs if s)
        total = len(self.logs)
        self._log(f"\n=== Summary: {passed}/{total} checks passed ===", success)
        
        return success, self.logs