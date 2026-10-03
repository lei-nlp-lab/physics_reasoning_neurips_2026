import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        # Inputs from problem context - ONLY from results dictionary
        self.m = solution_instance.results.get('m', 1.67e-27)
        self.e = solution_instance.results.get('e', 1.6e-19)
        self.U1 = solution_instance.results.get('U1', 1000.0)
        self.B1 = solution_instance.results.get('B1', 0.1)
        self.d = solution_instance.results.get('d', 0.01)
        self.B2 = solution_instance.results.get('B2', 0.05)
        # Result quantities from the steps
        self.v = solution_instance.results.get('v', None)
        self.U_2 = solution_instance.results.get('U_2', None)
        self.R = solution_instance.results.get('R', None)
        self.logs = []
        
    def log(self, msg, success):
        self.logs.append((msg, success))
        
    def test_invariants(self):
        """Verify energy conservation and force balance laws from KB"""
        try:
            # Check for zero values that would cause division errors
            # Use a small tolerance for floating point comparison
            if abs(self.m) < 1e-30 or abs(self.e) < 1e-30 or abs(self.d) < 1e-30:
                self.log("Zero value in parameters would cause division error", False)
                return False
                
            # 1. Work-energy theorem for acceleration stage (step 1)
            if self.v is not None:
                expected_v = math.sqrt(2 * self.e * self.U1 / self.m)
                # Use fuzzy matching with 2% tolerance
                if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"Work-energy violation: v={self.v:.3e}, expected √(2eU1/m)={expected_v:.3e}", False)
                    return False
                else:
                    self.log(f"Work-energy theorem satisfied for acceleration: v ≈ √(2eU1/m)", True)
            
            # 2. Force balance in velocity selector (step 2)
            if self.U_2 is not None and self.v is not None:
                # Electric force = magnetic force: e*(U_2/d) = e*v*B1
                electric_force = self.e * self.U_2 / self.d
                magnetic_force = self.e * self.v * self.B1
                if not math.isclose(electric_force, magnetic_force, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"Force balance violation in selector: eU2/d={electric_force:.3e}, evB1={magnetic_force:.3e}", False)
                    return False
                else:
                    self.log(f"Force balance satisfied in velocity selector: eU2/d ≈ evB1", True)
            
            # 3. Lorentz force as centripetal force (step 3)
            if self.R is not None and self.v is not None:
                if abs(self.R) < 1e-30:
                    self.log("Radius R is zero, cannot compute centripetal force", False)
                    return False
                magnetic_force = self.e * self.v * self.B2
                centripetal_force = self.m * self.v**2 / self.R
                if not math.isclose(magnetic_force, centripetal_force, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"Lorentz/centripetal mismatch: evB2={magnetic_force:.3e}, mv²/R={centripetal_force:.3e}", False)
                    return False
                else:
                    self.log(f"Lorentz force equals centripetal force: evB2 ≈ mv²/R", True)
                    
            return True
        except Exception as e:
            self.log(f"Invariant test failed with error: {str(e)}", False)
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results using symbolic relationships"""
        try:
            # Check for zero values with tolerance
            if abs(self.m) < 1e-30 or abs(self.e) < 1e-30 or abs(self.B2) < 1e-30:
                self.log("Zero value in parameters would cause division error", False)
                return False
                
            if None in (self.v, self.U_2, self.R):
                self.log("Missing some results for trace consistency check", False)
                return False
                
            # Check U_2 derived from v and step 2 equation
            expected_U2 = self.B1 * self.d * self.v
            if not math.isclose(self.U_2, expected_U2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Trace inconsistency: U_2={self.U_2:.3e}, but B1*d*v={expected_U2:.3e}", False)
                return False
                
            # Check R derived from v and step 3 equation (rearranged: R = mv/(eB2))
            expected_R = self.m * self.v / (self.e * self.B2)
            if not math.isclose(self.R, expected_R, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Trace inconsistency: R={self.R:.3e}, but mv/(eB2)={expected_R:.3e}", False)
                return False
                
            # Check that R can also be expressed as (1/B2)*√(2mU1/e)
            alt_R = (1/self.B2) * math.sqrt(2 * self.m * self.U1 / self.e)
            if not math.isclose(self.R, alt_R, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Alternative R mismatch: R={self.R:.3e}, (1/B2)√(2mU1/e)={alt_R:.3e}", False)
                return False
                
            self.log("All trace consistency checks passed", True)
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}", False)
            return False
            
    def test_monotonicity(self):
        """Perturb an input (U1) and verify expected trends"""
        try:
            # Check for zero values with tolerance
            if abs(self.m) < 1e-30 or abs(self.e) < 1e-30:
                self.log("Zero value in parameters would cause division error", False)
                return False
                
            # Store original values
            original_U1 = self.U1
            perturb_factor = 1.05  # 5% increase
            
            # Perturb U1
            self.U1 *= perturb_factor
            
            # Recalculate expected values using formulas from steps
            v_pert = math.sqrt(2 * self.e * self.U1 / self.m)
            U2_pert = self.B1 * self.d * v_pert
            R_pert = self.m * v_pert / (self.e * self.B2)
            
            # Restore original U1 immediately
            self.U1 = original_U1
            
            # Check monotonic trends (all should increase with U1)
            if self.v is not None and v_pert <= self.v:
                self.log(f"Monotonicity violation: v doesn't increase with U1 (ΔU1=+5%)", False)
                return False
            if self.U_2 is not None and U2_pert <= self.U_2:
                self.log(f"Monotonicity violation: U_2 doesn't increase with U1 (ΔU1=+5%)", False)
                return False
            if self.R is not None and R_pert <= self.R:
                self.log(f"Monotonicity violation: R doesn't increase with U1 (ΔU1=+5%)", False)
                return False
                
            self.log("Monotonic trends verified: v, U_2, R all increase with U1", True)
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}", False)
            return False
            
    def run_all_tests(self):
        """Execute all checks and return success status with logs"""
        self.logs.clear()
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.logs.append((f"{test_name}: FAILED", False))
                else:
                    self.logs.append((f"{test_name}: PASSED", True))
            except Exception as e:
                success = False
                self.logs.append((f"{test_name}: ERROR - {str(e)}", False))
                
        return success, self.logs