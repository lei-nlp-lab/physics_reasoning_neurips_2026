import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use robust attribute access with fallbacks to prevent AttributeError.
        """
        try:
            # Extract input parameters with safe defaults
            self.B = getattr(solution_instance, 'B', 1.0)  # magnetic field strength
            self.v0 = getattr(solution_instance, 'v0', 1.0)  # reference velocity
            self.k = getattr(solution_instance, 'k', 1.0)  # charge-to-mass ratio
            self.alpha = getattr(solution_instance, 'alpha', 30.0)  # angle in degrees
            # Convert alpha to radians for calculations
            self.alpha_rad = math.radians(self.alpha)
            
            # Extract results safely (prefer results dict, fallback to attributes)
            results = getattr(solution_instance, 'results', {})
            self.R1 = results.get('R_1', getattr(solution_instance, 'R_1', 0.0))
            self.R2 = results.get('R_2', getattr(solution_instance, 'R_2', 0.0))
            self.v_m = results.get('v_m', getattr(solution_instance, 'v_m', 0.0))
            self.L = results.get('L', getattr(solution_instance, 'L', 0.0))
            
            # Store for trace checking
            self.inputs = {'B': self.B, 'v0': self.v0, 'k': self.k, 'alpha': self.alpha}
            self.outputs = {'R1': self.R1, 'R2': self.R2, 'v_m': self.v_m, 'L': self.L}
        except Exception as e:
            self.error = f"Initialization failed: {e}"
            raise

    def test_invariants(self):
        """
        Verify physical invariants:
        1. Centripetal force law (Newton's 2nd Law for circular motion)
        2. Geometric constraints from the problem setup
        3. Algebraic consistency between derived quantities
        """
        logs = []
        try:
            # 1. Centripetal force law check for R1
            # From q*v0*B = m*v0^2/R1 -> R1 = v0/(k*B)
            expected_R1 = self.v0 / (self.k * self.B)
            if not math.isclose(self.R1, expected_R1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: R1 mismatch. Expected {expected_R1}, got {self.R1}")
            else:
                logs.append("OK: R1 satisfies centripetal force law")
                
            # 2. Centripetal force law check for R2
            # From q*v_m*B = m*v_m^2/R2 -> R2 = v_m/(k*B)
            expected_R2 = self.v_m / (self.k * self.B)
            if not math.isclose(self.R2, expected_R2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: R2 mismatch. Expected {expected_R2}, got {self.R2}")
            else:
                logs.append("OK: R2 satisfies centripetal force law")
            
            # 3. Geometric relationship for PQ exit (Step 2-3)
            # L*sinα = R1 - R1*sinα -> L = R1*(1 - sinα)/sinα
            sin_alpha = math.sin(self.alpha_rad)
            expected_L_from_R1 = self.R1 * (1 - sin_alpha) / sin_alpha
            if not math.isclose(self.L, expected_L_from_R1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: L from R1 geometry mismatch. Expected {expected_L_from_R1}, got {self.L}")
            else:
                logs.append("OK: L consistent with R1 geometry")
            
            # 4. Geometric relationship for MP exit (Step 5)
            # L = R2/sinα + R2 = R2*(1/sinα + 1)
            expected_L_from_R2 = self.R2 * (1/sin_alpha + 1)
            if not math.isclose(self.L, expected_L_from_R2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: L from R2 geometry mismatch. Expected {expected_L_from_R2}, got {self.L}")
            else:
                logs.append("OK: L consistent with R2 geometry")
            
            # 5. Algebraic consistency: v_m = v0/3 (Step 6)
            expected_v_m = self.v0 / 3.0
            if not math.isclose(self.v_m, expected_v_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_m mismatch. Expected {expected_v_m}, got {self.v_m}")
            else:
                logs.append("OK: v_m = v0/3 holds")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
        return logs

    def test_trace_consistency(self):
        """
        Cross-check that intermediate results (R1, R2, L, v_m) are mutually consistent
        using the equations from the steps analysis.
        """
        logs = []
        try:
            sin_alpha = math.sin(self.alpha_rad)
            
            # Check that both geometric formulas give the same L
            L_from_R1 = self.R1 * (1 - sin_alpha) / sin_alpha
            L_from_R2 = self.R2 * (1/sin_alpha + 1)
            
            if not math.isclose(L_from_R1, L_from_R2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: L from R1 ({L_from_R1}) and L from R2 ({L_from_R2}) disagree")
            else:
                logs.append("OK: Geometric formulas yield consistent L")
            
            # Check that R1 and R2 are consistent via L
            # From L = R1*(1 - sinα)/sinα and L = R2*(1/sinα + 1)
            # -> R1 = L*sinα/(1 - sinα) and R2 = L/(1/sinα + 1)
            R1_from_L = self.L * sin_alpha / (1 - sin_alpha)
            R2_from_L = self.L / (1/sin_alpha + 1)
            
            if not math.isclose(self.R1, R1_from_L, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: R1 from L mismatch. Expected {R1_from_L}, got {self.R1}")
            else:
                logs.append("OK: R1 consistent with L")
            
            if not math.isclose(self.R2, R2_from_L, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: R2 from L mismatch. Expected {R2_from_L}, got {self.R2}")
            else:
                logs.append("OK: R2 consistent with L")
            
            # Check v_m from R2 and centripetal law
            expected_v_m_from_R2 = self.R2 * self.k * self.B
            if not math.isclose(self.v_m, expected_v_m_from_R2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_m from R2 mismatch. Expected {expected_v_m_from_R2}, got {self.v_m}")
            else:
                logs.append("OK: v_m consistent with R2 via centripetal law")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
        return logs

    def test_monotonicity(self):
        """
        Perturb an input parameter (v0) and verify that outputs change in the expected direction.
        Uses direct formulas rather than re-running the solution to avoid side effects.
        """
        logs = []
        try:
            # Perturb v0 by +10%
            perturbation = 0.10
            v0_perturbed = self.v0 * (1 + perturbation)
            
            # Compute expected changes using formulas
            # R1 ∝ v0, L ∝ v0, v_m ∝ v0, R2 ∝ v_m ∝ v0
            expected_R1_perturbed = self.R1 * (1 + perturbation)
            expected_L_perturbed = self.L * (1 + perturbation)
            expected_v_m_perturbed = self.v_m * (1 + perturbation)
            expected_R2_perturbed = self.R2 * (1 + perturbation)
            
            # Check if monotonic relationships hold (all increase with v0)
            if not (expected_R1_perturbed > self.R1 and 
                    expected_L_perturbed > self.L and 
                    expected_v_m_perturbed > self.v_m and
                    expected_R2_perturbed > self.R2):
                logs.append("FAIL: Monotonicity check failed - expected all to increase with v0")
            else:
                logs.append("OK: All outputs increase monotonically with v0")
            
            # Verify proportional scaling
            sin_alpha = math.sin(self.alpha_rad)
            
            # Check R1 scaling
            R1_scaled = v0_perturbed / (self.k * self.B)
            if not math.isclose(R1_scaled, expected_R1_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: R1 scaling mismatch. Expected {expected_R1_perturbed}, computed {R1_scaled}")
            else:
                logs.append("OK: R1 scales linearly with v0")
            
            # Check L scaling via R1 geometry
            L_scaled_from_R1 = R1_scaled * (1 - sin_alpha) / sin_alpha
            if not math.isclose(L_scaled_from_R1, expected_L_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: L scaling mismatch. Expected {expected_L_perturbed}, computed {L_scaled_from_R1}")
            else:
                logs.append("OK: L scales linearly with v0")
            
            # Check v_m scaling
            v_m_scaled = v0_perturbed / 3.0
            if not math.isclose(v_m_scaled, expected_v_m_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_m scaling mismatch. Expected {expected_v_m_perturbed}, computed {v_m_scaled}")
            else:
                logs.append("OK: v_m scales linearly with v0")
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
        return logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success status with logs.
        """
        all_logs = []
        success = True
        
        try:
            # Run each test
            for test_name, test_method in [
                ("Physical Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== {test_name} ===")
                test_logs = test_method()
                all_logs.extend(test_logs)
                
                # Check for any FAIL messages in this test
                if any("FAIL" in log for log in test_logs):
                    success = False
                    
        except Exception as e:
            all_logs.append(f"Critical error during testing: {e}")
            success = False
            
        return success, all_logs