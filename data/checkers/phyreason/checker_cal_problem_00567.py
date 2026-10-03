import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        Avoid direct attribute access to solution_instance. Use getattr with fallback.
        """
        try:
            # Extract input parameters from solution_instance with safe defaults
            self.A = getattr(solution_instance, 'A', 0.05)  # Area in m²
            self.N = getattr(solution_instance, 'N', 100)   # Number of turns
            self.r = getattr(solution_instance, 'r', 1.0)   # Coil resistance in Ω
            self.R = getattr(solution_instance, 'R', 9.0)   # External resistance in Ω
            self.B = getattr(solution_instance, 'B', 1.0/math.pi)  # Magnetic field in T
            self.n = getattr(solution_instance, 'n', 300.0) # Rotational speed in rpm
            
            # Extract results from solution_instance.results dictionary with safe defaults
            results = getattr(solution_instance, 'results', {})
            
            # Extract angular velocity ω (omega) - must be numeric
            self.omega = float(results.get('$omega$', results.get('ω', 0.0)))
            # Extract instantaneous emf expression e (string) - handle separately
            e_expr_raw = results.get('e', '')
            # Convert to string if it's not already
            self.e_expr = str(e_expr_raw) if e_expr_raw is not None else ''
            # Extract instantaneous emf at t=1/60 s: e₁ - must be numeric
            self.e1 = float(results.get('$e_1$', results.get('e1', 0.0)))
            # Extract maximum emf: Eₘ - must be numeric
            self.E_m = float(results.get('$E_m$', results.get('Em', 0.0)))
            # Extract current I - must be numeric
            self.I = float(results.get('I', 0.0))
            # Extract voltage across external resistor U - must be numeric
            self.U = float(results.get('U', 0.0))
            # Extract effective emf E - must be numeric
            self.E = float(results.get('E', 0.0))
            
            # Additional derived parameters (if not provided, compute from inputs)
            if self.omega == 0.0:
                # ω = 2πn, where n is in rev/s (300 rpm = 5 rev/s)
                self.omega = 2 * math.pi * (self.n / 60.0)
            
            self.logs = []
            
        except Exception as e:
            self.logs.append(f"Initialization error: {traceback.format_exc()}")
            raise
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Ohm's Law relationships
        2. AC voltage relationships (max vs effective)
        3. Faraday's Law for rotating coil
        """
        logs = []
        try:
            # 1. Check Ohm's Law: U = I * R
            if self.I > 0 and self.U > 0 and self.R > 0:
                expected_U = self.I * self.R
                if not math.isclose(self.U, expected_U, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Ohm's Law violation: U={self.U:.3f}, I*R={expected_U:.3f}")
                else:
                    logs.append(f"Ohm's Law satisfied: U = I*R within 2% tolerance")
            
            # 2. Check effective value relationship: E = E_m / sqrt(2)
            if self.E_m > 0 and self.E > 0:
                expected_E = self.E_m / math.sqrt(2)
                if not math.isclose(self.E, expected_E, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Effective value violation: E={self.E:.3f}, E_m/√2={expected_E:.3f}")
                else:
                    logs.append(f"Effective value relationship satisfied: E = E_m/√2 within 2%")
            
            # 3. Check Faraday's Law: E_m = N * B * A * ω
            expected_Em = self.N * self.B * self.A * self.omega
            if not math.isclose(self.E_m, expected_Em, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Faraday's Law violation: E_m={self.E_m:.3f}, N*B*A*ω={expected_Em:.3f}")
            else:
                logs.append(f"Faraday's Law satisfied: E_m = N*B*A*ω within 2%")
            
            # 4. Check total resistance consistency: I = E / (r + R)
            if self.E > 0 and self.I > 0:
                expected_I = self.E / (self.r + self.R)
                if not math.isclose(self.I, expected_I, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Total resistance violation: I={self.I:.3f}, E/(r+R)={expected_I:.3f}")
                else:
                    logs.append(f"Total resistance relationship satisfied: I = E/(r+R) within 2%")
            
            # 5. Check instantaneous emf at t=1/60 s: e₁ = E_m * sin(ω * (1/60))
            if self.E_m > 0 and self.omega > 0:
                t_val = 1/60.0
                expected_e1 = self.E_m * math.sin(self.omega * t_val)
                if not math.isclose(self.e1, expected_e1, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Instantaneous emf violation: e₁={self.e1:.3f}, E_m*sin(ω/60)={expected_e1:.3f}")
                else:
                    logs.append(f"Instantaneous emf satisfied: e₁ = E_m*sin(ω/60) within 2%")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify the consistency of the step-by-step solution.
        """
        logs = []
        try:
            # Step 1: ω = 2πn, n = 300 rpm = 5 rev/s
            expected_omega = 2 * math.pi * (300.0 / 60.0)  # 10π ≈ 31.4159 rad/s
            if not math.isclose(self.omega, expected_omega, rel_tol=0.01, abs_tol=0.01):
                logs.append(f"Angular velocity mismatch: ω={self.omega:.3f}, expected 10π≈31.416 rad/s")
            else:
                logs.append(f"Angular velocity consistent: ω ≈ 31.416 rad/s")
            
            # Step 2: E_m = N * B * A * ω
            # B = 1/π ≈ 0.3183 T, A = 0.05 m², N = 100, ω = 10π ≈ 31.416 rad/s
            expected_Em = 100 * (1.0/math.pi) * 0.05 * (10 * math.pi)
            if not math.isclose(self.E_m, expected_Em, rel_tol=0.01, abs_tol=0.01):
                logs.append(f"Maximum emf mismatch: E_m={self.E_m:.3f}, expected 50 V")
            else:
                logs.append(f"Maximum emf consistent: E_m ≈ 50 V")
            
            # Step 3: e = E_m * sin(ωt)
            # Check if the expression string contains the right parameters
            expected_expr_partial = "50"
            if expected_expr_partial not in self.e_expr:
                logs.append(f"Instantaneous emf expression mismatch: e='{self.e_expr}', expected to contain '50'")
            else:
                logs.append(f"Instantaneous emf expression reasonable")
            
            # Step 4: e₁ at t=1/60 s
            # ωt = (10π)*(1/60) = π/6, sin(π/6)=0.5
            expected_e1_val = 50 * 0.5
            if not math.isclose(self.e1, expected_e1_val, rel_tol=0.01, abs_tol=0.01):
                logs.append(f"Instantaneous emf at t=1/60 mismatch: e₁={self.e1:.3f}, expected 25 V")
            else:
                logs.append(f"Instantaneous emf at t=1/60 consistent: e₁ ≈ 25 V")
            
            # Step 5: Effective value E = E_m/√2
            expected_E_val = 50 / math.sqrt(2)
            if not math.isclose(self.E, expected_E_val, rel_tol=0.01, abs_tol=0.01):
                logs.append(f"Effective emf mismatch: E={self.E:.3f}, expected 35.355 V")
            else:
                logs.append(f"Effective emf consistent: E ≈ 35.355 V")
            
            # Step 6: Current I = E/(r+R)
            expected_I_val = expected_E_val / (1 + 9)
            if not math.isclose(self.I, expected_I_val, rel_tol=0.01, abs_tol=0.01):
                logs.append(f"Current mismatch: I={self.I:.3f}, expected 3.5355 A")
            else:
                logs.append(f"Current consistent: I ≈ 3.5355 A")
            
            # Step 7: Voltage U = I * R
            expected_U_val = expected_I_val * 9
            if not math.isclose(self.U, expected_U_val, rel_tol=0.01, abs_tol=0.01):
                logs.append(f"Voltage mismatch: U={self.U:.3f}, expected 31.8195 V")
            else:
                logs.append(f"Voltage consistent: U ≈ 31.8195 V")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter (e.g., magnetic field B) and verify that
        the output changes in the expected direction.
        """
        logs = []
        try:
            # Original values
            B_original = self.B
            # Perturb B by +10%
            B_perturbed = B_original * 1.1
            
            # Recalculate expected changes using formulas
            # E_m ∝ B, so should increase by 10%
            expected_Em_ratio = 1.1
            # E ∝ B, so should increase by 10%
            expected_E_ratio = 1.1
            # I ∝ B, so should increase by 10%
            expected_I_ratio = 1.1
            # U ∝ B, so should increase by 10%
            expected_U_ratio = 1.1
            
            # Compute perturbed values using the same formulas
            Em_perturbed = self.N * B_perturbed * self.A * self.omega
            E_perturbed = Em_perturbed / math.sqrt(2)
            I_perturbed = E_perturbed / (self.r + self.R)
            U_perturbed = I_perturbed * self.R
            
            # Check if ratios are as expected (within 1% tolerance)
            actual_Em_ratio = Em_perturbed / self.E_m if self.E_m != 0 else 0
            actual_E_ratio = E_perturbed / self.E if self.E != 0 else 0
            actual_I_ratio = I_perturbed / self.I if self.I != 0 else 0
            actual_U_ratio = U_perturbed / self.U if self.U != 0 else 0
            
            if not math.isclose(actual_Em_ratio, expected_Em_ratio, rel_tol=0.01):
                logs.append(f"Monotonicity violation for E_m: ratio={actual_Em_ratio:.3f}, expected {expected_Em_ratio:.3f}")
            else:
                logs.append(f"E_m monotonicity satisfied: increases with B")
            
            if not math.isclose(actual_E_ratio, expected_E_ratio, rel_tol=0.01):
                logs.append(f"Monotonicity violation for E: ratio={actual_E_ratio:.3f}, expected {expected_E_ratio:.3f}")
            else:
                logs.append(f"E monotonicity satisfied: increases with B")
            
            if not math.isclose(actual_I_ratio, expected_I_ratio, rel_tol=0.01):
                logs.append(f"Monotonicity violation for I: ratio={actual_I_ratio:.3f}, expected {expected_I_ratio:.3f}")
            else:
                logs.append(f"I monotonicity satisfied: increases with B")
            
            if not math.isclose(actual_U_ratio, expected_U_ratio, rel_tol=0.01):
                logs.append(f"Monotonicity violation for U: ratio={actual_U_ratio:.3f}, expected {expected_U_ratio:.3f}")
            else:
                logs.append(f"U monotonicity satisfied: increases with B")
            
            # Restore original B (not strictly needed as we didn't modify instance, but for clarity)
            self.B = B_original
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
        
        return logs
    
    def run_all_tests(self):
        """
        Run all physics checks and return (success_bool, logs_list).
        Success is determined by having no critical violations.
        """
        all_logs = []
        success = True
        
        # Run each test
        invariant_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        monotonicity_logs = self.test_monotonicity()
        
        all_logs.extend(invariant_logs)
        all_logs.extend(trace_logs)
        all_logs.extend(monotonicity_logs)
        
        # Determine success: no "violation" messages in logs
        violation_keywords = ['violation', 'mismatch', 'Error']
        for log in all_logs:
            if any(keyword in log.lower() for keyword in violation_keywords):
                success = False
                break
        
        return success, all_logs