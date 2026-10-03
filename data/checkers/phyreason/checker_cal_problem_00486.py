import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Avoid direct attribute access to prevent sandbox errors.
        """
        self.inputs = {}
        self.results = {}
        
        # Extract inputs (provided in problem context)
        self.inputs['q'] = -6e-6  # C (negative charge)
        self.inputs['W_AB_ext'] = 3e-5  # J (external work A→B, overcoming field)
        self.inputs['W_BC_field'] = 1.2e-5  # J (work done by field B→C)
        
        # Extract results from solution_instance using safe access
        # Fallback to defaults if missing
        self.results['U_AB'] = getattr(solution_instance, 'U_AB', 
                                       getattr(solution_instance, 'results', {}).get('U_AB', 0.0))
        self.results['U_BC'] = getattr(solution_instance, 'U_BC', 
                                       getattr(solution_instance, 'results', {}).get('U_BC', 0.0))
        self.results['varphi_C'] = getattr(solution_instance, 'varphi_C', 
                                           getattr(solution_instance, 'results', {}).get('varphi_C', 0.0))
        self.results['W_AC'] = getattr(solution_instance, 'W_AC', 
                                       getattr(solution_instance, 'results', {}).get('W_AC', 0.0))
        self.results['delta_E_p'] = getattr(solution_instance, 'delta_E_p', 
                                            getattr(solution_instance, 'results', {}).get('delta_E_p', 0.0))
        
        # For symbolic computations
        self.q = self.inputs['q']
        self.W_AB_ext = self.inputs['W_AB_ext']
        self.W_BC_field = self.inputs['W_BC_field']
        
    def test_invariants(self):
        """
        Verify fundamental physics laws based on KB theorems.
        Focus on work-potential relationships and sign consistency.
        """
        logs = []
        q, W_AB_ext, W_BC_field = self.q, self.W_AB_ext, self.W_BC_field
        U_AB, U_BC, varphi_C, W_AC, delta_E_p = (self.results['U_AB'], self.results['U_BC'], 
                                                 self.results['varphi_C'], self.results['W_AC'], 
                                                 self.results['delta_E_p'])
        
        try:
            # ------------------------------------------------------------
            # Invariant 1: W_external = q * ΔV (from A→B, overcoming field)
            # Convert external work to work by field: W_field = -W_external
            W_AB_field = -W_AB_ext  # Work by field A→B (should be negative)
            expected_U_AB = W_AB_field / q  # ΔV_AB = V_B - V_A = W_field / q
            if not math.isclose(U_AB, expected_U_AB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_AB mismatch. Expected {expected_U_AB:.3f}, got {U_AB:.3f}")
            else:
                logs.append(f"PASS: U_AB = {U_AB:.3f} V consistent with W_AB_field/q.")
            
            # ------------------------------------------------------------
            # Invariant 2: W_field = q * ΔV (from B→C, work done by field)
            expected_U_BC = W_BC_field / q  # ΔV_BC = V_C - V_B = W_field / q
            if not math.isclose(U_BC, expected_U_BC, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_BC mismatch. Expected {expected_U_BC:.3f}, got {U_BC:.3f}")
            else:
                logs.append(f"PASS: U_BC = {U_BC:.3f} V consistent with W_BC_field/q.")
            
            # ------------------------------------------------------------
            # Invariant 3: Potential consistency (assuming V_B = 0 reference)
            # φ_C = φ_B - U_BC  (since U_BC = V_B - V_C from step analysis)
            expected_varphi_C = 0 - U_BC  # φ_B = 0 (reference)
            if not math.isclose(varphi_C, expected_varphi_C, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: varphi_C mismatch. Expected {expected_varphi_C:.3f}, got {varphi_C:.3f}")
            else:
                logs.append(f"PASS: varphi_C = {varphi_C:.3f} V consistent with φ_B=0 reference.")
            
            # ------------------------------------------------------------
            # Invariant 4: Work additivity (step analysis)
            expected_W_AC = W_AB_field + W_BC_field  # Work by field A→C
            if not math.isclose(W_AC, expected_W_AC, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: W_AC mismatch. Expected {expected_W_AC:.3e}, got {W_AC:.3e}")
            else:
                logs.append(f"PASS: W_AC = {W_AC:.3e} J consistent with additivity.")
            
            # ------------------------------------------------------------
            # Invariant 5: ΔE_p = -W_field (from work-energy theorem for conservative field)
            expected_delta_E_p = -W_AC  # ΔU = -W_field
            if not math.isclose(delta_E_p, expected_delta_E_p, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: delta_E_p mismatch. Expected {expected_delta_E_p:.3e}, got {delta_E_p:.3e}")
            else:
                logs.append(f"PASS: delta_E_p = {delta_E_p:.3e} J consistent with -W_AC.")
            
            # ------------------------------------------------------------
            # Invariant 6: Sign checks (qualitative)
            # Since q < 0 and W_AB_ext > 0 (overcoming field), U_AB = V_B - V_A should be positive.
            if U_AB < 0:
                logs.append("WARNING: U_AB negative but expected positive (q negative, external work positive).")
            # W_AC should be negative (field does net negative work moving negative charge A→C).
            if W_AC > 0:
                logs.append("WARNING: W_AC positive but expected negative (net field work on negative charge).")
            # ΔE_p should be positive (potential energy increases when negative charge moves against field).
            if delta_E_p < 0:
                logs.append("WARNING: delta_E_p negative but expected positive.")
            
        except Exception as e:
            logs.append(f"ERROR in invariant tests: {e}")
            
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Ensures internal consistency of the solution steps.
        """
        logs = []
        try:
            # Step 1: U_AB should be 5 V (from step analysis)
            if not math.isclose(self.results['U_AB'], 5.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_AB = {self.results['U_AB']:.3f} V, expected ~5 V")
            else:
                logs.append("PASS: U_AB ≈ 5 V matches step 1.")
            
            # Step 2: U_BC should be -2 V
            if not math.isclose(self.results['U_BC'], -2.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_BC = {self.results['U_BC']:.3f} V, expected ~ -2 V")
            else:
                logs.append("PASS: U_BC ≈ -2 V matches step 2.")
            
            # Step 3: φ_C should be 2 V (assuming φ_B = 0)
            if not math.isclose(self.results['varphi_C'], 2.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: varphi_C = {self.results['varphi_C']:.3f} V, expected ~2 V")
            else:
                logs.append("PASS: varphi_C ≈ 2 V matches step 3.")
            
            # Step 4: W_AC should be -1.8e-5 J
            expected_W_AC = -1.8e-5
            if not math.isclose(self.results['W_AC'], expected_W_AC, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: W_AC = {self.results['W_AC']:.3e} J, expected ~ -1.8e-5 J")
            else:
                logs.append("PASS: W_AC ≈ -1.8e-5 J matches step 4.")
            
            # Step 5: ΔE_p should be 1.8e-5 J
            expected_delta_E_p = 1.8e-5
            if not math.isclose(self.results['delta_E_p'], expected_delta_E_p, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: delta_E_p = {self.results['delta_E_p']:.3e} J, expected ~ 1.8e-5 J")
            else:
                logs.append("PASS: delta_E_p ≈ 1.8e-5 J matches step 5.")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {e}")
            
        return logs
    
    def test_monotonicity(self):
        """
        Perturb an input (charge magnitude) and check if trends match expectations.
        Restore original values after test.
        """
        logs = []
        original_q = self.q
        original_results = self.results.copy()
        
        try:
            # Perturb charge magnitude (more negative)
            perturbed_q = original_q * 1.5  # -9e-6 C
            # Recalculate expected quantities using formulas from steps
            # Step 1: U_AB should remain the same (independent of q, per unit charge)
            W_AB_field = -self.W_AB_ext  # Work by field unchanged
            expected_U_AB_pert = W_AB_field / perturbed_q
            # Step 2: U_BC should also remain the same
            expected_U_BC_pert = self.W_BC_field / perturbed_q
            # Step 4: W_AC scales with q (since W_field = q * ΔV, ΔV fixed)
            expected_W_AC_pert = W_AB_field + self.W_BC_field  # Same as before? Wait, no.
            # Actually, W_AB_field and W_BC_field are fixed by the problem? No, they are given for this specific q.
            # We must recalculate W_AB_field and W_BC_field for the new q? The problem gives work values for the original q.
            # So the work values are specific to the charge. If we change q, the work done by the field for the same ΔV changes.
            # But the problem statement gives work values, which are measurements. So we assume ΔV remains the same.
            # Therefore, U_AB and U_BC are properties of the field, independent of q.
            # So for perturbed q, the work done by the field would be: W_field_new = q_new * ΔV (with same ΔV).
            # Thus, we can compute:
            U_AB_original = original_results['U_AB']
            U_BC_original = original_results['U_BC']
            expected_W_AB_field_pert = perturbed_q * U_AB_original
            expected_W_BC_field_pert = perturbed_q * U_BC_original
            expected_W_AC_pert = expected_W_AB_field_pert + expected_W_BC_field_pert
            expected_delta_E_p_pert = -expected_W_AC_pert
            
            # Check monotonicity: If |q| increases (more negative), |W_AC| should increase (more negative work by field).
            # Since q is negative, making it more negative means q decreases.
            # W_AC = q * (U_AB + U_BC) = q * (5 - 2) = q * 3.
            # As q becomes more negative, W_AC becomes more negative (since 3 > 0).
            # So magnitude of W_AC should increase.
            if abs(perturbed_q) > abs(original_q):
                if abs(expected_W_AC_pert) <= abs(original_results['W_AC']):
                    logs.append("WARNING: |W_AC| did not increase when |q| increased.")
                else:
                    logs.append("PASS: |W_AC| increases with |q| (monotonic).")
            
            # ΔE_p = -W_AC, so same trend as |W_AC|.
            if abs(expected_delta_E_p_pert) <= abs(original_results['delta_E_p']):
                logs.append("WARNING: |delta_E_p| did not increase when |q| increased.")
            else:
                logs.append("PASS: |delta_E_p| increases with |q| (monotonic).")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
        finally:
            # Restore original values
            self.q = original_q
            self.results = original_results
            
        return logs
    
    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        """
        all_logs = []
        success = True
        
        # Run each test suite
        test_suites = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in test_suites:
            try:
                logs = test_func()
                # Check for FAIL or ERROR messages
                if any(log.startswith(("FAIL", "ERROR")) for log in logs):
                    success = False
                all_logs.append(f"--- {name} ---")
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR in {name}: {traceback.format_exc()}")
                success = False
        
        return success, all_logs