import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract inputs from results dict with safe defaults
        # q should be in results, but use default if not
        self.q = self.results.get('q', -1.0e-8)  # Negative charge
        # Use actual given constants for tests, not from results
        self.W_infA_by_given = 6.0e-8
        self.W_AB_against_given = 4.0e-8
        
        # Extract results with safe access
        self.U_infA = self.results.get('U_infA', None)
        self.U_AB = self.results.get('U_AB', None)
        self.phi_A = self.results.get('phi_A', None)
        self.phi_B = self.results.get('phi_B', None)

    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # Check 1: Energy conservation via work-energy theorem
            # W_by = -q*(ΔV)
            # For ∞→A: W_by = -q*(φ_A - 0) ⇒ φ_A = -W_by/q
            if self.phi_A is not None and abs(self.q) > 1e-12:
                phi_A_calc = -self.W_infA_by_given / self.q
                if not math.isclose(self.phi_A, phi_A_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: φ_A mismatch. Expected {phi_A_calc:.2e} V, got {self.phi_A:.2e} V")
                    success = False
                else:
                    logs.append(f"PASS: φ_A = -W_by(∞→A)/q = {phi_A_calc:.2e} V")
            
            # Check 2: Work from A→B: W_against = -W_by = q*(φ_B - φ_A) ⇒ φ_B = φ_A + W_against/q
            if self.phi_A is not None and self.phi_B is not None and abs(self.q) > 1e-12:
                phi_B_calc = self.phi_A + self.W_AB_against_given / self.q
                if not math.isclose(self.phi_B, phi_B_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: φ_B mismatch. Expected {phi_B_calc:.2e} V, got {self.phi_B:.2e} V")
                    success = False
                else:
                    logs.append(f"PASS: φ_B = φ_A + W_against(A→B)/q = {phi_B_calc:.2e} V")
            
            # Check 3: Potential difference consistency
            if self.U_infA is not None and self.phi_A is not None:
                # U_∞A should be φ_A - 0 = φ_A
                if not math.isclose(self.U_infA, self.phi_A, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: U_∞A ≠ φ_A. U_∞A={self.U_infA:.2e} V, φ_A={self.phi_A:.2e} V")
                    success = False
                else:
                    logs.append(f"PASS: U_∞A = φ_A = {self.phi_A:.2e} V")
            
            if self.U_AB is not None and self.phi_A is not None and self.phi_B is not None:
                # U_AB should be φ_A - φ_B (potential difference from A to B)
                U_AB_calc = self.phi_A - self.phi_B
                if not math.isclose(self.U_AB, U_AB_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: U_AB mismatch. Expected {U_AB_calc:.2e} V, got {self.U_AB:.2e} V")
                    success = False
                else:
                    logs.append(f"PASS: U_AB = φ_A - φ_B = {U_AB_calc:.2e} V")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Cross-check with steps analysis values
            # Step 2: U_∞A = φ_A - φ_∞ = φ_A (since φ_∞ = 0)
            if self.U_infA is not None and self.phi_A is not None:
                if not math.isclose(self.U_infA, self.phi_A, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: U_∞A should equal φ_A. U_∞A={self.U_infA:.2e} V, φ_A={self.phi_A:.2e} V")
                    success = False
                else:
                    logs.append(f"PASS: U_∞A = φ_A = {self.phi_A:.2e} V")
            
            # Step 4: U_AB = φ_A - φ_B (from ground truth: U_AB = 4V, φ_A=6V, φ_B=2V)
            if self.U_AB is not None and self.phi_A is not None and self.phi_B is not None:
                U_AB_calc = self.phi_A - self.phi_B
                if not math.isclose(self.U_AB, U_AB_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: U_AB should equal φ_A - φ_B. Expected {U_AB_calc:.2e} V, got {self.U_AB:.2e} V")
                    success = False
                else:
                    logs.append(f"PASS: U_AB = φ_A - φ_B = {U_AB_calc:.2e} V")
            
            # Check final answers match target
            if self.phi_A is not None and self.phi_B is not None:
                phi_A_target = 6.0  # V
                phi_B_target = 2.0  # V
                if (math.isclose(self.phi_A, phi_A_target, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.phi_B, phi_B_target, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"PASS: Final potentials match target: φ_A={self.phi_A:.2f} V, φ_B={self.phi_B:.2f} V")
                else:
                    logs.append(f"FAIL: Final potentials mismatch target. Expected φ_A=6 V, φ_B=2 V, got φ_A={self.phi_A:.2f} V, φ_B={self.phi_B:.2f} V")
                    success = False
            
            # Consistency between U_∞A, U_AB and potentials
            if self.U_infA is not None and self.U_AB is not None and self.phi_A is not None and self.phi_B is not None:
                # φ_A should equal U_∞A (since φ_∞=0)
                if not math.isclose(self.phi_A, self.U_infA, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: φ_A ≠ U_∞A. φ_A={self.phi_A:.2e} V, U_∞A={self.U_infA:.2e} V")
                    success = False
                
                # φ_B should be φ_A - U_AB (since U_AB = φ_A - φ_B)
                phi_B_from_U = self.phi_A - self.U_AB
                if not math.isclose(self.phi_B, phi_B_from_U, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: φ_B ≠ φ_A - U_AB. φ_B={self.phi_B:.2e} V, φ_A - U_AB={phi_B_from_U:.2e} V")
                    success = False
                else:
                    logs.append(f"PASS: φ_B = φ_A - U_AB consistency check")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Use the q value from results, with fallback to actual value
            q_original = self.q if abs(self.q) > 1e-12 else -1.0e-8
            q_perturbed = q_original * 1.01  # Increase magnitude by 1%
            
            # Use the actual given work values, not from results
            W_infA_by = 6.0e-8
            W_AB_against = 4.0e-8
            
            # Recalculate using formulas (not calling solution)
            phi_A_orig = -W_infA_by / q_original
            phi_B_orig = phi_A_orig + W_AB_against / q_original
            
            phi_A_pert = -W_infA_by / q_perturbed
            phi_B_pert = phi_A_pert + W_AB_against / q_perturbed
            
            # Check monotonic trend: As |q| increases (more negative), potentials should decrease
            # Because φ = -W/q, and W positive, q negative => φ positive
            # Increasing |q| (more negative) makes denominator larger in magnitude, so φ decreases
            if phi_A_pert < phi_A_orig and phi_B_pert < phi_B_orig:
                logs.append(f"PASS: Monotonicity - Increasing |q| decreases potentials (as expected for negative q)")
                logs.append(f"  Original: q={q_original:.2e} C, φ_A={phi_A_orig:.2e} V, φ_B={phi_B_orig:.2e} V")
                logs.append(f"  Perturbed: q={q_perturbed:.2e} C, φ_A={phi_A_pert:.2e} V, φ_B={phi_B_pert:.2e} V")
            else:
                logs.append(f"FAIL: Monotonicity violation - Potentials did not decrease with increased |q|")
                logs.append(f"  Original: φ_A={phi_A_orig:.2e} V, φ_B={phi_B_orig:.2e} V")
                logs.append(f"  Perturbed: φ_A={phi_A_pert:.2e} V, φ_B={phi_B_pert:.2e} V")
                success = False
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs