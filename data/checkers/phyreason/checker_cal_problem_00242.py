import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        try:
            # Extract inputs with safe fallbacks
            self.v = getattr(solution_instance, 'v', 9.6e4)
            self.B = getattr(solution_instance, 'B', 0.1)
            self.qm_M = getattr(solution_instance, 'qm_M', 4.8e6)  # specific charge for ion M
            
            # Extract results with safe fallbacks
            results = getattr(solution_instance, 'results', {})
            self.r = results.get('r', 0.0)               # radius for ion M
            self.OM = results.get('OM', 0.0)             # distance OM
            self.qm_N = results.get('fracqprimemprime', 0.0)  # specific charge for ion N
            
            # Store original values for monotonicity test
            self._original_v = self.v
            self._original_B = self.B
            self._original_qm_M = self.qm_M
            
            # Constants for geometric factor (from steps analysis)
            self.r_ratio = 1.1  # r'/r from step_3: r' = 1.1r
            
        except Exception as e:
            print(f"Warning: Error in initialization: {e}")
            # Set defaults if extraction fails
            self.v = 9.6e4
            self.B = 0.1
            self.qm_M = 4.8e6
            self.r = 0.0
            self.OM = 0.0
            self.qm_N = 0.0
            self._original_v = self.v
            self._original_B = self.B
            self._original_qm_M = self.qm_M
            self.r_ratio = 1.1

    def test_invariants(self):
        """
        Verify Lorentz force physics invariants using symbolic and numerical checks.
        """
        logs = []
        success = True
        
        try:
            # Define symbolic variables
            v_sym, B_sym, qm_sym, r_sym = sp.symbols('v B qm r', positive=True)
            
            # Symbolic Lorentz force as centripetal force: q*v*B = m*v^2/r => r = v/(qm*B)
            r_formula = v_sym / (qm_sym * B_sym)
            
            # Check formula consistency numerically for ion M
            r_calc = self.v / (self.qm_M * self.B)
            if not math.isclose(self.r, r_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Radius r mismatch. Expected ~{r_calc:.4f}, got {self.r}")
                success = False
            else:
                logs.append(f"PASS: Radius r = {self.r} consistent with Lorentz force.")
            
            # Check that OM = 2r (semicircle geometry)
            if not math.isclose(self.OM, 2*self.r, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: OM should be 2r = {2*self.r:.4f}, but got {self.OM}")
                success = False
            else:
                logs.append(f"PASS: OM = {self.OM} = 2r consistent.")
            
            # Check specific charge for ion N: q'/m' = v/(r'*B) with r' = 1.1r
            r_prime = self.r_ratio * self.r
            qm_N_calc = self.v / (r_prime * self.B)
            if not math.isclose(self.qm_N, qm_N_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: qm_N mismatch. Expected ~{qm_N_calc:.2e}, got {self.qm_N:.2e}")
                success = False
            else:
                logs.append(f"PASS: qm_N = {self.qm_N:.2e} consistent with Lorentz force and geometry.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        try:
            # Target answers from problem statement
            target_OM = 0.4  # m
            target_qm_N = 4.4e6  # C/kg
            
            # Check OM
            if not math.isclose(self.OM, target_OM, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: OM mismatch with target. Target {target_OM}, got {self.OM}")
                success = False
            else:
                logs.append(f"PASS: OM matches target {target_OM} m.")
            
            # Check specific charge for ion N
            if not math.isclose(self.qm_N, target_qm_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: qm_N mismatch with target. Target {target_qm_N:.2e}, got {self.qm_N:.2e}")
                success = False
            else:
                logs.append(f"PASS: qm_N matches target {target_qm_N:.2e} C/kg.")
            
            # Verify internal consistency: r should be target_OM/2
            if not math.isclose(self.r, target_OM/2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARNING: r = {self.r} not consistent with OM/2 = {target_OM/2}")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input parameter and verify the trend of results.
        """
        logs = []
        success = True
        
        try:
            # Perturb magnetic field B by +10%
            B_perturbed = self.B * 1.1
            
            # Recalculate expected results using formulas
            r_perturbed = self.v / (self.qm_M * B_perturbed)
            OM_perturbed = 2 * r_perturbed
            qm_N_perturbed = self.v / (self.r_ratio * r_perturbed * B_perturbed)
            
            # Check trends: if B increases, radius decreases
            if not (r_perturbed < self.r):
                logs.append(f"FAIL: Radius r should decrease when B increases.")
                success = False
            else:
                logs.append(f"PASS: r decreases from {self.r:.4f} to {r_perturbed:.4f} when B increases.")
            
            # OM should also decrease
            if not (OM_perturbed < self.OM):
                logs.append(f"FAIL: OM should decrease when B increases.")
                success = False
            else:
                logs.append(f"PASS: OM decreases from {self.OM:.4f} to {OM_perturbed:.4f}.")
            
            # qm_N should increase? Let's analyze:
            # qm_N = v/(1.1*r*B) and r = v/(qm_M*B) => qm_N = v/(1.1*(v/(qm_M*B))*B) = qm_M/1.1
            # So qm_N is independent of B! Actually, from the derivation:
            # qm_N = v/(1.1*r*B) and r = v/(qm_M*B) => qm_N = qm_M/1.1
            # Therefore, qm_N should NOT change with B. Check this:
            if not math.isclose(qm_N_perturbed, self.qm_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: qm_N should be invariant under B change, but changed.")
                success = False
            else:
                logs.append(f"PASS: qm_N invariant under B change (remains ~{self.qm_N:.2e}).")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run invariants test
        success, logs = self.test_invariants()
        all_logs.extend(["=== Invariants Test ==="] + logs)
        overall_success = overall_success and success
        
        # Run trace consistency test
        success, logs = self.test_trace_consistency()
        all_logs.extend(["=== Trace Consistency Test ==="] + logs)
        overall_success = overall_success and success
        
        # Run monotonicity test
        success, logs = self.test_monotonicity()
        all_logs.extend(["=== Monotonicity Test ==="] + logs)
        overall_success = overall_success and success
        
        return overall_success, all_logs