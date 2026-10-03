import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        try:
            # Extract results with safe access
            self.results = getattr(solution_instance, 'results', {})
            
            # Extract inputs from results or use defaults
            self.m = self.results.get('m', 4.0)
            self.F = self.results.get('F', 26.0)
            self.g = self.results.get('g', 10.0)  # Use 10.0 as default to match ground truth
            
            # Get theta in degrees (default 37)
            self.theta_deg = self.results.get('theta_deg', 37.0)
            
            # Extract required output variables
            self.F_N = self.results.get('F_N', 0.0)
            self.F_f = self.results.get('F_f', 0.0)
            
            # Calculate expected values using the SAME approximations as the solution
            # Use common approximations: cos37° = 0.8, sin37° = 0.6
            cos37 = 0.8
            sin37 = 0.6
            
            self.F_N_expected = self.m * self.g * cos37 + self.F * sin37
            self.F_f_expected = self.m * self.g * sin37 - self.F * cos37
            
        except Exception as e:
            self.error = f"Initialization error: {str(e)}"
            raise

    def test_invariants(self):
        """
        Verify that the object is in equilibrium:
        1. Net force perpendicular to the plane is zero -> F_N matches expected.
        2. Net force along the plane is zero -> F_f matches expected and is positive (upward).
        """
        logs = []
        success = True
        
        # 1. Check normal force
        if not math.isclose(self.F_N, self.F_N_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Normal force mismatch. Expected ~{self.F_N_expected:.2f}, got {self.F_N:.2f}")
            success = False
        else:
            logs.append(f"PASS: Normal force matches equilibrium condition (F_N = {self.F_N:.2f} N)")
        
        # 2. Check frictional force
        if not math.isclose(self.F_f, self.F_f_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Frictional force mismatch. Expected ~{self.F_f_expected:.2f}, got {self.F_f:.2f}")
            success = False
        else:
            logs.append(f"PASS: Frictional force matches equilibrium condition (F_f = {self.F_f:.2f} N)")
            
        # Check direction
        if self.F_f < -1e-2:
            logs.append(f"WARNING: Frictional force is negative ({self.F_f:.2f} N), but expected positive (upward along incline)")
        
        # 3. Check net force residuals
        perp_residual = self.F_N - self.F_N_expected
        if not math.isclose(perp_residual, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Net force perpendicular to plane is not zero (residual = {perp_residual:.2e} N)")
            success = False
        else:
            logs.append("PASS: Net force perpendicular to plane is zero within tolerance")
            
        along_residual = self.F_f - self.F_f_expected
        if not math.isclose(along_residual, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Net force along plane is not zero (residual = {along_residual:.2e} N)")
            success = False
        else:
            logs.append("PASS: Net force along plane is zero within tolerance")
        
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the steps analysis and target answers.
        """
        logs = []
        success = True
        
        # Step 1: Normal force = 47.6 N
        expected_F_N_step = 47.6
        if not math.isclose(self.F_N, expected_F_N_step, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Normal force {self.F_N:.2f} N does not match step value {expected_F_N_step} N")
            success = False
        else:
            logs.append(f"PASS: Normal force matches step value ({self.F_N:.2f} N)")
        
        # Step 2: Frictional force = 3.2 N, positive (upward)
        expected_F_f_step = 3.2
        if not math.isclose(self.F_f, expected_F_f_step, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Frictional force {self.F_f:.2f} N does not match step value {expected_F_f_step} N")
            success = False
        else:
            logs.append(f"PASS: Frictional force matches step value ({self.F_f:.2f} N)")
            if self.F_f > 0:
                logs.append("PASS: Frictional force direction is upward along the incline (positive)")
            else:
                logs.append("WARNING: Frictional force is not positive as expected")
        
        # Check against target answers
        target_F_N = 47.6
        target_F_f = 3.2
        if not (math.isclose(self.F_N, target_F_N, rel_tol=1e-2, abs_tol=1e-2) and 
                math.isclose(self.F_f, target_F_f, rel_tol=1e-2, abs_tol=1e-2)):
            logs.append(f"FAIL: Results do not match target answers: F_N={target_F_N} N, F_f={target_F_f} N")
            success = False
        else:
            logs.append("PASS: Results match both target answers within 1% tolerance")
        
        return success, logs

    def test_monotonicity(self):
        """
        Perturb mass m by +10%, recalculate F_N and F_f using formulas, check trends, then restore.
        """
        logs = []
        success = True
        
        original_m = self.m
        perturbed_m = original_m * 1.10  # +10%
        
        try:
            # Use the same approximations as the solution
            cos37 = 0.8
            sin37 = 0.6
            
            # Recalculate using the formulas with same approximations
            F_N_orig = self.F_N_expected
            F_f_orig = self.F_f_expected
            
            F_N_pert = perturbed_m * self.g * cos37 + self.F * sin37
            F_f_pert = perturbed_m * self.g * sin37 - self.F * cos37
            
            # Check monotonicity: F_N should increase with m
            if F_N_pert <= F_N_orig:
                logs.append(f"FAIL: Normal force did not increase when mass increased. Original: {F_N_orig:.2f} N, Perturbed: {F_N_pert:.2f} N")
                success = False
            else:
                logs.append(f"PASS: Normal force increases with mass (from {F_N_orig:.2f} N to {F_N_pert:.2f} N)")
            
            # F_f should increase with m (since m*g*sinθ term increases)
            if F_f_pert <= F_f_orig:
                logs.append(f"FAIL: Frictional force did not increase when mass increased. Original: {F_f_orig:.2f} N, Perturbed: {F_f_pert:.2f} N")
                success = False
            else:
                logs.append(f"PASS: Frictional force increases with mass (from {F_f_orig:.2f} N to {F_f_pert:.2f} N)")
        
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            success = False
        
        return success, logs

    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"--- {test_name} ---")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRASH: {test_name} failed with exception: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\nOverall result: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs