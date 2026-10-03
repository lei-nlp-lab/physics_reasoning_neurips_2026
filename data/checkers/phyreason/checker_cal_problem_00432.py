import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary data from the solution instance.
        """
        self.solution = solution_instance
        # Safely access results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract expected keys with safe defaults
        self.k = getattr(solution_instance, 'k', self.results.get('k', 0.0))
        self.l_0 = getattr(solution_instance, 'l_0', self.results.get('l_0', 0.0))
        self.F_1 = getattr(solution_instance, 'F_1', self.results.get('F_1', 0.0))
        self.F_2 = getattr(solution_instance, 'F_2', self.results.get('F_2', 0.0))
        
        # Problem constants
        self.l_1 = 0.09  # 9 cm in meters
        self.l_2 = 0.12  # 12 cm in meters
        
        # Store logs for reporting
        self.logs = []
        
    def log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def test_invariants(self):
        """Test fundamental physics laws: Hooke's Law for both cases."""
        self.log("Testing Hooke's Law invariants...")
        successes = []
        
        # Numerical check for compression case
        delta_x1 = self.l_0 - self.l_1
        F1_calc = self.k * delta_x1
        if math.isclose(F1_calc, self.F_1, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Compression Hooke's Law: {self.F_1} ≈ {self.k} * ({self.l_0} - {self.l_1}) = {F1_calc}", True)
            successes.append(True)
        else:
            self.log(f"Compression Hooke's Law FAILED: {self.F_1} ≠ {self.k} * ({self.l_0} - {self.l_1}) = {F1_calc}", False)
            successes.append(False)
        
        # Numerical check for tension case
        delta_x2 = self.l_2 - self.l_0
        F2_calc = self.k * delta_x2
        if math.isclose(F2_calc, self.F_2, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Tension Hooke's Law: {self.F_2} ≈ {self.k} * ({self.l_2} - {self.l_0}) = {F2_calc}", True)
            successes.append(True)
        else:
            self.log(f"Tension Hooke's Law FAILED: {self.F_2} ≠ {self.k} * ({self.l_2} - {self.l_0}) = {F2_calc}", False)
            successes.append(False)
        
        # Consistency between two equations (should give same k and l0)
        # Solve the system numerically using the physics equations
        # From F1 = k*(l0 - l1) and F2 = k*(l2 - l0)
        # We can solve for l0: l0 = (F1*l2 + F2*l1)/(F1 + F2)
        l0_calc = (self.F_1 * self.l_2 + self.F_2 * self.l_1) / (self.F_1 + self.F_2)
        k_calc = self.F_1 / (l0_calc - self.l_1)
        
        if (math.isclose(k_calc, self.k, rel_tol=1e-2, abs_tol=1e-2) and
            math.isclose(l0_calc, self.l_0, rel_tol=1e-2, abs_tol=1e-2)):
            self.log(f"Simultaneous equations yield consistent k={k_calc:.2f}, l0={l0_calc:.3f}", True)
            successes.append(True)
        else:
            self.log(f"Simultaneous equations yield k={k_calc:.2f}, l0={l0_calc:.3f} but expected k={self.k:.2f}, l0={self.l_0:.3f}", False)
            successes.append(False)
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Check that intermediate steps from KB match final results."""
        self.log("Testing trace consistency with KB steps...")
        successes = []
        
        # Step 1: Compression equation F1 = k*(l0 - l1)
        delta_x1 = self.l_0 - self.l_1
        F1_step = self.k * delta_x1
        if math.isclose(F1_step, 10.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Step 1 (compression): 10 ≈ {self.k} * ({self.l_0} - {self.l_1}) = {F1_step}", True)
            successes.append(True)
        else:
            self.log(f"Step 1 FAILED: 10 ≠ {self.k} * ({self.l_0} - {self.l_1}) = {F1_step}", False)
            successes.append(False)
        
        # Step 3: Tension equation F2 = k*(l2 - l0)
        delta_x2 = self.l_2 - self.l_0
        F2_step = self.k * delta_x2
        if math.isclose(F2_step, 20.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Step 3 (tension): 20 ≈ {self.k} * ({self.l_2} - {self.l_0}) = {F2_step}", True)
            successes.append(True)
        else:
            self.log(f"Step 3 FAILED: 20 ≠ {self.k} * ({self.l_2} - {self.l_0}) = {F2_step}", False)
            successes.append(False)
        
        # Step 5: Natural length (check both meter and cm representations)
        l0_m = self.l_0
        l0_cm = self.l_0 * 100
        if math.isclose(l0_m, 0.1, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Step 5 (l0 in meters): {l0_m:.3f} ≈ 0.1 m", True)
            successes.append(True)
        else:
            self.log(f"Step 5 FAILED: {l0_m:.3f} ≠ 0.1 m", False)
            successes.append(False)
        
        if math.isclose(l0_cm, 10.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Step 5 (l0 in cm): {l0_cm:.1f} ≈ 10 cm", True)
            successes.append(True)
        else:
            self.log(f"Step 5 FAILED: {l0_cm:.1f} ≠ 10 cm", False)
            successes.append(False)
        
        # Step 6: Spring constant
        if math.isclose(self.k, 1000.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Step 6 (spring constant): {self.k:.1f} ≈ 1000 N/m", True)
            successes.append(True)
        else:
            self.log(f"Step 6 FAILED: {self.k:.1f} ≠ 1000 N/m", False)
            successes.append(False)
        
        return all(successes)
    
    def test_monotonicity(self):
        """
        Perturb input force F1, recompute k and l0 using physics formulas,
        check expected trends, then restore original values.
        """
        self.log("Testing monotonicity with perturbed F1...")
        successes = []
        
        # Store original state
        orig_F1 = self.F_1
        orig_k = self.k
        orig_l0 = self.l_0
        
        # Perturbation: increase F1 by 5%
        perturbed_F1 = orig_F1 * 1.05
        
        # Solve for k and l0 using simultaneous equations (from physics)
        # Equations: 
        #   F1 = k*(l0 - l1)  --> k = F1/(l0 - l1)
        #   F2 = k*(l2 - l0)  --> k = F2/(l2 - l0)
        # Equate: F1/(l0 - l1) = F2/(l2 - l0)
        # Solve for l0: F1*(l2 - l0) = F2*(l0 - l1)
        #              F1*l2 - F1*l0 = F2*l0 - F2*l1
        #              F1*l2 + F2*l1 = l0*(F1 + F2)
        #              l0 = (F1*l2 + F2*l1)/(F1 + F2)
        
        # Calculate perturbed l0
        perturbed_l0 = (perturbed_F1 * self.l_2 + self.F_2 * self.l_1) / (perturbed_F1 + self.F_2)
        
        # Calculate perturbed k using either equation
        perturbed_k = perturbed_F1 / (perturbed_l0 - self.l_1)
        
        # Expected trends (from KB Theorem):
        # 1. If |F1| increases (compression force), spring constant k should increase 
        #    (stiffer spring to achieve same compression with more force)
        # 2. Natural length l0 should shift slightly
        
        # Check trend 1: k increases with increased compression force
        if perturbed_k > orig_k * 0.99:  # Allow 1% tolerance for numerical noise
            self.log(f"Trend: k increases from {orig_k:.1f} to {perturbed_k:.1f} when F1 increases", True)
            successes.append(True)
        else:
            self.log(f"Trend FAILED: k decreased from {orig_k:.1f} to {perturbed_k:.1f}", False)
            successes.append(False)
        
        # Check trend 2: l0 changes predictably
        # From formula: l0 = (F1*l2 + F2*l1)/(F1+F2)
        # Derivative dl0/dF1 = (l2*(F1+F2) - (F1*l2+F2*l1))/(F1+F2)^2
        #                   = (F2*(l2-l1))/(F1+F2)^2 > 0 since l2>l1
        # So l0 should increase with F1
        if perturbed_l0 > orig_l0 * 0.999:  # Very small expected change
            self.log(f"Trend: l0 increases from {orig_l0:.3f} to {perturbed_l0:.3f} when F1 increases", True)
            successes.append(True)
        else:
            self.log(f"Trend: l0 changed from {orig_l0:.3f} to {perturbed_l0:.3f}", False)
            successes.append(False)
        
        # Verify perturbed values still satisfy Hooke's Law
        delta_x1_pert = perturbed_l0 - self.l_1
        F1_pert_check = perturbed_k * delta_x1_pert
        if math.isclose(F1_pert_check, perturbed_F1, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Perturbed system satisfies Hooke's Law (compression)", True)
            successes.append(True)
        else:
            self.log(f"Perturbed Hooke's Law FAILED: {perturbed_F1} ≠ {perturbed_k} * ({perturbed_l0} - {self.l_1}) = {F1_pert_check}", False)
            successes.append(False)
        
        # Restore original values
        self.F_1 = orig_F1
        self.k = orig_k
        self.l_0 = orig_l0
        
        return all(successes)
    
    def run_all_tests(self):
        """Execute all tests and return overall success with logs."""
        self.logs = []
        overall_success = True
        
        try:
            self.log("=== Physics Checker Starting ===")
            
            # Test 1: Invariants
            if not self.test_invariants():
                overall_success = False
                self.log("Invariants test failed", False)
            
            # Test 2: Trace Consistency
            if not self.test_trace_consistency():
                overall_success = False
                self.log("Trace consistency test failed", False)
            
            # Test 3: Monotonicity
            if not self.test_monotonicity():
                overall_success = False
                self.log("Monotonicity test failed", False)
            
            # Final summary
            if overall_success:
                self.log("=== All Physics Checks PASSED ===", True)
            else:
                self.log("=== Some Physics Checks FAILED ===", False)
                
        except Exception as e:
            self.log(f"Critical error during testing: {str(e)}", False)
            self.log(traceback.format_exc(), False)
            overall_success = False
        
        return overall_success, self.logs