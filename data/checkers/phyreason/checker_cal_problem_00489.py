import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Inputs from problem statement (with safe fallbacks)
        # Use known constants from the problem, not from solution_instance
        self.q = 2e-5  # C (given in problem)
        self.delta_U_given = 0.2  # J (given in problem)
        self.d = 0.02  # m (given in problem)
        self.theta_deg = 60.0  # degrees (given in problem)
        
        # Results from the solution (with safe access via .results dictionary)
        results = getattr(solution_instance, 'results', {})
        self.W_AB = results.get('W_AB', 0.0)
        self.U_AB = results.get('U_AB', 0.0)
        self.d_AB = results.get('d_AB', 0.0)
        self.E = results.get('E', 0.0)
        
        # Derived constants
        self.theta_rad = math.radians(self.theta_deg)
        self.cos_theta = math.cos(self.theta_rad)
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Log storage
        self.logs = []

    def _log(self, message: str, success: bool) -> None:
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")

    def test_invariants(self) -> bool:
        """
        Verify fundamental physics laws from the KB.
        """
        success = True
        
        # 1. Work-energy theorem: W_AB = -ΔU (given ΔU = 0.2 J increase)
        if not math.isclose(self.W_AB, -self.delta_U_given, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Work-energy invariant violated: W_AB = {self.W_AB} != -ΔU = {-self.delta_U_given}", False)
            success = False
        else:
            self._log(f"Work-energy invariant holds: W_AB = {self.W_AB} ≈ -ΔU = {-self.delta_U_given}", True)
        
        # 2. Potential difference definition: U_AB = W_AB / q
        if abs(self.q) > 1e-12:  # Avoid division by zero
            expected_U_AB = self.W_AB / self.q
            if not math.isclose(self.U_AB, expected_U_AB, 
                                rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Potential difference invariant violated: U_AB = {self.U_AB} != W_AB/q = {expected_U_AB}", False)
                success = False
            else:
                self._log(f"Potential difference invariant holds: U_AB = {self.U_AB} ≈ W_AB/q = {expected_U_AB}", True)
        else:
            self._log("Skipped potential difference invariant (q ≈ 0)", False)
            success = False
        
        # 3. Geometric projection: d_AB = d * cosθ
        expected_d_AB = self.d * self.cos_theta
        if not math.isclose(self.d_AB, expected_d_AB, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Geometric invariant violated: d_AB = {self.d_AB} != d*cosθ = {expected_d_AB}", False)
            success = False
        else:
            self._log(f"Geometric invariant holds: d_AB = {self.d_AB} ≈ d*cosθ = {expected_d_AB}", True)
        
        # 4. Uniform field relationship: E = U_AB / d_AB
        if abs(self.d_AB) > 1e-12:  # Avoid division by zero
            expected_E = self.U_AB / self.d_AB
            if not math.isclose(self.E, expected_E, 
                                rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Uniform field invariant violated: E = {self.E} != U_AB/d_AB = {expected_E}", False)
                success = False
            else:
                self._log(f"Uniform field invariant holds: E = {self.E} ≈ U_AB/d_AB = {expected_E}", True)
        else:
            self._log("Skipped uniform field invariant (d_AB ≈ 0)", False)
            success = False
        
        # 5. Work formula: W_AB = q * E * d_AB (scalar form, signs matter)
        expected_W_from_E = self.q * self.E * self.d_AB
        if not math.isclose(self.W_AB, expected_W_from_E, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Work formula invariant violated: W_AB = {self.W_AB} != q*E*d_AB = {expected_W_from_E}", False)
            success = False
        else:
            self._log(f"Work formula invariant holds: W_AB = {self.W_AB} ≈ q*E*d_AB = {expected_W_from_E}", True)
        
        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results from the steps analysis.
        """
        success = True
        
        # Expected values from the steps analysis
        expected_W_AB = -0.2
        expected_U_AB = -10000.0
        expected_d_AB = 0.01
        expected_E_mag = 1000000.0
        
        # Compare with extracted results (using absolute tolerance for exact values)
        if not math.isclose(self.W_AB, expected_W_AB, rel_tol=1e-4, abs_tol=1e-4):
            self._log(f"Trace inconsistency in W_AB: {self.W_AB} != {expected_W_AB}", False)
            success = False
        else:
            self._log(f"W_AB trace consistent: {self.W_AB} ≈ {expected_W_AB}", True)
        
        if not math.isclose(self.U_AB, expected_U_AB, rel_tol=1e-4, abs_tol=1e-4):
            self._log(f"Trace inconsistency in U_AB: {self.U_AB} != {expected_U_AB}", False)
            success = False
        else:
            self._log(f"U_AB trace consistent: {self.U_AB} ≈ {expected_U_AB}", True)
        
        if not math.isclose(self.d_AB, expected_d_AB, rel_tol=1e-4, abs_tol=1e-4):
            self._log(f"Trace inconsistency in d_AB: {self.d_AB} != {expected_d_AB}", False)
            success = False
        else:
            self._log(f"d_AB trace consistent: {self.d_AB} ≈ {expected_d_AB}", True)
        
        # E magnitude check (ignore sign for magnitude)
        if not math.isclose(abs(self.E), expected_E_mag, rel_tol=1e-4, abs_tol=1e-4):
            self._log(f"Trace inconsistency in |E|: |{self.E}| != {expected_E_mag}", False)
            success = False
        else:
            self._log(f"|E| trace consistent: |{self.E}| ≈ {expected_E_mag}", True)
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb the distance d and verify the expected trend in E.
        """
        success = True
        
        # Store original values
        original_d = self.d
        original_d_AB = self.d_AB
        original_E = self.E
        
        # Perturbation: increase d by 1%
        perturbed_d = original_d * 1.01
        perturbed_d_AB = perturbed_d * self.cos_theta
        
        # Since delta_U and q are fixed, U_AB remains unchanged
        # Avoid division by zero
        if abs(perturbed_d_AB) > 1e-12:
            perturbed_E = self.U_AB / perturbed_d_AB
        else:
            self._log("Cannot compute perturbed_E (d_AB ≈ 0)", False)
            return False
        
        # Expected trend: E is inversely proportional to d_AB (and thus to d)
        # So ratio should be original_d_AB / perturbed_d_AB
        expected_ratio = original_d_AB / perturbed_d_AB
        if abs(original_E) > 1e-12:
            actual_ratio = perturbed_E / original_E
        else:
            self._log("Cannot compute ratio (original_E ≈ 0)", False)
            return False
        
        # Check if the ratio matches (allowing for sign differences)
        # Since U_AB is negative and d_AB is positive, E is negative
        # When d increases, the magnitude of E decreases but sign stays negative
        if not math.isclose(abs(actual_ratio), abs(expected_ratio), rel_tol=5e-3, abs_tol=5e-3):
            self._log(f"Monotonicity violated: E does not scale inversely with d. " 
                     f"Actual ratio magnitude |{actual_ratio}|, expected |{expected_ratio}|", False)
            success = False
        else:
            self._log(f"Monotonicity holds: E scales inversely with d (ratio magnitude |{actual_ratio}| ≈ |{expected_ratio}|)", True)
        
        # Additional check: if d increases, |E| should decrease (since U_AB fixed and negative)
        if abs(perturbed_E) < abs(original_E):
            self._log(f"Trend correct: |E| decreases when d increases (|{perturbed_E}| < |{original_E}|)", True)
        else:
            self._log(f"Trend incorrect: |E| does not decrease when d increases", False)
            success = False
        
        # Restore original values (important for subsequent tests)
        self.d = original_d
        self.d_AB = original_d_AB
        self.E = original_E
        
        return success

    def run_all_tests(self) -> tuple[bool, list[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        
        # Run each test
        test_functions = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for name, test_func in test_functions:
            self.logs.append(f"--- Running {name} Test ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test crashed: {e}", False)
                overall_success = False
        
        self.logs.append(f"=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs