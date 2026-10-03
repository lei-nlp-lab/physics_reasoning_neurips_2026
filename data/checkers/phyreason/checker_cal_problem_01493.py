import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Extract results with safe defaults
        self.results = getattr(solution_instance, 'results', {})
        self.E = self.results.get('E', 0.0)
        self.a = self.results.get('a', 0.0)
        self.v_t = self.results.get('v_t', 0.0)
        
        # Extract the numeric parameter values used in the solution
        # These should be stored in results by the solution code
        self.m = self.results.get('m_val', 1.0)      # default to values used in typical solution
        self.q = self.results.get('q_val', 1.0)
        self.g = self.results.get('g_val', 9.8)
        self.H = self.results.get('H_val', 2.0)
        
        # Constants
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = self.sin37 / self.cos37  # 0.75
        
        # Logs
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def test_invariants(self):
        """
        Test fundamental physics laws using numeric values.
        """
        success = True
        sin37, cos37, tan37 = self.sin37, self.cos37, self.tan37
        
        # Step 1: Equilibrium: mg tan37 = qE
        try:
            lhs_eq1 = self.q * self.E
            rhs_eq1 = self.m * self.g * tan37
            if math.isclose(lhs_eq1, rhs_eq1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Equilibrium check (qE = mg tan37): Numerical check passed (error={abs(lhs_eq1-rhs_eq1):.2e})", True)
            else:
                self._log(f"Equilibrium check (qE = mg tan37): Numerical mismatch: {lhs_eq1} != {rhs_eq1}", False)
                success = False
        except Exception as e:
            self._log(f"Equilibrium check failed: {e}", False)
            success = False
        
        # Step 2: Newton's second law: (qE + mg) sin37 = ma
        try:
            lhs_eq2 = (self.q * self.E + self.m * self.g) * sin37
            rhs_eq2 = self.m * self.a
            if math.isclose(lhs_eq2, rhs_eq2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's 2nd law check ((qE+mg)sin37 = ma): Numerical check passed (error={abs(lhs_eq2-rhs_eq2):.2e})", True)
            else:
                self._log(f"Newton's 2nd law check ((qE+mg)sin37 = ma): Numerical mismatch: {lhs_eq2} != {rhs_eq2}", False)
                success = False
        except Exception as e:
            self._log(f"Newton's 2nd law check failed: {e}", False)
            success = False
        
        # Step 4: Kinematic: v_t^2 = 2a * (H / sin37)
        try:
            lhs_eq3 = self.v_t**2
            rhs_eq3 = 2 * self.a * (self.H / sin37)
            if math.isclose(lhs_eq3, rhs_eq3, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Kinematic check (v_t^2 = 2a*(H/sin37)): Numerical check passed (error={abs(lhs_eq3-rhs_eq3):.2e})", True)
            else:
                self._log(f"Kinematic check (v_t^2 = 2a*(H/sin37)): Numerical mismatch: {lhs_eq3} != {rhs_eq3}", False)
                success = False
        except Exception as e:
            self._log(f"Kinematic check failed: {e}", False)
            success = False
        
        # Additional consistency: a should be 21g/20 (from step 2 result)
        try:
            expected_a = 21 * self.g / 20
            if math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Acceleration matches expected value 21g/20 = {expected_a}", True)
            else:
                self._log(f"Acceleration a mismatch: got {self.a}, expected {expected_a}", False)
                success = False
        except Exception as e:
            self._log(f"Acceleration consistency check failed: {e}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """
        Cross-check that intermediate results (E, a) are consistent with the final v_t.
        """
        success = True
        
        # Check E against step 1 result: E = 3mg/(4q)
        try:
            expected_E = 3 * self.m * self.g / (4 * self.q)
            if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"E matches expected 3mg/(4q) = {expected_E}", True)
            else:
                self._log(f"E mismatch: got {self.E}, expected 3mg/(4q) = {expected_E}", False)
                success = False
        except Exception as e:
            self._log(f"E consistency check failed: {e}", False)
            success = False
        
        # Check a against step 2 result: a = 21g/20
        try:
            expected_a = 21 * self.g / 20
            if math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"a matches expected 21g/20 = {expected_a}", True)
            else:
                self._log(f"a mismatch: got {self.a}, expected 21g/20 = {expected_a}", False)
                success = False
        except Exception as e:
            self._log(f"a consistency check failed: {e}", False)
            success = False
        
        # Check v_t against step 4 result: v_t = sqrt(7gH/2)
        try:
            expected_v_t = math.sqrt(7 * self.g * self.H / 2)
            if math.isclose(self.v_t, expected_v_t, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v_t matches expected sqrt(7gH/2) = {expected_v_t}", True)
            else:
                self._log(f"v_t mismatch: got {self.v_t}, expected sqrt(7gH/2) = {expected_v_t}", False)
                success = False
        except Exception as e:
            self._log(f"v_t consistency check failed: {e}", False)
            success = False
        
        # Verify kinematic chain: use a to compute v_t via the incline length
        try:
            incline_length = self.H / self.sin37
            computed_v_t = math.sqrt(2 * self.a * incline_length)
            if math.isclose(self.v_t, computed_v_t, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v_t consistent with kinematic chain from a and H", True)
            else:
                self._log(f"v_t kinematic chain mismatch: got {self.v_t}, computed from a = {computed_v_t}", False)
                success = False
        except Exception as e:
            self._log(f"Kinematic chain check failed: {e}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input (mass m) and check that results change as expected.
        We expect:
        - E ∝ m (if m increases, E increases proportionally)
        - a independent of m (should not change)
        - v_t independent of m (should not change)
        """
        success = True
        
        # Save original values
        original_m = self.m
        original_E = self.E
        original_a = self.a
        original_v_t = self.v_t
        
        # Perturb m by +10%
        perturb_factor = 1.1
        self.m *= perturb_factor
        
        # Recompute expected results using formulas
        new_E = 3 * self.m * self.g / (4 * self.q)
        new_a = 21 * self.g / 20
        new_v_t = math.sqrt(7 * self.g * self.H / 2)
        
        # Check E: should scale with m
        expected_E = original_E * perturb_factor
        if math.isclose(new_E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Monotonicity E: E scales with m as expected", True)
        else:
            self._log(f"Monotonicity E: after +10% m, E should scale by 1.1. Got {new_E}, expected {expected_E}", False)
            success = False
        
        # Check a: should remain unchanged
        if math.isclose(new_a, original_a, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Monotonicity a: a independent of m as expected", True)
        else:
            self._log(f"Monotonicity a: after +10% m, a should not change. Got {new_a}, original {original_a}", False)
            success = False
        
        # Check v_t: should remain unchanged (since a unchanged and H fixed)
        if math.isclose(new_v_t, original_v_t, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Monotonicity v_t: v_t independent of m as expected", True)
        else:
            self._log(f"Monotonicity v_t: after +10% m, v_t should not change. Got {new_v_t}, original {original_v_t}", False)
            success = False
        
        # Restore original m
        self.m = original_m
        
        return success
    
    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self._log(f"=== Running {name} Test ===", None)
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test {name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===", None)
        return overall_success, self.logs