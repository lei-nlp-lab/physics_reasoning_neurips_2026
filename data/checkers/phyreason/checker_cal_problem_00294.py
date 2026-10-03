import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all required quantities with safe defaults
        self.a1 = getattr(solution_instance, 'a_1', self.results.get('a_1', 0.0))
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.mu1 = getattr(solution_instance, 'mu_1', self.results.get('mu_1', 0.0))
        self.v0 = getattr(solution_instance, 'v_0', self.results.get('v_0', 0.0))
        self.a2 = getattr(solution_instance, 'a_2', self.results.get('a_2', 0.0))
        self.s = getattr(solution_instance, 's', self.results.get('s', 0.0))
        self.mu2 = getattr(solution_instance, 'mu_2', self.results.get('mu_2', 0.0))
        
        # Constants from problem statement
        self.g = 10.0
        self.M = 2.0  # M = 2m (we'll use m=1 for relative calculations)
        self.m = 1.0
        self.t1 = 0.5
        self.t2 = 1.5
        self.v1 = 2.0  # Block velocity at t=0.5s from graph
        
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def fuzzy_eq(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify fundamental physics laws based on KB.
        """
        checks = []
        
        # 1. Check μ1 = a1/g (from step 4)
        expected_mu1 = self.a1 / self.g
        if not self.fuzzy_eq(self.mu1, expected_mu1):
            checks.append(f"FAIL: μ1 = {self.mu1}, expected a1/g = {expected_mu1}")
        else:
            checks.append(f"OK: μ1 = a1/g")
        
        # 2. Check μ2 = a2/g (from step 7)
        expected_mu2 = self.a2 / self.g
        if not self.fuzzy_eq(self.mu2, expected_mu2):
            checks.append(f"FAIL: μ2 = {self.mu2}, expected a2/g = {expected_mu2}")
        else:
            checks.append(f"OK: μ2 = a2/g")
        
        # 3. Newton's 2nd Law for block during 0-0.5s
        # Friction force on block = m*a1 = μ1*m*g
        lhs = self.m * self.a1
        rhs = self.mu1 * self.m * self.g
        if not self.fuzzy_eq(lhs, rhs):
            checks.append(f"FAIL: Block N2L: m*a1 = {lhs}, μ1*m*g = {rhs}")
        else:
            checks.append(f"OK: Block Newton's 2nd Law holds")
        
        # 4. Newton's 2nd Law for system after 0.5s
        # Friction on system = (M+m)*a2 = μ2*(M+m)*g
        lhs_sys = (self.M + self.m) * self.a2
        rhs_sys = self.mu2 * (self.M + self.m) * self.g
        if not self.fuzzy_eq(lhs_sys, rhs_sys):
            checks.append(f"FAIL: System N2L: (M+m)*a2 = {lhs_sys}, μ2*(M+m)*g = {rhs_sys}")
        else:
            checks.append(f"OK: System Newton's 2nd Law holds")
        
        # 5. Check plate acceleration formula (step 9)
        # The formula gives magnitude: |a| = [μ1*m*g + μ2*(M+m)*g] / M
        expected_a_magnitude = (self.mu1*self.m*self.g + self.mu2*(self.M+self.m)*self.g) / self.M
        if not self.fuzzy_eq(self.a, expected_a_magnitude):
            checks.append(f"FAIL: a (magnitude) = {self.a}, expected [μ1*m*g + μ2*(M+m)*g]/M = {expected_a_magnitude}")
        else:
            checks.append(f"OK: Plate acceleration magnitude formula")
        
        # 6. Check v0 equation (step 10): v1 = v0 - a*t1 (since a is magnitude of deceleration)
        # Rearranged: v0 = v1 + a*t1
        expected_v0 = self.v1 + self.a * self.t1
        if not self.fuzzy_eq(self.v0, expected_v0):
            checks.append(f"FAIL: v0 = {self.v0}, expected v1 + a*t1 = {expected_v0}")
        else:
            checks.append(f"OK: v0 kinematic relation")
        
        # 7. Check displacement formula (step 11)
        # s = 0.5*(v0+v1)*t1 + 0.5*v1*(t2-t1)
        expected_s = 0.5*(self.v0 + self.v1)*self.t1 + 0.5*self.v1*(self.t2 - self.t1)
        if not self.fuzzy_eq(self.s, expected_s):
            checks.append(f"FAIL: s = {self.s}, expected kinematic sum = {expected_s}")
        else:
            checks.append(f"OK: Displacement formula")
        
        return all("FAIL" not in check for check in checks), checks

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        checks = []
        
        # Target answers from problem
        target_mu1 = 0.4
        target_mu2 = 0.2
        target_s = 2.625
        
        if not self.fuzzy_eq(self.mu1, target_mu1):
            checks.append(f"FAIL: μ1 = {self.mu1} doesn't match target {target_mu1}")
        else:
            checks.append(f"OK: μ1 matches target")
        
        if not self.fuzzy_eq(self.mu2, target_mu2):
            checks.append(f"FAIL: μ2 = {self.mu2} doesn't match target {target_mu2}")
        else:
            checks.append(f"OK: μ2 matches target")
        
        if not self.fuzzy_eq(self.s, target_s):
            checks.append(f"FAIL: s = {self.s} doesn't match target {target_s}")
        else:
            checks.append(f"OK: s matches target")
        
        # Check that a1 and a2 match graph slopes
        # a1 = (2-0)/0.5 = 4, a2 = (0-2)/1 = -2 (magnitude 2)
        if not self.fuzzy_eq(self.a1, 4.0):
            checks.append(f"WARN: a1 = {self.a1}, expected 4 from graph slope")
        if not self.fuzzy_eq(self.a2, 2.0):
            checks.append(f"WARN: a2 = {self.a2}, expected magnitude 2 from graph slope")
        
        return all("FAIL" not in check for check in checks), checks

    def test_monotonicity(self):
        """
        Perturb mass m and check trend consistency.
        Since M = 2m, we change both proportionally.
        """
        checks = []
        
        # Store original
        orig_m = self.m
        orig_M = self.M
        orig_a = self.a
        
        # Test with 10% increase
        self.m = 1.1
        self.M = 2.2
        
        # Recalculate expected values using formulas
        # μ1 and μ2 should remain unchanged (independent of mass)
        new_mu1 = self.mu1  # unchanged
        new_mu2 = self.mu2  # unchanged
        
        # Plate acceleration magnitude: a = [μ1*m*g + μ2*(M+m)*g] / M
        new_a_magnitude = (new_mu1*self.m*self.g + new_mu2*(self.M+self.m)*self.g) / self.M
        
        # Check trends
        # Since all masses increased, friction forces increase, so a (magnitude) should increase
        if new_a_magnitude > orig_a:
            checks.append(f"OK: a (magnitude) increases with mass (perturbation test)")
        else:
            checks.append(f"WARN: a (magnitude) doesn't increase with mass as expected")
        
        # Restore original values
        self.m = orig_m
        self.M = orig_M
        self.a = orig_a
        
        return True, checks

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_success = True
        all_logs = []
        
        try:
            # Test 1: Invariants
            success1, logs1 = self.test_invariants()
            all_logs.append("=== Test Invariants ===")
            all_logs.extend(logs1)
            if not success1:
                all_success = False
            
            # Test 2: Trace Consistency
            success2, logs2 = self.test_trace_consistency()
            all_logs.append("\n=== Test Trace Consistency ===")
            all_logs.extend(logs2)
            if not success2:
                all_success = False
            
            # Test 3: Monotonicity
            success3, logs3 = self.test_monotonicity()
            all_logs.append("\n=== Test Monotonicity ===")
            all_logs.extend(logs3)
            if not success3:
                all_success = False
            
        except Exception as e:
            all_success = False
            all_logs.append(f"ERROR during testing: {str(e)}")
            all_logs.append(traceback.format_exc())
        
        return all_success, all_logs