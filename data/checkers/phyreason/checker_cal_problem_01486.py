import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: get results and inputs from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Extract all needed quantities with safe defaults
        # Inputs
        self.r = self.inputs.get('r', 1.5)
        self.mu1 = self.inputs.get('mu1', 0.6)
        self.mu2 = self.inputs.get('mu2', 0.225)
        self.h = self.inputs.get('h', 0.8)
        self.g = self.inputs.get('g', 10.0)
        
        # Results (use 0.0 as fallback if missing)
        self.x = self.results.get('x', 0.0)
        self.F_fm = self.results.get('F_fm', 0.0)
        self.a = self.results.get('a', 0.0)
        self.F_N = self.results.get('F_N', 0.0)
        self.v_0 = self.results.get('v_0', 0.0)
        self.F_f = self.results.get('F_f', 0.0)
        self.R = self.results.get('R', 0.0)
        self.omega = self.results.get('omega', 0.0)
        
        # For symbolic checks
        self.m = sp.symbols('m', positive=True)  # mass cancels in many equations
        
    def test_invariants(self):
        logs = []
        
        try:
            # 1. Check centripetal force condition: F_fm = m * r * omega^2
            expected_F_fm = self.m * self.r * self.omega**2
            # Since mass is unknown, we check the ratio F_fm / (r * omega^2) = m
            # Instead, we can check that F_fm / (r * omega^2) is consistent with F_fm / (mu1 * m * g)
            # But we don't have m. Instead, use the condition from step 1 and step 2:
            # At threshold: F_fm = mu1 * F_N and F_N = m * g
            # So F_fm = mu1 * m * g
            # Therefore, mu1 * m * g = m * r * omega^2  => omega = sqrt(mu1 * g / r)
            # So we check omega:
            expected_omega = math.sqrt(self.mu1 * self.g / self.r)
            if not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: omega = {self.omega} does not match sqrt(mu1*g/r) = {expected_omega}")
            else:
                logs.append("PASS: Centripetal force condition (omega) satisfied.")
                
            # 2. Check vertical equilibrium: F_N = m * g (we cannot check numerically without m)
            # Instead, we can check that F_f / mu2 = F_N, because F_f = mu2 * F_N
            if self.F_f != 0 and self.F_N != 0:
                expected_F_f = self.mu2 * self.F_N
                if not math.isclose(self.F_f, expected_F_f, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: F_f = {self.F_f} does not match mu2 * F_N = {expected_F_f}")
                else:
                    logs.append("PASS: F_f = mu2 * F_N holds.")
            else:
                logs.append("SKIP: F_f or F_N not provided, skipping friction relation.")
                
            # 3. Check kinematic relation: v_0 = omega * r
            expected_v0 = self.omega * self.r
            if not math.isclose(self.v_0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_0 = {self.v_0} does not match omega * r = {expected_v0}")
            else:
                logs.append("PASS: v_0 = omega * r holds.")
                
            # 4. Check deceleration: a = - mu2 * g (magnitude)
            expected_a_mag = self.mu2 * self.g
            if not math.isclose(abs(self.a), expected_a_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: |a| = {abs(self.a)} does not match mu2 * g = {expected_a_mag}")
            else:
                logs.append("PASS: Deceleration magnitude matches mu2*g.")
                
            # 5. Check kinematic equation: v_0^2 = 2 * |a| * x
            if self.a != 0:
                expected_x = self.v_0**2 / (2 * abs(self.a))
                if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x = {self.x} does not match v_0^2/(2|a|) = {expected_x}")
                else:
                    logs.append("PASS: Kinematic equation v_0^2 = 2|a|x holds.")
            else:
                logs.append("SKIP: a is zero, cannot check kinematic equation.")
                
            # 6. Check Pythagorean theorem: R = sqrt(r^2 + x^2)
            expected_R = math.sqrt(self.r**2 + self.x**2)
            if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: R = {self.R} does not match sqrt(r^2 + x^2) = {expected_R}")
            else:
                logs.append("PASS: R = sqrt(r^2 + x^2) holds.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            
        return logs
    
    def test_trace_consistency(self):
        logs = []
        
        try:
            # Check that the given steps are consistent with each other
            # Step 3: omega = sqrt(mu1 * g / r)
            expected_omega = math.sqrt(self.mu1 * self.g / self.r)
            if not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: omega from step 3 inconsistent: {self.omega} vs {expected_omega}")
            else:
                logs.append("PASS: Step 3 (omega) is consistent with inputs.")
                
            # Step 6: v_0 = omega * r
            expected_v0 = self.omega * self.r
            if not math.isclose(self.v_0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_0 from step 6 inconsistent: {self.v_0} vs {expected_v0}")
            else:
                logs.append("PASS: Step 6 (v_0) is consistent with omega and r.")
                
            # Step 5: F_f = mu2 * m * g -> but we don't have m. Instead, check that a = F_f / m = mu2 * g
            if self.F_f != 0 and self.a != 0:
                # Since a = F_f / m, we expect F_f / a = m, and then mu2 * g should equal a in magnitude
                if not math.isclose(abs(self.a), self.mu2 * self.g, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a = {self.a} does not match mu2*g = {self.mu2*self.g}")
                else:
                    logs.append("PASS: Step 5 (F_f) and step 4 (a) are consistent.")
            else:
                logs.append("SKIP: F_f or a not provided, skipping step 5 consistency.")
                
            # Step 8: x = v_0^2 / (2 * a) with a = mu2 * g
            expected_x = self.v_0**2 / (2 * self.mu2 * self.g)
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x from step 8 inconsistent: {self.x} vs {expected_x}")
            else:
                logs.append("PASS: Step 8 (x) is consistent with v_0 and mu2*g.")
                
            # Step 9: R = sqrt(r^2 + x^2)
            expected_R = math.sqrt(self.r**2 + self.x**2)
            if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: R from step 9 inconsistent: {self.R} vs {expected_R}")
            else:
                logs.append("PASS: Step 9 (R) is consistent with r and x.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_mu1 = self.mu1
        
        try:
            # Perturb mu1 by +10% and recalculate omega and R
            perturbed_mu1 = original_mu1 * 1.1
            perturbed_omega = math.sqrt(perturbed_mu1 * self.g / self.r)
            # Recalculate v0, x, R
            perturbed_v0 = perturbed_omega * self.r
            perturbed_x = perturbed_v0**2 / (2 * self.mu2 * self.g)
            perturbed_R = math.sqrt(self.r**2 + perturbed_x**2)
            
            # Check monotonicity: if mu1 increases, omega should increase (since sqrt(mu1*g/r) is increasing in mu1)
            if perturbed_omega > self.omega:
                logs.append("PASS: omega increases with mu1 (as expected).")
            else:
                logs.append("FAIL: omega does not increase with mu1.")
                
            # Then R should also increase because x increases (since v0 increases)
            if perturbed_R > self.R:
                logs.append("PASS: R increases with mu1 (as expected).")
            else:
                logs.append("FAIL: R does not increase with mu1.")
                
            # Restore original mu1 (though not strictly necessary, but for clarity)
            self.mu1 = original_mu1
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            self.mu1 = original_mu1  # restore on error
            
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test and collect logs
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== {test_name} ===")
            logs = test_method()
            all_logs.extend(logs)
            # Check for any FAIL or ERROR in logs
            for log in logs:
                if log.startswith("FAIL") or log.startswith("ERROR"):
                    success = False
        
        return success, all_logs

# Example usage (for testing the checker):
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.inputs = {
                'r': 1.5,
                'mu1': 0.6,
                'mu2': 0.225,
                'h': 0.8,
                'g': 10.0
            }
            self.results = {
                'omega': 2.0,
                'v_0': 3.0,
                'a': -2.25,
                'x': 2.0,
                'R': 2.5,
                'F_f': 2.25,  # assuming m=1 for simplicity
                'F_N': 10.0,  # m=1 -> F_N = 10
                'F_fm': 6.0   # m=1 -> F_fm = mu1 * m * g = 0.6*1*10=6
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    print("Success:", success)
    for log in logs:
        print(log)