import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs with safe defaults
        self.length = getattr(solution_instance, 'length', getattr(solution_instance, 'L', 14.0))
        self.mu = getattr(solution_instance, 'mu', getattr(solution_instance, 'μ', 0.7))
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.v0 = getattr(solution_instance, 'v_0', solution_instance.results.get('v_0', 0.0))
        
        self.logs = []
        
    def test_invariants(self):
        """Verify Newton's 2nd Law and energy/work theorem."""
        try:
            # 1. Newton's second law: a = μ * g
            expected_a = self.mu * self.g
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: a = {self.a} but μ*g = {expected_a}")
                return False
            self.logs.append("PASS: Newton's 2nd Law (a = μ*g) holds.")
            
            # 2. Work-energy theorem: v0^2 = 2 * a * length (since v_final = 0)
            # Note: a is magnitude of deceleration, so positive.
            expected_v0_sq = 2 * self.a * self.length
            v0_sq = self.v0**2
            if not math.isclose(v0_sq, expected_v0_sq, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v0^2 = {v0_sq} but 2*a*L = {expected_v0_sq}")
                return False
            self.logs.append("PASS: Kinematic equation (v0^2 = 2*a*L) holds.")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answer."""
        try:
            # Step 1 from STEPS ANALYSIS: a = μ * g
            step1_a = self.mu * self.g
            if not math.isclose(self.a, step1_a, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step1 a={step1_a} vs result a={self.a}")
                return False
            self.logs.append("PASS: Step 1 (a = μ*g) consistent with result.")
            
            # Step 2 from STEPS ANALYSIS: v0 = sqrt(2 * a * length)
            step2_v0 = math.sqrt(2 * self.a * self.length)
            if not math.isclose(self.v0, step2_v0, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step2 v0={step2_v0} vs result v0={self.v0}")
                return False
            self.logs.append("PASS: Step 2 (v0 = sqrt(2*a*L)) consistent with result.")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        try:
            # Perturb μ: increase by 10%
            mu_perturbed = self.mu * 1.1
            a_perturbed = mu_perturbed * self.g
            v0_perturbed = math.sqrt(2 * a_perturbed * self.length)
            
            # Check trends: a should increase, v0 should increase
            if not (a_perturbed > self.a and v0_perturbed > self.v0):
                self.logs.append(f"FAIL: Perturb μ up -> a {a_perturbed} vs {self.a}, v0 {v0_perturbed} vs {self.v0}")
                return False
            self.logs.append("PASS: Increasing μ increases a and v0.")
            
            # Perturb length: increase by 10%
            length_perturbed = self.length * 1.1
            v0_perturbed_len = math.sqrt(2 * self.a * length_perturbed)
            
            # Check: v0 should increase with length for fixed a
            if not (v0_perturbed_len > self.v0):
                self.logs.append(f"FAIL: Perturb L up -> v0 {v0_perturbed_len} vs {self.v0}")
                return False
            self.logs.append("PASS: Increasing L increases v0 (for fixed a).")
            
            # Perturb g: increase by 10%
            g_perturbed = self.g * 1.1
            a_perturbed_g = self.mu * g_perturbed
            v0_perturbed_g = math.sqrt(2 * a_perturbed_g * self.length)
            
            # Check: a and v0 should increase with g
            if not (a_perturbed_g > self.a and v0_perturbed_g > self.v0):
                self.logs.append(f"FAIL: Perturb g up -> a {a_perturbed_g} vs {self.a}, v0 {v0_perturbed_g} vs {self.v0}")
                return False
            self.logs.append("PASS: Increasing g increases a and v0.")
            
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs.clear()
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test in tests:
            self.logs.append(f"\n--- Running {name} Test ---")
            if not test():
                success = False
        
        self.logs.append(f"\n--- Overall Result: {'PASS' if success else 'FAIL'} ---")
        return success, self.logs