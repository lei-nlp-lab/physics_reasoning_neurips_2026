import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs (problem constants)
        self.m = 1.0  # mass in kg (given)
        self.h = 8.0  # height in m (given)
        self.t = 2.0  # time in s (given)
        self.g = 10.0  # gravity in m/s² (given)
        
        # Safe attribute access for results
        self.v_t = getattr(solution_instance, 'v_t', solution_instance.results.get('v_t', 0.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', 0.0))
        
        # Initialize symbolic variables for invariant checking
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'h': sp.symbols('h', positive=True),
            't': sp.symbols('t', positive=True),
            'g': sp.symbols('g', positive=True),
            'v_t': sp.symbols('v_t', positive=True),
            'a': sp.symbols('a'),
            'f': sp.symbols('f', positive=True),
            'v_0': sp.symbols('v_0')
        }
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        
    def test_invariants(self):
        """Check physical invariants using KB formulas."""
        try:
            # 1. Check uniformly accelerated motion: h = (v_0 + v_t)/2 * t with v_0 = 0
            v_0 = 0.0
            expected_v_t = 2 * self.h / self.t - v_0  # From h = (v_0 + v_t)/2 * t
            if not math.isclose(self.v_t, expected_v_t, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: v_t invariant: got {self.v_t}, expected {expected_v_t}")
                return False
            self.log("PASS: v_t satisfies kinematic equation h = (v_0 + v_t)/2 * t")
            
            # 2. Check displacement equation: h = 0.5 * a * t^2
            expected_a = 2 * self.h / (self.t ** 2)
            if not math.isclose(self.a, expected_a, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: a invariant: got {self.a}, expected {expected_a}")
                return False
            self.log("PASS: a satisfies kinematic equation h = 0.5 * a * t^2")
            
            # 3. Check Newton's second law: mg - f = ma (downward positive)
            net_force = self.m * self.g - self.f
            expected_net_force = self.m * self.a
            if not math.isclose(net_force, expected_net_force, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: Newton's 2nd law: mg - f = {net_force}, ma = {expected_net_force}")
                return False
            self.log("PASS: Newton's 2nd law satisfied (mg - f = ma)")
            
            # 4. Check that a < g (air resistance present)
            if self.a >= self.g:
                self.log(f"FAIL: Acceleration a = {self.a} should be less than g = {self.g} due to air resistance")
                return False
            self.log(f"PASS: a = {self.a} < g = {self.g} (air resistance present)")
            
            # 5. Check force directions: f > 0 (opposes motion)
            if self.f <= 0:
                self.log(f"FAIL: Air resistance f = {self.f} should be positive")
                return False
            self.log(f"PASS: Air resistance f = {self.f} N > 0")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # From step_1 and step_2: v_t = 8 m/s, a = 4 m/s²
            # From step_4: f = m*(g - a) = 1*(10 - 4) = 6 N
            expected_f = self.m * (self.g - self.a)
            if not math.isclose(self.f, expected_f, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: Trace consistency: f = {self.f}, expected m*(g - a) = {expected_f}")
                return False
            self.log("PASS: Trace consistency: f = m*(g - a) holds")
            
            # Alternative: Check v_t from a and t: v_t = a*t (since v_0 = 0)
            expected_v_t_from_a = self.a * self.t
            if not math.isclose(self.v_t, expected_v_t_from_a, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: v_t from a: got {self.v_t}, expected a*t = {expected_v_t_from_a}")
                return False
            self.log("PASS: v_t = a*t consistent")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
            
    def test_monotonicity(self):
        """Perturb mass and check air resistance trend."""
        try:
            # Store original values
            original_m = self.m
            original_f = self.f
            original_a = self.a
            
            # Perturb mass by 10% (increase)
            perturbed_m = original_m * 1.1
            
            # Recalculate expected air resistance using f = m*(g - a)
            # Note: a is independent of mass in this kinematic setup (from h and t)
            perturbed_f = perturbed_m * (self.g - original_a)
            
            # Check monotonicity: f increases with m (since g - a > 0)
            if not perturbed_f > original_f:
                self.log(f"FAIL: Monotonicity: f({perturbed_m}) = {perturbed_f} ≤ f({original_m}) = {original_f}")
                return False
            self.log(f"PASS: Monotonicity: f increases with mass (f({original_m}) = {original_f}, f({perturbed_m}) ≈ {perturbed_f:.2f})")
            
            # Additional check: perturb mass downward
            perturbed_m2 = original_m * 0.9
            perturbed_f2 = perturbed_m2 * (self.g - original_a)
            if not perturbed_f2 < original_f:
                self.log(f"FAIL: Monotonicity: f({perturbed_m2}) = {perturbed_f2} ≥ f({original_m}) = {original_f}")
                return False
            self.log(f"PASS: Monotonicity: f decreases when mass decreases (f({perturbed_m2}) ≈ {perturbed_f2:.2f})")
            
            # Restore original mass (not strictly needed but good practice)
            self.m = original_m
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
            
    def run_all_tests(self):
        """Run all tests and return success and logs."""
        self.logs.clear()
        success = True
        
        tests = [
            ("Invariant tests", self.test_invariants),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} ---")
            if not test_func():
                success = False
                self.log(f"FAILED: {test_name}")
            else:
                self.log(f"PASSED: {test_name}")
                
        return success, self.logs