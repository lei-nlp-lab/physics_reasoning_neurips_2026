import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # SAFE ATTRIBUTE ACCESS: Use getattr with fallback to results dict or default
        self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', 0.0))
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 0.0))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 0.0))
        self.s = getattr(solution_instance, 's', solution_instance.results.get('s', 0.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        
        # Extract results with safe fallback
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v0', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
        
    def fuzzy_eq(self, a, b, msg):
        """Check if two numbers are close within 1-2% tolerance."""
        if math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"✓ {msg}: {a} ≈ {b}")
            return True
        else:
            self.log(f"✗ {msg}: {a} ≠ {b} (outside tolerance)")
            return False
            
    def test_invariants(self):
        """Test conservation of momentum and projectile motion laws."""
        success = True
        
        # 1. Horizontal projectile motion: v = s / t
        if self.t > 0:
            v_calc = self.s / self.t
            if not self.fuzzy_eq(self.v, v_calc, "Horizontal motion: v = s/t"):
                success = False
        else:
            self.log("⚠ Time t is zero or negative, skipping horizontal motion check")
            
        # 2. Vertical projectile motion: h = 0.5*g*t^2
        t_calc = math.sqrt(2 * self.h / self.g)
        if not self.fuzzy_eq(self.t, t_calc, "Vertical motion: t = √(2h/g)"):
            success = False
            
        # 3. Conservation of momentum: m*v0 = (M + m)*v
        left = self.m * self.v0
        right = (self.M + self.m) * self.v
        if not self.fuzzy_eq(left, right, "Momentum conservation: m*v0 = (M+m)*v"):
            success = False
            
        return success
        
    def test_trace_consistency(self):
        """Check intermediate results against final answer."""
        success = True
        
        # From steps: v = s * sqrt(g/(2h))
        v_expected = self.s * math.sqrt(self.g / (2 * self.h))
        if not self.fuzzy_eq(self.v, v_expected, "Velocity v from s and h"):
            success = False
            
        # Final answer should match: v0 = (M+m)/m * v
        v0_expected = (self.M + self.m) / self.m * self.v
        if not self.fuzzy_eq(self.v0, v0_expected, "v0 from momentum conservation"):
            success = False
            
        # Check against the given formula: v0 = (M+m)s√(2gh)/(2mh)
        v0_formula = (self.M + self.m) * self.s * math.sqrt(2 * self.g * self.h) / (2 * self.m * self.h)
        if not self.fuzzy_eq(self.v0, v0_formula, "v0 matches given formula"):
            success = False
            
        return success
        
    def test_monotonicity(self):
        """Test how v0 changes when mass m is perturbed."""
        try:
            original_m = self.m
            # Perturb m by +10%
            self.m = original_m * 1.10
            
            # Recalculate using the physics formula
            v_new = self.s * math.sqrt(self.g / (2 * self.h))
            v0_new = (self.M + self.m) / self.m * v_new
            
            # Restore original value
            self.m = original_m
            
            # v0 should DECREASE when m increases (since v0 = (M+m)/m * v)
            if v0_new < self.v0:
                self.log(f"✓ Monotonicity: v0 decreases when m increases ({self.v0} → {v0_new})")
                return True
            else:
                self.log(f"✗ Monotonicity: v0 did not decrease as expected ({self.v0} → {v0_new})")
                return False
                
        except Exception as e:
            self.log(f"⚠ Monotonicity test failed: {str(e)}")
            return False
            
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"❌ {test_name} crashed: {str(e)}")
                self.log(traceback.format_exc())
                overall_success = False
                
        self.log(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs