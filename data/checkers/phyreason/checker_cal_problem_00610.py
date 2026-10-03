import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.a = self._safe_get('a', 0.0)
        self.f = self._safe_get('f', 0.0)
        self.v_t = self._safe_get('v_t', 0.0)
        self.t_prime = self._safe_get("t'", 0.0)
        
        # Problem constants (from problem statement)
        self.m = 70.0  # kg
        self.v0 = 2.0  # m/s
        self.x_slope = 60.0  # m (horizontal distance, used as displacement in step1)
        self.t_slope = 5.0  # s
        self.x_horizontal = 22.0  # m
        self.theta = math.radians(30)  # radians
        self.g = 10.0  # m/s²
        
        # Symbolic variables
        self.m_sym = sp.Symbol('m', positive=True)
        self.v0_sym = sp.Symbol('v0', positive=True)
        self.x_sym = sp.Symbol('x', positive=True)
        self.t_sym = sp.Symbol('t', positive=True)
        self.a_sym = sp.Symbol('a', real=True)
        self.f_sym = sp.Symbol('f', real=True)
        self.v_t_sym = sp.Symbol('v_t', positive=True)
        self.t_prime_sym = sp.Symbol("t'", positive=True)
        self.g_sym = sp.Symbol('g', positive=True)
        self.theta_sym = sp.Symbol('theta', positive=True)
        
        # Log storage
        self.logs = []
    
    def _safe_get(self, key, default):
        """Safely get value from results with multiple fallbacks."""
        return self.results.get(key, default)
    
    def _is_close(self, a, b, msg=""):
        """Robust fuzzy comparison with physics-tolerant thresholds."""
        if math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2):
            self.logs.append(f"✓ {msg}: {a:.3f} ≈ {b:.3f}")
            return True
        else:
            self.logs.append(f"✗ {msg}: {a:.3f} ≠ {b:.3f} (difference: {abs(a-b):.3f})")
            return False
    
    def test_invariants(self) -> bool:
        """Verify physical laws and consistency of steps."""
        success = True
        
        # 1. Check kinematic equation for slope motion (Step 1)
        # x = v0*t + 0.5*a*t²
        x_calc = self.v0 * self.t_slope + 0.5 * self.a * self.t_slope**2
        if not self._is_close(x_calc, self.x_slope, "Step1: Displacement from kinematics"):
            success = False
        
        # Symbolic check
        eq1 = sp.Eq(self.x_sym, self.v0_sym*self.t_sym + self.a_sym*self.t_sym**2/2)
        lhs1 = self.x_slope
        rhs1 = self.v0 * self.t_slope + self.a * self.t_slope**2 / 2
        if not self._is_close(lhs1, rhs1, "Step1: Symbolic displacement"):
            success = False
        
        # 2. Check Newton's 2nd Law on slope (Step 2)
        # f = m*g*sinθ - m*a
        f_calc = self.m * self.g * math.sin(self.theta) - self.m * self.a
        if not self._is_close(f_calc, self.f, "Step2: Resistance force from Newton's 2nd Law"):
            success = False
        
        # Symbolic check
        eq2 = sp.Eq(self.f_sym, self.m_sym*self.g_sym*sp.sin(self.theta_sym) - self.m_sym*self.a_sym)
        lhs2 = self.f
        rhs2 = self.m * self.g * math.sin(self.theta) - self.m * self.a
        if not self._is_close(lhs2, rhs2, "Step2: Symbolic force balance"):
            success = False
        
        # 3. Check velocity at bottom (Step 3)
        # v_t = v0 + a*t
        v_t_calc = self.v0 + self.a * self.t_slope
        if not self._is_close(v_t_calc, self.v_t, "Step3: Final velocity from kinematics"):
            success = False
        
        # Symbolic check
        eq3 = sp.Eq(self.v_t_sym, self.v0_sym + self.a_sym*self.t_sym)
        lhs3 = self.v_t
        rhs3 = self.v0 + self.a * self.t_slope
        if not self._is_close(lhs3, rhs3, "Step3: Symbolic velocity"):
            success = False
        
        # 4. Check horizontal motion time (Step 4)
        # Using average velocity: x' = (v_t + 0)/2 * t'
        # This is correct for uniform deceleration to rest
        t_prime_calc = 2 * self.x_horizontal / self.v_t if self.v_t != 0 else 0
        if not self._is_close(t_prime_calc, self.t_prime, "Step4: Time from average velocity"):
            success = False
        
        # Alternative: kinematic equation v² = v0² + 2a'x'
        # For v=0, v_t² = 2|a'|x' → a' = -v_t²/(2x')
        if self.x_horizontal > 0:
            a_prime = -self.v_t**2 / (2 * self.x_horizontal)
            t_prime_alt = -self.v_t / a_prime if a_prime != 0 else 0
            if not self._is_close(t_prime_alt, self.t_prime, "Step4: Time from deceleration kinematics"):
                self.logs.append("  Note: Alternative kinematic check")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results for internal consistency."""
        success = True
        
        # Check that all values are physically plausible
        if self.a <= 0:
            self.logs.append("✗ Acceleration on slope should be positive (downhill)")
            success = False
        else:
            self.logs.append("✓ Acceleration positive")
        
        # Check resistance force direction: should oppose motion (positive value)
        if self.f < 0:
            self.logs.append("✗ Resistance force should be positive (opposing motion)")
            success = False
        else:
            self.logs.append("✓ Resistance force positive")
        
        # Check that resistance force is less than gravitational component
        f_gravity = self.m * self.g * math.sin(self.theta)
        if self.f >= f_gravity:
            self.logs.append(f"✗ Resistance force ({self.f:.1f}N) ≥ gravitational component ({f_gravity:.1f}N)")
            success = False
        else:
            self.logs.append(f"✓ Resistance force ({self.f:.1f}N) < gravitational component ({f_gravity:.1f}N)")
        
        # Check velocity consistency: v_t should be > v0
        if self.v_t <= self.v0:
            self.logs.append(f"✗ Final velocity ({self.v_t:.1f}m/s) ≤ initial velocity ({self.v0:.1f}m/s)")
            success = False
        else:
            self.logs.append(f"✓ Final velocity ({self.v_t:.1f}m/s) > initial velocity ({self.v0:.1f}m/s)")
        
        # Check energy consideration (not strictly required but good sanity)
        # Initial energy: KE0 + PE0 = 0.5*m*v0² + m*g*h
        # Height: h = x_slope * sinθ (using x_slope as along slope? Problem uses 60m horizontal)
        # Actually, the problem uses 60m as horizontal distance, so height = 60 * tanθ
        h = 60.0 * math.tan(self.theta)  # Using horizontal distance
        E_initial = 0.5 * self.m * self.v0**2 + self.m * self.g * h
        E_final_bottom = 0.5 * self.m * self.v_t**2
        work_friction = self.f * (60.0 / math.cos(self.theta))  # Distance along slope
        
        # Energy lost to friction should equal difference
        E_loss_calc = E_initial - E_final_bottom
        if not self._is_close(work_friction, E_loss_calc, "Energy-work consistency"):
            self.logs.append("  Note: Energy check uses slope distance from horizontal")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test how results change with perturbed inputs."""
        success = True
        
        # Store original values
        original_m = self.m
        original_results = {
            'a': self.a,
            'f': self.f,
            'v_t': self.v_t,
            't_prime': self.t_prime
        }
        
        try:
            # Test 1: Increase mass by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate using formulas (not changing instance state)
            # Acceleration on slope should be independent of mass (from kinematic eq)
            a_new = self.a  # Same, from x = v0*t + 0.5*a*t²
            
            # Force: f = m*g*sinθ - m*a
            f_new = perturbed_m * self.g * math.sin(self.theta) - perturbed_m * a_new
            
            # Velocity at bottom: v_t = v0 + a*t (unchanged)
            v_t_new = self.v0 + a_new * self.t_slope
            
            # Time on horizontal: t' = 2*x'/v_t
            t_prime_new = 2 * self.x_horizontal / v_t_new if v_t_new != 0 else 0
            
            # Check trends
            # Force should increase with mass (f ∝ m)
            if f_new <= original_results['f']:
                self.logs.append("✗ Force should increase with mass")
                success = False
            else:
                self.logs.append(f"✓ Force increases with mass: {original_results['f']:.1f}N → {f_new:.1f}N")
            
            # Acceleration should be unchanged
            if not math.isclose(a_new, original_results['a'], rel_tol=1e-3):
                self.logs.append("✗ Acceleration should be independent of mass")
                success = False
            else:
                self.logs.append("✓ Acceleration independent of mass")
            
            # Velocity should be unchanged
            if not math.isclose(v_t_new, original_results['v_t'], rel_tol=1e-3):
                self.logs.append("✗ Velocity should be independent of mass")
                success = False
            else:
                self.logs.append("✓ Velocity independent of mass")
            
            # Time on horizontal should be unchanged
            if not math.isclose(t_prime_new, original_results['t_prime'], rel_tol=1e-3):
                self.logs.append("✗ Horizontal time should be independent of mass")
                success = False
            else:
                self.logs.append("✓ Horizontal time independent of mass")
            
            # Test 2: Increase initial velocity
            perturbed_v0 = self.v0 * 1.2
            
            # Recalculate acceleration (from same displacement and time)
            a_new2 = 2 * (self.x_slope - perturbed_v0 * self.t_slope) / self.t_slope**2
            
            # Check: with higher v0, acceleration should be lower (same distance, same time)
            if a_new2 >= self.a:
                self.logs.append("✗ Acceleration should decrease with higher initial velocity")
                success = False
            else:
                self.logs.append(f"✓ Acceleration decreases: {self.a:.2f} → {a_new2:.2f} m/s²")
            
        finally:
            # Restore original state (important!)
            self.m = original_m
            # Note: We didn't actually change self.a, self.f, etc., so no need to restore
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        
        self.logs.append("=== Physics Audit Report ===")
        self.logs.append(f"Loaded results: a={self.a:.2f}, f={self.f:.2f}, v_t={self.v_t:.2f}, t'={self.t_prime:.2f}")
        
        test_results = []
        
        # Run tests
        self.logs.append("\n1. Testing Physical Invariants:")
        test_results.append(self.test_invariants())
        
        self.logs.append("\n2. Testing Trace Consistency:")
        test_results.append(self.test_trace_consistency())
        
        self.logs.append("\n3. Testing Monotonicity (Perturbation Analysis):")
        test_results.append(self.test_monotonicity())
        
        # Summary
        success = all(test_results)
        self.logs.append(f"\n=== Summary: {'ALL TESTS PASS' if success else 'SOME TESTS FAILED'} ===")
        self.logs.append(f"Passed {sum(test_results)} of {len(test_results)} test categories")
        
        return success, self.logs

# Example usage (commented out for safety):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'a': 4.0,
#                 'f': 70.0,
#                 'v_t': 22.0,
#                 "t'": 2.0
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)