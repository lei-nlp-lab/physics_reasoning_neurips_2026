import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of problem parameters
        self.d = getattr(solution_instance, 'd', getattr(solution_instance, 'width', 180.0))
        self.v1 = getattr(solution_instance, 'v1', getattr(solution_instance, 'current_speed', 2.5))
        self.v2 = getattr(solution_instance, 'v2', getattr(solution_instance, 'boat_speed', 5.0))
        
        # Safe extraction of results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.t = self.results.get('t', 0.0)  # Minimum time crossing
        self.x = self.results.get('x', 0.0)  # Displacement with minimum time
        self.t_prime = self.results.get('t\'', 0.0)  # Time for minimum distance
        self.x_prime = self.results.get('x\'', 0.0)  # Displacement with minimum distance
        self.v = self.results.get('v', 0.0)  # Resultant velocity
        self.alpha_deg = self.results.get('alpha', 0.0)  # Angle in degrees
        
        # Derived quantities
        self.alpha_rad = math.radians(self.alpha_deg) if self.alpha_deg else 0.0
        
        # Expected values from problem analysis
        self.expected_t_min = self.d / self.v2  # 36 s
        self.expected_v = math.sqrt(self.v1**2 + self.v2**2)  # ~5.59 m/s
        self.expected_x_min_time = self.expected_v * self.expected_t_min  # ~201.2 m
        self.expected_alpha = math.degrees(math.asin(self.v1/self.v2))  # 30°
        self.expected_t_prime = self.d / (self.v2 * math.cos(math.radians(self.expected_alpha)))  # ~41.57 s
        
        # Symbolic setup for physics laws
        self.symbols = sp.symbols('d v1 v2 t x t_prime x_prime v alpha')
        self.setup_symbolic_laws()
        
        self.logs = []
    
    def setup_symbolic_laws(self):
        """Define symbolic physics laws for verification"""
        d, v1, v2, t, x, t_prime, x_prime, v, alpha = self.symbols
        
        # Kinematics laws for minimum time crossing
        self.law_min_time = [
            sp.Eq(t, d / v2),  # Time to cross perpendicularly
            sp.Eq(v, sp.sqrt(v1**2 + v2**2)),  # Resultant velocity
            sp.Eq(x, v * t)  # Displacement
        ]
        
        # Laws for minimum distance crossing
        self.law_min_distance = [
            sp.Eq(sp.sin(alpha), v1/v2),  # Angle determination
            sp.Eq(t_prime, d / (v2 * sp.cos(alpha))),  # Time to cross
            sp.Eq(x_prime, d)  # Displacement equals width
        ]
    
    def fuzzy_equal(self, a, b, msg=""):
        """Fuzzy comparison with physics tolerance"""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            result = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
            self.logs.append(f"{msg}: {a} ≈ {b} -> {result}")
            return result
        return a == b
    
    def test_invariants(self):
        """Verify fundamental physics invariants"""
        checks = []
        
        # 1. Check minimum time crossing physics
        # Time should be width / perpendicular component
        checks.append((
            self.fuzzy_equal(self.t, self.expected_t_min, 
                           f"Minimum time t={self.t} vs d/v2={self.expected_t_min}"),
            "Minimum time crossing: t = d/v₂"
        ))
        
        # 2. Check resultant velocity Pythagorean theorem
        checks.append((
            self.fuzzy_equal(self.v, self.expected_v,
                           f"Resultant velocity v={self.v} vs √(v₁²+v₂²)={self.expected_v}"),
            "Velocity addition: v = √(v₁² + v₂²)"
        ))
        
        # 3. Check displacement for minimum time
        checks.append((
            self.fuzzy_equal(self.x, self.expected_x_min_time,
                           f"Displacement x={self.x} vs v*t={self.expected_x_min_time}"),
            "Displacement with minimum time: x = v * t"
        ))
        
        # 4. Check angle for minimum distance
        checks.append((
            self.fuzzy_equal(self.alpha_deg, self.expected_alpha,
                           f"Angle α={self.alpha_deg}° vs arcsin(v1/v2)={self.expected_alpha}°"),
            "Heading angle: sin(α) = v₁/v₂"
        ))
        
        # 5. Check time for minimum distance
        checks.append((
            self.fuzzy_equal(self.t_prime, self.expected_t_prime,
                           f"Time t'={self.t_prime} vs d/(v₂·cosα)={self.expected_t_prime}"),
            "Time for minimum distance: t' = d/(v₂·cosα)"
        ))
        
        # 6. Check displacement equals width for minimum distance
        checks.append((
            self.fuzzy_equal(self.x_prime, self.d,
                           f"Displacement x'={self.x_prime} vs width d={self.d}"),
            "Minimum distance crossing: x' = d"
        ))
        
        # Compile results
        success = all(check[0] for check in checks)
        for passed, msg in checks:
            if not passed:
                self.logs.append(f"FAILED: {msg}")
        
        self.logs.append(f"test_invariants: {'PASSED' if success else 'FAILED'}")
        return success
    
    def test_trace_consistency(self):
        """Verify intermediate results are consistent with final answers"""
        checks = []
        
        # Check that the two crossing strategies give different but consistent results
        if self.t > 0 and self.t_prime > 0:
            # Minimum time should be less than minimum distance time
            checks.append(self.t < self.t_prime)
            self.logs.append(f"t={self.t} < t'={self.t_prime}: {self.t < self.t_prime}")
        
        # Check displacement relationship
        if self.x > 0 and self.x_prime > 0:
            # Minimum time gives longer path (diagonal)
            checks.append(self.x > self.x_prime)
            self.logs.append(f"x={self.x} > x'={self.x_prime}: {self.x > self.x_prime}")
        
        # Verify angle is acute
        checks.append(0 <= self.alpha_deg <= 90)
        self.logs.append(f"0 ≤ α={self.alpha_deg}° ≤ 90°: {0 <= self.alpha_deg <= 90}")
        
        # Check velocity triangle consistency
        if self.v > 0 and self.alpha_deg > 0:
            v_perpendicular = self.v2 * math.cos(self.alpha_rad)
            checks.append(self.fuzzy_equal(v_perpendicular * self.t_prime, self.d,
                                          f"Perpendicular distance: {v_perpendicular*self.t_prime} ≈ {self.d}"))
        
        success = all(checks)
        self.logs.append(f"test_trace_consistency: {'PASSED' if success else 'FAILED'}")
        return success
    
    def test_monotonicity(self):
        """Check that perturbations produce expected trends"""
        original_v1 = self.v1
        perturbation_factor = 1.1  # 10% increase
        
        try:
            # Perturb current speed
            self.v1 *= perturbation_factor
            
            # Recalculate expected values with perturbed v1
            new_expected_v = math.sqrt(self.v1**2 + self.v2**2)
            new_expected_alpha = math.degrees(math.asin(self.v1/self.v2))
            new_expected_t_prime = self.d / (self.v2 * math.cos(math.radians(new_expected_alpha)))
            
            # Check monotonic trends
            checks = []
            
            # Resultant velocity increases with v1
            checks.append(new_expected_v > self.expected_v)
            self.logs.append(f"v increases with v1: {new_expected_v} > {self.expected_v}")
            
            # Angle increases with v1 (more upstream deflection needed)
            checks.append(new_expected_alpha > self.expected_alpha)
            self.logs.append(f"α increases with v1: {new_expected_alpha}° > {self.expected_alpha}°")
            
            # Time for minimum distance increases with v1
            checks.append(new_expected_t_prime > self.expected_t_prime)
            self.logs.append(f"t' increases with v1: {new_expected_t_prime} > {self.expected_t_prime}")
            
            success = all(checks)
            
        finally:
            # Restore original value
            self.v1 = original_v1
        
        self.logs.append(f"test_monotonicity: {'PASSED' if success else 'FAILED'}")
        return success
    
    def test_symbolic_consistency(self):
        """Verify results satisfy symbolic physics laws"""
        checks = []
        
        # Substitute numerical values into symbolic laws
        subs_dict = {
            self.symbols[0]: self.d,
            self.symbols[1]: self.v1,
            self.symbols[2]: self.v2,
            self.symbols[3]: self.t,
            self.symbols[4]: self.x,
            self.symbols[5]: self.t_prime,
            self.symbols[6]: self.x_prime,
            self.symbols[7]: self.v,
            self.symbols[8]: self.alpha_rad
        }
        
        # Check minimum time laws
        for law in self.law_min_time:
            lhs = law.lhs.subs(subs_dict)
            rhs = law.rhs.subs(subs_dict)
            checks.append(self.fuzzy_equal(float(lhs), float(rhs),
                                          f"Symbolic law: {law}"))
        
        # Check minimum distance laws
        for law in self.law_min_distance:
            lhs = law.lhs.subs(subs_dict)
            rhs = law.rhs.subs(subs_dict)
            checks.append(self.fuzzy_equal(float(lhs), float(rhs),
                                          f"Symbolic law: {law}"))
        
        success = all(checks)
        self.logs.append(f"test_symbolic_consistency: {'PASSED' if success else 'FAILED'}")
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = ["PhysicsChecker running..."]
        
        test_results = []
        try:
            test_results.append(("invariants", self.test_invariants()))
            test_results.append(("trace_consistency", self.test_trace_consistency()))
            test_results.append(("monotonicity", self.test_monotonicity()))
            test_results.append(("symbolic_consistency", self.test_symbolic_consistency()))
        except Exception as e:
            self.logs.append(f"ERROR during testing: {str(e)}")
            self.logs.append(traceback.format_exc())
            return False, self.logs
        
        success = all(result for _, result in test_results)
        
        # Summary
        self.logs.append("\n=== SUMMARY ===")
        for test_name, passed in test_results:
            self.logs.append(f"{test_name}: {'✓' if passed else '✗'}")
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs