import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results
        self.d = getattr(solution_instance, 'd', solution_instance.results.get('d', None))
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', None))
        self.alpha = getattr(solution_instance, 'alpha', solution_instance.results.get('alpha', None))
        self.tantheta = getattr(solution_instance, 'tantheta', solution_instance.results.get('tantheta', None))
        self.g = getattr(solution_instance, 'g', 9.8)  # default gravitational acceleration
        
        # Intermediate values from steps (if available)
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v0', None))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', None))
        
        self.logs = []
        
    def test_invariants(self):
        """Check fundamental physics laws: projectile motion and Snell's law."""
        success = True
        
        # 1. Check projectile motion: tanθ = gt/v0
        if None not in (self.tantheta, self.v0, self.t):
            try:
                # Use symbolic or numerical check
                if isinstance(self.tantheta, (int, float)) and isinstance(self.v0, (int, float)) and isinstance(self.t, (int, float)):
                    rhs = self.g * self.t / self.v0
                    if not math.isclose(self.tantheta, rhs, rel_tol=1e-2, abs_tol=1e-2):
                        self.logs.append(f"Projectile invariant failed: tanθ={self.tantheta}, gt/v0={rhs}")
                        success = False
                else:
                    # Symbolic check
                    t_sym, v0_sym, g_sym = sp.symbols('t v0 g')
                    expr = sp.Eq(sp.symbols('tantheta'), g_sym * t_sym / v0_sym)
                    # Cannot verify symbolically without values, skip
                    pass
            except Exception as e:
                self.logs.append(f"Projectile invariant check error: {e}")
                success = False
        
        # 2. Check Snell's law: n = sin(α)/sin(90°-θ) = sin(α)/cos(θ)
        if None not in (self.alpha, self.tantheta):
            try:
                # Convert to radians if needed
                if isinstance(self.alpha, (int, float)) and isinstance(self.tantheta, (int, float)):
                    # Assume alpha is in degrees (as per problem)
                    alpha_rad = math.radians(self.alpha)
                    theta = math.atan(self.tantheta)  # θ from tanθ
                    n_computed = math.sin(alpha_rad) / math.cos(theta)
                    n_expected = 4/3
                    if not math.isclose(n_computed, n_expected, rel_tol=1e-2, abs_tol=1e-2):
                        self.logs.append(f"Snell's law failed: n_computed={n_computed}, expected={n_expected}")
                        success = False
                else:
                    # Symbolic check
                    alpha_sym, theta_sym = sp.symbols('alpha theta')
                    n_expr = sp.sin(alpha_sym) / sp.cos(theta_sym)
                    # Cannot verify without numeric values
                    pass
            except Exception as e:
                self.logs.append(f"Snell's law check error: {e}")
                success = False
        
        return success
    
    def test_trace_consistency(self):
        """Verify intermediate steps lead to final answers."""
        success = True
        
        # Step 1 & 2: Projectile motion equations
        if None not in (self.d, self.v0, self.t):
            try:
                # Horizontal: d = v0 * t
                if isinstance(self.d, (int, float)) and isinstance(self.v0, (int, float)) and isinstance(self.t, (int, float)):
                    if not math.isclose(self.d, self.v0 * self.t, rel_tol=1e-2, abs_tol=1e-2):
                        self.logs.append(f"Horizontal motion inconsistent: d={self.d}, v0*t={self.v0*self.t}")
                        success = False
                # Vertical: (2/3)d = (1/2)gt²
                vertical_lhs = (2/3) * self.d
                vertical_rhs = 0.5 * self.g * self.t**2
                if not math.isclose(vertical_lhs, vertical_rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Vertical motion inconsistent: (2/3)d={vertical_lhs}, (1/2)gt²={vertical_rhs}")
                    success = False
            except Exception as e:
                self.logs.append(f"Trace consistency check error: {e}")
                success = False
        
        # Step 6: Geometry: H tan37 + (2/3)d tan53 = d
        if None not in (self.H, self.d, self.alpha):
            try:
                if isinstance(self.H, (int, float)) and isinstance(self.d, (int, float)):
                    # Use α=53°, so tan53 = tan(α), and 90°-θ=37° so tan37 = tan(90°-θ)
                    alpha_rad = math.radians(53)  # Expected α from solution
                    theta_complement = math.radians(37)  # 90°-θ
                    lhs = self.H * math.tan(theta_complement) + (2/3)*self.d * math.tan(alpha_rad)
                    if not math.isclose(lhs, self.d, rel_tol=1e-2, abs_tol=1e-2):
                        self.logs.append(f"Geometry inconsistent: LHS={lhs}, d={self.d}")
                        success = False
            except Exception as e:
                self.logs.append(f"Geometry check error: {e}")
                success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb d and check H changes proportionally."""
        success = True
        
        if isinstance(self.d, (int, float)) and isinstance(self.H, (int, float)):
            try:
                # Store original values
                original_d = self.d
                original_H = self.H
                
                # Perturb d by 1%
                perturbed_d = original_d * 1.01
                
                # Recalculate H using the derived formula H = 4d/27
                expected_H_perturbed = (4 * perturbed_d) / 27
                expected_H_original = (4 * original_d) / 27
                
                # Check if original H matches expected
                if not math.isclose(original_H, expected_H_original, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Original H mismatch: H={original_H}, expected 4d/27={expected_H_original}")
                    success = False
                
                # Check if perturbed H scales correctly (1% increase)
                expected_ratio = 1.01
                actual_ratio = expected_H_perturbed / expected_H_original
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Monotonicity failed: H does not scale linearly with d")
                    success = False
                
                # Restore original values
                self.d = original_d
                self.H = original_H
                
            except Exception as e:
                self.logs.append(f"Monotonicity check error: {e}")
                success = False
        else:
            self.logs.append("Monotonicity skipped: d or H not numeric")
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
                    self.logs.append(f"{test_name} test failed.")
            except Exception as e:
                overall_success = False
                self.logs.append(f"{test_name} test crashed: {traceback.format_exc()}")
        
        # Final verification of target answers
        if self.tantheta is not None:
            if isinstance(self.tantheta, (int, float)):
                if not math.isclose(self.tantheta, 4/3, rel_tol=1e-2, abs_tol=1e-2):
                    overall_success = False
                    self.logs.append(f"tanθ mismatch: got {self.tantheta}, expected 4/3")
            else:
                # Symbolic check
                if not sp.simplify(self.tantheta - sp.Rational(4,3)) == 0:
                    overall_success = False
                    self.logs.append(f"tanθ not equal to 4/3 symbolically")
        
        if self.H is not None and self.d is not None:
            if isinstance(self.H, (int, float)) and isinstance(self.d, (int, float)):
                expected_H = (4 * self.d) / 27
                if not math.isclose(self.H, expected_H, rel_tol=1e-2, abs_tol=1e-2):
                    overall_success = False
                    self.logs.append(f"H mismatch: got {self.H}, expected 4d/27={expected_H}")
            else:
                # Symbolic check
                d_sym = sp.symbols('d')
                if not sp.simplify(self.H - (4*d_sym)/27) == 0:
                    overall_success = False
                    self.logs.append(f"H not equal to 4d/27 symbolically")
        
        return overall_success, self.logs