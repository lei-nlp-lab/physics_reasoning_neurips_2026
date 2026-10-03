import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from the solution instance."""
        self.results = {}
        
        # Safe extraction using getattr with fallback to results dictionary
        keys_to_extract = [
            'a_1', 'F', 'F_0', 'F_net2', 'a_2', 'mu', 
            'F_net_up', 'F_net_down', 'F_net1'
        ]
        
        for key in keys_to_extract:
            # Try to get from results first, then from attributes
            value = getattr(solution_instance, key, None)
            if value is None:
                value = solution_instance.results.get(key, None)
            self.results[key] = value
        
        # Extract input parameters with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.theta = getattr(solution_instance, 'theta', math.radians(30))
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Expected values from the problem statement and steps analysis
        self.expected = {
            'a_1': 12.0,
            'a_2': 6.0,
            'F': 18.0,
            'mu': math.sqrt(3) / 15,
            'F_0_up': 9.9,
            'F_0_down': 2.2
        }
        
        # Trigonometric constants
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)
        
        self.logs = []

    def _log(self, message, success=True):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison with 1-2% tolerance."""
        # Handle None values
        if a is None or b is None:
            return False
        # Handle symbolic expressions by converting to float
        if hasattr(a, 'evalf'):
            a = float(a.evalf())
        if hasattr(b, 'evalf'):
            b = float(b.evalf())
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def _check_newton_phase1(self):
        """Check Newton's 2nd Law for phase 1 (with pulling force F)."""
        try:
            F = self.results.get('F')
            mu = self.results.get('mu')
            a1 = self.results.get('a_1')
            
            if None in [F, mu, a1]:
                return self._log("Missing values for phase 1 check", False)
            
            # Symbolic check
            m, g, sinθ, cosθ = sp.symbols('m g sinθ cosθ', positive=True)
            F_sym, mu_sym, a1_sym = sp.symbols('F mu a1')
            
            lhs = F_sym - m*g*sinθ - mu_sym*m*g*cosθ
            rhs = m*a1_sym
            
            # Substitute numerical values
            subs = {
                m: self.m,
                g: self.g,
                sinθ: self.sin_theta,
                cosθ: self.cos_theta,
                F_sym: F,
                mu_sym: mu,
                a1_sym: a1
            }
            
            lhs_val = float(lhs.subs(subs).evalf())
            rhs_val = float(rhs.subs(subs).evalf())
            
            if self._is_close(lhs_val, rhs_val):
                return self._log("Newton's 2nd Law holds for phase 1 (F - mgsinθ - μmgcosθ = ma₁)")
            else:
                return self._log(f"Phase 1 Newton mismatch: {lhs_val} ≠ {rhs_val}", False)
                
        except Exception as e:
            return self._log(f"Phase 1 Newton check error: {str(e)}", False)

    def _check_newton_phase2(self):
        """Check Newton's 2nd Law for phase 2 (after force removal)."""
        try:
            mu = self.results.get('mu')
            a2 = self.results.get('a_2')
            
            if None in [mu, a2]:
                return self._log("Missing values for phase 2 check", False)
            
            # Symbolic check
            m, g, sinθ, cosθ = sp.symbols('m g sinθ cosθ', positive=True)
            mu_sym, a2_sym = sp.symbols('mu a2')
            
            lhs = m*g*sinθ + mu_sym*m*g*cosθ
            rhs = m*a2_sym
            
            subs = {
                m: self.m,
                g: self.g,
                sinθ: self.sin_theta,
                cosθ: self.cos_theta,
                mu_sym: mu,
                a2_sym: a2
            }
            
            lhs_val = float(lhs.subs(subs).evalf())
            rhs_val = float(rhs.subs(subs).evalf())
            
            if self._is_close(lhs_val, rhs_val):
                return self._log("Newton's 2nd Law holds for phase 2 (mgsinθ + μmgcosθ = ma₂)")
            else:
                return self._log(f"Phase 2 Newton mismatch: {lhs_val} ≠ {rhs_val}", False)
                
        except Exception as e:
            return self._log(f"Phase 2 Newton check error: {str(e)}", False)

    def _check_friction_coefficient(self):
        """Verify the friction coefficient calculation."""
        try:
            mu = self.results.get('mu')
            if mu is None:
                return self._log("No friction coefficient found", False)
            
            # Calculate expected μ from a₂
            a2 = self.results.get('a_2', self.expected['a_2'])
            mu_expected = (a2 - self.g * self.sin_theta) / (self.g * self.cos_theta)
            
            if self._is_close(mu, mu_expected):
                return self._log(f"Friction coefficient correct: μ ≈ {mu:.4f}")
            else:
                return self._log(f"Friction coefficient mismatch: {mu} ≠ {mu_expected}", False)
                
        except Exception as e:
            return self._log(f"Friction coefficient check error: {str(e)}", False)

    def _check_horizontal_force(self):
        """Check the horizontal force F₀ calculations for up/down cases."""
        try:
            F0 = self.results.get('F_0')
            mu = self.results.get('mu')
            
            if F0 is None or mu is None:
                return self._log("Missing values for F₀ check", False)
            
            # Handle both single value and list of values
            if isinstance(F0, (list, tuple)):
                F0_values = F0
            else:
                F0_values = [F0]
            
            success = True
            for F0_val in F0_values:
                # Determine which case (up or down) based on closeness to expected values
                if self._is_close(F0_val, self.expected['F_0_up']):
                    case = "upward"
                    # Equation: F₀cosθ - mgsinθ - μ(F₀sinθ + mgcosθ) = ma
                    lhs = (F0_val * self.cos_theta - self.m * self.g * self.sin_theta - 
                           mu * (F0_val * self.sin_theta + self.m * self.g * self.cos_theta))
                    # Solve for acceleration a
                    a = lhs / self.m
                    if self._is_close(a, 2.0):  # From manual calculation
                        self._log(f"Horizontal force {F0_val:.1f}N gives upward a ≈ {a:.1f}m/s²")
                    else:
                        success = self._log(f"F₀={F0_val:.1f}N: a={a:.1f} ≠ 2.0m/s²", False)
                        
                elif self._is_close(F0_val, self.expected['F_0_down']):
                    case = "downward"
                    # Equation: F₀cosθ - mgsinθ + μ(F₀sinθ + mgcosθ) = -ma
                    lhs = (F0_val * self.cos_theta - self.m * self.g * self.sin_theta + 
                           mu * (F0_val * self.sin_theta + self.m * self.g * self.cos_theta))
                    a = -lhs / self.m
                    if self._is_close(a, 2.0):  # From manual calculation
                        self._log(f"Horizontal force {F0_val:.1f}N gives downward a ≈ {a:.1f}m/s²")
                    else:
                        success = self._log(f"F₀={F0_val:.1f}N: a={a:.1f} ≠ 2.0m/s²", False)
                else:
                    success = self._log(f"F₀={F0_val} not close to 9.9N or 2.2N", False)
            
            return success
            
        except Exception as e:
            return self._log(f"Horizontal force check error: {str(e)}", False)

    def test_invariants(self):
        """Test physical invariants (Newton's Laws)."""
        tests = [
            self._check_newton_phase1,
            self._check_newton_phase2,
            self._check_friction_coefficient
        ]
        
        success = all(test() for test in tests)
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            success = True
            
            # Check a₁ and a₂ against expected values
            a1 = self.results.get('a_1')
            a2 = self.results.get('a_2')
            
            if a1 is not None and not self._is_close(a1, self.expected['a_1']):
                success = self._log(f"a₁={a1} ≠ {self.expected['a_1']}", False)
            else:
                self._log(f"a₁={a1} correct")
            
            if a2 is not None and not self._is_close(a2, self.expected['a_2']):
                success = self._log(f"a₂={a2} ≠ {self.expected['a_2']}", False)
            else:
                self._log(f"a₂={a2} correct")
            
            # Check F against expected
            F = self.results.get('F')
            if F is not None and not self._is_close(F, self.expected['F']):
                success = self._log(f"F={F} ≠ {self.expected['F']}", False)
            else:
                self._log(f"F={F} correct")
            
            # Check μ against expected
            mu = self.results.get('mu')
            if mu is not None and not self._is_close(mu, self.expected['mu']):
                success = self._log(f"μ={mu} ≠ {self.expected['mu']:.4f}", False)
            else:
                self._log(f"μ={mu} correct")
            
            return success
            
        except Exception as e:
            return self._log(f"Trace consistency error: {str(e)}", False)

    def test_monotonicity(self):
        """Test mass perturbation trend."""
        try:
            original_m = self.m
            perturbed_m = original_m * 1.1  # 10% increase
            
            # Get current results
            a1 = self.results.get('a_1', self.expected['a_1'])
            a2 = self.results.get('a_2', self.expected['a_2'])
            mu = self.results.get('mu', self.expected['mu'])
            
            # Calculate F for original mass (from phase 1 equation)
            F_original = (original_m * a1 + original_m * self.g * self.sin_theta + 
                         mu * original_m * self.g * self.cos_theta)
            
            # Calculate F for perturbed mass
            F_perturbed = (perturbed_m * a1 + perturbed_m * self.g * self.sin_theta + 
                          mu * perturbed_m * self.g * self.cos_theta)
            
            # Check linear proportionality: F_perturbed/F_original ≈ perturbed_m/original_m
            ratio_F = F_perturbed / F_original
            ratio_m = perturbed_m / original_m
            
            if self._is_close(ratio_F, ratio_m):
                self._log(f"Mass perturbation: F ∝ m holds (ratio_F={ratio_F:.3f} ≈ ratio_m={ratio_m:.3f})")
                return True
            else:
                return self._log(f"Mass perturbation failed: {ratio_F:.3f} ≠ {ratio_m:.3f}", False)
                
        except Exception as e:
            return self._log(f"Monotonicity test error: {str(e)}", False)

    def run_all_tests(self):
        """Execute all tests and return results."""
        self.logs = []
        
        test_suites = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Horizontal Force", self._check_horizontal_force),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_success = True
        for name, test_func in test_suites:
            self.logs.append(f"\n--- Testing {name} ---")
            try:
                if not test_func():
                    all_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {name}: {traceback.format_exc()}")
                all_success = False
        
        return all_success, self.logs