import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access. Extract inputs and results.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Safe extraction of results with fallbacks
        self.F_max = getattr(solution_instance, 'F_max', self.results.get('F_max', None))
        self.F_min = getattr(solution_instance, 'F_min', self.results.get('F_min', None))
        self.F = getattr(solution_instance, 'F', self.results.get('F', None))
        
        # Input parameters (with defaults from problem statement)
        self.m = self.inputs.get('m', 5.0)
        self.mu = self.inputs.get('mu', 0.25)
        self.g = self.inputs.get('g', 10.0)
        self.theta_deg = self.inputs.get('theta_deg', 37.0)
        self.theta_rad = math.radians(self.theta_deg)
        self.sin_theta = self.inputs.get('sin_theta', 0.6)
        self.cos_theta = self.inputs.get('cos_theta', 0.8)
        
        # Expected values for comparison (computed from formulas)
        self.F_no_friction_expected = self.m * self.g * self.sin_theta / self.cos_theta  # 37.5 N
        numerator_min = self.m * self.g * (self.sin_theta - self.mu * self.cos_theta)
        denominator_min = self.cos_theta + self.mu * self.sin_theta
        self.F_min_expected = numerator_min / denominator_min  # 400/19 ≈ 21.0526
        numerator_max = self.m * self.g * (self.sin_theta + self.mu * self.cos_theta)
        denominator_max = self.cos_theta - self.mu * self.sin_theta
        self.F_max_expected = numerator_max / denominator_max  # 800/13 ≈ 61.5385
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
    
    def test_invariants(self):
        """
        Verify force equilibrium and friction laws.
        """
        try:
            # Test 1: No-friction case (if F is provided)
            if self.F is not None:
                F_no_friction_eq = sp.Eq(
                    sp.Symbol('F') * self.cos_theta,
                    self.m * self.g * self.sin_theta
                )
                expected_no_friction = self.F_no_friction_expected
                if not math.isclose(self.F, expected_no_friction, 
                                  rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"FAIL: No-friction F mismatch. Got {self.F}, expected {expected_no_friction}")
                    return False
                self.log("PASS: No-friction force matches.")
            
            # Test 2: Minimum force equilibrium (friction up incline)
            if self.F_min is not None:
                # Symbolic check
                F, N, f = sp.symbols('F N f')
                eq1 = sp.Eq(F * self.cos_theta + f, self.m * self.g * self.sin_theta)  # Horizontal
                eq2 = sp.Eq(F * self.sin_theta + self.m * self.g * self.cos_theta, N)  # Vertical
                eq3 = sp.Eq(f, self.mu * N)  # Friction relation
                # Solve symbolically
                sol = sp.solve([eq1, eq2, eq3], [F, N, f], dict=True)
                if sol:
                    F_min_symbolic = sol[0][F]
                    F_min_computed = float(F_min_symbolic.subs({
                        sp.Symbol('m'): self.m,
                        sp.Symbol('g'): self.g,
                        sp.Symbol('mu'): self.mu
                    }))
                    if not math.isclose(self.F_min, F_min_computed, 
                                      rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                        self.log(f"FAIL: F_min symbolic mismatch. Got {self.F_min}, expected {F_min_computed}")
                        return False
                # Numerical fallback
                if not math.isclose(self.F_min, self.F_min_expected,
                                  rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"FAIL: F_min numerical mismatch. Got {self.F_min}, expected {self.F_min_expected}")
                    return False
                self.log("PASS: F_min satisfies equilibrium with friction up.")
            
            # Test 3: Maximum force equilibrium (friction down incline)
            if self.F_max is not None:
                # Symbolic check
                F, N, f = sp.symbols('F N f')
                eq1 = sp.Eq(F * self.cos_theta, f + self.m * self.g * self.sin_theta)  # Horizontal
                eq2 = sp.Eq(F * self.sin_theta + self.m * self.g * self.cos_theta, N)  # Vertical
                eq3 = sp.Eq(f, self.mu * N)  # Friction relation
                sol = sp.solve([eq1, eq2, eq3], [F, N, f], dict=True)
                if sol:
                    F_max_symbolic = sol[0][F]
                    F_max_computed = float(F_max_symbolic.subs({
                        sp.Symbol('m'): self.m,
                        sp.Symbol('g'): self.g,
                        sp.Symbol('mu'): self.mu
                    }))
                    if not math.isclose(self.F_max, F_max_computed,
                                      rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                        self.log(f"FAIL: F_max symbolic mismatch. Got {self.F_max}, expected {F_max_computed}")
                        return False
                # Numerical fallback
                if not math.isclose(self.F_max, self.F_max_expected,
                                  rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"FAIL: F_max numerical mismatch. Got {self.F_max}, expected {self.F_max_expected}")
                    return False
                self.log("PASS: F_max satisfies equilibrium with friction down.")
            
            # Test 4: Range ordering
            if self.F_min is not None and self.F_max is not None:
                if self.F_min >= self.F_max:
                    self.log(f"FAIL: F_min ({self.F_min}) >= F_max ({self.F_max})")
                    return False
                if not math.isclose(self.F_no_friction_expected, (self.F_min + self.F_max)/2,
                                  rel_tol=0.1, abs_tol=1.0):  # Rough symmetry check
                    self.log("WARN: No-friction F not near midpoint of range (within 10%).")
                else:
                    self.log("PASS: Range ordering and approximate symmetry.")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate steps with final results.
        """
        try:
            # Check that F_min matches formula from step 4
            if self.F_min is not None:
                numerator = self.m * self.g * (self.sin_theta - self.mu * self.cos_theta)
                denominator = self.cos_theta + self.mu * self.sin_theta
                computed = numerator / denominator
                if not math.isclose(self.F_min, computed,
                                  rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"FAIL: F_min formula mismatch. Got {self.F_min}, computed {computed}")
                    return False
            
            # Check that F_max matches formula from step 7
            if self.F_max is not None:
                numerator = self.m * self.g * (self.sin_theta + self.mu * self.cos_theta)
                denominator = self.cos_theta - self.mu * self.sin_theta
                computed = numerator / denominator
                if not math.isclose(self.F_max, computed,
                                  rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"FAIL: F_max formula mismatch. Got {self.F_max}, computed {computed}")
                    return False
            
            self.log("PASS: Trace consistency (formulas match).")
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """
        Perturb mass and check expected trends.
        """
        try:
            original_m = self.m
            perturbed_m = original_m * 1.1  # Increase by 10%
            
            # Compute expected F_min and F_max with perturbed mass
            numerator_min = perturbed_m * self.g * (self.sin_theta - self.mu * self.cos_theta)
            denominator_min = self.cos_theta + self.mu * self.sin_theta
            F_min_perturbed = numerator_min / denominator_min
            
            numerator_max = perturbed_m * self.g * (self.sin_theta + self.mu * self.cos_theta)
            denominator_max = self.cos_theta - self.mu * self.sin_theta
            F_max_perturbed = numerator_max / denominator_max
            
            # Check monotonicity: increasing mass should increase both bounds
            if self.F_min is not None and not (F_min_perturbed > self.F_min):
                self.log(f"FAIL: F_min not monotonic with mass. Original {self.F_min}, perturbed {F_min_perturbed}")
                return False
            if self.F_max is not None and not (F_max_perturbed > self.F_max):
                self.log(f"FAIL: F_max not monotonic with mass. Original {self.F_max}, perturbed {F_max_perturbed}")
                return False
            
            # Restore original mass (not strictly necessary, but good practice)
            self.m = original_m
            self.log("PASS: Monotonicity with mass perturbation.")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} Test ---")
            if not test_func():
                success = False
                self.log(f"{test_name} FAILED.")
            else:
                self.log(f"{test_name} PASSED.")
        
        return success, self.logs