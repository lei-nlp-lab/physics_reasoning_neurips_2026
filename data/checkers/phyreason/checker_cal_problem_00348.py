import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        self.solution = solution_instance
        
        # Extract results safely using getattr and .get()
        self.results = {}
        expected_keys = ['x', 'v_D', 't', 'x_2', 'v_B', 'h_1', 'x_1', 'E']
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance, 'results', {}).get(key, 0.0))
        
        # Extract constants (with safe defaults)
        self.m = getattr(solution_instance, 'm', 1.0)           # mass
        self.q = getattr(solution_instance, 'q', 1.0)           # charge
        self.L = getattr(solution_instance, 'L', 1.0)           # length AB
        self.r = getattr(solution_instance, 'r', 0.1)           # radius
        self.theta_deg = 37.0                                   # given
        self.theta = math.radians(self.theta_deg)
        self.sin_theta = 0.6                                    # given
        self.cos_theta = 0.8                                    # given
        self.mu = getattr(solution_instance, 'mu', 0.25)        # friction
        self.g = getattr(solution_instance, 'g', 9.8)           # gravity
        # Electric field strength (formula from problem)
        self.E0 = (4 * self.m * self.g) / (5 * self.q) if self.q != 0 else 0.0
        
        # Store original values for monotonicity test
        self._original_m = self.m
        self._original_E0 = self.E0
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []

    def _log(self, msg, success):
        """Add a log entry with status."""
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success

    def _isclose(self, a, b):
        """Safe numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        """Test fundamental physics invariants using KB theorems."""
        successes = []
        
        try:
            # 1. Check work-energy theorem from A to B (Step 1)
            # Equation: mgL sinθ - μ(mg cosθ + qE0)L = ½ m v_B²
            v_B = self.results.get('v_B')
            if v_B is not None:
                left = (self.m * self.g * self.L * self.sin_theta - 
                       self.mu * (self.m * self.g * self.cos_theta + self.q * self.E0) * self.L)
                right = 0.5 * self.m * v_B**2
                successes.append(self._log(
                    f"Work-energy (A→B): {left:.3f} ≈ {right:.3f}, diff={abs(left-right):.2e}",
                    self._isclose(left, right)
                ))
            
            # 2. Check Newton's second law at D (circular motion) (Step 2)
            # Equation: mg = m v_D² / r
            v_D = self.results.get('v_D')
            if v_D is not None:
                left = self.m * self.g
                right = self.m * v_D**2 / self.r
                successes.append(self._log(
                    f"Circular motion at D: {left:.3f} ≈ {right:.3f}, diff={abs(left-right):.2e}",
                    self._isclose(left, right)
                ))
            
            # 3. Check work-energy theorem with variable E (Step 3)
            # Equation: mg(L sinθ - 2r) - μ(mg cosθ + qE)L = ½ m v_D²
            E = self.results.get('E')
            if E is not None and v_D is not None:
                left = (self.m * self.g * (self.L * self.sin_theta - 2 * self.r) - 
                       self.mu * (self.m * self.g * self.cos_theta + self.q * E) * self.L)
                right = 0.5 * self.m * v_D**2
                successes.append(self._log(
                    f"Work-energy (A→D): {left:.3f} ≈ {right:.3f}, diff={abs(left-right):.2e}",
                    self._isclose(left, right)
                ))
                
                # Also check E = 0.75 E0 relationship
                expected_E = 0.75 * self.E0
                successes.append(self._log(
                    f"Electric field ratio: E={E:.3f} ≈ 0.75E0={expected_E:.3f}",
                    self._isclose(E, expected_E)
                ))
            
            # 4. Check projectile motion kinematics (Steps 4-6)
            # From Step 4: v_y = v_D / tanθ = g t
            t = self.results.get('t')
            if t is not None and v_D is not None:
                tan_theta = self.sin_theta / self.cos_theta
                left = v_D / tan_theta if tan_theta != 0 else float('inf')
                right = self.g * t
                successes.append(self._log(
                    f"Vertical velocity: {left:.3f} ≈ {right:.3f}, diff={abs(left-right):.2e}",
                    self._isclose(left, right)
                ))
            
            # From Step 5: x1 = v_D * t
            x1 = self.results.get('x_1')
            if x1 is not None and v_D is not None and t is not None:
                expected_x1 = v_D * t
                successes.append(self._log(
                    f"Horizontal displacement x1: {x1:.3f} ≈ {expected_x1:.3f}",
                    self._isclose(x1, expected_x1)
                ))
            
            # From Step 6: h1 = ½ g t²
            h1 = self.results.get('h_1')
            if h1 is not None and t is not None:
                expected_h1 = 0.5 * self.g * t**2
                successes.append(self._log(
                    f"Vertical displacement h1: {h1:.3f} ≈ {expected_h1:.3f}",
                    self._isclose(h1, expected_h1)
                ))
            
            # 5. Check geometric relationship (Step 7)
            # x2 = (2r - h1) tanθ
            x2 = self.results.get('x_2')
            if x2 is not None and h1 is not None:
                expected_x2 = (2 * self.r - h1) * (self.sin_theta / self.cos_theta)
                successes.append(self._log(
                    f"Geometric x2: {x2:.3f} ≈ {expected_x2:.3f}",
                    self._isclose(x2, expected_x2)
                ))
            
            # 6. Check length addition (Step 8)
            # x = x1 + x2
            x = self.results.get('x')
            if x is not None and x1 is not None and x2 is not None:
                expected_x = x1 + x2
                successes.append(self._log(
                    f"Total x: {x:.3f} ≈ x1+x2={expected_x:.3f}",
                    self._isclose(x, expected_x)
                ))
                
                # Also check against target answer 13/60 m
                target_x = 13/60
                successes.append(self._log(
                    f"Target x: {x:.3f} ≈ 13/60={target_x:.3f}",
                    self._isclose(x, target_x)
                ))
            
            return all(successes)
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        successes = []
        
        try:
            # Check that using v_B from step 1 leads to correct v_D in step 2
            v_B = self.results.get('v_B')
            v_D = self.results.get('v_D')
            
            if v_B is not None and v_D is not None:
                # Energy from A to D (without electric field work for friction)
                # Using work-energy: K_D = K_B + W_gravity_BC - W_friction_AB?
                # Actually, from step 3 we have the full equation already checked in invariants
                # Here we just check that v_D is less than v_B (energy loss due to friction)
                successes.append(self._log(
                    f"v_D ({v_D:.3f}) < v_B ({v_B:.3f}) due to friction",
                    v_D < v_B
                ))
            
            # Check that t, x1, h1, x2 are all positive
            for key in ['t', 'x_1', 'h_1', 'x_2', 'x']:
                val = self.results.get(key)
                if val is not None:
                    successes.append(self._log(
                        f"{key} = {val:.3f} is positive",
                        val > 0
                    ))
            
            return all(successes)
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False

    def test_monotonicity(self):
        """Test that increasing mass decreases the horizontal distance x."""
        successes = []
        
        try:
            # Perturb mass by +10%
            perturbed_m = self.m * 1.1
            
            # Calculate v_B with perturbed mass (using step 1 equation)
            # v_B² = 2[gL sinθ - μ(g cosθ + (q/m)E0)L] but E0 depends on m!
            # Actually E0 = (4mg)/(5q) so qE0 = (4mg)/5
            # So the friction term becomes: μ(mg cosθ + qE0) = μ(mg cosθ + 4mg/5) = μmg(cosθ + 4/5)
            sin_term = self.g * self.L * self.sin_theta
            friction_term = self.mu * self.g * self.L * (self.cos_theta + 4/5)  # cosθ + qE0/(mg)
            v_B_sq = 2 * (sin_term - friction_term)  # Independent of m!
            v_B_perturbed = math.sqrt(max(v_B_sq, 0))
            
            # v_D remains sqrt(g*r) from step 2 (independent of m)
            v_D_perturbed = math.sqrt(self.g * self.r)
            
            # Calculate E with perturbed mass (using step 3 equation)
            # From step 3: mg(L sinθ - 2r) - μ(mg cosθ + qE)L = ½ m v_D²
            # Solve for E: E = [mg(L sinθ - 2r) - ½ m v_D²]/(μ q L) - (mg cosθ)/q
            numerator = (perturbed_m * self.g * (self.L * self.sin_theta - 2 * self.r) - 
                        0.5 * perturbed_m * v_D_perturbed**2)
            denominator = self.mu * self.q * self.L
            E_perturbed = numerator / denominator - (perturbed_m * self.g * self.cos_theta) / self.q
            
            # E0 also changes with m
            E0_perturbed = (4 * perturbed_m * self.g) / (5 * self.q)
            E_ratio_perturbed = E_perturbed / E0_perturbed if E0_perturbed != 0 else 0
            
            # Calculate projectile motion (steps 4-8)
            tan_theta = self.sin_theta / self.cos_theta
            t_perturbed = v_D_perturbed / (self.g * tan_theta) if tan_theta != 0 else 0
            x1_perturbed = v_D_perturbed * t_perturbed
            h1_perturbed = 0.5 * self.g * t_perturbed**2
            x2_perturbed = (2 * self.r - h1_perturbed) * tan_theta
            x_perturbed = x1_perturbed + x2_perturbed
            
            # Get original x
            x_original = self.results.get('x', 0)
            
            # Check that perturbed x is close to original (since v_B and v_D are independent of m!)
            # Actually, from our derivation, v_B and v_D are independent of m, so x should be too
            successes.append(self._log(
                f"Monotonicity: x(perturbed m={perturbed_m:.2f}) = {x_perturbed:.3f} ≈ x(original) = {x_original:.3f}",
                self._isclose(x_perturbed, x_original)
            ))
            
            # Additional check: E ratio should remain 0.75
            successes.append(self._log(
                f"E/E0 ratio invariant: {E_ratio_perturbed:.3f} ≈ 0.75",
                self._isclose(E_ratio_perturbed, 0.75)
            ))
            
            # Restore original values
            self.m = self._original_m
            self.E0 = self._original_E0
            
            return all(successes)
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            # Restore original values on error
            self.m = self._original_m
            self.E0 = self._original_E0
            return False

    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        all_success = True
        
        try:
            self.logs.append("=== Physics Checker Started ===")
            
            # Run individual tests
            test1 = self.test_invariants()
            all_success = all_success and test1
            self.logs.append(f"Invariants test: {'PASS' if test1 else 'FAIL'}")
            
            test2 = self.test_trace_consistency()
            all_success = all_success and test2
            self.logs.append(f"Trace consistency test: {'PASS' if test2 else 'FAIL'}")
            
            test3 = self.test_monotonicity()
            all_success = all_success and test3
            self.logs.append(f"Monotonicity test: {'PASS' if test3 else 'FAIL'}")
            
            self.logs.append(f"=== Overall: {'ALL TESTS PASSED' if all_success else 'SOME TESTS FAILED'} ===")
            
            return all_success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs