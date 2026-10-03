import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 2.0))
        self.k = getattr(solution_instance, 'k', getattr(solution_instance, 'inputs', {}).get('k', 200.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        self.L0 = getattr(solution_instance, 'L0', getattr(solution_instance, 'inputs', {}).get('L0', 0.08))
        self.L1 = getattr(solution_instance, 'L1', getattr(solution_instance, 'inputs', {}).get('L1', 0.11))
        self.L2 = getattr(solution_instance, 'L2', getattr(solution_instance, 'inputs', {}).get('L2', 0.105))

        # Safe attribute access for results with fallback values
        self.F_max = getattr(solution_instance, 'F_max', solution_instance.results.get('F_max', None))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', None))
        self.mu = getattr(solution_instance, 'mu', solution_instance.results.get('mu', None))
        
        # Expected values from direct calculation (for cross-check)
        self.dx1 = self.L1 - self.L0  # displacement at motion start
        self.dx2 = self.L2 - self.L0  # displacement at constant velocity
        self.F_max_expected = self.k * self.dx1
        self.f_expected = self.k * self.dx2
        self.mu_expected = self.f_expected / (self.m * self.g)
        
        self.logs = []
    
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        try:
            # 1. Static equilibrium at motion start: F_spring = F_max (static friction max)
            F_spring_start = self.k * self.dx1
            if self.F_max is not None:
                if not math.isclose(F_spring_start, self.F_max, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: F_spring_start {F_spring_start:.2f} ≠ F_max {self.F_max:.2f}")
                    return False
                else:
                    self.log(f"PASS: Static equilibrium F_spring = F_max = {F_spring_start:.2f} N")
            else:
                self.log("WARN: F_max not provided; skipping static equilibrium check")
            
            # 2. Dynamic equilibrium at constant velocity: F_spring = kinetic friction f
            F_spring_vel = self.k * self.dx2
            if self.f is not None:
                if not math.isclose(F_spring_vel, self.f, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: F_spring_vel {F_spring_vel:.2f} ≠ f {self.f:.2f}")
                    return False
                else:
                    self.log(f"PASS: Dynamic equilibrium F_spring = f = {F_spring_vel:.2f} N")
            else:
                self.log("WARN: f not provided; skipping dynamic equilibrium check")
            
            # 3. Friction force definition: f = μ * N
            N = self.m * self.g
            if self.mu is not None and self.f is not None:
                f_from_mu = self.mu * N
                if not math.isclose(f_from_mu, self.f, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: μ*N = {f_from_mu:.2f} ≠ f {self.f:.2f}")
                    return False
                else:
                    self.log(f"PASS: Friction definition f = μ*N = {f_from_mu:.2f} N")
            else:
                self.log("WARN: Missing mu or f; skipping friction definition check")
            
            # 4. Static friction ≥ kinetic friction (physical consistency)
            if self.F_max is not None and self.f is not None:
                if self.F_max < self.f - 1e-2:  # Allow tolerance
                    self.log(f"FAIL: Static friction {self.F_max:.2f} < kinetic friction {self.f:.2f}")
                    return False
                else:
                    self.log(f"PASS: Static friction {self.F_max:.2f} ≥ kinetic friction {self.f:.2f}")
            else:
                self.log("WARN: Missing F_max or f; skipping friction inequality check")
            
            # 5. Symbolic Hooke's Law check
            x = sp.symbols('x')
            F_spring_sym = -self.k * x  # Hooke's Law
            # At x = dx1, force magnitude should be F_max
            if self.F_max is not None:
                F_spring_val = abs(F_spring_sym.subs(x, self.dx1))
                if not math.isclose(float(F_spring_val), self.F_max, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Symbolic Hooke's Law gives {float(F_spring_val):.2f}, expected {self.F_max:.2f}")
                    return False
                else:
                    self.log("PASS: Symbolic Hooke's Law consistent")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        try:
            # Check that results match direct calculations from inputs
            checks_passed = True
            
            if self.F_max is not None:
                if not math.isclose(self.F_max, self.F_max_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: F_max {self.F_max:.2f} ≠ expected {self.F_max_expected:.2f}")
                    checks_passed = False
                else:
                    self.log(f"PASS: F_max matches expected {self.F_max_expected:.2f} N")
            
            if self.f is not None:
                if not math.isclose(self.f, self.f_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: f {self.f:.2f} ≠ expected {self.f_expected:.2f}")
                    checks_passed = False
                else:
                    self.log(f"PASS: f matches expected {self.f_expected:.2f} N")
            
            if self.mu is not None:
                if not math.isclose(self.mu, self.mu_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: μ {self.mu:.3f} ≠ expected {self.mu_expected:.3f}")
                    checks_passed = False
                else:
                    self.log(f"PASS: μ matches expected {self.mu_expected:.3f}")
            
            # Cross-check between steps: μ should be f/(m*g)
            if self.f is not None and self.mu is not None:
                mu_calc = self.f / (self.m * self.g)
                if not math.isclose(mu_calc, self.mu, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: μ from f/(m*g) = {mu_calc:.3f} ≠ reported μ {self.mu:.3f}")
                    checks_passed = False
                else:
                    self.log("PASS: μ consistent with f and m*g")
            
            return checks_passed
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        original_m = self.m
        original_results = (self.F_max, self.f, self.mu)
        
        try:
            # Perturb mass upward by 10%
            self.m = original_m * 1.1
            self.log(f"Testing monotonicity: perturbed m to {self.m:.2f} kg")
            
            # Recalculate expected values with perturbed mass
            # F_max and f depend only on spring, not mass, so they should NOT change
            F_max_new = self.k * self.dx1
            f_new = self.k * self.dx2
            mu_new = f_new / (self.m * self.g)
            
            # Check 1: F_max unchanged (independent of mass)
            if not math.isclose(F_max_new, self.F_max_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: F_max changed with mass perturbation - not independent")
                self.m = original_m
                return False
            else:
                self.log("PASS: F_max independent of mass (as expected)")
            
            # Check 2: f unchanged (independent of mass)
            if not math.isclose(f_new, self.f_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: f changed with mass perturbation - not independent")
                self.m = original_m
                return False
            else:
                self.log("PASS: f independent of mass (as expected)")
            
            # Check 3: μ decreases when mass increases (since μ = f/(m*g), f constant)
            if mu_new >= self.mu_expected - 1e-2:  # Should be strictly less
                self.log(f"FAIL: μ did not decrease with increased mass")
                self.m = original_m
                return False
            else:
                self.log(f"PASS: μ decreases with mass increase (from {self.mu_expected:.3f} to {mu_new:.3f})")
            
            # Restore original values
            self.m = original_m
            self.F_max, self.f, self.mu = original_results
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            self.m = original_m
            self.F_max, self.f, self.mu = original_results
            return False
    
    def run_all_tests(self):
        self.logs = []  # Reset logs
        success = True
        
        try:
            self.log("=== PhysicsChecker Started ===")
            self.log(f"Inputs: m={self.m} kg, k={self.k} N/m, g={self.g} N/kg")
            self.log(f"        L0={self.L0} m, L1={self.L1} m, L2={self.L2} m")
            self.log(f"Results: F_max={self.F_max}, f={self.f}, μ={self.mu}")
            
            # Run tests in sequence
            if not self.test_invariants():
                self.log("FAILED: test_invariants")
                success = False
            else:
                self.log("PASSED: test_invariants")
            
            if not self.test_trace_consistency():
                self.log("FAILED: test_trace_consistency")
                success = False
            else:
                self.log("PASSED: test_trace_consistency")
            
            if not self.test_monotonicity():
                self.log("FAILED: test_monotonicity")
                success = False
            else:
                self.log("PASSED: test_monotonicity")
            
        except Exception as e:
            self.log(f"CRITICAL ERROR: {traceback.format_exc()}")
            success = False
        
        self.log(f"=== PhysicsChecker Finished: {'SUCCESS' if success else 'FAILURE'} ===")
        return success, self.logs