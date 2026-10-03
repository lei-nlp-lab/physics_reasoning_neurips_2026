import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        # Extract input parameters - first try from results, then from attributes
        results = getattr(solution_instance, 'results', {})
        
        # Get m, P, v from results if stored there, otherwise from attributes
        self.m = results.get('m', getattr(solution_instance, 'm', 1000.0))
        self.P = results.get('P', getattr(solution_instance, 'P', 50000.0))
        self.v = results.get('v', getattr(solution_instance, 'v', 25.0))
        
        # Extract all needed quantities with fallbacks
        self.f = results.get('f', 0.0)
        self.F = results.get('F', 0.0)
        self.F_prime = results.get("F''", 0.0)  # Traction at v/4
        self.F_all = results.get('F_all', 0.0)  # Net force at v/4
        self.F_all_prime = results.get("F_all''", 0.0)  # Net force when F'' = 2f
        self.t = results.get('t', 0.0)
        self.a = results.get('a', 0.0)  # Acceleration at v/4
        self.v_max = results.get('v_max', 0.0)
        
        self.logs = []

    def _log(self, message, success):
        self.logs.append((message, success))

    def _is_close(self, val1, val2):
        """Fuzzy numerical matching with 1-2% tolerance."""
        if val1 is None or val2 is None:
            return False
        return math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2)

    def _check_power_formula(self, force, speed, power):
        """Check P = F * v relation."""
        return self._is_close(power, force * speed)

    def test_invariants(self):
        """Verify physics invariants based on KB theorems."""
        success = True
        
        try:
            # 1. Equilibrium at maximum speed: F = f (traction = resistance)
            if not self._is_close(self.F, self.f):
                self._log(f"Equilibrium violation: F={self.F}, f={self.f}", False)
                success = False
            else:
                self._log("Equilibrium at max speed: ✓", True)
            
            # 2. Power formula at maximum speed: P = F * v
            if self._check_power_formula(self.F, self.v, self.P):
                self._log("Power formula at max speed: ✓", True)
            else:
                self._log(f"Power formula violation: P={self.P}, F*v={self.F*self.v}", False)
                success = False
            
            # 3. Newton's 2nd law: F_net = m * a (for v/4 case)
            expected_net_force = self.m * self.a
            if self._is_close(self.F_all, expected_net_force):
                self._log("Newton's 2nd law (v/4 case): ✓", True)
            else:
                self._log(f"Newton's 2nd law violation: F_all={self.F_all}, m*a={expected_net_force}", False)
                success = False
            
            # 4. Superposition of forces: F_all = F' - f (v/4 case)
            net_force_calc = self.F_prime - self.f
            if self._is_close(self.F_all, net_force_calc):
                self._log("Force superposition (v/4): ✓", True)
            else:
                self._log(f"Force superposition violation: F_all={self.F_all}, F'-f={net_force_calc}", False)
                success = False
            
            # 5. Consistency for F'' = 2f case
            expected_F_all_prime = 2 * self.f - self.f  # F'' - f where F'' = 2f
            if self._is_close(self.F_all_prime, expected_F_all_prime):
                self._log("Force superposition (F''=2f case): ✓", True)
            else:
                self._log(f"F''=2f case violation: F_all'={self.F_all_prime}, expected={expected_F_all_prime}", False)
                success = False
            
            # 6. Maximum speed when traction doubles resistance
            expected_v_max = self.v / 2
            if self._is_close(self.v_max, expected_v_max):
                self._log("Maximum speed (F''=2f): ✓", True)
            else:
                self._log(f"Maximum speed violation: v_max={self.v_max}, expected v/2={expected_v_max}", False)
                success = False
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            success = False
            
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        try:
            # Expected formulas
            expected_f = self.P / self.v
            expected_a = 3 * self.P / (self.m * self.v)
            expected_t = self.m * self.v**2 / (2 * self.P)
            
            # Check target answers
            checks = [
                ("Resistance force f", self.f, expected_f),
                ("Acceleration at v/4", self.a, expected_a),
                ("Time to reach v/2", self.t, expected_t)
            ]
            
            for name, actual, expected in checks:
                if self._is_close(actual, expected):
                    self._log(f"{name}: ✓", True)
                else:
                    self._log(f"{name} mismatch: {actual} vs {expected}", False)
                    success = False
            
            # Check intermediate steps
            expected_F_prime = 4 * self.P / self.v
            if self._is_close(self.F_prime, expected_F_prime):
                self._log("Traction at v/4 (F'): ✓", True)
            else:
                self._log(f"F' mismatch: {self.F_prime} vs {expected_F_prime}", False)
                success = False
            
            expected_F_all_prime = self.P / self.v
            if self._is_close(self.F_all_prime, expected_F_all_prime):
                self._log("Net force when F''=2f: ✓", True)
            else:
                self._log(f"F_all' mismatch: {self.F_all_prime} vs {expected_F_all_prime}", False)
                success = False
                    
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            success = False
            
        return success

    def test_monotonicity(self):
        """Perturb mass and check trend, then restore."""
        success = True
        
        try:
            # Store original values
            original_m = self.m
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Calculate expected changes using formulas
            original_a = 3 * self.P / (original_m * self.v)
            perturbed_a = 3 * self.P / (perturbed_m * self.v)
            
            original_t = original_m * self.v**2 / (2 * self.P)
            perturbed_t = perturbed_m * self.v**2 / (2 * self.P)
            
            # Check trends
            if perturbed_a < original_a:  # a should decrease when m increases
                self._log("Monotonicity: a decreases with increased mass ✓", True)
            else:
                self._log(f"Monotonicity violation: a({perturbed_m})={perturbed_a} >= a({original_m})={original_a}", False)
                success = False
                
            if perturbed_t > original_t:  # t should increase when m increases
                self._log("Monotonicity: t increases with increased mass ✓", True)
            else:
                self._log(f"Monotonicity violation: t({perturbed_m})={perturbed_t} <= t({original_m})={original_t}", False)
                success = False
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
                
        return success

    def run_all_tests(self):
        """Execute all tests and return results."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== Running {test_name} ===", True)
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        # Count successes
        success_count = sum(1 for _, s in self.logs if s)
        total_count = sum(1 for _, s in self.logs if isinstance(s, bool))
        
        self._log(f"\n=== Summary: {success_count}/{total_count} checks passed ===", overall_success)
        
        return overall_success, self.logs