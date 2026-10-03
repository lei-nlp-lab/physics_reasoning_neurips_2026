import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all required inputs and results from solution_instance."""
        self.solution = solution_instance
        
        # Extract inputs with safe fallbacks
        self.s = getattr(solution_instance, 's', getattr(solution_instance, 'distance', 100.0))  # distance along arc
        self.r = getattr(solution_instance, 'r', getattr(solution_instance, 'radius', 20.0))      # radius
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'time', 10.0))        # time
        
        # Extract results from results dictionary or attributes
        results = getattr(solution_instance, 'results', {})
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        self.omega = getattr(solution_instance, 'omega', results.get('omega', 0.0))
        self.a = getattr(solution_instance, 'a', results.get('a', 0.0))
        
        # Symbolic variables for invariant checking
        self.sym_s, self.sym_r, self.sym_t, self.sym_v, self.sym_omega, self.sym_a = sp.symbols('s r t v omega a')
        
        # Expected values from steps analysis (for trace consistency)
        self.expected_v = 10.0
        self.expected_omega = 0.5
        self.expected_a = 5.0
        
        self.logs = []

    def log(self, msg, status='INFO'):
        """Helper to log messages with status."""
        self.logs.append(f'[{status}] {msg}')
    
    def test_invariants(self):
        """Verify fundamental physics relationships using symbolic and numerical checks."""
        checks_passed = True
        
        # 1. Linear velocity definition: v = s/t (since uniform circular motion)
        if not math.isclose(self.v, self.s / self.t, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Linear velocity invariant failed: v={self.v}, s/t={self.s/self.t:.2f}')
            checks_passed = False
        else:
            self.log('Linear velocity definition (v = s/t) satisfied')
        
        # 2. Relationship between linear and angular velocity: v = omega * r
        if not math.isclose(self.v, self.omega * self.r, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Velocity-angular velocity invariant failed: v={self.v}, omega*r={self.omega*self.r:.2f}')
            checks_passed = False
        else:
            self.log('Velocity-angular velocity relationship (v = ω*r) satisfied')
        
        # 3. Centripetal acceleration formulas: a = v²/r = ω²*r
        # Check a = v²/r
        if not math.isclose(self.a, self.v**2 / self.r, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Centripetal acceleration (v²/r) failed: a={self.a}, v²/r={self.v**2/self.r:.2f}')
            checks_passed = False
        else:
            self.log('Centripetal acceleration (a = v²/r) satisfied')
        
        # Check a = ω²*r
        if not math.isclose(self.a, self.omega**2 * self.r, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Centripetal acceleration (ω²*r) failed: a={self.a}, ω²*r={self.omega**2*self.r:.2f}')
            checks_passed = False
        else:
            self.log('Centripetal acceleration (a = ω²*r) satisfied')
        
        # Symbolic cross-check for completeness
        try:
            eq1 = sp.Eq(self.sym_v, self.sym_s / self.sym_t)
            eq2 = sp.Eq(self.sym_v, self.sym_omega * self.sym_r)
            eq3 = sp.Eq(self.sym_a, self.sym_v**2 / self.sym_r)
            eq4 = sp.Eq(self.sym_a, self.sym_omega**2 * self.sym_r)
            
            # Substitute numerical values into symbolic equations
            subs_dict = {self.sym_s: self.s, self.sym_r: self.r, self.sym_t: self.t,
                        self.sym_v: self.v, self.sym_omega: self.omega, self.sym_a: self.a}
            
            for eq, desc in [(eq1, 'v = s/t'), (eq2, 'v = ω*r'), 
                           (eq3, 'a = v²/r'), (eq4, 'a = ω²*r')]:
                if not eq.subs(subs_dict).simplify():
                    self.log(f'Symbolic check failed for {desc}', 'WARNING')
        except Exception as e:
            self.log(f'Symbolic invariant check error: {e}', 'WARNING')
        
        return checks_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results from steps analysis with final answers."""
        checks_passed = True
        
        # Check against expected values from steps analysis
        if not math.isclose(self.v, self.expected_v, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Linear velocity mismatch: got {self.v}, expected {self.expected_v}')
            checks_passed = False
        else:
            self.log(f'Linear velocity matches expected {self.expected_v} m/s')
        
        if not math.isclose(self.omega, self.expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Angular velocity mismatch: got {self.omega}, expected {self.expected_omega}')
            checks_passed = False
        else:
            self.log(f'Angular velocity matches expected {self.expected_omega} rad/s')
        
        if not math.isclose(self.a, self.expected_a, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Centripetal acceleration mismatch: got {self.a}, expected {self.expected_a}')
            checks_passed = False
        else:
            self.log(f'Centripetal acceleration matches expected {self.expected_a} m/s²')
        
        # Check sequential consistency: omega should equal v/r
        expected_omega_from_v = self.v / self.r
        if not math.isclose(self.omega, expected_omega_from_v, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f'Step 2 consistency failed: ω={self.omega}, v/r={expected_omega_from_v:.2f}')
            checks_passed = False
        else:
            self.log('Step 2 (ω = v/r) is consistent with computed v')
        
        # Check a should equal v²/r AND ω²*r
        a_from_v = self.v**2 / self.r
        a_from_omega = self.omega**2 * self.r
        if not (math.isclose(self.a, a_from_v, rel_tol=1e-2, abs_tol=1e-2) and 
                math.isclose(self.a, a_from_omega, rel_tol=1e-2, abs_tol=1e-2)):
            self.log(f'Step 3 consistency failed: a={self.a}, v²/r={a_from_v:.2f}, ω²*r={a_from_omega:.2f}')
            checks_passed = False
        else:
            self.log('Step 3 (a = v²/r = ω²*r) is consistent')
        
        return checks_passed
    
    def test_monotonicity(self):
        """Perturb inputs and verify output trends match physical expectations."""
        original_s = self.s
        checks_passed = True
        
        # Perturbation: increase distance traveled by 10%
        perturbed_s = original_s * 1.10
        
        # Recalculate using physics formulas (not calling solution methods)
        perturbed_v = perturbed_s / self.t  # v = s/t
        perturbed_omega = perturbed_v / self.r  # ω = v/r
        perturbed_a = perturbed_v**2 / self.r  # a = v²/r
        
        # Expected trends from KB:
        # 1. If s increases (t constant), v should increase (linear relationship)
        # 2. If v increases (r constant), ω should increase (linear relationship)
        # 3. If v increases (r constant), a should increase (quadratic relationship)
        
        if not (perturbed_v > self.v):
            self.log(f'Monotonicity failure: v did not increase when s increased. v={self.v}, perturbed_v={perturbed_v}')
            checks_passed = False
        else:
            self.log(f'Linear velocity increases with distance (v monotonic): {self.v:.2f} → {perturbed_v:.2f}')
        
        if not (perturbed_omega > self.omega):
            self.log(f'Monotonicity failure: ω did not increase when v increased. ω={self.omega}, perturbed_ω={perturbed_omega}')
            checks_passed = False
        else:
            self.log(f'Angular velocity increases with linear velocity (ω monotonic): {self.omega:.2f} → {perturbed_omega:.2f}')
        
        if not (perturbed_a > self.a):
            self.log(f'Monotonicity failure: a did not increase when v increased. a={self.a}, perturbed_a={perturbed_a}')
            checks_passed = False
        else:
            self.log(f'Centripetal acceleration increases with velocity squared (a monotonic): {self.a:.2f} → {perturbed_a:.2f}')
        
        # Restore original value (conceptually - we didn't actually modify solution)
        # No restoration needed as we only used local variables
        
        # Additional check: decrease time (t) should increase v
        original_t = self.t
        perturbed_t = original_t * 0.90  # 10% decrease
        perturbed_v2 = self.s / perturbed_t
        if not (perturbed_v2 > self.v):
            self.log(f'Monotonicity failure: v did not increase when t decreased. v={self.v}, perturbed_v={perturbed_v2}')
            checks_passed = False
        else:
            self.log(f'Linear velocity increases when time decreases (inverse relationship): {self.v:.2f} → {perturbed_v2:.2f}')
        
        return checks_passed
    
    def run_all_tests(self):
        """Execute all verification tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        test_results = []
        
        try:
            self.log('Starting PhysicsChecker for uniform circular motion problem')
            self.log(f'Inputs: s={self.s} m, r={self.r} m, t={self.t} s')
            self.log(f'Results: v={self.v} m/s, ω={self.omega} rad/s, a={self.a} m/s²')
            
            # Run individual tests
            test_results.append(('Invariants', self.test_invariants()))
            test_results.append(('Trace Consistency', self.test_trace_consistency()))
            test_results.append(('Monotonicity', self.test_monotonicity()))
            
            # Determine overall success
            success = all(passed for _, passed in test_results)
            
            # Log summary
            self.log('\n=== Test Summary ===')
            for test_name, passed in test_results:
                status = 'PASS' if passed else 'FAIL'
                self.log(f'{test_name}: {status}')
            
            self.log(f'\nOverall: {"PASS" if success else "FAIL"}')
            
            return success, self.logs
            
        except Exception as e:
            self.log(f'Unexpected error during testing: {e}\n{traceback.format_exc()}', 'ERROR')
            return False, self.logs