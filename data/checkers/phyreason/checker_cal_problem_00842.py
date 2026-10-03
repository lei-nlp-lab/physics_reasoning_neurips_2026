import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs safely
        self.inputs['a'] = getattr(solution_instance, 'a', 
                                 getattr(solution_instance, 'inputs', {}).get('a', 3.0))
        self.inputs['t'] = getattr(solution_instance, 't', 
                                 getattr(solution_instance, 'inputs', {}).get('t', 4.0))
        self.inputs['v0'] = getattr(solution_instance, 'v0', 
                                  getattr(solution_instance, 'inputs', {}).get('v0', 0.0))
        
        # Extract results safely
        self.results['v'] = getattr(solution_instance, 'v', 
                                  getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.results['s'] = getattr(solution_instance, 's', 
                                  getattr(solution_instance, 'results', {}).get('s', 0.0))
        
        # Constants
        self.tolerance = 0.02  # 2% tolerance for physics rounding
        self.logs = []
        
        # Symbolic variables
        self.v0_sym, self.a_sym, self.t_sym = sp.symbols('v0 a t')
        self.v_sym = self.v0_sym + self.a_sym * self.t_sym
        self.s_sym = self.v0_sym * self.t_sym + 0.5 * self.a_sym * self.t_sym**2
    
    def log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def test_invariants(self):
        """Test fundamental physics laws"""
        successes = []
        
        # Test 1: Velocity-time relationship v = v0 + a*t
        v_expected = self.inputs['v0'] + self.inputs['a'] * self.inputs['t']
        symbolic_v = self.v_sym.subs({
            self.v0_sym: self.inputs['v0'],
            self.a_sym: self.inputs['a'],
            self.t_sym: self.inputs['t']
        })
        
        numeric_match = math.isclose(
            self.results['v'], v_expected, 
            rel_tol=self.tolerance, abs_tol=self.tolerance
        )
        symbolic_match = math.isclose(
            float(symbolic_v), v_expected,
            rel_tol=self.tolerance, abs_tol=self.tolerance
        )
        
        successes.append(self.log(
            f"Velocity invariant: v = v0 + a·t = {v_expected:.2f} m/s, got {self.results['v']:.2f} m/s",
            numeric_match and symbolic_match
        ))
        
        # Test 2: Displacement-time relationship s = v0*t + 0.5*a*t²
        s_expected = self.inputs['v0'] * self.inputs['t'] + 0.5 * self.inputs['a'] * self.inputs['t']**2
        symbolic_s = self.s_sym.subs({
            self.v0_sym: self.inputs['v0'],
            self.a_sym: self.inputs['a'],
            self.t_sym: self.inputs['t']
        })
        
        numeric_match = math.isclose(
            self.results['s'], s_expected,
            rel_tol=self.tolerance, abs_tol=self.tolerance
        )
        symbolic_match = math.isclose(
            float(symbolic_s), s_expected,
            rel_tol=self.tolerance, abs_tol=self.tolerance
        )
        
        successes.append(self.log(
            f"Displacement invariant: s = v0·t + ½a·t² = {s_expected:.2f} m, got {self.results['s']:.2f} m",
            numeric_match and symbolic_match
        ))
        
        # Test 3: Energy consistency (kinetic energy = 0.5*m*v², work = F·s = m·a·s)
        # Assuming unit mass (m=1) for scaling
        ke = 0.5 * (self.results['v']**2 - self.inputs['v0']**2)
        work = self.inputs['a'] * self.results['s']
        
        successes.append(self.log(
            f"Work-Energy: ΔKE = {ke:.2f} J, Work = {work:.2f} J (scaled by m=1)",
            math.isclose(ke, work, rel_tol=self.tolerance, abs_tol=self.tolerance)
        ))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Cross-check with step-by-step analysis"""
        successes = []
        
        # From KB steps: v = a*t (since v0=0)
        step1_v = self.inputs['a'] * self.inputs['t']
        successes.append(self.log(
            f"Step 1 (velocity): v = a·t = {step1_v:.2f} m/s, got {self.results['v']:.2f} m/s",
            math.isclose(self.results['v'], step1_v, rel_tol=self.tolerance, abs_tol=self.tolerance)
        ))
        
        # From KB steps: s = 0.5*a*t² (since v0=0)
        step2_s = 0.5 * self.inputs['a'] * self.inputs['t']**2
        successes.append(self.log(
            f"Step 2 (displacement): s = ½a·t² = {step2_s:.2f} m, got {self.results['s']:.2f} m",
            math.isclose(self.results['s'], step2_s, rel_tol=self.tolerance, abs_tol=self.tolerance)
        ))
        
        # Consistency between steps: using v_avg = (v0+v)/2 = v/2, then s = v_avg*t = (v/2)*t
        v_avg = 0.5 * (self.inputs['v0'] + self.results['v'])
        s_from_vavg = v_avg * self.inputs['t']
        
        successes.append(self.log(
            f"Cross-check: s = v_avg·t = {s_from_vavg:.2f} m, got {self.results['s']:.2f} m",
            math.isclose(self.results['s'], s_from_vavg, rel_tol=self.tolerance, abs_tol=self.tolerance)
        ))
        
        return all(successes)
    
    def test_monotonicity(self):
        """Test how results change with parameter perturbations"""
        successes = []
        
        original_a = self.inputs['a']
        original_t = self.inputs['t']
        
        # Test 1: Increase acceleration by 10%
        perturbed_a = original_a * 1.1
        perturbed_v = self.inputs['v0'] + perturbed_a * original_t
        perturbed_s = self.inputs['v0'] * original_t + 0.5 * perturbed_a * original_t**2
        
        # Since a > 0, both v and s should increase
        v_increased = perturbed_v > self.results['v']
        s_increased = perturbed_s > self.results['s']
        
        successes.append(self.log(
            f"Monotonicity in a: v increased from {self.results['v']:.2f} to {perturbed_v:.2f} m/s",
            v_increased
        ))
        successes.append(self.log(
            f"Monotonicity in a: s increased from {self.results['s']:.2f} to {perturbed_s:.2f} m",
            s_increased
        ))
        
        # Test 2: Increase time by 5%
        perturbed_t = original_t * 1.05
        perturbed_v2 = self.inputs['v0'] + original_a * perturbed_t
        perturbed_s2 = self.inputs['v0'] * perturbed_t + 0.5 * original_a * perturbed_t**2
        
        # Since a > 0 and t > 0, both v and s should increase
        v_increased2 = perturbed_v2 > self.results['v']
        s_increased2 = perturbed_s2 > self.results['s']
        
        successes.append(self.log(
            f"Monotonicity in t: v increased from {self.results['v']:.2f} to {perturbed_v2:.2f} m/s",
            v_increased2
        ))
        successes.append(self.log(
            f"Monotonicity in t: s increased from {self.results['s']:.2f} to {perturbed_s2:.2f} m",
            s_increased2
        ))
        
        # Test 3: Linear scaling of v with a (when t fixed)
        # dv/da = t (symbolic derivative)
        dv_da_symbolic = sp.diff(self.v_sym, self.a_sym).subs({
            self.t_sym: original_t
        })
        dv_da_numeric = float(dv_da_symbolic)
        
        # Numeric derivative approximation
        delta_a = original_a * 0.01
        v_plus = self.inputs['v0'] + (original_a + delta_a) * original_t
        v_minus = self.inputs['v0'] + (original_a - delta_a) * original_t
        numeric_derivative = (v_plus - v_minus) / (2 * delta_a)
        
        successes.append(self.log(
            f"Linear scaling ∂v/∂a: symbolic = {dv_da_numeric:.2f}, numeric ≈ {numeric_derivative:.2f}",
            math.isclose(dv_da_numeric, numeric_derivative, rel_tol=0.1, abs_tol=0.1)
        ))
        
        # Restore original values (though they were never actually modified)
        self.inputs['a'] = original_a
        self.inputs['t'] = original_t
        
        return all(successes)
    
    def run_all_tests(self):
        """Execute all tests and return results"""
        self.logs = []  # Reset logs
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        all_passed = test1 and test2 and test3
        
        # Summary
        self.logs.append(f"\nSummary: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        self.logs.append(f"Inputs: a={self.inputs['a']} m/s², t={self.inputs['t']} s, v0={self.inputs['v0']} m/s")
        self.logs.append(f"Results: v={self.results['v']} m/s, s={self.results['s']} m")
        
        return all_passed, self.logs