import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr with fallback to .results dictionary.
        """
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results with safe defaults
        self.r = getattr(solution_instance, 'r', self.results.get('r', 0.0))  # refraction angle
        self.i = getattr(solution_instance, 'i', self.results.get('i', 0.0))  # incident angle
        self.n = getattr(solution_instance, 'n', self.results.get('n', 0.0))  # refractive index
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))  # speed in medium
        self.s = getattr(solution_instance, 's', self.results.get('s', 0.0))  # distance traveled
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))  # propagation time
        
        # Extract inputs with safe defaults
        self.l = getattr(solution_instance, 'l', getattr(solution_instance, 'inputs', {}).get('l', 1.0))  # prism side length
        self.c = getattr(solution_instance, 'c', getattr(solution_instance, 'inputs', {}).get('c', 1.0))  # speed of light in vacuum
        self.deviation = 15.0  # given deviation in degrees
        
        # For symbolic checks
        self.symbols = {
            'r': sp.symbols('r'),
            'i': sp.symbols('i'),
            'n': sp.symbols('n'),
            'v': sp.symbols('v'),
            'c': sp.symbols('c'),
            's': sp.symbols('s'),
            't': sp.symbols('t'),
            'l': sp.symbols('l')
        }
        
        self.logs = []
    
    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def test_snells_law(self):
        """Verify Snell's Law: n = sin(i)/sin(r)."""
        try:
            # Convert to radians
            i_rad = math.radians(self.i)
            r_rad = math.radians(self.r)
            
            # Symbolic check
            lhs = self.symbols['n']
            rhs = sp.sin(self.symbols['i']) / sp.sin(self.symbols['r'])
            symbolic_eq = sp.Eq(lhs, rhs)
            
            # Numerical check
            n_calculated = math.sin(i_rad) / math.sin(r_rad)
            numeric_ok = math.isclose(self.n, n_calculated, rel_tol=1e-2, abs_tol=1e-2)
            
            msg = f"Snell's Law: n = sin({self.i:.1f}°)/sin({self.r:.1f}°) = {n_calculated:.3f} vs {self.n:.3f}"
            return self._log(msg, numeric_ok)
        except Exception as e:
            return self._log(f"Snell's Law failed: {e}", False)
    
    def test_speed_index_relation(self):
        """Verify v = c/n (inverse relationship)."""
        try:
            # Symbolic check
            symbolic_eq = sp.Eq(self.symbols['v'], self.symbols['c'] / self.symbols['n'])
            
            # Numerical check
            v_calculated = self.c / self.n
            numeric_ok = math.isclose(self.v, v_calculated, rel_tol=1e-2, abs_tol=1e-2)
            
            msg = f"Speed-index: v = c/n = {self.c:.3f}/{self.n:.3f} = {v_calculated:.3f} vs {self.v:.3f}"
            return self._log(msg, numeric_ok)
        except Exception as e:
            return self._log(f"Speed-index relation failed: {e}", False)
    
    def test_distance_geometry(self):
        """Verify s = l/2 from geometry."""
        try:
            s_expected = self.l / 2.0
            numeric_ok = math.isclose(self.s, s_expected, rel_tol=1e-2, abs_tol=1e-2)
            
            msg = f"Distance geometry: s = l/2 = {self.l:.3f}/2 = {s_expected:.3f} vs {self.s:.3f}"
            return self._log(msg, numeric_ok)
        except Exception as e:
            return self._log(f"Distance geometry failed: {e}", False)
    
    def test_time_calculation(self):
        """Verify t = s/v."""
        try:
            # Symbolic check
            symbolic_eq = sp.Eq(self.symbols['t'], self.symbols['s'] / self.symbols['v'])
            
            # Numerical check
            t_calculated = self.s / self.v
            numeric_ok = math.isclose(self.t, t_calculated, rel_tol=1e-2, abs_tol=1e-2)
            
            msg = f"Time calculation: t = s/v = {self.s:.3f}/{self.v:.3f} = {t_calculated:.3f} vs {self.t:.3f}"
            return self._log(msg, numeric_ok)
        except Exception as e:
            return self._log(f"Time calculation failed: {e}", False)
    
    def test_deviation_consistency(self):
        """Verify that deviation = 15° is consistent with angles."""
        try:
            # From geometry: deviation = i - r (since light bends toward normal when entering denser medium)
            deviation_calculated = self.i - self.r
            numeric_ok = math.isclose(deviation_calculated, self.deviation, rel_tol=1e-2, abs_tol=1e-2)
            
            msg = f"Deviation: i - r = {self.i:.1f}° - {self.r:.1f}° = {deviation_calculated:.1f}° vs {self.deviation:.1f}°"
            return self._log(msg, numeric_ok)
        except Exception as e:
            return self._log(f"Deviation consistency failed: {e}", False)
    
    def test_target_answers(self):
        """Verify against target answers: √2 and (√2 l)/(2c)."""
        try:
            # Target 1: n = √2
            target_n = math.sqrt(2)
            n_ok = math.isclose(self.n, target_n, rel_tol=1e-2, abs_tol=1e-2)
            
            # Target 2: t = (√2 l)/(2c)
            target_t = (math.sqrt(2) * self.l) / (2 * self.c)
            t_ok = math.isclose(self.t, target_t, rel_tol=1e-2, abs_tol=1e-2)
            
            msg = f"Target answers: n={target_n:.3f} vs {self.n:.3f}, t={target_t:.3f} vs {self.t:.3f}"
            return self._log(msg, n_ok and t_ok)
        except Exception as e:
            return self._log(f"Target answers check failed: {e}", False)
    
    def test_monotonicity(self):
        """
        Perturb input l, recalculate using physics laws, check monotonic relationship.
        Restore original values after test.
        """
        original_l = self.l
        
        try:
            # Increase l by 10%
            perturbed_l = original_l * 1.1
            
            # Recalculate using physics formulas from steps
            s_perturbed = perturbed_l / 2.0  # s = l/2
            v_original = self.c / self.n     # v = c/n (unchanged)
            t_perturbed = s_perturbed / v_original  # t = s/v
            
            # Check monotonicity: if l increases, s increases, so t should increase
            monotonic_ok = t_perturbed > self.t
            
            # Restore original
            self.l = original_l
            
            msg = f"Monotonicity: l↑ ({original_l:.3f}→{perturbed_l:.3f}) → t↑ ({self.t:.3f}→{t_perturbed:.3f})"
            return self._log(msg, monotonic_ok)
        except Exception as e:
            self.l = original_l  # Ensure restore even if error
            return self._log(f"Monotonicity test failed: {e}", False)
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        tests = [
            self.test_snells_law,
            self.test_speed_index_relation,
            self.test_distance_geometry,
            self.test_time_calculation,
            self.test_deviation_consistency,
            self.test_target_answers,
            self.test_monotonicity
        ]
        
        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                self._log(f"Test crashed: {e}\n{traceback.format_exc()}", False)
                results.append(False)
        
        success = all(results)
        return success, self.logs