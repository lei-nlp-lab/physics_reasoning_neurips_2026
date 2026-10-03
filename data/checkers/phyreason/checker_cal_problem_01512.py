import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs: Earth radius (R), orbital period (T), gravity (g), near-Earth altitude (h)
        # Use getattr on instance, then results dict, then default to 1.0 for dimensionless checks
        self.R = self._safe_get('R', 1.0)
        self.T = self._safe_get('T', 1.0)
        self.g = self._safe_get('g', 9.8)
        self.h = self._safe_get('h', 0.0)  # altitude of near-Earth circular orbit
        
        # Results: velocity (v) and acceleration (a)
        self.v = self._safe_get('v', 0.0)
        self.a = self._safe_get('a', 0.0)
        
        # Derived constants
        self.r_geo = self.R + 5.6 * self.R  # geostationary orbit radius
        self.r_A = self.R + self.h          # near-Earth orbit radius at A
        
    def _safe_get(self, key, default):
        """Safely retrieve attribute from instance or results dict."""
        value = getattr(self.solution, key, None)
        if value is None:
            value = self.results.get(key, default)
        return float(value) if isinstance(value, (int, float, sp.Number)) else default
    
    def test_invariants(self):
        """Verify physical laws: centripetal force equals gravity for both orbits."""
        logs = []
        try:
            # 1. Geostationary orbit: gravitational force provides centripetal force
            # From Newton's law: g = G*M/R^2, so G*M = g*R^2
            # At geostationary orbit: a_geo = G*M / r_geo^2 = g * R^2 / r_geo^2
            a_geo_expected = self.g * self.R**2 / self.r_geo**2
            # Centripetal acceleration from velocity: a_geo_calc = v^2 / r_geo
            a_geo_calc = self.v**2 / self.r_geo
            
            if not math.isclose(a_geo_expected, a_geo_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Geostationary orbit invariant mismatch. "
                            f"Expected a={a_geo_expected:.3f} from gravity, "
                            f"got {a_geo_calc:.3f} from centripetal formula.")
            else:
                logs.append("PASS: Geostationary orbit satisfies Newton's law.")
            
            # 2. Near-Earth circular orbit at point A: a = g * R^2 / (R+h)^2
            a_A_expected = self.g * self.R**2 / self.r_A**2
            if not math.isclose(self.a, a_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration at A mismatch. "
                            f"Expected a={a_A_expected:.3f}, got {self.a:.3f}.")
            else:
                logs.append("PASS: Acceleration at A matches gravitational formula.")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check velocity formula: v = 2π * r_geo / T = 13.2πR / T
            v_expected = 2 * math.pi * self.r_geo / self.T
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Velocity formula mismatch. "
                            f"Expected v={v_expected:.3f}, got {self.v:.3f}.")
            else:
                logs.append("PASS: Velocity matches period-radius relation.")
            
            # Check acceleration formula: a = g * R^2 / (R+h)^2
            a_expected = self.g * self.R**2 / (self.R + self.h)**2
            if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration formula mismatch. "
                            f"Expected a={a_expected:.3f}, got {self.a:.3f}.")
            else:
                logs.append("PASS: Acceleration matches derived formula.")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        logs = []
        try:
            # Perturb Earth radius R by +10%
            R_orig = self.R
            delta = 0.1 * R_orig
            self.R = R_orig + delta
            
            # Recompute expected values using formulas
            v_new_expected = 2 * math.pi * (self.R + 5.6 * self.R) / self.T
            a_new_expected = self.g * self.R**2 / (self.R + self.h)**2
            
            # Restore original R
            self.R = R_orig
            
            # Original expected values (using formulas)
            v_orig_expected = 2 * math.pi * (self.R + 5.6 * self.R) / self.T
            a_orig_expected = self.g * self.R**2 / (self.R + self.h)**2
            
            # Check trends
            if v_new_expected > v_orig_expected:
                logs.append("PASS: Velocity increases with increased Earth radius (v ∝ R).")
            else:
                logs.append("FAIL: Velocity does not increase with increased Earth radius.")
            
            # For acceleration, note that a = g * R^2/(R+h)^2. As R increases, the derivative is positive if h < R, negative if h > R.
            # Typically h is small for near-Earth orbit (h << R), so a should increase.
            if a_new_expected > a_orig_expected:
                logs.append("PASS: Acceleration increases with increased Earth radius (typical for h << R).")
            else:
                logs.append("INFO: Acceleration trend depends on h/R ratio.")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs.append(f"\n--- {test_name} ---")
            test_logs = test_method()
            logs.extend(test_logs)
            # If any test log contains "FAIL", mark overall as False
            if any("FAIL" in log for log in test_logs):
                success = False
        
        return success, logs