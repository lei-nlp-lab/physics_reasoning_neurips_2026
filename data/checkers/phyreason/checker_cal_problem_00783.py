import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        # Extract given constants from the problem (always present)
        self.h1 = 4.0  # meters
        self.theta_i_deg = 37.0  # degrees
        self.theta_exit_deg = 53.0  # degrees (angle from vertical)
        self.x_total = 7.0  # meters
        self.c = 3.0e8  # m/s
        
        # Trigonometric values provided
        self.sin53 = 0.8
        self.cos53 = 0.6
        self.sin37 = 0.6  # derived from problem context
        self.cos37 = 0.8  # derived from problem context
        
        # Safely extract computed results from solution instance
        # Use only dictionary lookup from results
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract with careful type checking
        def safe_float_get(key, default=0.0):
            val = results_dict.get(key, default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        
        self.x1 = safe_float_get('x₁', 0.0)
        self.ratio = safe_float_get('x₂/h₂', 0.0)
        self.x_total_computed = safe_float_get('x', 0.0)
        self.h2 = safe_float_get('h₂', 0.0)
        self.n = safe_float_get('n', 0.0)
        self.v = safe_float_get('v', 0.0)
        self.t = safe_float_get('t', 0.0)
        
        # Store logs for reporting
        self.logs = []
    
    def _log(self, message, success):
        """Record test result."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def test_geometry_consistency(self):
        """Check geometric relationships from steps 1-4."""
        successes = []
        
        # Step 1: tan(37°) = x1 / h1
        x1_expected = self.h1 * self.sin37 / self.cos37  # tan37 = sin37/cos37
        successes.append(self._log(
            f"Step 1 geometry: x1 = {self.x1:.3f}, expected {x1_expected:.3f}",
            math.isclose(self.x1, x1_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Step 2: tan(53°) = x2/h2 = ratio
        ratio_expected = self.sin53 / self.cos53  # tan53 = sin53/cos53
        successes.append(self._log(
            f"Step 2 geometry: ratio = {self.ratio:.3f}, expected {ratio_expected:.3f}",
            math.isclose(self.ratio, ratio_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Step 3: x_total = x1 + x2
        # Calculate x2 from ratio and h2
        x2 = 0.0
        if self.h2 != 0:
            x2 = self.ratio * self.h2
        x_total_check = self.x1 + x2
        successes.append(self._log(
            f"Step 3 geometry: x1 + x2 = {x_total_check:.3f}, given x_total = {self.x_total:.3f}",
            math.isclose(x_total_check, self.x_total, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Step 4: h2 from simultaneous equations
        # x2 = x_total - x1, and h2 = x2 / tan(53°)
        x2_calc = self.x_total - self.x1
        if abs(self.sin53 / self.cos53) > 1e-10:  # Avoid division by zero
            h2_expected = x2_calc / (self.sin53 / self.cos53)
        else:
            h2_expected = 0.0
        successes.append(self._log(
            f"Step 4 geometry: h2 = {self.h2:.3f}, expected {h2_expected:.3f}",
            math.isclose(self.h2, h2_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        return all(successes)
    
    def test_refraction_invariants(self):
        """Check Snell's law and speed relationships."""
        successes = []
        
        # Snell's law: n_water * sin(37°) = n_air * sin(53°), with n_air = 1
        n_expected = self.sin53 / self.sin37
        successes.append(self._log(
            f"Snell's law: n = {self.n:.3f}, expected {n_expected:.3f}",
            math.isclose(self.n, n_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Speed in water: v = c / n
        if self.n > 1e-10:  # Avoid division by zero
            v_expected = self.c / n_expected
        else:
            v_expected = 0.0
        successes.append(self._log(
            f"Speed in water: v = {self.v:.3e}, expected {v_expected:.3e}",
            math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Consistency check: n should be > 1 (water denser than air)
        successes.append(self._log(
            f"Refractive index > 1: n = {self.n:.3f}",
            self.n > 1.0
        ))
        
        return all(successes)
    
    def test_time_calculation(self):
        """Check travel time computation."""
        # Time in water: t = h1 / (v * cos37°)
        # Note: The path length in water is h1/cos37°, speed is v
        if abs(self.v * self.cos37) > 1e-10:  # Avoid division by zero
            t_expected = self.h1 / (self.v * self.cos37)
        else:
            t_expected = 0.0
        
        return self._log(
            f"Travel time: t = {self.t:.3e} s, expected {t_expected:.3e} s",
            math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-10)
        )
    
    def test_target_answers(self):
        """Compare with given target answers."""
        successes = []
        
        # Target 1: h2 = 3 m
        successes.append(self._log(
            f"Target h2: {self.h2:.3f} m vs 3.0 m",
            math.isclose(self.h2, 3.0, rel_tol=1e-2, abs_tol=1e-2)
        ))
        
        # Target 2: t = (20/9) × 10^-8 s ≈ 2.222... × 10^-8 s
        t_target = (20.0/9.0) * 1e-8
        successes.append(self._log(
            f"Target t: {self.t:.3e} s vs {t_target:.3e} s",
            math.isclose(self.t, t_target, rel_tol=1e-2, abs_tol=1e-10)
        ))
        
        return all(successes)
    
    def test_monotonicity(self):
        """Perturb h1 and check logical trend in travel time."""
        try:
            original_h1 = self.h1
            perturbed_h1 = original_h1 * 1.1  # Increase by 10%
            
            # Recalculate using the geometric and physical relationships
            # Step 1: x1' = tan37 * h1'
            x1_prime = self.sin37 / self.cos37 * perturbed_h1
            
            # Step 2-4: h2' = (x_total - x1') / tan53
            if abs(self.sin53 / self.cos53) > 1e-10:
                h2_prime = (self.x_total - x1_prime) / (self.sin53 / self.cos53)
            else:
                h2_prime = 0.0
            
            # Refractive index remains the same
            n_prime = self.sin53 / self.sin37
            
            # Speed in water
            if n_prime > 1e-10:
                v_prime = self.c / n_prime
            else:
                v_prime = 0.0
            
            # Travel time
            if abs(v_prime * self.cos37) > 1e-10:
                t_prime = perturbed_h1 / (v_prime * self.cos37)
            else:
                t_prime = 0.0
            
            # Check monotonicity: if h1 increases, travel time should increase
            # (since path length increases while speed remains constant)
            trend_ok = t_prime > self.t
            
            return self._log(
                f"Monotonicity: t({original_h1:.2f}) = {self.t:.3e}, "
                f"t({perturbed_h1:.2f}) = {t_prime:.3e}, increasing = {trend_ok}",
                trend_ok
            )
        except Exception as e:
            return self._log(f"Monotonicity test failed: {str(e)}", False)
    
    def run_all_tests(self):
        """Execute all checks and return success status with logs."""
        self.logs = []
        
        tests = [
            ("Geometry Consistency", self.test_geometry_consistency),
            ("Refraction Invariants", self.test_refraction_invariants),
            ("Time Calculation", self.test_time_calculation),
            ("Target Answers", self.test_target_answers),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        all_passed = True
        for name, test_func in tests:
            try:
                if not test_func():
                    all_passed = False
            except Exception as e:
                self.logs.append(f"ERROR in {name}: {str(e)}")
                all_passed = False
        
        return all_passed, self.logs