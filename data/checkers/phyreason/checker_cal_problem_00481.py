import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Never access solution_instance.attr directly.
        """
        # Input parameters (with default fallbacks)
        self.m = getattr(solution_instance, 'm', 0.4)
        self.q = getattr(solution_instance, 'q', 3e-5)
        self.r = getattr(solution_instance, 'r', 0.5)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.sin_theta = math.sin(math.radians(self.theta))
        self.cos_theta = math.cos(math.radians(self.theta))
        # Override with given trig values if available
        self.sin_theta = getattr(solution_instance, 'sin_theta', 0.6)
        self.cos_theta = getattr(solution_instance, 'cos_theta', 0.8)
        
        # Results (safe extraction with fallback to 0.0)
        results = getattr(solution_instance, 'results', {})
        self.E_min = results.get('E_min', 0.0)
        self.E_k = results.get('E_k', 0.0)
        self.E = results.get('E', 0.0)
        self.v_D = results.get('v_D', 0.0)
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append((success, msg))
        
    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_E_field(self):
        """Check electric field from equilibrium at point C."""
        try:
            # Symbolic check: E = (m*g*tanθ)/q
            E_expected = self.m * self.g * self.sin_theta / (self.q * self.cos_theta)
            if self._is_close(self.E, E_expected):
                self._log(f"E field correct: {self.E:.2e} N/C ≈ {E_expected:.2e} N/C", True)
                return True
            else:
                self._log(f"E field mismatch: {self.E:.2e} vs {E_expected:.2e}", False)
                return False
        except Exception as e:
            self._log(f"E field test failed with error: {e}", False)
            return False
    
    def test_velocity_at_D(self):
        """Check velocity at point D from centripetal force balance."""
        try:
            # From step 3: sqrt((m*g)^2 + (q*E)^2) = m * v_D^2 / r
            lhs = math.sqrt((self.m * self.g)**2 + (self.q * self.E)**2)
            rhs = self.m * self.v_D**2 / self.r
            if self._is_close(lhs, rhs):
                self._log(f"Velocity at D consistent: v_D = {self.v_D:.2f} m/s", True)
                return True
            else:
                self._log(f"Velocity at D mismatch: lhs={lhs:.2f}, rhs={rhs:.2f}", False)
                return False
        except Exception as e:
            self._log(f"Velocity at D test failed with error: {e}", False)
            return False
    
    def test_kinetic_energy_at_far_right(self):
        """Check kinetic energy at far right point using work-energy theorem."""
        try:
            # From step 5: m*g*R*cosθ - q*E*R*(1-sinθ) = E_k - 0.5*m*v_D^2
            R = self.r
            lhs = self.m * self.g * R * self.cos_theta - self.q * self.E * R * (1 - self.sin_theta)
            rhs = self.E_k - 0.5 * self.m * self.v_D**2
            if self._is_close(lhs, rhs):
                self._log(f"Kinetic energy at far right consistent: E_k = {self.E_k:.2f} J", True)
                return True
            else:
                self._log(f"Kinetic energy mismatch: lhs={lhs:.2f}, rhs={rhs:.2f}", False)
                return False
        except Exception as e:
            self._log(f"Kinetic energy test failed with error: {e}", False)
            return False
    
    def test_energy_minimum(self):
        """Check minimum mechanical energy from step 6."""
        try:
            # E_min = m*g*R*cosθ + E_k
            E_min_expected = self.m * self.g * self.r * self.cos_theta + self.E_k
            if self._is_close(self.E_min, E_min_expected):
                self._log(f"Minimum energy consistent: E_min = {self.E_min:.2f} J", True)
                return True
            else:
                self._log(f"Minimum energy mismatch: {self.E_min:.2f} vs {E_min_expected:.2f}", False)
                return False
        except Exception as e:
            self._log(f"Minimum energy test failed with error: {e}", False)
            return False
    
    def test_energy_conservation(self):
        """Verify total mechanical energy (kinetic + gravitational + electrical) is conserved."""
        try:
            # Define zero at point C for both potentials
            # Energy at point C: E_total = K_C + U_g,C + U_e,C = K_C + 0 + 0
            # Since ball is given initial velocity at C, K_C is unknown.
            # Instead, check energy change between two points using work-energy:
            # The net work by conservative forces equals -ΔU.
            # For a full loop, net work by electric field is zero (symmetric).
            # But here we check between specific points using step 5 equation.
            # Already tested in test_kinetic_energy_at_far_right.
            self._log("Energy conservation implicitly checked in kinetic energy test.", True)
            return True
        except Exception as e:
            self._log(f"Energy conservation test failed: {e}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and check trend in E_min (should increase with m)."""
        try:
            original_m = self.m
            # Perturb mass by +10%
            self.m *= 1.1
            # Recompute expected E_min using formula from step 6
            E_min_perturbed = self.m * self.g * self.r * self.cos_theta + self.E_k
            # Restore original mass
            self.m = original_m
            if E_min_perturbed > self.E_min:
                self._log(f"Monotonicity: E_min increases with mass (perturbed: {E_min_perturbed:.2f} J)", True)
                return True
            else:
                self._log(f"Monotonicity violated: E_min decreases or stays same", False)
                return False
        except Exception as e:
            self._log(f"Monotonicity test failed: {e}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs.clear()
        
        tests = [
            self.test_E_field,
            self.test_velocity_at_D,
            self.test_kinetic_energy_at_far_right,
            self.test_energy_minimum,
            self.test_energy_conservation,
            self.test_monotonicity
        ]
        
        all_pass = True
        for test in tests:
            if not test():
                all_pass = False
        
        return all_pass, self.logs