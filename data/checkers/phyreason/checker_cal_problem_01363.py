import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results with fallbacks."""
        self.sol = solution_instance
        
        # Given constants (safe extraction)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.M = getattr(solution_instance, 'M', 4.0)
        self.H = getattr(solution_instance, 'H', 0.6)
        self.h = getattr(solution_instance, 'h', 0.15)
        self.R = getattr(solution_instance, 'R', 0.75)
        self.mu1 = getattr(solution_instance, 'mu1', 0.7)
        self.mu2 = getattr(solution_instance, 'mu2', 0.2)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        
        # Extract results with fallback to solution_instance.results dictionary
        results = getattr(solution_instance, 'results', {})
        self.v0 = getattr(solution_instance, 'v_0', results.get('v_0', 0.0))
        self.v1 = getattr(solution_instance, 'v_1', results.get('v_1', 0.0))
        self.v2 = getattr(solution_instance, 'v_2', results.get('v_2', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', 0.0))
        self.F_pressure = getattr(solution_instance, 'F_pressure', results.get('F_pressure', 0.0))
        self.l = getattr(solution_instance, 'l', results.get('l', 0.0))
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.v_y = getattr(solution_instance, 'v_y', results.get('v_y', 0.0))
        self.tantheta = getattr(solution_instance, 'tanθ', results.get('tanθ', results.get('tantheta', 0.0)))
        self.F_f = getattr(solution_instance, 'F_f', results.get('F_f', 0.0))
        self.F_f_prime = getattr(solution_instance, "F_f''", results.get("F_f''", results.get('F_f_prime', 0.0)))
        
        # Store logs
        self.logs = []
        
    def _log(self, msg, success=None):
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Energy conservation from A to C (smooth track)
            v2_sq_expected = self.v0**2 + 2*self.g*self.H
            energy_ok = math.isclose(self.v2**2, v2_sq_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Energy conservation: v2² = {self.v2**2:.3f}, expected {v2_sq_expected:.3f}", energy_ok)
            
            # 2. Centripetal force at C (Newton's 2nd Law)
            F_N_expected = self.m * self.g + self.m * self.v2**2 / self.R
            centripetal_ok = math.isclose(self.F_N, F_N_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Centripetal force at C: F_N = {self.F_N:.3f}, expected {F_N_expected:.3f}", centripetal_ok)
            
            # 3. Newton's 3rd Law: Pressure equals normal force
            third_law_ok = math.isclose(self.F_pressure, self.F_N, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Newton's 3rd Law: F_pressure = {self.F_pressure:.3f}, F_N = {self.F_N:.3f}", third_law_ok)
            
            # 4. Projectile motion kinematics from A to B
            t_expected = math.sqrt(2*(self.H - self.h)/self.g)
            time_ok = math.isclose(self.t, t_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Free-fall time A→B: t = {self.t:.3f}, expected {t_expected:.3f}", time_ok)
            
            v_y_expected = self.g * self.t
            vy_ok = math.isclose(self.v_y, v_y_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Vertical velocity at B: v_y = {self.v_y:.3f}, expected {v_y_expected:.3f}", vy_ok)
            
            # 5. Trigonometry at B
            tantheta_expected = self.v_y / self.v0 if self.v0 != 0 else 0
            trig_ok = math.isclose(self.tantheta, 0.75, rel_tol=0.02) and math.isclose(tantheta_expected, 0.75, rel_tol=0.02)
            self._log(f"tanθ at B: {self.tantheta:.3f}, expected ~0.75", trig_ok)
            
            # 6. Velocity magnitude at B
            v1_expected = math.sqrt(self.v0**2 + self.v_y**2)
            v1_ok = math.isclose(self.v1, v1_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Speed at B: v1 = {self.v1:.3f}, expected {v1_expected:.3f}", v1_ok)
            
            # 7. Friction forces
            F_f_expected = self.mu1 * self.m * self.g
            friction1_ok = math.isclose(self.F_f, F_f_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Block-plank friction: F_f = {self.F_f:.3f}, expected {F_f_expected:.3f}", friction1_ok)
            
            F_f_prime_expected = self.mu2 * (self.M + self.m) * self.g
            friction2_ok = math.isclose(self.F_f_prime, F_f_prime_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Plank-ground max friction: F_f' = {self.F_f_prime:.3f}, expected {F_f_prime_expected:.3f}", friction2_ok)
            
            # 8. Stopping distance on plank
            a_friction = -self.mu1 * self.g
            l_expected = -self.v2**2 / (2 * a_friction) if a_friction != 0 else 0
            stopping_ok = math.isclose(self.l, l_expected, rel_tol=0.02, abs_tol=0.02)
            self._log(f"Stopping distance: l = {self.l:.3f}, expected {l_expected:.3f}", stopping_ok)
            
            return all([energy_ok, centripetal_ok, third_law_ok, time_ok, vy_ok, trig_ok, v1_ok, friction1_ok, friction2_ok, stopping_ok])
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check that v0 and v1 match the first target answer
            v0_ok = math.isclose(self.v0, 4.0, rel_tol=0.02, abs_tol=0.02)
            v1_ok = math.isclose(self.v1, 5.0, rel_tol=0.02, abs_tol=0.02)
            self._log(f"v0 = {self.v0:.3f} (target 4.0)", v0_ok)
            self._log(f"v1 = {self.v1:.3f} (target 5.0)", v1_ok)
            
            # Check that F_pressure matches the second target answer
            F_target = 47.3
            pressure_ok = math.isclose(self.F_pressure, F_target, rel_tol=0.02, abs_tol=0.02)
            self._log(f"F_pressure = {self.F_pressure:.3f} (target 47.3)", pressure_ok)
            
            # Check that l matches the third target answer
            l_target = 2.0
            l_ok = math.isclose(self.l, l_target, rel_tol=0.02, abs_tol=0.02)
            self._log(f"l = {self.l:.3f} (target 2.0)", l_ok)
            
            # Additional consistency: v2 from energy should match v2 used in other calculations
            v2_from_energy = math.sqrt(self.v0**2 + 2*self.g*self.H)
            v2_consistency = math.isclose(self.v2, v2_from_energy, rel_tol=0.02, abs_tol=0.02)
            self._log(f"v2 from energy: {v2_from_energy:.3f}, stored v2: {self.v2:.3f}", v2_consistency)
            
            return all([v0_ok, v1_ok, pressure_ok, l_ok, v2_consistency])
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        original_m = self.m
        perturbed_results = []
        
        try:
            # Test 1: Increase mass m by 10%
            self.m *= 1.1
            
            # Recalculate expected values using physics formulas
            # v0 should remain unchanged (determined by projectile motion)
            t_pert = math.sqrt(2*(self.H - self.h)/self.g)
            v_y_pert = self.g * t_pert
            v0_pert = v_y_pert / 0.75  # tanθ fixed at 3/4
            
            # v1 from Pythagoras
            v1_pert = math.sqrt(v0_pert**2 + v_y_pert**2)
            
            # v2 from energy conservation
            v2_pert = math.sqrt(v0_pert**2 + 2*self.g*self.H)
            
            # F_N from centripetal force
            F_N_pert = self.m * self.g + self.m * v2_pert**2 / self.R
            
            # l from stopping distance
            l_pert = v2_pert**2 / (2 * self.mu1 * self.g)
            
            perturbed_results = [v0_pert, v1_pert, F_N_pert, l_pert]
            
            # Restore original mass
            self.m = original_m
            
            # Check trends
            # v0 and v1 should be unchanged (within tolerance)
            v0_trend = math.isclose(v0_pert, self.v0, rel_tol=0.02, abs_tol=0.02)
            v1_trend = math.isclose(v1_pert, self.v1, rel_tol=0.02, abs_tol=0.02)
            
            # F_N should increase proportionally to mass (10% increase)
            expected_F_N_increase = self.F_N * 1.1
            F_N_trend = math.isclose(F_N_pert, expected_F_N_increase, rel_tol=0.02, abs_tol=0.02)
            
            # l should remain unchanged
            l_trend = math.isclose(l_pert, self.l, rel_tol=0.02, abs_tol=0.02)
            
            self._log(f"Perturbed v0: {v0_pert:.3f} (original: {self.v0:.3f})", v0_trend)
            self._log(f"Perturbed v1: {v1_pert:.3f} (original: {self.v1:.3f})", v1_trend)
            self._log(f"Perturbed F_N: {F_N_pert:.3f} (expected ~{expected_F_N_increase:.3f})", F_N_trend)
            self._log(f"Perturbed l: {l_pert:.3f} (original: {self.l:.3f})", l_trend)
            
            return all([v0_trend, v1_trend, F_N_trend, l_trend])
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}")
            # Restore mass if an error occurred
            self.m = original_m
            return False
    
    def run_all_tests(self):
        """Execute all tests and return overall success."""
        self.logs = []
        overall_success = True
        
        try:
            self._log("="*50)
            self._log("Starting Physics Audit")
            self._log("="*50)
            
            # Run tests
            inv_success = self.test_invariants()
            self._log(f"Invariant tests: {'PASS' if inv_success else 'FAIL'}")
            overall_success &= inv_success
            
            trace_success = self.test_trace_consistency()
            self._log(f"Trace consistency tests: {'PASS' if trace_success else 'FAIL'}")
            overall_success &= trace_success
            
            mono_success = self.test_monotonicity()
            self._log(f"Monotonicity tests: {'PASS' if mono_success else 'FAIL'}")
            overall_success &= mono_success
            
            self._log("="*50)
            self._log(f"Overall audit: {'PASS' if overall_success else 'FAIL'}")
            self._log("="*50)
            
        except Exception as e:
            self._log(f"Fatal error in audit: {traceback.format_exc()}")
            overall_success = False
        
        return overall_success, self.logs