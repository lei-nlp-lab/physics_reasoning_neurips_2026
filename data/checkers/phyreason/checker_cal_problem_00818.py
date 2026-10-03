import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary results from the solution instance.
        Uses getattr and .get() to avoid AttributeError.
        Converts numeric values to float where possible.
        """
        self.solution = solution_instance
        # Get results dictionary, default to empty dict if not present
        results = getattr(solution_instance, 'results', {})
        
        # Extract required quantities with safe access
        self.lambda_val = float(getattr(solution_instance, 'λ', results.get('λ', 0.0)))
        self.T_period = float(getattr(solution_instance, 'T', results.get('T', 0.0)))
        self.v = float(getattr(solution_instance, 'v', results.get('v', 0.0)))
        self.x_Q = float(getattr(solution_instance, 'x_Q', results.get('x_Q', 0.0)))
        self.phi = float(getattr(solution_instance, 'phi', results.get('phi', 0.0)))
        self.y_t0 = float(getattr(solution_instance, 'y', results.get('y', 0.0)))
        self.t_eq = float(getattr(solution_instance, 't', results.get('t', 0.0)))
        self.delta_x = float(getattr(solution_instance, 'delta_x', results.get('delta_x', 0.0)))
        
        # Constants from problem (amplitude in cm, but we'll use in meters for consistency)
        self.A = 0.02  # 2 cm in meters
        self.omega = math.pi / 4  # rad/s, from T=8 s
        
        # Tolerance for numerical comparisons (1-2% relative)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        self.logs = []

    def _log(self, message, success=None):
        """Helper to log messages and test results."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def test_invariants(self):
        """Test fundamental physical invariants: wave speed, phase, equilibrium time."""
        success = True
        try:
            # 1. Wave speed relation: v = λ / T
            v_expected = self.lambda_val / self.T_period
            if not math.isclose(self.v, v_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Wave speed mismatch: v={self.v}, expected λ/T={v_expected}", False)
                success = False
            else:
                self._log(f"Wave speed v = λ/T holds: {self.v} ≈ {v_expected}", True)
            
            # 2. Phase from initial displacement: cos(φ) = y_t0 / A
            # Note: y_t0 and A must be in same units. We assume stored y_t0 is in meters (0.01 for 1 cm).
            cos_phi_expected = self.y_t0 / self.A
            if not math.isclose(math.cos(self.phi), cos_phi_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Phase mismatch: cos(φ)={math.cos(self.phi)}, expected y_t0/A={cos_phi_expected}", False)
                success = False
            else:
                self._log(f"Phase φ satisfies cos(φ) = y_t0/A: {math.cos(self.phi)} ≈ {cos_phi_expected}", True)
            
            # 3. Time to equilibrium: from ω t + φ = π/2 => t = (π/2 - φ)/ω
            t_expected = (math.pi/2 - self.phi) / self.omega
            if not math.isclose(self.t_eq, t_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Equilibrium time mismatch: t={self.t_eq}, expected (π/2 - φ)/ω={t_expected}", False)
                success = False
            else:
                self._log(f"Equilibrium time t = (π/2 - φ)/ω holds: {self.t_eq} ≈ {t_expected}", True)
            
            # 4. Distance traveled by wave: Δx = v * t
            delta_x_expected = self.v * self.t_eq
            if not math.isclose(self.delta_x, delta_x_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Δx mismatch: Δx={self.delta_x}, expected v*t={delta_x_expected}", False)
                success = False
            else:
                self._log(f"Δx = v*t holds: {self.delta_x} ≈ {delta_x_expected}", True)
                
        except Exception as e:
            self._log(f"Exception in test_invariants: {e}", False)
            success = False
        return success

    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        success = True
        try:
            # 1. Verify that the initial equilibrium position is 8 m (from step 9)
            initial_equilibrium = self.x_Q - self.delta_x
            if not math.isclose(initial_equilibrium, 8.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Initial equilibrium position mismatch: x_Q - Δx = {initial_equilibrium}, expected 8.0", False)
                success = False
            else:
                self._log(f"Initial equilibrium position = x_Q - Δx = {initial_equilibrium} ≈ 8.0", True)
            
            # 2. Verify that angular frequency ω = 2π/T
            omega_expected = 2 * math.pi / self.T_period
            if not math.isclose(self.omega, omega_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Angular frequency mismatch: ω={self.omega}, expected 2π/T={omega_expected}", False)
                success = False
            else:
                self._log(f"Angular frequency ω = 2π/T holds: {self.omega} ≈ {omega_expected}", True)
            
            # 3. Verify that the vibration equation at t=0 gives correct displacement
            y_t0_expected = self.A * math.cos(self.phi)
            if not math.isclose(self.y_t0, y_t0_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Initial displacement mismatch: y(0)={self.y_t0}, expected A*cos(φ)={y_t0_expected}", False)
                success = False
            else:
                self._log(f"Initial displacement y(0) = A*cos(φ) holds: {self.y_t0} ≈ {y_t0_expected}", True)
                
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {e}", False)
            success = False
        return success

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends in derived quantities."""
        success = True
        try:
            # Test 1: Increasing wavelength λ should increase wave speed v (with T fixed)
            original_v = self.v
            original_lambda = self.lambda_val
            new_lambda = original_lambda * 1.1  # +10%
            new_v = new_lambda / self.T_period
            if not new_v > original_v:
                self._log(f"Monotonicity fail: λ↑ from {original_lambda} to {new_lambda} did not cause v↑ (v={original_v} -> {new_v})", False)
                success = False
            else:
                self._log(f"λ↑ from {original_lambda} to {new_lambda} causes v↑ from {original_v} to {new_v}", True)
            
            # Test 2: Increasing period T should decrease wave speed v (with λ fixed)
            original_T = self.T_period
            new_T = original_T * 1.1  # +10%
            new_v2 = self.lambda_val / new_T
            if not new_v2 < original_v:
                self._log(f"Monotonicity fail: T↑ from {original_T} to {new_T} did not cause v↓ (v={original_v} -> {new_v2})", False)
                success = False
            else:
                self._log(f"T↑ from {original_T} to {new_T} causes v↓ from {original_v} to {new_v2}", True)
            
            # Test 3: Increasing amplitude A should increase displacement magnitude (at fixed phase)
            original_y = self.A * math.cos(self.phi)
            new_A = self.A * 1.2  # +20%
            new_y = new_A * math.cos(self.phi)
            if not abs(new_y) > abs(original_y):
                self._log(f"Monotonicity fail: A↑ from {self.A} to {new_A} did not cause |y|↑ (|y|={abs(original_y)} -> {abs(new_y)})", False)
                success = False
            else:
                self._log(f"A↑ from {self.A} to {new_A} causes |y|↑ from {abs(original_y)} to {abs(new_y)}", True)
                
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {e}", False)
            success = False
        return success

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        overall_success = True
        
        self._log("Starting PhysicsChecker for wave propagation problem.")
        
        # Run individual tests
        test_functions = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in test_functions:
            self._log(f"Running {name} test...")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Unexpected error in {name}: {traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"All tests completed. Overall success: {overall_success}")
        return overall_success, self.logs