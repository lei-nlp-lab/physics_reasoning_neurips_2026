import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Never directly access solution_instance.attribute to avoid sandbox errors.
        """
        # Physical constants
        self.c = 3.0e8  # Speed of light in m/s
        self.T_interval = 100e-6  # Emission interval in s (100 μs)
        self.duration = 1e-6  # Emission duration in s (1 μs)
        
        # Extract results with safe fallbacks - NO automatic conversion
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        self.s = getattr(solution_instance, 's', solution_instance.results.get('s', 0.0))
        
        # Expected values from steps analysis (in seconds and meters)
        self.t_expected = 50e-6  # 50 μs in seconds
        self.s_expected = 7.5e3  # 7.5 × 10³ m
        
        # Initialize logs
        self.logs = []

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((message, success))
    
    def test_invariants(self):
        """Verify fundamental physical laws and relationships."""
        try:
            # 1. Check that s = c * t / 2 (distance from round-trip time)
            s_calculated = self.c * self.t / 2
            if not math.isclose(self.s, s_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance invariant violated: s={self.s}, c*t/2={s_calculated}", False)
                return False
            self._log("Distance formula s = c*t/2 holds", True)
            
            # 2. Check that round-trip time is consistent with emission interval
            # Given ab = bc in the figure, the round-trip time should be half the interval
            # So t should equal T_interval/2
            if not math.isclose(self.t, self.T_interval/2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Time invariant violated: t={self.t}, T_interval/2={self.T_interval/2}", False)
                return False
            self._log(f"Round-trip time t = T_interval/2 holds (t={self.t:.2e}s)", True)
            
            # 3. Check that t is less than emission interval (reflection arrives before next pulse)
            if self.t >= self.T_interval:
                self._log(f"Timing error: round-trip time t={self.t} >= emission interval {self.T_interval}", False)
                return False
            self._log(f"Round-trip time t < emission interval (100 μs)", True)
            
            return True
        except Exception as e:
            self._log(f"Invariant test failed with error: {traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # 1. Check t matches expected from step_1
            if not math.isclose(self.t, self.t_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Round-trip time mismatch: t={self.t}, expected={self.t_expected}", False)
                return False
            self._log(f"Round-trip time t={self.t:.2e}s matches expected {self.t_expected:.2e}s", True)
            
            # 2. Check s matches expected from step_2
            if not math.isclose(self.s, self.s_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance mismatch: s={self.s}, expected={self.s_expected}", False)
                return False
            self._log(f"Distance s={self.s:.2e}m matches expected {self.s_expected:.2e}m", True)
            
            # 3. Verify the calculation chain: s = c * t / 2 with expected values
            s_from_expected_t = self.c * self.t_expected / 2
            if not math.isclose(self.s_expected, s_from_expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Calculation chain broken: s_expected={self.s_expected}, c*t_expected/2={s_from_expected_t}", False)
                return False
            self._log("Calculation chain s = c*t/2 is consistent with expected values", True)
            
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """Perturb the round-trip time and verify the distance changes accordingly."""
        try:
            # Save original values
            t_original = self.t
            s_original = self.s
            
            # Perturb t by +1%
            t_perturbed = t_original * 1.01
            s_perturbed_expected = self.c * t_perturbed / 2
            
            # Check monotonic increase: if t increases, s should increase
            if not (s_perturbed_expected > s_original):
                self._log("Monotonicity violated: increasing t does not increase s", False)
                return False
            
            # Check the derivative ds/dt = c/2 (linear relationship)
            ds_dt = (s_perturbed_expected - s_original) / (t_perturbed - t_original)
            if not math.isclose(ds_dt, self.c/2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Derivative mismatch: ds/dt={ds_dt}, c/2={self.c/2}", False)
                return False
            
            # Check the perturbation magnitude is consistent
            s_recalculated = self.c * t_perturbed / 2
            if not math.isclose(s_perturbed_expected, s_recalculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Perturbation calculation mismatch", False)
                return False
            
            self._log("Monotonicity holds: s increases linearly with t (ds/dt = c/2)", True)
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {traceback.format_exc()}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...", None)
            if not test_func():
                all_passed = False
        
        return all_passed, self.logs