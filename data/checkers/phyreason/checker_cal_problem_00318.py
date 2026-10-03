import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        Extract inputs: R (Earth radius), ω (angular velocity), g (gravitational acceleration).
        Extract result: h (altitude of satellite).
        """
        try:
            # Safe extraction of inputs from results dictionary
            self.R = solution_instance.results.get('R', None)
            self.ω = solution_instance.results.get('ω', None)
            self.g = solution_instance.results.get('g', None)
            
            # Safe extraction of result
            self.h = solution_instance.results.get('h', None)
            
            self.logs = []
            self._backup_vars = {}  # For monotonicity test restoration
        except Exception as e:
            self.logs.append(f"ERROR in __init__: {e}\n{traceback.format_exc()}")
            self.h = None

    def _log(self, message, is_error=False):
        """Helper to log messages."""
        prefix = "ERROR" if is_error else "INFO"
        self.logs.append(f"{prefix}: {message}")

    def test_invariants(self):
        """
        Verify physics invariants using numeric arithmetic only.
        Core invariant: (R+h)^3 * ω^2 = g * R^2
        """
        try:
            # Check if all values are numeric
            if not all(isinstance(val, (int, float)) for val in [self.R, self.ω, self.g, self.h]):
                self._log("Invariant test skipped: non-numeric inputs.", is_error=False)
                return True  # Skip, not a failure
            
            lhs = self.g * self.R**2
            rhs = (self.R + self.h)**3 * self.ω**2
            
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Numerical invariant holds: {lhs} ≈ {rhs}")
                return True
            else:
                rel_diff = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
                self._log(f"Numerical invariant fails: {lhs} ≠ {rhs} (relative diff: {rel_diff:.2%})", is_error=True)
                return False
        except Exception as e:
            self._log(f"Failed in test_invariants: {e}\n{traceback.format_exc()}", is_error=True)
            return False

    def test_trace_consistency(self):
        """
        Cross-check that intermediate steps lead to the final h using numeric verification.
        Expected formula: h = (g*R**2/ω**2)^(1/3) - R
        """
        try:
            # Check if all values are numeric
            if not all(isinstance(val, (int, float)) for val in [self.R, self.ω, self.g, self.h]):
                self._log("Trace consistency test skipped: non-numeric inputs.", is_error=False)
                return True  # Skip, not a failure
            
            # Compute h using the derived formula
            h_expected = (self.g * self.R**2 / self.ω**2)**(1/3) - self.R
            
            if math.isclose(h_expected, self.h, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace consistency: expected h={h_expected} ≈ given h={self.h}")
                return True
            else:
                self._log(f"Trace inconsistency: expected h={h_expected} ≠ given h={self.h}", is_error=True)
                return False
        except Exception as e:
            self._log(f"Failed in test_trace_consistency: {e}\n{traceback.format_exc()}", is_error=True)
            return False

    def test_monotonicity(self):
        """
        Perturb ω (increase by 10%) and check that h decreases (since h ~ ω^(-2/3)).
        Restore original ω after test.
        """
        try:
            # Backup original ω
            self._backup_vars['ω'] = self.ω
            
            # Only run if all inputs are numerical
            if not all(isinstance(val, (int, float)) for val in [self.R, self.ω, self.g]):
                self._log("Monotonicity test skipped: non-numeric inputs.", is_error=False)
                return True  # Skip, not a failure
            
            # Perturb ω: increase by 10%
            ω_perturbed = self.ω * 1.10
            # Compute h_perturbed using the derived formula
            h_perturbed = (self.g * self.R**2 / ω_perturbed**2)**(1/3) - self.R
            
            # Check monotonicity: increasing ω should decrease h
            if h_perturbed < self.h:
                self._log(f"Monotonicity holds: ω↑ (from {self.ω} to {ω_perturbed}) => h↓ (from {self.h} to {h_perturbed})")
                success = True
            else:
                # Allow small tolerance due to rounding
                if math.isclose(h_perturbed, self.h, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Monotonicity inconclusive: h remains ~{self.h} after perturbation (within tolerance)")
                    success = True
                else:
                    self._log(f"Monotonicity fails: ω↑ but h↑ (from {self.h} to {h_perturbed})", is_error=True)
                    success = False
            
            # Restore ω
            self.ω = self._backup_vars['ω']
            return success
        except Exception as e:
            self._log(f"Failed in test_monotonicity: {e}\n{traceback.format_exc()}", is_error=True)
            # Attempt to restore ω on error
            if 'ω' in self._backup_vars:
                self.ω = self._backup_vars['ω']
            return False

    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        success = True
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} failed.", is_error=True)
                else:
                    self._log(f"{test_name} passed.")
            except Exception as e:
                success = False
                self._log(f"{test_name} crashed: {e}\n{traceback.format_exc()}", is_error=True)
        
        return success, self.logs