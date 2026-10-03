import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: Extract all required inputs and results using safe access patterns.
        """
        try:
            # Extract inputs with safe defaults
            self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 1.7374e6))
            self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 1e5))
            self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', 7200.0))
            self.G = getattr(solution_instance, 'G', solution_instance.results.get('G', 6.6743e-11))
            
            # Extract results with safe defaults
            self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))
            self.rho = getattr(solution_instance, 'rho', solution_instance.results.get('rho', None))
            self.V = getattr(solution_instance, 'V', solution_instance.results.get('V', None))
            self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', None))
            
            # Store for trace consistency
            self.results = {
                'v': self.v,
                'rho': self.rho,
                'V': self.V,
                'M': self.M
            }
            
            self.logs = []
            
        except Exception as e:
            self.logs.append(f"Initialization error: {traceback.format_exc()}")
            raise

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching for physics data."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Verify physical invariants: circular motion and gravitational laws."""
        successes = []
        
        # 1. Check circular motion relation: v = 2π(R+h)/T
        try:
            if self.v is not None:
                expected_v = 2 * math.pi * (self.R + self.h) / self.T
                success = self._is_close(self.v, expected_v)
                successes.append(self._log(f"Circular motion relation v = 2π(R+h)/T", success))
            else:
                successes.append(self._log("Circular motion check skipped - missing v", False))
        except Exception as e:
            successes.append(self._log(f"Circular motion check failed: {e}", False))

        # 2. Check gravitational centripetal force balance: (2π/T)²(R+h) = GM/(R+h)²
        try:
            if self.M is not None:
                left_side = (2 * math.pi / self.T)**2 * (self.R + self.h)
                right_side = self.G * self.M / (self.R + self.h)**2
                success = self._is_close(left_side, right_side)
                successes.append(self._log("Gravitational force balance (Newton's 2nd Law)", success))
            else:
                successes.append(self._log("Gravitational balance check skipped - missing M", False))
        except Exception as e:
            successes.append(self._log(f"Gravitational balance check failed: {e}", False))

        # 3. Check density formula: ρ = 3π(R+h)³/(G T² R³)
        try:
            if self.rho is not None:
                expected_rho = 3 * math.pi * (self.R + self.h)**3 / (self.G * self.T**2 * self.R**3)
                success = self._is_close(self.rho, expected_rho)
                successes.append(self._log("Density formula ρ = 3π(R+h)³/(G T² R³)", success))
            else:
                successes.append(self._log("Density formula check skipped - missing ρ", False))
        except Exception as e:
            successes.append(self._log(f"Density formula check failed: {e}", False))

        return all(successes)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        successes = []
        
        # 1. Check volume formula if available: V = (4/3)πR³
        try:
            if self.V is not None:
                expected_V = (4/3) * math.pi * self.R**3
                success = self._is_close(self.V, expected_V)
                successes.append(self._log("Volume V = (4/3)πR³", success))
            else:
                successes.append(self._log("Volume check skipped - missing V", False))
        except Exception as e:
            successes.append(self._log(f"Volume check failed: {e}", False))

        # 2. Check density from mass and volume: ρ = M/V
        try:
            if all(x is not None for x in [self.rho, self.M, self.V]):
                rho_from_MV = self.M / self.V
                success = self._is_close(self.rho, rho_from_MV)
                successes.append(self._log("Density ρ = M/V", success))
            else:
                successes.append(self._log("Density consistency check skipped - missing data", False))
        except Exception as e:
            successes.append(self._log(f"Density consistency check failed: {e}", False))

        # 3. Check that the derived ρ matches the gravitational law expression
        try:
            if self.rho is not None:
                # From gravitational law: M = 4π²(R+h)³/(G T²)
                # Then ρ = M/V = [4π²(R+h)³/(G T²)] / [(4/3)πR³] = 3π(R+h)³/(G T² R³)
                expected_rho = 3 * math.pi * (self.R + self.h)**3 / (self.G * self.T**2 * self.R**3)
                success = self._is_close(self.rho, expected_rho)
                successes.append(self._log("ρ derived from gravitational law matches final ρ", success))
            else:
                successes.append(self._log("Gravitational derivation check skipped - missing ρ", False))
        except Exception as e:
            successes.append(self._log(f"Gravitational derivation check failed: {e}", False))

        return all(successes)

    def test_monotonicity(self):
        """Check monotonic trends by perturbing inputs."""
        successes = []
        
        try:
            # Perturb h and check v increases (for fixed R, T)
            if all(x is not None for x in [self.R, self.h, self.T]):
                h_perturbed = self.h * 1.01  # +1%
                v_original = 2 * math.pi * (self.R + self.h) / self.T
                v_perturbed = 2 * math.pi * (self.R + h_perturbed) / self.T
                success = v_perturbed > v_original
                successes.append(self._log("v increases with height h (positive derivative)", success))
            else:
                successes.append(self._log("v monotonicity check skipped - missing data", False))
        except Exception as e:
            successes.append(self._log(f"v monotonicity check failed: {e}", False))

        try:
            # Perturb h and check ρ increases (for fixed R, T, G)
            if all(x is not None for x in [self.R, self.h, self.T, self.G]):
                h_perturbed = self.h * 1.01  # +1%
                rho_original = 3 * math.pi * (self.R + self.h)**3 / (self.G * self.T**2 * self.R**3)
                rho_perturbed = 3 * math.pi * (self.R + h_perturbed)**3 / (self.G * self.T**2 * self.R**3)
                success = rho_perturbed > rho_original
                successes.append(self._log("ρ increases with height h (positive derivative)", success))
            else:
                successes.append(self._log("ρ monotonicity check skipped - missing data", False))
        except Exception as e:
            successes.append(self._log(f"ρ monotonicity check failed: {e}", False))

        return all(successes)

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        success = True
        
        try:
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self.logs.append(f"CRITICAL: Test execution failed: {traceback.format_exc()}")
            success = False
        
        return success, self.logs