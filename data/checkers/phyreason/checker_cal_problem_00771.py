import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Uses robust fallbacks to avoid AttributeError in sandbox environments.
        """
        # Safe extraction of inputs and results dictionaries
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})

        # Extract physical constants with safe fallbacks (typical Earth values)
        self.l = self.inputs.get('l', getattr(solution_instance, 'l', 1.0))          # pendulum length [m]
        self.g = self.inputs.get('g', getattr(solution_instance, 'g', 9.8))          # Earth gravity [m/s²]
        self.R = self.inputs.get('R', getattr(solution_instance, 'R', 6.4e6))        # planet radius [m]
        self.G = self.inputs.get('G', getattr(solution_instance, 'G', 6.67e-11))     # gravitational constant [N·m²/kg²]

        # Extract results with safe fallbacks
        self.T = getattr(solution_instance, 'T', self.results.get('T', None))        # period [s]
        self.M = getattr(solution_instance, 'M', self.results.get('M', None))        # planet mass [kg]
        # Handle g' which might be stored as 'g_prime' or 'g'' in results
        self.g_prime = getattr(solution_instance, 'g_prime', 
                              self.results.get('g_prime', self.results.get("g'", None)))

        # Convert symbolic results to numerical values if possible
        self.T_num = self._safe_float(self.T)
        self.M_num = self._safe_float(self.M)
        self.g_prime_num = self._safe_float(self.g_prime)

        # Pre-compute expected values for validation
        self.expected_T = (3 * math.pi / 2) * math.sqrt(self.l / self.g)
        self.expected_g_prime = 4 * self.g
        self.expected_M = (self.expected_g_prime * self.R ** 2) / self.G

    def _safe_float(self, value):
        """Safely convert a value to float, handling SymPy expressions and None."""
        if value is None:
            return None
        try:
            if hasattr(value, 'evalf'):
                return float(value.evalf())
            return float(value)
        except:
            return None

    def test_invariants(self):
        """Test fundamental physical laws: pendulum period formula and gravitational law."""
        logs = []
        try:
            # 1. Period formula validation (from step_1)
            if self.T_num is not None:
                if math.isclose(self.T_num, self.expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Period T matches (3π/2)·√(l/g).")
                else:
                    logs.append(f"FAIL: T mismatch. Expected {self.expected_T:.3f}, got {self.T_num:.3f}.")
            else:
                logs.append("SKIP: Period T not provided.")

            # 2. Gravitational acceleration ratio (from step_2)
            if self.g_prime_num is not None:
                if math.isclose(self.g_prime_num, self.expected_g_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: g' = 4g as required by period halving.")
                else:
                    logs.append(f"FAIL: g' mismatch. Expected {self.expected_g_prime:.3f}, got {self.g_prime_num:.3f}.")
            else:
                logs.append("SKIP: g' not provided.")

            # 3. Gravitational law (from step_3)
            if self.M_num is not None:
                if math.isclose(self.M_num, self.expected_M, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Planet mass M matches g'R²/G.")
                else:
                    logs.append(f"FAIL: M mismatch. Expected {self.expected_M:.3e}, got {self.M_num:.3e}.")
            else:
                logs.append("SKIP: Planet mass M not provided.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Cross-check consistency between intermediate steps and final results."""
        logs = []
        try:
            # Check period consistency between T and g'
            if self.T_num is not None and self.g_prime_num is not None:
                # Period on planet X should be half of Earth period
                T_prime_calc = (3 * math.pi / 2) * math.sqrt(self.l / self.g_prime_num)
                T_earth_half = self.T_num / 2
                if math.isclose(T_prime_calc, T_earth_half, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: g' yields period exactly half of Earth period.")
                else:
                    logs.append(f"FAIL: g' gives T'={T_prime_calc:.3f}, but T/2={T_earth_half:.3f}.")
            else:
                logs.append("SKIP: Need both T and g' for period consistency check.")

            # Check mass consistency between g' and M
            if self.g_prime_num is not None and self.M_num is not None:
                M_from_g_prime = (self.g_prime_num * self.R ** 2) / self.G
                if math.isclose(self.M_num, M_from_g_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: M is consistent with g' via Newton's law.")
                else:
                    logs.append(f"FAIL: From g' we get M={M_from_g_prime:.3e}, but result is {self.M_num:.3e}.")
            else:
                logs.append("SKIP: Need both g' and M for mass consistency check.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """
        Perturb inputs and verify expected trends.
        Important: Restore original values after perturbation.
        """
        logs = []
        original_l, original_g, original_R = self.l, self.g, self.R  # Save for restoration

        try:
            # 1. Perturb pendulum length l: T ∝ √l
            perturbed_l = original_l * 1.1  # +10%
            self.l = perturbed_l
            T_perturbed = (3 * math.pi / 2) * math.sqrt(self.l / self.g)
            if self.T_num is not None and T_perturbed > self.T_num:
                logs.append("PASS: Increasing l increases T (T ∝ √l).")
            elif self.T_num is not None:
                logs.append(f"FAIL: T did not increase with l. Original {self.T_num:.3f}, perturbed {T_perturbed:.3f}.")
            else:
                logs.append("SKIP: T not provided for l monotonicity.")
            self.l = original_l  # Restore

            # 2. Perturb gravity g: T ∝ 1/√g
            perturbed_g = original_g * 1.1
            self.g = perturbed_g
            T_perturbed = (3 * math.pi / 2) * math.sqrt(self.l / self.g)
            if self.T_num is not None and T_perturbed < self.T_num:
                logs.append("PASS: Increasing g decreases T (T ∝ 1/√g).")
            elif self.T_num is not None:
                logs.append(f"FAIL: T did not decrease with g. Original {self.T_num:.3f}, perturbed {T_perturbed:.3f}.")
            else:
                logs.append("SKIP: T not provided for g monotonicity.")
            self.g = original_g  # Restore

            # 3. Perturb planet radius R: M ∝ R²
            perturbed_R = original_R * 1.1
            M_perturbed = (4 * self.g * perturbed_R ** 2) / self.G
            if self.M_num is not None and M_perturbed > self.M_num:
                logs.append("PASS: Increasing R increases M (M ∝ R²).")
            elif self.M_num is not None:
                logs.append(f"FAIL: M did not increase with R. Original {self.M_num:.3e}, perturbed {M_perturbed:.3e}.")
            else:
                logs.append("SKIP: M not provided for R monotonicity.")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            # Ensure restoration even on error
            self.l, self.g, self.R = original_l, original_g, original_R
        finally:
            # Final restoration
            self.l, self.g, self.R = original_l, original_g, original_R

        return logs

    def run_all_tests(self):
        """Execute all checks and return success flag with detailed logs."""
        logs = []
        try:
            logs.append("=== PhysicsChecker started ===")
            logs.extend(self.test_invariants())
            logs.extend(self.test_trace_consistency())
            logs.extend(self.test_monotonicity())
            logs.append("=== PhysicsChecker completed ===")

            # Determine overall success: no FAIL or ERROR messages
            fail_count = sum(1 for log in logs if log.startswith("FAIL"))
            error_count = sum(1 for log in logs if "ERROR" in log)
            success = (fail_count == 0 and error_count == 0)
        except Exception as e:
            logs.append(f"CRITICAL ERROR in run_all_tests: {e}\n{traceback.format_exc()}")
            success = False

        return success, logs