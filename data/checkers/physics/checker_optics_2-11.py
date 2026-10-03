import math
import traceback

class PhysicsChecker:
    """
    Checker for soap film thickness problem (thin film interference, constructive second order).
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely. Never use numeric fallbacks.
        # Assume solution_instance may have .givens dict/object or direct attributes.
        self.lambda_val = None
        self.n = None
        self.theta_i = None  # incidence angle in degrees
        self.m = None  # order for second-order constructive (m=1)

        # attempt to get givens from .givens
        givens = getattr(solution_instance, 'givens', None)
        if givens is not None:
            if hasattr(givens, 'get'):
                self.lambda_val = givens.get('lambda', None)
                self.n = givens.get('n', None)
                self.theta_i = givens.get('theta_i', None)
                self.m = givens.get('m', None)
            else:
                self.lambda_val = getattr(givens, 'lambda', None)
                self.n = getattr(givens, 'n', None)
                self.theta_i = getattr(givens, 'theta_i', None)
                self.m = getattr(givens, 'm', None)
        else:
            # fallback: try direct attributes on solution_instance
            self.lambda_val = getattr(solution_instance, 'lambda', None)
            self.n = getattr(solution_instance, 'n', None)
            self.theta_i = getattr(solution_instance, 'theta_i', None)
            self.m = getattr(solution_instance, 'm', None)

        # Extract thickness t from results
        self.t = None
        results = getattr(solution_instance, 'results', None)
        if results is not None:
            if hasattr(results, 'get'):
                self.t = results.get('t', None)
            else:
                self.t = getattr(results, 't', None)
        else:
            self.t = getattr(solution_instance, 't', None)

        # Second order constructive means m = 1 (condition uses m + 1/2)
        if self.m is None:
            self.m = 1

        # Convert all givens to float for computation if possible
        try:
            if self.lambda_val is not None:
                self.lambda_val = float(self.lambda_val)
            if self.n is not None:
                self.n = float(self.n)
            if self.theta_i is not None:
                self.theta_i = float(self.theta_i)
            if self.m is not None:
                self.m = int(self.m)
            if self.t is not None:
                self.t = float(self.t)
        except (TypeError, ValueError):
            pass  # leave as non-numeric if conversion fails

    def _sin_theta(self, angle_deg):
        """Return sin of angle in degrees."""
        return math.sin(math.radians(angle_deg))

    def _cos_theta(self, angle_deg):
        """Return cos of angle in degrees."""
        return math.cos(math.radians(angle_deg))

    def _compute_theta_t(self):
        """Compute angle of refraction inside film (degrees) using Snell's law.
           Returns None if missing givens or if total internal reflection occurs.
        """
        if self.n is None or self.theta_i is None or self.n == 0:
            return None
        sin_theta_i = self._sin_theta(self.theta_i)
        sin_theta_t = sin_theta_i / self.n
        if abs(sin_theta_t) > 1.0:
            return None
        return math.degrees(math.asin(sin_theta_t))

    def _compute_cos_theta_t(self):
        """Compute cos of angle of refraction inside film."""
        theta_t = self._compute_theta_t()
        if theta_t is None:
            return None
        return math.cos(math.radians(theta_t))

    # --------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------

    def test_interface_required_result_keys(self):
        """Check that results['t'] is present and numeric."""
        if self.t is None:
            return (False, "SKIP: t not available in results")
        if not isinstance(self.t, (int, float)):
            return (False, f"t is not a number: {type(self.t)}")
        return (True, "results['t'] exists and is numeric")

    def test_interface_givens_exist(self):
        """Check that required givens (lambda, n, theta_i) are provided."""
        missing = []
        if self.lambda_val is None:
            missing.append('lambda')
        if self.n is None:
            missing.append('n')
        if self.theta_i is None:
            missing.append('theta_i')
        if missing:
            return (False, f"SKIP: missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # --------------------------------------------------------------
    # Invariant tests (physical laws)
    # --------------------------------------------------------------

    def test_invariant_thin_film_constructive(self):
        """Check the constructive condition: 2 n t cos(theta_t) ≈ (m + 0.5) lambda."""
        if self.t is None or self.lambda_val is None or self.n is None or self.m is None:
            return (False, "SKIP: missing required quantities for invariant check")
        cos_theta_t = self._compute_cos_theta_t()
        if cos_theta_t is None:
            return (False, "SKIP: could not compute cos(theta_t) from Snell's law")
        left = 2.0 * self.n * self.t * cos_theta_t
        right = (self.m + 0.5) * self.lambda_val
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Constructive condition satisfied: left={left:.2f}, right={right:.2f}")
        else:
            return (False, f"Constructive condition violated: left={left:.2f}, right={right:.2f} (tolerance 1%)")

    def test_invariant_snells_law(self):
        """Check Snell's law: sin(theta_i) = n sin(theta_t)."""
        if self.n is None or self.theta_i is None:
            return (False, "SKIP: missing n or theta_i")
        theta_t = self._compute_theta_t()
        if theta_t is None:
            return (False, "SKIP: could not compute refraction angle")
        sin_i = self._sin_theta(self.theta_i)
        sin_t = self._sin_theta(theta_t)
        if math.isclose(sin_i, self.n * sin_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Snell's law holds: sin({self.theta_i}°) ≈ {self.n} * sin({theta_t:.2f}°)")
        else:
            return (False, f"Snell's law mismatch: sin_i={sin_i:.4f}, n*sin_t={self.n*sin_t:.4f}")

    # --------------------------------------------------------------
    # Modeling / trace consistency tests
    # --------------------------------------------------------------

    def test_modeling_derived_thickness_formula(self):
        """Re-derive thickness from formula t = (m+0.5)*lambda / (2 n cos(theta_t)) and compare to given t."""
        if self.t is None or self.lambda_val is None or self.n is None or self.m is None:
            return (False, "SKIP: missing required quantities")
        cos_theta_t = self._compute_cos_theta_t()
        if cos_theta_t is None or cos_theta_t == 0:
            return (False, "SKIP: could not compute cos(theta_t)")
        expected_t = ((self.m + 0.5) * self.lambda_val) / (2.0 * self.n * cos_theta_t)
        if math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derived thickness matches given: given={self.t:.2f} Å, derived={expected_t:.2f} Å")
        else:
            return (False, f"Derived thickness mismatch: given={self.t:.2f} Å, derived={expected_t:.2f} Å (tolerance 1%)")

    def test_modeling_cos_theta_reasonable(self):
        """Check that cos(theta_t) is computed correctly from Snell's law and is plausible."""
        cos_theta_t = self._compute_cos_theta_t()
        if cos_theta_t is None:
            return (False, "SKIP: could not compute cos(theta_t)")
        if 0 <= cos_theta_t <= 1:
            return (True, f"cos(theta_t) = {cos_theta_t:.4f} is between 0 and 1")
        else:
            return (False, f"cos(theta_t) = {cos_theta_t:.4f} outside [0,1]")

    # --------------------------------------------------------------
    # Monotonicity tests (each must perturb an input and compare)
    # --------------------------------------------------------------

    def test_monotonicity_thickness_increases_with_wavelength(self):
        """If lambda increases by 1%, computed thickness should increase by ~1%."""
        if self.t is None or self.lambda_val is None or self.n is None or self.m is None:
            return (False, "SKIP: missing required quantities")
        cos_theta_t = self._compute_cos_theta_t()
        if cos_theta_t is None:
            return (False, "SKIP: could not compute cos(theta_t)")
        # baseline thickness derived from current lambda
        baseline = ((self.m + 0.5) * self.lambda_val) / (2.0 * self.n * cos_theta_t)
        new_lambda = self.lambda_val * 1.01  # +1%
        perturbed_t = ((self.m + 0.5) * new_lambda) / (2.0 * self.n * cos_theta_t)
        if perturbed_t > baseline * 0.999:
            return (True, f"Thickness increased: baseline={baseline:.2f}, perturbed={perturbed_t:.2f} (lambda +1%)")
        else:
            return (False, f"Thickness did not increase: baseline={baseline:.2f}, perturbed={perturbed_t:.2f}")

    def test_monotonicity_thickness_decreases_with_index(self):
        """If refractive index increases by 1%, computed thickness should decrease."""
        if self.t is None or self.lambda_val is None or self.n is None or self.m is None:
            return (False, "SKIP: missing required quantities")
        cos_theta_t = self._compute_cos_theta_t()
        if cos_theta_t is None:
            return (False, "SKIP: could not compute cos(theta_t)")
        # baseline using current n
        baseline = ((self.m + 0.5) * self.lambda_val) / (2.0 * self.n * cos_theta_t)
        new_n = self.n * 1.01  # +1%
        # recompute cos_theta_t because refraction angle changes with n
        sin_theta_i = self._sin_theta(self.theta_i) if self.theta_i is not None else 0.5
        sin_theta_t_new = sin_theta_i / new_n
        if abs(sin_theta_t_new) > 1:
            return (False, "SKIP: perturbation leads to total internal reflection")
        cos_theta_t_new = math.sqrt(1 - sin_theta_t_new ** 2)
        perturbed_t = ((self.m + 0.5) * self.lambda_val) / (2.0 * new_n * cos_theta_t_new)
        if perturbed_t < baseline * 1.001:
            return (True, f"Thickness decreased: baseline={baseline:.2f}, perturbed={perturbed_t:.2f} (n +1%)")
        else:
            return (False, f"Thickness did not decrease: baseline={baseline:.2f}, perturbed={perturbed_t:.2f}")

    # --------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # --------------------------------------------------------------

    def test_answer_sanity_reasonable_magnitude(self):
        """Check that thickness is in the range of hundreds to thousands of angstroms."""
        if self.t is None:
            return (False, "SKIP: t not available")
        if not math.isfinite(self.t):
            return (False, f"t is not finite: {self.t}")
        # Typical thin film thickness: 10^2 to 10^4 Å, expected ~4260 Å
        if 100 < self.t < 10000:
            return (True, f"t = {self.t} Å is within reasonable range (100-10000 Å)")
        else:
            return (False, f"t = {self.t} Å is outside expected range")

    # --------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------

    def run_all_tests(self):
        """Discover all test_* methods and run them, collecting results."""
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)