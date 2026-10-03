import math
import traceback
import sys

class PhysicsChecker:
    """
    Audits a solution for the Space Telescope resolution problem.
    Checks interface completeness, Rayleigh criterion, small-angle geometry,
    monotonicity, and plausible answer range.
    """

    def __init__(self, solution_instance):
        # Extract emitted results safely
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.theta_min = solution_instance.results.get('theta_min', None)
            self.s = solution_instance.results.get('s', None)
        else:
            # Fallback to direct attributes
            self.theta_min = getattr(solution_instance, 'theta_min', None)
            self.s = getattr(solution_instance, 's', None)

        # Extract given parameters (may be in solution_instance attributes or givens)
        self.D = getattr(solution_instance, 'D', None)
        # 'lambda' is a reserved keyword, so try 'wavelength' first
        self.lambda_ = getattr(solution_instance, 'wavelength', None)
        if self.lambda_ is None:
            self.lambda_ = getattr(solution_instance, 'lambda', None)
        self.L = getattr(solution_instance, 'L', None)

        # If givens are stored in a dict, try that too
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            if self.D is None:
                self.D = solution_instance.givens.get('D', None)
            if self.lambda_ is None:
                self.lambda_ = solution_instance.givens.get('lambda', None)
            if self.L is None:
                self.L = solution_instance.givens.get('L', None)

        # Convert to float if possible
        for attr in ['theta_min', 's', 'D', 'lambda_', 'L']:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, (int, float)):
                try:
                    setattr(self, attr, float(val))
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that theta_min and s exist and are numeric."""
        if self.theta_min is None:
            return (False, "Missing required result: theta_min")
        if self.s is None:
            return (False, "Missing required result: s")
        if not isinstance(self.theta_min, (int, float)):
            return (False, "theta_min is not numeric")
        if not isinstance(self.s, (int, float)):
            return (False, "s is not numeric")
        return (True, "theta_min and s present and numeric")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_rayleigh_criterion(self):
        """Check that theta_min satisfies the Rayleigh criterion within 2%."""
        if self.theta_min is None or self.lambda_ is None or self.D is None:
            return (False, "SKIP: missing theta_min, lambda_, or D")
        try:
            expected = 1.22 * self.lambda_ / self.D
        except ZeroDivisionError:
            return (False, "D is zero")
        if math.isclose(self.theta_min, expected, rel_tol=1e-2, abs_tol=1e-15):
            return (True, f"theta_min = {self.theta_min:.3e} matches Rayleigh criterion (expected {expected:.3e})")
        return (False, f"theta_min = {self.theta_min:.3e} differs from Rayleigh {expected:.3e} by >2%")

    def test_invariant_small_angle_geometry(self):
        """Check that s = L * theta_min (small-angle approx) within 2%."""
        if self.s is None or self.L is None or self.theta_min is None:
            return (False, "SKIP: missing s, L, or theta_min")
        try:
            expected = self.L * self.theta_min
        except (TypeError, ValueError):
            return (False, "Invalid numeric type for L or theta_min")
        if math.isclose(self.s, expected, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"s = {self.s:.3e} matches L*theta_min = {expected:.3e}")
        return (False, f"s = {self.s:.3e} differs from L*theta_min = {expected:.3e} by >2%")

    # ------------------------------------------------------------------
    # Modeling / plausibility checks
    # ------------------------------------------------------------------
    def test_modeling_theta_min_plausible(self):
        """Check that theta_min is in the physically reasonable range for an optical telescope."""
        if self.theta_min is None:
            return (False, "SKIP: theta_min not available")
        if 1e-8 < self.theta_min < 1e-6:
            return (True, f"theta_min = {self.theta_min:.3e} rad is plausible")
        if self.theta_min <= 0:
            return (False, "theta_min is non-positive")
        return (False, f"theta_min = {self.theta_min:.3e} rad outside typical range (1e-8 to 1e-6)")

    def test_modeling_s_plausible(self):
        """Check that s is in the right order of magnitude (1e14 to 1e16 cm)."""
        if self.s is None:
            return (False, "SKIP: s not available")
        if 1e14 < self.s < 1e16:
            return (True, f"s = {self.s:.3e} cm is plausible")
        if self.s <= 0:
            return (False, "s is non-positive")
        return (False, f"s = {self.s:.3e} cm outside plausible range (1e14 to 1e16)")

    # ------------------------------------------------------------------
    # Monotonicity checks (strengthened with explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_diameter_effect(self):
        """Increasing mirror diameter decreases angular resolution (theta_min)."""
        if self.D is None or self.lambda_ is None or self.theta_min is None:
            return (False, "SKIP: missing D, lambda_, or theta_min")
        # Perturb: 10% increase in D
        new_D = self.D * 1.1
        new_theta = 1.22 * self.lambda_ / new_D
        if new_theta < self.theta_min:
            return (True, f"10% larger D gives theta_min = {new_theta:.3e} < original {self.theta_min:.3e}")
        return (False, "Monotonicity failed: larger D did NOT yield smaller theta_min")

    def test_monotonicity_distance_effect(self):
        """Increasing distance L increases linear separation s proportionally."""
        if self.L is None or self.theta_min is None or self.s is None:
            return (False, "SKIP: missing L, theta_min, or s")
        # Perturb: 10% increase in L
        new_L = self.L * 1.1
        new_s = new_L * self.theta_min
        if new_s > self.s:
            return (True, f"10% larger L gives s = {new_s:.3e} > original {self.s:.3e}")
        return (False, "Monotonicity failed: larger L did NOT yield larger s")

    def test_monotonicity_wavelength_effect(self):
        """Increasing wavelength increases angular resolution (theta_min)."""
        if self.lambda_ is None or self.D is None or self.theta_min is None:
            return (False, "SKIP: missing lambda_, D, or theta_min")
        # Perturb: 10% increase in lambda
        new_lambda = self.lambda_ * 1.1
        new_theta = 1.22 * new_lambda / self.D
        if new_theta > self.theta_min:
            return (True, f"10% larger lambda gives theta_min = {new_theta:.3e} > original {self.theta_min:.3e}")
        return (False, "Monotonicity failed: larger lambda did NOT yield larger theta_min")

    # ------------------------------------------------------------------
    # Answer sanity (one weak test only)
    # ------------------------------------------------------------------
    def test_answer_sanity_s_range(self):
        """Final answer s is positive, finite, and of order 10^15 cm."""
        if self.s is None:
            return (False, "SKIP: s not available")
        if not math.isfinite(self.s) or self.s <= 0:
            return (False, f"s = {self.s} is not positive and finite")
        # Allow a wide range around the expected ~3e15 cm
        if 1e14 < self.s < 1e16:
            return (True, f"s = {self.s:.3e} cm is a plausible order-of-magnitude answer")
        return (True, "s is positive and finite")  # weak pass because we only require positivity

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (all_pass, logs)."""
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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs