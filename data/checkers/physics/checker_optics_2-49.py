import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs and results safely with None defaults
        # Inputs: wavelength lambda, slit width a, slit separation d
        self.lambda_val = getattr(solution_instance, 'lambda', None)
        if self.lambda_val is None:
            self.lambda_val = solution_instance.results.get('lambda', None)
        self.a = getattr(solution_instance, 'a', None)
        if self.a is None:
            self.a = solution_instance.results.get('a', None)
        self.d = getattr(solution_instance, 'd', None)
        if self.d is None:
            self.d = solution_instance.results.get('d', None)

        # Results: theta_d1, theta_i1_min, theta_i1_max
        self.theta_d1 = getattr(solution_instance, 'theta_d1', None)
        if self.theta_d1 is None:
            self.theta_d1 = solution_instance.results.get('theta_d1', None)
        self.theta_i1_min = getattr(solution_instance, 'theta_i1_min', None)
        if self.theta_i1_min is None:
            self.theta_i1_min = solution_instance.results.get('theta_i1_min', None)
        self.theta_i1_max = getattr(solution_instance, 'theta_i1_max', None)
        if self.theta_i1_max is None:
            self.theta_i1_max = solution_instance.results.get('theta_i1_max', None)

        # Optional: explanation_steps for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Auxiliary computed quantities (for tests, recomputed from results if possible)
        self.sin_theta_d1 = None
        self.sin_theta_i1_min = None
        self.sin_theta_i1_max = None
        if self.theta_d1 is not None:
            self.sin_theta_d1 = math.sin(abs(self.theta_d1))
        if self.theta_i1_min is not None:
            self.sin_theta_i1_min = math.sin(abs(self.theta_i1_min))
        if self.theta_i1_max is not None:
            self.sin_theta_i1_max = math.sin(abs(self.theta_i1_max))

    # ------------------------------------------------------------------ #
    # Interface checks: required keys and types
    # ------------------------------------------------------------------ #
    def test_interface_required_inputs(self):
        """Check that essential input parameters exist and are numeric."""
        if self.lambda_val is None:
            return (False, "Missing input: wavelength (lambda)")
        if self.a is None:
            return (False, "Missing input: slit width (a)")
        if self.d is None:
            return (False, "Missing input: slit separation (d)")
        # Type/validity check
        for name, val in [("lambda", self.lambda_val), ("a", self.a), ("d", self.d)]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return (False, f"Invalid type/value for {name}: {val}")
        return (True, "Required inputs present and valid")

    def test_interface_required_results(self):
        """Check that all three output angles exist and are numeric."""
        if self.theta_d1 is None:
            return (False, "Missing result: theta_d1 (first diffraction minimum)")
        if self.theta_i1_min is None:
            return (False, "Missing result: theta_i1_min (first interference minimum)")
        if self.theta_i1_max is None:
            return (False, "Missing result: theta_i1_max (first interference maximum)")
        for name, val in [("theta_d1", self.theta_d1),
                          ("theta_i1_min", self.theta_i1_min),
                          ("theta_i1_max", self.theta_i1_max)]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return (False, f"Invalid type/value for {name}: {val}")
        return (True, "Required results present and valid")

    def test_interface_angle_bounds(self):
        """The angles should be between -90° and 90° (radians: -π/2 to π/2)."""
        half_pi = math.pi / 2
        for name, val in [("theta_d1", self.theta_d1),
                          ("theta_i1_min", self.theta_i1_min),
                          ("theta_i1_max", self.theta_i1_max)]:
            if val is None:
                continue
            if not (-half_pi <= val <= half_pi):
                return (False, f"{name} = {val} rad out of [-π/2, π/2]")
        return (True, "All angles within physical bounds")

    # ------------------------------------------------------------------ #
    # Invariant checks (theorem-based, using math.isclose)
    # ------------------------------------------------------------------ #
    def test_invariant_diffraction_minimum(self):
        """First diffraction minimum: sinθ = λ / a (sinc zero)."""
        if any(v is None for v in [self.lambda_val, self.a, self.sin_theta_d1]):
            return (False, "SKIP: missing required data for diffraction minimum invariant")
        expected = self.lambda_val / self.a
        if math.isclose(self.sin_theta_d1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sinθ_d1 = {self.sin_theta_d1:.4f} ≈ λ/a = {expected:.4f}")
        else:
            return (False, f"sinθ_d1 = {self.sin_theta_d1:.4f}, expected {expected:.4f} (diff > 1%)")

    def test_invariant_interference_minimum(self):
        """First interference minimum: |2cosφ-1|=0 => φ=π/3 => sinθ = λ/(6d)."""
        if any(v is None for v in [self.lambda_val, self.d, self.sin_theta_i1_min]):
            return (False, "SKIP: missing data for interference minimum invariant")
        expected = self.lambda_val / (6 * self.d)
        if math.isclose(self.sin_theta_i1_min, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sinθ_i1_min = {self.sin_theta_i1_min:.4f} ≈ λ/(6d) = {expected:.4f}")
        else:
            return (False, f"sinθ_i1_min = {self.sin_theta_i1_min:.4f}, expected {expected:.4f}")

    def test_invariant_interference_maximum(self):
        """First interference maximum: max of |2cosφ-1| occurs at φ=π => sinθ = λ/(2d)."""
        if any(v is None for v in [self.lambda_val, self.d, self.sin_theta_i1_max]):
            return (False, "SKIP: missing data for interference maximum invariant")
        expected = self.lambda_val / (2 * self.d)
        if math.isclose(self.sin_theta_i1_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sinθ_i1_max = {self.sin_theta_i1_max:.4f} ≈ λ/(2d) = {expected:.4f}")
        else:
            return (False, f"sinθ_i1_max = {self.sin_theta_i1_max:.4f}, expected {expected:.4f}")

    def test_invariant_phase_shift_effect(self):
        """Check that the interference factor matches |2cosφ-1|^2 pattern.
        This test verifies that the 180° phase shift is correctly accounted for by checking
        that the first interference minimum results from |2cosφ-1|=0, not |3cosφ| etc.
        We recompute φ = 2π d sinθ/λ for the given θ_i1_min and verify cosφ ≈ 1/2.
        """
        if any(v is None for v in [self.lambda_val, self.d, self.theta_i1_min]):
            return (False, "SKIP: missing data for phase shift check")
        phi = 2 * math.pi * self.d * math.sin(self.theta_i1_min) / self.lambda_val
        # Take absolute value of cos(phi) because phi may be negative
        cos_phi = math.cos(abs(phi))
        if math.isclose(cos_phi, 0.5, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"cos(phi) ≈ {cos_phi:.4f} ≈ 0.5 (confirms |2cosφ-1|=0)")
        else:
            return (False, f"Interference minimum: cos(phi) = {cos_phi:.4f}, expected 0.5")

    # ------------------------------------------------------------------ #
    # Modeling/trace checks (based on explanation steps)
    # ------------------------------------------------------------------ #
    def test_trace_intensity_factor(self):
        """Check that the intensity factor from explanation steps is consistent.
        The derived factor is [cos(3γ)/cos(γ)]^2, which has first zero at γ=π/6.
        For the interference minimum, we have γ = π/6, and for the maximum γ = π/2.
        Verify that the computed sinθ_i1_min gives γ ≈ π/6.
        """
        if any(v is None for v in [self.lambda_val, self.d, self.theta_i1_min]):
            return (False, "SKIP: missing data for trace intensity factor")
        gamma = math.pi * self.d * math.sin(self.theta_i1_min) / self.lambda_val
        expected_gamma = math.pi / 6  # 30°, cause of first zero
        if math.isclose(gamma, expected_gamma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"γ = {gamma:.4f} ≈ π/6 = {expected_gamma:.4f}, consistent with cos(3γ)=0")
        else:
            return (False, f"γ = {gamma:.4f} expected {expected_gamma:.4f}")

    def test_trace_relative_order(self):
        """Check relative ordering of angles for typical input (λ < a, λ < 6d).
        Usually first diffraction min is larger than interference max/min.
        But we only enforce the trivial: θ_i1_min < θ_i1_max because λ/(6d) < λ/(2d).
        """
        if any(v is None for v in [self.sin_theta_i1_min, self.sin_theta_i1_max]):
            return (False, "SKIP: missing data for relative order check")
        # Compare magnitudes (angles are symmetric)
        if self.sin_theta_i1_min < self.sin_theta_i1_max:
            return (True, "Interference minimum sinθ < interference maximum sinθ, as expected")
        else:
            return (False, f"Interference minimum sinθ ({self.sin_theta_i1_min:.4f}) "
                           f"should be less than maximum sinθ ({self.sin_theta_i1_max:.4f})")

    # ------------------------------------------------------------------ #
    # Monotonicity checks (trend via input perturbation)
    # ------------------------------------------------------------------ #
    def test_monotonicity_wavelength(self):
        """Increasing wavelength should increase all sinθ values (positive effect)."""
        if any(v is None for v in [self.lambda_val, self.a, self.d]):
            return (False, "SKIP: missing inputs for monotonicity check")
        # Perturb wavelength by +5%
        lambda_pert = self.lambda_val * 1.05
        # Compute expected sinθ after perturbation according to the formulas
        new_sin_d1 = lambda_pert / self.a
        new_sin_i1_min = lambda_pert / (6 * self.d)
        new_sin_i1_max = lambda_pert / (2 * self.d)

        # Baseline sin values from results (if available) or compute from theta
        if self.sin_theta_d1 is None or self.sin_theta_i1_min is None or self.sin_theta_i1_max is None:
            return (False, "SKIP: missing sinθ values for baseline")

        # Check monotonicity: new values > old values
        pass_checks = True
        msgs = []
        if not (new_sin_d1 > self.sin_theta_d1):
            pass_checks = False
            msgs.append(f"Diffraction: sinθ should increase (baseline {self.sin_theta_d1:.4f}, "
                        f"perturbed {new_sin_d1:.4f})")
        if not (new_sin_i1_min > self.sin_theta_i1_min):
            pass_checks = False
            msgs.append(f"Interference min: sinθ should increase (baseline {self.sin_theta_i1_min:.4f}, "
                        f"perturbed {new_sin_i1_min:.4f})")
        if not (new_sin_i1_max > self.sin_theta_i1_max):
            pass_checks = False
            msgs.append(f"Interference max: sinθ should increase (baseline {self.sin_theta_i1_max:.4f}, "
                        f"perturbed {new_sin_i1_max:.4f})")
        if pass_checks:
            return (True, "Wavelength increase leads to increased sinθ for all three cases")
        else:
            return (False, "; ".join(msgs))

    # ------------------------------------------------------------------ #
    # Weak answer sanity checks (no math.isclose, no re-derivations)
    # ------------------------------------------------------------------ #
    def test_answer_sanity_sign_and_magnitude(self):
        """Very weak check: all sinθ values are positive and less than 1."""
        if any(v is None for v in [self.sin_theta_d1, self.sin_theta_i1_min, self.sin_theta_i1_max]):
            return (False, "SKIP: missing sinθ values")
        # Check positivity (absolute values)
        if self.sin_theta_d1 <= 0 or self.sin_theta_i1_min <= 0 or self.sin_theta_i1_max <= 0:
            return (False, "One or more sinθ values are non-positive (should be positive)")
        # Check less than 1 (physical)
        if self.sin_theta_d1 >= 1 or self.sin_theta_i1_min >= 1 or self.sin_theta_i1_max >= 1:
            return (False, "One or more sinθ values exceed 1 (unphysical)")
        return (True, "All sinθ values are positive and less than 1 (physically reasonable)")

    # ------------------------------------------------------------------ #
    # Master test runner (do not modify, except adding new test methods above)
    # ------------------------------------------------------------------ #
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with test_
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)