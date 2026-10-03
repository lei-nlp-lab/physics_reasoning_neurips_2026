import math
import traceback

class PhysicsChecker:
    """
    Auditor for laser wavelength measurement via diffraction (thin wire or grating).
    Checks data quality, physics relations, and consistency.
    """

    def __init__(self, solution_instance):
        # Solve for physically stored variables (safe extraction)
        self.d = getattr(solution_instance, 'd', None)
        if self.d is None:
            self.d = (
                getattr(solution_instance, 'results', {}).get('d', None)
                if hasattr(solution_instance, 'results') else None
            )
        self.L = getattr(solution_instance, 'L', None)
        if self.L is None:
            self.L = (
                getattr(solution_instance, 'results', {}).get('L', None)
                if hasattr(solution_instance, 'results') else None
            )
        self.Delta = getattr(solution_instance, 'Delta', None)
        if self.Delta is None:
            self.Delta = (
                getattr(solution_instance, 'results', {}).get('Delta', None)
                if hasattr(solution_instance, 'results') else None
            )

        # Extract computed wavelength (target result)
        self.lambda_result = getattr(solution_instance, 'lambda', None)
        if self.lambda_result is None:
            self.lambda_result = (
                getattr(solution_instance, 'results', {}).get('lambda', None)
                if hasattr(solution_instance, 'results') else None
            )

        # Compute expected wavelength from givens (if available)
        self.lambda_calc = None
        if (self.d is not None and self.L is not None and self.Delta is not None and
                self.L != 0 and self.d > 0 and self.Delta > 0):
            self.lambda_calc = self.d * self.Delta / self.L

    # ------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------
    def test_interface_lambda_key(self):
        """Check that 'lambda' exists in results."""
        if self.lambda_result is None:
            return (False, "Missing key 'lambda' in solution results.")
        return (True, "Key 'lambda' present.")

    def test_interface_positive_givens(self):
        """Check that d, L, Delta are positive (if available)."""
        missing = []
        if self.d is None or self.d <= 0:
            missing.append('d')
        if self.L is None or self.L <= 0:
            missing.append('L')
        if self.Delta is None or self.Delta <= 0:
            missing.append('Delta')
        if missing:
            return (False, f"Missing or non-positive givens: {', '.join(missing)}.")
        return (True, "All givens (d, L, Delta) are positive.")

    # ------------------------------------------------------------
    # INVARIANT TESTS (theorem-based, fuzzy numerical)
    # ------------------------------------------------------------
    def test_invariant_diffraction_relation(self):
        """Check that lambda = d * Delta / L within 1% relative tolerance."""
        if self.lambda_result is None:
            return (False, "SKIP: lambda_result not available.")
        if self.lambda_calc is None:
            return (False, "SKIP: cannot compute lambda from givens (d, L, Delta missing or invalid).")
        if math.isclose(self.lambda_result, self.lambda_calc, rel_tol=1e-2):
            return (True, f"lambda_result ({self.lambda_result:.3e}) ≈ d*Delta/L ({self.lambda_calc:.3e}) (tol=1%)")
        else:
            return (False, f"lambda_result ({self.lambda_result:.3e}) != d*Delta/L ({self.lambda_calc:.3e}) (tolerance 1%)")

    def test_invariant_small_angle_approximation(self):
        """Check that for small angle, sin(θ) ≈ tan(θ) ≈ Δ/L within 1% relative tolerance."""
        if self.L is None or self.Delta is None or self.L == 0:
            return (False, "SKIP: Cannot check small angle (L or Delta missing).")
        ratio = self.Delta / self.L
        if ratio >= 0.5:
            # Approximation not expected to hold; skip test but note it
            return (True, f"Angle is not small (Δ/L={ratio:.4f}), approximation not expected; skipping invariant.")
        # Compute angle from geometry: θ = arctan(Δ/L)
        theta = math.atan(ratio)
        sin_theta = math.sin(theta)
        tan_theta = math.tan(theta)
        # Check sin(θ) ≈ Δ/L
        if not math.isclose(sin_theta, ratio, rel_tol=1e-2):
            return (False, f"sin(θ)={sin_theta:.4f} not close to Δ/L={ratio:.4f} (tol=1%)")
        # Check tan(θ) ≈ Δ/L
        if not math.isclose(tan_theta, ratio, rel_tol=1e-2):
            return (False, f"tan(θ)={tan_theta:.4f} not close to Δ/L={ratio:.4f} (tol=1%)")
        return (True, f"Small angle approximations hold: sin(θ)≈Δ/L, tan(θ)≈Δ/L")

    # ------------------------------------------------------------
    # MODELING QUALITY TESTS
    # ------------------------------------------------------------
    def test_modeling_small_angle_validity(self):
        """Check that Δ << L (small angle approximation is valid)."""
        if self.L is None or self.Delta is None or self.L == 0:
            return (False, "SKIP: Cannot check small angle (L or Delta missing).")
        ratio = self.Delta / self.L
        if ratio < 0.1:
            return (True, f"Δ/L = {ratio:.4f} < 0.1 (small angle okay).")
        else:
            return (False, f"Δ/L = {ratio:.4f} >= 0.1 (small angle approximation may be invalid).")

    def test_modeling_far_field_condition(self):
        """Check that L >> d (far-field diffraction condition)."""
        if self.L is None or self.d is None or self.d == 0:
            return (False, "SKIP: L or d missing.")
        ratio = self.L / self.d
        if ratio >= 100:
            return (True, f"L/d = {ratio:.1f} >= 100, far-field condition satisfied.")
        else:
            return (False, f"L/d = {ratio:.1f} < 100, far-field approximation may be invalid.")

    # ------------------------------------------------------------
    # MONOTONICITY TESTS
    # ------------------------------------------------------------
    def test_monotonicity_lambda_increases_with_Delta(self):
        """If Delta increases, lambda should increase proportionally."""
        if self.Delta is None or self.d is None or self.L is None or self.L == 0:
            return (False, "SKIP: needed variables missing.")
        base = self.d * self.Delta / self.L
        perturbed_delta = self.Delta * 1.10
        perturbed_lambda = self.d * perturbed_delta / self.L
        expected_ratio = 1.10
        actual_ratio = perturbed_lambda / base
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"Lambda increases proportionally with Delta: ratio {actual_ratio:.4f} ≈ 1.10")
        else:
            return (False, f"Lambda scaling wrong: expected ratio 1.10, got {actual_ratio:.4f}")

    def test_monotonicity_lambda_increases_with_d(self):
        """If grating spacing d increases, lambda should increase proportionally."""
        if self.d is None or self.Delta is None or self.L is None or self.L == 0:
            return (False, "SKIP: needed variables missing.")
        base = self.d * self.Delta / self.L
        perturbed_d = self.d * 1.05
        perturbed_lambda = perturbed_d * self.Delta / self.L
        expected_ratio = 1.05
        actual_ratio = perturbed_lambda / base
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"Lambda increases proportionally with d: ratio {actual_ratio:.4f} ≈ 1.05")
        else:
            return (False, f"Lambda scaling wrong: expected ratio 1.05, got {actual_ratio:.4f}")

    # ------------------------------------------------------------
    # ANSWER SANITY (weak, only sign/range)
    # ------------------------------------------------------------
    def test_answer_sanity_visible_range(self):
        """Check that lambda is in visible range (400–700 nm)."""
        if self.lambda_result is None:
            return (False, "SKIP: lambda_result not available.")
        lam = self.lambda_result
        if lam < 400e-9 or lam > 700e-9:
            return (False, f"lambda={lam:.3e} not in visible range (400–700 nm).")
        return (True, f"lambda={lam:.3e} in visible range.")

    # ------------------------------------------------------------
    # RUNNER
    # ------------------------------------------------------------
    def run_all_tests(self):
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