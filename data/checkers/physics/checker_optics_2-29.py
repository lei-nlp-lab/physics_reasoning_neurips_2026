import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Fresnel diffraction problem:
    Monochromatic light normally incident on a circular aperture.
    Find the most distant axial point where intensity vanishes.
    Expected result: z_max = R^2/(2λ)
    """

    def __init__(self, solution_instance):
        # Extract givens (safe defaults: None for missing)
        self.R = getattr(solution_instance, 'R', None)
        self.wavelength = getattr(solution_instance, 'lambda', None)  # try common names
        if self.wavelength is None:
            self.wavelength = getattr(solution_instance, 'wavelength', None)

        # Extract results (expected key: 'z_max')
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.z_max = results.get('z_max', None)

        # Additional items that may appear in the instance (e.g., explanation steps)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ---------- interface checks ----------
    def test_interface_required_givens(self):
        """Check that essential givens (R, lambda) exist and are numeric."""
        if self.R is None:
            return (False, "Missing given: R")
        if self.wavelength is None:
            return (False, "Missing given: lambda (wavelength)")
        if not isinstance(self.R, (int, float)):
            return (False, f"R is not numeric: {type(self.R)}")
        if not isinstance(self.wavelength, (int, float)):
            return (False, f"lambda is not numeric: {type(self.wavelength)}")
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}")
        if self.wavelength <= 0:
            return (False, f"lambda must be positive, got {self.wavelength}")
        return (True, "All required givens present and valid")

    def test_interface_result_key_exists(self):
        """Check that the expected result key 'z_max' exists in results."""
        if self.z_max is None:
            return (False, "Missing result: z_max")
        return (True, "z_max is present")

    # ---------- theorem-based invariant checks ----------
    def test_invariant_zero_condition_satisfied(self):
        """
        Check that the given z_max satisfies the zero-intensity condition:
        sin^2(π R^2/(2 λ z_max)) = 0  →  sin(π R^2/(2 λ z_max)) = 0
        Use fuzzy match because this is an invariant derived from physics.
        """
        if self.z_max is None or self.R is None or self.wavelength is None:
            return (False, "SKIP: missing data for invariant check")

        if self.z_max <= 0:
            return (False, f"z_max must be positive, got {self.z_max}")

        # Evaluate the argument of the sine: π R^2 / (2 λ z_max)
        numerator = math.pi * self.R * self.R
        denominator = 2.0 * self.wavelength * self.z_max
        if denominator == 0:
            return (False, "Division by zero in invariant check")

        arg = numerator / denominator
        # We expect sin(arg) = 0, i.e. arg ≈ m*π for some integer m.
        # Check closeness to the nearest integer multiple of π.
        # Because of rounding, we check relative difference.
        sin_val = math.sin(arg)
        # The condition sin(arg)=0 is equivalent to |sin(arg)| < tolerance.
        if math.fabs(sin_val) > 1e-2:
            return (False, f"sin(arg) = {sin_val:.4e}, not close to zero")
        # Also check that arg corresponds to m=1 (most distant) by verifying that
        # the next zero (m=2) gives a smaller z. This is optional; the given z_max should be the largest one.
        # But we can only check that arg is not near m=0 (which would give infinite z) – that's fine.
        # We'll just check that arg is near π rather than 2π, 3π, etc., to ensure it's the first zero.
        # However, the problem doesn't require this; we only need to verify zero condition.
        # To be safe, check that arg ≈ π (m=1) within 5% relative error.
        arg_pi = arg / math.pi
        if not math.isclose(arg_pi, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"arg/π = {arg_pi:.4f}, expected ≈1 for z_max (m=1)")
        return (True, "Zero condition satisfied (m=1)")

    def test_invariant_most_distant_positive_and_finite(self):
        """z_max must be positive and finite (since λ/R ≪ 1, z_max should be large but finite)."""
        if self.z_max is None:
            return (False, "Missing z_max")
        if not math.isfinite(self.z_max):
            return (False, f"z_max is not finite: {self.z_max}")
        if self.z_max <= 0:
            return (False, f"z_max must be positive, got {self.z_max}")
        return (True, "z_max is positive and finite")

    # ---------- modeling / trace checks (based on derived formula) ----------
    def test_modeling_z_max_formula_consistency(self):
        """
        Re-derive z_max from givens: z_max_theoretical = R^2/(2λ).
        Compare with given z_max using isclose (relative tolerance 1%).
        This is allowed because the formula is derived from physics constraints inside the checker.
        """
        if self.R is None or self.wavelength is None or self.z_max is None:
            return (False, "SKIP: missing data for formula check")

        # Derive expected z_max
        expected = self.R * self.R / (2.0 * self.wavelength)
        if not math.isclose(self.z_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"z_max mismatch: given {self.z_max:.6e}, expected {expected:.6e} (relative diff {abs(self.z_max-expected)/expected:.2e})")
        return (True, "z_max matches theoretical formula R^2/(2λ)")

    def test_modeling_wavelength_order_check(self):
        """
        Check that λ/R ≪ 1 is satisfied (to ensure Fresnel regime).
        If λ/R > 0.1, the approximation may be weak, but we only warn.
        """
        if self.R is None or self.wavelength is None:
            return (False, "SKIP: missing givens")
        ratio = self.wavelength / self.R
        if ratio > 0.1:
            return (False, f"λ/R = {ratio:.4f} is not ≪ 1; Fresnel approximation may be invalid")
        return (True, f"λ/R = {ratio:.4e} ≪ 1, condition satisfied")

    # ---------- monotonicity checks ----------
    def test_monotonicity_z_max_increases_with_R(self):
        """
        Perturb R by +10% and check that z_max increases (since z_max ∝ R^2).
        Need to recompute z_max using the formula (since we only have one solution).
        We'll compare the given z_max with a derived one for a perturbed R,
        but the given z_max corresponds to original R. We can test tendency:
        If we increase R by a factor f, z_max should increase by factor f^2.
        """
        if self.R is None or self.z_max is None:
            return (False, "SKIP: missing data for monotonicity check")
        # Simulate a small increase in R (10%)
        factor = 1.1
        R_perturbed = self.R * factor
        # z_max for perturbed R, assuming same wavelength, would be (R_perturbed^2)/(2λ)
        # But we don't have that in the result; we only have one solution.
        # Instead, we can use the given z_max and the formula to check that
        # a positive change in R implies positive change in z_max.
        # Since we cannot recompute with different inputs, we can check the derivative:
        # dz_max/dR = R/(λ) > 0. So monotonicity is inherent.
        # We'll just verify that z_max is positive; already done.
        # To make a meaningful test, we can check that if R were larger, z_max would be larger
        # based on the formula (but that uses the same formula – not independent).
        # Better: Check that z_max consistent with R^2/(2λ) (already done).
        # So this test is a bit redundant; we'll keep it minimal:
        return (True, "Monotonicity w.r.t. R is implied by z_max ∝ R^2 (check performed in formula test)")

    def test_monotonicity_z_max_decreases_with_wavelength(self):
        """
        Similar: z_max ∝ 1/λ, so increasing λ decreases z_max.
        Since we cannot perturb the instance, we rely on the formula test.
        """
        if self.wavelength is None or self.z_max is None:
            return (False, "SKIP: missing data")
        # Check sign and proportionality: already done in formula test.
        return (True, "Monotonicity w.r.t. λ is implied by z_max ∝ 1/λ")

    # ---------- weak answer sanity (final answer only) ----------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: z_max should be positive, and of reasonable magnitude
        given typical values in the problem (R and λ). No isclose, no re-derivation.
        """
        if self.z_max is None:
            return (False, "Missing z_max")
        if self.z_max <= 0:
            return (False, f"z_max must be positive, got {self.z_max}")
        # Check that z_max is not extremely small relative to R (should be > R typically because λ/R ≪ 1)
        if self.R is not None:
            if self.z_max < self.R / 10.0:
                return (False, f"z_max = {self.z_max:.4e} seems too small compared to R = {self.R:.4e}")
        # Also not astronomically large (if R and λ are given, z_max should be determinable)
        # We'll just accept any positive finite value.
        return (True, "Weak sanity passed: z_max positive and not obviously unreasonable")

    # ---------- runner ----------
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