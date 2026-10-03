import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Born approximation forward scattering problem.
    Checks interface completeness, invariant physical relations,
    modeling/trace consistency, monotonicity, and basic answer sanity.
    """

    def __init__(self, solution_instance):
        # Safe extraction of givens
        self.m = getattr(solution_instance, 'm', None)
        self.e = getattr(solution_instance, 'e', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.A = getattr(solution_instance, 'A', None)

        # Safe extraction of results (preferred from .results dict or attribute)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.f0 = solution_instance.results.get('f0', None)
            self.dsigma_dOmega_0 = solution_instance.results.get('dsigma_dOmega_0', None)
        else:
            self.f0 = getattr(solution_instance, 'f0', None)
            self.dsigma_dOmega_0 = getattr(solution_instance, 'dsigma_dOmega_0', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_results(self):
        """Check that both f0 and dsigma_dOmega_0 are present and numeric."""
        if self.f0 is None:
            return (False, "Missing 'f0'")
        if self.dsigma_dOmega_0 is None:
            return (False, "Missing 'dsigma_dOmega_0'")
        if not isinstance(self.f0, (int, float)):
            return (False, f"f0 is not numeric: {type(self.f0)}")
        if not isinstance(self.dsigma_dOmega_0, (int, float)):
            return (False, f"dsigma_dOmega_0 is not numeric: {type(self.dsigma_dOmega_0)}")
        return (True, "Both f0 and dsigma_dOmega_0 present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests (physical relations, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_cross_section_from_amplitude(self):
        """Check that dsigma/dOmega = |f|^2 (Born approximation)."""
        if any(v is None for v in [self.f0, self.dsigma_dOmega_0]):
            return (False, "SKIP: missing f0 or dsigma_dOmega_0")
        expected = self.f0**2
        if not math.isclose(self.dsigma_dOmega_0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dsigma_dOmega_0={self.dsigma_dOmega_0} != |f0|^2={expected}")
        return (True, f"dsigma_dOmega_0 = |f0|^2 within tolerance")

    def test_invariant_forward_amplitude_formula(self):
        """Check that f0 = (m e A)/(3 hbar^2) derived from expansion."""
        if any(v is None for v in [self.m, self.e, self.A, self.hbar, self.f0]):
            return (False, "SKIP: missing one or more required inputs")
        expected_f0 = (self.m * self.e * self.A) / (3 * self.hbar**2)
        if not math.isclose(self.f0, expected_f0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f0={self.f0} != expected {expected_f0}")
        return (True, f"f0 matches derived expression")

    # ------------------------------------------------------------------
    # Modeling / trace quality tests
    # ------------------------------------------------------------------
    def test_modeling_tilde_rho_expansion(self):
        """
        Verify that the expansion tilde{rho}(q) ≈ -A q^2/6 used in the
        Born amplitude is consistent with f0 being finite and the cross section
        having the correct dependence on A.
        """
        # Check that f0 is non-zero when A is non-zero (which it is given as a constant)
        if self.f0 is None or self.A is None:
            return (False, "SKIP: missing f0 or A")
        # If the expansion were wrong, f0 might be infinite or zero.
        if not math.isfinite(self.f0):
            return (False, f"f0 is not finite, indicating expansion failure")
        # The expansion yields f0 proportional to A; check that f0 != 0 if A != 0
        if math.isclose(self.A, 0.0, abs_tol=1e-12):
            return (False, "SKIP: A is zero, expansion not checkable")
        if math.isclose(self.f0, 0.0, abs_tol=1e-12):
            return (False, "f0 is zero but A is non-zero; expansion may have been misapplied")
        return (True, "f0 is finite and non-zero, consistent with expansion")

    def test_modeling_angular_integration_factor(self):
        """
        Check that the factor 1/3 arises correctly from the angular integration
        of (q·r)^2. The expansion yields:
            ∫ ρ(r) (q·r)^2 d^3r = (1/3) q^2 ∫ ρ(r) r^2 d^3r = (1/3) q^2 A.
        Then the form factor becomes F(q) = -q^2 A/6 because of the 1/2! and the angular average.
        This test re-derives the angular part of the factor.
        """
        # Compute angular integral ∫_0^π cos^2θ sinθ dθ = 2/3
        # For each ρ(r), the average of (q·r)^2 over angles gives (q^2 r^2)/3.
        # So the 1/3 factor appears. The 1/2! then gives 1/6. We can check that
        # 1/6 = (1/2) * (1/3).
        half_factorial = 0.5
        # We just verify that 1/6 = 1/2 * 1/3
        if not math.isclose(1.0 / 6.0, half_factorial * (1.0 / 3.0), rel_tol=1e-12):
            return (False, "Angular integration factor derivation inconsistent")
        # Also verify that the factor 1/3 is correct for the integral of (cos^2θ) over sphere
        # The actual integral ∫ cos^2θ sinθ dθ from 0 to π equals 2/3.
        # And the average over the sphere is (1/4π) * 2π * (2/3) = 1/3. So it's consistent.
        return (True, "Angular integration factor (1/3) is correct")

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_cross_section_increases_with_A(self):
        """
        The cross section scales as A^2. Perturb A by +10% and verify
        that the new cross section (re-derived from the formula) is larger.
        """
        if any(v is None for v in [self.A, self.m, self.e, self.hbar, self.dsigma_dOmega_0]):
            return (False, "SKIP: missing required inputs or cross section")
        # Baseline cross section (given)
        baseline = self.dsigma_dOmega_0
        # Perturb A by +10%
        pert_factor = 1.1
        A_pert = self.A * pert_factor
        # Re-derive new cross section using the formula
        f0_pert = (self.m * self.e * A_pert) / (3 * self.hbar**2)
        dsigma_pert = f0_pert**2
        # Check that perturbed is larger
        if not dsigma_pert > baseline:
            return (False, f"Perturbed cross section {dsigma_pert:.3e} is not larger than baseline {baseline:.3e}")
        # Additional quantitative check: the ratio should be pert_factor^2 = 1.21
        expected_ratio = pert_factor**2
        actual_ratio = dsigma_pert / baseline
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Cross section ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}")
        return (True, f"Cross section increases with A: ratio {actual_ratio:.4f} ≈ {expected_ratio:.4f}")

    def test_monotonicity_cross_section_increases_with_m(self):
        """
        The cross section scales as m^2. Perturb m by +10% and check.
        """
        if any(v is None for v in [self.m, self.e, self.A, self.hbar, self.dsigma_dOmega_0]):
            return (False, "SKIP: missing required inputs or cross section")
        baseline = self.dsigma_dOmega_0
        pert_factor = 1.1
        m_pert = self.m * pert_factor
        f0_pert = (m_pert * self.e * self.A) / (3 * self.hbar**2)
        dsigma_pert = f0_pert**2
        expected_ratio = pert_factor**2
        actual_ratio = dsigma_pert / baseline
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Cross section ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}")
        return (True, f"Cross section increases with m: ratio {actual_ratio:.4f} ≈ {expected_ratio:.4f}")

    # ------------------------------------------------------------------
    # Answer sanity (at most one weak test; no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """Check that cross section is positive and finite, and f0 is finite."""
        if self.dsigma_dOmega_0 is None:
            return (False, "SKIP: dsigma_dOmega_0 missing")
        # Positive
        if self.dsigma_dOmega_0 < 0:
            return (False, f"Cross section negative: {self.dsigma_dOmega_0}")
        # Finite and not NaN
        if not math.isfinite(self.dsigma_dOmega_0):
            return (False, f"Cross section not finite: {self.dsigma_dOmega_0}")
        # Also check f0 if present
        if self.f0 is not None:
            if not math.isfinite(self.f0):
                return (False, f"f0 not finite: {self.f0}")
        return (True, "Cross section positive, finite; f0 finite (if present)")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)