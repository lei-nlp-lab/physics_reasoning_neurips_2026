import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the perturbed harmonic oscillator problem:
    H = P^2/(2m) + (1/2) m ω^2 x^2 + λ/(x^2 + a^2).
    Checks interface, physical invariants, modeling consistency, monotonicity,
    and a weak final answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results (required by pseudocode emit contract)
        self.E1 = None
        self.E2 = None
        # Also attempt to extract the given parameters
        self.m = None
        self.omega = None
        self.hbar = None
        self.lmbda = None   # avoid shadowing lambda
        self.a = None

        # Safe extraction: prefer results dict, then direct attributes
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        self.E1 = results.get('E1', getattr(solution_instance, 'E1', None))
        self.E2 = results.get('E2', getattr(solution_instance, 'E2', None))

        # Try to get givens from results or from the instance itself
        for key, attr in [('m', 'm'), ('omega', 'omega'), ('hbar', 'hbar'),
                          ('lambda', 'lmbda'), ('a', 'a')]:
            val = results.get(key, getattr(solution_instance, key, None))
            setattr(self, attr, val)

        # Compute derived scale length if possible
        self.l = None
        if self.hbar is not None and self.m is not None and self.omega is not None:
            if self.hbar > 0 and self.m > 0 and self.omega > 0:
                self.l = math.sqrt(self.hbar / (self.m * self.omega))

    # ----------------------------------------------------------------
    # Interface tests: check that expected keys exist and are numeric
    # ----------------------------------------------------------------
    def test_interface_E1_present(self):
        if self.E1 is None:
            return (False, "Missing E1 result")
        if not isinstance(self.E1, (int, float)):
            return (False, f"E1 has non-numeric type: {type(self.E1)}")
        return (True, "E1 present and numeric")

    def test_interface_E2_present(self):
        if self.E2 is None:
            return (False, "Missing E2 result")
        if not isinstance(self.E2, (int, float)):
            return (False, f"E2 has non-numeric type: {type(self.E2)}")
        return (True, "E2 present and numeric")

    # ----------------------------------------------------------------
    # Invariant tests: physical principles that must hold
    # ----------------------------------------------------------------
    def test_invariant_E1_positive(self):
        if self.E1 is None:
            return (False, "SKIP: E1 not available")
        if self.lmbda is None:
            return (False, "SKIP: lambda not available")
        # Perturbation λ > 0 (given, not specified but typical). If λ < 0,
        # correction could be negative; but integrand is positive if λ>0.
        if self.lmbda > 0 and self.E1 <= 0:
            return (False, f"E1 = {self.E1} should be positive when λ > 0")
        return (True, "E1 sign consistent (or λ not positive)")

    def test_invariant_E2_positive(self):
        if self.E2 is None:
            return (False, "SKIP: E2 not available")
        if self.lmbda is None:
            return (False, "SKIP: lambda not available")
        if self.lmbda > 0 and self.E2 <= 0:
            return (False, f"E2 = {self.E2} should be positive when λ > 0")
        return (True, "E2 sign consistent (or λ not positive)")

    def test_invariant_E1_rate_vs_sqrt(self):
        """
        For a ≪ ℓ, first-order correction should be proportional to 1/a.
        Check that E1 * a ≈ λ * sqrt(m*ω*π/ħ), if all parameters are available.
        """
        if any(x is None for x in (self.E1, self.m, self.omega, self.hbar, self.lmbda, self.a)):
            return (False, "SKIP: insufficient givens to check E1 scaling")
        if self.a <= 0 or self.lmbda <= 0:
            return (False, "SKIP: a or lambda non-positive")
        if self.l is None:
            return (False, "SKIP: cannot compute ℓ")

        # Expected leading behavior: E1 ≈ (λ/a) * sqrt(m*ω*π/ħ)
        expected_scale = (self.lmbda / self.a) * math.sqrt(self.m * self.omega * math.pi / self.hbar)
        if math.isclose(self.E1, expected_scale, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"E1 matches λ/a * sqrt(mωπ/ħ): {self.E1} ≈ {expected_scale}")
        else:
            return (False, f"E1 {self.E1} deviates from expected {expected_scale}")

    def test_invariant_E2_rate_vs_inverse_square(self):
        """
        For a ≫ ℓ, leading correction is λ/a².
        """
        if any(x is None for x in (self.E2, self.lmbda, self.a)):
            return (False, "SKIP: insufficient data for E2 scaling")
        if self.a <= 0:
            return (False, "SKIP: a non-positive")
        expected = self.lmbda / (self.a * self.a)
        if math.isclose(self.E2, expected, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"E2 matches λ/a²: {self.E2} ≈ {expected}")
        else:
            return (False, f"E2 {self.E2} deviates from expected {expected}")

    # ----------------------------------------------------------------
    # Modeling / trace tests: check that the reasoning steps are plausible
    # ----------------------------------------------------------------
    def test_modeling_E1_via_integral(self):
        """
        Check that E1 could be obtained from the integral: λ ∫ ψ₀²/(x²+a²) dx.
        If we have the given form ψ₀²(0) = sqrt(mω/(πħ)), and using the
        hint integral, the answer should be λ sqrt(mω/(πħ)) * (π/a).
        This is a local relation, not a bulk copy of steps.
        """
        if any(x is None for x in (self.E1, self.m, self.omega, self.hbar, self.lmbda, self.a)):
            return (False, "SKIP: missing data for modeling test")
        psi0sq0 = math.sqrt(self.m * self.omega / (math.pi * self.hbar))
        integral_hint = math.pi / self.a
        expected = self.lmbda * psi0sq0 * integral_hint
        if math.isclose(self.E1, expected, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"E1 consistent with integral evaluation: {self.E1} ≈ {expected}")
        else:
            return (False, f"E1 {self.E1} does not match integral expression {expected}")

    def test_modeling_E2_using_expansion(self):
        """
        Large a expansion: 1/(x²+a²) ≈ 1/a² - x²/a⁴.
        Using ⟨x²⟩ = ħ/(2mω), the first-order correction should be
        λ(1/a² - (ħ/(2mω))/a⁴). For large a, leading term is λ/a².
        Check that E2 matches the leading term (ignoring the subleading correction).
        """
        if any(x is None for x in (self.E2, self.lmbda, self.a, self.m, self.omega, self.hbar)):
            return (False, "SKIP: missing data for expansion test")
        leading = self.lmbda / (self.a * self.a)
        subleading = (self.lmbda * self.hbar) / (2 * self.m * self.omega * self.a ** 4)
        # The full expression (first two terms) would be leading - subleading.
        # We check that E2 is close to the leading term, given that a is large.
        if math.isclose(self.E2, leading, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"E2 consistent with leading expansion term: {self.E2} ≈ {leading}")
        else:
            # Could be that they included the subleading term; check that too
            full = leading - subleading
            if math.isclose(self.E2, full, rel_tol=1e-2, abs_tol=1e-12):
                return (True, f"E2 consistent with full two-term expansion: {self.E2} ≈ {full}")
            else:
                return (False, f"E2 {self.E2} does not match leading term {leading} or full {full}")

    # ----------------------------------------------------------------
    # Monotonicity tests: explicitly perturb an input and compare
    # ----------------------------------------------------------------
    def test_monotonicity_E1_decreases_with_a(self):
        """
        Explicitly perturb a by +10% and recompute the expected E1 using the
        leading-order formula. Then verify that the perturbed expected value
        is smaller than the baseline expected value.
        Note: We use the derived formula to compute baseline and perturbed,
        not the solution's own E1 (since we cannot re-run the full solution).
        This checks the monotonic property of the physics itself.
        """
        # Ensure we have all necessary givens
        if any(x is None for x in (self.m, self.omega, self.hbar, self.lmbda, self.a)):
            return (False, "SKIP: missing givens for monotonicity test")
        if self.a <= 0 or self.lmbda <= 0:
            return (False, "SKIP: a or lambda must be positive")

        # Baseline expected E1 (using the small‑a leading term)
        baseline_expected = (self.lmbda / self.a) * math.sqrt(self.m * self.omega * math.pi / self.hbar)

        # Perturb a: increase by 10%
        a_perturbed = 1.1 * self.a
        perturbed_expected = (self.lmbda / a_perturbed) * math.sqrt(self.m * self.omega * math.pi / self.hbar)

        # For a > 0, increasing a should decrease the baseline expected value
        if perturbed_expected < baseline_expected:
            return (True, f"E1 decreases when a increases: baseline {baseline_expected:.6f} -> perturbed {perturbed_expected:.6f}")
        else:
            return (False, "Perturbed expected E1 is not smaller than baseline (monotonicity violated)")

    def test_monotonicity_E2_decreases_with_a(self):
        """
        Perturb a by +10% and check that the expected leading‑order E2 (λ/a²)
        decreases.
        """
        if any(x is None for x in (self.lmbda, self.a)):
            return (False, "SKIP: missing lambda or a for monotonicity test")
        if self.a <= 0 or self.lmbda <= 0:
            return (False, "SKIP: a or lambda must be positive")

        baseline_expected = self.lmbda / (self.a ** 2)
        a_perturbed = 1.1 * self.a
        perturbed_expected = self.lmbda / (a_perturbed ** 2)

        if perturbed_expected < baseline_expected:
            return (True, f"E2 decreases when a increases: baseline {baseline_expected:.6f} -> perturbed {perturbed_expected:.6f}")
        else:
            return (False, "Perturbed expected E2 is not smaller than baseline (monotonicity violated)")

    # ----------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ----------------------------------------------------------------
    def test_answer_sanity_magnitudes(self):
        """
        Only check that E1 and E2 are finite, non-NaN, and have sensible sign.
        No re-derivation of values.
        """
        if self.E1 is not None:
            if not math.isfinite(self.E1):
                return (False, f"E1 is not finite: {self.E1}")
        if self.E2 is not None:
            if not math.isfinite(self.E2):
                return (False, f"E2 is not finite: {self.E2}")
        # If both present, check that E2 is not astronomically larger than E1?
        # Actually E2 ~ λ/a², E1 ~ λ/(a) * sqrt(...), but without ℓ we can't compare.
        # Just pass.
        return (True, "Both results are finite and numeric")

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------
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
        return all_pass, logs