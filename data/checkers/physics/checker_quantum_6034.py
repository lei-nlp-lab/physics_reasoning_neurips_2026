import math
import traceback

class PhysicsChecker:
    """
    Audits neutron-neutron zero-energy s-wave scattering cross section calculation.
    """
    def __init__(self, solution_instance):
        # Extract givens (safe defaults: None)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.a = givens.get('a', None)
        self.V0 = givens.get('V0', None)
        self.mu = givens.get('mu', None)
        self.hbar = givens.get('hbar', None)

        # Extract results (safe defaults: None)
        results = getattr(solution_instance, 'results', {}) or {}
        self.k = results.get('k', None)
        self.a_s = results.get('a_s', None)
        self.sigma = results.get('sigma', None)

        # Additional check: if solution_instance has attributes directly (fallback)
        if self.k is None:
            self.k = getattr(solution_instance, 'k', None)
        if self.a_s is None:
            self.a_s = getattr(solution_instance, 'a_s', None)
        if self.sigma is None:
            self.sigma = getattr(solution_instance, 'sigma', None)

    # ---------- Interface checks ----------
    def test_interface_has_k(self):
        """Check that result key 'k' exists and is not None."""
        if self.k is None:
            return False, "k is missing from results"
        return True, "k present"

    def test_interface_has_a_s(self):
        """Check that result key 'a_s' exists and is not None."""
        if self.a_s is None:
            return False, "a_s is missing from results"
        return True, "a_s present"

    def test_interface_has_sigma(self):
        """Check that result key 'sigma' exists and is not None."""
        if self.sigma is None:
            return False, "sigma is missing from results"
        return True, "sigma present"

    def test_interface_all_givens_present(self):
        """Check that all required given parameters are provided."""
        missing = []
        if self.a is None:
            missing.append("a")
        if self.V0 is None:
            missing.append("V0")
        if self.mu is None:
            missing.append("mu")
        if self.hbar is None:
            missing.append("hbar")
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All givens present"

    # ---------- Invariant / theorem checks (use math.isclose) ----------
    def test_invariant_k_definition(self):
        """Verify that k = sqrt(6*mu*V0)/hbar."""
        if None in (self.mu, self.V0, self.hbar, self.k):
            return False, "SKIP: insufficient data to check k definition"
        expected_k = math.sqrt(6 * self.mu * self.V0) / self.hbar
        if math.isclose(self.k, expected_k, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"k≈{expected_k:.4f}"
        return False, f"k computed {self.k:.4f} but expected {expected_k:.4f}"

    def test_invariant_a_s_definition(self):
        """Check scattering length formula: a_s = a - tan(k*a)/k."""
        if None in (self.a, self.k, self.a_s):
            return False, "SKIP: insufficient data to check a_s definition"
        expected_a_s = self.a - math.tan(self.k * self.a) / self.k
        if math.isclose(self.a_s, expected_a_s, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"a_s≈{expected_a_s:.4f}"
        return False, f"a_s computed {self.a_s:.4f} but expected {expected_a_s:.4f}"

    def test_invariant_sigma_definition(self):
        """Check total cross section formula: sigma = 4π * a_s^2."""
        if None in (self.a_s, self.sigma):
            return False, "SKIP: insufficient data to check sigma definition"
        expected_sigma = 4 * math.pi * self.a_s**2
        if math.isclose(self.sigma, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"sigma≈{expected_sigma:.4f}"
        return False, f"sigma computed {self.sigma:.4f} but expected {expected_sigma:.4f}"

    def test_invariant_k_positive(self):
        """Wave number k must be positive (real potential)."""
        if self.k is None:
            return False, "SKIP: k not provided"
        if self.k > 0:
            return True, "k > 0"
        return False, f"k = {self.k} is not positive"

    def test_invariant_sigma_nonnegative(self):
        """Cross section must be nonnegative."""
        if self.sigma is None:
            return False, "SKIP: sigma not provided"
        if self.sigma >= 0:
            return True, "sigma nonnegative"
        return False, f"sigma = {self.sigma} is negative"

    # ---------- Modeling / trace checks ----------
    def test_modeling_only_singlet_contributes(self):
        """
        According to the Pauli principle, for s-wave (spatially symmetric),
        the spin must be antisymmetric -> singlet (S=0).
        The effective potential depth for singlet is 3V0.
        Check that the k used corresponds to sqrt(6*mu*V0)/hbar,
        i.e., depth = 3V0 (so k^2 = 2*mu*(3V0)/hbar^2 = 6muV0/hbar^2).
        This is already tested in invariant_k_definition, but we also check
        that the result is not based on triplet (depth V0).
        """
        if None in (self.mu, self.V0, self.hbar, self.k):
            return False, "SKIP: insufficient data for singlet check"
        # Correct singlet k:
        k_singlet = math.sqrt(6 * self.mu * self.V0) / self.hbar
        # Triplet would use sqrt(2*mu*V0)/hbar
        k_triplet = math.sqrt(2 * self.mu * self.V0) / self.hbar
        # Check if k is close to singlet
        if math.isclose(self.k, k_singlet, rel_tol=1e-2, abs_tol=1e-2):
            return True, "k corresponds to singlet (depth 3V0)"
        elif math.isclose(self.k, k_triplet, rel_tol=1e-2, abs_tol=1e-2):
            return False, "k corresponds to triplet (depth V0) – violates Pauli principle"
        else:
            return False, f"k = {self.k} does not match singlet or triplet expectation"

    def test_modeling_s_wave_only(self):
        """
        Zero-energy scattering: only s-wave (l=0) contributes.
        The result should not include higher partial waves.
        This is a consequence of the formula using only scattering length.
        We can check that sigma = 4π a_s^2 (already tested), which is the s-wave total cross section.
        No additional check needed here, but we assert consistency.
        """
        # Indirectly verified by sigma formula. We'll just pass if sigma is defined.
        if self.sigma is None:
            return False, "SKIP: sigma not available"
        return True, "s-wave only assumption consistent with sigma formula"

    # ---------- Monotonicity checks ----------
    def test_monotonicity_sigma_increases_with_V0(self):
        """
        If V0 increases (potential well deeper), k increases,
        and for small-to-moderate k a, a_s becomes more negative,
        so |a_s| increases, hence sigma = 4π a_s^2 increases.
        We test by perturbing V0 by +1% and checking that sigma increases.
        """
        if None in (self.a, self.V0, self.mu, self.hbar, self.k, self.a_s, self.sigma):
            return False, "SKIP: insufficient data for monotonicity test"
        # Baseline
        base_sigma = self.sigma
        # Perturb V0 by +1%
        V0_perturbed = self.V0 * 1.01
        k_pert = math.sqrt(6 * self.mu * V0_perturbed) / self.hbar
        a_s_pert = self.a - math.tan(k_pert * self.a) / k_pert
        sigma_pert = 4 * math.pi * a_s_pert**2
        if sigma_pert > base_sigma:
            return True, f"sigma increases with 1% V0 increase (baseline {base_sigma:.4f} -> perturbed {sigma_pert:.4f})"
        else:
            return False, f"sigma did not increase: baseline {base_sigma:.4f}, perturbed {sigma_pert:.4f}"

    def test_monotonicity_sigma_increases_with_a_larger(self):
        """
        If potential radius a increases while keeping V0 fixed,
        the product k a increases, and for small ka, sigma ~ (ka)^4 a^2? Let's compute.
        Actually sigma increases monotonically with a? Not always, but for shallow wells it does.
        We assume a is positive. We test a +1% increase.
        """
        if None in (self.a, self.V0, self.mu, self.hbar, self.k, self.a_s, self.sigma):
            return False, "SKIP: insufficient data"
        a_pert = self.a * 1.01
        k_pert = math.sqrt(6 * self.mu * self.V0) / self.hbar  # same k
        a_s_pert = a_pert - math.tan(k_pert * a_pert) / k_pert
        sigma_pert = 4 * math.pi * a_s_pert**2
        if sigma_pert > self.sigma:
            return True, f"sigma increases with 1% a increase (baseline {self.sigma:.4f} -> perturbed {sigma_pert:.4f})"
        else:
            # Not strictly monotonic if ka is near resonance, but that's rare for zero-energy.
            # We'll still report but not fail; just note.
            return False, f"sigma did not increase: baseline {self.sigma:.4f}, perturbed {sigma_pert:.4f}"

    # ---------- Weak answer sanity (only one allowed) ----------
    def test_answer_sanity_sigma_finite(self):
        """Check that sigma is finite and not extremely large (within physical bounds)."""
        if self.sigma is None:
            return False, "SKIP: sigma not provided"
        if not math.isfinite(self.sigma):
            return False, f"sigma = {self.sigma} is not finite"
        if self.sigma > 1e10:  # arbitrary huge value, probably unrealistic
            return False, f"sigma = {self.sigma} is suspiciously large"
        if self.sigma < 0:
            return False, "sigma negative"
        return True, "sigma is finite and nonnegative"

    # ---------- Main runner ----------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all test methods
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs