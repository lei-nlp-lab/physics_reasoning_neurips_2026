import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of givens
        self.r0 = getattr(solution_instance, 'r0', None)
        self.dr = getattr(solution_instance, 'dr', None)
        self.rho0 = getattr(solution_instance, 'rho0', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Safe extraction of results
        result_dict = getattr(solution_instance, 'results', {})
        self.E1 = result_dict.get('E1', None)
        self.E2 = result_dict.get('E2', None)
        self.E3 = result_dict.get('E3', None)
        self.sigma_inner = result_dict.get('sigma_inner', None)
        self.sigma_outer_isolated = result_dict.get('sigma_outer_isolated', None)
        self.E_grounded = result_dict.get('E_grounded', None)
        self.sigma_outer_grounded = result_dict.get('sigma_outer_grounded', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present (not None)."""
        keys = ['E1', 'E2', 'E3', 'sigma_inner', 'sigma_outer_isolated',
                'E_grounded', 'sigma_outer_grounded']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_givens(self):
        """Check that essential physical givens are available."""
        missing = []
        for g in ['r0', 'dr', 'rho0', 'epsilon0']:
            if getattr(self, g) is None:
                missing.append(g)
        if missing:
            return (False, f"Missing givens: {missing}. Cannot fully verify physics.")
        return (True, "Essential givens present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_E_inside_conductor(self):
        """E = 0 inside the conducting shell (r0 < r < r0+dr)."""
        if self.E2 is None:
            return (False, "SKIP: E2 not available")
        if not math.isclose(self.E2, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E2 = {self.E2} is not zero inside conductor")
        return (True, "E2 zero inside conductor")

    def test_invariant_cavity_charge_cancel(self):
        """
        On the inner surface, the induced charge must cancel the enclosed
        cavity charge. For a Gaussian cylinder at r just inside the conductor:
        sigma_inner * 2π r0 = -π r0^2 ρ0  →  sigma_inner = -ρ0 r0 / 2.
        """
        if None in (self.sigma_inner, self.r0, self.rho0):
            return (False, "SKIP: missing sigma_inner, r0, or rho0")
        expected = -self.rho0 * self.r0 / 2.0
        if not math.isclose(self.sigma_inner, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_inner = {self.sigma_inner}, expected ~ {expected}")
        return (True, "sigma_inner consistent with cavity charge cancelation")

    def test_invariant_isolated_net_zero(self):
        """Total charge on the isolated hollow cylinder is zero."""
        if None in (self.sigma_inner, self.sigma_outer_isolated, self.r0, self.dr):
            return (False, "SKIP: missing required values")
        net = self.sigma_inner * self.r0 + self.sigma_outer_isolated * (self.r0 + self.dr)
        if not math.isclose(net, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Net charge per unit length not zero: {net}")
        return (True, "Net charge zero for isolated cylinder")

    def test_invariant_grounded_E(self):
        """For a grounded cylinder, the field outside must be zero."""
        if self.E_grounded is None:
            return (False, "SKIP: E_grounded not available")
        if not math.isclose(self.E_grounded, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_grounded = {self.E_grounded} not zero")
        return (True, "E_grounded zero as expected")

    def test_invariant_grounded_outer_sigma(self):
        """For a grounded cylinder, the outer surface charge is zero."""
        if self.sigma_outer_grounded is None:
            return (False, "SKIP: sigma_outer_grounded not available")
        if not math.isclose(self.sigma_outer_grounded, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_outer_grounded = {self.sigma_outer_grounded} not zero")
        return (True, "sigma_outer_grounded zero")

    # ------------------------------------------------------------------
    # Modeling tests (based on explanation steps / derived formulas)
    # ------------------------------------------------------------------
    def test_modeling_sigma_inner_formula(self):
        """Check sigma_inner against the analytic formula σ = -ρ0 r0 / 2."""
        if None in (self.rho0, self.r0, self.sigma_inner):
            return (False, "SKIP: missing rho0, r0, sigma_inner")
        expected = -self.rho0 * self.r0 / 2.0
        if not math.isclose(self.sigma_inner, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_inner = {self.sigma_inner}, expected ~ {expected}")
        return (True, "sigma_inner matches formula")

    def test_modeling_sigma_outer_isolated_formula(self):
        """Check sigma_outer_isolated against σ = ρ0 r0^2 / [2(r0+dr)]."""
        if None in (self.rho0, self.r0, self.dr, self.sigma_outer_isolated):
            return (False, "SKIP: missing required values")
        expected = self.rho0 * self.r0**2 / (2.0 * (self.r0 + self.dr))
        if not math.isclose(self.sigma_outer_isolated, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_outer_isolated = {self.sigma_outer_isolated}, expected ~ {expected}")
        return (True, "sigma_outer_isolated matches formula")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_outer_vs_dr(self):
        """Increasing shell thickness dr should reduce sigma_outer_isolated
        (denominator grows, numerator constant). We compute baseline and
        perturbed by a small positive δdr."""
        if None in (self.rho0, self.r0, self.dr, self.sigma_outer_isolated):
            return (False, "SKIP: missing required values")
        if self.dr == 0.0:
            return (False, "SKIP: dr cannot be zero for perturbation")
        # Baseline sigma (recomputed for consistency)
        baseline_sigma = self.rho0 * self.r0**2 / (2.0 * (self.r0 + self.dr))
        # Perturb dr by +10%
        delta_dr = 0.1 * self.dr
        perturbed_sigma = self.rho0 * self.r0**2 / (2.0 * (self.r0 + (self.dr + delta_dr)))
        # Monotonicity: increasing dr → decreasing sigma
        if not (perturbed_sigma < baseline_sigma):
            return (False, f"Perturbed sigma {perturbed_sigma} not less than baseline {baseline_sigma}")
        return (True, "sigma_outer_isolated decreases with increased dr")

    def test_monotonicity_sigma_inner_vs_rho0(self):
        """Increasing ρ0 increases |sigma_inner| (more negative)."""
        if None in (self.rho0, self.r0, self.sigma_inner):
            return (False, "SKIP: missing required values")
        baseline_sigma = -self.rho0 * self.r0 / 2.0
        # Perturb ρ0 by +5%
        delta_rho = 0.05 * self.rho0
        perturbed_sigma = -(self.rho0 + delta_rho) * self.r0 / 2.0
        # Since sigma_inner is negative, more negative means larger magnitude.
        if not (perturbed_sigma < baseline_sigma):   # more negative
            return (False, f"Perturbed sigma {perturbed_sigma} not more negative than baseline {baseline_sigma}")
        return (True, "sigma_inner becomes more negative with increased ρ0")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs(self):
        """Only check signs and finiteness of fields and charges."""
        if None in (self.E1, self.E3, self.sigma_inner, self.sigma_outer_isolated):
            return (False, "SKIP: missing some values for sign check")
        checks = []
        # E1 should be positive for positive ρ0 and r>0
        if self.E1 < 0:
            checks.append("E1 negative (unexpected sign)")
        # E3 should be positive for positive ρ0
        if self.E3 < 0:
            checks.append("E3 negative (unexpected sign)")
        # sigma_inner should be negative for positive ρ0
        if self.sigma_inner > 0:
            checks.append("sigma_inner positive (expected negative)")
        # sigma_outer_isolated should be positive for positive ρ0
        if self.sigma_outer_isolated < 0:
            checks.append("sigma_outer_isolated negative (expected positive)")
        # All should be finite
        for name, val in [("E1", self.E1), ("E3", self.E3),
                         ("sigma_inner", self.sigma_inner),
                         ("sigma_outer_isolated", self.sigma_outer_isolated)]:
            if math.isinf(val) or math.isnan(val):
                checks.append(f"{name} is infinite or NaN")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All sign and finiteness checks passed")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover test methods
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