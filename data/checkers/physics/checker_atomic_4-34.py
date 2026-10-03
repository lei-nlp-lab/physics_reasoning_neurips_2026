import math
import sys

class PhysicsChecker:
    """
    Physics checker for antiproton-proton annihilation problem.
    Validates results for pion and photon production.
    """

    def __init__(self, solution_instance):
        # Extract givens safely - assume they might be in instance attributes or dicts
        self.m_p = None
        self.m_pi = None
        self.hc = None
        if hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            if isinstance(givens, dict):
                self.m_p = givens.get('m_p', None)
                self.m_pi = givens.get('m_pi', None)
                self.hc = givens.get('hc', None)
        # Also fallback to direct attributes if present
        if self.m_p is None:
            self.m_p = getattr(solution_instance, 'm_p', None)
        if self.m_pi is None:
            self.m_pi = getattr(solution_instance, 'm_pi', None)
        if self.hc is None:
            self.hc = getattr(solution_instance, 'hc', None)

        # Extract results safely
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.K_avg = self.results.get('K_avg', None)
        self.p = self.results.get('p', None)
        self.beta = self.results.get('beta', None)
        self.lambda_val = self.results.get('lambda', None)

        # Extract explanation_steps for context (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Extract pseudocode if needed (used conservatively)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

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

    # -------------------- Interface Tests --------------------
    def test_interface_required_results(self):
        """Check that all four required result keys exist and are not None."""
        required = {
            'K_avg': self.K_avg,
            'p': self.p,
            'beta': self.beta,
            'lambda': self.lambda_val
        }
        missing = [key for key, val in required.items() if val is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required results present")

    def test_interface_required_givens(self):
        """Check that required givens are provided."""
        needed = [('m_p', self.m_p), ('m_pi', self.m_pi), ('hc', self.hc)]
        missing = [name for name, val in needed if val is None]
        if missing:
            return (False, f"Missing givens: {missing}")
        # Also check positivity
        for name, val in needed:
            if val is not None and val <= 0:
                return (False, f"Given {name} is not positive: {val}")
        return (True, "All givens present and positive")

    # -------------------- Invariant Tests (use math.isclose) --------------------
    def test_invariant_energy_conservation(self):
        """
        The total initial rest energy (2*m_p) must equal the total final energy
        of four pions: 4*(K_avg + m_pi).
        """
        if self.m_p is None or self.m_pi is None or self.K_avg is None:
            return (False, "SKIP: missing data for energy conservation check")
        # Compute expected total final energy from result
        total_final = 4 * (self.K_avg + self.m_pi)
        total_initial = 2 * self.m_p
        if not math.isclose(total_initial, total_final, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Energy not conserved: initial {total_initial}, final {total_final}")
        return (True, f"Energy conserved within tolerance")

    def test_invariant_relativistic_relation(self):
        """
        Check relativistic energy-momentum: p^2 = E^2 - m_pi^2,
        where E = K_avg + m_pi.
        """
        if self.K_avg is None or self.m_pi is None or self.p is None:
            return (False, "SKIP: missing data for relativistic check")
        E = self.K_avg + self.m_pi
        expected_p_sq = E**2 - self.m_pi**2
        p_sq = self.p**2
        if not math.isclose(expected_p_sq, p_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Relativistic relation fails: expected p^2={expected_p_sq:.1f}, got {p_sq:.1f}")
        return (True, f"Relativistic energy-momentum holds")

    def test_invariant_velocity(self):
        """Check beta = p / E with E = K_avg + m_pi."""
        if self.K_avg is None or self.m_pi is None or self.p is None:
            return (False, "SKIP: missing data for velocity check")
        if self.beta is None:
            return (False, "SKIP: beta not provided")
        E = self.K_avg + self.m_pi
        expected_beta = self.p / E
        if not math.isclose(expected_beta, self.beta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Beta mismatch: expected {expected_beta:.3f}, got {self.beta:.3f}")
        return (True, f"Beta value correct")

    # -------------------- Modeling/Trace Quality Tests --------------------
    def test_modeling_pion_kinetic_energy_formula(self):
        """
        Verify that K_avg was derived from the correct energy conservation equation:
        K_avg = (2*m_p - 4*m_pi) / 4
        This is re-derived from the given masses inside the checker.
        """
        if self.m_p is None or self.m_pi is None or self.K_avg is None:
            return (False, "SKIP: missing data for modeling test")
        expected_K = (2 * self.m_p - 4 * self.m_pi) / 4.0
        if not math.isclose(expected_K, self.K_avg, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"K_avg derived formula mismatch: expected {expected_K:.1f}, got {self.K_avg:.1f}")
        return (True, f"K_avg matches energy conservation formula")

    def test_modeling_photon_wavelength(self):
        """
        For photon case: lambda = hc / m_p.
        This tests the modeling of part (d).
        """
        if self.m_p is None or self.hc is None or self.lambda_val is None:
            return (False, "SKIP: missing data for photon wavelength check")
        expected_lambda = self.hc / self.m_p
        if not math.isclose(expected_lambda, self.lambda_val, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Photon wavelength mismatch: expected {expected_lambda:.3e}, got {self.lambda_val:.3e}")
        return (True, f"Photon wavelength matches derived value")

    def test_modeling_momentum_from_energy(self):
        """
        Verify that the momentum p is consistent with relativistic formula
        p = sqrt( (K_avg + m_pi)^2 - m_pi^2 ) re-derived from givens.
        This confirms the modeling step for part (b).
        """
        if self.m_pi is None or self.K_avg is None or self.p is None:
            return (False, "SKIP: missing data for momentum modeling test")
        E = self.K_avg + self.m_pi
        expected_p = math.sqrt(E**2 - self.m_pi**2)
        if not math.isclose(expected_p, self.p, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Momentum mismatch: expected {expected_p:.1f}, got {self.p:.1f}")
        return (True, f"Momentum consistent with relativistic formula")

    # -------------------- Monotonicity Test --------------------
    def test_monotonicity_K_avg_with_m_p(self):
        """
        Perturb m_p by +5% and check that K_avg increases (since more initial energy).
        Use stored givens to recompute perturbed K_avg.
        """
        if self.m_p is None or self.m_pi is None or self.K_avg is None:
            return (False, "SKIP: missing data for monotonicity test")
        # Original K_avg
        K_orig = self.K_avg
        # Perturb m_p upward by 5%
        m_p_pert = self.m_p * 1.05
        # Recompute K_avg with same logic
        K_pert = (2 * m_p_pert - 4 * self.m_pi) / 4.0
        # Check monotonicity: increase in m_p leads to increase in K
        if K_pert <= K_orig:
            return (False, f"Monotonicity violated: m_p increased 5% to {m_p_pert:.0f} but K_avg did not increase (original {K_orig:.1f}, perturbed {K_pert:.1f})")
        return (True, f"K_avg increases with m_p: original {K_orig:.1f}, perturbed {K_pert:.1f}")

    # -------------------- Answer Sanity (weak, no isclose, no re-derivation) --------------------
    def test_answer_sanity_weak(self):
        """
        Weak sanity checks on final answers: signs, ranges, no re-derivation.
        """
        if any(v is None for v in [self.K_avg, self.p, self.beta, self.lambda_val]):
            return (False, "SKIP: missing results for sanity check")
        issues = []
        # Kinetic energy should be positive and less than total initial (1876 MeV)
        if not (0 < self.K_avg < 2000):
            issues.append(f"K_avg out of plausible range: {self.K_avg}")
        # Momentum positive and not too large (max roughly sqrt((2m_p)^2 - (4m_pi)^2)/4 ~ 500)
        if not (0 < self.p < 1000):
            issues.append(f"p out of plausible range: {self.p}")
        # Beta between 0 and 1
        if not (0 < self.beta < 1):
            issues.append(f"beta out of [0,1]: {self.beta}")
        # Lambda positive and very small (around 1e-13 cm)
        if not (self.lambda_val > 0 and self.lambda_val < 1e-10):
            issues.append(f"lambda out of plausible range: {self.lambda_val}")
        if issues:
            return (False, "Sanity issues: " + "; ".join(issues))
        return (True, "All sanity checks pass")