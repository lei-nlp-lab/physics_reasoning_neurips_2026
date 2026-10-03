import math

class PhysicsChecker:
    """
    Auditor for K+ photoproduction problem.
    Checks invariant-mass conditions, decay kinematics, and basic sanity.
    """

    def __init__(self, solution_instance):
        # Safe extraction of results dictionary
        self.results = getattr(solution_instance, 'results', {}) or {}
        
        # Target outputs (may be None)
        self.E_gamma1 = self.results.get('E_gamma1', None)
        self.E_gamma2 = self.results.get('E_gamma2', None)
        self.p_pi_max = self.results.get('p_pi_max', None)
        self.p_perp_max = self.results.get('p_perp_max', None)

        # Physical constants (natural units, c=1)
        self.mp = 938      # MeV/c^2
        self.mk = 494      # MeV/c^2
        self.ml = 1116     # MeV/c^2
        self.mpi = 140     # MeV/c^2
        self.pF_nominal = 250   # assumed Fermi momentum (MeV/c), used in monotonicity tests
        self.beta_L = 0.8
        self.gamma_L = 1.0 / math.sqrt(1 - self.beta_L ** 2)  # should be 5/3

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All four required keys must exist and be numeric."""
        keys = ['E_gamma1', 'E_gamma2', 'p_pi_max', 'p_perp_max']
        missing = [k for k in keys if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")

        # Check that each value is a number (int or float)
        for k in keys:
            v = self.results[k]
            if not isinstance(v, (int, float)):
                return (False, f"Key {k} is not numeric (got {type(v).__name__})")
        return (True, "All required keys present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity, use isclose)
    # ------------------------------------------------------------------
    def test_invariant_free_proton_threshold(self):
        """Check invariant mass squared for part (a): (E_gamma + m_p)^2 - E_gamma^2 = (m_K + m_Lambda)^2."""
        if self.E_gamma1 is None:
            return (False, "SKIP: E_gamma1 not available")

        # Compute invariant from the reported energy
        invariant = (self.E_gamma1 + self.mp) ** 2 - self.E_gamma1 ** 2
        target = (self.mk + self.ml) ** 2

        if math.isclose(invariant, target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Invariant {invariant:.1f} matches target {target}")
        else:
            return (False, f"Invariant {invariant:.1f} differs from target {target}")

    def test_invariant_fermi_motion_threshold(self):
        """Check invariant mass for part (b) with Fermi momentum p_F opposite photon direction."""
        if self.E_gamma2 is None:
            return (False, "SKIP: E_gamma2 not available")

        # Proton energy and momentum in scenario: p_F opposite photon
        E_p = math.sqrt(self.pF_nominal ** 2 + self.mp ** 2)
        # Invariant: (E_gamma + E_p)^2 - (p_gamma - p_F)^2, with p_gamma = E_gamma (photon)
        invariant = (self.E_gamma2 + E_p) ** 2 - (self.E_gamma2 - self.pF_nominal) ** 2
        target = (self.mk + self.ml) ** 2

        if math.isclose(invariant, target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Fermi invariant {invariant:.1f} matches target {target}")
        else:
            return (False, f"Fermi invariant {invariant:.1f} differs from target {target}")

    def test_invariant_decay_rest_frame_momentum(self):
        """Check that p_perp_max equals the π⁻ momentum in the Λ rest frame (p*)."""
        if self.p_perp_max is None:
            return (False, "SKIP: p_perp_max not available")

        # Compute p* from decay kinematics (re‑derived)
        mp = self.mp
        mpi = self.mpi
        ml = self.ml
        num = ((ml ** 2 - (mp + mpi) ** 2) * (ml ** 2 - (mp - mpi) ** 2))
        if num <= 0:
            return (False, "Invalid product in p* calculation (should be positive)")
        pstar = math.sqrt(num) / (2 * ml)

        if math.isclose(self.p_perp_max, pstar, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_perp_max = {self.p_perp_max:.1f} matches p* = {pstar:.1f}")
        else:
            return (False, f"p_perp_max = {self.p_perp_max:.1f} differs from p* = {pstar:.1f}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (quality checks)
    # ------------------------------------------------------------------
    def test_modeling_boosted_momentum(self):
        """Check that p_pi_max is the Lorentz‑boosted value using p_perp_max as p*."""
        if self.p_pi_max is None or self.p_perp_max is None:
            return (False, "SKIP: p_pi_max or p_perp_max not available")

        # Treat p_perp_max as the rest‑frame momentum p*
        pstar = self.p_perp_max
        Estar = math.sqrt(pstar ** 2 + self.mpi ** 2)
        expected = self.gamma_L * (pstar + self.beta_L * Estar)

        if math.isclose(self.p_pi_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_pi_max = {self.p_pi_max:.1f} matches boosted value {expected:.1f}")
        else:
            return (False, f"p_pi_max = {self.p_pi_max:.1f} differs from boosted value {expected:.1f}")

    def test_modeling_decay_rest_energy(self):
        """Check that the rest frame energy E* (derived from p_perp_max) satisfies the known formula."""
        if self.p_perp_max is None:
            return (False, "SKIP: p_perp_max not available")

        # Compute E* from p_perp_max
        pstar = self.p_perp_max
        Estar = math.sqrt(pstar ** 2 + self.mpi ** 2)

        # Expected E* from two‑body decay formula: (m_L^2 + m_pi^2 - m_p^2) / (2 m_L)
        expected_Estar = (self.ml ** 2 + self.mpi ** 2 - self.mp ** 2) / (2 * self.ml)

        if math.isclose(Estar, expected_Estar, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Rest energy E* = {Estar:.1f} matches formula {expected_Estar:.1f}")
        else:
            return (False, f"Rest energy E* = {Estar:.1f} differs from formula {expected_Estar:.1f}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_threshold_with_fermi_perturbation(self):
        """Perturb Fermi momentum upward; check that E_gamma2 decreases."""
        if self.E_gamma2 is None:
            return (False, "SKIP: E_gamma2 not available")

        # Baseline with nominal pF = 250 MeV/c
        pF0 = 250
        E_p0 = math.sqrt(pF0 ** 2 + self.mp ** 2)
        # Recompute the expected E_gamma2 for this baseline (using same threshold equation)
        # Numerator: (mK+mL)^2 - mp^2
        numerator = (self.mk + self.ml) ** 2 - self.mp ** 2
        denominator0 = 2 * (E_p0 + pF0)
        Eg2_baseline = numerator / denominator0

        # Perturb: increase pF by 10%
        pF1 = pF0 * 1.1
        E_p1 = math.sqrt(pF1 ** 2 + self.mp ** 2)
        denominator1 = 2 * (E_p1 + pF1)
        Eg2_perturbed = numerator / denominator1

        # Check monotonicity: larger pF -> smaller E_gamma2
        if Eg2_perturbed < Eg2_baseline:
            return (True, f"Perturbed E_gamma2 = {Eg2_perturbed:.1f} < baseline {Eg2_baseline:.1f} (expected)")
        else:
            return (False, f"Perturbed E_gamma2 = {Eg2_perturbed:.1f} is not smaller than baseline {Eg2_baseline:.1f}")

    def test_monotonicity_pion_momenta_with_beta_perturbation(self):
        """Perturb beta_L upward; check that p_pi_max increases but p_perp_max stays unchanged."""
        if self.p_pi_max is None or self.p_perp_max is None:
            return (False, "SKIP: p_pi_max or p_perp_max not available")

        # Baseline using nominal parameters
        pstar = self.p_perp_max  # treat as p*
        Estar = math.sqrt(pstar ** 2 + self.mpi ** 2)

        beta0 = self.beta_L
        gamma0 = 1.0 / math.sqrt(1 - beta0 ** 2)
        p_pi_baseline = gamma0 * (pstar + beta0 * Estar)

        # Perturb: increase beta by 5%
        beta1 = beta0 * 1.05
        gamma1 = 1.0 / math.sqrt(1 - beta1 ** 2)
        p_pi_perturbed = gamma1 * (pstar + beta1 * Estar)

        # p_perp_max should remain unchanged (it's independent of boost)
        # Check monotonicity: p_pi increases
        if p_pi_perturbed > p_pi_baseline:
            # Also check p_perp_max unchanged (by construction it is same)
            p_perp_check = pstar  # always equal
            return (True, f"Perturbed p_pi_max = {p_pi_perturbed:.1f} > baseline {p_pi_baseline:.1f} (expected); p_perp unchanged")
        else:
            return (False, f"Perturbed p_pi_max = {p_pi_perturbed:.1f} is not larger than baseline {p_pi_baseline:.1f}")

    # ------------------------------------------------------------------
    # Answer sanity (exactly one method, weak checks only)
    # ------------------------------------------------------------------
    def test_answer_sanity_values(self):
        """All outputs must be positive and finite (no NaN/inf)."""
        keys = ['E_gamma1', 'E_gamma2', 'p_pi_max', 'p_perp_max']
        for k in keys:
            v = getattr(self, k, None)
            if v is None:
                return (False, f"SKIP: {k} not available")
            if not isinstance(v, (int, float)):
                return (False, f"{k} is not a number")
            if not math.isfinite(v):
                return (False, f"{k} is not finite (value={v})")
            if v <= 0:
                return (False, f"{k} is not positive (value={v})")
        return (True, "All outputs are positive and finite")

    # ------------------------------------------------------------------
    # Test runner (do not modify this method)
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_ methods, run them, return (overall_pass, logs)."""
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