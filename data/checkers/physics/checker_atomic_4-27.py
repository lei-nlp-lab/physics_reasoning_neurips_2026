import math
import traceback

class PhysicsChecker:
    """
    Checks a solution for the Z0 production and decay problem.
    """

    def __init__(self, solution_instance):
        # Extract results safely – only expect these keys
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.E_e_coll = results.get('E_e_coll', None)
        self.E_beam_fixed = results.get('E_beam_fixed', None)
        self.E_Z = results.get('E_Z', None)
        self.beta_Z = results.get('beta_Z', None)
        self.E_mu_max = results.get('E_mu_max', None)
        # Optional intermediate p_Z (might be present)
        self.p_Z = results.get('p_Z', None)

        # Extract givens (M_Z, m_e) – problem must provide them
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}
        self.M_Z = givens.get('M_Z', None)
        self.m_e = givens.get('m_e', None)

    # ----------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------
    def test_interface_required_keys(self):
        """All five required result keys must exist."""
        keys = ['E_e_coll', 'E_beam_fixed', 'E_Z', 'beta_Z', 'E_mu_max']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing results keys: {missing}")
        return (True, "All required keys present")

    def test_interface_values_positive_finite(self):
        """All five results must be positive finite numbers."""
        vals = [self.E_e_coll, self.E_beam_fixed, self.E_Z, self.beta_Z, self.E_mu_max]
        for v in vals:
            if v is None:
                return (False, "One of the values is None (likely test_interface_required_keys failed)")
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                return (False, f"Value {v} is not a positive finite number")
        return (True, "All values are positive finite numbers")

    # ----------------------------------------------------------------
    # Invariant / theorem checks (using math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_colliding_beam(self):
        """Check that 2*E_e_coll = M_Z (colliding beam threshold)."""
        MZ = self.M_Z
        E = self.E_e_coll
        if MZ is None or E is None:
            return (False, "SKIP: missing M_Z or E_e_coll")
        if not math.isclose(2 * E, MZ, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"2*E_e_coll = {2*E} != M_Z = {MZ}")
        return (True, f"2*E_e_coll = {2*E} ~ M_Z = {MZ}")

    def test_invariant_fixed_target_mass(self):
        """Invariant mass squared for fixed target: 2 m_e E_beam + 2 m_e^2 = M_Z^2."""
        MZ = self.M_Z
        me = self.m_e
        E_beam = self.E_beam_fixed
        if MZ is None or me is None or E_beam is None:
            return (False, "SKIP: missing M_Z, m_e, or E_beam_fixed")
        expected_s = MZ ** 2
        computed_s = 2 * me * E_beam + 2 * me ** 2
        if not math.isclose(computed_s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Invariant mass^2 check: {computed_s} != {expected_s}")
        return (True, f"Invariant mass squared consistent: {computed_s} ~ {expected_s}")

    def test_invariant_Z_energy_from_beam(self):
        """E_Z must equal E_beam_fixed + m_e."""
        E_beam = self.E_beam_fixed
        me = self.m_e
        EZ = self.E_Z
        if E_beam is None or me is None or EZ is None:
            return (False, "SKIP: missing E_beam_fixed, m_e, or E_Z")
        expected = E_beam + me
        if not math.isclose(EZ, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_Z = {EZ} != E_beam + m_e = {expected}")
        return (True, f"E_Z = {EZ} ~ {expected}")

    def test_invariant_Z_beta_from_energy(self):
        """beta_Z = sqrt(1 - (M_Z/E_Z)^2)."""
        MZ = self.M_Z
        EZ = self.E_Z
        beta = self.beta_Z
        if MZ is None or EZ is None or beta is None:
            return (False, "SKIP: missing M_Z, E_Z, or beta_Z")
        if EZ <= MZ:
            return (False, f"E_Z = {EZ} <= M_Z = {MZ}, impossible for a real Z")
        expected_beta = math.sqrt(1 - (MZ / EZ) ** 2)
        if not math.isclose(beta, expected_beta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"beta_Z = {beta} != expected {expected_beta}")
        return (True, f"beta_Z = {beta} ~ {expected_beta}")

    def test_invariant_muon_max_energy(self):
        """E_mu_max = (E_Z + p_Z)/2, with p_Z = sqrt(E_Z^2 - M_Z^2)."""
        MZ = self.M_Z
        EZ = self.E_Z
        E_mu = self.E_mu_max
        # p_Z may be provided or we can compute
        pZ = self.p_Z
        if MZ is None or EZ is None or E_mu is None:
            return (False, "SKIP: missing M_Z, E_Z, or E_mu_max")
        if pZ is None:
            pZ = math.sqrt(max(0, EZ**2 - MZ**2))   # guard against tiny negative
        expected = (EZ + pZ) / 2.0
        if not math.isclose(E_mu, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_mu_max = {E_mu} != (E_Z+p_Z)/2 = {expected}")
        return (True, f"E_mu_max = {E_mu} ~ {expected}")

    # ----------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------
    def test_modeling_Z_energy_not_less_than_MZ(self):
        """E_Z must be >= M_Z (since Z is produced on shell)."""
        MZ = self.M_Z
        EZ = self.E_Z
        if MZ is None or EZ is None:
            return (False, "SKIP: missing M_Z or E_Z")
        if EZ < MZ - 1e-9:
            return (False, f"E_Z = {EZ} < M_Z = {MZ}")
        return (True, f"E_Z = {EZ} >= M_Z = {MZ}")

    def test_modeling_muon_max_energy_less_than_EZ(self):
        """Maximum muon energy cannot exceed total Z energy in lab."""
        EZ = self.E_Z
        E_mu = self.E_mu_max
        if EZ is None or E_mu is None:
            return (False, "SKIP: missing E_Z or E_mu_max")
        if E_mu > EZ + 1e-9:
            return (False, f"E_mu_max = {E_mu} > E_Z = {EZ}")
        return (True, f"E_mu_max = {E_mu} <= E_Z = {EZ}")

    # ----------------------------------------------------------------
    # Monotonicity checks (input perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_beta_increases_with_EZ(self):
        """
        Beta_Z should increase when E_Z increases (holding M_Z fixed).
        Perturb E_Z by +1% and verify the new beta is larger.
        """
        MZ = self.M_Z
        EZ = self.E_Z
        beta = self.beta_Z
        if MZ is None or EZ is None or beta is None:
            return (False, "SKIP: missing M_Z, E_Z, or beta_Z")
        if EZ <= MZ:
            return (False, "E_Z <= M_Z, cannot compute beta")
        # Baseline
        baseline_beta = math.sqrt(1 - (MZ / EZ) ** 2)
        # Perturb E_Z by +1%
        EZ_pert = EZ * 1.01
        perturbed_beta = math.sqrt(1 - (MZ / EZ_pert) ** 2)
        if perturbed_beta <= baseline_beta - 1e-12:
            return (False, f"Perturbed beta = {perturbed_beta} <= baseline {baseline_beta}")
        return (True, f"Beta increases with E_Z: baseline {baseline_beta:.6f} -> perturbed {perturbed_beta:.6f}")

    # ----------------------------------------------------------------
    # Weak final‑answer sanity only
    # ----------------------------------------------------------------
    def test_answer_sanity_broad_ranges(self):
        """Very loose bounds: everything should be plausible."""
        msgs = []
        # E_e_coll should be around 46 GeV
        E_coll = self.E_e_coll
        if E_coll is not None:
            if not (10 < E_coll < 100):
                msgs.append(f"E_e_coll = {E_coll} outside [10,100] GeV")
        # E_beam_fixed should be enormous (> 1e6)
        E_beam = self.E_beam_fixed
        if E_beam is not None:
            if E_beam < 1e6:
                msgs.append(f"E_beam_fixed = {E_beam} < 1e6 GeV, unlikely")
        # E_Z similarly huge
        EZ = self.E_Z
        if EZ is not None:
            if EZ < 1e6:
                msgs.append(f"E_Z = {EZ} < 1e6 GeV, unlikely")
        # beta_Z must be very close to 1
        beta = self.beta_Z
        if beta is not None:
            if not (0.99 < beta <= 1.0):
                msgs.append(f"beta_Z = {beta} not in (0.99, 1.0]")
        # E_mu_max positive and not larger than E_Z (already checked elsewhere)
        E_mu = self.E_mu_max
        if E_mu is not None:
            if E_mu <= 0:
                msgs.append(f"E_mu_max = {E_mu} not positive")
        if msgs:
            return (False, "; ".join(msgs))
        return (True, "All quantities pass broad sanity ranges")

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)