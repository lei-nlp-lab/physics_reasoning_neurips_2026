import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a SolutionInstance for correctness regarding electrostatic energy
    reduction in nuclear fission.
    """

    def __init__(self, solution_instance):
        # ----------------------------------------------------------------------
        # 1. Extract the main result (delta_U_MeV) safely
        # ----------------------------------------------------------------------
        results = getattr(solution_instance, 'results', None)
        if results is not None and isinstance(results, dict):
            self.delta_U_MeV = results.get('delta_U_MeV', None)
        else:
            self.delta_U_MeV = None

        # ----------------------------------------------------------------------
        # 2. Extract given constants and parameters safely
        #    Expected: Z, A, R0 (radius constant), k, e, MeV (conversion)
        #    They may be in solution_instance.givens, solution_instance.params,
        #    or solution_instance.results.  Try several sources.
        # ----------------------------------------------------------------------
        # Helper to fetch from a dict attribute
        def _get_from(source, key, default=None):
            if source is None:
                return default
            return source.get(key, default)

        givens = getattr(solution_instance, 'givens', None)
        params = getattr(solution_instance, 'params', None)

        # try also results (some implementors put constants there)
        self.Z = _get_from(givens, 'Z', _get_from(params, 'Z', None))
        self.A = _get_from(givens, 'A', _get_from(params, 'A', None))
        self.R0 = _get_from(givens, 'R0', _get_from(params, 'R0', 1.2e-15))  # in meters
        self.k = _get_from(givens, 'k', _get_from(params, 'k', 8.987551787e9))
        self.e = _get_from(givens, 'e', _get_from(params, 'e', 1.602176634e-19))
        self.MeV_factor = _get_from(givens, 'MeV', _get_from(params, 'MeV', 1.602176634e-13))

        # optional intermediate results (U_i, U_f, delta_U_J) – not mandatory
        self.U_i = results.get('U_i', None) if results else None
        self.U_f = results.get('U_f', None) if results else None
        self.delta_U_J = results.get('delta_U_J', None) if results else None

    # ------------------------------------------------------------------
    # Helper: compute ΔU (MeV) from given Z, A
    # ------------------------------------------------------------------
    def _compute_delta_U_MeV(self, Z, A):
        """
        Return ΔU in MeV using constants stored in the instance.
        If a constant is missing, return None.
        """
        if None in (Z, A, self.R0, self.k, self.e, self.MeV_factor):
            return None
        R = self.R0 * (A ** (1.0 / 3.0))
        Rf = self.R0 * ((A / 2.0) ** (1.0 / 3.0))
        Ui = (3.0 / 5.0) * self.k * ((Z * self.e) ** 2) / R
        Uf = 2.0 * (3.0 / 5.0) * self.k * (((Z / 2.0) * self.e) ** 2) / Rf
        delta_J = Ui - Uf
        return delta_J / self.MeV_factor

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------

    # -------- interface tests --------

    def test_interface_delta_U_MeV_exists(self):
        if self.delta_U_MeV is None:
            return (False, "delta_U_MeV is missing from results")
        return (True, "delta_U_MeV is present")

    def test_interface_delta_U_MeV_numeric(self):
        if self.delta_U_MeV is None:
            return (False, "SKIP: delta_U_MeV not available")
        if not isinstance(self.delta_U_MeV, (int, float)):
            return (False, f"delta_U_MeV is not numeric: {type(self.delta_U_MeV)}")
        if math.isnan(self.delta_U_MeV) or math.isinf(self.delta_U_MeV):
            return (False, f"delta_U_MeV is NaN or Inf: {self.delta_U_MeV}")
        return (True, "delta_U_MeV is a finite numeric value")

    # -------- invariant tests (theorem-based, use math.isclose) --------

    def test_invariant_radius_ratio(self):
        """Check that Rf / R = (1/2)^(1/3)."""
        if self.Z is None or self.A is None:
            return (False, "SKIP: Z or A not provided")
        R = self.R0 * (self.A ** (1.0 / 3.0))
        Rf = self.R0 * ((self.A / 2.0) ** (1.0 / 3.0))
        expected_ratio = (0.5) ** (1.0 / 3.0)
        if math.isclose(Rf / R, expected_ratio, rel_tol=1e-2):
            return (True, f"Rf/R = {Rf/R:.5f} ≈ {expected_ratio:.5f}")
        else:
            return (False, f"Rf/R = {Rf/R:.5f} but expected ~{expected_ratio:.5f}")

    def test_invariant_energy_reduction_positive(self):
        if self.delta_U_MeV is None:
            return (False, "SKIP: delta_U_MeV not available")
        if self.delta_U_MeV > 0:
            return (True, f"ΔU = {self.delta_U_MeV:.2f} MeV > 0")
        else:
            return (False, f"ΔU must be positive, got {self.delta_U_MeV:.2f} MeV")

    def test_invariant_initial_self_energy_positive(self):
        """If U_i stored, check > 0."""
        if self.U_i is None:
            return (False, "SKIP: U_i not provided")
        if self.U_i > 0:
            return (True, f"U_i = {self.U_i:.2e} J > 0")
        else:
            return (False, f"U_i must be positive, got {self.U_i:.2e} J")

    def test_invariant_final_self_energy_positive(self):
        if self.U_f is None:
            return (False, "SKIP: U_f not provided")
        if self.U_f > 0:
            return (True, f"U_f = {self.U_f:.2e} J > 0")
        else:
            return (False, f"U_f must be positive, got {self.U_f:.2e} J")

    # -------- modeling / trace tests (check relations from steps) --------

    def test_modeling_conversion_J_to_MeV(self):
        """If both delta_U_J and MeV factor are available, check conversion."""
        if self.delta_U_J is None or self.delta_U_MeV is None or self.MeV_factor is None:
            return (False, "SKIP: delta_U_J or delta_U_MeV or MeV factor missing")
        expected = self.delta_U_J / self.MeV_factor
        if math.isclose(self.delta_U_MeV, expected, rel_tol=1e-2):
            return (True, f"delta_U_MeV from delta_U_J: {self.delta_U_MeV:.2f} ≈ {expected:.2f}")
        else:
            return (False, f"Conversion mismatch: delta_U_MeV={self.delta_U_MeV:.2f} but delta_U_J/MeV={expected:.2f}")

    def test_modeling_self_energy_consistency(self):
        """
        If Z, A, and the stored U_i, U_f are all given, check that they
        satisfy the theoretical formula to within ~2%.
        """
        if None in (self.Z, self.A, self.U_i, self.U_f):
            return (False, "SKIP: missing Z, A, or stored U_i/U_f")
        R = self.R0 * (self.A ** (1.0 / 3.0))
        Rf = self.R0 * ((self.A / 2.0) ** (1.0 / 3.0))
        expected_Ui = (3.0 / 5.0) * self.k * ((self.Z * self.e) ** 2) / R
        expected_Uf = 2.0 * (3.0 / 5.0) * self.k * (((self.Z / 2.0) * self.e) ** 2) / Rf
        ok_i = math.isclose(self.U_i, expected_Ui, rel_tol=1e-2, abs_tol=1e-2)
        ok_f = math.isclose(self.U_f, expected_Uf, rel_tol=1e-2, abs_tol=1e-2)
        if ok_i and ok_f:
            return (True, "Stored U_i and U_f match theoretical formula")
        msgs = []
        if not ok_i:
            msgs.append(f"U_i discrepancy: stored {self.U_i:.2e}, expected {expected_Ui:.2e}")
        if not ok_f:
            msgs.append(f"U_f discrepancy: stored {self.U_f:.2e}, expected {expected_Uf:.2e}")
        return (False, "; ".join(msgs))

    def test_modeling_delta_U_derived_from_givens(self):
        """
        If Z, A are given, recompute ΔU ourselves and compare with stored value.
        This is allowed because we re‑derive it inside the checker.
        """
        if None in (self.Z, self.A) or self.delta_U_MeV is None:
            return (False, "SKIP: missing Z, A, or delta_U_MeV")
        recomputed = self._compute_delta_U_MeV(self.Z, self.A)
        if recomputed is None:
            return (False, "SKIP: cannot recompute ΔU (constants missing)")
        if math.isclose(self.delta_U_MeV, recomputed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Self-consistency: stored ΔU = {self.delta_U_MeV:.1f} MeV, "
                          f"recomputed ≈ {recomputed:.1f} MeV")
        else:
            return (False, f"ΔU mismatch: stored {self.delta_U_MeV:.1f} MeV, "
                          f"recomputed {recomputed:.1f} MeV")

    # -------- monotonicity tests (perturb input, check sign of Δ) --------

    def test_monotonicity_Z_increases_deltaU(self):
        """
        If Z increases by 10%, ΔU should increase (since ΔU ∝ Z^2 approximately).
        We compute our own ΔU for baseline Z and for perturbed Z.
        """
        if self.Z is None or self.A is None:
            return (False, "SKIP: Z or A not provided")
        base = self._compute_delta_U_MeV(self.Z, self.A)
        perturbed_Z = self.Z * 1.10
        perturbed = self._compute_delta_U_MeV(perturbed_Z, self.A)
        if base is None or perturbed is None:
            return (False, "SKIP: cannot compute ΔU (constants missing)")
        msg = f"Z={self.Z} → ΔU={base:.1f} MeV; Z={perturbed_Z:.1f} → ΔU={perturbed:.1f} MeV"
        if perturbed > base:
            return (True, f"Z↑ ⇒ ΔU↑ (monotonic) – {msg}")
        else:
            return (False, f"Z↑ should increase ΔU but got {msg}")

    def test_monotonicity_A_decreases_deltaU(self):
        """
        If A increases by 10%, R increases so ΔU should decrease (1/R factor).
        """
        if self.Z is None or self.A is None:
            return (False, "SKIP: Z or A not provided")
        base = self._compute_delta_U_MeV(self.Z, self.A)
        perturbed_A = self.A * 1.10
        perturbed = self._compute_delta_U_MeV(self.Z, perturbed_A)
        if base is None or perturbed is None:
            return (False, "SKIP: cannot compute ΔU (constants missing)")
        msg = f"A={self.A} → ΔU={base:.1f} MeV; A={perturbed_A:.1f} → ΔU={perturbed:.1f} MeV"
        if perturbed < base:
            return (True, f"A↑ ⇒ ΔU↓ (monotonic) – {msg}")
        else:
            return (False, f"A↑ should decrease ΔU but got {msg}")

    # -------- weak answer sanity (no exact value, no isclose, no re-derivation) --------

    def test_answer_sanity_delta_U_MeV_magnitude(self):
        """Only check sign, finiteness, and broad order of magnitude."""
        if self.delta_U_MeV is None:
            return (False, "SKIP: delta_U_MeV not available")
        if not (100 < self.delta_U_MeV < 500):
            return (False, f"ΔU = {self.delta_U_MeV:.1f} MeV seems unreasonable "
                           f"(expected around a few hundred MeV)")
        return (True, f"ΔU = {self.delta_U_MeV:.1f} MeV is in plausible range (100–500 MeV)")

    # ------------------------------------------------------------------
    # Runner: discover all test_ methods and run them
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