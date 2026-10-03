import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results from solution_instance.results
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.v_F = solution_instance.results.get('v_F', None)
            self.v_avg = solution_instance.results.get('v_avg', None)
        else:
            self.v_F = None
            self.v_avg = None

        # Attempt to extract given physical inputs (optional)
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
        else:
            givens = {}

        self.n = givens.get('n', None)
        self.hbar = givens.get('hbar', None)
        self.m = givens.get('m', None)

        # Store explanation steps (if present) for reference
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that both v_F and v_avg are present in results."""
        if self.v_F is None:
            return (False, "Missing key: v_F")
        if self.v_avg is None:
            return (False, "Missing key: v_avg")
        return (True, "Both v_F and v_avg are present")

    def test_interface_v_F_type(self):
        """Check that v_F is a finite numeric value."""
        if self.v_F is None:
            return (False, "SKIP: v_F not available")
        try:
            val = float(self.v_F)
            if math.isnan(val) or math.isinf(val):
                return (False, f"v_F is NaN or infinite: {self.v_F}")
            return (True, "v_F is a finite number")
        except (TypeError, ValueError):
            return (False, f"v_F is not numeric: {self.v_F}")

    def test_interface_v_avg_type(self):
        """Check that v_avg is a finite numeric value."""
        if self.v_avg is None:
            return (False, "SKIP: v_avg not available")
        try:
            val = float(self.v_avg)
            if math.isnan(val) or math.isinf(val):
                return (False, f"v_avg is NaN or infinite: {self.v_avg}")
            return (True, "v_avg is a finite number")
        except (TypeError, ValueError):
            return (False, f"v_avg is not numeric: {self.v_avg}")

    # ------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ------------------------------------------------------------------
    def test_invariant_fermi_relation(self):
        """Check v_avg ≈ sqrt(3/5) * v_F (Fermi distribution relation)."""
        if self.v_F is None or self.v_avg is None:
            return (False, "SKIP: v_F or v_avg not available")
        try:
            vF = float(self.v_F)
            vAvg = float(self.v_avg)
        except (TypeError, ValueError):
            return (False, "Cannot convert v_F or v_avg to float")

        expected = math.sqrt(3.0 / 5.0) * vF
        if math.isclose(vAvg, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_avg ≈ sqrt(3/5)*v_F: v_avg={vAvg:.3e}, expected={expected:.3e}")
        else:
            return (False, f"v_avg deviates from sqrt(3/5)*v_F: v_avg={vAvg:.3e}, expected={expected:.3e}")

    def test_invariant_v_avg_less_than_v_F(self):
        """Physical invariant: average random speed < Fermi velocity (v_avg = sqrt(3/5) v_F < v_F)."""
        if self.v_F is None or self.v_avg is None:
            return (False, "SKIP: v_F or v_avg not available")
        try:
            vF = float(self.v_F)
            vAvg = float(self.v_avg)
        except (TypeError, ValueError):
            return (False, "Cannot convert v_F or v_avg to float")

        if vAvg < vF:
            return (True, f"v_avg ({vAvg:.3e}) < v_F ({vF:.3e}) holds")
        else:
            return (False, f"v_avg ({vAvg:.3e}) is not less than v_F ({vF:.3e})")

    def test_invariant_fermi_velocity_formula(self):
        """If n, hbar, m are provided, verify v_F ≈ (hbar/m)*(3π^2 n)^(1/3)."""
        if self.v_F is None or self.n is None or self.hbar is None or self.m is None:
            return (False, "SKIP: required inputs (v_F, n, hbar, m) not all present")
        try:
            vF = float(self.v_F)
            n_val = float(self.n)
            hbar_val = float(self.hbar)
            m_val = float(self.m)
        except (TypeError, ValueError):
            return (False, "Cannot convert one of the inputs to float")

        vF_expected = (hbar_val / m_val) * (3.0 * math.pi**2 * n_val) ** (1.0 / 3.0)
        if math.isclose(vF, vF_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_F ≈ (hbar/m)*(3π²n)^(1/3): v_F={vF:.3e}, expected={vF_expected:.3e}")
        else:
            return (False, f"v_F deviates from formula: v_F={vF:.3e}, expected={vF_expected:.3e}")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_fermi_velocity_with_typical_density(self):
        """
        If v_F is given, verify it is consistent with the Fermi velocity formula
        using typical metal parameters (n = 8.5e22 cm^-3).
        """
        if self.v_F is None:
            return (False, "SKIP: v_F not available")
        try:
            vF = float(self.v_F)
        except (TypeError, ValueError):
            return (False, "v_F not float-convertible")

        # Typical constants for copper (cgs units)
        n_typical = 8.5e22  # cm^-3
        hbar = 1.054571817e-27  # erg*s
        m_e = 9.10938356e-28   # g

        vF_expected = (hbar / m_e) * (3 * math.pi**2 * n_typical) ** (1.0 / 3.0)
        # vF_expected is in cm/s
        if math.isclose(vF, vF_expected, rel_tol=0.5, abs_tol=1e-2):
            return (True, f"v_F ≈ typical Fermi velocity: v_F={vF:.3e}, typical={vF_expected:.3e}")
        else:
            return (False, f"v_F deviates from typical Fermi velocity: v_F={vF:.3e}, typical={vF_expected:.3e}")

    def test_modeling_v_avg_order_of_magnitude(self):
        """
        Check that v_avg is within a factor of 3 of the expected value
        derived from the Fermi gas model with typical metal parameters.
        This is a loose consistency check.
        """
        if self.v_avg is None:
            return (False, "SKIP: v_avg not available")
        try:
            vAvg = float(self.v_avg)
        except (TypeError, ValueError):
            return (False, "v_avg not float-convertible")

        # Compute expected v_avg from Fermi model using typical copper parameters
        n_typical = 8.5e22
        hbar = 1.054571817e-27
        m_e = 9.10938356e-28
        vF_typical = (hbar / m_e) * (3 * math.pi**2 * n_typical) ** (1.0 / 3.0)
        vAvg_expected = math.sqrt(3.0 / 5.0) * vF_typical  # ~7.75e7 cm/s

        # Accept within a factor of 3 (i.e., rel_tol = 2.0)
        if math.isclose(vAvg, vAvg_expected, rel_tol=2.0, abs_tol=1e-2):
            return (True, f"v_avg consistent with Fermi model: v_avg={vAvg:.3e}, expected~{vAvg_expected:.3e}")
        else:
            return (False, f"v_avg inconsistent with Fermi model: v_avg={vAvg:.3e}, expected~{vAvg_expected:.3e}")

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------
    def test_monotonicity_v_F_with_density(self):
        """
        If v_F and number density n are given, check that a small increase in n
        leads to an increase in v_F (monotonic dependence v_F ∝ n^(1/3)).
        """
        if self.v_F is None or self.n is None:
            return (False, "SKIP: v_F or n not available")

        try:
            vF = float(self.v_F)
            n0 = float(self.n)
        except (TypeError, ValueError):
            return (False, "Cannot convert v_F or n to float")

        # Perturb n by +1%
        n1 = n0 * 1.01
        # Recompute v_F using Fermi velocity formula (requires hbar, m; use defaults if missing)
        hbar = float(self.hbar) if self.hbar is not None else 1.054571817e-27
        m_e = float(self.m) if self.m is not None else 9.10938356e-28

        vF_pert = (hbar / m_e) * (3 * math.pi**2 * n1) ** (1.0 / 3.0)
        if vF_pert > vF:
            return (True, f"Increasing n by 1% increases v_F (vF_pert={vF_pert:.3e} > vF={vF:.3e})")
        else:
            return (False, f"Monotonicity violated: vF_pert={vF_pert:.3e} <= vF={vF:.3e}")

    def test_monotonicity_v_avg_with_v_F(self):
        """
        If both v_avg and v_F are given, check that increasing v_F by 10%
        (as a proxy) leads to an increase in v_avg (since v_avg ∝ v_F).
        We recompute v_avg from v_F using the theoretical relation.
        """
        if self.v_F is None or self.v_avg is None:
            return (False, "SKIP: v_F or v_avg not available")

        try:
            vF = float(self.v_F)
            vAvg = float(self.v_avg)
        except (TypeError, ValueError):
            return (False, "Cannot convert v_F or v_avg to float")

        # Baseline v_avg from given v_F
        vAvg_baseline = math.sqrt(3.0 / 5.0) * vF
        # Perturb v_F by +10%
        vF_pert = vF * 1.10
        vAvg_pert = math.sqrt(3.0 / 5.0) * vF_pert
        # The perturbed v_avg should be larger than original v_avg
        if vAvg_pert > vAvg:
            return (True, f"Increasing v_F by 10% increases v_avg (vAvg_pert={vAvg_pert:.3e} > vAvg={vAvg:.3e})")
        else:
            return (False, f"Monotonicity violated: vAvg_pert={vAvg_pert:.3e} <= vAvg={vAvg:.3e}")

    # ------------------------------------------------------------------
    # Weak answer sanity check (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_range(self):
        """
        Broad sanity: v_avg should be positive and less than the speed of light,
        and within plausible range for degenerate electron gas (10^5–10^8 cm/s).
        This is very weak – it does not compute any expected value.
        """
        if self.v_avg is None:
            return (False, "SKIP: v_avg not available")
        try:
            vAvg = float(self.v_avg)
        except (TypeError, ValueError):
            return (False, "v_avg not float-convertible")

        if vAvg <= 0:
            return (False, f"v_avg is not positive: {vAvg}")
        if math.isinf(vAvg) or math.isnan(vAvg):
            return (False, "v_avg is infinite or NaN")
        # Speed of light ~ 3e10 cm/s
        if vAvg > 3e10:
            return (False, f"v_avg exceeds speed of light: {vAvg:.3e} > 3e10")
        # Reasonable lower bound (classical thermal speeds ~10^5)
        if vAvg < 1e5:
            return (False, f"v_avg too small for degenerate electrons: {vAvg:.3e} < 1e5")
        return (True, f"v_avg is plausible: {vAvg:.3e} cm/s")

    # ------------------------------------------------------------------
    # Run all tests
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