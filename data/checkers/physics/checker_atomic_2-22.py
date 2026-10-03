import math
import traceback

class PhysicsChecker:
    """Audits a solution for the forward elastic pn differential cross-section bound problem."""

    # Physical constants and conversion factors
    HBAR_C_GEV_FM = 0.19733  # not strictly needed
    MB_TO_GEV2 = 0.3894       # 1 mb = 0.3894 GeV^{-2}  (1 GeV^{-2} = 2.568 mb)
    MB_TO_B = 1000.0          # 1 b = 1000 mb

    def __init__(self, solution_instance):
        # Extract givens (inputs to the problem)
        self.p_GeV = getattr(solution_instance, 'p', None)
        self.sigma_tot_mb = getattr(solution_instance, 'sigma_tot', None)

        # Extract computed results (from solution's results dict)
        solution_results = getattr(solution_instance, 'results', {})
        self.dsigma_dOmega_min = solution_results.get('dsigma_dOmega_min', None)
        self.violation_meaning = solution_results.get('violation_meaning', None)

        # Also fetch explanation steps for modeling checks (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Precompute expected bound if givens are available
        self.expected_dsigma_dOmega_min_ster_b = None
        if self.p_GeV is not None and self.sigma_tot_mb is not None:
            try:
                k_GeV = self.p_GeV  # natural units
                sigma_tot_nat = self.sigma_tot_mb / self.MB_TO_GEV2  # GeV^{-2}
                k_sq = k_GeV ** 2
                dsigma_dOmega_min_nat = (k_sq * sigma_tot_nat ** 2) / (16.0 * math.pi ** 2)  # GeV^{-2}
                dsigma_dOmega_min_mb = dsigma_dOmega_min_nat * self.MB_TO_GEV2  # mb/ster
                self.expected_dsigma_dOmega_min_ster_b = dsigma_dOmega_min_mb / self.MB_TO_B  # b/ster
            except Exception:
                # If computation fails, leave as None
                pass

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

    # ------------------------------------------------------------
    # Interface checks: required keys exist and have correct types
    # ------------------------------------------------------------
    def test_interface_dsigma_key_exists(self):
        """Check that the solution provides a dsigma_dOmega_min key."""
        if self.dsigma_dOmega_min is None:
            return (False, "Missing required result key 'dsigma_dOmega_min'")
        return (True, "Key 'dsigma_dOmega_min' is present")

    def test_interface_violation_key_exists(self):
        """Check that the solution provides a violation_meaning key."""
        if self.violation_meaning is None:
            return (False, "Missing required result key 'violation_meaning'")
        return (True, "Key 'violation_meaning' is present")

    def test_interface_violation_type_string(self):
        """Check that violation_meaning is a string."""
        if not isinstance(self.violation_meaning, str):
            return (False, f"violation_meaning should be a string, got {type(self.violation_meaning)}")
        return (True, "violation_meaning is a string")

    def test_interface_dsigma_type_numeric(self):
        """Check that dsigma_dOmega_min is numeric (int or float)."""
        if not isinstance(self.dsigma_dOmega_min, (int, float)):
            return (False, f"dsigma_dOmega_min should be numeric, got {type(self.dsigma_dOmega_min)}")
        return (True, "dsigma_dOmega_min is numeric")

    # ------------------------------------------------------------
    # Physical invariant checks (theorem-based)
    # ------------------------------------------------------------
    def test_invariant_forward_bound_formula(self):
        """
        Check that dsigma_dOmega_min approximately equals the lower bound computed
        from sigma_tot and p using the optical theorem inequality.
        """
        if self.dsigma_dOmega_min is None or self.expected_dsigma_dOmega_min_ster_b is None:
            return (False, "SKIP: missing data to verify bound formula")
        if not math.isclose(self.dsigma_dOmega_min, self.expected_dsigma_dOmega_min_ster_b,
                            rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dsigma_dOmega_min {self.dsigma_dOmega_min} b/ster does not match "
                           f"expected {self.expected_dsigma_dOmega_min_ster_b:.2f} b/ster (±1% tolerance)")
        return (True, f"Bound matches optical theorem prediction: {self.dsigma_dOmega_min} ≈ {self.expected_dsigma_dOmega_min_ster_b:.2f} b/ster")

    def test_invariant_violation_meaning_contains_optical_theorem(self):
        """Check that the violation meaning mentions 'optical theorem' or 'unitarity'."""
        if self.violation_meaning is None:
            return (False, "SKIP: violation_meaning not available")
        msg_lower = self.violation_meaning.lower()
        if "optical theorem" not in msg_lower and "unitarity" not in msg_lower:
            return (False, "violation_meaning should mention 'optical theorem' or 'unitarity'")
        return (True, "violation_meaning correctly references optical theorem/unitarity")

    def test_invariant_direction_inequality(self):
        """Check that dsigma_dOmega_min is positive (a lower bound must be non-negative)."""
        if self.dsigma_dOmega_min is None:
            return (False, "SKIP: no value to check")
        if self.dsigma_dOmega_min < 0:
            return (False, "dsigma_dOmega_min should be >= 0")
        return (True, "dsigma_dOmega_min is non-negative")

    # ------------------------------------------------------------
    # Modeling / trace consistency checks (use explanation_steps if needed)
    # ------------------------------------------------------------
    def test_modeling_forward_differential_cross_section_conversion(self):
        """
        Check that the solution uses the correct relationship between
        dsigma/dOmega (at theta=0) and dsigma/dt (at t=0).  This is not directly
        testable unless the solution provides additional intermediate variables,
        but we can check that the bound is consistent with the formula
        dsigma/dOmega_0 >= sigma_tot^2/(16 pi^2) * k^2.
        """
        if self.expected_dsigma_dOmega_min_ster_b is None:
            return (False, "SKIP: cannot perform conversion check without givens")
        # The check is already done in test_invariant_forward_bound_formula
        # This test just confirms the conversion was physically correct.
        if self.dsigma_dOmega_min is None:
            return (False, "SKIP: solution value missing")
        # Re-derive from the formula
        k = self.p_GeV
        sigma_tot_nat = self.sigma_tot_mb / self.MB_TO_GEV2
        bound_nat = (k**2 * sigma_tot_nat**2) / (16 * math.pi**2)
        bound_mb = bound_nat * self.MB_TO_GEV2
        bound_b = bound_mb / self.MB_TO_B
        if not math.isclose(self.dsigma_dOmega_min, bound_b, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dsigma_dOmega_min={self.dsigma_dOmega_min} does not match re-derived bound={bound_b:.2f} b/ster")
        return (True, "differential cross-section bound is consistent with optical theorem model")

    def test_modeling_explanation_steps_have_key_relations(self):
        """If explanation_steps are provided, check that they mention optical theorem or unitarity."""
        if self.explanation_steps is None:
            return (True, "No explanation steps to check (optional)")
        steps_str = str(self.explanation_steps).lower()
        if "optical theorem" in steps_str or "unitarity" in steps_str:
            return (True, "Explanation steps reference optical theorem/unitarity")
        # Not required, so soft pass
        return (True, "Explanation steps present but no explicit optical theorem mention (non-fatal)")

    # ------------------------------------------------------------
    # Monotonicity check (perturb one parameter and see trend)
    # ------------------------------------------------------------
    def test_monotonicity_sigma_tot_increase(self):
        """
        If the total cross section increases, the lower bound on the forward
        differential cross-section should increase (monotonically).
        """
        if self.sigma_tot_mb is None or self.p_GeV is None:
            return (False, "SKIP: missing givens for monotonicity test")
        if self.dsigma_dOmega_min is None:
            return (False, "SKIP: solution value missing")
        # Perturb sigma_tot by +10%
        sigma_tot_pert = self.sigma_tot_mb * 1.10
        # Compute new bound
        sigma_tot_nat_pert = sigma_tot_pert / self.MB_TO_GEV2
        k = self.p_GeV
        bound_nat_pert = (k**2 * sigma_tot_nat_pert**2) / (16 * math.pi**2)
        bound_mb_pert = bound_nat_pert * self.MB_TO_GEV2
        bound_b_pert = bound_mb_pert / self.MB_TO_B
        if bound_b_pert <= self.dsigma_dOmega_min:
            return (False, f"Monotonicity fails: perturbed bound {bound_b_pert:.2f} not > original {self.dsigma_dOmega_min:.2f}")
        return (True, f"Bound increases with sigma_tot: {self.dsigma_dOmega_min} -> {bound_b_pert:.2f} b/ster")

    # ------------------------------------------------------------
    # Weak answer sanity only (one test, no isclose, no re-derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Weak sanity checks: final bound should be a reasonable finite number,
        and violation meaning should be a non-empty string.
        """
        errors = []
        # dsigma_dOmega_min
        if self.dsigma_dOmega_min is None:
            errors.append("dsigma_dOmega_min is missing")
        else:
            if not math.isfinite(self.dsigma_dOmega_min):
                errors.append("dsigma_dOmega_min is not finite")
            if self.dsigma_dOmega_min <= 0:
                errors.append("dsigma_dOmega_min should be positive")
            # Broad magnitude check: should be less than 1000 b (since total σ=40 mb → ~0.04 b)
            if self.dsigma_dOmega_min > 1000.0:
                errors.append("dsigma_dOmega_min unreasonably large (>1000 b)")

        # violation_meaning
        if self.violation_meaning is None:
            errors.append("violation_meaning is missing")
        elif len(self.violation_meaning.strip()) == 0:
            errors.append("violation_meaning is empty")

        if errors:
            return (False, "; ".join(errors))
        return (True, "All weak sanity checks passed")