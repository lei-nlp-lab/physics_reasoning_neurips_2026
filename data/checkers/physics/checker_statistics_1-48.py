import math
import traceback


class PhysicsChecker:
    """
    Audits a solution instance for a Carnot refrigerator problem.
    Checks interface completeness, physical invariants, modeling consistency,
    monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {}) or {}
        self.W = results.get('W', None)

        # Extract given parameters (if any)
        # They might be stored as attributes of the solution instance or inside its results.
        # We use getattr with None default.
        self.T_h = getattr(solution_instance, 'T_h', None)
        self.T_c = getattr(solution_instance, 'T_c', None)
        self.m = getattr(solution_instance, 'm', None)
        self.L_f = getattr(solution_instance, 'L_f', None)

        # If givens are not directly present, try to fetch from the results dict
        if self.T_h is None:
            self.T_h = results.get('T_h', None)
        if self.T_c is None:
            self.T_c = results.get('T_c', None)
        if self.m is None:
            self.m = results.get('m', None)
        if self.L_f is None:
            self.L_f = results.get('L_f', None)

        # Convert to float if possible
        for attr in ['T_h', 'T_c', 'm', 'L_f', 'W']:
            val = getattr(self, attr, None)
            if val is not None:
                try:
                    setattr(self, attr, float(val))
                except (ValueError, TypeError):
                    pass  # leave as is, will be caught by interface tests

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_result_W_exists(self):
        """Check that the required result key 'W' is present and numeric."""
        if self.W is None:
            return (False, "Result key 'W' is missing or None.")
        if not isinstance(self.W, (int, float)):
            return (False, f"W is not numeric, got {type(self.W)}.")
        if math.isnan(self.W) or math.isinf(self.W):
            return (False, f"W is NaN or infinite.")
        return (True, "W is present and finite.")

    def test_interface_givens_available(self):
        """Check that all required given parameters are available for further tests."""
        missing = []
        for name, val in [('T_h', self.T_h), ('T_c', self.T_c),
                          ('m', self.m), ('L_f', self.L_f)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}. Cannot run physics tests.")
        return (True, "All necessary givens are present.")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based sanity, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_carnot_cop_relation(self):
        """Check that COP = Q_c / W matches Carnot COP = T_c/(T_h - T_c)."""
        if any(v is None for v in (self.W, self.T_h, self.T_c, self.m, self.L_f)):
            return (False, "SKIP: missing required data for Carnot COP check.")
        if self.W == 0:
            return (False, "W is zero, cannot compute COP.")
        # Compute expected COP from Carnot formula
        cop_carnot = self.T_c / (self.T_h - self.T_c)
        # Compute actual COP from work and latent heat
        Q_c = self.m * self.L_f
        cop_actual = Q_c / self.W
        if math.isclose(cop_actual, cop_carnot, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"COP matches Carnot: actual={cop_actual:.4f}, Carnot={cop_carnot:.4f}.")
        return (False, f"COP mismatch: actual={cop_actual:.4f}, Carnot={cop_carnot:.4f}.")

    def test_invariant_work_positive(self):
        """Work input must be positive for a refrigerator."""
        if self.W is None:
            return (False, "SKIP: W not available.")
        if self.W > 0:
            return (True, f"W = {self.W:.4f} > 0.")
        return (False, f"W = {self.W:.4f} is not positive.")

    def test_invariant_heat_removed_positive(self):
        """Heat removed from cold reservoir (Q_c) must be positive."""
        if any(v is None for v in (self.m, self.L_f)):
            return (False, "SKIP: missing m or L_f.")
        Q_c = self.m * self.L_f
        if Q_c > 0:
            return (True, f"Q_c = {Q_c:.4f} > 0.")
        return (False, f"Q_c = {Q_c:.4f} is not positive.")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_work_formula_derived(self):
        """Check that W computed from m, L_f, T_h, T_c via Carnot relation is consistent with result W."""
        if any(v is None for v in (self.W, self.m, self.L_f, self.T_h, self.T_c)):
            return (False, "SKIP: missing data for work formula check.")
        # Expected minimum work: W = m * L_f * (T_h - T_c) / T_c
        expected_W = self.m * self.L_f * (self.T_h - self.T_c) / self.T_c
        if math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"W = {self.W:.4f} matches expected {expected_W:.4f}.")
        return (False, f"W = {self.W:.4f} differs from expected {expected_W:.4f}.")

    def test_modeling_latent_heat_magnitude(self):
        """L_f should be around 334 kJ/kg (3.34e5 J/kg)."""
        if self.L_f is None:
            return (False, "SKIP: L_f not available.")
        if 3.2e5 <= self.L_f <= 3.5e5:
            return (True, f"L_f = {self.L_f} is within reasonable range for latent heat of fusion of water.")
        return (False, f"L_f = {self.L_f} is outside typical range 3.2e5-3.5e5 J/kg.")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_work_increases_with_th(self):
        """Perturb T_h up and verify that the Carnot work increases."""
        if any(v is None for v in (self.m, self.L_f, self.T_c, self.T_h)):
            return (False, "SKIP: missing data for monotonicity test.")
        # baseline work
        W_baseline = self.m * self.L_f * (self.T_h - self.T_c) / self.T_c
        # perturb T_h upward by 5%
        T_h_perturbed = self.T_h * 1.05
        W_perturbed = self.m * self.L_f * (T_h_perturbed - self.T_c) / self.T_c
        if W_perturbed > W_baseline:
            return (True, "Work increases with T_h (perturbation +5%).")
        return (False, f"Work did not increase: baseline={W_baseline}, perturbed={W_perturbed}.")

    def test_monotonicity_work_decreases_with_tc(self):
        """Perturb T_c up and verify that the Carnot work decreases."""
        if any(v is None for v in (self.m, self.L_f, self.T_c, self.T_h)):
            return (False, "SKIP: missing data for monotonicity test.")
        # baseline work
        W_baseline = self.m * self.L_f * (self.T_h - self.T_c) / self.T_c
        # perturb T_c upward by 2%
        T_c_perturbed = self.T_c * 1.02
        # recompute with same T_h
        W_perturbed = self.m * self.L_f * (self.T_h - T_c_perturbed) / T_c_perturbed
        if W_perturbed < W_baseline:
            return (True, "Work decreases with increased T_c (perturbation +2%).")
        return (False, f"Work did not decrease: baseline={W_baseline}, perturbed={W_perturbed}.")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """Check that W is in the physically plausible range for freezing 3 kg of water."""
        if self.W is None:
            return (False, "SKIP: W not available.")
        # Typical Carnot work for this problem is a few tens of kJ.
        # Accept any value between 10 kJ and 500 kJ.
        if 1e4 < self.W < 5e5:
            return (True, f"W = {self.W:.4f} J is in plausible range (10^4 - 5×10^5 J).")
        return (False, f"W = {self.W:.4f} J is outside plausible range (10^4 - 5×10^5 J).")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (all_pass, logs)."""
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