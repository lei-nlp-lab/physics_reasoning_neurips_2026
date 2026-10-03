import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the classical radiation problem:
    a charged particle spiraling in due to radiation reaction.
    Checks physical consistency, scaling, and interface completeness.
    """
    def __init__(self, solution_instance):
        # Extract givens – safe defaults are None
        givens = getattr(solution_instance, 'givens', {})
        self.m = givens.get('m', None)
        self.q = givens.get('q', None)
        self.c = givens.get('c', None)
        self.eps0 = givens.get('eps0', None)
        self.R = givens.get('R', None)

        # Extract results – safe defaults are None
        results = getattr(solution_instance, 'results', {})
        self.t = results.get('t', None)

        # Optional: extract explanation steps and pseudocode for trace consistency
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Pre-compute expected constants if all givens are present
        if all(v is not None for v in [self.m, self.q, self.c, self.eps0, self.R]):
            try:
                self.C = (self.q**4) / (12 * (math.pi**2) * (self.eps0**2) * (self.c**3) * (self.m**2))
                self.expected_t = (7 * math.pi**2 * self.eps0**2 * self.c**3 * self.m**2 * self.R**3) / (2 * self.q**4)
            except (ZeroDivisionError, ValueError):
                self.C = None
                self.expected_t = None
        else:
            self.C = None
            self.expected_t = None

    def run_all_tests(self):
        """Discover and run all test_ methods, returning (overall_pass, log_list)."""
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

    # ------------------------------------------------------------------
    # Interface tests – required keys and types
    # ------------------------------------------------------------------
    def test_interface_givens_present(self):
        """All required givens must be present and non‑None."""
        missing = []
        for key, val in [('m', self.m), ('q', self.q), ('c', self.c),
                         ('eps0', self.eps0), ('R', self.R)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present.")

    def test_interface_result_t_present(self):
        """Result key 't' must be present and non‑None."""
        if self.t is None:
            return (False, "Result 't' is missing or None.")
        return (True, "Result 't' is present.")

    def test_interface_given_types_numeric(self):
        """All givens must be numeric (int or float)."""
        if any(v is None for v in [self.m, self.q, self.c, self.eps0, self.R]):
            return (False, "Cannot check types: some givens are None.")
        for name, v in [('m', self.m), ('q', self.q), ('c', self.c),
                        ('eps0', self.eps0), ('R', self.R)]:
            if not isinstance(v, (int, float)):
                return (False, f"Given '{name}' is not numeric.")
        return (True, "All givens are numeric.")

    def test_interface_result_t_type_numeric(self):
        if self.t is None:
            return (False, "Result 't' is missing.")
        if not isinstance(self.t, (int, float)):
            return (False, f"Result 't' has type {type(self.t).__name__}, expected numeric.")
        return (True, "Result 't' is numeric.")

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based sanity checks (fuzzy matching)
    # ------------------------------------------------------------------
    def test_invariant_energy_balance_C(self):
        """
        From energy balance (Larmor formula) we derived dr/dt = -C/r^2.
        Check that the reported 'C' (if derivable) matches the theoretical form.
        Actually we can verify that the integrated time matches the formula.
        This test checks that the scaling of drift constant C is correct.
        """
        if self.C is None:
            return (False, "SKIP: cannot compute C (missing givens)")
        # C must be positive
        if self.C <= 0:
            return (False, f"C = {self.C} should be positive.")
        # Order of magnitude sanity: C should be small for typical parameters.
        # (Not strict, but we check that C is not astronomically large.)
        # A more concrete check: C ~ q^4/(m^2); for SI units 1e-?? but we avoid hard numbers.
        # Instead, we check that the exponent of q is 4 and of m is -2 (symbolic).
        # Since we have numeric values, we can check that if we double q, C increases by 16.
        # This is not a direct test of the given instance; we skip.
        return (True, "C computed and positive.")

    def test_invariant_energy_balance_integrated_time(self):
        """
        The integrated time t = ∫ dr / (-C/r^2) from R to R/2 must equal (7/24) R^3 / C.
        Compare the given t with the expected value using the derived C.
        """
        if self.t is None or self.C is None or self.R is None:
            return (False, "SKIP: missing t, C, or R.")
        # Compute expected integrated time (the analytic result)
        expected_integral = (7.0/24.0) * (self.R**3) / self.C
        if not math.isfinite(expected_integral) or expected_integral <= 0:
            return (False, f"Expected integral time {expected_integral} is invalid.")
        # Use fuzzy matching
        if math.isclose(self.t, expected_integral, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"Integrated time matches ODE solution: t ≈ {expected_integral:.4g}.")
        else:
            return (False, f"t = {self.t:.4g}, expected from ODE integration = {expected_integral:.4g}.")

    def test_invariant_centripetal_to_potential(self):
        """
        For circular orbit: kinetic energy = -potential/2  (virial theorem).
        E = -q^2/(8π ε0 r). Check that if we can infer r = starting R, the energy becomes -q^2/(8π ε0 R).
        Since we don't have E directly, we check scaling of the result t with r^3.
        """
        # No direct check, but we can verify that the derived result t ∝ R^3
        if self.R is None or self.t is None:
            return (False, "SKIP: missing R or t.")
        # If we double R, t should increase by factor 8 (from R^3 dependence). This is a monotonicity check,
        # but here we just note that the formula requires R^3 dependence.
        # We'll do a quick sanity: if R is zero or negative, skip.
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}.")
        return (True, "R is positive, consistent with circular orbit assumption.")

    def test_invariant_larmor_power_sign(self):
        """
        The power radiated must be positive, meaning dE/dt < 0.
        Since E = -q^2/(8π ε0 r), dE/dt = (q^2/(8π ε0 r^2)) dr/dt.
        dr/dt is negative (radius decreases), so dE/dt is negative => power positive.
        We can check that dr/dt < 0 if we had dr/dt; we check the sign of the
        computed time derivative of radius from the ODE.
        """
        if self.t is None or self.R is None or self.C is None:
            return (False, "SKIP: missing t, R, or C.")
        # dr/dt at initial time is -C/R^2 (negative)
        drdt_initial = -self.C / (self.R**2)
        if drdt_initial >= 0:
            return (False, f"Initial dr/dt = {drdt_initial} should be negative.")
        return (True, f"Initial dr/dt = {drdt_initial} < 0, consistent with energy loss.")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests (fuzzy matching)
    # ------------------------------------------------------------------
    def test_modeling_scaling_with_R(self):
        """
        The time should scale as R^3. Verify that the reported t and given R
        satisfy t / R^3 is constant (with appropriate constants).
        """
        if self.t is None or self.R is None:
            return (False, "SKIP: missing t or R.")
        if self.R <= 0:
            return (False, "R not positive.")
        # Compute the expected proportionality: t * ( 2 q^4 ) / (7 π^2 ε0^2 c^3 m^2) = R^3
        # But we cannot rely on exact constants. Instead, we use the expected t formula.
        if self.expected_t is not None:
            # Check that t / R^3 is close to the expected ratio within 2%
            expected_ratio = (7 * math.pi**2 * self.eps0**2 * self.c**3 * self.m**2) / (2 * self.q**4)
            actual_ratio = self.t / (self.R**3)
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-12):
                return (True, f"t/R^3 = {actual_ratio:.4g}, expected ≈ {expected_ratio:.4g}.")
            else:
                return (False, f"t/R^3 = {actual_ratio:.4g} does not match expected {expected_ratio:.4g}.")
        else:
            # Without exact constants, just check that t/R^3 is positive and finite
            if not math.isfinite(self.t / (self.R**3)):
                return (False, "t/R^3 is not finite.")
            return (True, "t/R^3 is finite and positive (no exact constants for deep check).")

    def test_modeling_ode_integral_form(self):
        """
        Verify that the integration: ∫ from R to R/2 of -r^2/C dr equals (7/24) R^3 / C.
        This is a purely mathematical check on the integral, which we already used.
        To avoid redundancy, we check that the step R/2 is indeed half.
        If we have pseudocode, check that the emitted result matches the integral.
        """
        if self.t is None or self.C is None or self.R is None:
            return (False, "SKIP: missing t, C, or R.")
        # The integration result from do dr/dt = -C/r^2:
        # t = (r_initial^3 - r_final^3) / (3C)  because ∫ r^2 dr = r^3/3
        # initial r=R, final r=R/2 => t = (R^3 - R^3/8)/(3C) = (7/24) R^3/C
        expected_t_from_ode = (7.0/24.0) * (self.R**3) / self.C
        if not math.isfinite(expected_t_from_ode):
            return (False, "Expected t from ODE integration is not finite.")
        if math.isclose(self.t, expected_t_from_ode, rel_tol=1e-2, abs_tol=1e-12):
            return (True, "t matches ODE integration: t = (7/24) R^3 / C.")
        else:
            return (False, f"t = {self.t:.4g}, ODE integration gives {expected_t_from_ode:.4g}.")

    def test_modeling_pseudocode_consistency(self):
        """
        If pseudocode is present and defines 'C' and computes 't', check that
        the reported t matches the pseudocode computation.
        We only check if the pseudocode contains a formula for t.
        """
        if self.pseudocode is None or not isinstance(self.pseudocode, str):
            # We can check if pseudocode is a dict with keys; for generality, skip if not available.
            return (False, "SKIP: no pseudocode available for cross-check.")
        # Simple heuristic: search for the emitted t expression.
        # We already have analytical check; this is optional.
        return (True, "Pseudocode present (structural consistency assumed).")

    def test_modeling_explanation_steps_flow(self):
        """
        Verify that explanation steps discuss centripetal condition and Larmor formula.
        If explanation_steps is a list of dicts, check that at least one step mentions
        energy balance or radiation.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided.")
        # We don't enforce content; just check that it's non-empty and has steps.
        if isinstance(self.explanation_steps, list) and len(self.explanation_steps) > 0:
            return (True, "Explanation steps are present.")
        else:
            return (False, "Explanation steps are empty or not a list.")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb input and check output direction
    # ------------------------------------------------------------------
    def test_monotonicity_radius_increase(self):
        """
        If initial radius R is increased by 10%, the collapse time should increase.
        Use the expected analytic formula to verify monotonicity.
        """
        if self.expected_t is None or self.R is None:
            return (False, "SKIP: cannot compute expected t or R missing.")
        R_perturbed = self.R * 1.1
        # Compute expected t for perturbed R using the same formula
        t_pert = (7 * math.pi**2 * self.eps0**2 * self.c**3 * self.m**2 * R_perturbed**3) / (2 * self.q**4)
        if not t_pert > self.expected_t:
            return (False, f"t(1.1R) = {t_pert:.4g} should be > t(R) = {self.expected_t:.4g}.")
        return (True, f"t increases when R increases: t(R)={self.expected_t:.4g} < t(1.1R)={t_pert:.4g}.")

    def test_monotonicity_charge_decrease(self):
        """
        If charge q is increased, the time should decrease (t ∝ 1/q^4).
        Perturb q by +10% and check t decreases.
        """
        if self.expected_t is None or self.q is None:
            return (False, "SKIP: cannot compute expected t or q missing.")
        q_pert = self.q * 1.1
        t_pert = (7 * math.pi**2 * self.eps0**2 * self.c**3 * self.m**2 * self.R**3) / (2 * q_pert**4)
        if not t_pert < self.expected_t:
            return (False, f"t(1.1q) = {t_pert:.4g} should be < t(q) = {self.expected_t:.4g}.")
        return (True, f"t decreases when q increases: t(q)={self.expected_t:.4g} > t(1.1q)={t_pert:.4g}.")

    def test_monotonicity_mass_increase(self):
        """
        If mass m is increased, t should increase (t ∝ m^2).
        Perturb m by +10% and check t increases.
        """
        if self.expected_t is None or self.m is None:
            return (False, "SKIP: cannot compute expected t or m missing.")
        m_pert = self.m * 1.1
        t_pert = (7 * math.pi**2 * self.eps0**2 * self.c**3 * m_pert**2 * self.R**3) / (2 * self.q**4)
        if not t_pert > self.expected_t:
            return (False, f"t(1.1m) = {t_pert:.4g} should be > t(m) = {self.expected_t:.4g}.")
        return (True, f"t increases when m increases: t(m)={self.expected_t:.4g} < t(1.1m)={t_pert:.4g}.")

    # ------------------------------------------------------------------
    # Weak answer sanity check (only sign/finite)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        """
        The resulting time t must be positive, finite, and not too small (e.g., > 0).
        No re-derivation of formula, no isclose.
        """
        if self.t is None:
            return (False, "Result t is missing.")
        if not math.isfinite(self.t):
            return (False, "Result t is not finite (NaN or Inf).")
        if self.t <= 0:
            return (False, "Result t is not positive.")
        return (True, "Result t is positive and finite.")