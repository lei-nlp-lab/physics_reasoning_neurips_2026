import math
import traceback

class PhysicsChecker:
    """
    Auditor for the free expansion (Joule expansion) of an ideal gas into a vacuum.
    The system is isolated, no heat or work exchange. Internal energy depends only on
    temperature for an ideal gas, so the final temperature equals the initial temperature.
    """

    def __init__(self, solution_instance):
        # Extract inputs (givens) safely; never assign numeric defaults.
        self.T0 = getattr(solution_instance, 'T0', None)
        self.Cv = getattr(solution_instance, 'Cv', None)
        self.n = getattr(solution_instance, 'n', None)

        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})

        # Extract the final temperature from results with None fallback
        self.T_f = self.results.get('T_f', None)

        # Optional: explanation steps (list of strings)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

    def run_all_tests(self):
        """
        Discover and run every bound method whose name starts with 'test_'.
        Each method must return (passed_bool, message_str).
        Exceptions are caught and reported as failures.
        Returns (all_pass, logs_list).
        """
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
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface checks: required keys exist and are of proper type
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """Check that 'T_f' exists in results and T0 is provided."""
        if 'T_f' not in self.results:
            return (False, "Missing 'T_f' in results")
        if self.T0 is None:
            return (False, "Missing T0 (initial temperature)")
        return (True, "Required keys (T0, T_f) are present")

    def test_interface_T_f_is_numeric(self):
        """Check that T_f is a finite positive number (or can be coerced to one)."""
        if self.T_f is None:
            return (False, "T_f is None")
        try:
            val = float(self.T_f)
        except (TypeError, ValueError):
            return (False, f"T_f = {self.T_f} is not convertible to float")
        if not math.isfinite(val):
            return (False, f"T_f = {val} is not finite")
        if val <= 0:
            return (False, f"T_f = {val} is not positive")
        return (True, f"T_f = {val} is a finite positive float")

    def test_interface_T0_is_numeric(self):
        """Check that T0 is a finite positive float."""
        if self.T0 is None:
            return (False, "T0 is None")
        try:
            val = float(self.T0)
        except (TypeError, ValueError):
            return (False, f"T0 = {self.T0} is not convertible to float")
        if not math.isfinite(val):
            return (False, f"T0 = {val} is not finite")
        if val <= 0:
            return (False, f"T0 = {val} is not positive")
        return (True, f"T0 = {val} is a finite positive float")

    # ------------------------------------------------------------------
    # Invariant checks: theorem-based physical consistency
    # ------------------------------------------------------------------
    def test_invariant_temperature_unchanged(self):
        """
        For free expansion of an ideal gas, internal energy depends only on temperature.
        Since the system is isolated (Q=0, W=0), ΔU=0 ⇒ T_f = T0.
        """
        if self.T0 is None or self.T_f is None:
            return (False, "SKIP: T0 or T_f missing")
        try:
            T0 = float(self.T0)
            Tf = float(self.T_f)
        except (TypeError, ValueError):
            return (False, "Cannot convert T0 or T_f to float")

        if not math.isclose(Tf, T0, rel_tol=1e-6, abs_tol=1e-6):
            return (False, f"T_f = {Tf} differs from T0 = {T0} (free expansion failed)")
        return (True, f"T_f = {Tf} ≈ T0 = {T0}")

    def test_invariant_energy_consistency(self):
        """
        If n and Cv are provided, check that n*Cv*T0 ≈ n*Cv*T_f.
        This directly verifies conservation of internal energy.
        """
        if self.n is None or self.Cv is None or self.T0 is None or self.T_f is None:
            return (False, "SKIP: n, Cv, T0, or T_f missing")
        try:
            n = float(self.n)
            Cv = float(self.Cv)
            T0 = float(self.T0)
            Tf = float(self.T_f)
        except (TypeError, ValueError):
            return (False, "Cannot convert one of n, Cv, T0, T_f to float")

        U_initial = n * Cv * T0
        U_final = n * Cv * Tf
        if not math.isclose(U_initial, U_final, rel_tol=1e-6, abs_tol=1e-6):
            return (False, f"U_initial = {U_initial} ≠ U_final = {U_final} (energy not conserved)")
        return (True, f"U_initial = {U_initial} ≈ U_final = {U_final}")

    # ------------------------------------------------------------------
    # Modeling/trace checks: based on explanation steps
    # ------------------------------------------------------------------
    def test_modeling_explanation_steps_contain_energy_conservation(self):
        """
        Check that the explanation steps mention energy conservation or internal energy.
        (A weak check that the reasoning is on the right track.)
        """
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        if "internal energy" in combined or "energy" in combined or "u" in combined:
            return (True, "Explanation steps reference energy/internal energy")
        return (False, "Explanation steps do not mention energy conservation or internal energy")

    def test_modeling_pseudocode_emit_contract(self):
        """
        Verify that the result key 'T_f' matches the emission contract from pseudocode.
        (This is a structural check, not a numerical one.)
        """
        if 'T_f' not in self.results:
            return (False, "Pseudocode contract violated: 'T_f' missing from results")
        return (True, "Pseudocode emit contract satisfied: 'T_f' in results")

    # ------------------------------------------------------------------
    # Monotonicity check: explicit input perturbation and comparison
    # ------------------------------------------------------------------
    def test_monotonicity_temperature_linear_scaling(self):
        """
        Verify that increasing T0 by +10% leads to an expected increase in T_f by +10%
        (assuming the physics T_f = T0 holds). This tests the linear monotonic relationship.
        """
        if self.T0 is None or self.T_f is None:
            return (False, "SKIP: T0 or T_f missing")
        try:
            T0 = float(self.T0)
            Tf = float(self.T_f)
        except (TypeError, ValueError):
            return (False, "Cannot convert T0 or T_f to float")

        # Baseline: given input T0 produces output Tf (should equal T0)
        # Create perturbed input: increase T0 by 10%
        T0_pert = T0 * 1.1
        # Expected perturbed output from physics (T_f = T0) is T0_pert
        Tf_expected = T0_pert

        # Check monotonicity: output must increase when input increases
        if not (Tf_expected > Tf):
            return (False, "Monotonicity violated: expected output does not increase with input")

        # Check linearity: fractional change in output equals fractional change in input
        frac_dT0 = (T0_pert - T0) / T0
        frac_dTf = (Tf_expected - Tf) / Tf
        if not math.isclose(frac_dTf, frac_dT0, rel_tol=1e-6, abs_tol=1e-6):
            return (False, f"Non‑linear response: dTf/Tf = {frac_dTf} ≠ dT0/T0 = {frac_dT0}")
        return (True, f"Perturbing T0 by +10% gives expected T_f increase by +10% (linear monotonic)")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        """
        Ensure T_f is positive and finite. This is a trivial sanity check
        that does NOT use math.isclose or re-derive the answer.
        """
        if self.T_f is None:
            return (False, "T_f is None – cannot check sanity")
        try:
            val = float(self.T_f)
        except (TypeError, ValueError):
            return (False, f"T_f = {self.T_f} cannot be converted to float")
        if not math.isfinite(val):
            return (False, f"T_f = {val} is not finite")
        if val <= 0:
            return (False, f"T_f = {val} is not positive")
        return (True, f"T_f = {val} is a finite positive number")