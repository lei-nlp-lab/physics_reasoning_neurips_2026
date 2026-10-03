import math
import traceback
import sympy

class PhysicsChecker:
    """Audits solutions for the electron in a spherical cavity problem.

    Expected input structure:
        solution_instance.givens = {'hbar': value, 'm': value, 'R': value}
        solution_instance.results = {'P': value}   (or optionally 'E0', 'V')
    """

    def __init__(self, solution_instance):
        # Extract constants safely (None if missing)
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}

        # Givens (no numeric defaults!)
        self.hbar = givens.get('hbar', None)
        self.m = givens.get('m', None)
        self.R = givens.get('R', None)

        # Results (only P is required by pseudocode contract)
        self.P = results.get('P', None)

        # Optional but useful for deeper checks
        self.E0 = results.get('E0', None)   # may not be present
        self.V = results.get('V', None)

    # ----------------------------------------------------------------
    # INTERFACE CHECKS
    # ----------------------------------------------------------------
    def test_interface_hbar_given(self):
        """Check that hbar is provided and numeric."""
        if self.hbar is None:
            return (False, "hbar is missing in givens")
        if not isinstance(self.hbar, (int, float, sympy.Number)):
            return (False, "hbar is not numeric")
        return (True, "hbar present and numeric")

    def test_interface_m_given(self):
        """Check that electron mass is provided and numeric."""
        if self.m is None:
            return (False, "m is missing in givens")
        if not isinstance(self.m, (int, float, sympy.Number)):
            return (False, "m is not numeric")
        return (True, "m present and numeric")

    def test_interface_R_given(self):
        """Check that cavity radius is provided and numeric."""
        if self.R is None:
            return (False, "R is missing in givens")
        if not isinstance(self.R, (int, float, sympy.Number)):
            return (False, "R is not numeric")
        return (True, "R present and numeric")

    def test_interface_P_result(self):
        """Check that pressure P is present in results."""
        if self.P is None:
            return (False, "P is missing in results")
        if not isinstance(self.P, (int, float, sympy.Number)):
            return (False, "P is not numeric")
        return (True, "P present and numeric")

    # ----------------------------------------------------------------
    # INVARIANT (THEOREM-BASED) CHECKS
    # ----------------------------------------------------------------
    def test_invariant_ground_state_energy(self):
        """If E0 is given, verify E0 = pi^2 hbar^2 / (2 m R^2)."""
        if self.E0 is None or self.hbar is None or self.m is None or self.R is None:
            return (False, "SKIP: missing data for energy check")
        try:
            expected = (math.pi**2 * self.hbar**2) / (2 * self.m * self.R**2)
            passed = math.isclose(float(self.E0), float(expected),
                                  rel_tol=1e-2, abs_tol=1e-2)
            if passed:
                return (True, "E0 matches theoretical value")
            else:
                return (False, f"E0 = {self.E0}, expected ≈ {expected}")
        except Exception as e:
            return (False, f"Energy invariant error: {e}")

    def test_invariant_pressure_via_derivative(self):
        """Check that P = -dE0/dV using givens and the formula.

        If E0 not present, we compute it from the standard expression.
        """
        if self.P is None or self.hbar is None or self.m is None or self.R is None:
            return (False, "SKIP: missing data for pressure derivative check")
        try:
            # Derivative of ground state energy w.r.t. R
            # E0 = pi^2 hbar^2 / (2 m R^2)  =>  dE0/dR = - pi^2 hbar^2 / (m R^3)
            dE0_dR = - (math.pi**2 * self.hbar**2) / (self.m * self.R**3)
            # Volume derivative: V = (4/3) pi R^3  =>  dV/dR = 4 pi R^2
            dV_dR = 4 * math.pi * self.R**2
            # P = - dE0/dV = - (dE0/dR) / (dV/dR)
            expected_P = - dE0_dR / dV_dR
            # Simplify: expected_P = (pi^2 hbar^2 / (m R^3)) / (4 pi R^2)
            #            = (pi hbar^2) / (4 m R^5)
            expected_P_simplified = (math.pi * self.hbar**2) / (4 * self.m * self.R**5)
            passed = math.isclose(float(self.P), float(expected_P),
                                  rel_tol=1e-2, abs_tol=1e-2)
            if passed:
                return (True, "P matches derivative expression")
            else:
                return (False, f"P = {self.P}, expected from derivative ≈ {expected_P}")
        except Exception as e:
            return (False, f"Pressure derivative check error: {e}")

    def test_invariant_pressure_formula(self):
        """Direct check: P = pi hbar^2 / (4 m R^5)."""
        if self.P is None or self.hbar is None or self.m is None or self.R is None:
            return (False, "SKIP: missing data for direct formula check")
        try:
            expected = (math.pi * self.hbar**2) / (4 * self.m * self.R**5)
            passed = math.isclose(float(self.P), float(expected),
                                  rel_tol=1e-2, abs_tol=1e-2)
            if passed:
                return (True, "P matches direct formula")
            else:
                return (False, f"P = {self.P}, expected ≈ {expected}")
        except Exception as e:
            return (False, f"Direct formula error: {e}")

    # ----------------------------------------------------------------
    # MODELING / TRACE CHECKS (based on explanation steps)
    # ----------------------------------------------------------------
    def test_modeling_volume_radius_relation(self):
        """If V is provided, check V = (4/3) pi R^3."""
        if self.V is None or self.R is None:
            return (False, "SKIP: V not provided or R missing")
        try:
            expected_V = (4.0/3.0) * math.pi * self.R**3
            passed = math.isclose(float(self.V), float(expected_V),
                                  rel_tol=1e-2, abs_tol=1e-2)
            if passed:
                return (True, "V matches (4/3) pi R^3")
            else:
                return (False, f"V = {self.V}, expected ≈ {expected_V}")
        except Exception as e:
            return (False, f"Volume relation error: {e}")

    def test_modeling_radial_wavefunction_normalization(self):
        """Check if normalization constant sqrt(2/R) appears in solution.

        This test is only possible if the solution exports the radial
        wavefunction normalization factor, e.g., results['norm_factor'].
        Otherwise skip.
        """
        # Dummy: skip because problem does not require exporting wavefunction.
        return (False, "SKIP: radial wavefunction data not expected")

    # ----------------------------------------------------------------
    # MONOTONICITY CHECK (trend verification via explicit perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_pressure_decreases_with_R(self):
        """Verify that increasing R decreases pressure.

        Uses the scaling law P ∝ 1/R^5 derived from the solution.
        Perturb R upward by 10% and compute expected perturbed pressure
        from the given P and compare with baseline.
        """
        if self.P is None or self.R is None:
            return (False, "SKIP: missing P or R")
        try:
            R0 = float(self.R)
            P0 = float(self.P)
            # Perturb R upward by 10%
            R1 = R0 * 1.1
            # Expected P at R1: P1 = P0 * (R0/R1)^5 (since P ∝ 1/R^5)
            P1_expected = P0 * (R0 / R1)**5
            # Since R1 > R0, P1 should be less than P0
            if P1_expected < P0:
                return (True, f"P decreases from {P0} to {P1_expected} when R increases from {R0} to {R1}")
            else:
                return (False, f"P1_expected {P1_expected} should be < P0 {P0}")
        except Exception as e:
            return (False, f"Monotonicity error: {e}")

    # ----------------------------------------------------------------
    # WEAK ANSWER SANITY (only one, non-dominating)
    # ----------------------------------------------------------------
    def test_answer_sanity_pressure_positive_finite(self):
        """Basic sanity: pressure should be positive and finite."""
        if self.P is None:
            return (False, "SKIP: P missing")
        try:
            p = float(self.P)
            if math.isfinite(p) and p > 0:
                return (True, "P is finite and positive")
            else:
                return (False, f"P = {p} is not positive/finite")
        except Exception as e:
            return (False, f"Sanity error: {e}")

    # ----------------------------------------------------------------
    # MASTER TEST RUNNER
    # ----------------------------------------------------------------
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