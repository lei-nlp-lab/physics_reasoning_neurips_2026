import math
import traceback

class PhysicsChecker:
    """
    Physics Checker for energy in a capacitor.
    Expected result: results['U3'] = 0.5 * C3 * V3^2.
    Givens (optionally available): E, C1, C2, C3 (series circuit).
    """
    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Required output
        self.U3 = self.results.get('U3', None)

        # Optional givens (could be attributes of solution_instance or in results)
        self.E = getattr(solution_instance, 'E', self.results.get('E', None))
        self.C1 = getattr(solution_instance, 'C1', self.results.get('C1', None))
        self.C2 = getattr(solution_instance, 'C2', self.results.get('C2', None))
        self.C3 = getattr(solution_instance, 'C3', self.results.get('C3', None))

        # Optional intermediate results (not required but useful for checks)
        self.V3 = self.results.get('V3', None)
        self.Q = self.results.get('Q', None)
        self.Ceq = self.results.get('Ceq', None)

        # All capacitances in Farads (input likely in µF, we keep raw; formulas dimensionless)
        # We will not convert units – assume consistent.

    def _all_series_givens(self):
        """Check if all required series givens are present."""
        return (self.E is not None and self.C1 is not None
                and self.C2 is not None and self.C3 is not None)

    def _compute_expected_U3(self):
        """Compute U3 from series circuit formulas using givens."""
        # Ceq = 1/(1/C1 + 1/C2 + 1/C3)
        inv = 1.0/self.C1 + 1.0/self.C2 + 1.0/self.C3
        Ceq = 1.0 / inv
        Q = Ceq * self.E
        V3 = Q / self.C3
        U3 = 0.5 * self.C3 * V3 * V3
        return U3

    # ========== Interface Tests ==========
    def test_interface_U3_exists(self):
        """Check that results['U3'] exists."""
        if self.U3 is None:
            return (False, "U3 key missing in results")
        return (True, "U3 present")

    def test_interface_U3_is_positive(self):
        """Check that U3 is positive (energy store)."""
        if self.U3 is None:
            return (False, "SKIP: U3 not available")
        if not isinstance(self.U3, (int, float)):
            return (False, "U3 is not numeric")
        if self.U3 <= 0:
            return (False, f"U3 = {self.U3} is not positive")
        return (True, "U3 positive")

    # ========== Invariant Tests (theorem-based) ==========
    def test_invariant_energy_formula(self):
        """If V3 and C3 are known, verify U3 = 0.5*C3*V3^2."""
        if self.V3 is None or self.C3 is None:
            return (False, "SKIP: V3 or C3 not available")
        expected = 0.5 * self.C3 * self.V3 * self.V3
        if not math.isclose(self.U3, expected, rel_tol=1e-2, abs_tol=1e-12):
            return (False, f"U3 mismatch: got {self.U3}, expected {expected}")
        return (True, "Energy formula holds")

    def test_invariant_series_charge(self):
        """If Q and C3 are known, check V3 = Q/C3."""
        if self.Q is None or self.C3 is None or self.V3 is None:
            return (False, "SKIP: Q, C3 or V3 not available")
        expected_V3 = self.Q / self.C3
        if not math.isclose(self.V3, expected_V3, rel_tol=1e-2):
            return (False, f"V3 mismatch: got {self.V3}, expected {expected_V3}")
        return (True, "Q = C * V holds")

    # ========== Modeling / Trace Tests ==========
    def test_modeling_series_Ceq(self):
        """If Ceq and all capacitances are known, check series formula."""
        if self.Ceq is None or not self._all_series_givens():
            return (False, "SKIP: Ceq or givens missing")
        inv = 1.0/self.C1 + 1.0/self.C2 + 1.0/self.C3
        expected_Ceq = 1.0 / inv
        if not math.isclose(self.Ceq, expected_Ceq, rel_tol=1e-2):
            return (False, f"Ceq mismatch: got {self.Ceq}, expected {expected_Ceq}")
        return (True, "Series Ceq formula holds")

    def test_modeling_voltage_division(self):
        """If Q, Ceq, C3 are known, verify V3 = (Ceq/C3)*E (if E known)."""
        # This is a combination check; requires E, Ceq, C3 (or Q)
        if self.E is None or self.C3 is None or self.Ceq is None:
            return (False, "SKIP: E, C3, Ceq required")
        expected_V3 = (self.Ceq / self.C3) * self.E
        if self.V3 is not None:
            if not math.isclose(self.V3, expected_V3, rel_tol=1e-2):
                return (False, f"V3 from voltage division mismatch: got {self.V3}, expected {expected_V3}")
        else:
            # V3 not provided, but we can still check consistency via U3 if known
            if self.U3 is not None:
                derived_V3 = math.sqrt(2.0 * self.U3 / self.C3)
                if not math.isclose(derived_V3, expected_V3, rel_tol=1e-2):
                    return (False, f"Derived V3 from energy mismatch voltage division: {derived_V3} vs {expected_V3}")
            else:
                return (False, "SKIP: neither V3 nor U3 to compare")
        return (True, "Voltage division consistent")

    # ========== Monotonicity Tests ==========
    def test_monotonicity_U3_increases_with_E(self):
        """If all givens present, verify that increasing E increases U3."""
        if not self._all_series_givens():
            return (False, "SKIP: givens missing")
        baseline_U3 = self._compute_expected_U3()
        # Perturb E by +10%
        E_new = self.E * 1.1
        # Compute new U3 using same formulas
        inv = 1.0/self.C1 + 1.0/self.C2 + 1.0/self.C3
        Ceq = 1.0 / inv
        Q_new = Ceq * E_new
        V3_new = Q_new / self.C3
        new_U3 = 0.5 * self.C3 * V3_new * V3_new
        if new_U3 <= baseline_U3:
            return (False, f"Monotonicity failed: baseline U3={baseline_U3}, new U3 with +10% E={new_U3}")
        return (True, "U3 increases with E")

    def test_monotonicity_U3_decreases_with_C3(self):
        """If all givens present, verify that increasing C3 decreases U3."""
        if not self._all_series_givens():
            return (False, "SKIP: givens missing")
        baseline_U3 = self._compute_expected_U3()
        # Perturb C3 by +10%
        C3_new = self.C3 * 1.1
        inv_new = 1.0/self.C1 + 1.0/self.C2 + 1.0/C3_new
        Ceq_new = 1.0 / inv_new
        Q_new = Ceq_new * self.E
        V3_new = Q_new / C3_new
        new_U3 = 0.5 * C3_new * V3_new * V3_new
        if new_U3 >= baseline_U3:
            return (False, f"Monotonicity failed: baseline U3={baseline_U3}, new U3 with +10% C3={new_U3}")
        return (True, "U3 decreases with increasing C3")

    # ========== Answer Sanity (weak) ==========
    def test_answer_sanity_U3_reasonable(self):
        """Broad magnitude check: energy should be between 1e-12 J and 1 J."""
        if self.U3 is None:
            return (False, "SKIP: U3 not available")
        if not 1e-12 <= self.U3 <= 1.0:
            return (False, f"U3 = {self.U3} outside reasonable range [1e-12, 1] J")
        return (True, f"U3 = {self.U3} is within reasonable bounds")

    # ========== Runner ==========
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