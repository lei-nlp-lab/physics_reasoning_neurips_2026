import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits the entropy change solution for a thermally conducting bar.
    Extracts givens and results from solution_instance, then runs small
    focused tests for interface, invariants, modeling, monotonicity, and answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract givens; default None for missing
        self.L = getattr(solution_instance, 'L', None)
        self.A = getattr(solution_instance, 'A', None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.cp = getattr(solution_instance, 'cp', None)
        self.T_H = getattr(solution_instance, 'T_H', None)
        self.T_C = getattr(solution_instance, 'T_C', None)

        # Possibly givens live in a dict
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.L = self.L if self.L is not None else givens.get('L', None)
        self.A = self.A if self.A is not None else givens.get('A', None)
        self.rho = self.rho if self.rho is not None else givens.get('rho', None)
        self.cp = self.cp if self.cp is not None else givens.get('cp', None)
        self.T_H = self.T_H if self.T_H is not None else givens.get('T_H', None)
        self.T_C = self.T_C if self.T_C is not None else givens.get('T_C', None)

        # Result: Delta_S
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.Delta_S = results.get('Delta_S', None)

        # Compute derived quantities that tests will need
        self.Cp = None
        self.T_f = None
        if all(v is not None for v in (self.cp, self.rho, self.A, self.L,
                                        self.T_H, self.T_C)):
            self.Cp = self.cp * self.rho * self.A * self.L
            self.T_f = (self.T_H + self.T_C) / 2.0

    # ----------------------------------------------------------------------
    # Helper: compute ΔS from givens (same formula as in the problem)
    # ----------------------------------------------------------------------
    def _compute_Delta_S_from_givens(self, T_H, T_C, Cp):
        """
        Return ΔS = Cp * (1 + ln T_f + (T_C/(T_H - T_C))*ln T_C
                          - (T_H/(T_H - T_C))*ln T_H)
        where T_f = (T_H + T_C)/2.
        Assumes T_H > T_C > 0 and T_H != T_C.
        """
        if T_H == T_C:
            return 0.0
        T_f = (T_H + T_C) / 2.0
        return Cp * (1.0 + math.log(T_f) +
                     (T_C / (T_H - T_C)) * math.log(T_C) -
                     (T_H / (T_H - T_C)) * math.log(T_H))

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that Delta_S result is present and givens are available."""
        if self.Delta_S is None:
            return (False, "Missing Delta_S in results")
        missing = []
        for name in ['L', 'A', 'rho', 'cp', 'T_H', 'T_C']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Check that numerical values are floats (or ints/float-like)."""
        vals = [self.L, self.A, self.rho, self.cp, self.T_H, self.T_C, self.Delta_S]
        for v in vals:
            if v is not None and not isinstance(v, (int, float)):
                return (False, f"Expected numeric type, got {type(v).__name__}: {v}")
        return (True, "All values are numeric")

    # ----------------------------------------------------------------------
    # Invariant / theorem‑based tests (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_positivity(self):
        """Delta_S must be positive when T_H > T_C > 0."""
        if self.Delta_S is None or self.T_H is None or self.T_C is None:
            return (False, "SKIP: missing data for positivity check")
        if self.T_H <= 0 or self.T_C <= 0:
            return (False, "Temperatures must be positive")
        if self.T_H <= self.T_C:
            return (False, "Test requires T_H > T_C to ensure positive entropy change")
        if self.Delta_S <= 0:
            return (False, f"Delta_S = {self.Delta_S} <= 0, but expected positive")
        return (True, f"Delta_S = {self.Delta_S} > 0")

    def test_invariant_limit_equality(self):
        """If T_H == T_C, Delta_S should be zero (within tolerance)."""
        if self.Delta_S is None or self.T_H is None or self.T_C is None:
            return (False, "SKIP: missing data for limit check")
        if self.T_H != self.T_C:
            return (True, "Not applicable (T_H != T_C), skipping limit test")
        if math.isclose(self.Delta_S, 0.0, rel_tol=1e-6, abs_tol=1e-6):
            return (True, "Delta_S is zero when T_H = T_C")
        else:
            return (False, f"Delta_S = {self.Delta_S}, expected ~0 when T_H = T_C")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency tests (use math.isclose)
    # ----------------------------------------------------------------------
    def test_modeling_Cp_consistency(self):
        """Check that Cp = cp * rho * A * L matches the given values."""
        if any(v is None for v in [self.cp, self.rho, self.A, self.L, self.Cp]):
            return (False, "SKIP: missing givens for Cp check")
        expected_Cp = self.cp * self.rho * self.A * self.L
        if math.isclose(self.Cp, expected_Cp, rel_tol=1e-9):
            return (True, f"Cp = {self.Cp} matches expected {expected_Cp}")
        else:
            return (False, f"Cp = {self.Cp}, expected {expected_Cp}")

    def test_modeling_entropy_formula(self):
        """
        Re‑derive Delta_S from the given givens using the known formula and
        compare to the result's Delta_S using a fuzzy tolerance.
        """
        if any(v is None for v in [self.Delta_S, self.Cp, self.T_H, self.T_C, self.T_f]):
            return (False, "SKIP: missing data for formula test")
        if math.isclose(self.T_H, self.T_C, rel_tol=1e-9):
            # limit already tested; formula would involve division by zero
            return (True, "SKIP: T_H ≈ T_C, formula not applicable")
        try:
            expected = self._compute_Delta_S_from_givens(self.T_H, self.T_C, self.Cp)
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Could not compute expected Delta_S: {e}")
        if math.isclose(self.Delta_S, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Delta_S = {self.Delta_S} ≈ expected {expected}")
        else:
            return (False, f"Delta_S = {self.Delta_S} differs from expected {expected}")

    # ----------------------------------------------------------------------
    # Monotonicity test (perturb an input and compare perturbed output to baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_T_H_increase(self):
        """
        Perturb T_H upward by 5% and verify that the re‑derived ΔS increases.
        The entropy change should be monotonic increasing in T_H.
        """
        if any(v is None for v in [self.T_H, self.T_C, self.Cp]):
            return (False, "SKIP: missing givens for monotonicity test")
        if math.isclose(self.T_H, self.T_C, rel_tol=1e-9):
            # If temperatures equal, ΔS = 0 and perturbation may not produce monotonicity
            return (True, "SKIP: T_H ≈ T_C, monotonicity test not meaningful")
        try:
            baseline = self._compute_Delta_S_from_givens(self.T_H, self.T_C, self.Cp)
            T_H_pert = self.T_H * 1.05
            perturbed = self._compute_Delta_S_from_givens(T_H_pert, self.T_C, self.Cp)
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Error computing Delta_S: {e}")
        # Check that perturbed > baseline (with a tiny tolerance for floating errors)
        if perturbed > baseline - 1e-12:
            return (True, f"Perturbed ΔS = {perturbed} > baseline ΔS = {baseline} (T_H increased by 5%)")
        else:
            return (False, f"Perturbed ΔS = {perturbed} not > baseline ΔS = {baseline}")

    def test_monotonicity_T_C_decrease(self):
        """
        Perturb T_C downward by 5% and verify that the re‑derived ΔS increases.
        Lowering cold temperature should increase the temperature gradient and thus ΔS.
        """
        if any(v is None for v in [self.T_H, self.T_C, self.Cp]):
            return (False, "SKIP: missing givens for monotonicity test")
        if math.isclose(self.T_H, self.T_C, rel_tol=1e-9):
            return (True, "SKIP: T_H ≈ T_C, monotonicity test not meaningful")
        try:
            baseline = self._compute_Delta_S_from_givens(self.T_H, self.T_C, self.Cp)
            T_C_pert = self.T_C * 0.95
            # Ensure T_C_pert > 0
            if T_C_pert <= 0:
                return (False, "Perturbed T_C would be <= 0, test invalid")
            perturbed = self._compute_Delta_S_from_givens(self.T_H, T_C_pert, self.Cp)
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Error computing Delta_S: {e}")
        if perturbed > baseline - 1e-12:
            return (True, f"Perturbed ΔS = {perturbed} > baseline ΔS = {baseline} (T_C decreased by 5%)")
        else:
            return (False, f"Perturbed ΔS = {perturbed} not > baseline ΔS = {baseline}")

    # ----------------------------------------------------------------------
    # Answer‑sanity test (weak, no isclose, no re‑derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_range(self):
        """Check that Delta_S is finite, not NaN, and within a plausible range."""
        if self.Delta_S is None:
            return (False, "SKIP: Delta_S missing")
        if not math.isfinite(self.Delta_S):
            return (False, f"Delta_S is not finite: {self.Delta_S}")
        if self.Cp is not None:
            if abs(self.Delta_S) > 1e6 * abs(self.Cp):
                return (False, f"Delta_S magnitude {self.Delta_S} too large relative to Cp={self.Cp}")
        return (True, f"Delta_S = {self.Delta_S} is finite and plausible")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
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