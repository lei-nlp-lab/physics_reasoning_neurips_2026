import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics auditor for the capacitor action potential problem.
    Checks:
        - Required result keys exist and are numeric.
        - Capacitance relation Q = C V holds within tolerance (if givens available).
        - Stored energy relation E = 0.5 C V^2 holds within tolerance (if givens available).
        - Signs are positive.
        - Broad magnitude order for final answers (weak sanity).
        - Monotonicity: increasing voltage increases both charge and energy.
    """
    def __init__(self, solution_instance):
        # Extract results and givens safely, using None for missing values.
        self.results = getattr(solution_instance, 'results', {})
        self.Q = self.results.get('Q', None)
        self.E = self.results.get('E', None)

        # Try to obtain V and C from several common places
        self.V = getattr(solution_instance, 'V', None)
        if self.V is None:
            self.V = self.results.get('V', None)
        if self.V is None:
            givens = getattr(solution_instance, 'givens', {})
            self.V = givens.get('V', None)

        self.C = getattr(solution_instance, 'C', None)
        if self.C is None:
            self.C = self.results.get('C', None)
        if self.C is None:
            givens = getattr(solution_instance, 'givens', {})
            self.C = givens.get('C', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_Q_exists(self):
        """Check that result key 'Q' exists and is a number."""
        if self.Q is None:
            return (False, "'Q' is missing or None in results.")
        if not isinstance(self.Q, (int, float)):
            return (False, f"'Q' is not numeric, got {type(self.Q).__name__}.")
        return (True, "'Q' present and numeric.")

    def test_interface_E_exists(self):
        """Check that result key 'E' exists and is a number."""
        if self.E is None:
            return (False, "'E' is missing or None in results.")
        if not isinstance(self.E, (int, float)):
            return (False, f"'E' is not numeric, got {type(self.E).__name__}.")
        return (True, "'E' present and numeric.")

    # ------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_capacitance_relation(self):
        """
        Verify that Q ≈ C·V when V and C are available.
        Uses math.isclose with tolerance 1e-2.
        """
        if self.V is None or self.C is None:
            return (False, "SKIP: V or C not available, cannot verify capacitance relation.")
        if self.Q is None:
            return (False, "SKIP: Q not available.")
        expected_Q = self.C * self.V
        if math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q = {self.Q} ~ C·V = {expected_Q}")
        else:
            return (False, f"Q = {self.Q} is not close to C·V = {expected_Q}.")

    def test_invariant_energy_relation(self):
        """
        Verify that E ≈ 0.5·C·V² when V and C are available.
        Uses math.isclose with tolerance 1e-2.
        """
        if self.V is None or self.C is None:
            return (False, "SKIP: V or C not available, cannot verify energy relation.")
        if self.E is None:
            return (False, "SKIP: E not available.")
        expected_E = 0.5 * self.C * self.V**2
        if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E = {self.E} ~ 0.5·C·V² = {expected_E}")
        else:
            return (False, f"E = {self.E} is not close to 0.5·C·V² = {expected_E}.")

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks (positive, finite)
    # ------------------------------------------------------------------
    def test_modeling_charge_positive(self):
        """Charge should be positive (as stated in constraints)."""
        if self.Q is None:
            return (False, "SKIP: Q not available.")
        if self.Q > 0:
            return (True, f"Q = {self.Q} > 0.")
        else:
            return (False, f"Q = {self.Q} is not positive.")

    def test_modeling_energy_positive(self):
        """Energy should be positive."""
        if self.E is None:
            return (False, "SKIP: E not available.")
        if self.E > 0:
            return (True, f"E = {self.E} > 0.")
        else:
            return (False, f"E = {self.E} is not positive.")

    def test_modeling_finite_values(self):
        """Both Q and E should be finite (not inf or NaN)."""
        if self.Q is None or self.E is None:
            return (False, "SKIP: Q or E not available.")
        if not (math.isfinite(self.Q) and math.isfinite(self.E)):
            return (False, "Q or E is infinite or NaN.")
        return (True, "Both Q and E are finite numbers.")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturbing an input and comparing output)
    # ------------------------------------------------------------------
    def test_monotonicity_voltage_increase(self):
        """
        Verify that increasing voltage (by 1%) increases charge and energy.
        Uses only given V and C to compute baseline and perturbed values.
        """
        if self.V is None or self.C is None:
            return (False, "SKIP: V or C not available for monotonicity test.")
        V0 = self.V
        C = self.C
        V1 = V0 * 1.01
        Q0 = C * V0
        Q1 = C * V1
        E0 = 0.5 * C * V0**2
        E1 = 0.5 * C * V1**2

        # Q should increase by 1%, E by (1.01)^2 = 1.0201
        q_ratio_ok = math.isclose(Q1 / Q0, 1.01, rel_tol=1e-6)
        e_ratio_ok = math.isclose(E1 / E0, 1.0201, rel_tol=1e-6)
        if not (q_ratio_ok and e_ratio_ok):
            return (False, f"Monotonicity failed: Q ratio {Q1/Q0} (expected 1.01), "
                           f"E ratio {E1/E0} (expected 1.0201).")
        return (True, "Voltage increase by 1% gives expected linear (Q) and quadratic (E) increases.")

    # ------------------------------------------------------------------
    # Weak answer sanity (broad magnitude range only)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_order(self):
        """
        Very weak sanity: check that Q lies roughly between 1e-12 and 1e-9,
        and E between 1e-13 and 1e-10.
        These ranges are deliberately wide to avoid over‑constraining.
        """
        if self.Q is None or self.E is None:
            return (False, "SKIP: Q or E not available.")
        q_ok = 1e-12 <= self.Q <= 1e-9
        e_ok = 1e-13 <= self.E <= 1e-10
        if q_ok and e_ok:
            return (True, f"Q = {self.Q} and E = {self.E} are within broad expected ranges.")
        else:
            msg = ""
            if not q_ok:
                msg += f"Q = {self.Q} not in [1e-12, 1e-9]. "
            if not e_ok:
                msg += f"E = {self.E} not in [1e-13, 1e-10]. "
            return (False, msg.strip())

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all methods starting with test_
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