import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Helium boiling temperature problem.
    Checks definitions, Clausius-Clapeyron relationship, and monotonicity.
    """
    R = 8.314  # J/(mol·K)
    TROUTON_K = 85.0  # J/(mol·K) from problem context

    def __init__(self, solution_instance):
        # Extract inputs
        self.P1 = getattr(solution_instance, 'P1', None)
        self.T1 = getattr(solution_instance, 'T1', None)
        self.P2 = getattr(solution_instance, 'P2', None)
        # Extract results
        results = getattr(solution_instance, 'results', {}) or {}
        self.def_complete = results.get('def_complete', None)
        self.L = results.get('L', None)
        self.T2 = results.get('T2', None)
        # Explanation steps (if present)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_def_complete(self):
        """def_complete must be a boolean and True."""
        if self.def_complete is None:
            return (False, "def_complete is missing")
        if not isinstance(self.def_complete, bool):
            return (False, f"def_complete should be bool, got {type(self.def_complete)}")
        if self.def_complete is not True:
            return (False, "def_complete must be True")
        return (True, "def_complete is present and True")

    def test_interface_L(self):
        """L must be positive finite number."""
        if self.L is None:
            return (False, "L is missing")
        try:
            L_val = float(self.L)
        except (TypeError, ValueError):
            return (False, f"L cannot be converted to float: {self.L}")
        if not math.isfinite(L_val) or L_val <= 0:
            return (False, f"L must be positive finite, got {L_val}")
        return (True, f"L = {L_val} J/mol")

    def test_interface_T2(self):
        """T2 must be positive finite number."""
        if self.T2 is None:
            return (False, "T2 is missing")
        try:
            T2_val = float(self.T2)
        except (TypeError, ValueError):
            return (False, f"T2 cannot be converted to float: {self.T2}")
        if not math.isfinite(T2_val) or T2_val <= 0:
            return (False, f"T2 must be positive finite, got {T2_val}")
        return (True, f"T2 = {T2_val} K")

    # ------------------------------------------------------------------
    # Invariant/modeling checks (using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_definitions_consistency(self):
        """
        Check that the definitions (critical point & triple point) are
        consistent with expectation.  We only verify that def_complete
        is True (already done), and optionally that explanation steps
        mention the correct criteria.
        """
        if self.def_complete is not True:
            return (False, "def_complete indicates definitions were not properly emitted")
        # If explanation steps are available, quickly check they contain
        # key phrases.  This is a weak sanity, not a hard constraint.
        if self.explanation_steps is not None and isinstance(self.explanation_steps, str):
            text = self.explanation_steps.lower()
            if 'critical' not in text or 'triple' not in text:
                return (False, "explanation_steps missing references to critical or triple point")
        return (True, "Definition consistency verified")

    def test_modeling_trouton_rule(self):
        """
        Check that L = k * T1 (Trouton's rule) is satisfied.
        Requires T1 to be available.
        """
        if self.T1 is None:
            return (False, "SKIP: T1 not available")
        if self.L is None:
            return (False, "SKIP: L not available")
        L_expected = PhysicsChecker.TROUTON_K * self.T1
        if not math.isclose(self.L, L_expected, rel_tol=1e-2):
            return (False, f"L = {self.L} but expected {L_expected:.1f} (k*T1)")
        return (True, f"L = {self.L} J/mol ≈ k*T1 = {L_expected:.1f}")

    def test_modeling_clausius_clapeyron(self):
        """
        Verify the integrated Clausius-Clapeyron equation:
        ln(P2/P1) = - (L/R) * (1/T2 - 1/T1)
        """
        missing = []
        if self.P1 is None: missing.append("P1")
        if self.P2 is None: missing.append("P2")
        if self.T1 is None: missing.append("T1")
        if self.T2 is None: missing.append("T2")
        if self.L is None: missing.append("L")
        if missing:
            return (False, f"SKIP: missing {', '.join(missing)}")

        try:
            P1 = float(self.P1)
            P2 = float(self.P2)
            T1 = float(self.T1)
            T2 = float(self.T2)
            L = float(self.L)
        except (TypeError, ValueError) as e:
            return (False, f"Cannot convert inputs to float: {e}")

        R = PhysicsChecker.R
        # Compute left and right sides
        lhs = math.log(P2 / P1) if P1 > 0 and P2 > 0 else None
        if lhs is None:
            return (False, "P1 and P2 must be positive")
        rhs = - (L / R) * (1.0 / T2 - 1.0 / T1)

        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Clausius-Clapeyron mismatch: ln(P2/P1)={lhs:.4f}, "
                           f"rhs={rhs:.4f}")
        return (True, f"Clausius-Clapeyron satisfied: ln({P2}/{P1})={lhs:.4f}")

    def test_invariant_T2_less_than_T1(self):
        """At lower pressure, boiling temperature must be lower."""
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: T1 or T2 not available")
        if self.T2 >= self.T1:
            return (False, f"T2={self.T2} >= T1={self.T1}, but lower pressure should give lower boiling point")
        return (True, f"T2={self.T2} < T1={self.T1} (consistent with pressure reduction)")

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_pressure_drop_decreases_temperature(self):
        """
        If we perturb P2 upward (multiply by 1.1), the boiling temperature
        should increase.  We recompute T2' using the integrated CC equation
        and the same L, T1, P1.
        """
        if any(x is None for x in [self.P1, self.P2, self.T1, self.L]):
            return (False, "SKIP: missing one of P1, P2, T1, L")
        try:
            P1 = float(self.P1)
            P2 = float(self.P2)
            T1 = float(self.T1)
            L = float(self.L)
        except (TypeError, ValueError) as e:
            return (False, f"Cannot convert inputs: {e}")

        # Perturb P2 upward by +10%
        P2_prime = P2 * 1.1
        R = PhysicsChecker.R
        # Compute T2' from CC:
        # 1/T2' = 1/T1 + (R/L)*ln(P1/P2')
        # ln(P1/P2') positive because P1 > P2'
        ratio = math.log(P1 / P2_prime)
        inv_T2_prime = 1.0 / T1 + (R / L) * ratio
        T2_prime = 1.0 / inv_T2_prime
        # T2_prime should be higher than T2 (if T2 exists)
        if self.T2 is not None:
            try:
                T2_actual = float(self.T2)
            except:
                T2_actual = None
            if T2_actual is not None and not (T2_prime > T2_actual):
                return (False, f"Monotonicity violation: perturbed P2={P2_prime:.3f} gives "
                               f"T2'={T2_prime:.4f}, but actual T2={T2_actual:.4f} (expected T2' > T2)")
        return (True, f"Monotonicity: higher P2 gives higher T2'={T2_prime:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_T2(self):
        """Quick sanity: T2 should be a small positive number (< 10 K)."""
        if self.T2 is None:
            return (False, "SKIP: T2 not available")
        try:
            T2_val = float(self.T2)
        except (TypeError, ValueError):
            return (False, "T2 not a number")
        if not math.isfinite(T2_val):
            return (False, f"T2 is not finite: {T2_val}")
        if T2_val <= 0:
            return (False, f"T2 is not positive: {T2_val}")
        if T2_val > 10.0:
            return (False, f"T2 too large for helium near 1 mm Hg: {T2_val} K")
        return (True, f"T2 = {T2_val} K passes weak sanity")

    # ------------------------------------------------------------------
    # Runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)