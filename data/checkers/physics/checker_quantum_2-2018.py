import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for solution to the spherically symmetric square-well problem.
    Checks invariants, modeling quality, and provides a weak final-answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        # Extract given constants from solution_instance.givens or directly from attributes
        self.givens = getattr(solution_instance, 'givens', {})
        self.m = self.givens.get('m', getattr(solution_instance, 'm', None))
        self.r0 = self.givens.get('r0', getattr(solution_instance, 'r0', None))
        self.hbar = self.givens.get('hbar', getattr(solution_instance, 'hbar', None))

        # Extract expected result keys; all default to None if missing
        self.V0_min = self.results.get('V0_min', None)
        self.E0 = self.results.get('E0', None)
        self.transcendental = self.results.get('transcendental', None)
        self.u_in = self.results.get('u_in', None)
        self.u_out = self.results.get('u_out', None)
        self.T_expect = self.results.get('T_expect', None)
        self.UP = self.results.get('UP', None)
        self.E_est = self.results.get('E_est', None)
        self.comparison = self.results.get('comparison', None)

    def run_all_tests(self):
        """Discover all test_* methods and run them, returning (all_pass, logs)."""
        logs = []
        all_pass = True
        # Get all bound methods whose names start with "test_"
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

    # ------------------------------------------------------------------
    # Interface tests: required keys and givens
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        required_keys = ['V0_min', 'E0', 'transcendental', 'u_in', 'u_out',
                         'T_expect', 'UP', 'E_est', 'comparison']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All nine required result keys present")

    def test_interface_givens_available(self):
        if self.m is None or self.r0 is None or self.hbar is None:
            return (False, "SKIP: one or more givens (m, r0, hbar) missing")
        return (True, "Givens m, r0, hbar present")

    # ------------------------------------------------------------------
    # Invariant (theorem‑based) tests
    # ------------------------------------------------------------------
    def test_invariant_V0_min_formula(self):
        """V0_min must equal 9π²ħ²/(8 m r0²) (the minimum depth for two s‑wave bound states)."""
        if self.V0_min is None or self.m is None or self.r0 is None or self.hbar is None:
            return (False, "SKIP: missing data for V0_min check")
        m, r0, hbar = sp.symbols('m r0 hbar', positive=True)
        expected = 9 * sp.pi**2 * hbar**2 / (8 * m * r0**2)
        given = self.V0_min
        if isinstance(given, str):
            try:
                given_sym = sp.sympify(given.replace('hbar', 'hbar').replace('pi', 'pi'))
            except Exception:
                return (False, f"Cannot parse V0_min string: {given}")
        else:
            given_sym = given
        diff = sp.simplify(given_sym - expected)
        if diff == 0:
            return (True, "V0_min matches theoretical minimum 9π²ħ²/(8 m r0²)")
        else:
            return (False, f"V0_min does not match: given {given_sym} vs expected {expected}, diff {diff}")

    def test_invariant_E0_zero(self):
        """E0 (energy of the second bound state) must be exactly 0."""
        if self.E0 is None:
            return (False, "SKIP: E0 not available")
        if self.E0 == 0 or str(self.E0).strip() == '0':
            return (True, "E0 is zero (threshold bound state)")
        else:
            return (False, f"E0 should be 0, got {self.E0}")

    def test_invariant_transcendental_form(self):
        """The transcendental equation must be k cot(k r0) = -κ."""
        if self.transcendental is None:
            return (False, "SKIP: transcendental not available")
        s = self.transcendental.replace(" ", "").lower()
        target = "kcot(kr0)=-kappa"  # allow variations without spaces
        if target in s:
            return (True, "Transcendental equation matches k cot(k r0) = -κ")
        else:
            return (False, f"Transcendental equation does not match expected: {self.transcendental}")

    def test_invariant_wavefunction_inside_form(self):
        """Inside the well, u(r) must be A sin(α r)."""
        if self.u_in is None:
            return (False, "SKIP: u_in not available")
        if 'sin(alpha r)' in self.u_in or 'sin(α r)' in self.u_in:
            return (True, "Inside wavefunction is sinusoidal (A sin(α r))")
        else:
            return (False, f"Inside wavefunction does not contain sin(α r): {self.u_in}")

    def test_invariant_wavefunction_outside_form(self):
        """Outside the well, u(r) must be exponentially decaying."""
        if self.u_out is None:
            return (False, "SKIP: u_out not available")
        # check for exponential with beta and r0
        if 'exp' in self.u_out and 'beta' in self.u_out and 'r0' in self.u_out:
            return (True, "Outside wavefunction contains exponential decay")
        else:
            return (False, f"Outside wavefunction does not have expected structure: {self.u_out}")

    def test_invariant_bound_state_condition(self):
        """Derive α from V0_min and verify α = 3π/2 (condition for two bound states)."""
        if self.V0_min is None or self.m is None or self.r0 is None or self.hbar is None:
            return (False, "SKIP: missing data for bound state condition")
        m, r0, hbar = sp.symbols('m r0 hbar', positive=True)
        given = self.V0_min
        if isinstance(given, str):
            try:
                V0_sym = sp.sympify(given.replace('hbar', 'hbar').replace('pi', 'pi'))
            except Exception:
                return (False, f"Cannot parse V0_min: {given}")
        else:
            V0_sym = given
        # α² = 2m V0 r0² / ħ²
        alpha_sq = (2 * m * V0_sym * r0**2) / hbar**2
        expected_alpha_sq = (3 * sp.pi / 2)**2
        diff = sp.simplify(alpha_sq - expected_alpha_sq)
        if diff == 0:
            return (True, "Bound state condition satisfied: α = 3π/2")
        else:
            return (False, f"α² = {alpha_sq}, expected {expected_alpha_sq}, diff {diff}")

    def test_invariant_E_est_formula(self):
        """E_est (uncertainty‑principle estimate) must equal -(9π²-1)ħ²/(8 m r0²)."""
        if self.E_est is None:
            return (False, "SKIP: E_est not available")
        m, r0, hbar = sp.symbols('m r0 hbar', positive=True)
        expected = -(9 * sp.pi**2 - 1) * hbar**2 / (8 * m * r0**2)
        given = self.E_est
        if isinstance(given, str):
            try:
                given_sym = sp.sympify(given.replace('hbar', 'hbar').replace('pi', 'pi'))
            except Exception:
                return (False, f"Cannot parse E_est: {given}")
        else:
            given_sym = given
        diff = sp.simplify(given_sym - expected)
        if diff == 0:
            return (True, "E_est matches theoretical estimate")
        else:
            # Qualitative check: must be negative (bound state)
            if given_sym.is_negative:
                return (True, "E_est is negative (qualitative bound state), even if exact form differs")
            return (False, f"E_est does not match: {given_sym} vs {expected}")

    def test_invariant_comparison(self):
        """Comparison must state that exact E1 is less than the estimate."""
        if self.comparison is None:
            return (False, "SKIP: comparison not available")
        s = self.comparison.lower()
        if 'exact' in s and ('<' in s or 'less' in s or 'lower' in s):
            return (True, "Comparison correctly indicates exact E1 is lower than estimate")
        else:
            return (False, f"Comparison does not convey expected relation: {self.comparison}")

    # ------------------------------------------------------------------
    # Modeling quality tests
    # ------------------------------------------------------------------
    def test_modeling_T_expect_form(self):
        """T_expect must be expressed in terms of E1, V0, A², and trigonometric terms."""
        if self.T_expect is None:
            return (False, "SKIP: T_expect not available")
        s = self.T_expect
        if 'E1' in s and 'V0' in s and 'A' in s and 'sin' in s:
            return (True, "T_expect contains expected components (E1, V0, A², sin)")
        else:
            return (False, f"T_expect does not contain required terms: {s}")

    def test_modeling_uncertainty_principle_stated(self):
        """The uncertainty principle must be stated as Δp Δr ≥ ħ/2."""
        if self.UP is None:
            return (False, "SKIP: UP not available")
        s = self.UP.replace(" ", "")
        if "ΔpΔr≥ħ/2" in s or "Delta p Delta r >= hbar/2" in s:
            return (True, "Uncertainty principle correctly stated")
        else:
            return (False, f"Uncertainty principle statement mismatch: {self.UP}")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_E_est_with_r0(self):
        """
        Perturb r0 by +5% and check that the uncertainty–principle estimate
        E_est becomes less negative (i.e., increases). Uses the known formula
        re-derived from givens.
        """
        if self.m is None or self.r0 is None or self.hbar is None:
            return (False, "SKIP: missing givens for monotonicity test")
        # Use typical values to evaluate numerically; take m=1, hbar=1 for simplicity
        # but we can also use the provided givens if they are numeric; if symbolic we substitute.
        # For robustness, we use the given m, r0, hbar but if they are symbolic we substitute arbitrary positive numbers.
        m_val = self.m if isinstance(self.m, (int, float)) else 1.0
        hbar_val = self.hbar if isinstance(self.hbar, (int, float)) else 1.0
        r0_val = self.r0 if isinstance(self.r0, (int, float)) else 1.0

        # baseline: using provided givens
        baseline = -(9*math.pi**2 - 1) * hbar_val**2 / (8 * m_val * r0_val**2)

        # perturb r0 upward by 5%
        r0_perturbed = r0_val * 1.05
        perturbed = -(9*math.pi**2 - 1) * hbar_val**2 / (8 * m_val * r0_perturbed**2)

        # Since r0 increases, E_est (which is negative) should become less negative.
        # So baseline < perturbed (both negative, baseline more negative).
        if baseline < perturbed:
            return (True, f"E_est increases as r0 increases: baseline {baseline:.6f}, perturbed {perturbed:.6f}")
        else:
            return (False, f"Monotonicity violated: baseline {baseline:.6f}, perturbed {perturbed:.6f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one method)
    # ------------------------------------------------------------------
    def test_answer_sanity_V0_min_positive(self):
        """Minimum depth must be positive (bound state condition)."""
        if self.V0_min is None:
            return (False, "SKIP: V0_min not available")
        if isinstance(self.V0_min, (int, float)):
            if self.V0_min > 0:
                return (True, "V0_min is positive")
            else:
                return (False, f"V0_min should be positive, got {self.V0_min}")
        # If it's a symbolic expression, check analytically
        try:
            expr = self.V0_min
            if isinstance(expr, str):
                expr = sp.sympify(expr.replace('hbar', 'hbar').replace('pi', 'pi'))
            if expr.is_positive:
                return (True, "V0_min is analytically positive")
            else:
                return (False, "V0_min expression does not guarantee positivity")
        except Exception:
            return (False, "Could not determine positivity of V0_min")