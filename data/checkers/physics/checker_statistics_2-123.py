import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics auditor for statistical mechanics problem:
    partition function, heat capacity relation, two-level system heat capacity.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract givens dict safely (expected keys: Delta, k, T, etc.)
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Expected keys from pseudocode
        self.z_def = self.results.get('z_def', None)
        self.C_relation = self.results.get('C_relation', None)
        self.C2 = self.results.get('C2', None)
        self.U2 = self.results.get('U2', None)

        # Extract givens for numerical checks
        self.Delta = self.givens.get('Delta', None)
        self.k = self.givens.get('k', None)
        self.T = self.givens.get('T', None)

        # Derived or missing givens: we'll compute beta if possible
        self.beta = None
        if self.k is not None and self.T is not None and self.k != 0 and self.T != 0:
            self.beta = 1.0 / (self.k * self.T)

        # Helper: numeric evaluation of expressions (if they are numeric or symbolic)
        self._numeric_U2 = None
        self._numeric_C2 = None
        if self.U2 is not None:
            try:
                if isinstance(self.U2, (int, float)):
                    self._numeric_U2 = float(self.U2)
                else:
                    sub_dict = {}
                    if self.k is not None and self.T is not None:
                        sub_dict[sp.symbols('k')] = self.k
                        sub_dict[sp.symbols('T')] = self.T
                    if self.Delta is not None:
                        sub_dict[sp.symbols('Delta')] = self.Delta
                    if self.beta is not None:
                        sub_dict[sp.symbols('beta')] = self.beta
                    self._numeric_U2 = float(self.U2.evalf(subs=sub_dict))
            except Exception:
                self._numeric_U2 = None

        if self.C2 is not None:
            try:
                if isinstance(self.C2, (int, float)):
                    self._numeric_C2 = float(self.C2)
                else:
                    sub_dict = {}
                    if self.k is not None and self.T is not None:
                        sub_dict[sp.symbols('k')] = self.k
                        sub_dict[sp.symbols('T')] = self.T
                    if self.Delta is not None:
                        sub_dict[sp.symbols('Delta')] = self.Delta
                    if self.beta is not None:
                        sub_dict[sp.symbols('beta')] = self.beta
                    self._numeric_C2 = float(self.C2.evalf(subs=sub_dict))
            except Exception:
                self._numeric_C2 = None

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

    # ----------------------------------------------------------------------
    # Interface tests: ensure required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that results contain the four required keys from pseudocode."""
        missing = []
        for key in ['z_def', 'C_relation', 'C2', 'U2']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present.")

    def test_interface_z_def_format(self):
        """Check that z_def is a non-empty string or sympy expression."""
        if self.z_def is None:
            return (False, "SKIP: z_def not provided")
        if isinstance(self.z_def, str) and len(self.z_def.strip()) > 0:
            return (True, "z_def is a non-empty string")
        try:
            sp.sympify(self.z_def)
            return (True, "z_def is a valid sympy expression")
        except Exception:
            return (False, "z_def is not a valid string or expression")

    def test_interface_C_relation_format(self):
        """Check that C_relation is a non-empty string or expression containing derivative."""
        if self.C_relation is None:
            return (False, "SKIP: C_relation not provided")
        if isinstance(self.C_relation, str) and len(self.C_relation.strip()) > 0:
            if 'd' in self.C_relation or '∂' in self.C_relation:
                return (True, "C_relation is a string with derivative notation")
            else:
                return (True, "C_relation is a non-empty string")
        try:
            sp.sympify(self.C_relation)
            return (True, "C_relation is a valid sympy expression")
        except Exception:
            return (False, "C_relation is not a valid expression")

    # ----------------------------------------------------------------------
    # Invariant tests: theorem-based physics sanity, use math.isclose
    # ----------------------------------------------------------------------
    def test_invariant_U2_formula(self):
        """Check that U2 matches the expected analytic form: U = Δ/(e^{βΔ}+1)."""
        if self._numeric_U2 is None:
            return (False, "SKIP: U2 numeric evaluation not possible")
        if self.Delta is None or self.beta is None:
            return (False, "SKIP: missing Delta or beta for expected U")
        expected = self.Delta / (math.exp(self.beta * self.Delta) + 1.0)
        if math.isclose(self._numeric_U2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U2 ≈ {expected:.4f}, matches expected.")
        return (False, f"U2 = {self._numeric_U2:.4f}, expected {expected:.4f}")

    def test_invariant_C2_formula(self):
        """Check that C2 matches the expected form: C = k (Δ/(kT))^2 e^{βΔ}/(1+e^{βΔ})^2."""
        if self._numeric_C2 is None:
            return (False, "SKIP: C2 numeric evaluation not possible")
        if self.Delta is None or self.beta is None or self.k is None:
            return (False, "SKIP: missing parameters for expected C")
        x = self.beta * self.Delta  # βΔ
        expected = self.k * (x**2) * math.exp(x) / ((1 + math.exp(x)) ** 2)
        if math.isclose(self._numeric_C2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C2 ≈ {expected:.4f}, matches expected.")
        return (False, f"C2 = {self._numeric_C2:.4f}, expected {expected:.4f}")

    def test_invariant_C2_nonnegative(self):
        """Sanity: C2 must be non-negative for any temperature."""
        if self._numeric_C2 is None:
            return (False, "SKIP: C2 not numeric")
        if self._numeric_C2 < -1e-10:
            return (False, f"C2 is negative: {self._numeric_C2}")
        return (True, "C2 is non-negative.")

    # ----------------------------------------------------------------------
    # Modeling tests: check consistency of derived relations
    # ----------------------------------------------------------------------
    def test_modeling_derivative_relation(self):
        """
        Check relation: C = -k β^2 dU/dβ.
        We compute numerical derivative of U formula and compare to C2.
        """
        if self._numeric_U2 is None or self._numeric_C2 is None:
            return (False, "SKIP: need numeric U2 and C2")
        if self.beta is None or self.k is None:
            return (False, "SKIP: missing beta or k")
        if self.Delta is None:
            return (False, "SKIP: Delta needed for derivative check.")
        # Use analytic U function for finite difference
        def U_func(b):
            return self.Delta / (math.exp(b * self.Delta) + 1.0)
        delta_beta = 1e-6
        beta_plus = self.beta + delta_beta
        beta_minus = self.beta - delta_beta
        dU_dbeta = (U_func(beta_plus) - U_func(beta_minus)) / (2 * delta_beta)
        derived_C = -self.k * (self.beta ** 2) * dU_dbeta
        if math.isclose(self._numeric_C2, derived_C, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C2 ≈ {derived_C:.4f} from derivative relation.")
        return (False, f"C2 = {self._numeric_C2:.4f}, from derivative = {derived_C:.4f}")

    def test_modeling_partition_function_ln_second_derivative(self):
        """
        Check C2 = k β^2 d^2 ln(z)/dβ^2 for two-level system.
        Compute ln(z) = ln(1+e^{-βΔ}), then second derivative numerically.
        """
        if self._numeric_C2 is None:
            return (False, "SKIP: need numeric C2")
        if self.Delta is None or self.beta is None or self.k is None:
            return (False, "SKIP: missing parameters")
        delta_beta = 1e-6
        def ln_z(b):
            return math.log(1 + math.exp(-b * self.Delta))
        lz_plus = ln_z(self.beta + delta_beta)
        lz_minus = ln_z(self.beta - delta_beta)
        lz_0 = ln_z(self.beta)
        second_deriv = (lz_plus - 2*lz_0 + lz_minus) / (delta_beta ** 2)
        derived_C = self.k * (self.beta ** 2) * second_deriv
        if math.isclose(self._numeric_C2, derived_C, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C2 ≈ {derived_C:.4f} from second derivative of ln(z).")
        return (False, f"C2 = {self._numeric_C2:.4f}, from second deriv = {derived_C:.4f}")

    # ----------------------------------------------------------------------
    # Monotonicity tests: trend based on physical parameter changes
    # ----------------------------------------------------------------------
    def test_monotonicity_C2_increasing_low_T(self):
        """
        For very low temperatures (β large), C2 should increase with increasing T.
        Compare two temperatures: T1 = Δ/(10k), T2 = Δ/(5k). Expect C(T2) > C(T1).
        """
        if self.Delta is None or self.k is None:
            return (False, "SKIP: need Delta and k")
        def C_func(T):
            beta_T = 1.0 / (self.k * T)
            x = beta_T * self.Delta
            return self.k * (x**2) * math.exp(x) / ((1 + math.exp(x))**2)
        T1 = self.Delta / (10.0 * self.k)
        T2 = self.Delta / (5.0 * self.k)
        C1 = C_func(T1)
        C2_T = C_func(T2)
        if C2_T > C1 + 1e-12:
            return (True, f"C increases with T at low T: C({T2:.3f}) = {C2_T:.6f} > C({T1:.3f}) = {C1:.6f}")
        else:
            return (False, f"Low T monotonicity failed: C({T2:.3f}) = {C2_T:.6f} not > C({T1:.3f}) = {C1:.6f}")

    def test_monotonicity_C2_decreasing_high_T(self):
        """
        For high temperatures, C2 should decrease with increasing T.
        Compare T1 = 5Δ/k, T2 = 10Δ/k. Expect C(T1) > C(T2).
        """
        if self.Delta is None or self.k is None:
            return (False, "SKIP: need Delta and k")
        def C_func(T):
            beta_T = 1.0 / (self.k * T)
            x = beta_T * self.Delta
            return self.k * (x**2) * math.exp(x) / ((1 + math.exp(x))**2)
        T1 = 5.0 * self.Delta / self.k
        T2 = 10.0 * self.Delta / self.k
        C1 = C_func(T1)
        C2_T = C_func(T2)
        if C1 > C2_T + 1e-12:
            return (True, f"C decreases with T at high T: C({T1:.3f}) = {C1:.6f} > C({T2:.3f}) = {C2_T:.6f}")
        else:
            return (False, f"High T monotonicity failed: C({T1:.3f}) = {C1:.6f} not > C({T2:.3f}) = {C2_T:.6f}")

    # ----------------------------------------------------------------------
    # Single weak answer-sanity test (finite/non-NaN check only)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Check that both U2 and C2 are finite numbers (not inf, not NaN)."""
        results_ok = True
        if self._numeric_U2 is None and self._numeric_C2 is None:
            return (False, "SKIP: neither U2 nor C2 numeric")
        if self._numeric_U2 is not None:
            if not math.isfinite(self._numeric_U2):
                return (False, f"U2 is not finite: {self._numeric_U2}")
        if self._numeric_C2 is not None:
            if not math.isfinite(self._numeric_C2):
                return (False, f"C2 is not finite: {self._numeric_C2}")
        return (True, "All available answer quantities are finite.")