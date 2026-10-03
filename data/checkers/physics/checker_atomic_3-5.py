import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsAuditor for G-parity problem.
    Checks interface, invariants, modeling quality, monotonicity, and weak sanity.
    """

    def __init__(self, solution_instance):
        """
        Extract all expected result keys safely with None defaults.
        """
        results = getattr(solution_instance, 'results', {})
        self.G_op = results.get('G_op', None)
        self.G_pi = results.get('G_pi', None)
        self.G_npi = results.get('G_npi', None)
        self.G_rho = results.get('G_rho', None)
        self.G_omega = results.get('G_omega', None)
        self.G_phi = results.get('G_phi', None)
        self.G_eta = results.get('G_eta', None)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _is_symbolic_or_string(self, value):
        return isinstance(value, (sp.Basic, str))

    def _to_numeric(self, val):
        """Try to convert val to a float. Returns None if not possible."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, sp.Basic):
            try:
                return float(val.evalf())
            except Exception:
                return None
        if isinstance(val, str):
            try:
                return float(sp.sympify(val).evalf())
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Interface tests: existence and type checks
    # ------------------------------------------------------------------
    def test_interface_G_op_exists(self):
        if self.G_op is None:
            return (False, "G_op missing")
        if not self._is_symbolic_or_string(self.G_op):
            return (False, f"G_op has unexpected type: {type(self.G_op)}")
        return (True, "G_op present and symbolic/string")

    def test_interface_G_pi_exists(self):
        if self.G_pi is None:
            return (False, "G_pi missing")
        if not isinstance(self.G_pi, (int, float, sp.Basic, str)):
            return (False, f"G_pi type unsupported: {type(self.G_pi)}")
        return (True, "G_pi present")

    def test_interface_G_npi_exists(self):
        if self.G_npi is None:
            return (False, "G_npi missing")
        if not isinstance(self.G_npi, (sp.Basic, str)):
            return (False, f"G_npi type unsupported: {type(self.G_npi)}")
        return (True, "G_npi present")

    def test_interface_G_rho_exists(self):
        if self.G_rho is None:
            return (False, "G_rho missing")
        if not isinstance(self.G_rho, (int, float, sp.Basic, str)):
            return (False, f"G_rho type: {type(self.G_rho)}")
        return (True, "G_rho present")

    def test_interface_G_omega_exists(self):
        if self.G_omega is None:
            return (False, "G_omega missing")
        if not isinstance(self.G_omega, (int, float, sp.Basic, str)):
            return (False, f"G_omega type: {type(self.G_omega)}")
        return (True, "G_omega present")

    def test_interface_G_phi_exists(self):
        if self.G_phi is None:
            return (False, "G_phi missing")
        if not isinstance(self.G_phi, (int, float, sp.Basic, str)):
            return (False, f"G_phi type: {type(self.G_phi)}")
        return (True, "G_phi present")

    def test_interface_G_eta_exists(self):
        if self.G_eta is None:
            return (False, "G_eta missing")
        if not isinstance(self.G_eta, (int, float, sp.Basic, str)):
            return (False, f"G_eta type: {type(self.G_eta)}")
        return (True, "G_eta present")

    # ------------------------------------------------------------------
    # Invariant tests: physics correctness
    # ------------------------------------------------------------------
    def test_invariant_G_op_product_form(self):
        """Check that G_op contains C and exp(iπ I2) structure."""
        if self.G_op is None:
            return (False, "SKIP: G_op not available")
        s = str(self.G_op)
        has_C = 'C' in s
        has_exp = 'exp' in s or ('e' in s and 'π' in s) or ('e' in s and 'pi' in s)
        if not (has_C and has_exp):
            return (False, f"G_op does not clearly show C * exp(iπ I2): {s}")
        return (True, "G_op has product form C * exp(iπ I2)")

    def test_invariant_G_pi_value(self):
        """G eigenvalue for any pion is -1."""
        if self.G_pi is None:
            return (False, "SKIP: G_pi not available")
        val = self._to_numeric(self.G_pi)
        if val is None:
            s = str(self.G_pi)
            if '-1' in s:
                return (True, "G_pi is -1 (symbolic check)")
            return (False, f"Cannot confirm G_pi = -1 from {s}")
        if not math.isclose(val, -1.0, rel_tol=1e-9, abs_tol=1e-9):
            return (False, f"G_pi should be -1, got {val}")
        return (True, f"G_pi = {val} (should be -1)")

    def test_invariant_G_npi_form(self):
        """G for n pions should be (-1)^n or equivalent."""
        if self.G_npi is None:
            return (False, "SKIP: G_npi not available")
        s = str(self.G_npi)
        if '(-1)^n' in s or '(-1)**n' in s or '(-1)**n' in s.replace(' ', ''):
            return (True, "G_npi has correct form (-1)^n")
        return (False, f"G_npi does not appear as (-1)^n, got: {s}")

    def test_invariant_G_rho_value(self):
        return self._check_meson_value(self.G_rho, 1.0, "G_rho")

    def test_invariant_G_omega_value(self):
        return self._check_meson_value(self.G_omega, -1.0, "G_omega")

    def test_invariant_G_phi_value(self):
        return self._check_meson_value(self.G_phi, -1.0, "G_phi")

    def test_invariant_G_eta_value(self):
        return self._check_meson_value(self.G_eta, 1.0, "G_eta")

    def _check_meson_value(self, val, expected, name):
        if val is None:
            return (False, f"SKIP: {name} not available")
        numeric = self._to_numeric(val)
        if numeric is None:
            s = str(val).strip()
            if (expected == 1 and s in ('+1', '1', '1.0')) or \
               (expected == -1 and s in ('-1', '-1.0')):
                return (True, f"{name} = {s} (symbolic match)")
            return (False, f"Cannot confirm {name} = {expected} from {s}")
        if not math.isclose(numeric, expected, rel_tol=1e-9, abs_tol=1e-9):
            return (False, f"{name} should be {expected}, got {numeric}")
        return (True, f"{name} = {numeric} (expected {expected})")

    # ------------------------------------------------------------------
    # Modeling quality / trace checks
    # ------------------------------------------------------------------
    def test_modeling_G_npi_consistent_with_G_pi(self):
        """Check that G_npi = (G_pi)^n."""
        if self.G_pi is None or self.G_npi is None:
            return (False, "SKIP: Missing G_pi or G_npi")
        n = sp.Symbol('n', integer=True, nonnegative=True)
        G_pi_num = self._to_numeric(self.G_pi)
        if G_pi_num is not None:
            expected = G_pi_num ** n
        else:
            expected = (-1) ** n
        # Simplify given expression
        given = sp.sympify(str(self.G_npi)) if isinstance(self.G_npi, str) else self.G_npi
        if sp.simplify(expected - given) == 0:
            return (True, "G_npi consistent with (G_pi)^n")
        if str(sp.simplify(expected)) in str(self.G_npi):
            return (True, "G_npi consistent (string match)")
        return (False, f"G_npi = {self.G_npi} but expected {sp.simplify(expected)}")

    def test_modeling_meson_G_via_decay_consistency(self):
        """Check meson G values match strong decay rules (rho→2π → +1, ω,φ→3π → -1)."""
        if None in (self.G_rho, self.G_omega, self.G_phi):
            return (False, "SKIP: missing meson G values")
        rho_ok = self._check_meson_value(self.G_rho, 1.0, "")[0]
        omega_ok = self._check_meson_value(self.G_omega, -1.0, "")[0]
        phi_ok = self._check_meson_value(self.G_phi, -1.0, "")[0]
        if rho_ok and omega_ok and phi_ok:
            return (True, "Meson G values consistent with strong decay selection rules")
        return (False, "One or more meson G values mismatch")

    # ------------------------------------------------------------------
    # Monotonicity test (must perturb input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_G_npi_sign_flip(self):
        """
        Monotonicity for n-pion system: G changes sign when n increases by 1.
        Perturb n -> n+1 and check that G_{n+1} = -G_n.
        """
        if self.G_npi is None:
            return (False, "SKIP: G_npi not available")
        # Convert to sympy expression
        if isinstance(self.G_npi, str):
            try:
                expr = sp.sympify(self.G_npi)
            except Exception:
                return (False, "Cannot parse G_npi as symbolic expression")
        elif isinstance(self.G_npi, sp.Basic):
            expr = self.G_npi
        else:
            return (False, f"Unexpected type for G_npi: {type(self.G_npi)}")

        n = sp.Symbol('n', integer=True, nonnegative=True)
        # Baseline: G(n)
        G_n = expr
        # Perturbed: G(n+1)
        G_n_plus_1 = expr.subs(n, n+1)
        # Check if G(n+1) + G(n) == 0 algebraically
        if sp.simplify(G_n_plus_1 + G_n) == 0:
            return (True, "G_{n+1} = -G_n confirmed by algebraic simplification")
        # If not fully symbolic, try numeric check for n=0,1,2
        try:
            for n_test in [0, 1, 2]:
                val_base = float(expr.subs(n, n_test).evalf())
                val_pert = float(expr.subs(n, n_test+1).evalf())
                if not math.isclose(val_pert, -val_base, rel_tol=1e-9, abs_tol=1e-9):
                    return (False, f"At n={n_test}, base={val_base}, perturbed={val_pert}, expected opposite sign.")
        except Exception as e:
            return (False, f"Could not evaluate numerically: {e}")
        return (True, "Monotonicity: sign flips for n->n+1 verified")

    # ------------------------------------------------------------------
    # Weak sanity (at most one method)
    # ------------------------------------------------------------------
    def test_answer_sanity_all_finite(self):
        """Weak sanity: all G values should be finite (or symbolic) and not None."""
        keys = ['G_op', 'G_pi', 'G_npi', 'G_rho', 'G_omega', 'G_phi', 'G_eta']
        missing = []
        for k in keys:
            val = getattr(self, k, None)
            if val is None:
                missing.append(k)
        if missing:
            return (False, f"Missing keys: {missing}")
        # For numeric values, check they are finite
        for k in ('G_pi', 'G_rho', 'G_omega', 'G_phi', 'G_eta'):
            val = getattr(self, k)
            numeric = self._to_numeric(val)
            if numeric is not None and (math.isnan(numeric) or math.isinf(numeric)):
                return (False, f"{k} is NaN or Inf")
        return (True, "All G values are present and finite (where numeric)")

    # ------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)