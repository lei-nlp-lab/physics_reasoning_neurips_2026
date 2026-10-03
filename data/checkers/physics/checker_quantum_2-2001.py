import math
import re

class PhysicsChecker:
    """
    Checker for 3D infinite potential well problem.
    Verifies interface, structural invariance, modeling quality,
    monotonicity (via numeric perturbation), and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely; fallback to empty dict
        self.results = getattr(solution_instance, 'results', None) or {}
        # Optional givens, not strictly needed for structural checks
        self.givens = getattr(solution_instance, 'givens', None) or {}

        # Store specific expected output keys with None defaults
        self.SE = self.results.get('SE', None)
        self.psi_111 = self.results.get('psi_111', None)
        self.N = self.results.get('N', None)

    # ---------- Interface tests ----------
    def test_interface_results_dict(self):
        if not isinstance(self.results, dict):
            return (False, "results is not a dict")
        return (True, "results is a dict")

    def test_interface_key_SE(self):
        if self.SE is None:
            return (False, "SE key missing in results")
        return (True, "SE key present")

    def test_interface_key_psi_111(self):
        if self.psi_111 is None:
            return (False, "psi_111 key missing in results")
        return (True, "psi_111 key present")

    def test_interface_key_N(self):
        if self.N is None:
            return (False, "N key missing in results")
        return (True, "N key present")

    def test_interface_type_SE(self):
        if self.SE is None:
            return (False, "SKIP: SE is missing")
        if not isinstance(self.SE, str):
            return (False, f"SE is not a string, type={type(self.SE)}")
        return (True, "SE is a string")

    def test_interface_type_psi_111(self):
        if self.psi_111 is None:
            return (False, "SKIP: psi_111 is missing")
        if not isinstance(self.psi_111, str):
            return (False, f"psi_111 is not a string, type={type(self.psi_111)}")
        return (True, "psi_111 is a string")

    def test_interface_type_N(self):
        if self.N is None:
            return (False, "SKIP: N is missing")
        if not isinstance(self.N, str):
            return (False, f"N is not a string, type={type(self.N)}")
        return (True, "N is a string")

    # ---------- Invariant tests (theorem-based physical sanity) ----------
    def test_invariant_SE_contains_hbar_and_nabla(self):
        if self.SE is None:
            return (False, "SKIP: SE is missing")
        s = self.SE.lower()
        has_hbar = 'hbar' in s
        has_nabla = 'nabla^2' in s or 'nabla^2' in s.replace(' ', '')
        has_partial = 'partial' in s or '\partial' in s
        if not (has_hbar and has_nabla and has_partial):
            missing = []
            if not has_hbar: missing.append('hbar')
            if not has_nabla: missing.append('nabla^2')
            if not has_partial: missing.append('partial/∂')
            return (False, f"SE missing key elements: {', '.join(missing)}")
        return (True, "SE contains hbar, nabla^2, and partial derivative marker")

    def test_invariant_SE_time_dependent_form(self):
        if self.SE is None:
            return (False, "SKIP: SE is missing")
        s = self.SE.lower()
        has_i = 'i' in s and 'hbar' in s
        has_d_dt = 'd/dt' in s or '∂/∂t' in s or ('partial' in s and 't' in s)
        if not (has_i and has_d_dt):
            if not has_i:
                return (False, "SE does not contain 'i' with 'hbar' (time-dependent form)")
            return (False, "SE does not contain time derivative (∂/∂t)")
        return (True, "SE has time-dependent form with iħ∂/∂t")

    def test_invariant_psi_111_form(self):
        if self.psi_111 is None:
            return (False, "SKIP: psi_111 is missing")
        s = self.psi_111.lower()
        has_sin_x = 'sin(pi x/l)' in s
        has_sin_y = 'sin(pi y/l)' in s
        has_sin_z = 'sin(pi z/l)' in s
        if not (has_sin_x and has_sin_y and has_sin_z):
            missing = []
            if not has_sin_x: missing.append('sin(pi x/L)')
            if not has_sin_y: missing.append('sin(pi y/L)')
            if not has_sin_z: missing.append('sin(pi z/L)')
            return (False, f"psi_111 missing: {', '.join(missing)}")
        has_factor = '(2/l)^{3/2}' in s or '(2/l)^(3/2)' in s
        if not has_factor:
            return (False, "psi_111 missing normalization factor (2/L)^{3/2}")
        return (True, "psi_111 has correct sine product and normalization factor")

    def test_invariant_psi_111_ground_state(self):
        if self.psi_111 is None:
            return (False, "SKIP: psi_111 is missing")
        s = self.psi_111.lower()
        # Check that only n=1 terms appear (no 2 in sine argument)
        if 'sin(2 pi x/l)' in s or 'sin(2\\pi x/l)' in s:
            return (False, "psi_111 contains sin(2πx/L) – not ground state")
        if 'sin(2 pi y/l)' in s or 'sin(2\\pi y/l)' in s:
            return (False, "psi_111 contains sin(2πy/L) – not ground state")
        if 'sin(2 pi z/l)' in s or 'sin(2\\pi z/l)' in s:
            return (False, "psi_111 contains sin(2πz/L) – not ground state")
        return (True, "psi_111 uses n=1 for each dimension")

    def test_invariant_N_form(self):
        if self.N is None:
            return (False, "SKIP: N is missing")
        s = self.N.lower()
        has_E = 'e' in s
        has_hbar = 'hbar' in s
        has_pi = 'pi' in s or 'π' in s
        has_3over2 = '3/2' in s or '1.5' in s or '^{3/2}' in s or '**1.5' in s
        if not (has_E and has_hbar and has_pi and has_3over2):
            missing = []
            if not has_E: missing.append('E')
            if not has_hbar: missing.append('hbar')
            if not has_pi: missing.append('pi')
            if not has_3over2: missing.append('3/2 exponent')
            return (False, f"N expression missing: {', '.join(missing)}")
        return (True, "N expression contains E, hbar, π, and exponent 3/2")

    def test_invariant_N_positive_exponent(self):
        if self.N is None:
            return (False, "SKIP: N is missing")
        s = self.N.lower()
        if '3/2' not in s and '1.5' not in s and '^{3/2}' not in s and '**1.5' not in s:
            return (False, "N expression does not contain positive exponent 3/2")
        if '-3/2' in s:
            return (False, "N expression contains negative exponent -3/2")
        return (True, "N exponent is positive (3/2)")

    # ---------- Modeling tests (based on explanation steps) ----------
    def test_modeling_psi_111_boundary_consistency(self):
        if self.psi_111 is None:
            return (False, "SKIP: psi_111 is missing")
        s = self.psi_111.lower()
        if 'cos' in s:
            return (False, "psi_111 contains cosine terms; must be sine only for Dirichlet BC")
        if 'exp' in s:
            return (False, "psi_111 contains exponential terms; expected sine product")
        return (True, "psi_111 has only sine functions, consistent with boundary conditions")

    def test_modeling_N_counting_continuum(self):
        if self.N is None:
            return (False, "SKIP: N is missing")
        s = self.N.lower()
        if 'sum' in s or '\\sum' in s or 'Σ' in s:
            return (False, "N expression appears as a sum; expected continuum integral")
        return (True, "N expression is algebraic (continuum approximation)")

    # ---------- Monotonicity test (numeric perturbation) ----------
    def _safe_eval_N(self, E_val):
        """
        Try to evaluate the N string for a given numeric E_val.
        Assumes m=1, L=1, hbar=1.  Returns the computed value or None on failure.
        """
        if self.N is None:
            return None
        expr = self.N
        # Replace common LaTeX and symbolic patterns
        replacements = [
            (r'\hbar', '1'),
            (r'\pi', str(math.pi)),
            (r'π', str(math.pi)),
            (r'\^', '**'),
            ('^', '**'),
            (r'm', '1'),
            (r'L', '1'),
            (r'E', str(E_val)),
        ]
        for pattern, replacement in replacements:
            # Use regex to replace whole word? Simple string replace:
            expr = expr.replace(pattern, replacement)
        # Clean up any LaTeX brackets
        expr = expr.replace('{', '(').replace('}', ')')
        # Remove whitespace
        expr = expr.replace(' ', '')
        # Remove trailing . if any
        try:
            # Ensure only allowed names are used
            allowed_names = {'__builtins__': None}
            # Use math namespace for functions like sqrt, sin, etc.
            result = eval(expr, {"__builtins__": None}, {"math": math})
            return result
        except Exception:
            return None

    def test_monotonicity_N_increases_with_E(self):
        """
        Perturb energy E from 1.0 to 2.0 and verify N increases.
        Uses arbitrary numeric constants m=1, L=1, hbar=1.
        """
        if self.N is None:
            return (False, "SKIP: N is missing")
        # Baseline at E=1
        N1 = self._safe_eval_N(1.0)
        if N1 is None:
            return (False, "SKIP: Could not evaluate N expression numerically")
        # Perturb to E=2
        N2 = self._safe_eval_N(2.0)
        if N2 is None:
            return (False, "SKIP: Could not evaluate N expression at E=2.0")
        # Check monotonic increase with a generous tolerance (should be strictly greater)
        if N2 <= N1:
            return (False, f"N does not increase with E: N(1.0)={N1:.6f}, N(2.0)={N2:.6f}")
        # Also check relative increase is positive and roughly (2)^{3/2} = 2.828
        rel_increase = N2 / N1
        expected_rel = 2.0 ** 1.5  # ~2.828
        if not math.isclose(rel_increase, expected_rel, rel_tol=0.1):
            return (False, f"Relative increase {rel_increase:.4f} deviates from expected {expected_rel:.4f}")
        return (True, f"N increases with E: N(1.0)={N1:.6f}, N(2.0)={N2:.6f}")

    # ---------- Weak answer sanity (at most one) ----------
    def test_answer_sanity_basic(self):
        if self.N is None:
            return (False, "SKIP: N is missing")
        if not isinstance(self.N, str) or len(self.N.strip()) == 0:
            return (False, "N is empty or not a string")
        s = self.N.lower()
        if 'e' not in s:
            return (False, "N does not contain 'E' – answer sanity weak check failed")
        if '3/2' not in s and '1.5' not in s:
            return (False, "N does not contain exponent 3/2 – answer sanity weak check failed")
        return (True, "N passes basic sanity: non-empty, contains E and 3/2")

    # ---------- Runner ----------
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