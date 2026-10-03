import math
import sympy as sp
import traceback

class PhysicsChecker:
    """Checker for the electromagnetic wave propagation in a plasma problem."""

    def __init__(self, solution_instance):
        # Extract results with safe defaults
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract each required key, defaulting to None
        self.J = self.results.get('J', None)
        self.wave_eq = self.results.get('wave_eq', None)
        self.k_sq = self.results.get('k_sq', None)
        self.propagation_cond = self.results.get('propagation_cond', None)
        self.n_e_max = self.results.get('n_e_max', None)

        # Define symbols used in physics expressions
        self.sym_omega = sp.Symbol('omega', positive=True)
        self.sym_n_e = sp.Symbol('n_e', positive=True)
        self.sym_e = sp.Symbol('e', positive=True)
        self.sym_m_e = sp.Symbol('m_e', positive=True)
        self.sym_epsilon0 = sp.Symbol('epsilon0', positive=True)
        self.sym_c = sp.Symbol('c', positive=True)
        self.sym_omega_p = sp.Symbol('omega_p', positive=True)
        self.sym_E = sp.Symbol('E')
        self.sym_sigma = sp.Symbol('sigma')

    def _parse_to_expr(self, val):
        """Convert a value (string or sympy expression) to a sympy expression, or None on failure."""
        if val is None:
            return None
        if isinstance(val, sp.Expr):
            return val
        if isinstance(val, str):
            try:
                # Replace common physics notation for sympy
                expr_str = val.replace('\\nabla^2', 'nabla2').replace('\\nabla', 'nabla')
                return sp.sympify(expr_str)
            except Exception:
                return None
        return None

    # ----------------------------------------------------------------------
    # Interface tests: verify that required result keys are present and not None
    # ----------------------------------------------------------------------
    def test_interface_has_J(self):
        """J result key exists and is not None."""
        if self.J is None:
            return False, "SKIP: J not provided"
        return True, "J present"

    def test_interface_has_wave_eq(self):
        """wave_eq result key exists and is not None."""
        if self.wave_eq is None:
            return False, "SKIP: wave_eq not provided"
        return True, "wave_eq present"

    def test_interface_has_k_sq(self):
        """k_sq result key exists and is not None."""
        if self.k_sq is None:
            return False, "SKIP: k_sq not provided"
        return True, "k_sq present"

    def test_interface_has_propagation_cond(self):
        """propagation_cond result key exists and is not None."""
        if self.propagation_cond is None:
            return False, "SKIP: propagation_cond not provided"
        return True, "propagation_cond present"

    def test_interface_has_n_e_max(self):
        """n_e_max result key exists and is not None."""
        if self.n_e_max is None:
            return False, "SKIP: n_e_max not provided"
        return True, "n_e_max present"

    # ----------------------------------------------------------------------
    # Invariant tests: theorem-based physical checks using symbolic equivalence
    # ----------------------------------------------------------------------
    def test_invariant_J_correct_form(self):
        """J should be proportional to i * n_e * e^2 * E / (m_e * omega)."""
        if self.J is None:
            return False, "SKIP: J not available"
        expr = self._parse_to_expr(self.J)
        if expr is None:
            return False, "Cannot parse J expression"
        expected = sp.I * self.sym_n_e * self.sym_e**2 * self.sym_E / (self.sym_m_e * self.sym_omega)
        try:
            diff = sp.simplify(expr - expected)
            if diff == 0:
                return True, "J matches expected form i n_e e^2 E/(m_e omega)"
            expected_neg = -sp.I * self.sym_n_e * self.sym_e**2 * self.sym_E / (self.sym_m_e * self.sym_omega)
            diff2 = sp.simplify(expr - expected_neg)
            if diff2 == 0:
                return False, "J sign reversed; should be +i * n_e e^2/(omega m_e) E"
            else:
                return False, f"J expression does not match expected; diff = {diff}"
        except Exception as e:
            return False, f"Error comparing J expression: {e}"

    def test_invariant_wave_eq_structure(self):
        """Wave equation should contain nabla^2 and (omega^2 - omega_p^2)/c^2 term."""
        if self.wave_eq is None:
            return False, "SKIP: wave_eq not available"
        wave_str = str(self.wave_eq).replace(" ", "")
        # Check for essential pieces
        has_nabla = 'nabla2' in wave_str or '\\nabla^2' in wave_str
        has_omega_term = ('omega**2' in wave_str and '-omega_p**2' in wave_str) or \
                         ('omega^2-omega_p^2' in wave_str) or \
                         ('1-omega_p**2/omega**2' in wave_str) or \
                         ('1-\\omega_P^2/\\omega^2' in wave_str)
        if has_nabla and has_omega_term:
            return True, "Wave equation structure includes nabla^2 and (omega^2 - omega_p^2)/c^2"
        else:
            return False, f"Wave equation missing required structure: {wave_str}"

    def test_invariant_k_sq_correct(self):
        """k^2 should equal (omega^2 - omega_p^2)/c^2."""
        if self.k_sq is None:
            return False, "SKIP: k_sq not available"
        expr = self._parse_to_expr(self.k_sq)
        if expr is None:
            return False, "Cannot parse k_sq expression"
        expected = (self.sym_omega**2 - self.sym_omega_p**2) / self.sym_c**2
        try:
            diff = sp.simplify(expr - expected)
            if diff == 0:
                return True, "k_sq matches (omega^2 - omega_p^2)/c^2"
            alt_expected = (self.sym_omega**2 / self.sym_c**2) * (1 - self.sym_omega_p**2 / self.sym_omega**2)
            diff2 = sp.simplify(expr - alt_expected)
            if diff2 == 0:
                return True, "k_sq matches alternative form omega^2/c^2 (1 - omega_p^2/omega^2)"
            else:
                return False, f"k_sq does not match; diff = {diff}"
        except Exception as e:
            return False, f"Error comparing k_sq: {e}"

    def test_invariant_propagation_cond(self):
        """Propagation condition should state omega^2 > omega_p^2 or n_e < epsilon0*m_e*omega^2/e^2."""
        if self.propagation_cond is None:
            return False, "SKIP: propagation_cond not available"
        cond_str = str(self.propagation_cond).replace(" ", "")
        if 'omega**2>omega_p**2' in cond_str or 'omega^2>omega_p^2' in cond_str:
            return True, "Propagation condition: omega^2 > omega_p^2"
        elif 'n_e<epsilon0*m_e*omega**2/e**2' in cond_str or 'n_e<epsilon0*m_e*omega^2/e^2' in cond_str:
            return True, "Propagation condition: n_e < epsilon0*m_e*omega^2/e^2"
        else:
            if 'n_e' in cond_str and 'epsilon0' in cond_str and 'omega' in cond_str and 'e' in cond_str and '<' in cond_str:
                return True, "Propagation condition in terms of n_e (recognized pattern)"
            elif 'omega' in cond_str and 'omega_p' in cond_str and '>' in cond_str:
                return True, "Propagation condition in terms of omega (recognized pattern)"
            else:
                return False, f"Propagation condition not recognized: {cond_str}"

    def test_invariant_n_e_max_correct(self):
        """n_e_max should equal epsilon0*m_e*omega^2/e^2."""
        if self.n_e_max is None:
            return False, "SKIP: n_e_max not available"
        expr = self._parse_to_expr(self.n_e_max)
        if expr is None:
            return False, "Cannot parse n_e_max expression"
        expected = self.sym_epsilon0 * self.sym_m_e * self.sym_omega**2 / self.sym_e**2
        try:
            diff = sp.simplify(expr - expected)
            if diff == 0:
                return True, "n_e_max matches epsilon0*m_e*omega^2/e^2"
            else:
                return False, f"n_e_max does not match; diff = {diff}"
        except Exception as e:
            return False, f"Error comparing n_e_max: {e}"

    # ----------------------------------------------------------------------
    # Modeling / trace tests: check consistency of model steps
    # ----------------------------------------------------------------------
    def test_modeling_sigma_from_J(self):
        """From J, conductivity sigma = J/E should be i n_e e^2/(omega m_e)."""
        if self.J is None:
            return False, "SKIP: J not available"
        expr = self._parse_to_expr(self.J)
        if expr is None:
            return False, "Cannot parse J expression"
        # Compute sigma = J / E
        sigma_expr = sp.simplify(expr / self.sym_E)
        expected_sigma = sp.I * self.sym_n_e * self.sym_e**2 / (self.sym_omega * self.sym_m_e)
        diff = sp.simplify(sigma_expr - expected_sigma)
        if diff == 0:
            return True, "sigma from J matches i n_e e^2/(omega m_e)"
        else:
            return False, f"sigma does not match; diff = {diff}"

    def test_modeling_dispersion_from_wave_eq(self):
        """From wave_eq, substituting plane wave should yield k^2 expression."""
        if self.wave_eq is None or self.k_sq is None:
            return False, "SKIP: wave_eq or k_sq not available"
        # We cannot fully parse the wave_eq as an equation, but we can check that
        # the term (omega^2 - omega_p^2)/c^2 appears in both.
        k_expr = self._parse_to_expr(self.k_sq)
        if k_expr is None:
            return False, "Cannot parse k_sq"
        # Check that k_sq is consistent with wave equation structure
        expected_k = (self.sym_omega**2 - self.sym_omega_p**2) / self.sym_c**2
        diff = sp.simplify(k_expr - expected_k)
        if diff == 0:
            return True, "k_sq matches the coefficient expected from wave equation"
        else:
            return False, f"k_sq does not match the expected coefficient; diff = {diff}"

    def test_modeling_k_sq_positivity_implies_condition(self):
        """If k_sq > 0 then condition omega^2 > omega_p^2 should be equivalent."""
        if self.k_sq is None or self.propagation_cond is None:
            return False, "SKIP: k_sq or propagation_cond not available"
        k_expr = self._parse_to_expr(self.k_sq)
        if k_expr is None:
            return False, "Cannot parse k_sq"
        # Symbolically solve for condition k_sq > 0
        # Inequality: (omega^2 - omega_p^2)/c^2 > 0  => omega^2 > omega_p^2
        cond_str = str(self.propagation_cond).replace(" ", "")
        # Check that propagation_cond mentions something like omega^2 > omega_p^2
        if 'omega**2>' in cond_str or 'omega^2>' in cond_str:
            if 'omega_p**2' in cond_str or 'omega_p^2' in cond_str:
                return True, "Propagation condition is consistent with k_sq > 0"
        # Also check the n_e version: n_e < epsilon0*m_e*omega^2/e^2
        if 'n_e<' in cond_str and 'epsilon0' in cond_str and 'omega**2' in cond_str:
            return True, "Propagation condition (n_e version) is consistent with k_sq > 0 when substituting omega_p^2"
        return False, "Could not verify consistency between k_sq > 0 and propagation_cond"

    # ----------------------------------------------------------------------
    # Monotonicity tests: explicit perturbation of an input
    # ----------------------------------------------------------------------
    def test_monotonicity_k_sq_increases_with_omega(self):
        """Define baseline omega=5, omega_p=3, c=1; perturb omega by +0.1 => k^2 increases."""
        if self.k_sq is None:
            return False, "SKIP: k_sq not available"
        expr = self._parse_to_expr(self.k_sq)
        if expr is None:
            return False, "Cannot parse k_sq expression"
        # Choose numeric values for baseline (omega must be > omega_p)
        subs_base = {self.sym_omega: 5.0, self.sym_omega_p: 3.0, self.sym_c: 1.0}
        subs_perturbed = {self.sym_omega: 5.1, self.sym_omega_p: 3.0, self.sym_c: 1.0}
        try:
            base_val = expr.subs(subs_base).evalf()
            perturbed_val = expr.subs(subs_perturbed).evalf()
            if perturbed_val > base_val:
                return True, f"k^2 increased: base={base_val}, perturbed={perturbed_val}"
            else:
                return False, f"k^2 did not increase: base={base_val}, perturbed={perturbed_val}"
        except Exception as e:
            return False, f"Error in monotonicity test: {e}"

    def test_monotonicity_k_sq_decreases_with_omega_p(self):
        """Define baseline omega=5, omega_p=3, c=1; perturb omega_p by +0.1 => k^2 decreases."""
        if self.k_sq is None:
            return False, "SKIP: k_sq not available"
        expr = self._parse_to_expr(self.k_sq)
        if expr is None:
            return False, "Cannot parse k_sq expression"
        subs_base = {self.sym_omega: 5.0, self.sym_omega_p: 3.0, self.sym_c: 1.0}
        subs_perturbed = {self.sym_omega: 5.0, self.sym_omega_p: 3.1, self.sym_c: 1.0}
        try:
            base_val = expr.subs(subs_base).evalf()
            perturbed_val = expr.subs(subs_perturbed).evalf()
            if perturbed_val < base_val:
                return True, f"k^2 decreased: base={base_val}, perturbed={perturbed_val}"
            else:
                return False, f"k^2 did not decrease: base={base_val}, perturbed={perturbed_val}"
        except Exception as e:
            return False, f"Error in monotonicity test: {e}"

    # ----------------------------------------------------------------------
    # Weak answer sanity (optional, only one method)
    # ----------------------------------------------------------------------
    def test_answer_sanity_nonempty(self):
        """All required results should be non-empty strings (or expressions)."""
        keys = ['J', 'wave_eq', 'k_sq', 'propagation_cond', 'n_e_max']
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                return False, f"{key} is None"
            if isinstance(val, str) and len(val.strip()) == 0:
                return False, f"{key} is empty string"
        return True, "All required results are present and non-empty"

    # ----------------------------------------------------------------------
    # Test runner: discover all test_ methods
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)