import math
import sympy
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits a solution instance for the two-photon pair production problem.
    Checks: structural formula correctness, invariant mass threshold,
    positivity, monotonicity, and a weak numeric sanity for the minimum energy.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Extract known physical constants (if provided by solution) ---
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        self.m_e_c2 = getattr(solution_instance, 'm_e_c2', None)

        # --- Extract results dictionary safely ---
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.results = results
        self.beta = results.get('beta', None)
        self.E_min_formula = results.get('E_min_formula', None)
        self.E_min_numeric = results.get('E_min_numeric', None)

        # --- Predefine symbolic variables for formula checks ---
        self.sym_E, self.sym_eps, self.sym_m = sympy.symbols('E epsilon m')
        self.c = 1  # natural units (included for completeness)

    # ----------------------------------------------------------------
    # Helper to evaluate a formula string/sympy at given numeric values
    # ----------------------------------------------------------------
    def _eval_formula(self, formula, E_val, eps_val) -> float:
        """Evaluate a beta or E_min formula at given E and epsilon."""
        if isinstance(formula, str):
            expr = sympy.sympify(formula)
        else:
            expr = formula
        # map symbols to values
        subs_dict = {self.sym_E: E_val, self.sym_eps: eps_val}
        # also handle possible use of 'm' (mass) if needed
        if hasattr(self, 'sym_m') and self.m_e_c2 is not None:
            subs_dict[self.sym_m] = self.m_e_c2
        return float(expr.subs(subs_dict).evalf())

    # ----------------------------------------------------------------
    # Interface tests – required keys and types
    # ----------------------------------------------------------------
    def test_interface_results_keys(self) -> Tuple[bool, str]:
        """Check that the results dictionary contains the three required keys."""
        if self.results is None:
            return (False, "results is None")
        required = ['beta', 'E_min_formula', 'E_min_numeric']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_beta_type(self) -> Tuple[bool, str]:
        """beta should be a string or a sympy expression (not None)."""
        if self.beta is None:
            return (False, "SKIP: beta is None")
        if isinstance(self.beta, str):
            return (True, "beta is a string")
        if isinstance(self.beta, (sympy.Expr,)):
            return (True, "beta is a sympy expression")
        return (False, f"Unexpected type for beta: {type(self.beta).__name__}")

    def test_interface_epsilon_is_positive(self) -> Tuple[bool, str]:
        """If epsilon is provided, check it is positive."""
        if self.epsilon is None:
            return (False, "SKIP: epsilon not provided")
        if self.epsilon > 0:
            return (True, f"epsilon = {self.epsilon} > 0")
        return (False, f"epsilon is not positive: {self.epsilon}")

    # ----------------------------------------------------------------
    # Modeling tests – formula correctness via symbolic algebra
    # ----------------------------------------------------------------
    def test_modeling_beta_formula(self) -> Tuple[bool, str]:
        """
        Check that beta matches the analytical expression (E-ε)/(E+ε).
        Uses symbolic manipulation; requires beta to be parseable.
        """
        if self.beta is None:
            return (False, "SKIP: beta not available")
        try:
            # Convert beta to sympy expression if it's a string
            if isinstance(self.beta, str):
                beta_expr = sympy.sympify(self.beta)
            else:
                beta_expr = self.beta
            # Expected form
            expected = (self.sym_E - self.sym_eps) / (self.sym_E + self.sym_eps)
            # Check algebraic equivalence by subtracting and simplifying
            diff = sympy.simplify(beta_expr - expected)
            if diff == 0:
                return (True, "Symbolic beta matches (E-ε)/(E+ε)")
            else:
                # try relaxed simplification (e.g., factor)
                diff2 = sympy.simplify(beta_expr - expected)
                if sympy.simplify(diff2 - 0) == 0:
                    return (True, "Symbolic beta equivalent after simplification")
                return (False, f"Symbolic difference not zero: {diff}")
        except Exception as e:
            return (False, f"Error checking beta formula: {e}")

    def test_modeling_E_min_formula(self) -> Tuple[bool, str]:
        """
        Check E_min_formula equals m_e_c2^2 / epsilon symbolically.
        """
        if self.E_min_formula is None:
            return (False, "SKIP: E_min_formula not available")
        try:
            if isinstance(self.E_min_formula, str):
                formula_expr = sympy.sympify(self.E_min_formula)
            else:
                formula_expr = self.E_min_formula
            expected = self.sym_m**2 / self.sym_eps
            diff = sympy.simplify(formula_expr - expected)
            if diff == 0:
                return (True, "E_min_formula matches m^2/ε")
            else:
                return (False, f"Symbolic difference: {diff}")
        except Exception as e:
            return (False, f"Error checking E_min formula: {e}")

    # ----------------------------------------------------------------
    # Invariant tests – physics constraints
    # ----------------------------------------------------------------
    def test_invariant_beta_less_than_one(self) -> Tuple[bool, str]:
        """
        For positive E, ε the velocity β is always less than 1.
        We can verify this using the formula property.
        """
        if self.beta is None:
            return (False, "SKIP: beta not available")
        try:
            # Use numeric proof: choose arbitrary positive E, ε
            test_E = 100.0
            test_eps = 10.0
            beta_val = self._eval_formula(self.beta, test_E, test_eps)
            if beta_val < 1.0:
                return (True, f"beta = {beta_val:.4f} < 1 (for E={test_E}, ε={test_eps})")
            else:
                return (False, f"beta is not < 1: {beta_val}")
        except Exception as e:
            return (False, f"Could not evaluate beta < 1: {e}")

    def test_invariant_energy_sign(self) -> Tuple[bool, str]:
        """
        Both photon energies and the minimum E must be positive.
        Check any provided numeric values.
        """
        sign_ok = True
        msgs = []
        if self.epsilon is not None and self.epsilon <= 0:
            sign_ok = False
            msgs.append(f"epsilon={self.epsilon} non-positive")
        if self.E_min_numeric is not None and self.E_min_numeric <= 0:
            sign_ok = False
            msgs.append(f"E_min_numeric={self.E_min_numeric} non-positive")
        if sign_ok:
            return (True, "All provided energies are positive")
        return (False, "; ".join(msgs))

    def test_invariant_threshold_check(self) -> Tuple[bool, str]:
        """
        If E_min_numeric and epsilon are known, verify the invariant mass condition:
        s = 4 * E_min * epsilon ≈ (2*m_e c^2)^2.
        Uses math.isclose with 1% relative tolerance.
        """
        if self.E_min_numeric is None or self.epsilon is None or self.m_e_c2 is None:
            return (False, "SKIP: E_min_numeric, epsilon, and m_e_c2 required")
        try:
            E = float(self.E_min_numeric)
            eps = float(self.epsilon)
            m = float(self.m_e_c2)
            # Invariant mass squared s = 4 * E * epsilon
            s_calc = 4 * E * eps
            # Threshold value (assuming natural units c=1)
            s_threshold = (2 * m) ** 2
            if math.isclose(s_calc, s_threshold, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"s = {s_calc:.2e} ≈ {s_threshold:.2e} within 1%")
            else:
                return (False, f"s = {s_calc:.2e} not close to threshold {s_threshold:.2e}")
        except Exception as e:
            return (False, f"Error in threshold check: {e}")

    # ----------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------
    def test_monotonicity_beta_with_E(self) -> Tuple[bool, str]:
        """
        For fixed epsilon, beta = (E-ε)/(E+ε) increases with E.
        Perturb E upward by 10% and check the sign of change.
        """
        if self.beta is None:
            return (False, "SKIP: beta not available")
        try:
            # Use generic positive numbers (epsilon fixed)
            eps = 50.0
            E_base = 100.0
            E_pert = 110.0  # +10%
            beta_base = self._eval_formula(self.beta, E_base, eps)
            beta_pert = self._eval_formula(self.beta, E_pert, eps)
            if beta_pert > beta_base:
                return (True, f"beta increased from {beta_base:.4f} to {beta_pert:.4f} when E increased from {E_base} to {E_pert}")
            else:
                return (False, f"beta did NOT increase: base={beta_base:.4f}, pert={beta_pert:.4f}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    def test_monotonicity_beta_with_epsilon(self) -> Tuple[bool, str]:
        """
        For fixed E, beta = (E-ε)/(E+ε) decreases with ε.
        Perturb epsilon upward by 10% and check the sign of change.
        """
        if self.beta is None:
            return (False, "SKIP: beta not available")
        try:
            E = 100.0
            eps_base = 50.0
            eps_pert = 55.0  # +10%
            beta_base = self._eval_formula(self.beta, E, eps_base)
            beta_pert = self._eval_formula(self.beta, E, eps_pert)
            if beta_pert < beta_base:
                return (True, f"beta decreased from {beta_base:.4f} to {beta_pert:.4f} when ε increased from {eps_base} to {eps_pert}")
            else:
                return (False, f"beta did NOT decrease: base={beta_base:.4f}, pert={beta_pert:.4f}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    def test_monotonicity_E_min_with_epsilon(self) -> Tuple[bool, str]:
        """
        For E_min = m_e_c2^2 / epsilon, increasing epsilon decreases E_min.
        Use the formula to compute at two epsilon values.
        """
        if self.E_min_formula is None or self.m_e_c2 is None:
            return (False, "SKIP: need E_min_formula and m_e_c2")
        try:
            # We need to evaluate the formula at two epsilon values
            eps_base = 1.0
            eps_pert = 1.5  # 50% increase
            # Create evaluation dictionary: we know m_e_c2 from solution
            m = float(self.m_e_c2)
            # Evaluate formula (it may depend on m and epsilon)
            if isinstance(self.E_min_formula, str):
                expr = sympy.sympify(self.E_min_formula)
            else:
                expr = self.E_min_formula
            subs_base = {self.sym_m: m, self.sym_eps: eps_base}
            subs_pert = {self.sym_m: m, self.sym_eps: eps_pert}
            E_base = float(expr.subs(subs_base).evalf())
            E_pert = float(expr.subs(subs_pert).evalf())
            if E_pert < E_base:
                return (True, f"E_min decreased from {E_base:.2e} to {E_pert:.2e} when ε increased from {eps_base} to {eps_pert}")
            else:
                return (False, f"E_min did NOT decrease: base={E_base:.2e}, pert={E_pert:.2e}")
        except Exception as e:
            return (False, f"Error in E_min monotonicity test: {e}")

    # ----------------------------------------------------------------
    # Answer sanity test – weak, no isclose, no re-derivation
    # ----------------------------------------------------------------
    def test_answer_sanity_E_min_numeric(self) -> Tuple[bool, str]:
        """
        Weak sanity: E_min_numeric should be in the range 1e10 to 1e12 eV
        (expected ~261 GeV = 2.61e11 eV).
        """
        if self.E_min_numeric is None:
            return (False, "SKIP: E_min_numeric not available")
        try:
            val = float(self.E_min_numeric)
            if val <= 0:
                return (False, f"E_min_numeric <= 0: {val}")
            if 1e10 < val < 1e12:
                return (True, f"E_min_numeric = {val:.2e} in plausible range (10^10 – 10^12 eV)")
            else:
                return (False, f"E_min_numeric = {val:.2e} out of plausible range")
        except Exception as e:
            return (False, f"Could not process E_min_numeric: {e}")

    # ----------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, execute them safely, collect results.
        Returns (all_pass, list_of_logs).
        """
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
        return all_pass, logs