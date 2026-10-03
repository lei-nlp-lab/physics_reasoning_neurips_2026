import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Checks the physics correctness of a solution to the cylindrical capacitor problem.
    """
    
    EPSILON0 = 8.854187817e-12  # F/m
    
    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if not isinstance(self.results, dict):
            self.results = {}
        
        # Extract givens dictionary safely
        self.givens = getattr(solution_instance, 'givens', None)
        if not isinstance(self.givens, dict):
            self.givens = {}
        
        # Extract required results — use None defaults exclusively
        self.a_opt_V = self.results.get('a_opt_V', None)
        self.a_opt_U = self.results.get('a_opt_U', None)
        self.V_max_A = self.results.get('V_max_A', None)
        self.V_max_B = self.results.get('V_max_B', None)
        
        # Extract givens — no numeric defaults
        self.R_out = self.givens.get('R_out', None)
        self.E_break = self.givens.get('E_break', None)
        
        # Also check if solution_instance has these as attributes
        if self.R_out is None:
            self.R_out = getattr(solution_instance, 'R_out', None)
        if self.E_break is None:
            self.E_break = getattr(solution_instance, 'E_break', None)
        if self.a_opt_V is None:
            self.a_opt_V = getattr(solution_instance, 'a_opt_V', None)
        if self.a_opt_U is None:
            self.a_opt_U = getattr(solution_instance, 'a_opt_U', None)
        if self.V_max_A is None:
            self.V_max_A = getattr(solution_instance, 'V_max_A', None)
        if self.V_max_B is None:
            self.V_max_B = getattr(solution_instance, 'V_max_B', None)
    
    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        if not isinstance(self.results, dict):
            return (False, "SKIP: results is not a dict")
        required = ['a_opt_V', 'a_opt_U', 'V_max_A', 'V_max_B']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")
    
    def test_interface_key_types(self) -> Tuple[bool, str]:
        if not isinstance(self.results, dict):
            return (False, "SKIP: results is not a dict")
        numeric_keys = ['a_opt_V', 'a_opt_U', 'V_max_A', 'V_max_B']
        for k in numeric_keys:
            val = self.results.get(k, None)
            if val is None:
                return (False, f"{k} is None")
            if not isinstance(val, (int, float, complex)):
                return (False, f"{k} is not numeric, got {type(val).__name__}")
        return (True, "All required keys are numeric")
    
    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_V_formula(self) -> Tuple[bool, str]:
        """Check V_max_A = E_break * a_opt_V * ln(R_out / a_opt_V)"""
        if any(x is None for x in [self.a_opt_V, self.V_max_A, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_V, V_max_A, R_out, or E_break")
        expected = self.E_break * self.a_opt_V * math.log(self.R_out / self.a_opt_V)
        if math.isclose(self.V_max_A, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V_max_A = {self.V_max_A}, derived = {expected}")
        else:
            return (False, f"V_max_A = {self.V_max_A}, derived = {expected}, mismatch")
    
    def test_invariant_U_derived_V(self) -> Tuple[bool, str]:
        """Check V_max_B = E_break * a_opt_U * ln(R_out / a_opt_U)"""
        if any(x is None for x in [self.a_opt_U, self.V_max_B, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_U, V_max_B, R_out, or E_break")
        expected = self.E_break * self.a_opt_U * math.log(self.R_out / self.a_opt_U)
        if math.isclose(self.V_max_B, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V_max_B = {self.V_max_B}, derived = {expected}")
        else:
            return (False, f"V_max_B = {self.V_max_B}, derived = {expected}, mismatch")
    
    def test_invariant_derivative_V_zero(self) -> Tuple[bool, str]:
        """Check dV/da at a_opt_V is near zero using central difference."""
        if any(x is None for x in [self.a_opt_V, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_V, R_out, or E_break")
        a = self.a_opt_V
        h = 1e-6 * a if a != 0 else 1e-10
        V_plus = self.E_break * (a + h) * math.log(self.R_out / (a + h))
        V_minus = self.E_break * (a - h) * math.log(self.R_out / (a - h))
        deriv = (V_plus - V_minus) / (2 * h)
        if math.isclose(deriv, 0.0, abs_tol=1e-2):
            return (True, f"dV/da at a_opt_V ≈ {deriv}")
        else:
            return (False, f"dV/da = {deriv}, expected near 0")
    
    def test_invariant_derivative_U_zero(self) -> Tuple[bool, str]:
        """Check dU/da at a_opt_U is near zero."""
        if any(x is None for x in [self.a_opt_U, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_U, R_out, or E_break")
        a = self.a_opt_U
        h = 1e-6 * a if a != 0 else 1e-10
        pi = math.pi
        eps0 = self.EPSILON0
        U_plus = pi * eps0 * self.E_break**2 * (a + h)**2 * math.log(self.R_out / (a + h))
        U_minus = pi * eps0 * self.E_break**2 * (a - h)**2 * math.log(self.R_out / (a - h))
        deriv = (U_plus - U_minus) / (2 * h)
        if math.isclose(deriv, 0.0, abs_tol=1e-2):
            return (True, f"dU/da at a_opt_U ≈ {deriv}")
        else:
            return (False, f"dU/da = {deriv}, expected near 0")
    
    # ------------------------------------------------------------------
    # Modeling / Trace tests
    # ------------------------------------------------------------------
    def test_modeling_optima_order(self) -> Tuple[bool, str]:
        """Check a_opt_V < a_opt_U (1/e < 1/sqrt(e) for e>1)."""
        if any(x is None for x in [self.a_opt_V, self.a_opt_U]):
            return (False, "SKIP: missing a_opt_V or a_opt_U")
        if self.a_opt_V < self.a_opt_U:
            return (True, f"a_opt_V = {self.a_opt_V} < a_opt_U = {self.a_opt_U}")
        else:
            return (False, f"Expected a_opt_V < a_opt_U, got {self.a_opt_V} >= {self.a_opt_U}")
    
    def test_modeling_V_expression_symbolic(self) -> Tuple[bool, str]:
        """Symbolic check that derivative of V gives a_opt = R_out/e."""
        if self.R_out is None or self.a_opt_V is None:
            return (False, "SKIP: missing R_out or a_opt_V")
        try:
            a = sp.symbols('a', positive=True)
            R_sym = sp.symbols('R_out', positive=True)
            V_expr = a * sp.log(R_sym / a)
            deriv = sp.diff(V_expr, a)
            sol = sp.solve(sp.simplify(deriv), a)
            expected = R_sym / sp.E
            if any(sp.simplify(s - expected) == 0 for s in sol):
                return (True, "Symbolic derivative gives a_opt = R_out/e")
            else:
                return (False, f"Symbolic solutions {sol} do not match R_out/e")
        except Exception as e:
            return (False, f"Symbolic check failed: {e}")
    
    def test_modeling_U_expression_symbolic(self) -> Tuple[bool, str]:
        """Symbolic check that derivative of energy gives a_opt = R_out/sqrt(e)."""
        if self.R_out is None or self.a_opt_U is None:
            return (False, "SKIP: missing R_out or a_opt_U")
        try:
            a = sp.symbols('a', positive=True)
            R_sym = sp.symbols('R_out', positive=True)
            U_expr = a**2 * sp.log(R_sym / a)
            deriv = sp.diff(U_expr, a)
            sol = sp.solve(sp.simplify(deriv), a)
            expected = R_sym / sp.sqrt(sp.E)
            if any(sp.simplify(s - expected) == 0 for s in sol):
                return (True, "Symbolic derivative gives a_opt = R_out/sqrt(e)")
            else:
                return (False, f"Symbolic solutions {sol} do not match R_out/sqrt(e)")
        except Exception as e:
            return (False, f"Symbolic check failed: {e}")
    
    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # Each test selects a baseline input, applies a small relative perturbation,
    # and compares the output to the baseline output.
    # ------------------------------------------------------------------
    def test_monotonicity_V_increasing_before_opt(self) -> Tuple[bool, str]:
        """Before the optimum, increasing a by +1% should increase V."""
        if any(x is None for x in [self.a_opt_V, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_V, R_out, or E_break")
        # Choose a point well before the optimum
        a_base = 0.5 * self.a_opt_V
        # Apply a small relative perturbation: +1%
        perturbation = 0.01  # 1%
        a_perturbed = a_base * (1 + perturbation)
        V_base = self.E_break * a_base * math.log(self.R_out / a_base)
        V_pert = self.E_break * a_perturbed * math.log(self.R_out / a_perturbed)
        # Since we are before the optimum, larger a gives larger V
        if V_pert > V_base - 1e-12:
            return (True, f"V(base)={V_base}, V(+1%)={V_pert}, V increased")
        else:
            return (False, f"V(base)={V_base}, V(+1%)={V_pert}, V did not increase")
    
    def test_monotonicity_V_decreasing_after_opt(self) -> Tuple[bool, str]:
        """After the optimum, increasing a by +1% should decrease V."""
        if any(x is None for x in [self.a_opt_V, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_V, R_out, or E_break")
        # Choose a point after the optimum, but stay below R_out
        a_base = 1.5 * self.a_opt_V
        if a_base >= self.R_out:
            a_base = 0.9 * self.R_out
        perturbation = 0.01
        a_perturbed = a_base * (1 + perturbation)
        V_base = self.E_break * a_base * math.log(self.R_out / a_base)
        V_pert = self.E_break * a_perturbed * math.log(self.R_out / a_perturbed)
        if V_pert < V_base + 1e-12:
            return (True, f"V(base)={V_base}, V(+1%)={V_pert}, V decreased")
        else:
            return (False, f"V(base)={V_base}, V(+1%)={V_pert}, V did not decrease")
    
    def test_monotonicity_U_increasing_before_opt(self) -> Tuple[bool, str]:
        """Before the optimum, increasing a by +1% should increase U."""
        if any(x is None for x in [self.a_opt_U, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_U, R_out, or E_break")
        a_base = 0.5 * self.a_opt_U
        perturbation = 0.01
        a_perturbed = a_base * (1 + perturbation)
        pi = math.pi
        eps0 = self.EPSILON0
        U_base = pi * eps0 * self.E_break**2 * a_base**2 * math.log(self.R_out / a_base)
        U_pert = pi * eps0 * self.E_break**2 * a_perturbed**2 * math.log(self.R_out / a_perturbed)
        if U_pert > U_base - 1e-20:
            return (True, f"U(base)={U_base}, U(+1%)={U_pert}, U increased")
        else:
            return (False, f"U(base)={U_base}, U(+1%)={U_pert}, U did not increase")
    
    def test_monotonicity_U_decreasing_after_opt(self) -> Tuple[bool, str]:
        """After the optimum, increasing a by +1% should decrease U."""
        if any(x is None for x in [self.a_opt_U, self.R_out, self.E_break]):
            return (False, "SKIP: missing a_opt_U, R_out, or E_break")
        a_base = 1.5 * self.a_opt_U
        if a_base >= self.R_out:
            a_base = 0.9 * self.R_out
        perturbation = 0.01
        a_perturbed = a_base * (1 + perturbation)
        pi = math.pi
        eps0 = self.EPSILON0
        U_base = pi * eps0 * self.E_break**2 * a_base**2 * math.log(self.R_out / a_base)
        U_pert = pi * eps0 * self.E_break**2 * a_perturbed**2 * math.log(self.R_out / a_perturbed)
        if U_pert < U_base + 1e-20:
            return (True, f"U(base)={U_base}, U(+1%)={U_pert}, U decreased")
        else:
            return (False, f"U(base)={U_base}, U(+1%)={U_pert}, U did not decrease")
    
    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, no re‑derivation, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """Check V_max_A and V_max_B are finite and positive (no math.isclose)."""
        issues = []
        for name, val in [('V_max_A', self.V_max_A), ('V_max_B', self.V_max_B)]:
            if val is None:
                issues.append(f"{name} is None")
            elif not isinstance(val, (int, float)):
                issues.append(f"{name} is not numeric")
            elif not math.isfinite(val):
                issues.append(f"{name} is not finite")
            elif val <= 0:
                issues.append(f"{name} is not positive")
        if issues:
            return (False, "; ".join(issues))
        return (True, "V_max_A and V_max_B are positive and finite")
    
    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                msg = f"CRASHED with {type(e).__name__}: {e}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)