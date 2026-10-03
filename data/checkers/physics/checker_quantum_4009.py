import math
import sympy as sp
from sympy.parsing.latex import parse_latex
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics checker for the time reversal/electron magnetic moment problem.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract required results from the solution instance.
        Use safe defaults: None for missing keys.
        """
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        self.psi_T: Optional[str] = self.results.get('psi_T', None)
        self.H_hat: Optional[str] = self.results.get('H_hat', None)
        self.dmu_dt: Optional[str] = self.results.get('dmu_dt', None)

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _check_contains(self, substring: str, full_string: Optional[str]) -> bool:
        if full_string is None:
            return False
        return substring in full_string

    def _check_contains_any(self, candidates: List[str], full_string: Optional[str]) -> bool:
        if full_string is None:
            return False
        for cand in candidates:
            if cand in full_string:
                return True
        return False

    def _try_parse_latex(self, latex: str) -> Optional[sp.Expr]:
        """Try to parse a LaTeX string into a sympy expression. Return None if fails."""
        try:
            # Replace common LaTeX commands with sympy-friendly forms
            expr = parse_latex(latex)
            return sp.simplify(expr)
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_psi_T_present(self) -> Tuple[bool, str]:
        if self.psi_T is None:
            return (False, "SKIP: psi_T not found in results")
        if not isinstance(self.psi_T, str):
            return (False, f"psi_T is not a string, got {type(self.psi_T)}")
        return (True, "psi_T is present and is a string")

    def test_interface_H_hat_present(self) -> Tuple[bool, str]:
        if self.H_hat is None:
            return (False, "SKIP: H_hat not found in results")
        if not isinstance(self.H_hat, str):
            return (False, f"H_hat is not a string, got {type(self.H_hat)}")
        return (True, "H_hat is present and is a string")

    def test_interface_dmu_dt_present(self) -> Tuple[bool, str]:
        if self.dmu_dt is None:
            return (False, "SKIP: dmu_dt not found in results")
        if not isinstance(self.dmu_dt, str):
            return (False, f"dmu_dt is not a string, got {type(self.dmu_dt)}")
        return (True, "dmu_dt is present and is a string")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based sanity)
    # ----------------------------------------------------------------------
    def test_invariant_time_reversal_antiunitary(self) -> Tuple[bool, str]:
        """Time reversal involves complex conjugation and time reversal."""
        if self.psi_T is None:
            return (False, "SKIP: psi_T not available")
        if self._check_contains("^*", self.psi_T) and self._check_contains("-t", self.psi_T):
            return (True, "psi_T includes complex conjugate and time reversal")
        return (False, f"psi_T does not contain both complex conjugate and reversed time: {self.psi_T}")

    def test_invariant_spin_commutation(self) -> Tuple[bool, str]:
        """Hamiltonian must contain a Pauli matrix (sigma_z) for spin-1/2."""
        if self.H_hat is None:
            return (False, "SKIP: H_hat not available")
        if self._check_contains("sigma_z", self.H_hat) or self._check_contains("\\sigma_z", self.H_hat):
            return (True, "H_hat contains sigma_z (implied spin commutation)")
        return (False, f"H_hat does not contain sigma_z: {self.H_hat}")

    def test_invariant_heisenberg_structure(self) -> Tuple[bool, str]:
        """dmu/dt must contain a cross product structure."""
        if self.dmu_dt is None:
            return (False, "SKIP: dmu_dt not available")
        cross_indicators = ['\\times', 'cross', '×']
        if self._check_contains_any(cross_indicators, self.dmu_dt):
            return (True, "dmu_dt contains cross product notation")
        return (False, f"dmu_dt does not contain cross product: {self.dmu_dt}")

    # ----------------------------------------------------------------------
    # Modeling/trace tests
    # ----------------------------------------------------------------------
    def test_modeling_psi_T_form(self) -> Tuple[bool, str]:
        """Time-reversed wavefunction should be psi^*(-t)."""
        if self.psi_T is None:
            return (False, "SKIP: psi_T not available")
        # Accept variations: psi^* (-t) or psi^*(-t)
        if self._check_contains("psi^* (-t)", self.psi_T) or \
           self._check_contains("psi^*(-t)", self.psi_T):
            return (True, "psi_T matches expected pattern")
        if self._check_contains("^*", self.psi_T) and self._check_contains("-t", self.psi_T):
            return (True, "psi_T contains complex conjugate and time reversal")
        return (False, f"psi_T does not match expected form: {self.psi_T}")

    def test_modeling_H_hat_form(self) -> Tuple[bool, str]:
        """Hamiltonian should be (e*hbar/(2*m*c))*sigma_z*H_z."""
        if self.H_hat is None:
            return (False, "SKIP: H_hat not available")
        checks = []
        if self._check_contains("sigma_z", self.H_hat) or self._check_contains("\\sigma_z", self.H_hat):
            checks.append("sigma_z")
        else:
            return (False, f"Missing sigma_z in H_hat: {self.H_hat}")
        if self._check_contains("H_z", self.H_hat):
            checks.append("H_z")
        else:
            return (False, f"Missing H_z in H_hat: {self.H_hat}")
        # Check factor e*hbar/(2mc)
        factor_variants = ["e/(2mc)", "e/(2 m c)", "e\\hbar/(2mc)", "\\frac{e\\hbar}{2mc}",
                          "\\frac{e\\hbar}{2 m c}"]
        if self._check_contains_any(factor_variants, self.H_hat):
            checks.append("factor e*hbar/(2mc)")
        else:
            # Accept if contains e, hbar, 2, m, c in some order
            if (self._check_contains("e", self.H_hat) and
                self._check_contains("hbar", self.H_hat) and
                self._check_contains("2", self.H_hat) and
                self._check_contains("m", self.H_hat) and
                self._check_contains("c", self.H_hat)):
                checks.append("factor e*hbar/(2*m*c)")
            else:
                return (False, f"Cannot identify factor in H_hat: {self.H_hat}")
        return (True, "H_hat contains " + ", ".join(checks))

    def test_modeling_dmu_dt_form(self) -> Tuple[bool, str]:
        """dmu/dt should be (e/(mc)) * H × mu."""
        if self.dmu_dt is None:
            return (False, "SKIP: dmu_dt not available")
        checks = []
        # Cross product
        if self._check_contains_any(["\\times", "×", "cross"], self.dmu_dt):
            checks.append("cross product")
        else:
            return (False, f"Missing cross product in dmu_dt: {self.dmu_dt}")
        # Factor e/(mc)
        factor_variants = ["e/(m*c)", "e/(mc)", "\\frac{e}{mc}", "\\frac{e}{m c}"]
        if self._check_contains_any(factor_variants, self.dmu_dt):
            checks.append("factor e/(mc)")
        else:
            if (self._check_contains("e", self.dmu_dt) and
                self._check_contains("m", self.dmu_dt) and
                self._check_contains("c", self.dmu_dt)):
                checks.append("factor e,m,c present")
            else:
                return (False, f"Cannot identify factor in dmu_dt: {self.dmu_dt}")
        # Must contain magnetic field symbol and mu
        if self._check_contains_any(["H", "\\mathbf{H}", "B", "\\mathbf{B}"], self.dmu_dt):
            checks.append("magnetic field")
        else:
            return (False, f"Missing magnetic field symbol in dmu_dt: {self.dmu_dt}")
        if self._check_contains_any(["\\mu", "mu", "μ"], self.dmu_dt):
            checks.append("mu")
        else:
            return (False, f"Missing mu symbol in dmu_dt: {self.dmu_dt}")
        return (True, "dmu_dt contains " + ", ".join(checks))

    # ----------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation and comparison)
    # ----------------------------------------------------------------------
    def test_monotonicity_H_hat_linear_in_H_z(self) -> Tuple[bool, str]:
        """
        The Hamiltonian H_hat is proportional to H_z.
        Perturb H_z by a factor of 2; the expression should scale by 2.
        We parse the LaTeX expression and substitute.
        """
        if self.H_hat is None:
            return (False, "SKIP: H_hat not available")
        expr = self._try_parse_latex(self.H_hat)
        if expr is None:
            return (False, "SKIP: Could not parse H_hat as LaTeX")
        # Identify symbols: there should be a symbol representing H_z
        free_symbols = list(expr.free_symbols)
        # Look for symbol that looks like H_z (maybe a string "H_z")
        H_z_sym = None
        for s in free_symbols:
            name = str(s)
            if "H_z" in name or "H_{z}" in name or "Hz" in name:
                H_z_sym = s
                break
        if H_z_sym is None:
            return (False, "SKIP: H_z symbol not found in parsed expression")
        # Baseline: evaluate at H_z = 1 (symbolically substitute)
        baseline = sp.simplify(expr.subs(H_z_sym, 1))
        # Perturbation: H_z = 2
        perturbed = sp.simplify(expr.subs(H_z_sym, 2))
        # Check if perturbed == 2 * baseline
        if sp.simplify(perturbed - 2*baseline) == 0:
            return (True, "H_hat scales linearly with H_z (2*H_z gives 2*original)")
        else:
            return (False, f"H_hat does not scale properly: baseline={baseline}, perturbed={perturbed}")

    def test_monotonicity_dmu_dt_linear_in_H(self) -> Tuple[bool, str]:
        """
        dmu/dt is proportional to H (the magnetic field vector).
        Perturb a generic magnitude factor (like replacing H with 2*H vector).
        Since it's a vector cross product, we need to parse the expression.
        We'll try to see if it contains a scalar coefficient; if not, we check
        that the expression is homogeneous of degree 1 in H symbols.
        """
        if self.dmu_dt is None:
            return (False, "SKIP: dmu_dt not available")
        expr = self._try_parse_latex(self.dmu_dt)
        if expr is None:
            return (False, "SKIP: Could not parse dmu_dt as LaTeX")
        # Find symbols that might represent magnetic field components or generic H
        free_symbols = list(expr.free_symbols)
        # Look for any symbol containing 'H' or 'B' (case matters)
        H_symbols = [s for s in free_symbols if 'H' in str(s) or 'B' in str(s)]
        if not H_symbols:
            return (False, "SKIP: No magnetic field symbols found in parsed expression")
        # Choose the first one as representative. We'll substitute it with 2*itself
        # but careful: the expression may contain H_y, H_z etc. We'll pick one and double it.
        H_sym = H_symbols[0]
        baseline = sp.simplify(expr)
        perturbed = sp.simplify(expr.subs(H_sym, 2*H_sym))
        # Check if perturbed == 2 * baseline (assuming linear in that symbol)
        # This may fail if the expression has cross terms; but for a single component it should work.
        # However, the expression may contain e/(m*c) as a factor. That's fine.
        if sp.simplify(perturbed - 2*baseline) == 0:
            return (True, f"dmu_dt is linear in {H_sym} (doubling symbol gives double)")
        else:
            # Try a different approach: perturb all H-like symbols by factor
            # This is safer: multiply each H symbol by a constant λ and check homogeneity.
            lambda_sym = sp.Symbol('lambda', real=True)
            substituted = expr
            for s in H_symbols:
                substituted = substituted.subs(s, lambda_sym * s)
            # Check if substituted equals lambda_sym * expr (up to factors like constant)
            # Better: subtract lambda_sym*expr and simplify; should be zero if homogeneous degree 1.
            diff = sp.simplify(substituted - lambda_sym * expr)
            if diff == 0:
                return (True, "dmu_dt is homogeneous of degree 1 in magnetic field symbols")
            else:
                return (False, f"dmu_dt does not scale linearly with magnetic field: difference={diff}")

    # ----------------------------------------------------------------------
    # Answer sanity test (weak, at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_nonempty(self) -> Tuple[bool, str]:
        """Very weak sanity: all three result strings are non-empty."""
        if self.psi_T is None or self.H_hat is None or self.dmu_dt is None:
            return (False, "SKIP: one or more results missing")
        if len(self.psi_T.strip()) == 0:
            return (False, "psi_T is an empty string")
        if len(self.H_hat.strip()) == 0:
            return (False, "H_hat is an empty string")
        if len(self.dmu_dt.strip()) == 0:
            return (False, "dmu_dt is an empty string")
        return (True, "All three result strings are non-empty")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass: bool = True
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