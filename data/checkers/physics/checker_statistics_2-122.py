import sympy as sp
import math
from typing import Any, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits solutions for the 3-level identical particle problem.
    Checks partition functions and average energies for Fermi, Bose, Boltzmann statistics,
    plus the classical limit condition.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results dict safely (default None)
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Extract givens if available (for numerical checks)
        self.givens = getattr(solution_instance, 'givens', None) or {}
        # Also try direct attributes on the instance
        self.beta = self._get_given('beta', default=None)
        self.epsilon = self._get_given('epsilon', default=None)
        self.k = self._get_given('k', default=None)  # Boltzmann constant

        # Extract result quantities (use .get with None)
        self.Z_F = self.results.get('Z_F', None)
        self.E_F = self.results.get('E_F', None)
        self.Z_B = self.results.get('Z_B', None)
        self.E_B = self.results.get('E_B', None)
        self.Z_Boltz = self.results.get('Z_Boltz', None)
        self.E_Boltz = self.results.get('E_Boltz', None)
        self.cond = self.results.get('cond', None)

        # Pre-convert symbolic expressions (if they are strings or sympy objects)
        self.sym_Z_F = self._to_sympy(self.Z_F)
        self.sym_E_F = self._to_sympy(self.E_F)
        self.sym_Z_B = self._to_sympy(self.Z_B)
        self.sym_E_B = self._to_sympy(self.E_B)
        self.sym_Z_Boltz = self._to_sympy(self.Z_Boltz)
        self.sym_E_Boltz = self._to_sympy(self.E_Boltz)

    def _get_given(self, name: str, default: Any = None) -> Any:
        """Try to extract a numeric given from solution_instance or self.givens."""
        # First, try the givens dict
        if isinstance(self.givens, dict):
            val = self.givens.get(name)
            if val is not None:
                return val
        # Second, try direct attribute on the solution_instance
        if hasattr(self.givens, name):  # if givens is an object
            return getattr(self.givens, name)
        return default

    def _to_sympy(self, expr: Any) -> Optional[sp.Basic]:
        """Convert a possible string/sympy expression to a sympy expression.
        Returns None if conversion fails.
        """
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        if isinstance(expr, str):
            try:
                # First, try plain sympify (works for math expressions)
                return sp.sympify(expr)
            except (sp.SympifyError, TypeError, ValueError):
                pass
            # If it fails, attempt to clean LaTeX syntax
            clean = expr.replace('\\beta', 'beta').replace('\\varepsilon', 'epsilon')\
                         .replace('\\', '').replace('{', '').replace('}', '')
            # replace e^{-...} with exp(-...)
            clean = clean.replace('e^{-', 'exp(-').replace('}^{', '**')
            try:
                return sp.sympify(clean)
            except (sp.SympifyError, TypeError, ValueError):
                return None
        return None

    def get_numeric_beta_epsilon(self) -> Tuple[Optional[float], Optional[float]]:
        """Return numeric beta and epsilon if both are available."""
        beta = self.beta
        eps = self.epsilon
        if beta is None and hasattr(self, 'T') and self.T is not None and self.k is not None:
            beta = 1.0 / (self.k * self.T)
        if beta is None or eps is None:
            return None, None
        try:
            beta = float(beta)
            eps = float(eps)
        except (TypeError, ValueError):
            return None, None
        return beta, eps

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist."""
        keys = ['Z_F', 'E_F', 'Z_B', 'E_B', 'Z_Boltz', 'E_Boltz', 'cond']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present.")

    # ----------------------------------------------------------------------
    # Positivity and sign invariants
    # ----------------------------------------------------------------------
    def _test_invariant_positive(self, Z_sym, name: str) -> Tuple[bool, str]:
        """Generic: the partition function expression must be positive for positive beta, epsilon."""
        if Z_sym is None:
            return (False, f"SKIP: {name} not available for positivity check")
        beta, eps = sp.symbols('beta epsilon', positive=True)
        try:
            val = Z_sym.subs({beta: 1.0, eps: 1.0}).evalf()
            if val > 0:
                return (True, f"{name} = {val} > 0 for sample β=1, ε=1")
            else:
                return (False, f"{name} = {val} is not positive")
        except Exception:
            return (False, f"{name} could not be evaluated numerically")

    def test_invariant_Fermi_Z_positive(self) -> Tuple[bool, str]:
        return self._test_invariant_positive(self.sym_Z_F, "Z_F")

    def test_invariant_Bose_Z_positive(self) -> Tuple[bool, str]:
        return self._test_invariant_positive(self.sym_Z_B, "Z_B")

    def test_invariant_Boltz_Z_positive(self) -> Tuple[bool, str]:
        return self._test_invariant_positive(self.sym_Z_Boltz, "Z_Boltz")

    def test_invariant_energy_nonnegative(self) -> Tuple[bool, str]:
        """Check that energies expression is non-negative for positive beta, epsilon."""
        beta, eps = sp.symbols('beta epsilon', positive=True)
        for name, E_sym in [('E_F', self.sym_E_F), ('E_B', self.sym_E_B), ('E_Boltz', self.sym_E_Boltz)]:
            if E_sym is None:
                continue
            try:
                val = E_sym.subs({beta: 1.0, eps: 1.0}).evalf()
                if val < -1e-9:
                    return (False, f"{name} = {val} is negative for β=1, ε=1")
            except Exception:
                continue
        return (True, "All available energies are non-negative for sample point.")

    # ----------------------------------------------------------------------
    # Modeling checks: consistency of energy with derivative of partition function
    # ----------------------------------------------------------------------
    def _check_energy_derivative(self, Z_sym, E_sym, name_Z, name_E) -> Tuple[bool, str]:
        """Check that given energy equals -d(log Z)/dbeta."""
        if Z_sym is None or E_sym is None:
            return (False, f"SKIP: {name_Z} or {name_E} missing")
        beta, eps = sp.symbols('beta epsilon', positive=True)
        try:
            log_Z = sp.log(Z_sym)
            dlogZ = sp.diff(log_Z, beta)
            expected_E = -dlogZ
            diff = sp.simplify(expected_E - E_sym)
            # If symbolic simplification gives 0, it's exact
            if diff == 0:
                return (True, f"E = -d(log {name_Z})/dbeta holds symbolically.")
            # Otherwise, test numerically at a random point
            beta0, eps0 = 1.2, 0.8
            num_exp = float(expected_E.subs({beta: beta0, eps: eps0}).evalf())
            num_given = float(E_sym.subs({beta: beta0, eps: eps0}).evalf())
            if math.isclose(num_exp, num_given, rel_tol=1e-5, abs_tol=1e-5):
                return (True, f"E = -d(log {name_Z})/dbeta holds at β={beta0}, ε={eps0}.")
            else:
                return (False, f"Numerical mismatch: expected {num_exp}, got {num_given}.")
        except Exception as e:
            return (False, f"Error in derivative check: {e}")

    def test_modeling_Fermi_energy_derivative(self) -> Tuple[bool, str]:
        return self._check_energy_derivative(self.sym_Z_F, self.sym_E_F, 'Z_F', 'E_F')

    def test_modeling_Bose_energy_derivative(self) -> Tuple[bool, str]:
        return self._check_energy_derivative(self.sym_Z_B, self.sym_E_B, 'Z_B', 'E_B')

    def test_modeling_Boltz_energy_derivative(self) -> Tuple[bool, str]:
        return self._check_energy_derivative(self.sym_Z_Boltz, self.sym_E_Boltz, 'Z_Boltz', 'E_Boltz')

    # ----------------------------------------------------------------------
    # Low-T and high-T limits (numerical invariants)
    # ----------------------------------------------------------------------
    def _check_limit(self, Z_sym, E_sym, T, limit_Z, limit_E, name: str) -> Tuple[bool, str]:
        """Check partition function and energy at a given temperature.
        Here T is a string: 'high' (beta->0) or 'low' (beta->inf).
        limit_Z and limit_E are expected numeric limits (re-derived inside checker).
        """
        beta, eps = sp.symbols('beta epsilon', positive=True)
        if T == 'high':
            beta_val = 0.001
        elif T == 'low':
            beta_val = 1000.0
        else:
            return (False, f"Unknown temperature limit: {T}")
        eps_val = 1.0
        try:
            Z_val = float(Z_sym.subs({beta: beta_val, eps: eps_val}).evalf())
            E_val = float(E_sym.subs({beta: beta_val, eps: eps_val}).evalf())
            Z_ok = math.isclose(Z_val, limit_Z, rel_tol=1e-2, abs_tol=0.1)
            E_ok = math.isclose(E_val, limit_E, rel_tol=1e-2, abs_tol=0.1)
            msg = f"{name} at T={T}: Z={Z_val}, E={E_val}"
            if Z_ok and E_ok:
                return (True, msg)
            else:
                return (False, f"{msg} (expected Z≈{limit_Z}, E≈{limit_E})")
        except Exception:
            return (False, f"Cannot evaluate {name} at {T} limit")

    def test_invariant_Fermi_high_T_limit(self) -> Tuple[bool, str]:
        if self.sym_Z_F is None or self.sym_E_F is None:
            return (False, "SKIP: Z_F or E_F missing")
        return self._check_limit(self.sym_Z_F, self.sym_E_F, 'high', limit_Z=6.0, limit_E=1.5, name='Fermi')

    def test_invariant_Bose_high_T_limit(self) -> Tuple[bool, str]:
        if self.sym_Z_B is None or self.sym_E_B is None:
            return (False, "SKIP: Z_B or E_B missing")
        return self._check_limit(self.sym_Z_B, self.sym_E_B, 'high', limit_Z=10.0, limit_E=1.5, name='Bose')

    def test_invariant_Boltz_high_T_limit(self) -> Tuple[bool, str]:
        if self.sym_Z_Boltz is None or self.sym_E_Boltz is None:
            return (False, "SKIP: Z_Boltz or E_Boltz missing")
        return self._check_limit(self.sym_Z_Boltz, self.sym_E_Boltz, 'high', limit_Z=16.0, limit_E=1.5, name='Boltz')

    def test_invariant_Fermi_low_T_limit(self) -> Tuple[bool, str]:
        if self.sym_Z_F is None or self.sym_E_F is None:
            return (False, "SKIP: Z_F or E_F missing")
        return self._check_limit(self.sym_Z_F, self.sym_E_F, 'low', limit_Z=1.0, limit_E=0.0, name='Fermi')

    def test_invariant_Bose_low_T_limit(self) -> Tuple[bool, str]:
        if self.sym_Z_B is None or self.sym_E_B is None:
            return (False, "SKIP: Z_B or E_B missing")
        return self._check_limit(self.sym_Z_B, self.sym_E_B, 'low', limit_Z=3.0, limit_E=0.0, name='Bose')

    def test_invariant_Boltz_low_T_limit(self) -> Tuple[bool, str]:
        if self.sym_Z_Boltz is None or self.sym_E_Boltz is None:
            return (False, "SKIP: Z_Boltz or E_Boltz missing")
        return self._check_limit(self.sym_Z_Boltz, self.sym_E_Boltz, 'low', limit_Z=4.0, limit_E=0.0, name='Boltz')

    # ----------------------------------------------------------------------
    # Monotonicity: energy decreases as beta increases (higher T -> higher energy)
    # Each test explicitly perturbs beta by +10% from a baseline and compares.
    # ----------------------------------------------------------------------
    def test_monotonicity_energy_Fermi(self) -> Tuple[bool, str]:
        """Check that E_F decreases when beta increases (temperature decreases)."""
        if self.sym_E_F is None:
            return (False, "SKIP: E_F not available")
        beta, eps = sp.symbols('beta epsilon', positive=True)
        # Baseline: beta=1.0, epsilon=1.0
        beta0 = 1.0
        eps0 = 1.0
        try:
            baseline = float(self.sym_E_F.subs({beta: beta0, eps: eps0}).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate baseline E_F: {e}")
        # Perturb beta by +10% (to 1.1)
        beta_perturbed = beta0 * 1.1
        try:
            perturbed = float(self.sym_E_F.subs({beta: beta_perturbed, eps: eps0}).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate perturbed E_F: {e}")
        if perturbed < baseline - 1e-12:
            return (True, f"E_F: baseline β={beta0}, ε={eps0} -> E={baseline:.6f}; perturbed β={beta_perturbed} -> E={perturbed:.6f}; energy decreased as β increased (correct monotonicity).")
        elif math.isclose(perturbed, baseline, rel_tol=1e-10):
            return (True, "E_F: energy unchanged within tolerance (likely zero at these parameters).")
        else:
            return (False, f"E_F: energy increased when β increased (should decrease). Baseline={baseline:.6f}, perturbed={perturbed:.6f}")

    def test_monotonicity_energy_Bose(self) -> Tuple[bool, str]:
        """Check that E_B decreases when beta increases."""
        if self.sym_E_B is None:
            return (False, "SKIP: E_B not available")
        beta, eps = sp.symbols('beta epsilon', positive=True)
        beta0 = 1.0
        eps0 = 1.0
        try:
            baseline = float(self.sym_E_B.subs({beta: beta0, eps: eps0}).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate baseline E_B: {e}")
        beta_perturbed = beta0 * 1.1
        try:
            perturbed = float(self.sym_E_B.subs({beta: beta_perturbed, eps: eps0}).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate perturbed E_B: {e}")
        if perturbed < baseline - 1e-12:
            return (True, f"E_B: baseline β={beta0}, ε={eps0} -> E={baseline:.6f}; perturbed β={beta_perturbed} -> E={perturbed:.6f}; energy decreased as β increased (correct).")
        elif math.isclose(perturbed, baseline, rel_tol=1e-10):
            return (True, "E_B: energy unchanged within tolerance.")
        else:
            return (False, f"E_B: energy increased when β increased. Baseline={baseline:.6f}, perturbed={perturbed:.6f}")

    def test_monotonicity_energy_Boltz(self) -> Tuple[bool, str]:
        """Check that E_Boltz decreases when beta increases."""
        if self.sym_E_Boltz is None:
            return (False, "SKIP: E_Boltz not available")
        beta, eps = sp.symbols('beta epsilon', positive=True)
        beta0 = 1.0
        eps0 = 1.0
        try:
            baseline = float(self.sym_E_Boltz.subs({beta: beta0, eps: eps0}).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate baseline E_Boltz: {e}")
        beta_perturbed = beta0 * 1.1
        try:
            perturbed = float(self.sym_E_Boltz.subs({beta: beta_perturbed, eps: eps0}).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate perturbed E_Boltz: {e}")
        if perturbed < baseline - 1e-12:
            return (True, f"E_Boltz: baseline β={beta0}, ε={eps0} -> E={baseline:.6f}; perturbed β={beta_perturbed} -> E={perturbed:.6f}; energy decreased as β increased (correct).")
        elif math.isclose(perturbed, baseline, rel_tol=1e-10):
            return (True, "E_Boltz: energy unchanged within tolerance.")
        else:
            return (False, f"E_Boltz: energy increased when β increased. Baseline={baseline:.6f}, perturbed={perturbed:.6f}")

    # ----------------------------------------------------------------------
    # Modeling check: classical limit condition
    # ----------------------------------------------------------------------
    def test_modeling_condition_string(self) -> Tuple[bool, str]:
        if self.cond is None:
            return (False, "SKIP: condition string missing")
        if isinstance(self.cond, str) and ('N/V' in self.cond or 'density' in self.cond or 'h' in self.cond):
            return (True, f"Condition provided: {self.cond}")
        else:
            try:
                sym_cond = self._to_sympy(self.cond)
                if sym_cond is not None and sym_cond.free_symbols:
                    return (True, "Condition is a symbolic expression")
                else:
                    return (False, "Condition string not recognized as valid")
            except:
                return (False, "Condition not in expected form")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_nonnegative(self) -> Tuple[bool, str]:
        """Check that partition functions and energies are finite and non-negative at a sample point."""
        beta, eps = self.get_numeric_beta_epsilon()
        if beta is None or eps is None:
            return (False, "SKIP: need numeric beta and epsilon")
        expressions = {
            'Z_F': self.Z_F, 'E_F': self.E_F,
            'Z_B': self.Z_B, 'E_B': self.E_B,
            'Z_Boltz': self.Z_Boltz, 'E_Boltz': self.E_Boltz,
        }
        for name, expr in expressions.items():
            if expr is None:
                continue
            sym_expr = self._to_sympy(expr)
            if sym_expr is None:
                continue
            try:
                val = float(sym_expr.subs({sp.Symbol('beta'): beta, sp.Symbol('epsilon'): eps}).evalf())
            except Exception:
                continue
            if val < 0:
                return (False, f"{name} = {val} is negative")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
        return (True, "All available expressions yield finite non-negative values.")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)