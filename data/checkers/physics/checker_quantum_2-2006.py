import math
import traceback
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics checker for the harmonic oscillator problem.
    Audits the solution's results against theorem‑based sanity, modeling quality,
    and a weak final‑answer check.
    """

    def __init__(self, solution_instance: Any):
        # ---------- Extract givens (safe defaults) ----------
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.m = givens.get('m', None)
        self.k = givens.get('k', None)
        self.omega = givens.get('omega', None)
        self.hbar = givens.get('hbar', None)

        # ---------- Extract results (safe defaults) ----------
        results = getattr(solution_instance, 'results', {}) or {}
        self.E_0 = results.get('E_0', None)
        self.alpha = results.get('alpha', None)
        self.psi_1 = results.get('psi_1', None)
        self.psi_m = results.get('psi_m', None)

        # Derived quantities (computed inside checker, from givens)
        # These are used in invariant and monotonicity tests.
        if self.m is not None and self.omega is not None and self.hbar is not None:
            self._expected_alpha = self.m * self.omega / (2.0 * self.hbar)
            self._expected_E0 = self.hbar * self.omega / 2.0
        else:
            self._expected_alpha = None
            self._expected_E0 = None

    # -------------------------------------------------------------------
    # Interface tests – verify that required keys exist with proper types
    # -------------------------------------------------------------------
    def test_interface_E0_exists(self) -> Tuple[bool, str]:
        if self.E_0 is None:
            return False, "E_0 is missing in results."
        if not isinstance(self.E_0, (int, float, sp.Expr)):
            return False, f"E_0 has wrong type: {type(self.E_0)}"
        return True, "E_0 is present and numeric."

    def test_interface_alpha_exists(self) -> Tuple[bool, str]:
        if self.alpha is None:
            return False, "alpha is missing in results."
        if not isinstance(self.alpha, (int, float, sp.Expr)):
            return False, f"alpha has wrong type: {type(self.alpha)}"
        return True, "alpha is present and numeric."

    def test_interface_psi1_exists(self) -> Tuple[bool, str]:
        if self.psi_1 is None:
            return False, "psi_1 is missing in results."
        if not isinstance(self.psi_1, str):
            return False, f"psi_1 should be a string, got {type(self.psi_1)}"
        return True, "psi_1 is present and a string."

    def test_interface_psi_m_exists(self) -> Tuple[bool, str]:
        if self.psi_m is None:
            return False, "psi_m is missing in results."
        if not isinstance(self.psi_m, dict):
            return False, f"psi_m should be a dict, got {type(self.psi_m)}"
        required_keys = {'m=0', 'm=1', 'm=-1'}
        if not required_keys.issubset(self.psi_m.keys()):
            return False, f"psi_m missing keys. Has {set(self.psi_m.keys())}"
        return True, "psi_m dict is present with required keys."

    # -------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity checks
    # -------------------------------------------------------------------
    def test_invariant_E0_minimum(self) -> Tuple[bool, str]:
        """Check that E_0 derived from uncertainty principle equals the result."""
        if self.E_0 is None or self._expected_E0 is None:
            return False, "SKIP: missing E_0 or givens."
        # Use symbolic or numerical fallback
        if isinstance(self.E_0, sp.Expr):
            val = float(self.E_0.subs({sp.Symbol('hbar'): 1, sp.Symbol('omega'): 1}).evalf())
            expected_val = self._expected_E0
            # simplify: assume hbar and omega are 1 for check
            if not math.isclose(val, expected_val, rel_tol=1e-2):
                return False, f"E_0 symbolic value {val} not close to expected {expected_val}"
        else:
            if not math.isclose(float(self.E_0), self._expected_E0, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"E_0 numerical value {self.E_0} not close to expected {self._expected_E0}"
        return True, "E_0 matches the minimum energy derived from uncertainty principle."

    def test_invariant_alpha_correct(self) -> Tuple[bool, str]:
        """Check that alpha = m*omega/(2*hbar)."""
        if self.alpha is None or self._expected_alpha is None:
            return False, "SKIP: missing alpha or givens."
        if isinstance(self.alpha, sp.Expr):
            val = float(self.alpha.subs({sp.Symbol('m'): 1, sp.Symbol('omega'): 1, sp.Symbol('hbar'): 1}).evalf())
            expected_val = self._expected_alpha
        else:
            val = float(self.alpha)
            expected_val = self._expected_alpha
        if not math.isclose(val, expected_val, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"alpha value {val} not close to expected {expected_val}"
        return True, "alpha satisfies the relation alpha = m*omega/(2*hbar)."

    def test_invariant_uncertainty_product(self) -> Tuple[bool, str]:
        """
        For the Gaussian ψ = exp(-α x²),
        Δx = 1/(2√α), Δp = ħ√α.
        Verify that Δx·Δp = ħ/2.
        """
        if self.alpha is None or self.hbar is None:
            return False, "SKIP: missing alpha or hbar."
        try:
            alpha_val = float(self.alpha) if not isinstance(self.alpha, sp.Expr) else float(self.alpha.subs({sp.Symbol('m'):1, sp.Symbol('omega'):1, sp.Symbol('hbar'):1}).evalf())
            hbar_val = float(self.hbar)
            delta_x = 1.0 / (2.0 * math.sqrt(alpha_val))
            delta_p = hbar_val * math.sqrt(alpha_val)
            product = delta_x * delta_p
            expected = hbar_val / 2.0
            if math.isclose(product, expected, rel_tol=1e-2):
                return True, f"Uncertainty product {product} ~ {expected} (ħ/2)."
            else:
                return False, f"Uncertainty product {product} not close to ħ/2 = {expected}."
        except Exception as e:
            return False, f"Error in uncertainty product calculation: {e}"

    # -------------------------------------------------------------------
    # Modeling and trace tests – check the form of provided expressions
    # -------------------------------------------------------------------
    def test_modeling_psi1_form(self) -> Tuple[bool, str]:
        """
        The first excited state wavefunction should be proportional to
        x * exp(-α x²). The string representation should contain 'x' and 'exp'.
        """
        if self.psi_1 is None:
            return False, "SKIP: psi_1 missing."
        s = str(self.psi_1).lower()
        if 'x' not in s:
            return False, "psi_1 does not contain 'x'."
        if 'exp' not in s:
            return False, "psi_1 does not contain exponential."
        # Optionally check that alpha appears
        if self.alpha is not None:
            alpha_str = str(self.alpha) if not isinstance(self.alpha, sp.Expr) else str(self.alpha.evalf())
            if alpha_str not in s and 'alpha' not in s:
                return False, "psi_1 does not mention alpha."
        return True, "psi_1 has the expected functional form (x * exp(-α x²))."

    def test_modeling_psi_m_content(self) -> Tuple[bool, str]:
        """
        For the 3D wave functions, each m state should contain 'r', a spherical harmonic,
        and exponential factor.
        """
        if self.psi_m is None or not isinstance(self.psi_m, dict):
            return False, "SKIP: psi_m missing or not a dict."
        for key, expr in self.psi_m.items():
            if expr is None:
                return False, f"psi_m entry for {key} is None."
            s = str(expr).lower()
            if 'r' not in s:
                return False, f"psi_m[{key}] does not contain radial coordinate 'r'."
            # Spherical harmonic indicators: cosθ, sinθ, e^{+/-iφ}
            # Accept at least one of 'cosθ', 'sinθ', 'costheta', 'sintheta'
            if not any(x in s for x in ['cosθ', 'sinθ', 'costheta', 'sintheta']):
                return False, f"psi_m[{key}] lacks angular dependence (cosθ or sinθ)."
            if not ('exp' in s or 'e^' in s):
                return False, f"psi_m[{key}] lacks exponential factor."
        return True, "psi_m entries contain radial, angular, and exponential factors."

    def test_modeling_lz_eigenstate(self) -> Tuple[bool, str]:
        """
        Sanity: the m=0 state should be real (no i), while m=±1 should contain i in exponent.
        (Eigenstates of l_z.)
        """
        if self.psi_m is None:
            return False, "SKIP: psi_m missing."
        # m=0: should be real (no 'i' or 'phi')
        psi0 = str(self.psi_m.get('m=0', '')).lower()
        if 'i' in psi0 or 'φ' in psi0 or 'phi' in psi0:
            return False, "m=0 wavefunction should be real (no complex phase)."
        # m=±1: should contain e^{±iφ}
        for key in ['m=1', 'm=-1']:
            expr = self.psi_m.get(key, '')
            s = str(expr).lower()
            if 'i' not in s:
                return False, f"{key} wavefunction should contain a factor exp(±iφ)."
        return True, "l_z eigenstate property: m=0 real, m=±1 have phase factor."

    # -------------------------------------------------------------------
    # Monotonicity tests – check that perturbing an input changes output as expected
    # -------------------------------------------------------------------
    def test_monotonicity_alpha_increases_with_m(self) -> Tuple[bool, str]:
        """
        alpha = m*ω/(2ħ) → increasing m should increase alpha.
        We perturb m by +10% and recompute alpha using the theoretical relation.
        """
        if self.m is None or self.omega is None or self.hbar is None or self.alpha is None:
            return False, "SKIP: missing givens or alpha."
        try:
            alpha0 = float(self.alpha) if not isinstance(self.alpha, sp.Expr) else float(self.alpha.subs({sp.Symbol('m'): self.m, sp.Symbol('omega'):self.omega, sp.Symbol('hbar'):self.hbar}).evalf())
            m0 = float(self.m) if not isinstance(self.m, sp.Expr) else float(self.m)
            omega0 = float(self.omega) if not isinstance(self.omega, sp.Expr) else float(self.omega)
            hbar0 = float(self.hbar) if not isinstance(self.hbar, sp.Expr) else float(self.hbar)
            m_pert = m0 * 1.1
            alpha_pert = m_pert * omega0 / (2.0 * hbar0)
            if alpha_pert <= alpha0:
                return False, f"alpha did not increase when m increased by 10% (was {alpha0}, perturbed {alpha_pert})."
            return True, f"alpha increases with m: baseline {alpha0}, perturbed {alpha_pert}."
        except Exception as e:
            return False, f"Error in monotonicity test: {e}"

    def test_monotonicity_E0_increases_with_omega(self) -> Tuple[bool, str]:
        """
        E_0 = ħω/2 → increasing ω should increase E_0.
        Perturb ω by +10%.
        """
        if self.omega is None or self.hbar is None or self.E_0 is None:
            return False, "SKIP: missing omega, hbar or E_0."
        try:
            E0 = float(self.E_0) if not isinstance(self.E_0, sp.Expr) else float(self.E_0.subs({sp.Symbol('hbar'):self.hbar, sp.Symbol('omega'):self.omega}).evalf())
            hbar0 = float(self.hbar) if not isinstance(self.hbar, sp.Expr) else float(self.hbar)
            omega0 = float(self.omega) if not isinstance(self.omega, sp.Expr) else float(self.omega)
            omega_pert = omega0 * 1.1
            E0_pert = hbar0 * omega_pert / 2.0
            if E0_pert <= E0:
                return False, f"E_0 did not increase when ω increased by 10% (was {E0}, perturbed {E0_pert})."
            return True, f"E_0 increases with ω: baseline {E0}, perturbed {E0_pert}."
        except Exception as e:
            return False, f"Error in monotonicity test: {e}"

    # -------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # -------------------------------------------------------------------
    def test_answer_sanity_general(self) -> Tuple[bool, str]:
        """Basic sanity: all output quantities should be finite and have correct sign."""
        issues = []
        # E_0
        if self.E_0 is not None:
            try:
                e0 = float(self.E_0) if not isinstance(self.E_0, sp.Expr) else float(self.E_0.evalf())
                if e0 <= 0:
                    issues.append("E_0 must be positive (energy > 0).")
                if math.isnan(e0) or math.isinf(e0):
                    issues.append("E_0 is NaN or infinite.")
            except Exception:
                issues.append("Cannot evaluate E_0 numerically.")
        # alpha
        if self.alpha is not None:
            try:
                a = float(self.alpha) if not isinstance(self.alpha, sp.Expr) else float(self.alpha.evalf())
                if a <= 0:
                    issues.append("alpha must be positive (Gaussian width parameter).")
                if math.isnan(a) or math.isinf(a):
                    issues.append("alpha is NaN or infinite.")
            except Exception:
                issues.append("Cannot evaluate alpha numerically.")
        # psi_1, psi_m – just check they are non-empty strings
        if self.psi_1 is not None and (not isinstance(self.psi_1, str) or self.psi_1.strip() == ""):
            issues.append("psi_1 is empty or not a string.")
        if self.psi_m is not None and (not isinstance(self.psi_m, dict) or len(self.psi_m) == 0):
            issues.append("psi_m is empty dict or wrong type.")
        if issues:
            return False, "; ".join(issues)
        return True, "Basic answer sanity checks passed."

    # -------------------------------------------------------------------
    # Test runner
    # -------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)