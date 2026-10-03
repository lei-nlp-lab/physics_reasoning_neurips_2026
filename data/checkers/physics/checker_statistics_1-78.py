import sympy as sp
import math
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

class PhysicsChecker:
    """
    Auditor for the thermodynamics of a classical paramagnetic system.
    Checks symbolic expressions for dQ, dS, and S against first-law, equation-of-state,
    and entropy differential constraints.
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction of result keys (all from pseudocode emit)
        self.dQ_coeff_M = self._safe_get(solution_instance, 'dQ_coeff_M')
        self.dQ_coeff_B = self._safe_get(solution_instance, 'dQ_coeff_B')
        self.dS_coeff_M = self._safe_get(solution_instance, 'dS_coeff_M')
        self.dS_coeff_B = self._safe_get(solution_instance, 'dS_coeff_B')
        self.S = self._safe_get(solution_instance, 'S')

        # Symbols used in the problem
        self.B, self.M, self.T, self.C = sp.symbols('B M T C', real=True, positive=True)
        self.S0 = sp.symbols('S0', real=True)

        # Precompute reference symbolic expressions
        self._ref_dQ_M = -self.B
        self._ref_dQ_B = sp.Integer(0)
        self._ref_dS_M = -self.M / self.C
        self._ref_dS_B = sp.Integer(0)
        self._ref_S = self.S0 - self.M**2 / (2 * self.C)

    @staticmethod
    def _safe_get(instance: Any, key: str) -> Any:
        """Retrieve a value from a dict attribute or an object attribute safely."""
        # Try results dictionary first
        if hasattr(instance, 'results') and isinstance(instance.results, dict):
            return instance.results.get(key, None)
        # Then try as attribute
        return getattr(instance, key, None)

    @staticmethod
    def _to_sympy(value: Any) -> Optional[sp.Expr]:
        """Convert a string or SymPy expression to a SymPy expression, else None."""
        if value is None:
            return None
        if isinstance(value, sp.Expr):
            return value
        try:
            return sp.sympify(value)
        except (sp.SympifyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Interface tests: existence and type
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        """All five result keys must be present (not None)."""
        keys = ['dQ_coeff_M', 'dQ_coeff_B', 'dS_coeff_M', 'dS_coeff_B', 'S']
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                return False, f"Key '{key}' is missing or None"
        return True, "All five result keys are present"

    def test_interface_keys_valid_sympy(self) -> Tuple[bool, str]:
        """All results should be convertible to valid SymPy expressions."""
        for key in ['dQ_coeff_M', 'dQ_coeff_B', 'dS_coeff_M', 'dS_coeff_B', 'S']:
            val = getattr(self, key, None)
            if val is None:
                continue  # handled elsewhere
            expr = self._to_sympy(val)
            if expr is None:
                return False, f"Key '{key}' cannot be parsed as a SymPy expression: {val}"
        return True, "All present result keys are valid SymPy expressions"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose for numeric fallback)
    # ------------------------------------------------------------------
    def test_invariant_first_law_dQ_via_dU_dW(self) -> Tuple[bool, str]:
        """
        Check that the reported dQ coefficients satisfy dQ = dU + dW.
        dU = -B dM - M dB,  dW = M dB  =>  dQ = -B dM.
        """
        dQ_M = self._to_sympy(self.dQ_coeff_M)
        dQ_B = self._to_sympy(self.dQ_coeff_B)
        if dQ_M is None or dQ_B is None:
            return False, "SKIP: insufficient dQ data"
        # Expected from first law
        expected_dQ_M = -self.B
        expected_dQ_B = sp.Integer(0)
        # Symbolic equality check
        if sp.simplify(dQ_M - expected_dQ_M) == 0 and sp.simplify(dQ_B - expected_dQ_B) == 0:
            return True, "dQ coefficients exactly match first law"
        # Fallback numeric check (if symbolic fails due to unknown variables)
        # Provide a concrete numeric instantiation, e.g., B=1.5, M=2.0, C=1.0, T=1.5
        subs_dict = {self.B: 1.5, self.M: 2.0, self.C: 1.0, self.T: 1.5}
        try:
            dQ_M_num = float(dQ_M.subs(subs_dict))
            dQ_B_num = float(dQ_B.subs(subs_dict))
            exp_M_num = float(expected_dQ_M.subs(subs_dict))
            exp_B_num = float(expected_dQ_B.subs(subs_dict))
            if math.isclose(dQ_M_num, exp_M_num, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(dQ_B_num, exp_B_num, rel_tol=1e-2, abs_tol=1e-2):
                return True, "dQ coefficients match first law numerically"
            else:
                return False, f"Numeric mismatch: dQ_M={dQ_M_num} vs {exp_M_num}, dQ_B={dQ_B_num} vs {exp_B_num}"
        except:
            return False, "Cannot evaluate dQ coefficients numerically for invariant test"

    def test_invariant_entropy_via_dQ_T(self) -> Tuple[bool, str]:
        """
        Check that reported dS coefficients satisfy dS = dQ / T,
        with T from equation of state T = C B / M.
        """
        dS_M = self._to_sympy(self.dS_coeff_M)
        dS_B = self._to_sympy(self.dS_coeff_B)
        dQ_M = self._to_sympy(self.dQ_coeff_M)
        dQ_B = self._to_sympy(self.dQ_coeff_B)
        if dS_M is None or dS_B is None or dQ_M is None or dQ_B is None:
            return False, "SKIP: insufficient dQ or dS data"
        # Compute expected dS from dQ/T = (-B dM) / T = -B/T dM
        T_expr = self.C * self.B / self.M   # from equation of state
        expected_dS_M = sp.simplify(self._ref_dQ_M / T_expr)
        expected_dS_B = sp.simplify(self._ref_dQ_B / T_expr)
        # Symbolic check
        if sp.simplify(dS_M - expected_dS_M) == 0 and sp.simplify(dS_B - expected_dS_B) == 0:
            return True, "dS coefficients exactly match dQ/T"
        # Numeric fallback
        subs_dict = {self.B: 2.0, self.M: 0.5, self.C: 1.0, self.T: 4.0}
        try:
            dS_M_num = float(dS_M.subs(subs_dict))
            dS_B_num = float(dS_B.subs(subs_dict))
            exp_M_num = float(expected_dS_M.subs(subs_dict))
            exp_B_num = float(expected_dS_B.subs(subs_dict))
            if math.isclose(dS_M_num, exp_M_num, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(dS_B_num, exp_B_num, rel_tol=1e-2, abs_tol=1e-2):
                return True, "dS coefficients match dQ/T numerically"
            else:
                return False, f"Numeric mismatch: dS_M={dS_M_num} vs {exp_M_num}, dS_B={dS_B_num} vs {exp_B_num}"
        except:
            return False, "Cannot evaluate dS coefficients numerically"

    def test_invariant_equation_of_state_consistency(self) -> Tuple[bool, str]:
        """
        Check that the reported dS coefficients are consistent with the
        equation of state M = C B / T.
        """
        dS_M = self._to_sympy(self.dS_coeff_M)
        dS_B = self._to_sympy(self.dS_coeff_B)
        if dS_M is None or dS_B is None:
            return False, "SKIP: insufficient dS data"
        # Expected from dQ/T after substituting T = C B/M gives dS = -M/C dM.
        expected_dS_M_given_EoS = sp.simplify(-self.M / self.C)
        expected_dS_B_given_EoS = sp.Integer(0)
        if sp.simplify(dS_M - expected_dS_M_given_EoS) == 0 and \
           sp.simplify(dS_B - expected_dS_B_given_EoS) == 0:
            return True, "dS coefficients match the form derived using state equation"
        # Numeric fallback (arbitrary values)
        subs = {self.B: 1.0, self.M: 2.0, self.C: 0.5}
        try:
            dS_M_num = float(dS_M.subs(subs))
            dS_B_num = float(dS_B.subs(subs))
            exp_M_num = float(expected_dS_M_given_EoS.subs(subs))
            exp_B_num = float(expected_dS_B_given_EoS.subs(subs))
            if math.isclose(dS_M_num, exp_M_num, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(dS_B_num, exp_B_num, rel_tol=1e-2, abs_tol=1e-2):
                return True, "dS coefficients numerically consistent with equation of state"
            else:
                return False, f"Numeric mismatch: dS_M={dS_M_num} vs {exp_M_num}, dS_B={dS_B_num} vs {exp_B_num}"
        except:
            return False, "Cannot evaluate numeric consistency of dS with EoS"

    # ------------------------------------------------------------------
    # Modeling tests (quality of the derived expressions)
    # ------------------------------------------------------------------
    def test_modeling_dQ_form(self) -> Tuple[bool, str]:
        """Check that dQ = -B dM exactly (i.e., coefficient of dB is zero)."""
        dQ_M = self._to_sympy(self.dQ_coeff_M)
        dQ_B = self._to_sympy(self.dQ_coeff_B)
        if dQ_M is None or dQ_B is None:
            return False, "SKIP: missing dQ coefficients"
        # Exact symbolic check
        if sp.simplify(dQ_M - self._ref_dQ_M) == 0 and sp.simplify(dQ_B - self._ref_dQ_B) == 0:
            return True, "dQ_form: coefficient of dM = -B, coefficient of dB = 0"
        else:
            return False, f"dQ_form mismatch: expected ({self._ref_dQ_M}, {self._ref_dQ_B}) but got ({dQ_M}, {dQ_B})"

    def test_modeling_dS_form(self) -> Tuple[bool, str]:
        """Check that dS = -(M/C) dM exactly."""
        dS_M = self._to_sympy(self.dS_coeff_M)
        dS_B = self._to_sympy(self.dS_coeff_B)
        if dS_M is None or dS_B is None:
            return False, "SKIP: missing dS coefficients"
        if sp.simplify(dS_M - self._ref_dS_M) == 0 and sp.simplify(dS_B - self._ref_dS_B) == 0:
            return True, "dS_form: coefficient of dM = -M/C, coefficient of dB = 0"
        else:
            return False, f"dS_form mismatch: expected ({self._ref_dS_M}, {self._ref_dS_B}) but got ({dS_M}, {dS_B})"

    def test_modeling_S_via_integration(self) -> Tuple[bool, str]:
        """Check that the derivative of the reported S expression yields -M/C."""
        S = self._to_sympy(self.S)
        if S is None:
            return False, "SKIP: missing S expression"
        # The derivative of S w.r.t. M should be -M/C (since S = S0 - M^2/(2C))
        dS_dM = sp.diff(S, self.M)
        expected_deriv = sp.simplify(-self.M / self.C)
        if sp.simplify(dS_dM - expected_deriv) == 0:
            return True, "Derivative of S yields the expected dS coeff for dM"
        # Fallback numeric check
        subs = {self.M: 1.0, self.C: 2.0, self.S0: 0.0}
        try:
            deriv_num = float(dS_dM.subs(subs))
            exp_num = float(expected_deriv.subs(subs))
            if math.isclose(deriv_num, exp_num, rel_tol=1e-2, abs_tol=1e-2):
                return True, "Derivative of S numerically matches -M/C"
            else:
                return False, f"Numeric derivative mismatch: {deriv_num} vs {exp_num}"
        except:
            return False, "Cannot evaluate derivative numerically"

    # ------------------------------------------------------------------
    # Monotonicity test: perturb input M and compare S output
    # ------------------------------------------------------------------
    def test_monotonicity_entropy_decreases_with_M(self) -> Tuple[bool, str]:
        """
        Perturb magnetization M (increase by 10%) and check that entropy S decreases.
        Uses the reported S expression evaluated at two different M values.
        """
        S = self._to_sympy(self.S)
        if S is None:
            return False, "SKIP: missing S expression"

        # Choose baseline values (arbitrary but positive)
        baseline_M = 1.0
        C_val = 1.0
        S0_val = 0.0
        pert_ratio = 0.10  # 10% increase
        perturbed_M = baseline_M * (1 + pert_ratio)

        try:
            subs_base = {self.M: baseline_M, self.C: C_val, self.S0: S0_val}
            subs_pert = {self.M: perturbed_M, self.C: C_val, self.S0: S0_val}
            S_base = float(S.subs(subs_base))
            S_pert = float(S.subs(subs_pert))
        except Exception as e:
            return False, f"Cannot evaluate S numerically: {e}"

        # Entropy should decrease when M increases, i.e., S_pert < S_base
        if S_pert < S_base:
            return True, f"S decreased from {S_base:.4f} to {S_pert:.4f} when M increased by {pert_ratio*100:.0f}%"
        else:
            return False, f"S did not decrease: baseline S={S_base:.4f}, perturbed S={S_pert:.4f} (expected decrease)"

    # ------------------------------------------------------------------
    # Weak answer sanity (one test only)
    # ------------------------------------------------------------------
    def test_answer_sanity_S_form(self) -> Tuple[bool, str]:
        """
        Weak check: the entropy S should be of the form S0 - M^2/(2C).
        Do not re-derive, just check that it contains a quadratic term in M.
        """
        S = self._to_sympy(self.S)
        if S is None:
            return False, "SKIP: missing S expression"
        # Check that S depends on M and has a negative M^2 term.
        try:
            expanded = sp.expand(S)
            coeff_M2 = expanded.coeff(self.M**2)
            if coeff_M2 is sp.Integer(0) or coeff_M2 is None:
                return False, "S does not contain a non-zero M^2 term"
            # The coefficient should be negative
            # Evaluate numerically for positive C
            val = float(coeff_M2.subs({self.C: 1.0}))
            if val < 0:
                return True, f"S contains a negative quadratic term in M (coeff {val})"
            else:
                return False, f"Quadratic coefficient must be negative, got {val}"
        except Exception as e:
            return False, f"Cannot analyze S form: {e}"

    # ------------------------------------------------------------------
    # Main test runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)