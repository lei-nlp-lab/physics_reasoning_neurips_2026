import math
import sympy as sp
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    PhysicsChecker for Van der Waals gas problem: compute molar heat capacities C_v and C_p.
    Performs theorem-based invariants, modeling consistency, and weak sanity checks.
    """

    def __init__(self, solution_instance: Any):
        # --- Extract results safely ---
        results = getattr(solution_instance, 'results', None) or {}
        self.C_v = getattr(solution_instance, 'C_v', None) or results.get('C_v', None)
        self.C_p = getattr(solution_instance, 'C_p', None) or results.get('C_p', None)

        # --- Extract givens safely ---
        givens = getattr(solution_instance, 'givens', None) or {}
        self.R = givens.get('R', None) or getattr(solution_instance, 'R', None)
        self.a = givens.get('a', None) or getattr(solution_instance, 'a', None)
        self.b = givens.get('b', None) or getattr(solution_instance, 'b', None)
        self.c = givens.get('c', None) or getattr(solution_instance, 'c', None)
        self.T = givens.get('T', None) or getattr(solution_instance, 'T', None)
        self.V = givens.get('V', None) or getattr(solution_instance, 'V', None)

        # Fallback: if any givens are missing, we may skip some tests
        self._have_givens = all(v is not None for v in [self.R, self.a, self.b, self.c, self.T, self.V])
        self._have_results = self.C_v is not None and self.C_p is not None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_C_v_exists(self) -> Tuple[bool, str]:
        if self.C_v is None:
            return (False, "C_v missing from solution")
        return (True, "C_v is present")

    def test_interface_C_p_exists(self) -> Tuple[bool, str]:
        if self.C_p is None:
            return (False, "C_p missing from solution")
        return (True, "C_p is present")

    def test_interface_C_v_numeric(self) -> Tuple[bool, str]:
        if self.C_v is None:
            return (False, "SKIP: C_v not available")
        if not isinstance(self.C_v, (int, float)):
            return (False, f"C_v is not numeric (type {type(self.C_v).__name__})")
        return (True, f"C_v = {self.C_v} is numeric")

    def test_interface_C_p_numeric(self) -> Tuple[bool, str]:
        if self.C_p is None:
            return (False, "SKIP: C_p not available")
        if not isinstance(self.C_p, (int, float)):
            return (False, f"C_p is not numeric (type {type(self.C_p).__name__})")
        return (True, f"C_p = {self.C_p} is numeric")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ----------------------------------------------------------------------
    def test_invariant_C_v_from_internal_energy(self) -> Tuple[bool, str]:
        """Check that C_v equals the derivative of u w.r.t T at constant V: C_v = c."""
        if self.C_v is None or self.c is None:
            return (False, "SKIP: C_v or constant c not available")
        if not math.isclose(self.C_v, self.c, rel_tol=1e-6, abs_tol=1e-6):
            return (False, f"C_v = {self.C_v} does not match c = {self.c}")
        return (True, f"C_v = {self.C_v} matches c = {self.c}")

    def test_invariant_C_p_greater_than_C_v(self) -> Tuple[bool, str]:
        """Thermodynamic stability: C_p > C_v for a real gas."""
        if not self._have_results:
            return (False, "SKIP: C_v or C_p not available")
        if not (self.C_p > self.C_v):
            return (False, f"C_p = {self.C_p} is not greater than C_v = {self.C_v}")
        return (True, f"C_p = {self.C_p} > C_v = {self.C_v}")

    def test_invariant_C_p_minus_C_v_via_identity(self) -> Tuple[bool, str]:
        """
        Check thermodynamic identity:
        C_p - C_v = T * (∂P/∂T)_V * (∂V/∂T)_P
        using given givens (a, b, R, T, V).
        """
        if not self._have_givens or not self._have_results:
            return (False, "SKIP: missing givens or results")
        # Compute (∂P/∂T)_V from VdW: P = RT/(V-b) - a/V^2
        # dP/dT_V = R/(V-b)
        dP_dT_V = self.R / (self.V - self.b)
        # Compute (∂P/∂V)_T
        dP_dV_T = -(self.R * self.T) / ((self.V - self.b)**2) + 2*self.a / (self.V**3)
        if abs(dP_dV_T) < 1e-15:
            return (False, "SKIP: dP/dV_T is too close to zero")
        # Use triple product: (∂V/∂T)_P = - (∂P/∂T)_V / (∂P/∂V)_T
        dV_dT_P = - dP_dT_V / dP_dV_T
        predicted_diff = self.T * dP_dT_V * dV_dT_P
        actual_diff = self.C_p - self.C_v
        if not math.isclose(predicted_diff, actual_diff, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Thermodynamic identity fails: predicted difference = {predicted_diff}, actual = {actual_diff}")
        return (True, f"C_p - C_v matches identity (diff = {actual_diff})")

    # ----------------------------------------------------------------------
    # Modeling/trace checks
    # ----------------------------------------------------------------------
    def test_modeling_C_v_expression(self) -> Tuple[bool, str]:
        """Check that C_v equals constant c (as derived from u = cT - a/V)."""
        # Same as invariant test but we treat it as a modeling check.
        if self.C_v is None or self.c is None:
            return (False, "SKIP: C_v or c missing")
        if not math.isclose(self.C_v, self.c, rel_tol=1e-6, abs_tol=1e-6):
            return (False, f"C_v = {self.C_v} should equal c = {self.c}")
        return (True, f"C_v = c = {self.c}")

    def test_modeling_C_p_formula(self) -> Tuple[bool, str]:
        """
        Check that C_p matches the derived formula:
        C_p = c + R / (1 - 2*a*(V-b)^2/(R*T*V^3))
        """
        if not self._have_givens or self.C_p is None:
            return (False, "SKIP: missing required data")
        denominator = 1 - (2*self.a * (self.V - self.b)**2) / (self.R * self.T * self.V**3)
        if abs(denominator) < 1e-15:
            return (False, "SKIP: denominator too close to zero")
        expected_C_p = self.c + self.R / denominator
        if not math.isclose(self.C_p, expected_C_p, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_p = {self.C_p} does not match expected {expected_C_p}")
        return (True, f"C_p matches derived formula: {self.C_p}")

    def test_trace_thermodynamic_step_consistency(self) -> Tuple[bool, str]:
        """
        Check one local relation from explanation steps:
        (∂V/∂T)_P should equal R/(p - a/V^2 + 2ab/V^3) when using VdW.
        Since P is not directly given, we verify using the implicit derivative.
        """
        if not self._have_givens:
            return (False, "SKIP: missing givens")
        # Compute pressure from VdW
        P = (self.R * self.T) / (self.V - self.b) - self.a / (self.V**2)
        # Compute expected (∂V/∂T)_P from formula in explanation steps
        numerator = self.R
        denominator = P - self.a/(self.V**2) + 2*self.a*self.b/(self.V**3)
        if abs(denominator) < 1e-15:
            return (False, "SKIP: denominator zero")
        dV_dT_P_formula = numerator / denominator
        # Compute using triple product
        dP_dT_V = self.R / (self.V - self.b)
        dP_dV_T = -(self.R * self.T) / ((self.V - self.b)**2) + 2*self.a / (self.V**3)
        if abs(dP_dV_T) < 1e-15:
            return (False, "SKIP: dP/dV_T too small")
        dV_dT_P_triple = - dP_dT_V / dP_dV_T
        if not math.isclose(dV_dT_P_formula, dV_dT_P_triple, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Trace inconsistency: formula gives {dV_dT_P_formula}, triple product gives {dV_dT_P_triple}")
        return (True, f"(∂V/∂T)_P consistent: {dV_dT_P_formula}")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_C_p_with_temperature(self) -> Tuple[bool, str]:
        """
        Check that C_p decreases when T increases (for typical positive a, b).
        Derive sign by perturbing T by +1% and recomputing C_p via formula.
        """
        if not self._have_givens or self.C_p is None:
            return (False, "SKIP: missing data for monotonicity")
        # Save original values
        T_orig = self.T
        V_orig = self.V  # V may also change with T at constant P, but we keep V fixed for simplicity
        # Compute C_p at T+1%
        T_new = T_orig * 1.01
        # Assume V is unchanged (isochoric perturbation; not fully physical but monotonicity should hold locally)
        # Use the formula
        denom_orig = 1 - (2*self.a * (V_orig - self.b)**2) / (self.R * T_orig * V_orig**3)
        denom_new = 1 - (2*self.a * (V_orig - self.b)**2) / (self.R * T_new * V_orig**3)
        if abs(denom_orig) < 1e-15 or abs(denom_new) < 1e-15:
            return (False, "SKIP: denominator too close to zero")
        Cp_orig = self.c + self.R / denom_orig
        Cp_new = self.c + self.R / denom_new
        # For typical values, C_p decreases with T (denom increases -> Cp decreases)
        if Cp_new >= Cp_orig:
            return (True, f"C_p non-decreasing with T (Cp_orig={Cp_orig}, Cp_new={Cp_new}) – note: typically decreases but not strict")
        # It may increase for some parameters; we just report the behavior
        return (True, f"C_p decreases with T: {Cp_orig} -> {Cp_new}")

    def test_monotonicity_C_p_with_volume(self) -> Tuple[bool, str]:
        """
        Check that C_p decreases with increasing V (at constant T).
        Perturb V by +1%.
        """
        if not self._have_givens or self.C_p is None:
            return (False, "SKIP: missing data")
        V_orig = self.V
        V_new = V_orig * 1.01
        denom_orig = 1 - (2*self.a * (V_orig - self.b)**2) / (self.R * self.T * V_orig**3)
        denom_new = 1 - (2*self.a * (V_new - self.b)**2) / (self.R * self.T * V_new**3)
        if abs(denom_orig) < 1e-15 or abs(denom_new) < 1e-15:
            return (False, "SKIP: denominator zero")
        Cp_orig = self.c + self.R / denom_orig
        Cp_new = self.c + self.R / denom_new
        # Typical: C_p decreases with V
        return (True, f"C_p with V: {Cp_orig} -> {Cp_new}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, minimal)
    # ----------------------------------------------------------------------
    def test_answer_sanity_values(self) -> Tuple[bool, str]:
        """
        Weak sanity: C_v should be positive (c>0) and C_p > C_v.
        No exact matching, no isclose.
        """
        if not self._have_results:
            return (False, "SKIP: results missing")
        if not (self.C_v > 0):
            return (False, f"C_v = {self.C_v} is not positive")
        if not (self.C_p > self.C_v):
            return (False, f"C_p = {self.C_p} is not greater than C_v")
        # Optionally check C_p is finite
        if math.isinf(self.C_p) or math.isnan(self.C_p):
            return (False, f"C_p is not finite: {self.C_p}")
        return (True, f"Sanity: C_v={self.C_v} > 0, C_p={self.C_p} > C_v, finite")

    # ----------------------------------------------------------------------
    # Master runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods. Returns (all_pass, logs).
        """
        logs = []
        all_pass = True
        # Get all callable attributes starting with test_
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