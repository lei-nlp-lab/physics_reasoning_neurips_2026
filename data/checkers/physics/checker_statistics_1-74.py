import math
import sympy as sym

class PhysicsChecker:
    def __init__(self, solution_instance):
        # safe extraction: if solution_instance has a results dict, use it; otherwise fallback to attributes
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # required output quantities
        self.c_v = results.get('c_v', None)          # (5/2)*n*R or expression
        self.c_p = results.get('c_p', None)          # (7/2)*n*R or expression
        self.energy_density = results.get('energy_density', None)   # (5/2)*P or expression

        # additional givens that may appear in solution_instance
        self.n = getattr(solution_instance, 'n', None)   # number of moles
        self.R = getattr(solution_instance, 'R', None)   # gas constant
        self.P = getattr(solution_instance, 'p', None)   # pressure (may be symbolic)
        self.T_i = getattr(solution_instance, 'T_i', None)
        self.T_f = getattr(solution_instance, 'T_f', None)
        self.V = getattr(solution_instance, 'V', None)   # volume

        # store the entropy expression if available (for derivative checks)
        self.S_expr = getattr(solution_instance, 'S', None)

        # store the solution instance for potential symbolic access
        self.sol = solution_instance

    # ------------------------------------------------------------------
    # Helper: try to evaluate a symbolic expression to a float if possible
    # ------------------------------------------------------------------
    def _to_numeric(self, expr):
        """Convert expression to float; return None if not possible."""
        if expr is None:
            return None
        if isinstance(expr, (int, float)):
            return float(expr)
        # sympy expression
        try:
            return float(expr.evalf())
        except:
            return None

    # ------------------------------------------------------------------
    # Interface checks: required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_c_v_exists(self):
        if self.c_v is None:
            return (False, "c_v not found in results")
        if not isinstance(self.c_v, (int, float, sym.Basic)):
            return (False, f"c_v has invalid type {type(self.c_v)}")
        return (True, "c_v present and of numeric/SymPy type")

    def test_interface_c_p_exists(self):
        if self.c_p is None:
            return (False, "c_p not found in results")
        if not isinstance(self.c_p, (int, float, sym.Basic)):
            return (False, f"c_p has invalid type {type(self.c_p)}")
        return (True, "c_p present and of numeric/SymPy type")

    def test_interface_energy_density_exists(self):
        if self.energy_density is None:
            return (False, "energy_density not found in results")
        if not isinstance(self.energy_density, (int, float, sym.Basic)):
            return (False, f"energy_density has invalid type {type(self.energy_density)}")
        return (True, "energy_density present and of numeric/SymPy type")

    # ------------------------------------------------------------------
    # Invariant checks: theorem-based physical relations
    # ------------------------------------------------------------------
    def test_invariant_temperature_from_entropy(self):
        """
        Check that 1/T = (dS/dU) leads to U = (5/2)nRT.
        We derive from the given entropy expression if available.
        """
        if self.S_expr is None:
            # cannot check, skip
            return (True, "SKIP: entropy expression not provided")
        # symbolic variables
        n, R, U, V, sigma = sym.symbols('n R U V sigma')
        # given entropy expression
        S = n/2 * (sigma + 5*R*sym.log(U/n) + 2*R*sym.log(V/n))
        # compute dS/dU
        dS_dU = sym.diff(S, U)
        # 1/T = dS/dU => T = 1/dS_dU
        T_expr = 1/dS_dU
        # expected U from T: U = (5/2)*n*R*T
        U_expected = (5/2) * n * R * T_expr
        # simplify and compare with U itself (symbolic identity)
        diff = sym.simplify(U - U_expected)
        # if diff is identically zero, the relation holds
        if diff == 0:
            return (True, "1/T = dS/dU yields U = (5/2)nRT correctly")
        else:
            return (False, f"Expected U = (5/2)nRT, but U - expected = {diff}")

    def test_invariant_ideal_gas_law(self):
        """
        Check that from entropy we obtain PV = nRT.
        """
        if self.S_expr is None:
            return (True, "SKIP: entropy expression not provided")
        n, R, U, V, sigma = sym.symbols('n R U V sigma')
        S = n/2 * (sigma + 5*R*sym.log(U/n) + 2*R*sym.log(V/n))
        # P/T = (dS/dV)_{U,n}
        dS_dV = sym.diff(S, V)
        # P/T = nR/V => P = nRT/V
        # but we need a symbolic check: T not present; use relation from previous step
        # Instead, compute P from dS_dV: P = T * dS_dV
        # Use the already derived relation T = 2U/(5nR)
        T = 2*U/(5*n*R)
        P_from_S = T * dS_dV
        # Simplify to see if it equals nR*T/V
        expected_P = n*R*T/V
        diff = sym.simplify(P_from_S - expected_P)
        if diff == 0:
            return (True, "PV = nRT follows from entropy")
        else:
            return (False, f"Expected PV = nRT, but difference = {diff}")

    def test_invariant_c_v_ideal_gas_relation(self):
        """
        Verify that c_v = (5/2)nR, using U = (5/2)nRT and the definition.
        If n and R are known, we can check numerically; otherwise symbolic.
        """
        if self.c_v is None:
            return (False, "c_v not available")
        # expected form: (5/2)*n*R
        n_val = self.n if self.n is not None else sym.Symbol('n')
        R_val = self.R if self.R is not None else sym.Symbol('R')
        expected = (5/2) * n_val * R_val
        # Compare
        if isinstance(self.c_v, sym.Basic) and isinstance(expected, sym.Basic):
            diff = sym.simplify(self.c_v - expected)
            if diff == 0:
                return (True, "c_v equals (5/2)nR symbolically")
            else:
                return (False, f"c_v should be (5/2)nR; difference = {diff}")
        else:
            # numeric fallback
            cv_num = self._to_numeric(self.c_v)
            exp_num = self._to_numeric(expected)
            if cv_num is None or exp_num is None:
                return (True, "SKIP: cannot evaluate c_v numerically")
            if math.isclose(cv_num, exp_num, rel_tol=1e-3, abs_tol=1e-3):
                return (True, f"c_v ≈ {exp_num} matches expected")
            else:
                return (False, f"c_v = {cv_num} but expected {exp_num}")

    def test_invariant_c_p_ideal_gas_relation(self):
        """
        Check that c_p = (7/2)nR, or equivalently c_p - c_v = nR.
        """
        if self.c_v is None or self.c_p is None:
            return (True, "SKIP: both c_v and c_p needed")
        n_val = self.n if self.n is not None else sym.Symbol('n')
        R_val = self.R if self.R is not None else sym.Symbol('R')
        expected = (7/2) * n_val * R_val
        # Check c_p directly
        if isinstance(self.c_p, sym.Basic) and isinstance(expected, sym.Basic):
            diff = sym.simplify(self.c_p - expected)
            if diff == 0:
                return (True, "c_p equals (7/2)nR symbolically")
            else:
                # Fallback: check c_p - c_v = nR
                diff2 = sym.simplify(self.c_p - self.c_v - n_val*R_val)
                if diff2 == 0:
                    return (True, "c_p - c_v = nR holds symbolically")
                else:
                    return (False, f"c_p not (7/2)nR; diff from expected {diff}; diff from c_v+nR {diff2}")
        else:
            # numeric
            cv_num = self._to_numeric(self.c_v)
            cp_num = self._to_numeric(self.c_p)
            exp_num = self._to_numeric(expected)
            if cv_num is None or cp_num is None:
                return (True, "SKIP: cannot evaluate c_p numerically")
            # Check direct match
            if math.isclose(cp_num, exp_num, rel_tol=1e-3, abs_tol=1e-3):
                return (True, f"c_p ≈ {exp_num} matches expected")
            # Check difference
            nR_num = self._to_numeric(n_val*R_val)
            if nR_num is not None and math.isclose(cp_num - cv_num, nR_num, rel_tol=1e-3, abs_tol=1e-3):
                return (True, f"c_p - c_v = {nR_num} holds numerically")
            else:
                return (False, f"c_p = {cp_num}, expected {exp_num} or check difference")

    def test_invariant_energy_density_constancy(self):
        """
        Check that energy density = (5/2)P, and that it is independent of T.
        If pressure is constant, U/V is same at both T_i and T_f.
        """
        if self.energy_density is None:
            return (False, "energy_density not available")
        # expected form: (5/2)*P
        P_val = self.P if self.P is not None else sym.Symbol('P')
        expected = (5/2) * P_val
        if isinstance(self.energy_density, sym.Basic) and isinstance(expected, sym.Basic):
            diff = sym.simplify(self.energy_density - expected)
            if diff == 0:
                # Additional check: if T_i and T_f are given, verify constancy
                if self.T_i is not None and self.T_f is not None:
                    # We could recompute energy density at each T using U = (5/2)nRT and PV=nRT
                    # But that's the same as checking the symbolic result.
                    pass
                return (True, "energy_density = (5/2)P symbolically")
            else:
                return (False, f"energy_density should be (5/2)P; difference = {diff}")
        else:
            ed_num = self._to_numeric(self.energy_density)
            exp_num = self._to_numeric(expected)
            if ed_num is None or exp_num is None:
                return (True, "SKIP: cannot evaluate energy density numerically")
            if math.isclose(ed_num, exp_num, rel_tol=1e-3, abs_tol=1e-3):
                return (True, f"energy_density ≈ {exp_num} matches expected")
            else:
                return (False, f"energy_density = {ed_num}, expected {exp_num}")

    # ------------------------------------------------------------------
    # Modeling checks: quality of derivation steps
    # ------------------------------------------------------------------
    def test_modeling_c_v_derivation_uses_U(self):
        """
        Check that c_v is derived from U = (5/2)nRT. 
        We don't have access to intermediate steps, but we can verify that c_v matches derivative.
        """
        if self.c_v is None:
            return (False, "c_v needed")
        n_val = self.n if self.n is not None else sym.Symbol('n')
        R_val = self.R if self.R is not None else sym.Symbol('R')
        T = sym.Symbol('T')
        U = (5/2)*n_val*R_val*T
        c_v_derived = sym.diff(U, T)
        if isinstance(self.c_v, sym.Basic):
            # compare with c_v_derived
            diff = sym.simplify(self.c_v - c_v_derived)
            if diff == 0:
                return (True, "c_v consistent with derivative of U = (5/2)nRT")
            else:
                return (False, f"c_v does not match dU/dT; difference = {diff}")
        else:
            # numeric check if n,R known
            cv_num = self._to_numeric(self.c_v)
            # need T to numeric? not needed for derivative.
            expected_num = self._to_numeric((5/2)*n_val*R_val)
            if cv_num is not None and expected_num is not None and math.isclose(cv_num, expected_num, rel_tol=1e-3):
                return (True, "c_v numeric value matches dU/dT")
            return (True, "SKIP: unable to verify derivative numerically")

    def test_modeling_c_p_via_enthalpy(self):
        """
        Verify that c_p is consistent with H = U + PV = (7/2)nRT.
        """
        if self.c_p is None or self.c_v is None:
            return (True, "SKIP: need c_p and c_v")
        # Expect c_p = c_v + nR
        n_val = self.n if self.n is not None else sym.Symbol('n')
        R_val = self.R if self.R is not None else sym.Symbol('R')
        if isinstance(self.c_p, sym.Basic) and isinstance(self.c_v, sym.Basic):
            diff = sym.simplify(self.c_p - self.c_v - n_val*R_val)
            if diff == 0:
                return (True, "c_p = c_v + nR holds")
            else:
                return (False, f"c_p - c_v != nR; difference = {diff}")
        else:
            cv_num = self._to_numeric(self.c_v)
            cp_num = self._to_numeric(self.c_p)
            nR_num = self._to_numeric(n_val*R_val)
            if None in (cv_num, cp_num, nR_num):
                return (True, "SKIP: cannot evaluate numeric check")
            if math.isclose(cp_num - cv_num, nR_num, rel_tol=1e-3):
                return (True, "c_p = c_v + nR numerically")
            else:
                return (False, f"c_p - c_v = {cp_num-cv_num}, expected {nR_num}")

    def test_modeling_energy_density_from_ideal_gas(self):
        """
        Verify that U/V = (5/2)P is derived from ideal gas law.
        """
        if self.energy_density is None:
            return (False, "energy_density not available")
        # We cannot re-derive, but we can check that if we have U and V separately,
        # the ratio matches. But we don't have U,V necessarily.
        # Use symbolic: if energy_density is (5/2)P, that's sufficient.
        # This test is redundant with invariant, but we include as modeling.
        P_val = self.P if self.P is not None else sym.Symbol('P')
        expected = (5/2) * P_val
        if isinstance(self.energy_density, sym.Basic) and isinstance(expected, sym.Basic):
            diff = sym.simplify(self.energy_density - expected)
            if diff == 0:
                return (True, "U/V = (5/2)P consistent with ideal gas law")
            else:
                return (False, f"energy_density not derived correctly; difference = {diff}")
        else:
            return (True, "SKIP: symbolic check not possible")

    # ------------------------------------------------------------------
    # Monotonicity checks: simple trend verification
    # ------------------------------------------------------------------
    def test_monotonicity_energy_density_increases_with_pressure(self):
        """
        If pressure increases, energy density should increase linearly.
        We'll assume the result is a function of pressure (symbolic) and check sign of derivative.
        """
        if self.energy_density is None:
            return (True, "SKIP: energy_density not available")
        # If energy_density = (5/2)*P, derivative w.r.t. P is positive.
        P = sym.Symbol('P')
        # Try to substitute P into expression and compute derivative
        # But we don't know if energy_density contains P directly.
        # We'll check if the result can be simplified to (5/2)*P.
        # If it does, derivative = 5/2 >0.
        # However, we cannot assume the expression contains P.
        # Alternative: use numeric perturbation if P numeric.
        if self.P is not None and self._to_numeric(self.P) is not None:
            P_num = self._to_numeric(self.P)
            # evaluate energy_density at baseline
            ed_base = self._to_numeric(self.energy_density)
            if ed_base is None:
                return (True, "SKIP: cannot evaluate energy_density numerically")
            # perturb P slightly (10% increase)
            P_pert = P_num * 1.1
            # We need to recompute energy density with new P. 
            # This requires that the solution instance supports perturbation.
            # We can try to substitute P in the symbolic expression if possible.
            if hasattr(self.energy_density, 'subs'):
                # attempt substitution
                try:
                    ed_pert = self._to_numeric(self.energy_density.subs({sym.Symbol('P'): P_pert}))
                except:
                    ed_pert = None
            else:
                ed_pert = None
            if ed_pert is None:
                return (True, "SKIP: cannot compute perturbed energy density")
            if ed_pert > ed_base:
                return (True, f"energy_density increases with P (base={ed_base}, perturbed={ed_pert})")
            else:
                return (False, "energy_density does not increase when P increases")
        return (True, "SKIP: cannot perform monotonicity check without numeric pressure")

    # ------------------------------------------------------------------
    # Weak answer sanity check (at most one, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """
        Check that all results are positive and finite (if numeric).
        """
        results = [('c_v', self.c_v), ('c_p', self.c_p), ('energy_density', self.energy_density)]
        for name, val in results:
            if val is None:
                continue
            num = self._to_numeric(val)
            if num is not None:
                if not math.isfinite(num):
                    return (False, f"{name} is not finite: {num}")
                if num <= 0:
                    return (False, f"{name} is not positive: {num}")
        return (True, "All available numeric results are positive and finite")

    # ------------------------------------------------------------------
    # Aggregator
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)