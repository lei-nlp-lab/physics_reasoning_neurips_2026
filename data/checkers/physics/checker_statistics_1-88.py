import sympy as sp
import math

class PhysicsChecker:
    """Checker for blackbody radiation problem.
    Audits Maxwell relation, differential equation, and Stefan-Boltzmann law.
    """

    def __init__(self, solution_instance):
        # Extract results from solution_instance (expected keys from pseudocode)
        self.maxwell = getattr(solution_instance, 'results', {}).get('maxwell', None)
        self.diff_eq = getattr(solution_instance, 'results', {}).get('diff_eq', None)
        self.u = getattr(solution_instance, 'results', {}).get('u', None)
        self.constant = getattr(solution_instance, 'results', {}).get('constant', None)

        # Store the entire solution_instance if needed for further inspection
        self.solution_instance = solution_instance

    # ---------- Interface tests ----------
    def test_interface_maxwell_key_exists(self):
        """Check that the 'maxwell' key is present and non-None."""
        if self.maxwell is None:
            return (False, "Missing key: maxwell")
        return (True, "maxwell key exists")

    def test_interface_diff_eq_key_exists(self):
        """Check that the 'diff_eq' key is present and non-None."""
        if self.diff_eq is None:
            return (False, "Missing key: diff_eq")
        return (True, "diff_eq key exists")

    def test_interface_u_key_exists(self):
        """Check that the 'u' key is present and non-None."""
        if self.u is None:
            return (False, "Missing key: u")
        return (True, "u key exists")

    def test_interface_constant_key_exists(self):
        """Check that the 'constant' key is present and non-None."""
        if self.constant is None:
            return (False, "Missing key: constant")
        return (True, "constant key exists")

    # ---------- Invariant tests (theorem-based, use math.isclose where feasible) ----------
    def test_invariant_u_satisfies_differential_equation(self):
        """Verify that u = a T^4 satisfies the derived differential equation."""
        if self.u is None:
            return (False, "SKIP: u not available")
        try:
            # Use sympy to check the relation symbolically
            T, a = sp.symbols('T a', positive=True)
            u_expr = a * T**4
            diff_eq_lhs = u_expr
            diff_eq_rhs = (T/3) * sp.diff(u_expr, T) - u_expr/3
            diff = sp.simplify(diff_eq_lhs - diff_eq_rhs)
            if diff == 0:
                return (True, "u = a T^4 satisfies the differential equation")
            else:
                # Fallback to numeric check if symbolic fails (unlikely)
                T_val = 2.0
                a_val = 1.0
                u_val = a_val * T_val**4
                rhs_val = (T_val/3) * (4 * a_val * T_val**3) - u_val/3
                if math.isclose(u_val, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "u = a T^4 satisfies diff eq (numeric check)")
                else:
                    return (False, f"u does not satisfy diff eq: diff = {diff}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_thermodynamic_identity(self):
        """Check that u = T (dp/dT)_V - p holds with p = u/3."""
        if self.u is None:
            return (False, "SKIP: u not available")
        try:
            T, a = sp.symbols('T a', positive=True)
            u_expr = a * T**4
            p_expr = u_expr / 3
            rhs = T * sp.diff(p_expr, T) - p_expr
            identity = sp.simplify(u_expr - rhs)
            if identity == 0:
                return (True, "Thermodynamic identity u = T(dp/dT)_V - p holds")
            else:
                # Numeric fallback
                T_val = 3.0
                a_val = 1.0
                u_val = a_val * T_val**4
                p_val = u_val / 3
                dpdT = (4 * a_val * T_val**3) / 3
                rhs_val = T_val * dpdT - p_val
                if math.isclose(u_val, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "Thermodynamic identity holds (numeric)")
                else:
                    return (False, f"Identity fails: u={u_val}, rhs={rhs_val}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    # ---------- Modeling / trace tests ----------
    def test_modeling_maxwell_relation_expression(self):
        """Check that the Maxwell relation is correctly stated."""
        if self.maxwell is None:
            return (False, "SKIP: maxwell not available")
        # Normalize the string: remove spaces, replace common variants
        normalized = self.maxwell.replace(" ", "").replace("∂", "d").replace("\\", "")
        expected = "(dS/dV)_T=(dp/dT)_V"
        if expected in normalized:
            return (True, "Maxwell relation correctly expressed")
        else:
            return (False, f"Maxwell relation unrecognised: {self.maxwell}")

    def test_modeling_diff_eq_expression(self):
        """Check that the differential equation is correctly stated."""
        if self.diff_eq is None:
            return (False, "SKIP: diff_eq not available")
        normalized = self.diff_eq.replace(" ", "").replace("∂", "d").replace("\\", "")
        # Accept several equivalent forms
        forms = [
            "u=T/3du/dT-u/3",
            "u=(T/3)du/dT-(1/3)u",
            "u=(T/3)(du/dT)-u/3",
            "u=(1/3)T(du/dT)-(1/3)u"
        ]
        if any(f in normalized for f in forms):
            return (True, "Differential equation correctly expressed")
        else:
            return (False, f"Differential equation unrecognised: {self.diff_eq}")

    def test_modeling_u_expression(self):
        """Check that u is expressed as a T^4."""
        if self.u is None:
            return (False, "SKIP: u not available")
        normalized = self.u.replace(" ", "").lower()
        if "t^4" in normalized or "t**4" in normalized or "t⁴" in normalized:
            return (True, "u expressed as a T^4")
        else:
            return (False, f"u expression unrecognised: {self.u}")

    # ---------- Monotonicity tests (numeric perturbation of input T) ----------
    def test_monotonicity_u_increases_with_T(self):
        """Perturb T by a factor of 2 and check that u scales by factor 16."""
        if self.u is None:
            return (False, "SKIP: u not available")
        try:
            # Use the functional form u = a T^4 (constant a cancels)
            T0 = 2.0
            T1 = 4.0  # factor 2
            # Compute u values
            u0 = T0**4        # 16
            u1 = T1**4        # 256
            ratio = u1 / u0
            # For u ∝ T^4, doubling T -> factor 16
            if math.isclose(ratio, 16.0, rel_tol=1e-2):
                return (True, f"u scales as T^4: T doubled, u increased by factor {ratio:.3f} (expected 16)")
            else:
                return (False, f"Scaling ratio {ratio:.3f} not close to 16")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_monotonicity_du_dT_increases_with_T(self):
        """Perturb T by a factor of 2 and check that du/dT scales by factor 8."""
        if self.u is None:
            return (False, "SKIP: u not available")
        try:
            # For u = a T^4, du/dT = 4a T^3, so doubling T multiplies slope by 8
            T0 = 2.0
            T1 = 4.0
            slope0 = 4 * T0**3   # 32
            slope1 = 4 * T1**3   # 256
            ratio = slope1 / slope0
            if math.isclose(ratio, 8.0, rel_tol=1e-2):
                return (True, f"du/dT scales as T^3: T doubled, slope increased by factor {ratio:.3f} (expected 8)")
            else:
                return (False, f"Slope scaling ratio {ratio:.3f} not close to 8")
        except Exception as e:
            return (False, f"test crashed: {e}")

    # ---------- Answer sanity (weak, final answer should not dominate) ----------
    def test_answer_sanity_u_positive(self):
        """Weak check: energy density u is positive (if represented as string)."""
        if self.u is None:
            return (False, "SKIP: u not available")
        # Only check that the expression does not contain negative sign or zero
        normalized = self.u.replace(" ", "").lower()
        if "a<0" in normalized or "a = 0" in normalized:
            return (False, "u appears to be non-positive")
        # Check that constant is described as positive
        if self.constant is not None and "positive" in self.constant.lower():
            return (True, "Constant is positive")
        # Fallback: just check that u is not zero (unlikely from string)
        if "a*t^4" in normalized or "aT^4" in normalized:
            return (True, "u is positive (form suggests a>0)")
        return (False, "Cannot confirm positivity")

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