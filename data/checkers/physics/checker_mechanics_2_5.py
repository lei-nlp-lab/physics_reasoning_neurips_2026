import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Checker for the rotating-frame velocity-dependent potential problem.
    Validates that the derived U has the correct form and physics properties
    using theorem-based and modeling checks. No hardcoded target expression comparisons.
    """

    def __init__(self, solution_instance):
        # Safely extract U from results
        self.U = solution_instance.results.get('U', None)

        # Convert string to sympy expression if needed
        self._U_sym = None
        if self.U is not None:
            if isinstance(self.U, str):
                try:
                    self._U_sym = sp.parsing.sympy_parser.parse_expr(self.U)
                except Exception:
                    self._U_sym = None
            elif isinstance(self.U, sp.Basic):
                self._U_sym = self.U
            else:
                self._U_sym = None

        # Symbols for independent variables (plain symbols for matching)
        self.m, self.omega = sp.symbols('m omega', real=True, positive=True)
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.xdot, self.ydot, self.zdot = sp.symbols('xdot ydot zdot', real=True)

        # Symbols for function-of-time approach (force derivation)
        self.t = sp.symbols('t', real=True)
        self.x_f = sp.Function('x')(self.t)
        self.y_f = sp.Function('y')(self.t)
        self.z_f = sp.Function('z')(self.t)

        # Expected force from U (derived from physics: Coriolis + centrifugal)
        # These are theorem-based, not from a target answer.
        self.expected_Fx = 2 * self.m * self.omega * sp.diff(self.y_f, self.t) + self.m * self.omega**2 * self.x_f
        self.expected_Fy = -2 * self.m * self.omega * sp.diff(self.x_f, self.t) + self.m * self.omega**2 * self.y_f
        self.expected_Fz = sp.Integer(0)

    # ----------------------------------------------------------------------
    # Helper: convert plain-symbol U to function-of-time U
    # ----------------------------------------------------------------------
    def _to_functional_U(self, U_sym):
        """
        Attempt to convert U_sym (with plain symbols) to an expression
        using function-of-time variables. Returns None if conversion fails.
        """
        if U_sym is None:
            return None
        subs = {
            self.x: self.x_f,
            self.y: self.y_f,
            self.z: self.z_f,
            self.xdot: sp.diff(self.x_f, self.t),
            self.ydot: sp.diff(self.y_f, self.t),
            self.zdot: sp.diff(self.z_f, self.t),
        }
        try:
            return U_sym.subs(subs)
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Interface check
    # ----------------------------------------------------------------------
    def test_interface_key_exists(self):
        if self.U is None:
            return (False, "SKIP: key 'U' missing in solution.results")
        if isinstance(self.U, str) or isinstance(self.U, sp.Basic):
            return (True, f"U present as {type(self.U).__name__}")
        return (False, f"U has unsupported type: {type(self.U).__name__}")

    # ----------------------------------------------------------------------
    # Invariant: U depends on velocities correctly (xdot, ydot but not zdot)
    # ----------------------------------------------------------------------
    def test_invariant_U_velocity_dependence(self):
        if self._U_sym is None:
            return (False, "SKIP: no sympy U available")
        try:
            dU_dxdot = sp.diff(self._U_sym, self.xdot)
            dU_dydot = sp.diff(self._U_sym, self.ydot)
            dU_dzdot = sp.diff(self._U_sym, self.zdot)
        except Exception as e:
            return (False, f"Derivative computation failed: {e}")
        # Check that dxdot and dydot are not zero
        if dU_dxdot == 0:
            return (False, "U does not depend on xdot (should, for Coriolis term)")
        if dU_dydot == 0:
            return (False, "U does not depend on ydot (should, for Coriolis term)")
        # Check zdot derivative is zero (no z motion)
        if dU_dzdot != 0:
            return (False, f"U unexpectedly depends on zdot (derivative = {dU_dzdot})")
        return (True, "U depends on xdot, ydot but not zdot")

    # ----------------------------------------------------------------------
    # Invariant: Force derived from U matches the known rotating-frame forces
    # ----------------------------------------------------------------------
    def test_invariant_U_force(self):
        if self._U_sym is None:
            return (False, "SKIP: no sympy U available")
        U_func = self._to_functional_U(self._U_sym)
        if U_func is None:
            return (False, "SKIP: cannot convert U to functional form")
        try:
            dU_dxdot = sp.diff(U_func, sp.diff(self.x_f, self.t))
            dU_dydot = sp.diff(U_func, sp.diff(self.y_f, self.t))
            dU_dzdot = sp.diff(U_func, sp.diff(self.z_f, self.t))

            ddt_dU_dxdot = sp.diff(dU_dxdot, self.t)
            ddt_dU_dydot = sp.diff(dU_dydot, self.t)
            ddt_dU_dzdot = sp.diff(dU_dzdot, self.t)

            dU_dx = sp.diff(U_func, self.x_f)
            dU_dy = sp.diff(U_func, self.y_f)
            dU_dz = sp.diff(U_func, self.z_f)

            # Force components
            Fx = ddt_dU_dxdot - dU_dx
            Fy = ddt_dU_dydot - dU_dy
            Fz = ddt_dU_dzdot - dU_dz
        except Exception as e:
            return (False, f"Force derivation failed: {e}")

        # Simplify differences with expected forces (theorem-based)
        diff_x = sp.simplify(Fx - self.expected_Fx)
        diff_y = sp.simplify(Fy - self.expected_Fy)
        diff_z = sp.simplify(Fz - self.expected_Fz)

        if diff_x == 0 and diff_y == 0 and diff_z == 0:
            return (True, "Force from U matches Coriolis+centrifugal forces")
        else:
            return (False,
                    f"Force mismatches:\n  Fx diff: {diff_x}\n  Fy diff: {diff_y}\n  Fz diff: {diff_z}")

    # ----------------------------------------------------------------------
    # Modeling / trace: U is a polynomial of appropriate degree
    # ----------------------------------------------------------------------
    def test_modeling_U_polynomial_degree(self):
        if self._U_sym is None:
            return (False, "SKIP: no sympy U available")
        vars_ = [self.x, self.y, self.xdot, self.ydot]
        try:
            poly = sp.Poly(self._U_sym, *vars_)
        except sp.PolynomialError:
            return (False, "U is not a polynomial in x,y,xdot,ydot")
        monoms = poly.monoms()
        for mon in monoms:
            deg_x = mon[0]
            deg_y = mon[1]
            deg_xdot = mon[2]
            deg_ydot = mon[3]
            if (deg_x + deg_y) > 2 or (deg_xdot + deg_ydot) > 1:
                return (False, f"U contains higher-degree monomial: {mon}")
        return (True, f"U is a polynomial of total degree ≤2 in positions and ≤1 in velocities")

    # ----------------------------------------------------------------------
    # Modeling / trace: U contains the correct Coriolis structure
    # ----------------------------------------------------------------------
    def test_modeling_U_coriolis_structure(self):
        if self._U_sym is None:
            return (False, "SKIP: no sympy U available")
        expanded = sp.expand(self._U_sym)
        # Coefficient of x*ydot should be -m*omega
        coeff = expanded.coeff(self.x * self.ydot)
        expected_coeff = -self.m * self.omega
        if sp.simplify(coeff - expected_coeff) != 0:
            return (False, f"coefficient of x*ydot is {coeff}, expected {expected_coeff}")
        # Coefficient of xdot*y should be +m*omega
        coeff2 = expanded.coeff(self.xdot * self.y)
        expected_coeff2 = self.m * self.omega
        if sp.simplify(coeff2 - expected_coeff2) != 0:
            return (False, f"coefficient of xdot*y is {coeff2}, expected {expected_coeff2}")
        return (True, "U contains correct Coriolis terms (x*ydot and xdot*y)")

    # ----------------------------------------------------------------------
    # Monotonicity: For fixed position and zero velocity, U decreases with omega
    # ----------------------------------------------------------------------
    def test_monotonicity_U_vs_omega(self):
        if self._U_sym is None:
            return (False, "SKIP: no sympy U available")
        # Baseline parameters
        base_subs = {self.m: 1, self.x: 1, self.y: 1, self.xdot: 0, self.ydot: 0}
        try:
            # Baseline at omega = 1
            baseline = self._U_sym.subs({**base_subs, self.omega: 1.0}).evalf()
            # Perturb: increase omega by 10% (omega = 1.1)
            perturbed = self._U_sym.subs({**base_subs, self.omega: 1.1}).evalf()
            # Additionally, another perturbation at 20% increase for robustness
            perturbed2 = self._U_sym.subs({**base_subs, self.omega: 1.2}).evalf()
        except Exception as e:
            return (False, f"Evaluation failed: {e}")

        # Check that both perturbed values are less than baseline (U becomes more negative)
        if perturbed < baseline and perturbed2 < baseline:
            return (True, f"U decreases when omega increases (baseline={baseline}, +10%={perturbed}, +20%={perturbed2})")
        else:
            return (False, f"U does not monotonically decrease with omega: baseline={baseline}, +10%={perturbed}, +20%={perturbed2}")

    # ----------------------------------------------------------------------
    # Weak answer sanity: for canonical test point, U is negative
    # ----------------------------------------------------------------------
    def test_answer_sanity_U_sign(self):
        if self._U_sym is None:
            return (False, "SKIP: no sympy U available")
        subs = {self.m: 1, self.omega: 1, self.x: 1, self.y: 1, self.xdot: 0, self.ydot: 0}
        try:
            U_val = self._U_sym.subs(subs).evalf()
        except Exception as e:
            return (False, f"Evaluation failed: {e}")
        if U_val < 0:
            return (True, f"U is negative (value={U_val}) as expected for this non-rotating case")
        else:
            return (False, f"U is not negative for a static particle at (1,1); got {U_val}")

    # ----------------------------------------------------------------------
    # Test suite runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)