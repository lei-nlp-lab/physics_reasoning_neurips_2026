import sympy as sp
import math
import random
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Checks a solution for the charge density induced on a conducting plane
    by a fixed electric dipole.
    """

    def __init__(self, solution_instance: Any):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.sigma = results.get('sigma', None)

        # Extract givens safely; no numeric fallbacks
        givens = getattr(solution_instance, 'givens', {})
        self.P = givens.get('P', None)
        self.theta = givens.get('theta', None)
        self.z0 = givens.get('z0', None)
        self.epsilon0 = givens.get('epsilon0', None)

    # ------------------------------------------------------------------
    # Helper: evaluate sigma numerically at a point
    # ------------------------------------------------------------------
    def _eval_sigma_numeric(self, x_val: float, y_val: float,
                            P_val: float = 1.0, theta_val: float = 0.5,
                            z0_val: float = 1.0) -> float:
        if self.sigma is None:
            raise ValueError("sigma missing")
        expr = sp.sympify(self.sigma)
        val = expr.subs({sp.Symbol('x'): x_val,
                         sp.Symbol('y'): y_val,
                         sp.Symbol('P'): P_val,
                         sp.Symbol('theta'): theta_val,
                         sp.Symbol('z0'): z0_val})
        return float(val.evalf())

    # ------------------------------------------------------------------
    # run_all_tests – discover and run all test_ methods
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_sigma_exists(self) -> Tuple[bool, str]:
        if self.sigma is None:
            return (False, "SKIP: sigma not provided")
        return (True, "sigma present")

    def test_interface_sigma_is_expression(self) -> Tuple[bool, str]:
        if self.sigma is None:
            return (False, "SKIP: sigma missing")
        try:
            sp.sympify(self.sigma)
            return (True, "sigma can be parsed as sympy expression")
        except Exception as e:
            return (False, f"sigma parse error: {e}")

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        missing = []
        for name in ['P', 'theta', 'z0', 'epsilon0']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Theorem-based invariants
    # ------------------------------------------------------------------
    def test_invariant_total_charge_zero(self) -> Tuple[bool, str]:
        """
        The total induced charge must be zero (dipole has no net charge).
        Numerically integrate sigma over a large disk.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma missing")
        if self.P is None or self.z0 is None or self.theta is None:
            return (False, "SKIP: missing givens for numeric integration")

        P_val = float(self.P)
        z0_val = float(self.z0)
        theta_val = float(self.theta)

        # Use a disk of radius 20*z0, with 200 radial bins and 20 angular samples
        r_max = 20 * z0_val
        N_r = 200
        N_phi = 20
        dr = r_max / N_r
        total = 0.0
        for i in range(N_r):
            r = (i + 0.5) * dr
            avg_sigma = 0.0
            for j in range(N_phi):
                phi = (j + 0.5) * (2 * math.pi / N_phi)
                x = r * math.cos(phi)
                y = r * math.sin(phi)
                val = self._eval_sigma_numeric(x, y, P_val, theta_val, z0_val)
                avg_sigma += val
            avg_sigma /= N_phi
            total += avg_sigma * (2 * math.pi * r * dr)

        if math.isclose(total, 0.0, abs_tol=1e-2):
            return (True, f"Total charge ≈0 (integral = {total:.4e})")
        else:
            return (False, f"Total charge not zero: {total:.4e}")

    def test_invariant_field_tangential_zero(self) -> Tuple[bool, str]:
        """
        On the conducting plane (z=0), Ex and Ey must vanish (from V = V_dip + V_img).
        Uses symbolic computation and a few numeric checks.
        """
        if self.P is None or self.theta is None or self.z0 is None:
            return (False, "SKIP: missing givens for field computation")

        # Symbolic variables
        P, theta, z0, x, y, z, eps0 = sp.symbols('P theta z0 x y z epsilon0',
                                                  positive=True, real=True)
        r_vec = sp.Matrix([x, y, z])
        r0_vec = sp.Matrix([0, 0, z0])
        r_img_vec = sp.Matrix([0, 0, -z0])
        P_vec = sp.Matrix([P*sp.sin(theta), 0, P*sp.cos(theta)])
        P_img_vec = sp.Matrix([-P*sp.sin(theta), 0, P*sp.cos(theta)])

        def dipole_pot(Pv, rd, rf):
            diff = rf - rd
            r = sp.sqrt(diff.dot(diff))
            return (Pv.dot(diff)) / (4*sp.pi*eps0 * r**3)

        V = dipole_pot(P_vec, r0_vec, r_vec) + dipole_pot(P_img_vec, r_img_vec, r_vec)
        Ex = -sp.diff(V, x)
        Ey = -sp.diff(V, y)
        Ex0 = sp.simplify(Ex.subs(z, 0))
        Ey0 = sp.simplify(Ey.subs(z, 0))

        # Try symbolic zero
        if Ex0 == 0 and Ey0 == 0:
            return (True, "Ex and Ey are identically zero on plane (symbolic)")

        # Numeric spot check
        passed = True
        for _ in range(5):
            xv = random.uniform(-3, 3)
            yv = random.uniform(-3, 3)
            subs = {x: xv, y: yv, P: 1.0, theta: math.pi/4, z0: 1.0, eps0: 1.0}
            try:
                ex_val = float(Ex0.subs(subs).evalf())
                ey_val = float(Ey0.subs(subs).evalf())
                if abs(ex_val) > 1e-5 or abs(ey_val) > 1e-5:
                    passed = False
                    break
            except Exception:
                passed = False
                break
        if passed:
            return (True, "Ex and Ey vanish on plane (numeric spot check)")
        else:
            return (False, "Ex and/or Ey not zero on plane")

    # ------------------------------------------------------------------
    # Modeling / trace consistency
    # ------------------------------------------------------------------
    def test_modeling_sigma_from_derivative(self) -> Tuple[bool, str]:
        """
        Compute sigma = -epsilon0 dV/dz|_{z=0} symbolically and compare numerically
        to the provided sigma at several points.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma missing")
        if self.P is None or self.theta is None or self.z0 is None:
            return (False, "SKIP: missing givens")

        # Symbolic computation of sigma from V
        P, theta, z0, x, y, z, eps0 = sp.symbols('P theta z0 x y z epsilon0',
                                                  positive=True, real=True)
        r_vec = sp.Matrix([x, y, z])
        r0_vec = sp.Matrix([0, 0, z0])
        r_img_vec = sp.Matrix([0, 0, -z0])
        P_vec = sp.Matrix([P*sp.sin(theta), 0, P*sp.cos(theta)])
        P_img_vec = sp.Matrix([-P*sp.sin(theta), 0, P*sp.cos(theta)])

        def dipole_pot(Pv, rd, rf):
            diff = rf - rd
            r = sp.sqrt(diff.dot(diff))
            return (Pv.dot(diff)) / (4*sp.pi*eps0 * r**3)

        V = dipole_pot(P_vec, r0_vec, r_vec) + dipole_pot(P_img_vec, r_img_vec, r_vec)
        sigma_from_V = -eps0 * sp.diff(V, z).subs(z, 0)
        sigma_from_V_simplified = sp.simplify(sigma_from_V)

        # Compare at several points
        provided = sp.sympify(self.sigma)
        passed = True
        for _ in range(5):
            xv = random.uniform(-3, 3)
            yv = random.uniform(-3, 3)
            Pv = 1.0
            thetav = math.pi/4
            z0v = 1.0
            subs = {x: xv, y: yv, P: Pv, theta: thetav, z0: z0v, eps0: 1.0}
            try:
                val1 = float(sigma_from_V_simplified.subs(subs).evalf())
                val2 = float(provided.subs({sp.Symbol('x'): xv, sp.Symbol('y'): yv,
                                            sp.Symbol('P'): Pv, sp.Symbol('theta'): thetav,
                                            sp.Symbol('z0'): z0v}).evalf())
                if not math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2):
                    passed = False
                    break
            except Exception:
                passed = False
                break
        if passed:
            return (True, "sigma agrees with -epsilon0 dV/dz at z=0")
        else:
            return (False, "sigma does not match derivative")

    def test_modeling_sigma_form(self) -> Tuple[bool, str]:
        """
        Check that sigma contains terms proportional to 1/(rho^2+z0^2)^(3/2)
        and x/(rho^2+z0^2)^(5/2) and 1/(rho^2+z0^2)^(5/2).
        """
        if self.sigma is None:
            return (False, "SKIP: sigma missing")
        try:
            expr = sp.expand(sp.sympify(self.sigma))
        except Exception as e:
            return (False, f"parse error: {e}")

        x, y, z0 = sp.symbols('x y z0', real=True)
        R = sp.sqrt(x**2 + y**2 + z0**2)
        # Check that expr * R^5 is polynomial in x, y, z0 (no powers less than 0)
        # Actually we just check that denominators contain only R^3 and R^5.
        # Get all denominators via together? Instead check that after rationalising, denominator is R^5.
        import sympy as sp
        numer, denom = sp.fraction(sp.together(expr))
        # denom should be something like (x^2+y^2+z0^2)**(5/2) => but sympy may keep sqrt.
        # A robust check: compute expr * (x^2+y^2+z0^2)**(5/2) and see if it's polynomial
        check = sp.simplify(expr * R**5)
        if check.is_polynomial(x, y, z0):
            return (True, "sigma has expected functional form (denominator R^3 and R^5)")
        else:
            return (False, "sigma does not have the expected rational structure")

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_theta_increases_decreases_sigma_at_origin(self) -> Tuple[bool, str]:
        """
        At (x=0,y=0), sigma = (P cosθ) / (π z0^3).  If theta increases,
        cosθ decreases, so sigma should decrease (if P>0).
        We perturb theta by +10% and check that sigma decreases.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma missing")
        if self.P is None or self.z0 is None or self.theta is None:
            return (False, "SKIP: missing givens for monotonicity test")

        P_val = float(self.P)
        z0_val = float(self.z0)
        theta_base = float(self.theta)

        # Baseline sigma at origin
        try:
            sigma_base = self._eval_sigma_numeric(0, 0, P_val, theta_base, z0_val)
        except Exception as e:
            return (False, f"Failed to compute baseline sigma: {e}")

        # Perturb theta: +10%
        theta_pert = theta_base * 1.1
        try:
            sigma_pert = self._eval_sigma_numeric(0, 0, P_val, theta_pert, z0_val)
        except Exception as e:
            return (False, f"Failed to compute perturbed sigma: {e}")

        # Expect sigma to decrease if cos decreases (i.e., theta in (0, pi/2))
        # Check that sigma_pert < sigma_base (within tolerance)
        if sigma_pert < sigma_base:
            return (True, f"Theta increase reduces sigma at origin: base {sigma_base:.4f}, perturbed {sigma_pert:.4f}")
        else:
            return (False, f"Sigma did not decrease with theta increase: base {sigma_base:.4f}, perturbed {sigma_pert:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_real(self) -> Tuple[bool, str]:
        """
        Check that sigma evaluates to finite real numbers for typical inputs.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma missing")
        try:
            for _ in range(5):
                xv = random.uniform(-5, 5)
                yv = random.uniform(-5, 5)
                val = self._eval_sigma_numeric(xv, yv)
                if not math.isfinite(val):
                    return (False, f"sigma not finite at ({xv}, {yv})")
            return (True, "sigma finite at random sample points")
        except Exception as e:
            return (False, f"Evaluation error: {e}")