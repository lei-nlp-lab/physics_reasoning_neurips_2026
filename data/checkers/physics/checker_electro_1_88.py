import sys
import math
import traceback
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution for the electrostatic problem:
      (a) potential of a dipole outside a grounded conducting sphere
      (b) two conducting spheres in a uniform electric field
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely – only keys guaranteed by the pseudocode [emit] contract
        results = getattr(solution_instance, 'results', {})
        self.phi_total = results.get('phi_total', None)       # symbolic expression or numeric
        self.p_total_approx = results.get('p_total_approx', None)  # numeric or symbolic
        self.Q_zz = results.get('Q_zz', None)                 # numeric or symbolic

        # Try to extract problem parameters from the solution instance
        givens = getattr(solution_instance, 'givens', {})
        self.d = givens.get('d', None)          # dipole magnitude
        self.L = givens.get('L', None)          # distance from sphere centre
        self.a = givens.get('a', None)          # sphere radius
        self.epsilon0 = givens.get('epsilon0', None)   # permittivity
        self.E0 = givens.get('E0', None)        # applied uniform field (part b)
        self.R = givens.get('R', None)          # centre-to-centre separation (part b)

        # For sympy operations we always use symbolic eps0 if not provided
        self._eps0 = self.epsilon0 if self.epsilon0 is not None else sp.Symbol('epsilon0', positive=True)

    # ------------------------------------------------------------------
    # interface tests – check that required keys exist and have plausible types
    # ------------------------------------------------------------------
    def test_interface_keys_exist(self) -> Tuple[bool, str]:
        """Check that all three required results are present (not None)."""
        missing = []
        if self.phi_total is None:
            missing.append('phi_total')
        if self.p_total_approx is None:
            missing.append('p_total_approx')
        if self.Q_zz is None:
            missing.append('Q_zz')
        if missing:
            return (False, f"Missing required result(s): {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check that emitted values are sympy expressions or numeric values."""
        issues = []
        if self.phi_total is not None and not isinstance(self.phi_total, (sp.Expr, int, float)):
            issues.append(f"phi_total has unexpected type {type(self.phi_total).__name__}")
        if self.p_total_approx is not None and not isinstance(self.p_total_approx, (sp.Expr, int, float)):
            issues.append(f"p_total_approx has unexpected type {type(self.p_total_approx).__name__}")
        if self.Q_zz is not None and not isinstance(self.Q_zz, (sp.Expr, int, float)):
            issues.append(f"Q_zz has unexpected type {type(self.Q_zz).__name__}")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All results have acceptable types (sympy expression or numeric)")

    # ------------------------------------------------------------------
    # invariant tests – theorem‑based physical sanity checks
    # ------------------------------------------------------------------
    def test_invariant_sphere_potential_zero(self) -> Tuple[bool, str]:
        """
        For part (a): the grounded sphere must have zero potential on its surface.
        Evaluate phi_total at r = a (using sample numerical values if needed)
        and check that it is ≈ 0.
        """
        if self.phi_total is None:
            return (False, "SKIP: phi_total not available")

        # If phi_total is a sympy expression, try symbolic simplification at r = a
        if isinstance(self.phi_total, sp.Expr):
            # define symbols used in the expression (r, theta, L, a, d, eps0)
            r, theta, L, a, d, eps0 = sp.symbols('r theta L a d eps0', positive=True, real=True)
            subs_dict = {r: a}   # only substitute r -> a, leave others symbolic
            potential_on_surface = sp.simplify(self.phi_total.subs(subs_dict))
            # If simplification gives exactly 0, we are done
            if potential_on_surface == 0:
                return (True, "phi_total symbolically vanishes at r = a (surface)")
            # Otherwise try a numeric spot check with typical values
            try:
                # assign numeric values
                num_vals = {a: 1.0, L: 2.5, d: 1.0, eps0: 1.0, theta: 0.5}
                numeric_val = float(potential_on_surface.subs(num_vals).evalf())
                if math.isclose(numeric_val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"phi_total ≈ {numeric_val:.2e} at r = a (numerical check)")
                else:
                    return (False, f"phi_total at r = a gives {numeric_val:.3f}, far from zero")
            except Exception as e:
                return (False, f"Cannot evaluate phi_total at r = a: {e}")

        # If phi_total is numeric (unlikely for a symbolic problem), do direct numeric test
        try:
            val = float(self.phi_total)
            # no direct r dependence; cannot check surface condition properly
            return (False, "SKIP: phi_total appears to be a single number, cannot test r=a")
        except (ValueError, TypeError):
            return (False, "phi_total cannot be evaluated")

    def test_invariant_dipole_moment_self_consistency(self) -> Tuple[bool, str]:
        """
        For part (b): the total induced dipole moment on each sphere must satisfy
            p_total ≈ 4π ε0 a^3 (E0 + p_total/(2π ε0 R^3))
        which to order a^3/R^3 gives  p_total ≈ p_self * (1 + 2 a^3 / R^3).
        Check this relation numerically.
        """
        if self.p_total_approx is None:
            return (False, "SKIP: p_total_approx not available")
        # Need numeric values for a, E0, R, ε0
        if None in (self.a, self.E0, self.R, self.epsilon0):
            return (False, "SKIP: missing givens (a, E0, R, epsilon0) to evaluate relation")
        try:
            a_num = float(self.a)
            E0_num = float(self.E0)
            R_num = float(self.R)
            eps0_num = float(self.epsilon0)
        except (ValueError, TypeError) as e:
            return (False, f"Cannot convert givens to numbers: {e}")

        p_self = 4 * math.pi * eps0_num * a_num**3 * E0_num
        expected_p_total = p_self * (1 + 2 * a_num**3 / R_num**3)

        # Get the numerical value of p_total_approx
        try:
            if isinstance(self.p_total_approx, sp.Expr):
                # substitute numeric values if expression contains symbols
                p_val = float(self.p_total_approx.subs(
                    {sp.Symbol('a'): a_num, sp.Symbol('E0'): E0_num,
                     sp.Symbol('R'): R_num, sp.Symbol('epsilon0'): eps0_num}
                ).evalf())
            else:
                p_val = float(self.p_total_approx)
        except Exception as e:
            return (False, f"Cannot evaluate p_total_approx numerically: {e}")

        if math.isclose(p_val, expected_p_total, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_total_approx ≈ {p_val:.4e}, expected ~ {expected_p_total:.4e}")
        else:
            return (False, f"p_total_approx = {p_val:.4e}, expected ~ {expected_p_total:.4e}")

    def test_invariant_quadrupole_relation(self) -> Tuple[bool, str]:
        """
        Check that the quadrupole moment Q_zz satisfies
            Q_zz ≈ 4π ε0 a^5 * gradE_z
        with gradE_z coming from the other sphere's dipole field:
            gradE_z = -3 p_total / (2π ε0 R^4)
        =>  Q_zz ≈ -6 a^5 p_total / R^4
        """
        if None in (self.Q_zz, self.p_total_approx):
            return (False, "SKIP: Q_zz or p_total_approx missing")
        if None in (self.a, self.R, self.epsilon0):
            return (False, "SKIP: missing givens (a, R, epsilon0)")

        try:
            a_num = float(self.a)
            R_num = float(self.R)
            eps0_num = float(self.epsilon0)
            # p_total value
            if isinstance(self.p_total_approx, sp.Expr):
                p_val = float(self.p_total_approx.subs(
                    {sp.Symbol('a'): a_num, sp.Symbol('R'): R_num,
                     sp.Symbol('epsilon0'): eps0_num}
                ).evalf())
            else:
                p_val = float(self.p_total_approx)
            # Q_zz value
            if isinstance(self.Q_zz, sp.Expr):
                Q_val = float(self.Q_zz.subs(
                    {sp.Symbol('a'): a_num, sp.Symbol('R'): R_num,
                     sp.Symbol('epsilon0'): eps0_num}
                ).evalf())
            else:
                Q_val = float(self.Q_zz)
        except Exception as e:
            return (False, f"Numerical evaluation failed: {e}")

        expected_Q = -6 * a_num**5 * p_val / R_num**4
        if math.isclose(Q_val, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_zz ≈ {Q_val:.4e}, expected ~ {expected_Q:.4e}")
        else:
            return (False, f"Q_zz = {Q_val:.4e}, expected ~ {expected_Q:.4e}")

    # ------------------------------------------------------------------
    # modeling/trace tests – check that expansions and solution structure are correct
    # ------------------------------------------------------------------
    def test_modeling_large_R_limit(self) -> Tuple[bool, str]:
        """
        As R → ∞, the induced dipole moment should approach that of an isolated sphere.
        p_total_approx should tend to p_self = 4π ε0 a^3 E0.
        """
        if self.p_total_approx is None:
            return (False, "SKIP: p_total_approx not available")
        if any(g is None for g in (self.a, self.E0, self.epsilon0)):
            return (False, "SKIP: missing givens (a, E0, epsilon0)")

        try:
            a_num = float(self.a)
            E0_num = float(self.E0)
            eps0_num = float(self.epsilon0)
            p_self = 4 * math.pi * eps0_num * a_num**3 * E0_num
            # use a large R (e.g., 100*a)
            large_R = 100 * a_num
            if isinstance(self.p_total_approx, sp.Expr):
                p_val = float(self.p_total_approx.subs(
                    {sp.Symbol('a'): a_num, sp.Symbol('E0'): E0_num,
                     sp.Symbol('R'): large_R, sp.Symbol('epsilon0'): eps0_num}
                ).evalf())
            else:
                # If it is a fixed number, we cannot check the limit easily
                return (False, "SKIP: p_total_approx is a constant, cannot test R→∞ limit")
        except Exception as e:
            return (False, f"Evaluation error: {e}")

        if math.isclose(p_val, p_self, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"For R={large_R}, p_total_approx ≈ {p_val:.4e} ~ p_self = {p_self:.4e}")
        else:
            return (False, f"For R={large_R}, p_total_approx = {p_val:.4e}, expected ~ {p_self:.4e}")

    def test_modeling_p_total_approx_expansion(self) -> Tuple[bool, str]:
        """
        Check the symbolic expansion of p_total_approx in a/R.
        It should be p_self * (1 + 2*a^3/R^3) to leading order.
        """
        if self.p_total_approx is None:
            return (False, "SKIP: p_total_approx not available")
        if not isinstance(self.p_total_approx, sp.Expr):
            return (False, "SKIP: p_total_approx is not symbolic, cannot expand")

        try:
            # assume the expression uses symbols 'a', 'R', 'E0', 'epsilon0'
            a_sym = sp.Symbol('a', positive=True)
            R_sym = sp.Symbol('R', positive=True)
            E0_sym = sp.Symbol('E0', positive=True)
            eps0_sym = sp.Symbol('epsilon0', positive=True)

            # replace a with epsilon*R to expand in small epsilon = a/R
            epsilon = sp.Symbol('epsilon', positive=True)
            expr_eps = self.p_total_approx.subs({a_sym: epsilon * R_sym})

            # series expansion up to epsilon^3
            series = sp.series(expr_eps, epsilon, 0, 4).removeO()

            # substitute back epsilon = a/R
            expanded = series.subs(epsilon, a_sym / R_sym)
            expanded_simplified = sp.simplify(expanded)

            # expected expansion: p_self * (1 + 2*a^3/R^3)
            p_self = 4 * sp.pi * eps0_sym * a_sym**3 * E0_sym
            expected = p_self * (1 + 2 * a_sym**3 / R_sym**3)

            if sp.simplify(expanded_simplified - expected) == 0:
                return (True, "p_total_approx expands correctly to order a^3/R^3")
            else:
                return (False, f"Expansion mismatch: got {expanded_simplified}, expected {expected}")
        except Exception as e:
            return (False, f"Expansion test error: {e}")

    # ------------------------------------------------------------------
    # monotonicity tests – perturb an input and check the direction of change
    # ------------------------------------------------------------------
    def test_monotonicity_potential_decreases_with_L(self) -> Tuple[bool, str]:
        """
        For part (a): as the dipole moves farther away (L increases),
        the magnitude of the potential at a fixed point outside the sphere
        should decrease.
        """
        if self.phi_total is None or self.L is None:
            return (False, "SKIP: phi_total or L not available")
        if not isinstance(self.phi_total, sp.Expr):
            return (False, "SKIP: phi_total is not symbolic, cannot perturb L")

        try:
            r, theta, L, a, d, eps0 = sp.symbols('r theta L a d eps0', positive=True, real=True)
            # choose a test point outside the sphere
            test_r = 2.0 * float(self.a) if self.a is not None else 2.0
            test_theta = 0.3
            # baseline L and perturbed L (larger)
            L_base = float(self.L)
            L_pert = L_base * 1.5   # 50% increase
            # prepare substitution dictionaries
            subs_base = {r: test_r, theta: test_theta, L: L_base, a: test_r/2, d: 1.0, eps0: 1.0}
            subs_pert = dict(subs_base)
            subs_pert[L] = L_pert

            phi_base = float(self.phi_total.subs(subs_base).evalf())
            phi_pert = float(self.phi_total.subs(subs_pert).evalf())

            if abs(phi_pert) < abs(phi_base):
                return (True, f"|phi| decreases: base L={L_base:.2f} -> |phi|={abs(phi_base):.4e}, perturb L={L_pert:.2f} -> |phi|={abs(phi_pert):.4e}")
            else:
                return (False, f"|phi| increased or stayed same: base |phi|={abs(phi_base):.4e}, perturb |phi|={abs(phi_pert):.4e}")
        except Exception as e:
            return (False, f"Evaluation error: {e}")

    def test_monotonicity_pmoment_decreases_with_R(self) -> Tuple[bool, str]:
        """
        For part (b): as the separation R increases, the induced dipole moment
        p_total_approx should decrease (the interaction correction decreases).
        """
        if self.p_total_approx is None or self.R is None:
            return (False, "SKIP: p_total_approx or R not available")
        if None in (self.a, self.E0, self.epsilon0):
            return (False, "SKIP: missing a, E0, epsilon0 for evaluation")

        try:
            a_num = float(self.a)
            E0_num = float(self.E0)
            eps0_num = float(self.epsilon0)
            R_base = float(self.R)
            R_pert = R_base * 1.2   # 20% increase

            if isinstance(self.p_total_approx, sp.Expr):
                subs = {sp.Symbol('a'): a_num, sp.Symbol('E0'): E0_num,
                        sp.Symbol('epsilon0'): eps0_num}
                p_base = float(self.p_total_approx.subs({**subs, sp.Symbol('R'): R_base}).evalf())
                p_pert = float(self.p_total_approx.subs({**subs, sp.Symbol('R'): R_pert}).evalf())
            else:
                # numeric constant: cannot test monotonicity
                return (False, "SKIP: p_total_approx is a constant, cannot test R variation")

            if p_pert < p_base:
                return (True, f"p_total decreases: base R={R_base:.2f} -> {p_base:.4e}, perturb R={R_pert:.2f} -> {p_pert:.4e}")
            else:
                return (False, f"p_total increased or stayed same: base {p_base:.4e}, perturb {p_pert:.4e}")
        except Exception as e:
            return (False, f"Evaluation error: {e}")

    # ------------------------------------------------------------------
    # weak answer sanity (single test, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self) -> Tuple[bool, str]:
        """
        Check that results are real, finite, and have plausible signs.
        Does NOT compare against expected numeric values.
        """
        issues = []
        # phi_total
        if self.phi_total is not None:
            try:
                if isinstance(self.phi_total, sp.Expr):
                    r, theta, L, a, d, eps0 = sp.symbols('r theta L a d eps0')
                    subs = {r: 2.0, theta: 0.5, L: 3.0, a: 1.0, d: 1.0, eps0: 1.0}
                    val = complex(self.phi_total.subs(subs).evalf())
                    if val.imag != 0 and abs(val.imag) > 1e-10:
                        issues.append("phi_total has imaginary part")
                    elif math.isnan(val.real):
                        issues.append("phi_total is NaN")
                else:
                    val = float(self.phi_total)
                    if math.isnan(val):
                        issues.append("phi_total is NaN")
            except Exception as e:
                issues.append(f"Cannot evaluate phi_total: {e}")

        # p_total_approx
        if self.p_total_approx is not None:
            try:
                if isinstance(self.p_total_approx, sp.Expr):
                    a_val = float(self.a) if self.a is not None else 1.0
                    E0_val = float(self.E0) if self.E0 is not None else 1.0
                    R_val = float(self.R) if self.R is not None else 10.0
                    eps0_val = float(self.epsilon0) if self.epsilon0 is not None else 1.0
                    subs = {sp.Symbol('a'): a_val, sp.Symbol('E0'): E0_val,
                            sp.Symbol('R'): R_val, sp.Symbol('epsilon0'): eps0_val}
                    p_val = float(self.p_total_approx.subs(subs).evalf())
                else:
                    p_val = float(self.p_total_approx)
                if math.isnan(p_val) or math.isinf(p_val):
                    issues.append("p_total_approx is NaN or Inf")
                elif p_val < 0:
                    issues.append("p_total_approx is negative (should be positive for parallel field)")
            except Exception as e:
                issues.append(f"Cannot evaluate p_total_approx: {e}")

        # Q_zz
        if self.Q_zz is not None:
            try:
                if isinstance(self.Q_zz, sp.Expr):
                    a_val = float(self.a) if self.a is not None else 1.0
                    R_val = float(self.R) if self.R is not None else 10.0
                    eps0_val = float(self.epsilon0) if self.epsilon0 is not None else 1.0
                    subs = {sp.Symbol('a'): a_val, sp.Symbol('R'): R_val,
                            sp.Symbol('epsilon0'): eps0_val}
                    q_val = complex(self.Q_zz.subs(subs).evalf())
                else:
                    q_val = complex(self.Q_zz)
                if math.isnan(q_val.real) or math.isinf(q_val.real):
                    issues.append("Q_zz is NaN or Inf")
            except Exception as e:
                issues.append(f"Cannot evaluate Q_zz: {e}")

        if not issues:
            return (True, "All results appear finite, real, and with plausible signs")
        return (False, "; ".join(issues))

    # ------------------------------------------------------------------
    # main test runner – discover all test_ methods
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)