import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Callable

class PhysicsChecker:
    """
    Robust physics checker for neutron spin precession problem.
    Covers interface, invariants, modeling/trace, and monotonicity.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract results dict (default to empty dict)
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        # Safely extract givens (attributes on instance or inside results)
        self.mu_n = getattr(solution_instance, 'mu_n', None)
        self.B1 = getattr(solution_instance, 'B1', None)
        self.B2 = getattr(solution_instance, 'B2', None)
        self.v = getattr(solution_instance, 'v', None)
        self.d = getattr(solution_instance, 'd', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.omega2 = getattr(solution_instance, 'omega2', None)
        self.t = getattr(solution_instance, 't', None)  # may be symbolic

        # Result keys: 'state', 'p', 'cond'
        self.state = results.get('state', None)
        self.p = results.get('p', None)
        self.cond = results.get('cond', None)

    # ----------------------------------------------------------------
    # test_interface_* methods
    # ----------------------------------------------------------------
    def test_interface_state_exists(self) -> Tuple[bool, str]:
        if self.state is None:
            return (False, "SKIP: state not provided")
        if not isinstance(self.state, (list, tuple)) or len(self.state) != 2:
            return (False, "state must be a list/tuple of length 2")
        return (True, "state is a two-component list/tuple")

    def test_interface_p_exists(self) -> Tuple[bool, str]:
        if self.p is None:
            return (False, "SKIP: p not provided")
        if not isinstance(self.p, (list, tuple)) or len(self.p) != 3:
            return (False, "p must be a list/tuple of length 3")
        return (True, "p is a three-component list/tuple")

    def test_interface_cond_exists(self) -> Tuple[bool, str]:
        if self.cond is None:
            return (False, "SKIP: cond not provided")
        if not isinstance(self.cond, str):
            return (False, "cond must be a string")
        return (True, "cond is a string")

    # ----------------------------------------------------------------
    # test_invariant_* methods (theorem-based physical sanity)
    # ----------------------------------------------------------------
    def test_invariant_state_norm(self) -> Tuple[bool, str]:
        if self.state is None:
            return (False, "SKIP: state not available")
        try:
            a, b = self.state[0], self.state[1]
            # Attempt numeric evaluation; use dummy substitution if needed
            subs = {}
            if self.t is not None and self.omega2 is not None:
                if hasattr(self.t, 'subs'):
                    subs[self.t] = 1
                    subs[self.omega2] = 1
                else:
                    subs = {}
            a_num = sp.N(a.subs(subs, simultaneous=True)) if hasattr(a, 'subs') else a
            b_num = sp.N(b.subs(subs, simultaneous=True)) if hasattr(b, 'subs') else b
            norm_sq = abs(complex(a_num))**2 + abs(complex(b_num))**2
            if math.isclose(norm_sq, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"norm^2 = {norm_sq:.4f} ≈ 1")
            else:
                return (False, f"norm^2 = {norm_sq:.4f}, expected ~1")
        except Exception as e:
            return (False, f"Could not compute norm: {e}")

    def test_invariant_p_x_zero(self) -> Tuple[bool, str]:
        if self.p is None:
            return (False, "SKIP: p not available")
        px = self.p[0]
        try:
            px_num = sp.N(px) if hasattr(px, 'subs') else px
            if math.isclose(float(px_num), 0.0, abs_tol=1e-2):
                return (True, f"p_x = {float(px_num):.4f} ≈ 0")
            else:
                return (False, f"p_x = {float(px_num):.4f}, expected 0")
        except Exception as e:
            return (False, f"Could not evaluate p_x: {e}")

    def test_invariant_p_yz_sum_squares(self) -> Tuple[bool, str]:
        if self.p is None:
            return (False, "SKIP: p not available")
        py, pz = self.p[1], self.p[2]
        try:
            py_num = sp.N(py) if hasattr(py, 'subs') else py
            pz_num = sp.N(pz) if hasattr(pz, 'subs') else pz
            sum_sq = float(py_num)**2 + float(pz_num)**2
            if math.isclose(sum_sq, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"p_y^2 + p_z^2 = {sum_sq:.4f} ≈ 1")
            else:
                return (False, f"p_y^2 + p_z^2 = {sum_sq:.4f}, expected ~1")
        except Exception as e:
            return (False, f"Could not evaluate p_y, p_z: {e}")

    # ----------------------------------------------------------------
    # test_modeling_* / test_trace_* methods (modeling quality)
    # ----------------------------------------------------------------
    def test_modeling_state_form(self) -> Tuple[bool, str]:
        """
        Check that the state has the expected symbolic form:
        [cos(omega2*t), -i*sin(omega2*t)] (ignoring global phase).
        """
        if self.state is None or self.omega2 is None or self.t is None:
            return (False, "SKIP: state, omega2, or t not available for form check")
        a, b = self.state[0], self.state[1]
        try:
            # Build expected symbolic forms using sympy
            omega_sym = sp.Symbol('omega2')
            t_sym = sp.Symbol('t')
            expected_a = sp.cos(omega_sym * t_sym)
            expected_b = -sp.I * sp.sin(omega_sym * t_sym)
            # Simplify difference
            diff_a = sp.simplify(a - expected_a.subs({omega_sym: self.omega2, t_sym: self.t}))
            diff_b = sp.simplify(b - expected_b.subs({omega_sym: self.omega2, t_sym: self.t}))
            if diff_a == 0 and diff_b == 0:
                return (True, "State matches expected form exactly")
            else:
                # If not exact, try numeric substitution with typical values
                subs = {self.t: 1, self.omega2: 1}
                a_num = sp.N(a.subs(subs))
                b_num = sp.N(b.subs(subs))
                expected_a_num = math.cos(1)
                expected_b_num = -1j * math.sin(1)
                if (math.isclose(abs(complex(a_num) - expected_a_num), 0, abs_tol=1e-2) and
                    math.isclose(abs(complex(b_num) - expected_b_num), 0, abs_tol=1e-2)):
                    return (True, "State matches expected form numerically")
                else:
                    return (False, "State does not match expected form")
        except Exception as e:
            return (False, f"Error checking state form: {e}")

    def test_modeling_p_yz_sinusoidal(self) -> Tuple[bool, str]:
        """
        Check that p_y and p_z follow sin(2ωt) and cos(2ωt) (up to sign).
        """
        if self.p is None or self.omega2 is None or self.t is None:
            return (False, "SKIP: p, omega2, or t not available")
        py, pz = self.p[1], self.p[2]
        try:
            # Expected forms: p_y = -sin(2ωt), p_z = cos(2ωt)
            omega_sym = sp.Symbol('omega2')
            t_sym = sp.Symbol('t')
            expected_py = -sp.sin(2 * omega_sym * t_sym)
            expected_pz = sp.cos(2 * omega_sym * t_sym)
            diff_py = sp.simplify(py - expected_py.subs({omega_sym: self.omega2, t_sym: self.t}))
            diff_pz = sp.simplify(pz - expected_pz.subs({omega_sym: self.omega2, t_sym: self.t}))
            if diff_py == 0 and diff_pz == 0:
                return (True, "p_y and p_z match expected sinusoidal form exactly")
            else:
                # Numeric fallback with test values
                subs = {self.t: 1, self.omega2: 1}
                py_num = float(sp.N(py.subs(subs)))
                pz_num = float(sp.N(pz.subs(subs)))
                expected_py_num = -math.sin(2)
                expected_pz_num = math.cos(2)
                if (math.isclose(py_num, expected_py_num, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(pz_num, expected_pz_num, rel_tol=1e-2, abs_tol=1e-2)):
                    return (True, "p_y and p_z match expected sinusoidal form numerically")
                else:
                    return (False, f"p_y={py_num:.3f} (expected {expected_py_num:.3f}), p_z={pz_num:.3f} (expected {expected_pz_num:.3f})")
        except Exception as e:
            return (False, f"Error checking p_yz form: {e}")

    # ----------------------------------------------------------------
    # test_monotonicity_* methods (explicit input perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_pz_decreases_with_time(self) -> Tuple[bool, str]:
        """
        For small t (near t=0), p_z = cos(2ωt) is decreasing.
        Perturb time by a small positive amount and check that p_z becomes smaller.
        """
        if self.p is None or self.omega2 is None or self.t is None:
            return (False, "SKIP: p, omega2, or t not available")
        if not hasattr(self.omega2, 'subs') and not hasattr(self.t, 'subs'):
            # Both are numeric; but need ability to vary t to compute perturbed output
            # We'll still attempt by treating current values as baseline and perturbing manually.
            pass  # Handle below

        # Extract numeric baseline for t and omega2
        try:
            # Baseline numeric values
            t_val = sp.N(self.t) if hasattr(self.t, 'subs') else self.t
            omega_val = sp.N(self.omega2) if hasattr(self.omega2, 'subs') else self.omega2
            t0 = float(t_val)
            omega0 = float(omega_val)
        except Exception as e:
            return (False, f"Could not extract numeric baseline: {e}")

        # Choose small relative perturbation: +1% of t
        delta = 0.01 * t0 if t0 != 0 else 0.01  # avoid zero t? but t may be symbolic positive
        t1 = t0 + delta

        # Evaluate p_z at t0 and t1 (using the same omega)
        # Need to evaluate p_z expression with substituted time
        pz_expr = self.p[2]
        try:
            if hasattr(pz_expr, 'subs'):
                # Substitute t and omega
                pz0 = float(sp.N(pz_expr.subs({self.t: t0, self.omega2: omega0})))
                pz1 = float(sp.N(pz_expr.subs({self.t: t1, self.omega2: omega0})))
            else:
                # Assume pz is numeric? That would be odd; but handle
                pz0 = float(pz_expr)
                pz1 = float(pz_expr)  # no variation -> skip
                return (False, "SKIP: pz does not depend on time (expected time dependence)")
        except Exception as e:
            return (False, f"Could not evaluate pz at perturbed time: {e}")

        # For small t, pz should decrease as t increases (since cos(2ωt) is decreasing near zero)
        # Check pz1 < pz0 (with tolerance for numerical noise)
        if pz1 < pz0 + 1e-6:  # allow tiny epsilon
            return (True, f"p_z at t={t1:.4f} = {pz1:.4f} < p_z at t={t0:.4f} = {pz0:.4f} (decreasing as expected)")
        else:
            return (False, f"p_z at t={t1:.4f} = {pz1:.4f} is not less than p_z at t={t0:.4f} = {pz0:.4f}")

    def test_monotonicity_py_becomes_more_negative_with_time(self) -> Tuple[bool, str]:
        """
        For small t, p_y = -sin(2ωt) is negative and becomes more negative as t increases.
        Perturb time by a small positive amount and check that p_y decreases (i.e., becomes more negative).
        """
        if self.p is None or self.omega2 is None or self.t is None:
            return (False, "SKIP: p, omega2, or t not available")
        try:
            t_val = sp.N(self.t) if hasattr(self.t, 'subs') else self.t
            omega_val = sp.N(self.omega2) if hasattr(self.omega2, 'subs') else self.omega2
            t0 = float(t_val)
            omega0 = float(omega_val)
        except Exception as e:
            return (False, f"Could not extract numeric baseline: {e}")

        delta = 0.01 * t0 if t0 != 0 else 0.01
        t1 = t0 + delta

        py_expr = self.p[1]
        try:
            if hasattr(py_expr, 'subs'):
                py0 = float(sp.N(py_expr.subs({self.t: t0, self.omega2: omega0})))
                py1 = float(sp.N(py_expr.subs({self.t: t1, self.omega2: omega0})))
            else:
                py0 = float(py_expr)
                py1 = float(py_expr)
                return (False, "SKIP: py does not depend on time")
        except Exception as e:
            return (False, f"Could not evaluate py at perturbed time: {e}")

        # For small t, py is negative and becomes more negative (decreases)
        if py1 < py0 + 1e-6:
            return (True, f"p_y at t={t1:.4f} = {py1:.4f} < p_y at t={t0:.4f} = {py0:.4f} (becomes more negative as expected)")
        else:
            return (False, f"p_y at t={t1:.4f} = {py1:.4f} is not less than p_y at t={t0:.4f} = {py0:.4f}")

    # ----------------------------------------------------------------
    # test_answer_sanity_* method (weak, at most one)
    # ----------------------------------------------------------------
    def test_answer_sanity_cond_keywords(self) -> Tuple[bool, str]:
        """
        Weak sanity: the condition string must contain basic inequality and physical constants.
        """
        if self.cond is None:
            return (False, "SKIP: cond not provided")
        required = ['h', 'mu_n', 'B2', '<', '<<']
        found = all(kw in self.cond for kw in required)
        if found:
            return (True, "cond contains necessary physical constants and inequality")
        else:
            missing = [kw for kw in required if kw not in self.cond]
            return (False, f"cond missing keywords: {missing}")

    # ----------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------
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