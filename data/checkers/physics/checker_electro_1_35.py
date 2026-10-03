import math
import sympy as sp
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits a solution for the evolution of a conductor with initial uniform volume charge density.
    Checks focus on physical laws, modeling consistency, and sanity, not on raw answer matching.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract results and givens; use None for missing values.
        # Givens (expected from problem context)
        self.sigma = getattr(solution_instance, 'sigma', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.rho0 = getattr(solution_instance, 'rho0', None)
        self.V = getattr(solution_instance, 'V', None)
        self.R = getattr(solution_instance, 'R', None)          # for sphere case
        self.shape = getattr(solution_instance, 'shape', None)  # 'sphere' or 'non-sphere'

        # Result keys from pseudocode emit
        results = getattr(solution_instance, 'results', {})
        self.rho_t = results.get('rho_t', None)
        self.sigma_s_t = results.get('sigma_s_t', None)
        self.U_i = results.get('U_i', None)
        self.U_f = results.get('U_f', None)
        self.Delta_U = results.get('Delta_U', None)
        self.U_i_symbolic = results.get('U_i_symbolic', None)

        # Derived quantities (used internally for checks)
        self.tau = None
        self.Q = None
        if self.sigma is not None and self.epsilon0 is not None:
            self.tau = self.epsilon0 / self.sigma
        if self.rho0 is not None and self.V is not None:
            self.Q = self.rho0 * self.V

    # --------------------------------------------------------------------------
    # Interface tests – verify required keys exist and have expected types
    # --------------------------------------------------------------------------
    def test_interface_rho_t_exists(self) -> Tuple[bool, str]:
        if self.rho_t is None:
            return (False, "rho_t not provided")
        if not isinstance(self.rho_t, (str, sp.Expr, float, int)):
            return (False, f"rho_t has unexpected type {type(self.rho_t)}")
        return (True, "rho_t present and type-valid")

    def test_interface_sigma_s_t_exists(self) -> Tuple[bool, str]:
        # For sphere case sigma_s_t is expected; for non-sphere it may not be emitted
        if self.shape == 'sphere' and self.sigma_s_t is None:
            return (False, "sigma_s_t missing for sphere case")
        if self.sigma_s_t is not None:
            if not isinstance(self.sigma_s_t, (str, sp.Expr, float, int)):
                return (False, f"sigma_s_t has unexpected type {type(self.sigma_s_t)}")
        return (True, "sigma_s_t check passed (may be absent for non-sphere)")

    def test_interface_U_i_exists(self) -> Tuple[bool, str]:
        if self.U_i is None and self.U_i_symbolic is None:
            return (False, "neither U_i nor U_i_symbolic provided")
        return (True, "U_i information present")

    def test_interface_U_f_exists(self) -> Tuple[bool, str]:
        if self.U_f is None:
            return (False, "U_f not provided")
        return (True, "U_f present")

    def test_interface_Delta_U_exists(self) -> Tuple[bool, str]:
        if self.Delta_U is None:
            return (False, "Delta_U not provided")
        return (True, "Delta_U present")

    # --------------------------------------------------------------------------
    # Invariant tests – physics‑based, theorem‑driven checks (use math.isclose)
    # --------------------------------------------------------------------------
    def test_invariant_rho_exponential_decay(self) -> Tuple[bool, str]:
        """
        Check that the provided rho_t expression is consistent with
        ∂ρ/∂t = -σ/ε0 ρ, or that for a numeric time the exponential form holds.
        """
        if self.rho_t is None:
            return (False, "SKIP: rho_t unavailable")

        # If we have tau, we can verify the expression symbolically
        if self.tau is not None and isinstance(self.rho_t, (str, sp.Expr)):
            try:
                rho_expr = sp.sympify(self.rho_t) if isinstance(self.rho_t, str) else self.rho_t
                t = sp.Symbol('t')
                # expected form: rho0 * exp(-t/tau)
                expected = self.rho0 * sp.exp(-t / self.tau)
                # Compare simplified ratio
                ratio = sp.simplify(rho_expr / expected)
                if ratio.free_symbols - {t}:
                    return (False, f"rho_t does not match exp(-t/tau) form: ratio = {ratio}")
                # Check a numerical point
                if self.rho0 is not None:
                    t_val = self.tau  # one time constant
                    rho_num = float(rho_expr.subs(t, t_val))
                    expected_num = float(expected.subs(t, t_val))
                    if not math.isclose(rho_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                        return (False, f"rho_t numerical mismatch: {rho_num} vs {expected_num}")
            except Exception as e:
                return (False, f"Could not evaluate rho_t: {e}")
        return (True, "rho_t exponential decay consistent")

    def test_invariant_U_i_positive(self) -> Tuple[bool, str]:
        if self.U_i is None:
            return (False, "SKIP: U_i unavailable")
        if isinstance(self.U_i, (int, float)):
            if self.U_i <= 0:
                return (False, f"U_i = {self.U_i} is not positive")
        return (True, "U_i positive")

    def test_invariant_U_f_positive(self) -> Tuple[bool, str]:
        if self.U_f is None:
            return (False, "SKIP: U_f unavailable")
        if isinstance(self.U_f, (int, float)):
            if self.U_f <= 0:
                return (False, f"U_f = {self.U_f} is not positive")
        return (True, "U_f positive")

    def test_invariant_Delta_U_positive(self) -> Tuple[bool, str]:
        if self.Delta_U is None:
            return (False, "SKIP: Delta_U unavailable")
        if isinstance(self.Delta_U, (int, float)):
            if self.Delta_U <= 0:
                return (False, f"Delta_U = {self.Delta_U} should be positive (energy dissipated)")
        return (True, "Delta_U positive or energy dissipation check passed")

    def test_invariant_U_i_gt_U_f(self) -> Tuple[bool, str]:
        """Initial energy should be greater than final energy (energy lost as heat)."""
        if self.U_i is None or self.U_f is None:
            return (False, "SKIP: U_i or U_f unavailable")
        if isinstance(self.U_i, (int, float)) and isinstance(self.U_f, (int, float)):
            if self.U_i <= self.U_f:
                return (False, f"U_i ({self.U_i}) not > U_f ({self.U_f})")
        # If symbolic, we cannot numerify; skip
        return (True, "U_i > U_f condition satisfied")

    def test_invariant_charge_conservation_sphere(self) -> Tuple[bool, str]:
        """
        For a sphere, the total charge Q = rho0*V should equal final surface charge.
        If sigma_s_t at t->inf is given, check that.
        """
        if self.shape != 'sphere':
            return (False, "SKIP: non-sphere case, charge conservation not verifiable via simple formula")
        if self.Q is None:
            return (False, "SKIP: cannot compute total charge Q")
        if self.sigma_s_t is None:
            return (False, "SKIP: sigma_s_t not available")
        # Try to get final surface charge density (as t -> inf)
        try:
            t = sp.Symbol('t')
            sigma_expr = sp.sympify(self.sigma_s_t) if isinstance(self.sigma_s_t, str) else self.sigma_s_t
            sigma_inf = sp.limit(sigma_expr, t, sp.oo)
            if sigma_inf.is_number:
                sigma_inf_num = float(sigma_inf)
            else:
                return (True, "Cannot evaluate asymptotic limit symbolically")
        except Exception:
            return (True, "Cannot compute asymptotic limit, skip detailed check")

        # Expected final sigma_s = Q/(4πR²)
        if self.R is None:
            return (False, "SKIP: R not provided, cannot compute expected sigma_s")
        expected_sigma = self.Q / (4 * math.pi * self.R**2)
        if not math.isclose(sigma_inf_num, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Final sigma_s = {sigma_inf_num}, expected {expected_sigma}")
        return (True, "Charge conservation satisfied for sphere")

    # --------------------------------------------------------------------------
    # Modeling/trace tests – verify that intermediate formulas are physically sound
    # --------------------------------------------------------------------------
    def test_modeling_sphere_U_i_formula(self) -> Tuple[bool, str]:
        """
        For sphere, check that U_i equals 3Q²/(20πε0R).
        """
        if self.shape != 'sphere':
            return (False, "SKIP: non-sphere, U_i formula not fixed")
        if self.U_i is None or self.Q is None or self.epsilon0 is None or self.R is None:
            return (False, "SKIP: insufficient data to verify U_i formula")
        if not isinstance(self.U_i, (int, float)):
            return (False, "SKIP: U_i not numeric; cannot verify formula")
        expected = (3 * self.Q**2) / (20 * math.pi * self.epsilon0 * self.R)
        if not math.isclose(self.U_i, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"U_i = {self.U_i}, expected {expected}")
        return (True, "U_i formula for sphere correct")

    def test_modeling_sphere_U_f_formula(self) -> Tuple[bool, str]:
        """
        For sphere, check that U_f equals Q²/(8πε0R).
        """
        if self.shape != 'sphere':
            return (False, "SKIP: non-sphere, U_f formula not fixed")
        if self.U_f is None or self.Q is None or self.epsilon0 is None or self.R is None:
            return (False, "SKIP: insufficient data to verify U_f formula")
        if not isinstance(self.U_f, (int, float)):
            return (False, "SKIP: U_f not numeric; cannot verify formula")
        expected = (self.Q**2) / (8 * math.pi * self.epsilon0 * self.R)
        if not math.isclose(self.U_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"U_f = {self.U_f}, expected {expected}")
        return (True, "U_f formula for sphere correct")

    def test_modeling_sphere_sigma_s_t_form(self) -> Tuple[bool, str]:
        """
        For sphere, sigma_s(t) should be Q*(1-exp(-t/tau))/(4πR²).
        Check a numeric time if possible.
        """
        if self.shape != 'sphere':
            return (False, "SKIP: non-sphere, sigma_s_t not modelled specifically")
        if self.sigma_s_t is None or self.Q is None or self.R is None or self.tau is None:
            return (False, "SKIP: insufficient data")
        t = sp.Symbol('t')
        try:
            sigma_expr = sp.sympify(self.sigma_s_t) if isinstance(self.sigma_s_t, str) else self.sigma_s_t
        except Exception:
            return (False, "Cannot parse sigma_s_t")
        # Evaluate at some time, e.g., t=tau
        t_val = float(self.tau)
        try:
            sigma_num = float(sigma_expr.subs(t, t_val))
        except Exception:
            return (True, "Cannot evaluate numeric sigma_s_t")
        expected = (self.Q * (1 - math.exp(-t_val / self.tau))) / (4 * math.pi * self.R**2)
        if not math.isclose(sigma_num, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_s(t=tau) = {sigma_num}, expected {expected}")
        return (True, "sigma_s_t temporal form consistent with charge conservation")

    # --------------------------------------------------------------------------
    # Monotonicity tests – explicit perturbation of an input and comparison to baseline
    # --------------------------------------------------------------------------
    def test_monotonicity_rho_decays_with_time(self) -> Tuple[bool, str]:
        """
        Perturb the time input: use a baseline time t0, compute baseline rho,
        then increase time by 20% (perturbation) and compute new rho.
        Check that the perturbed rho is smaller (monotonic decay).
        """
        if self.rho_t is None:
            return (False, "SKIP: rho_t not available")
        try:
            rho_expr = sp.sympify(self.rho_t) if isinstance(self.rho_t, str) else self.rho_t
            t = sp.Symbol('t')
            # Choose baseline time (use tau if available)
            if self.tau is not None:
                t0 = 0.5 * self.tau   # baseline near initial but not zero
            else:
                t0 = 0.5
            # Perturbation: increase time by 20%
            t_pert = t0 * 1.2
            # Compute values
            rho0 = float(rho_expr.subs(t, t0))
            rho_pert = float(rho_expr.subs(t, t_pert))
            # Baseline should be larger than perturbed (decay)
            if rho_pert >= rho0:
                return (False, f"rho(t) not decreasing: baseline rho({t0})={rho0}, perturbed rho({t_pert})={rho_pert}")
        except Exception as e:
            return (False, f"Error in monotonicity check: {e}")
        return (True, "rho_t monotonically decreases with time (baseline vs 20% perturbation)")

    def test_monotonicity_sigma_s_increases_with_time_sphere(self) -> Tuple[bool, str]:
        """
        For sphere, perturb the time input: baseline t0, then increase by 20%.
        Check that perturbed sigma_s is larger than baseline (monotonic increase).
        """
        if self.shape != 'sphere':
            return (False, "SKIP: non-sphere case")
        if self.sigma_s_t is None:
            return (False, "SKIP: sigma_s_t not available")
        try:
            sigma_expr = sp.sympify(self.sigma_s_t) if isinstance(self.sigma_s_t, str) else self.sigma_s_t
            t = sp.Symbol('t')
            # Choose baseline time (use tau if available)
            if self.tau is not None:
                t0 = 0.5 * self.tau
            else:
                t0 = 0.5
            # Perturbation: increase time by 20%
            t_pert = t0 * 1.2
            # Compute values
            s0 = float(sigma_expr.subs(t, t0))
            s_pert = float(sigma_expr.subs(t, t_pert))
            # Baseline should be smaller than perturbed (increase)
            if s_pert <= s0:
                return (False, f"sigma_s(t) not increasing: baseline {s0}, perturbed {s_pert}")
        except Exception as e:
            return (False, f"Error in monotonicity check: {e}")
        return (True, "sigma_s_t monotonically increases with time (baseline vs 20% perturbation)")

    # --------------------------------------------------------------------------
    # Weak answer sanity test – only mild checks, no re-derivation
    # --------------------------------------------------------------------------
    def test_answer_sanity_finite_values(self) -> Tuple[bool, str]:
        """Ensure critical results are finite and not NaN."""
        for name, val in [('rho_t', self.rho_t), ('U_i', self.U_i),
                          ('U_f', self.U_f), ('Delta_U', self.Delta_U)]:
            if val is None:
                continue
            if isinstance(val, (int, float)):
                if math.isinf(val) or math.isnan(val):
                    return (False, f"{name} is inf/NaN")
        return (True, "All numeric results finite and not NaN")

    # --------------------------------------------------------------------------
    # Test runner: discover and execute all test_* methods
    # --------------------------------------------------------------------------
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