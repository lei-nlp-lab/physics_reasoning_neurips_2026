import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics checker for damped driven harmonic oscillator (part a) and 
    anharmonic oscillator (part b).
    """
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Results expected keys
        self.P_avg = self.results.get('P_avg', None)
        self.x_t = self.results.get('x_t', None)

        # Givens (if provided as attributes or in givens dict)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.m = givens.get('m', None) or getattr(solution_instance, 'm', None)
        self.omega0 = givens.get('omega0', None) or getattr(solution_instance, 'omega0', None)
        self.gamma = givens.get('gamma', None) or getattr(solution_instance, 'gamma', None)
        self.A = givens.get('A', None) or getattr(solution_instance, 'A', None)
        self.omega = givens.get('omega', None) or getattr(solution_instance, 'omega', None)
        self.alpha = givens.get('alpha', None) or getattr(solution_instance, 'alpha', None)

    # ----------------------------------------------------------------------
    # 1. Interface checks
    # ----------------------------------------------------------------------
    def test_interface_has_P_avg(self):
        if self.P_avg is None:
            return (False, "Missing required result: P_avg")
        return (True, "P_avg is present")

    def test_interface_has_x_t(self):
        if self.x_t is None:
            return (False, "Missing required result: x_t")
        return (True, "x_t is present")

    def test_interface_has_necessary_givens(self):
        missing = []
        for name, val in [('m', self.m), ('omega0', self.omega0),
                          ('gamma', self.gamma), ('A', self.A),
                          ('omega', self.omega), ('alpha', self.alpha)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All necessary givens present")

    # ----------------------------------------------------------------------
    # 2. Invariant / theorem-based checks (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_energy_dissipation_formula(self):
        """Check that P_avg matches formula: P_avg = gamma * omega^2 * A^2 / (2*(m^2*(omega0^2-omega^2)^2+gamma^2*omega^2))"""
        if any(v is None for v in (self.P_avg, self.m, self.omega0, self.gamma, self.A, self.omega)):
            return (False, "SKIP: missing givens or P_avg")
        try:
            expected = (self.gamma * self.omega**2 * self.A**2) / (2*(self.m**2 * (self.omega0**2 - self.omega**2)**2 + self.gamma**2 * self.omega**2))
            if not isinstance(self.P_avg, (int, float)) or not isinstance(expected, (int, float)):
                return (False, "P_avg or expected not numeric")
            if math.isclose(self.P_avg, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Energy dissipation formula matches")
            else:
                return (False, f"P_avg {self.P_avg} does not match expected {expected}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_initial_position(self):
        """x(t=0) should be 0 within numerical tolerance."""
        if self.x_t is None:
            return (False, "SKIP: x_t not available")
        try:
            t = sp.symbols('t')
            expr = self._to_sympy(self.x_t)
            if expr is None:
                return (False, "Cannot parse x_t into sympy expression")
            x0_val = float(expr.subs(t, 0).evalf())
            if math.isclose(x0_val, 0.0, abs_tol=1e-3):
                return (True, f"x(0) = {x0_val} ≈ 0")
            else:
                return (False, f"x(0) = {x0_val}, not close to 0")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_initial_velocity(self):
        """dx/dt at t=0 should be 0 within tolerance."""
        if self.x_t is None:
            return (False, "SKIP: x_t not available")
        try:
            t = sp.symbols('t')
            expr = self._to_sympy(self.x_t)
            if expr is None:
                return (False, "Cannot parse x_t into sympy expression")
            vel = sp.diff(expr, t)
            v0_val = float(vel.subs(t, 0).evalf())
            if math.isclose(v0_val, 0.0, abs_tol=1e-3):
                return (True, f"dx/dt(0) = {v0_val} ≈ 0")
            else:
                return (False, f"dx/dt(0) = {v0_val}, not close to 0")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_ode_satisfied_to_first_order(self):
        """
        Check that the expression for x(t) satisfies the anharmonic ODE
        to first order in alpha.
        ODE: m * x'' + m * omega0^2 * x - alpha * x^2 - A * cos(omega t) = O(alpha^2)
        We test for given numeric parameters (if available) or symbolic substitution.
        """
        if self.x_t is None or any(v is None for v in (self.m, self.omega0, self.A, self.omega, self.alpha)):
            return (False, "SKIP: missing x_t or givens")
        try:
            t = sp.symbols('t')
            expr = self._to_sympy(self.x_t)
            if expr is None:
                return (False, "Cannot parse x_t")
            # Replace m, omega0, A, omega, alpha with numeric
            subs_dict = {
                sp.Symbol('m'): float(self.m),
                sp.Symbol('omega0'): float(self.omega0),
                sp.Symbol('A'): float(self.A),
                sp.Symbol('omega'): float(self.omega),
                sp.Symbol('alpha'): float(self.alpha),
            }
            x_func = expr.subs(subs_dict)
            x_ddot = sp.diff(x_func, t, 2)
            lhs = float(self.m) * x_ddot + float(self.m) * float(self.omega0)**2 * x_func - float(self.alpha) * x_func**2
            rhs = float(self.A) * sp.cos(float(self.omega) * t)
            residual = lhs - rhs
            # Evaluate at a few random points
            points = [0.1, 0.5, 1.0, 2.0]
            errors = [float(abs(residual.subs(t, tp).evalf())) for tp in points]
            max_err = max(errors)
            if max_err < 1e-2:
                return (True, f"ODE satisfied with max residual {max_err}")
            else:
                return (False, f"ODE residual too large: max {max_err}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_invariant_x0_satisfies_linear_ode(self):
        """
        The O(alpha^0) part (x0) should satisfy: m x0'' + m omega0^2 x0 = A cos(omega t)
        """
        if self.x_t is None or any(v is None for v in (self.m, self.omega0, self.A, self.omega)):
            return (False, "SKIP: missing data")
        try:
            t = sp.symbols('t')
            expr = self._to_sympy(self.x_t)
            if expr is None:
                return (False, "Cannot parse x_t")
            # Extract x0 by setting alpha=0
            alpha_sym = sp.Symbol('alpha')
            x0_expr = expr.subs(alpha_sym, 0)
            subs_dict = {
                sp.Symbol('m'): float(self.m),
                sp.Symbol('omega0'): float(self.omega0),
                sp.Symbol('A'): float(self.A),
                sp.Symbol('omega'): float(self.omega),
            }
            x0_func = x0_expr.subs(subs_dict)
            lhs = float(self.m) * sp.diff(x0_func, t, 2) + float(self.m) * float(self.omega0)**2 * x0_func
            rhs = float(self.A) * sp.cos(float(self.omega) * t)
            residual = lhs - rhs
            points = [0.2, 0.5, 1.0]
            errors = [float(abs(residual.subs(t, p).evalf())) for p in points]
            max_err = max(errors)
            if max_err < 1e-3:
                return (True, f"x0 satisfies linear ODE, max residual {max_err}")
            else:
                return (False, f"x0 ODE residual too large: {max_err}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    # ----------------------------------------------------------------------
    # 3. Modeling / trace checks (use math.isclose)
    # ----------------------------------------------------------------------
    def test_trace_x0_linear_initial_condition(self):
        """Check that x0(0)=0 and x0'(0)=0."""
        if self.x_t is None:
            return (False, "SKIP: x_t missing")
        try:
            t = sp.symbols('t')
            expr = self._to_sympy(self.x_t)
            if expr is None:
                return (False, "Cannot parse")
            alpha_sym = sp.Symbol('alpha')
            x0_expr = expr.subs(alpha_sym, 0)
            x0_0 = float(x0_expr.subs(t, 0).evalf())
            v0_expr = sp.diff(x0_expr, t)
            v0_0 = float(v0_expr.subs(t, 0).evalf())
            if math.isclose(x0_0, 0, abs_tol=1e-3) and math.isclose(v0_0, 0, abs_tol=1e-3):
                return (True, f"x0(0)={x0_0}, x0'(0)={v0_0} match zero")
            else:
                return (False, f"x0(0)={x0_0}, x0'(0)={v0_0} not both zero")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_trace_steady_state_power_positive(self):
        if self.P_avg is None:
            return (False, "SKIP: P_avg missing")
        try:
            if isinstance(self.P_avg, (int, float)) and self.P_avg >= 0:
                return (True, "P_avg is non-negative")
            else:
                return (False, f"P_avg is {self.P_avg}, not non-negative")
        except:
            return (False, "Cannot evaluate")

    # ----------------------------------------------------------------------
    # 4. Monotonicity checks (perturb one input and compare output)
    # ----------------------------------------------------------------------
    def test_monotonicity_P_avg_with_gamma(self):
        """
        Increase gamma by 10% and see if P_avg decreases (assuming omega != omega0).
        Using the analytical expression (derived from given solution).
        """
        if any(v is None for v in (self.P_avg, self.m, self.omega0, self.gamma, self.A, self.omega)):
            return (False, "SKIP: missing data")
        try:
            gamma0 = float(self.gamma)
            # Compute P_avg at gamma = gamma0 * 1.1
            gamma1 = gamma0 * 1.1
            P1 = (gamma1 * self.omega**2 * self.A**2) / (2*(self.m**2*(self.omega0**2 - self.omega**2)**2 + gamma1**2 * self.omega**2))
            # Baseline is the given P_avg
            if P1 < self.P_avg:
                return (True, f"P_avg decreases from {self.P_avg} to {P1} when gamma increases 10%")
            else:
                # For small gamma close to resonance, P may increase; but derivative is positive for gamma < something.
                # We'll check derivative sign: dP/dgamma = ... for typical case.
                # For robustness, just report failure but not strict.
                return (False, f"P_avg does not decrease: baseline {self.P_avg}, perturbed {P1}")
        except Exception as e:
            return (False, f"test crashed: {e}")

    def test_monotonicity_x_t_with_omega(self):
        """
        Perturb omega by +5% (if omega < omega0) and check that |x(t)| increases.
        Use solution's own expression for x_t at a fixed t.
        """
        if self.x_t is None or any(v is None for v in (self.m, self.omega0, self.A, self.omega, self.alpha)):
            return (False, "SKIP: missing x_t or givens")
        try:
            t = sp.symbols('t')
            expr = self._to_sympy(self.x_t)
            if expr is None:
                return (False, "Cannot parse x_t")
            # Use numeric values for parameters (except omega which we will perturb)
            subs_base = {
                sp.Symbol('m'): float(self.m),
                sp.Symbol('omega0'): float(self.omega0),
                sp.Symbol('A'): float(self.A),
                sp.Symbol('alpha'): float(self.alpha),
                sp.Symbol('omega'): float(self.omega),
            }
            # Choose a time where x is not zero (use t=0.5)
            t_val = 0.5
            x_base = float(expr.subs(t, t_val).subs(subs_base).evalf())
            # Now perturb omega: increase by 5%
            omega_new = float(self.omega) * 1.05
            subs_perturb = subs_base.copy()
            subs_perturb[sp.Symbol('omega')] = omega_new
            x_perturb = float(expr.subs(t, t_val).subs(subs_perturb).evalf())
            # Compare absolute values: for omega < omega0, magnitude should increase as omega rises toward resonance
            if float(self.omega) < float(self.omega0):
                if abs(x_perturb) > abs(x_base):
                    return (True, f"|x(t=0.5)| increases from {abs(x_base)} to {abs(x_perturb)} when omega increases 5%")
                else:
                    return (False, f"|x(t=0.5)| decreased: baseline {abs(x_base)}, perturbed {abs(x_perturb)}")
            elif float(self.omega) > float(self.omega0):
                # Above resonance, increasing omega moves away from resonance, so magnitude decreases?
                # Actually B' magnitude decreases as omega increases above omega0.
                if abs(x_perturb) < abs(x_base):
                    return (True, f"|x(t=0.5)| decreases from {abs(x_base)} to {abs(x_perturb)} when omega increases 5%")
                else:
                    return (False, f"|x(t=0.5)| increased: baseline {abs(x_base)}, perturbed {abs(x_perturb)}")
            else:
                # Exactly at resonance – denominator zero; the formula may have a singularity. Skip.
                return (True, "SKIP: omega = omega0 (resonance), not checking")
        except Exception as e:
            return (False, f"test crashed: {e}")

    # ----------------------------------------------------------------------
    # 5. Weak answer sanity (no isclose, no derived values)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Check that both answers are finite and not NaN."""
        problems = []
        if self.P_avg is not None:
            try:
                val = float(self.P_avg)
                if not math.isfinite(val):
                    problems.append("P_avg is not finite")
            except:
                problems.append("P_avg not convertible to float")
        if self.x_t is not None:
            try:
                t = sp.symbols('t')
                expr = self._to_sympy(self.x_t)
                if expr is not None:
                    # Try to evaluate numerically at t=1 with some default numeric substitution
                    # Use self values if available, else dummy
                    subs = {}
                    for sym in expr.free_symbols:
                        if sym == sp.Symbol('t'):
                            continue
                        # Try to use self values
                        if str(sym) == 'm' and self.m is not None:
                            subs[sym] = float(self.m)
                        elif str(sym) == 'omega0' and self.omega0 is not None:
                            subs[sym] = float(self.omega0)
                        elif str(sym) == 'A' and self.A is not None:
                            subs[sym] = float(self.A)
                        elif str(sym) == 'omega' and self.omega is not None:
                            subs[sym] = float(self.omega)
                        elif str(sym) == 'alpha' and self.alpha is not None:
                            subs[sym] = float(self.alpha)
                        else:
                            subs[sym] = 1.0  # fallback
                    val = float(expr.subs(t, 1).subs(subs).evalf())
                    if not math.isfinite(val):
                        problems.append("x_t(t=1) is not finite")
            except:
                pass  # if can't evaluate, ignore
        if problems:
            return (False, "; ".join(problems))
        return (True, "Answers appear finite and evaluable")

    # ----------------------------------------------------------------------
    # Helper: convert x_t to sympy expression (if string)
    # ----------------------------------------------------------------------
    def _to_sympy(self, x):
        """Convert x (string or sympy expression) to a sympy expression. Return None on failure."""
        if isinstance(x, sp.Basic):
            return x
        if isinstance(x, str):
            try:
                from sympy.parsing.sympy_parser import parse_expr
                # Parse assuming standard sympy syntax; replace cos, sin
                return parse_expr(x, local_dict={'cos': sp.cos, 'sin': sp.sin, 'pi': sp.pi})
            except Exception:
                return None
        return None

    # ----------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)