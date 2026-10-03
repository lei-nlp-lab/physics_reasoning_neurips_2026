import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}
        # Symbols for symbolic manipulation
        self.symbols = sp.symbols('omega R V t x y')
        self.omega_sym, self.R_sym, self.V_sym, self.t_sym, self.x_sym, self.y_sym = self.symbols

        # Parse results
        self.x_approx_raw = self.results.get('x_approx', None)
        self.y_approx_raw = self.results.get('y_approx', None)
        self.trajectory_eq_raw = self.results.get('trajectory_eq', None)

        self.x_approx = self._to_sympy(self.x_approx_raw)
        self.y_approx = self._to_sympy(self.y_approx_raw)
        self.trajectory_eq = self._parse_equation(self.trajectory_eq_raw)

        # Numeric givens (may be None)
        self.omega = self.givens.get('omega', None)
        self.R_val = self.givens.get('R', None)
        self.V_val = self.givens.get('V', None)

    def _to_sympy(self, expr):
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception:
                return None
        return None

    def _parse_equation(self, eq_str):
        if eq_str is None:
            return None
        if isinstance(eq_str, sp.Equality):
            return eq_str
        if isinstance(eq_str, str):
            try:
                parts = eq_str.split('=')
                if len(parts) == 2:
                    lhs = sp.sympify(parts[0].strip(), locals={
                        'omega': self.omega_sym, 'R': self.R_sym,
                        'V': self.V_sym, 'x': self.x_sym, 'y': self.y_sym
                    })
                    rhs = sp.sympify(parts[1].strip(), locals={
                        'omega': self.omega_sym, 'R': self.R_sym,
                        'V': self.V_sym, 'x': self.x_sym, 'y': self.y_sym
                    })
                    return sp.Eq(lhs, rhs)
            except Exception:
                return None
        return None

    def _eval_approx(self, expr, t_val, omega_val, V_val, R_val):
        """Evaluate a sympy expression at given numeric parameters."""
        if expr is None:
            return None
        subs_dict = {
            self.t_sym: t_val,
            self.omega_sym: omega_val,
            self.V_sym: V_val,
            self.R_sym: R_val
        }
        try:
            return float(expr.subs(subs_dict).evalf())
        except Exception:
            return None

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        missing = []
        for key in ['x_approx', 'y_approx', 'trajectory_eq']:
            if key not in self.results or self.results[key] is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_sympy_parsing(self):
        errors = []
        if self.x_approx is None:
            errors.append("x_approx not parseable")
        if self.y_approx is None:
            errors.append("y_approx not parseable")
        if self.trajectory_eq is None or not isinstance(self.trajectory_eq, sp.Equality):
            errors.append("trajectory_eq not parseable as sympy equality")
        if errors:
            return (False, " | ".join(errors))
        return (True, "All results parsed successfully")

    # ---------- Invariant tests (use isclose) ----------
    def test_invariant_initial_position(self):
        if self.x_approx is None or self.y_approx is None:
            return (False, "SKIP: x_approx or y_approx not available")
        t0 = 0
        x0 = self.x_approx.subs({self.t_sym: t0})
        y0 = self.y_approx.subs({self.t_sym: t0})
        if sp.simplify(x0 - self.R_sym) != 0:
            return (False, f"x(t=0) does not simplify to R, got {x0}")
        if sp.simplify(y0) != 0:
            return (False, f"y(t=0) does not simplify to 0, got {y0}")
        return (True, "Initial positions satisfied")

    def test_invariant_initial_velocity(self):
        if self.x_approx is None or self.y_approx is None:
            return (False, "SKIP: x_approx or y_approx not available")
        dxdt = sp.diff(self.x_approx, self.t_sym).subs({self.t_sym: 0})
        dydt = sp.diff(self.y_approx, self.t_sym).subs({self.t_sym: 0})
        if sp.simplify(dxdt + self.V_sym) != 0:
            return (False, f"dx/dt(t=0) does not simplify to -V, got {dxdt}")
        if sp.simplify(dydt) != 0:
            return (False, f"dy/dt(t=0) does not simplify to 0, got {dydt}")
        return (True, "Initial velocities satisfied")

    def test_invariant_coriolis_y_acceleration(self):
        if self.x_approx is None or self.y_approx is None:
            return (False, "SKIP: approximations not available")
        y_ddot = sp.diff(self.y_approx, self.t_sym, 2)
        expected = 2 * self.omega_sym * self.V_sym
        if sp.simplify(y_ddot - expected) != 0:
            if self.omega is not None and self.V_val is not None:
                t_val = 0.01
                y_ddot_num = float(y_ddot.subs({
                    self.omega_sym: self.omega,
                    self.V_sym: self.V_val,
                    self.t_sym: t_val
                }).evalf())
                expected_num = 2 * self.omega * self.V_val
                if math.isclose(y_ddot_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "y'' numerically close to 2ωV")
            return (False, f"y'' is not exactly 2ωV, got {sp.simplify(y_ddot)}")
        return (True, "y'' equals 2ωV exactly")

    # ---------- Modeling/trace tests ----------
    def test_modeling_trajectory_equation(self):
        if self.trajectory_eq is None:
            return (False, "SKIP: trajectory_eq not available")
        expected_eq = sp.Eq(self.y_sym, (self.omega_sym / self.V_sym) * (self.R_sym - self.x_sym)**2)
        diff = sp.simplify(self.trajectory_eq.lhs - self.trajectory_eq.rhs)
        expected_diff = sp.simplify(expected_eq.lhs - expected_eq.rhs)
        if diff != expected_diff:
            if self.omega is not None and self.V_val is not None and self.R_val is not None:
                x_test = 0.5 * self.R_val
                lhs_val = float(self.trajectory_eq.lhs.subs({
                    self.x_sym: x_test,
                    self.omega_sym: self.omega,
                    self.R_sym: self.R_val,
                    self.V_sym: self.V_val
                }).evalf())
                rhs_val = float(self.trajectory_eq.rhs.subs({
                    self.x_sym: x_test,
                    self.omega_sym: self.omega,
                    self.R_sym: self.R_val,
                    self.V_sym: self.V_val
                }).evalf())
                if math.isclose(lhs_val, rhs_val, rel_tol=1e-2):
                    return (True, "Equation numerically satisfied")
            return (False, "Trajectory equation does not match expected form")
        return (True, "Trajectory equation matches y = (omega/V)*(R-x)^2")

    def test_modeling_x_approximation(self):
        if self.x_approx is None:
            return (False, "SKIP: x_approx not available")
        expected = self.R_sym - self.V_sym * self.t_sym
        if sp.simplify(self.x_approx - expected) != 0:
            if self.omega is not None and self.V_val is not None and self.R_val is not None:
                t_sample = 0.01
                x_val = self._eval_approx(self.x_approx, t_sample, self.omega, self.V_val, self.R_val)
                expected_num = self.R_val - self.V_val * t_sample
                if math.isclose(x_val, expected_num, rel_tol=1e-2):
                    return (True, "x_approx numerically close to R - V*t")
            return (False, f"x_approx is not R - V*t, got {sp.simplify(self.x_approx)}")
        return (True, "x_approx equals R - V*t")

    def test_modeling_y_approximation(self):
        if self.y_approx is None:
            return (False, "SKIP: y_approx not available")
        expected = self.omega_sym * self.V_sym * self.t_sym**2
        if sp.simplify(self.y_approx - expected) != 0:
            if self.omega is not None and self.V_val is not None:
                t_sample = 0.01
                y_val = self._eval_approx(self.y_approx, t_sample, self.omega, self.V_val, self.R_val)
                expected_num = self.omega * self.V_val * t_sample**2
                if math.isclose(y_val, expected_num, rel_tol=1e-2):
                    return (True, "y_approx numerically close to omega*V*t^2")
            return (False, f"y_approx is not omega*V*t^2, got {sp.simplify(self.y_approx)}")
        return (True, "y_approx equals omega*V*t^2")

    # ---------- Monotonicity tests (explicit perturbation) ----------
    def test_monotonicity_y_increases_with_omega(self):
        """Perturb omega > baseline, check y_approx increases."""
        if self.y_approx is None or self.V_val is None or self.R_val is None:
            return (False, "SKIP: missing numeric givens or y_approx")
        t_fixed = 0.01
        omega_base = 1.0
        omega_pert = 1.1   # +10%
        # Use numeric values for V and R
        R_use = self.R_val if self.R_val is not None else 1.0
        V_use = self.V_val if self.V_val is not None else 1.0
        y_base = self._eval_approx(self.y_approx, t_fixed, omega_base, V_use, R_use)
        y_pert = self._eval_approx(self.y_approx, t_fixed, omega_pert, V_use, R_use)
        if y_base is None or y_pert is None:
            return (False, "Could not evaluate y_approx")
        if y_pert > y_base - 1e-12:   # allow tiny negative due to floating point
            return (True, f"y increased from {y_base:.6f} to {y_pert:.6f} when omega increased from {omega_base} to {omega_pert}")
        else:
            return (False, f"y did not increase: base={y_base:.6f}, perturbed={y_pert:.6f}")

    def test_monotonicity_y_increases_with_V(self):
        """Perturb V > baseline, check y_approx increases."""
        if self.y_approx is None or self.omega is None or self.R_val is None:
            return (False, "SKIP: missing numeric givens or y_approx")
        t_fixed = 0.01
        V_base = 1.0
        V_pert = 1.1
        omega_use = self.omega
        R_use = self.R_val
        y_base = self._eval_approx(self.y_approx, t_fixed, omega_use, V_base, R_use)
        y_pert = self._eval_approx(self.y_approx, t_fixed, omega_use, V_pert, R_use)
        if y_base is None or y_pert is None:
            return (False, "Could not evaluate y_approx")
        if y_pert > y_base - 1e-12:
            return (True, f"y increased from {y_base:.6f} to {y_pert:.6f} when V increased from {V_base} to {V_pert}")
        else:
            return (False, f"y did not increase: base={y_base:.6f}, perturbed={y_pert:.6f}")

    def test_monotonicity_x_decreases_with_t(self):
        """Perturb t > baseline, check x_approx decreases."""
        if self.x_approx is None or self.V_val is None or self.R_val is None:
            return (False, "SKIP: missing numeric givens or x_approx")
        t_base = 0.0
        t_pert = 0.01
        omega_use = self.omega if self.omega is not None else 1.0
        R_use = self.R_val
        V_use = self.V_val
        x_base = self._eval_approx(self.x_approx, t_base, omega_use, V_use, R_use)
        x_pert = self._eval_approx(self.x_approx, t_pert, omega_use, V_use, R_use)
        if x_base is None or x_pert is None:
            return (False, "Could not evaluate x_approx")
        if x_pert < x_base + 1e-12:
            return (True, f"x decreased from {x_base:.6f} to {x_pert:.6f} when t increased from {t_base} to {t_pert}")
        else:
            return (False, f"x did not decrease: base={x_base:.6f}, perturbed={x_pert:.6f}")

    # ---------- Weak sanity check ----------
    def test_answer_sanity_parabola_form(self):
        if self.trajectory_eq_raw is None:
            return (False, "SKIP: trajectory_eq not available")
        eq_str = self.trajectory_eq_raw
        keywords = ['omega', 'V', 'R', 'x', 'y']
        missing = [kw for kw in keywords if kw not in eq_str]
        if missing:
            return (False, f"Equation missing keywords: {missing}")
        if '(R-x)' not in eq_str and '(R - x)' not in eq_str:
            return (False, "Equation does not contain (R-x) squared pattern")
        return (True, "Equation plausible: contains required variables and (R-x)^2")

    # ---------- Test runner ----------
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