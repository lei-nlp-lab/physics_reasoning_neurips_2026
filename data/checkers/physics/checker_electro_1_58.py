import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # --- Givens ---
        # Try to get givens dict from solution_instance
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}
        self.q0 = givens.get('q0', None)
        self.sigma = givens.get('sigma', None)
        self.epsilon = givens.get('epsilon', None)
        self.a = givens.get('a', None)
        self.b = givens.get('b', None)

        # --- Results ---
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.q_t = results.get('q_t', None)
        self.I_t = results.get('I_t', None)
        self.H = results.get('H', None)
        self.U0 = results.get('U0', None)

        # Store t symbol for symbolic checks
        self.t = sp.Symbol('t', real=True, nonnegative=True)

    def _to_sympy(self, expr):
        """Try to convert an expression to sympy; return None if impossible."""
        try:
            return sp.sympify(expr)
        except (sp.SympyError, TypeError):
            return None

    def _numerical_value(self, expr):
        """Try to extract a numerical float from a sympy expression."""
        if expr is None:
            return None
        try:
            val = float(expr)
        except (TypeError, ValueError, sp.SympyError):
            # If it's symbolic, try to substitute given values
            if all(v is not None for v in [self.q0, self.sigma, self.epsilon, self.a, self.b]):
                sub = {
                    sp.Symbol('q0'): self.q0,
                    sp.Symbol('sigma'): self.sigma,
                    sp.Symbol('epsilon'): self.epsilon,
                    sp.Symbol('a'): self.a,
                    sp.Symbol('b'): self.b,
                    sp.Symbol('pi'): math.pi
                }
                try:
                    val = float(expr.subs(sub))
                except:
                    val = None
            else:
                val = None
        return val

    # ---------- Interface checks ----------
    def test_interface_required_keys(self):
        """Check that all required result keys exist and are not None."""
        keys = ['q_t', 'I_t', 'H', 'U0']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return False, f"Missing results: {missing}"
        return True, "All required keys present."

    def test_interface_givens_present(self):
        """Check that all necessary givens are provided."""
        givens_required = ['q0', 'sigma', 'epsilon', 'a', 'b']
        missing = [g for g in givens_required if getattr(self, g) is None]
        if missing:
            return False, f"Missing givens: {missing}"
        return True, "All required givens present."

    # ---------- Invariant / theorem checks ----------
    def test_invariant_charge_ode(self):
        """Check that q(t) satisfies dq/dt = -(sigma/epsilon)*q."""
        if self.q_t is None or self.sigma is None or self.epsilon is None:
            return False, "SKIP: missing q_t, sigma, or epsilon"

        q_sym = self._to_sympy(self.q_t)
        if q_sym is None:
            return False, "SKIP: could not convert q_t to sympy expression"

        # Compute derivative
        dq_dt = sp.diff(q_sym, self.t)
        rhs = -(self.sigma / self.epsilon) * q_sym
        # Check symbolic equality
        if sp.simplify(dq_dt - rhs) == 0:
            return True, "q(t) satisfies dq/dt = -(sigma/epsilon)*q"
        else:
            # Fallback to numerical check
            try:
                # Pick a time t=tau and evaluate numerically
                tau = self.epsilon / self.sigma if self.sigma != 0 else 1.0
                t_val = 0.5 * tau
                q_num = float(q_sym.subs({self.t: t_val}))
                # Approximate derivative using small epsilon
                dt = 1e-6 * self.epsilon / abs(self.sigma) if self.sigma != 0 else 1e-6
                if dt <= 0:
                    dt = 1e-6
                dq_dt_num = (float(q_sym.subs({self.t: t_val+dt})) - float(q_sym.subs({self.t: t_val-dt}))) / (2*dt)
                rhs_num = -(self.sigma / self.epsilon) * q_num
                if math.isclose(dq_dt_num, rhs_num, rel_tol=1e-2, abs_tol=1e-4):
                    return True, "q(t) satisfies ODE numerically"
                else:
                    return False, f"Numerical ODE check failed: dq/dt={dq_dt_num:.3e}, rhs={rhs_num:.3e}"
            except Exception as e:
                return False, f"Numerical fallback failed: {e}"

    def test_invariant_current_relation(self):
        """Check that I(t) = (sigma/epsilon) * q(t)."""
        if self.I_t is None or self.q_t is None or self.sigma is None or self.epsilon is None:
            return False, "SKIP: missing I_t, q_t, sigma, or epsilon"

        I_sym = self._to_sympy(self.I_t)
        q_sym = self._to_sympy(self.q_t)
        if I_sym is None or q_sym is None:
            return False, "SKIP: could not convert I_t or q_t to sympy"

        expected_I = (self.sigma / self.epsilon) * q_sym
        if sp.simplify(I_sym - expected_I) == 0:
            return True, "I(t) = (sigma/epsilon)*q(t) holds symbolically"
        else:
            # Numerical fallback
            try:
                t_val = 0.5 * self.epsilon / self.sigma if self.sigma != 0 else 0.5
                I_num = float(I_sym.subs({self.t: t_val}))
                q_num = float(q_sym.subs({self.t: t_val}))
                expected_num = (self.sigma / self.epsilon) * q_num
                if math.isclose(I_num, expected_num, rel_tol=1e-2, abs_tol=1e-4):
                    return True, "I(t) = (sigma/epsilon)*q(t) holds numerically"
                else:
                    return False, f"Numerical check failed: I={I_num:.3e}, expected={expected_num:.3e}"
            except Exception as e:
                return False, f"Numerical fallback failed: {e}"

    def test_invariant_energy_equality(self):
        """Check that Joule heat H equals initial electrostatic energy U0."""
        if self.H is None or self.U0 is None:
            return False, "SKIP: missing H or U0"

        H_val = self._numerical_value(self.H)
        U0_val = self._numerical_value(self.U0)
        if H_val is not None and U0_val is not None:
            if math.isclose(H_val, U0_val, rel_tol=1e-2, abs_tol=1e-4):
                return True, f"H = U0 = {U0_val:.3e} (rel diff = {abs(H_val-U0_val)/max(1e-9, U0_val):.2e})"
            else:
                return False, f"H = {H_val:.3e}, U0 = {U0_val:.3e}, not equal"
        else:
            # Symbolic check
            H_sym = self._to_sympy(self.H)
            U0_sym = self._to_sympy(self.U0)
            if H_sym is not None and U0_sym is not None:
                if sp.simplify(H_sym - U0_sym) == 0:
                    return True, "H == U0 symbolically"
            return False, "Could not verify equality of H and U0"

    # ---------- Monotonicity checks ----------
    def test_monotonicity_charge_decay_with_conductivity(self):
        """Check that increasing conductivity reduces charge at a fixed time."""
        if self.q_t is None or self.sigma is None or self.epsilon is None:
            return False, "SKIP: missing q_t, sigma, or epsilon"

        q_sym = self._to_sympy(self.q_t)
        if q_sym is None:
            return False, "SKIP: could not convert q_t to sympy"

        # Compute derivative w.r.t. sigma (treating sigma as a parameter)
        sigma_sym = sp.Symbol('sigma', real=True, positive=True)
        q_expr = self.q_t  # keep original expression
        # Replace sigma with symbolic variable if it's numeric
        if not hasattr(self.sigma, 'free_symbols'):
            # numeric sigma → substitute into symbolic expression
            q_expr = sp.sympify(self.q_t)
            if hasattr(q_expr, 'free_symbols'):
                # expression might contain sigma as a symbol? Replace with numeric?
                # Better: evaluate derivative numerically around its value
                pass
        # We'll do a numerical perturbation test
        try:
            t_val = 0.5 * self.epsilon / self.sigma if self.sigma != 0 else 0.5
            q_original = self._numerical_value(q_sym.subs({self.t: t_val}))
            # Perturb sigma upward by 5%
            sigma_pert = self.sigma * 1.05
            if sigma_pert == 0:
                return False, "SKIP: sigma is zero"
            # We need to recompute q_t with perturbed sigma. Assuming q_t expression contains sigma.
            q_expr_sub = self.q_t
            # Use sympy substitution if possible
            q_pert_sym = sp.sympify(q_expr_sub).subs({sp.Symbol('sigma'): sigma_pert, self.t: t_val})
            q_pert = self._numerical_value(q_pert_sym)
            if q_original is None or q_pert is None:
                return False, "Cannot evaluate charge numerically"
            if q_pert < q_original - 1e-10:
                return True, "Increasing conductivity decreases charge (monotonicity holds)"
            else:
                return False, f"Monotonicity violated: q0={q_original:.3e}, q_pert={q_pert:.3e} (expected lower)"
        except Exception as e:
            return False, f"Monotonicity test crashed: {e}"

    # ---------- Modeling / trace checks ----------
    def test_modeling_current_expression(self):
        """Check that I_t expression has correct form: I(t) = (sigma*q0/epsilon)*exp(-sigma*t/epsilon)."""
        if self.I_t is None or self.q0 is None or self.sigma is None or self.epsilon is None:
            return False, "SKIP: missing I_t, q0, sigma, or epsilon"
        I_sym = self._to_sympy(self.I_t)
        if I_sym is None:
            return False, "SKIP: cannot convert I_t"
        # Expected form
        expected = (self.sigma * self.q0 / self.epsilon) * sp.exp(-self.sigma/self.epsilon * self.t)
        if sp.simplify(I_sym - expected) == 0:
            return True, "I(t) has the expected exponential form"
        else:
            # Attempt numerical check at t=0
            try:
                I0_num = float(I_sym.subs({self.t: 0}))
                expected0 = (self.sigma * self.q0 / self.epsilon)
                if math.isclose(I0_num, expected0, rel_tol=1e-2, abs_tol=1e-4):
                    return True, "I(0) matches sigma*q0/epsilon"
                else:
                    return False, f"I(0) = {I0_num:.3e}, expected {expected0:.3e}"
            except Exception as e:
                return False, f"Modeling check failed: {e}"

    def test_modeling_energy_formula(self):
        """Check that H and U0 have the correct form: q0^2/(8πϵ)(1/a - 1/b)."""
        if self.H is None or self.U0 is None:
            return False, "SKIP: missing H or U0"
        expected_form = self.q0**2 / (8 * math.pi * self.epsilon) * (1/self.a - 1/self.b) if all(v is not None for v in [self.q0, self.epsilon, self.a, self.b]) else None
        if expected_form is None:
            return False, "SKIP: missing givens to compute expected form"
        H_val = self._numerical_value(self.H)
        U0_val = self._numerical_value(self.U0)
        if H_val is not None and U0_val is not None:
            if math.isclose(H_val, expected_form, rel_tol=1e-2, abs_tol=1e-4) and math.isclose(U0_val, expected_form, rel_tol=1e-2, abs_tol=1e-4):
                return True, "H and U0 match expected formula"
            else:
                return False, f"H={H_val:.3e}, U0={U0_val:.3e}, expected={expected_form:.3e}"
        else:
            return False, "Could not evaluate H or U0 numerically"

    # ---------- Answer sanity (weak, only positivity and finiteness) ----------
    def test_answer_sanity_positivity(self):
        """Check that final answers are positive and finite."""
        fails = []
        for name, val in [('I_t', self.I_t), ('H', self.H), ('U0', self.U0)]:
            if val is None:
                fails.append(f"{name} missing")
                continue
            v = self._numerical_value(val)
            if v is None:
                fails.append(f"{name} not evaluable")
            elif v <= 0 or math.isinf(v) or math.isnan(v):
                fails.append(f"{name} = {v} invalid")
        if fails:
            return False, "; ".join(fails)
        return True, "All answers positive and finite."

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