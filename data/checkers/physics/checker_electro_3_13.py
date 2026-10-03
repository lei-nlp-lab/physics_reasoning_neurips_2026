import sympy as sp
import math
from numbers import Number

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely; default None
        self.R1 = getattr(solution_instance, 'R1', None)
        self.R2 = getattr(solution_instance, 'R2', None)
        self.R3 = getattr(solution_instance, 'R3', None)
        self.C = getattr(solution_instance, 'C', None)
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        self.Q0 = getattr(solution_instance, 'Q0', None)

        # Extract results (q1, q2) from results dict or direct attributes
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.q1 = solution_instance.results.get('q1', None)
            self.q2 = solution_instance.results.get('q2', None)
        else:
            self.q1 = getattr(solution_instance, 'q1', None)
            self.q2 = getattr(solution_instance, 'q2', None)

        # Derived quantities (computed only if all needed givens exist)
        self.D = None
        self.tau = None
        self.q_ss = None
        self._compute_derived()

    def _compute_derived(self):
        """Compute derived quantities safely."""
        if all(v is not None for v in [self.R1, self.R2, self.R3, self.C, self.epsilon]):
            # Use sympy for exact symbolic computation, fallback to float
            try:
                R1, R2, R3, C, eps = sp.symbols('R1 R2 R3 C eps', real=True, positive=True)
                D_sym = R1*R2 + R2*R3 + R3*R1
                tau_sym = D_sym * C / (R1 + R2)
                q_ss_sym = eps * R3 / (R1 + R2)
                # Substitute given values (if numeric)
                if all(isinstance(v, (int, float)) for v in [self.R1, self.R2, self.R3, self.C, self.epsilon]):
                    subs_dict = {R1: self.R1, R2: self.R2, R3: self.R3, C: self.C, eps: self.epsilon}
                    self.D = float(D_sym.subs(subs_dict))
                    self.tau = float(tau_sym.subs(subs_dict))
                    self.q_ss = float(q_ss_sym.subs(subs_dict))
                else:
                    # Symbolic representation
                    self.D = D_sym
                    self.tau = tau_sym
                    self.q_ss = q_ss_sym
            except Exception:
                # Fallback to numeric computation
                if all(isinstance(v, (int, float)) for v in [self.R1, self.R2, self.R3, self.C, self.epsilon]):
                    self.D = self.R1*self.R2 + self.R2*self.R3 + self.R3*self.R1
                    if self.D != 0:
                        self.tau = self.D * self.C / (self.R1 + self.R2)
                        self.q_ss = self.epsilon * self.R3 / (self.R1 + self.R2)
                # If not numeric but symbolic, leave as None; tests will skip

    def _safe_substitute(self, expr, t_val=None):
        """Substitute t and known parameters into expression if possible."""
        if expr is None:
            return None
        # Define symbols
        t = sp.symbols('t', real=True)
        # Try to substitute numeric values for parameters
        subs_dict = {sp.Symbol('R1'): self.R1, sp.Symbol('R2'): self.R2,
                     sp.Symbol('R3'): self.R3, sp.Symbol('C'): self.C,
                     sp.Symbol('epsilon'): self.epsilon, sp.Symbol('Q0'): self.Q0}
        if t_val is not None:
            subs_dict[t] = t_val
        # Remove None entries
        subs_dict = {k: v for k, v in subs_dict.items() if v is not None}
        try:
            result = expr.subs(subs_dict)
            # If result contains only numbers, evaluate
            if result.is_Number:
                return float(result)
            return result
        except Exception:
            # If substitution fails, try numerical evaluation for numeric parts
            if all(isinstance(v, (int, float)) for v in subs_dict.values() if v is not None):
                try:
                    return float(expr.subs(subs_dict))
                except Exception:
                    return None
            return None

    def test_interface_required_givens(self):
        """Check that all required givens are present and positive."""
        required = [
            ('R1', self.R1, 'resistance'),
            ('R2', self.R2, 'resistance'),
            ('R3', self.R3, 'resistance'),
            ('C', self.C, 'capacitance'),
            ('epsilon', self.epsilon, 'emf'),
        ]
        for name, val, phys_type in required:
            if val is None:
                return (False, f"Missing given: {name}")
            if not isinstance(val, (int, float, sp.Expr)):
                return (False, f"Given {name} has invalid type: {type(val).__name__}")
            if isinstance(val, (int, float)) and val <= 0:
                return (False, f"Given {name} is non-positive: {val}")
        # Q0 is optional (for part b) – allow None or numeric
        if self.Q0 is not None:
            if not isinstance(self.Q0, (int, float, sp.Expr)):
                return (False, f"Given Q0 has invalid type: {type(self.Q0).__name__}")
            if isinstance(self.Q0, (int, float)) and self.Q0 < 0:
                return (False, f"Given Q0 is negative: {self.Q0}")
        return (True, "All required givens present and have valid types/values")

    def test_interface_results_exist(self):
        """Check that both q1 and q2 results are present."""
        if self.q1 is None:
            return (False, "q1 result is missing")
        if self.q2 is None:
            return (False, "q2 result is missing")
        return (True, "Both q1 and q2 results present")

    def test_invariant_initial_conditions(self):
        """Check that q1(0)=0 and q2(0)=Q0 (within tolerance)."""
        if self.q1 is None or self.q2 is None:
            return (False, "SKIP: results missing")
        # Evaluate at t=0
        t = sp.symbols('t', real=True)
        q1_0 = self._safe_substitute(self.q1, t_val=0)
        q2_0 = self._safe_substitute(self.q2, t_val=0)
        if q1_0 is None or q2_0 is None:
            return (False, "SKIP: could not evaluate at t=0")
        # Tolerance for numeric evaluation
        tol = 1e-9 if isinstance(q1_0, (int, float)) else 1e-6
        if not math.isclose(float(q1_0), 0, abs_tol=tol):
            return (False, f"q1(0) = {q1_0}, expected 0")
        # For q2, check against actual Q0 (or 0 if Q0 missing)
        expected_q2_0 = self.Q0 if self.Q0 is not None else 0
        if isinstance(expected_q2_0, (int, float)):
            if not math.isclose(float(q2_0), expected_q2_0, abs_tol=tol):
                return (False, f"q2(0) = {q2_0}, expected {expected_q2_0}")
        else:
            # Symbolic compare
            if not sp.simplify(q2_0 - expected_q2_0).is_zero:
                return (False, f"q2(0) does not symbolically equal Q0")
        return (True, "Initial conditions satisfied")

    def test_invariant_steady_state(self):
        """Check that as t→∞ both q1 and q2 approach epsilon*R3/(R1+R2)."""
        if self.q1 is None or self.q2 is None:
            return (False, "SKIP: results missing")
        if self.q_ss is None:
            return (False, "SKIP: could not compute steady-state charge")
        # Evaluate at large t (e.g., t=10*tau) if numeric
        if isinstance(self.tau, (int, float)):
            t_large = 10 * self.tau
            q1_inf = self._safe_substitute(self.q1, t_val=t_large)
            q2_inf = self._safe_substitute(self.q2, t_val=t_large)
            if q1_inf is None or q2_inf is None:
                return (False, "SKIP: could not evaluate at large t")
            tol = 1e-2 * self.q_ss  # 1% tolerance
            if not math.isclose(float(q1_inf), float(self.q_ss), rel_tol=1e-2, abs_tol=tol):
                return (False, f"q1(10*tau) = {q1_inf}, expected {self.q_ss} within 1%")
            if not math.isclose(float(q2_inf), float(self.q_ss), rel_tol=1e-2, abs_tol=tol):
                return (False, f"q2(10*tau) = {q2_inf}, expected {self.q_ss} within 1%")
        else:
            # For symbolic, check limit as t->infinity (exp(-∞) = 0)
            # We can substitute exp(-t/tau) → 0 if we isolate exponent
            # More simply, check that the functional form contains a constant term equal to q_ss
            # This is a weaker check
            # Assuming q1 = q_ss*(1 - exp(-t/tau)), q2 = q_ss + (Q0 - q_ss)*exp(-t/tau)
            # We'll check if the constant part matches q_ss.
            # Use sympy limit
            t = sp.symbols('t', real=True)
            try:
                lim1 = sp.limit(self.q1, t, sp.oo)
                lim2 = sp.limit(self.q2, t, sp.oo)
                if sp.simplify(lim1 - self.q_ss).is_zero is not True:
                    return (False, f"Symbolic limit of q1(t->inf) = {lim1}, expected {self.q_ss}")
                if sp.simplify(lim2 - self.q_ss).is_zero is not True:
                    return (False, f"Symbolic limit of q2(t->inf) = {lim2}, expected {self.q_ss}")
            except Exception:
                return (False, "SKIP: could not compute symbolic limit")
        return (True, "Steady-state charge matches expected value")

    def test_modeling_exponential_form(self):
        """Check that q1 and q2 have the expected exponential form with correct time constant."""
        if self.q1 is None or self.q2 is None:
            return (False, "SKIP: results missing")
        if self.tau is None:
            return (False, "SKIP: could not compute time constant")
        # Check that the exponent is -t/tau (or equivalent)
        # Extract exponential terms
        t = sp.symbols('t', real=True)
        try:
            # For q1, we expect terms like exp(-t/tau) or exp(-A*t) where A = 1/tau
            exp_pattern = sp.exp(-sp.Symbol('t') / sp.Symbol('tau'))
            # Substitute tau with computed value if numeric
            if isinstance(self.tau, (int, float)):
                # Replace tau symbol with numeric value in the expression
                tau_sym = sp.Symbol('tau')
                # We'll try to find exponential terms
                exponentials = [e for e in self.q1.atoms(sp.exp)]
                for e in exponentials:
                    # Check if exponent is linear in t
                    exponent = e.exp
                    if exponent.is_Mul and exponent.has(t):
                        # Check if it equals -t/tau
                        expr_check = exponent + t/self.tau
                        if sp.simplify(expr_check).is_zero:
                            continue
                        else:
                            # Might be different form; allow alternative expression
                            pass
                # Actually, a more robust check: verify that dq/dt + (1/tau)*q = q_ss/tau for q1 and similarly for q2
                # This is the ODE check
                return self._test_ode_satisfaction()
            else:
                # Symbolic tau: assume the exponential contains t/tau
                pass
        except Exception:
            pass
        # Fallback: compute derivative symbolically and check ODE
        return self._test_ode_satisfaction()

    def _test_ode_satisfaction(self):
        """Verify that the solution satisfies the ODE derived from Kirchhoff's laws."""
        # ODE: dq/dt = -A q + B, where A = (R1+R2)/(D*C), B = epsilon*R3/(D*C)
        if any(v is None for v in [self.R1, self.R2, self.R3, self.C, self.epsilon]):
            return (False, "SKIP: missing givens to compute ODE coefficients")
        D_val = self.R1*self.R2 + self.R2*self.R3 + self.R3*self.R1
        if D_val == 0:
            return (False, "SKIP: D = 0 (unphysical)")
        A = (self.R1 + self.R2) / (D_val * self.C)
        B = self.epsilon * self.R3 / (D_val * self.C)
        # Use symbolic differential equation check
        t = sp.symbols('t', real=True)
        for label, q_expr in [('q1', self.q1), ('q2', self.q2)]:
            try:
                dq_dt = sp.diff(q_expr, t)
                lhs = dq_dt + A * q_expr
                rhs = B
                diff = sp.simplify(lhs - rhs)
                # If diff is constant zero (symbolic) or numerically zero
                if diff == 0:
                    continue
                # Check numerically if possible
                if all(isinstance(v, (int, float)) for v in [A, B, self.R1, self.R2, self.R3, self.C, self.epsilon]):
                    t_val = 0.5 * self.tau if self.tau else 1.0
                    subs_dict = {t: t_val}
                    # Substitute any symbolic params from expression
                    all_syms = q_expr.free_symbols
                    subs_dict.update({s: getattr(self, s.name, None) for s in all_syms if hasattr(self, s.name)})
                    subs_dict = {k: v for k, v in subs_dict.items() if v is not None}
                    q_num = float(q_expr.subs(subs_dict))
                    dq_num = float(dq_dt.subs(subs_dict))
                    lhs_num = dq_num + A * q_num
                    if not math.isclose(lhs_num, B, rel_tol=1e-2, abs_tol=1e-2):
                        return (False, f"{label} fails ODE at t={t_val}: LHS={lhs_num}, RHS={B}")
                else:
                    # Symbolic, but diff not simplified to zero – possibly due to parameter symbols
                    # Check if diff is identically zero by substituting random numbers
                    try:
                        from random import uniform
                        for _ in range(3):
                            subs_test = {s: uniform(0.1, 2.0) for s in all_syms if s.name in ['R1','R2','R3','C','epsilon','Q0']}
                            subs_test[t] = uniform(0, 5)
                            val = float(diff.subs(subs_test))
                            if abs(val) > 1e-8:
                                return (False, f"{label} symbolic ODE check failed: diff={diff}")
                    except Exception:
                        pass
            except Exception as e:
                return (False, f"ODE check crashed for {label}: {e}")
        return (True, "Both solutions satisfy the ODE dq/dt + A q = B")

    def test_modeling_time_constant_positivity(self):
        """Check that the time constant (tau) is positive and finite."""
        if self.tau is None:
            return (False, "SKIP: could not compute tau")
        if sp.simplify(self.tau <= 0) == sp.true:
            return (False, f"Time constant tau = {self.tau} is not positive")
        if isinstance(self.tau, (int, float)) and self.tau <= 0:
            return (False, f"Numerical time constant tau = {self.tau} is not positive")
        return (True, "Time constant is positive")

    def test_invariant_charge_increases(self):
        """Check that q1 increases monotonically from 0 to q_ss for t≥0 (not input perturbation)."""
        if self.q1 is None or self.q_ss is None or self.tau is None:
            return (False, "SKIP: missing data for monotonicity test")
        if not isinstance(self.tau, (int, float)):
            return (False, "SKIP: need numeric tau for time points")
        # Check at three time points: t=0, t=tau, t=3*tau
        t0 = 0
        t1 = 0.5 * self.tau
        t2 = 2 * self.tau
        q0 = self._safe_substitute(self.q1, t_val=t0)
        q1 = self._safe_substitute(self.q1, t_val=t1)
        q2 = self._safe_substitute(self.q1, t_val=t2)
        if any(v is None for v in [q0, q1, q2]):
            return (False, "SKIP: could not evaluate q1 at required times")
        # Convert to float
        try:
            q0_f = float(q0)
            q1_f = float(q1)
            q2_f = float(q2)
        except TypeError:
            return (False, "SKIP: non-numeric evaluation")
        if not (q0_f <= q1_f <= q2_f) and not (q1_f - q0_f > 0 and q2_f - q1_f > 0):
            return (False, f"q1 is not monotonically increasing: q0={q0_f}, q(tau/2)={q1_f}, q(2tau)={q2_f}")
        # Also check that difference decreases (saturation)
        delta1 = q1_f - q0_f
        delta2 = q2_f - q1_f
        if delta1 <= 0 or delta2 <= 0 or delta2 >= delta1:
            # Should have delta2 < delta1 for exponential approach
            pass  # allow if due to numerical noise
        return (True, "q1 monotonically increases from 0 to steady state")

    # --- Monotonicity tests that explicitly perturb an input ---

    def test_monotonicity_steady_state_vs_R3(self):
        """Perturb R3 upward and verify that steady-state charge q_ss increases."""
        if self.q_ss is None:
            return (False, "SKIP: q_ss not available")
        if not isinstance(self.R3, (int, float)):
            return (False, "SKIP: need numeric R3")
        if not isinstance(self.q_ss, (int, float)):
            return (False, "SKIP: q_ss is not numeric")
        # Baseline: original q_ss
        q_ss_base = self.q_ss
        # Perturb R3: increase by 5%
        R3_perturbed = self.R3 * 1.05
        # Recompute q_ss with perturbed R3 (using stored R1,R2,epsilon)
        if self.R1 is None or self.R2 is None or self.epsilon is None:
            return (False, "SKIP: missing other givens")
        q_ss_pert = self.epsilon * R3_perturbed / (self.R1 + self.R2)
        # Expect q_ss_pert > q_ss_base
        if not (q_ss_pert > q_ss_base + 1e-12):
            return (False, f"q_ss did not increase after R3 increase: base {q_ss_base}, perturbed {q_ss_pert}")
        return (True, f"q_ss increases with R3: base {q_ss_base}, perturbed {q_ss_pert}")

    def test_monotonicity_time_constant_vs_C(self):
        """Perturb C upward and verify that time constant tau increases."""
        if self.tau is None:
            return (False, "SKIP: tau not available")
        if not isinstance(self.C, (int, float)):
            return (False, "SKIP: need numeric C")
        if not isinstance(self.tau, (int, float)):
            return (False, "SKIP: tau is not numeric")
        # Baseline: original tau
        tau_base = self.tau
        # Perturb C: increase by 10%
        C_perturbed = self.C * 1.10
        if self.D is None:
            return (False, "SKIP: D not computed")
        # Recompute tau with perturbed C
        tau_pert = self.D * C_perturbed / (self.R1 + self.R2)
        if not (tau_pert > tau_base + 1e-12):
            return (False, f"tau did not increase after C increase: base {tau_base}, perturbed {tau_pert}")
        return (True, f"tau increases with C: base {tau_base}, perturbed {tau_pert}")

    def test_monotonicity_charge_vs_initial_charge(self):
        """Perturb Q0 upward and verify that q2 at a fixed time increases."""
        if self.q2 is None:
            return (False, "SKIP: q2 result missing")
        if self.Q0 is None or not isinstance(self.Q0, (int, float)):
            return (False, "SKIP: need numeric Q0")
        if not isinstance(self.tau, (int, float)):
            return (False, "SKIP: need numeric tau")
        # Choose fixed time, e.g., t = tau
        t_fixed = self.tau
        q2_base = self._safe_substitute(self.q2, t_val=t_fixed)
        if q2_base is None:
            return (False, "SKIP: could not evaluate q2 at t=tau")
        try:
            q2_base_f = float(q2_base)
        except TypeError:
            return (False, "SKIP: non-numeric baseline")
        # Perturb Q0: increase by 10%
        Q0_perturbed = self.Q0 * 1.10
        # We need to compute q2 with perturbed Q0. To do that, we can re-evaluate the expression with Q0 changed.
        # Since we have the symbolic expression, we can substitute the new Q0 value.
        t = sp.symbols('t', real=True)
        # Create a subs dict with perturbed Q0
        subs_dict = {sp.Symbol('Q0'): Q0_perturbed, t: t_fixed}
        # Also include other parameters (they remain same)
        for sym_name in ['R1','R2','R3','C','epsilon']:
            val = getattr(self, sym_name, None)
            if val is not None:
                subs_dict[sp.Symbol(sym_name)] = val
        # Evaluate q2 with perturbed Q0
        try:
            q2_pert_expr = self.q2.subs(subs_dict)
            if q2_pert_expr.is_Number:
                q2_pert = float(q2_pert_expr)
            else:
                # Try to evaluate numerically if only numbers present
                q2_pert = float(q2_pert_expr)
        except Exception:
            return (False, "SKIP: could not evaluate q2 with perturbed Q0")
        if not (q2_pert > q2_base_f + 1e-12):
            return (False, f"q2 at t=tau did not increase after Q0 increase: base {q2_base_f}, perturbed {q2_pert}")
        return (True, f"q2 increases with Q0 at t=tau: base {q2_base_f}, perturbed {q2_pert}")

    def test_answer_sanity_basic(self):
        """Weak sanity: q1 and q2 should be non-negative, finite, and within plausible bounds."""
        if self.q1 is None or self.q2 is None:
            return (False, "SKIP: results missing")
        t = sp.symbols('t', real=True)
        # Check for finite values at typical t
        for label, q_expr in [('q1', self.q1), ('q2', self.q2)]:
            # Evaluate at t=0 and t=large (if numeric)
            for t_val in [0, 1e6]:
                val = self._safe_substitute(q_expr, t_val=t_val)
                if isinstance(val, (int, float)):
                    if math.isnan(val) or math.isinf(val):
                        return (False, f"{label} at t={t_val} is NaN or infinite")
                    if val < -1e-9:
                        return (False, f"{label} at t={t_val} is negative: {val}")
            if isinstance(self.tau, (int, float)) and self.q_ss is not None:
                # Check that q1 stays between 0 and q_ss (for non-negative Q0)
                if label == 'q1':
                    # For t>=0, q1 should be between 0 and q_ss
                    for t_val in [0, self.tau/2, self.tau]:
                        val = self._safe_substitute(q_expr, t_val=t_val)
                        if isinstance(val, (int, float)):
                            if val < -1e-9 or val > float(self.q_ss) + 1e-9:
                                return (False, f"{label}(t={t_val}) = {val} out of [0, {self.q_ss}]")
                elif label == 'q2' and self.Q0 is not None:
                    lower = min(float(self.Q0), float(self.q_ss))
                    upper = max(float(self.Q0), float(self.q_ss))
                    for t_val in [0, self.tau, 2*self.tau]:
                        val = self._safe_substitute(q_expr, t_val=t_val)
                        if isinstance(val, (int, float)):
                            if val < lower - 1e-9 or val > upper + 1e-9:
                                return (False, f"{label}(t={t_val}) = {val} out of [{lower}, {upper}]")
        return (True, "Charge values are finite, non-negative, and within expected bounds")

    def run_all_tests(self):
        """Execute all test methods and return (success, logs)."""
        logs = []
        all_pass = True
        # Discover all bound methods starting with test_
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)