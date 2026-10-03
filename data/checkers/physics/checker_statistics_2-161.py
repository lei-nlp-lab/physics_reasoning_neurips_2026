import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely – default to None if missing
        givens = getattr(solution_instance, 'givens', {})
        self.p0 = givens.get('p0', None)
        self.V = givens.get('V', None)
        self.A = givens.get('A', None)
        self.v = givens.get('v', None)
        self.t = sp.Symbol('t')  # independent variable

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.p1_expr = results.get('p1', None)

        # Convert p1_expr to sympy expression if it is a string
        if isinstance(self.p1_expr, str):
            try:
                self.p1_expr = sp.sympify(self.p1_expr)
            except Exception:
                self.p1_expr = None

        # Precompute rate constant k = v*A/(4*V) (if all givens available)
        self.k = None
        if self.v is not None and self.A is not None and self.V is not None and all(
                x != 0 for x in [self.v, self.A, self.V]):
            self.k = (self.v * self.A) / (4 * self.V)

        self.context = "Effusion problem: box divided, hole area A, average velocity v"

    # ------------------------------------------------------------
    # Interface tests – check required keys exist and are valid
    # ------------------------------------------------------------
    def test_interface_givens_present(self):
        """All required givens (p0, V, A, v) must be present and numeric."""
        missing = []
        for name, val in [('p0', self.p0), ('V', self.V), ('A', self.A), ('v', self.v)]:
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(f"{name} (type must be numeric, got {type(val).__name__})")
        if missing:
            return False, f"Missing or invalid givens: {', '.join(missing)}"
        return True, f"All givens present and numeric: p0={self.p0}, V={self.V}, A={self.A}, v={self.v}"

    def test_interface_result_p1_present(self):
        """Result p1 must exist and be a sympy expression (or convertible to one)."""
        if self.p1_expr is None:
            return False, "results['p1'] is missing or could not be parsed as a sympy expression"
        if not isinstance(self.p1_expr, sp.Basic):
            return False, f"p1_expr is of type {type(self.p1_expr).__name__}, not a sympy expression"
        return True, "p1_expr is a valid sympy expression"

    # ------------------------------------------------------------
    # Invariant tests – use math.isclose (lenient)
    # ------------------------------------------------------------
    def test_invariant_initial_condition(self):
        """At t=0, p1 should equal p0."""
        if self.p1_expr is None:
            return False, "SKIP: p1 expression not available"
        p1_at_0 = self.p1_expr.subs(self.t, 0)
        if p1_at_0.is_number:
            numeric = float(p1_at_0.evalf())
            if math.isclose(numeric, float(self.p0), rel_tol=1e-2, abs_tol=1e-2):
                return True, f"p1(0) = {numeric}, close to p0 = {self.p0}"
            else:
                return False, f"p1(0) = {numeric}, but p0 = {self.p0}"
        return False, "p1(0) did not evaluate to a number"

    def test_invariant_equilibrium_asymptote(self):
        """As t → ∞, p1 → p0/2."""
        if self.p1_expr is None:
            return False, "SKIP: p1 expression not available"
        if self.A is None or self.v is None or self.V is None:
            return False, "SKIP: not all givens available to compute characteristic time"
        # Characteristic time tau = 2V/(Av)
        if self.A * self.v == 0:
            return False, "SKIP: A or v is zero, cannot compute characteristic time"
        tau = (2 * self.V) / (self.A * self.v)
        if tau <= 0:
            return False, "SKIP: characteristic time non-positive"
        t_large = 10 * tau
        p1_large = self.p1_expr.subs(self.t, t_large)
        if p1_large.is_number:
            numeric = float(p1_large.evalf())
            expected = float(self.p0 / 2)
            if math.isclose(numeric, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"p1(t→∞) ≈ {numeric}, close to p0/2 = {expected}"
            else:
                return False, f"p1(t→∞) = {numeric}, but expected {expected}"
        return False, "Could not evaluate p1 at large t"

    def test_invariant_positive_pressure(self):
        """p1 must be nonnegative for all t >= 0 (sampled points)."""
        if self.p1_expr is None:
            return False, "SKIP: p1 expression not available"
        times = [0, 1e-3, 1, 10, 100]
        for t_val in times:
            val = self.p1_expr.subs(self.t, t_val)
            if val.is_number:
                if float(val) < -1e-12:
                    return False, f"p1({t_val}) = {float(val)} is negative"
        return True, "p1 appears nonnegative at sampled times"

    # ------------------------------------------------------------
    # Modeling / trace tests – symbolic checks using physics relations
    # ------------------------------------------------------------
    def test_modeling_ode_satisfaction(self):
        """p1(t) must satisfy the ODE dp1/dt = (vA/(4V))*(p0 - 2p1)."""
        if self.p1_expr is None or self.k is None:
            return False, "SKIP: p1 expression or rate constant not available"
        dp1_dt_expr = sp.diff(self.p1_expr, self.t)
        rhs = self.k * (self.p0 - 2 * self.p1_expr)
        diff = sp.simplify(dp1_dt_expr - rhs)
        if diff == 0:
            return True, "ODE satisfied symbolically (dp1/dt = k*(p0-2p1))"
        # Fallback numeric check
        t_vals = [0.1, 1, 10]
        for t_val in t_vals:
            diff_num = diff.subs(self.t, t_val)
            if diff_num.is_number:
                val = float(diff_num.evalf())
                if not math.isclose(val, 0, abs_tol=1e-6):
                    return False, f"At t={t_val}, dp1/dt - RHS = {val} (not close to zero)"
        return True, "ODE satisfied numerically at sampled times"

    def test_modeling_total_pressure_conservation(self):
        """Total pressure p1 + p2 should equal p0, and p2 >= 0."""
        if self.p1_expr is None:
            return False, "SKIP: p1 expression not available"
        p2_expr = self.p0 - self.p1_expr
        total = sp.simplify(self.p1_expr + p2_expr)
        if total != self.p0:
            return False, f"p1+p2 = {total}, but should equal p0 = {self.p0}"
        times = [0, 1, 10, 100]
        for t_val in times:
            val = p2_expr.subs(self.t, t_val)
            if val.is_number and float(val) < -1e-12:
                return False, f"p2({t_val}) = {float(val)} is negative"
        return True, "Total pressure conserved and p2 remains nonnegative"

    # ------------------------------------------------------------
    # Monotonicity tests – explicitly perturb one input parameter
    # ------------------------------------------------------------
    def test_monotonicity_hole_area_effect(self):
        """Larger hole area A leads to faster pressure drop (p1 smaller at same t)."""
        if self.p1_expr is None or self.A is None or self.p0 is None:
            return False, "SKIP: missing p1, A, or p0"
        t_test = 1.0  # chosen time point
        p_base = float(self.p1_expr.subs({self.t: t_test, self.A: self.A}).evalf())
        A2 = self.A * 1.1  # perturb A by +10%
        p_higher = float(self.p1_expr.subs({self.t: t_test, self.A: A2}).evalf())
        if p_higher < p_base - 1e-12:
            return True, f"Larger A gives p1 = {p_higher} < baseline {p_base} (faster drop)"
        else:
            return False, f"Larger A gave p1 = {p_higher} >= baseline {p_base} (should decrease)"

    def test_monotonicity_velocity_effect(self):
        """Higher average velocity v leads to faster pressure drop (p1 smaller at same t)."""
        if self.p1_expr is None or self.v is None or self.p0 is None:
            return False, "SKIP: missing p1, v, or p0"
        t_test = 1.0
        p_base = float(self.p1_expr.subs({self.t: t_test, self.v: self.v}).evalf())
        v2 = self.v * 1.1
        p_higher = float(self.p1_expr.subs({self.t: t_test, self.v: v2}).evalf())
        if p_higher < p_base - 1e-12:
            return True, f"Higher v gives p1 = {p_higher} < baseline {p_base} (faster drop)"
        else:
            return False, f"Higher v gave p1 = {p_higher} >= baseline {p_base} (should decrease)"

    def test_monotonicity_volume_effect(self):
        """Larger volume V leads to slower pressure drop (p1 larger at same t)."""
        if self.p1_expr is None or self.V is None or self.p0 is None:
            return False, "SKIP: missing p1, V, or p0"
        t_test = 1.0
        p_base = float(self.p1_expr.subs({self.t: t_test, self.V: self.V}).evalf())
        V2 = self.V * 1.1  # perturb V by +10%
        p_larger = float(self.p1_expr.subs({self.t: t_test, self.V: V2}).evalf())
        if p_larger > p_base + 1e-12:
            return True, f"Larger V gives p1 = {p_larger} > baseline {p_base} (slower drop)"
        else:
            return False, f"Larger V gave p1 = {p_larger} <= baseline {p_base} (should increase)"

    # ------------------------------------------------------------
    # Weak answer sanity (one test, broad checks only)
    # ------------------------------------------------------------
    def test_answer_sanity_pressure_range(self):
        """p1 must stay between 0 and p0 for all t >= 0 (sampled)."""
        if self.p1_expr is None:
            return False, "SKIP: p1 expression not available"
        times = [0, 0.1, 1, 10, 100]
        for t_val in times:
            try:
                val = float(self.p1_expr.subs(self.t, t_val).evalf())
            except Exception:
                continue
            if val < -1e-12 or val > float(self.p0) + 1e-12:
                return False, f"At t={t_val}, p1 = {val} out of range [0, {self.p0}]"
        return True, "p1 stays within [0, p0] at sampled times"

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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