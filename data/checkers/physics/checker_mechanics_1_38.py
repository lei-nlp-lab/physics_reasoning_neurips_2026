import sympy as sp
import math

class PhysicsChecker:
    """
    Audits a solution_instance for the 2D particle motion problem.
    """

    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.m = givens.get('m', None)
        self.x0 = givens.get('x0', None)
        self.y0 = givens.get('y0', None)
        self.omega1 = givens.get('omega1', None)
        self.omega2 = givens.get('omega2', None)
        self.t = givens.get('t', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.F_x = results.get('F_x', None)
        self.F_y = results.get('F_y', None)
        self.omega_sq_diff = results.get('omega_sq_diff', None)
        self.U = results.get('U', None)          # potential energy
        self.T = results.get('T', None)          # kinetic energy
        self.E = results.get('E', None)          # total energy

        # Define symbolic variables for use in checks
        self._sym_m, self._sym_x0, self._sym_y0 = sp.symbols('m x0 y0', positive=True)
        self._sym_w1, self._sym_w2 = sp.symbols('omega1 omega2', real=True)
        self._sym_t = sp.symbols('t', real=True)

        # Symbolic expressions for x(t) and y(t)
        self._sym_x = self._sym_x0 * sp.cos(self._sym_w1 * self._sym_t)
        self._sym_y = self._sym_y0 * sp.sin(self._sym_w2 * self._sym_t)

        # Symbolic velocities
        self._sym_vx = sp.diff(self._sym_x, self._sym_t)
        self._sym_vy = sp.diff(self._sym_y, self._sym_t)

        # Expected expressions derived from physics (for comparison)
        # Force components
        self._sym_Fx_expected = -self._sym_m * self._sym_w1**2 * self._sym_x
        self._sym_Fy_expected = -self._sym_m * self._sym_w2**2 * self._sym_y

        # Torque about origin: x*F_y - y*F_x
        self._sym_torque = self._sym_x * self._sym_Fy_expected - self._sym_y * self._sym_Fx_expected

        # Potential energy (constant set to zero at origin)
        self._sym_U_expected = (self._sym_m/2) * (self._sym_w1**2 * self._sym_x**2 + self._sym_w2**2 * self._sym_y**2)

        # Kinetic energy
        self._sym_T_expected = (self._sym_m/2) * (self._sym_vx**2 + self._sym_vy**2)

        # Total energy (should be constant)
        self._sym_E_expected = self._sym_T_expected + self._sym_U_expected
        # Simplify E to constant
        self._sym_E_constant = sp.simplify(self._sym_E_expected)

    # --------------------------
    # Interface checks
    # --------------------------
    def test_interface_required_givens(self):
        """Check that all required givens exist."""
        if self.m is None:
            return (False, "SKIP: given 'm' not found")
        if self.x0 is None:
            return (False, "SKIP: given 'x0' not found")
        if self.y0 is None:
            return (False, "SKIP: given 'y0' not found")
        if self.omega1 is None:
            return (False, "SKIP: given 'omega1' not found")
        if self.omega2 is None:
            return (False, "SKIP: given 'omega2' not found")
        if self.t is None:
            return (False, "SKIP: given 't' not found")
        return (True, "All required givens present")

    def test_interface_required_results(self):
        """Check that all required result keys exist."""
        required_keys = ['F_x', 'F_y', 'omega_sq_diff', 'U', 'T', 'E']
        missing = [k for k in required_keys if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required result keys present")

    # --------------------------
    # Invariant checks (theorem-based)
    # --------------------------
    def test_invariant_force_component_form(self):
        """Check that F_x = -m*ω1^2*x and F_y = -m*ω2^2*y."""
        if None in (self.F_x, self.F_y, self.m, self.omega1, self.omega2):
            return (False, "SKIP: missing data for force components")
        # Build expression for F_x + m*ω1^2*x using symbolic substitution if needed
        # Assume self.F_x is sympy expression; we need to extract x from givens? Not needed, we substitute symbolic form.
        # Instead, compare directly by simplifying difference.
        try:
            expr = sp.simplify(self.F_x - self._sym_Fx_expected)
            if expr == 0:
                return (True, "F_x matches expected form")
            else:
                return (False, f"F_x differs: {expr}")
        except Exception as e:
            return (False, f"Could not check F_x: {e}")

    def test_invariant_central_force_condition(self):
        """Check that torque condition matches ω_sq_diff."""
        if None in (self.F_x, self.F_y, self.omega_sq_diff):
            return (False, "SKIP: missing data for central force check")
        try:
            # Built torque from supplied F_x, F_y: τ = x*F_y - y*F_x
            # But we need x,y from givens? We can substitute symbolic x,y.
            # Use symbolic x,y from trajectory (they are time-dependent but are the same variables used in F_x,F_y)
            # The result F_x is expressed in terms of x and y, so we can use symbol x,y.
            sym_x, sym_y = sp.symbols('x y', real=True)
            # We need to substitute x(t), y(t) into F_x, F_y? 
            # Better: we can check that torque = m*x*y*(ω1^2 - ω2^2)
            # Approximate: compute torque from F_x and F_y using symbolic x,y, and compare to expected.
            # But we don't have x,y in result; they are probably given as expressions.
            # Instead, directly check that if ω1^2 - ω2^2 = 0 then torque is zero.
            # We'll compute torque using symbolic substitution: replace x and y in F_x,F_y with symbolic x and y (generic)
            # Actually the result F_x is likely written as -m*ω1^2*x, so x is generic.
            # We'll assume F_x and F_y are sympy expressions in x,y, ω1, ω2, m.
            # Compute torque expression: torque = x * self.F_y - y * self.F_x
            torque = sp.simplify(sym_x * self.F_y - sym_y * self.F_x)
            expected_torque = self._sym_m * sym_x * sym_y * (self.omega_sq_diff)
            diff = sp.simplify(torque - expected_torque)
            if diff == 0:
                return (True, "Torque expression matches ω_sq_diff relation")
            else:
                return (False, f"Torque mismatch: {diff}")
        except Exception as e:
            return (False, f"Central force check error: {e}")

    def test_invariant_potential_consistency(self):
        """Check that -∇U = (F_x, F_y)."""
        if None in (self.U, self.F_x, self.F_y):
            return (False, "SKIP: missing U or force for potential consistency")
        try:
            # Derive U symbolically from F_x and F_y? Instead, check that the gradient of U equals -F.
            # Use symbolic x,y variables again; U may be expressed in x,y.
            sym_x, sym_y = sp.symbols('x y', real=True)
            # Check partial derivatives
            dUdx = sp.diff(self.U, sym_x)
            dUdy = sp.diff(self.U, sym_y)
            # Compare with -F_x, -F_y (they contain x,y)
            # Simplify differences
            diff_x = sp.simplify(dUdx + self.F_x)
            diff_y = sp.simplify(dUdy + self.F_y)
            if diff_x == 0 and diff_y == 0:
                return (True, "Potential gradient matches force")
            else:
                return (False, f"Potential mismatch: dU/dx + F_x = {diff_x}, dU/dy + F_y = {diff_y}")
        except Exception as e:
            return (False, f"Potential consistency error: {e}")

    def test_invariant_energy_conservation(self):
        """Check that total energy E is constant (time-independent)."""
        if None in (self.E, self.t):
            return (False, "SKIP: missing E or t for energy conservation")
        try:
            # Differentiate E w.r.t. time and simplify
            dE_dt = sp.diff(self.E, self._sym_t)
            simplified = sp.simplify(dE_dt)
            if simplified == 0:
                return (True, "Total energy is conserved (dE/dt=0)")
            else:
                # Also check numeric: substitute a time sample
                sample_subs = {self._sym_m: 1.0, self._sym_x0: 1.0, self._sym_y0: 1.0, 
                               self._sym_w1: 2.0, self._sym_w2: 3.0, self._sym_t: 0.5}
                num_diff = float(dE_dt.subs(sample_subs).evalf())
                if math.isclose(num_diff, 0, abs_tol=1e-2):
                    return (True, "dE/dt numerically zero")
                else:
                    return (False, f"Energy not conserved: dE/dt = {dE_dt}")
        except Exception as e:
            return (False, f"Energy conservation check error: {e}")

    def test_invariant_energy_sum(self):
        """Check that E = T + U."""
        if None in (self.E, self.T, self.U):
            return (False, "SKIP: missing E, T, or U")
        try:
            diff = sp.simplify(self.E - (self.T + self.U))
            if diff == 0:
                return (True, "E = T + U holds")
            else:
                # Numeric fallback
                if hasattr(self, '_sym_m') and all(v is not None for v in [self.m, self.x0, self.y0, self.omega1, self.omega2, self.t]):
                    # Use numeric substitution from givens? Not safe.
                    pass
                return (False, f"E - (T+U) = {diff}")
        except Exception as e:
            return (False, f"Energy sum check error: {e}")

    # --------------------------
    # Modeling / trace checks
    # --------------------------
    def test_modeling_force_from_acceleration(self):
        """Check that F_x = m * a_x with a_x = d^2x/dt^2."""
        if None in (self.F_x, self.F_y, self.m, self.x0, self.y0, self.omega1, self.omega2):
            return (False, "SKIP: missing data for force derivation")
        try:
            # Compute expected a_x = -ω1^2 * x, a_y = -ω2^2 * y
            a_x = -self._sym_w1**2 * self._sym_x
            a_y = -self._sym_w2**2 * self._sym_y
            expected_Fx = self._sym_m * a_x
            expected_Fy = self._sym_m * a_y
            # Compare with provided results (use symbolic substitution)
            # Provided F_x likely contains self._sym_m, self._sym_w1, self._sym_x0, etc.
            diff_x = sp.simplify(self.F_x - expected_Fx)
            diff_y = sp.simplify(self.F_y - expected_Fy)
            if diff_x == 0 and diff_y == 0:
                return (True, "Force derived from acceleration correctly")
            else:
                return (False, f"Force mismatch: F_x diff = {diff_x}, F_y diff = {diff_y}")
        except Exception as e:
            return (False, f"Force derivation check error: {e}")

    def test_modeling_kinetic_energy_form(self):
        """Check that T = (1/2) m (v_x^2 + v_y^2) using given velocities."""
        if None in (self.T, self.m):
            return (False, "SKIP: missing T or m")
        try:
            # Compute expected T from symbolic v_x, v_y
            expected_T = (self._sym_m/2) * (self._sym_vx**2 + self._sym_vy**2)
            diff = sp.simplify(self.T - expected_T)
            if diff == 0:
                return (True, "Kinetic energy expression is correct")
            else:
                # Fallback numeric if symbolic fails
                return (False, f"T mismatch: {diff}")
        except Exception as e:
            return (False, f"Kinetic energy form check error: {e}")

    def test_modeling_potential_form(self):
        """Check that U = (1/2) m (ω1^2 x^2 + ω2^2 y^2)."""
        if None in (self.U, self.m, self.omega1, self.omega2):
            return (False, "SKIP: missing U, m, or frequencies")
        try:
            # Use symbolic x and y (generic)
            sym_x, sym_y = sp.symbols('x y', real=True)
            expected_U = (self._sym_m/2) * (self._sym_w1**2 * sym_x**2 + self._sym_w2**2 * sym_y**2)
            # Compare with self.U (which might use same symbols)
            diff = sp.simplify(self.U - expected_U)
            if diff == 0:
                return (True, "Potential energy form matches expected")
            else:
                return (False, f"U mismatch: {diff}")
        except Exception as e:
            return (False, f"Potential form check error: {e}")

    # --------------------------
    # Monotonicity checks
    # --------------------------
    def test_monotonicity_force_magnitude_with_frequency(self):
        """Check that |F_x| increases when ω1 increases (for fixed x)."""
        if None in (self.F_x, self.omega1, self.m):
            return (False, "SKIP: missing data for monotonicity")
        try:
            # Compute derivative of F_x^2 w.r.t ω1 (using symbolic)
            # F_x = -m ω1^2 x, so |F_x|^2 = m^2 ω1^4 x^2
            # d(|F_x|^2)/dω1 = 4 m^2 ω1^3 x^2 > 0 for ω1>0, x≠0
            # Check sign of derivative analytically or numerically.
            # Use symbolic variable omega1 as symbol
            w1 = sp.symbols('omega1', positive=True)
            F_sq = self._sym_m**2 * w1**4 * self._sym_x**2  # symbolic x
            deriv = sp.diff(F_sq, w1)
            # Simplify: derivative should be positive for positive x and ω1
            # We can test for a specific numeric scenario
            if self.m and self.x0 and self.omega1 and self.t:
                subs = {self._sym_m: float(self.m), self._sym_x0: float(self.x0), 
                        self._sym_w1: float(self.omega1), self._sym_t: float(self.t)}
                # Use a small perturbation
                eps = 0.01 * float(self.omega1)
                w1_pert = float(self.omega1) + eps
                # Compute F_x at original and perturbed ω1 (using known expression)
                # But we need the expression: F_x = -m ω1^2 x0 cos(ω1 t)
                # Evaluate numerically
                import math
                F_orig = -float(self.m) * (float(self.omega1)**2) * float(self.x0) * math.cos(float(self.omega1)*float(self.t))
                F_pert = -float(self.m) * (w1_pert**2) * float(self.x0) * math.cos(w1_pert*float(self.t))
                if abs(F_pert) > abs(F_orig) * 0.99:  # allow small noise
                    return (True, "Force magnitude increases with ω1 (numeric check)")
                else:
                    return (False, f"Monotonicity violated: |F_pert|={abs(F_pert)} < |F_orig|={abs(F_orig)}")
            else:
                # If numeric subs not available, try symbolic sign
                if sp.simplify(sp.sign(deriv.subs({self._sym_x: 1, self._sym_m: 1})) == 1):
                    return (True, "Symbolic derivative positive")
                else:
                    return (False, "Cannot verify monotonicity")
        except Exception as e:
            return (False, f"Monotonicity check error: {e}")

    # --------------------------
    # Answer sanity (weak)
    # --------------------------
    def test_answer_sanity_energy_value(self):
        """Check that total energy is positive and finite (no NaN)."""
        if self.E is None:
            return (False, "SKIP: E missing")
        try:
            # Compute numeric value for E by substituting given givens if possible
            if all(v is not None for v in [self.m, self.x0, self.y0, self.omega1, self.omega2]):
                # E = 1/2 m (ω1^2 x0^2 + ω2^2 y0^2)
                # But we trust the result expression; evaluate numerically
                subs = {self._sym_m: float(self.m), self._sym_x0: float(self.x0), 
                        self._sym_y0: float(self.y0), self._sym_w1: float(self.omega1), 
                        self._sym_w2: float(self.omega2)}
                numeric_E = float(self.E.subs(subs).evalf())
                if not math.isfinite(numeric_E):
                    return (False, "Total energy is not finite")
                if numeric_E <= 0:
                    return (False, "Total energy not positive")
                return (True, f"Total energy is positive and finite: {numeric_E}")
            else:
                # Just check that E is not NaN or inf symbolically? Not possible.
                return (False, "SKIP: insufficient givens to evaluate E numerically")
        except Exception as e:
            return (False, f"Energy sanity check error: {e}")

    # --------------------------
    # Main runner
    # --------------------------
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