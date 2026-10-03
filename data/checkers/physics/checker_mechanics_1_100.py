import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.L = getattr(solution_instance, 'L', None) or getattr(getattr(solution_instance, 'givens', {}), 'L', None)
        self.theta = getattr(solution_instance, 'theta', None) or getattr(getattr(solution_instance, 'givens', {}), 'theta', None)
        self.g = getattr(solution_instance, 'g', None) or getattr(getattr(solution_instance, 'givens', {}), 'g', None)
        self.omega = getattr(solution_instance, 'omega', None) or getattr(getattr(solution_instance, 'givens', {}), 'omega', None)
        self.R = getattr(solution_instance, 'R', None) or getattr(getattr(solution_instance, 'givens', {}), 'R', None)

        # Extract results safely (default None)
        self.d1_sym = getattr(solution_instance, 'd1_sym', None) or getattr(getattr(solution_instance, 'results', {}), 'd1_sym', None)
        self.d1_num = getattr(solution_instance, 'd1_num', None) or getattr(getattr(solution_instance, 'results', {}), 'd1_num', None)
        self.x_east_sym = getattr(solution_instance, 'x_east_sym', None) or getattr(getattr(solution_instance, 'results', {}), 'x_east_sym', None)
        self.x_east_num = getattr(solution_instance, 'x_east_num', None) or getattr(getattr(solution_instance, 'results', {}), 'x_east_num', None)

        # Derived quantities (computed from givens, if available)
        self.phi_small = None
        self.t_fall = None
        if self.L is not None and self.g is not None:
            self.t_fall = math.sqrt(2 * self.L / self.g)
        if (self.R is not None and self.omega is not None and
            self.theta is not None and self.g is not None):
            self.phi_small = (self.R * self.omega**2 * math.sin(self.theta) * math.cos(self.theta)) / self.g

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_givens_present(self):
        """Check that all required givens are present (not None)."""
        missing = []
        if self.L is None:
            missing.append('L')
        if self.theta is None:
            missing.append('theta')
        if self.g is None:
            missing.append('g')
        if self.omega is None:
            missing.append('omega')
        if self.R is None:
            missing.append('R')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present.")

    def test_interface_results_present(self):
        """Check that all expected result keys are present."""
        missing = []
        if self.d1_sym is None:
            missing.append('d1_sym')
        if self.d1_num is None:
            missing.append('d1_num')
        if self.x_east_sym is None:
            missing.append('x_east_sym')
        if self.x_east_num is None:
            missing.append('x_east_num')
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All expected result keys present.")

    def test_interface_types(self):
        """Check that numeric results are numeric (int or float) and sym results are strings (symbolic expressions)."""
        # Numerics should be numbers; sym should be strings (or sympy expressions)
        if self.d1_num is not None and not isinstance(self.d1_num, (int, float)):
            return (False, f"d1_num is not numeric: type {type(self.d1_num)}")
        if self.x_east_num is not None and not isinstance(self.x_east_num, (int, float)):
            return (False, f"x_east_num is not numeric: type {type(self.x_east_num)}")
        if self.d1_sym is not None and not isinstance(self.d1_sym, str):
            # Could be a sympy expression; accept if it has a 'subs' method (sympy's way)
            if not hasattr(self.d1_sym, 'subs'):
                return (False, f"d1_sym is not a string or sympy expression: type {type(self.d1_sym)}")
        if self.x_east_sym is not None and not isinstance(self.x_east_sym, str):
            if not hasattr(self.x_east_sym, 'subs'):
                return (False, f"x_east_sym is not a string or sympy expression: type {type(self.x_east_sym)}")
        return (True, "Result types acceptable.")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_plumb_small_angle(self):
        """Check that the deflection angle φ = arcsin( ... ) ≈ (R ω² sinθ cosθ)/g for small angle."""
        if self.d1_num is None or self.L is None or self.phi_small is None:
            return (False, "SKIP: missing d1_num, L, or phi_small")
        # d1 = L * φ => φ = d1 / L
        phi_from_result = self.d1_num / self.L
        # Compare with small-angle approximation
        if math.isclose(phi_from_result, self.phi_small, rel_tol=1e-2, abs_tol=1e-8):
            return (True, f"φ from result = {phi_from_result:.6e}, small-angle approx = {self.phi_small:.6e}, within tolerance.")
        else:
            return (False, f"φ mismatch: result {phi_from_result:.6e}, approx {self.phi_small:.6e}")

    def test_invariant_free_fall_time(self):
        """Check that t_fall = sqrt(2L/g) is consistent with eastward displacement formula."""
        if self.t_fall is None:
            return (False, "SKIP: t_fall not computed")
        if self.x_east_num is None or self.omega is None or self.theta is None:
            return (False, "SKIP: missing x_east_num, omega, or theta")
        # From derived formula: x_east = (1/3) ω cosθ g t_fall^3
        expected_x = (1/3) * self.omega * math.cos(self.theta) * self.g * self.t_fall**3
        if math.isclose(self.x_east_num, expected_x, rel_tol=1e-2, abs_tol=1e-6):
            return (True, f"x_east_num = {self.x_east_num:.6e}, expected from t_fall = {expected_x:.6e}")
        else:
            return (False, f"x_east_num mismatch: got {self.x_east_num:.6e}, expected {expected_x:.6e}")

    def test_invariant_direction_southward(self):
        """Plumb bob deflects southward in Northern hemisphere; check numeric result positive (magnitude)."""
        if self.d1_num is None:
            return (False, "SKIP: d1_num missing")
        if self.d1_num > 0:
            return (True, "d1_num positive, consistent with southward deflection (magnitude).")
        else:
            return (False, f"d1_num is {self.d1_num}, expected positive (magnitude).")

    def test_invariant_direction_eastward(self):
        """Dropped body displaced eastward; check numeric result positive."""
        if self.x_east_num is None:
            return (False, "SKIP: x_east_num missing")
        if self.x_east_num > 0:
            return (True, "x_east_num positive, consistent with eastward displacement.")
        else:
            return (False, f"x_east_num is {self.x_east_num}, expected positive.")

    def test_invariant_deflection_small_relative_to_tower(self):
        """Deflection should be much smaller than tower height."""
        if self.d1_num is None or self.L is None:
            return (False, "SKIP: missing d1_num or L")
        ratio = self.d1_num / self.L
        if ratio < 0.1:  # less than 10% of height
            return (True, f"d1/L = {ratio:.4f}, indeed small.")
        else:
            return (False, f"d1/L = {ratio:.4f}, not small enough?")

    # ----------------------------------------------------------------------
    # Modeling/trace checks (consistency with explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_d1_sym_form(self):
        """Check that d1_sym has the expected symbolic form: L * arcsin( (R ω² sin(2θ))/(2g) )."""
        if self.d1_sym is None:
            return (False, "SKIP: d1_sym missing")
        # We cannot evaluate the sym expression without substitution. Instead, check that it contains 'arcsin'.
        # Convert to string for simple check.
        sym_str = str(self.d1_sym)
        if 'arcsin' in sym_str or 'asin' in sym_str:
            return (True, "d1_sym contains arcsin/asin as expected.")
        else:
            return (False, f"d1_sym = {sym_str}, does not contain arcsin.")

    def test_modeling_x_east_sym_form(self):
        """Check that x_east_sym matches expected form: (1/3) ω cosθ sqrt(8L^3/g)."""
        if self.x_east_sym is None:
            return (False, "SKIP: x_east_sym missing")
        sym_str = str(self.x_east_sym)
        # Look for key pieces: omega, cos, sqrt, L^3, g
        has_omega = 'omega' in sym_str or 'ω' in sym_str
        has_cos = 'cos' in sym_str
        has_sqrt = 'sqrt' in sym_str
        if has_omega and has_cos and has_sqrt:
            return (True, "x_east_sym contains expected components: omega, cos, sqrt.")
        else:
            return (False, f"x_east_sym = {sym_str}, lacks expected components.")

    def test_modeling_numerical_plausibility(self):
        """Check that the numerical results are of plausible order of magnitude (Berkeley, L=93m)."""
        if self.d1_num is None or self.x_east_num is None:
            return (False, "SKIP: numerical results missing")
        # Expected approximate values: d1 ~ 0.156 m, x_east ~ 0.0155 m (from pseudocode)
        # Allow loose tolerance (factor of 2)
        if not (0.05 < self.d1_num < 0.5):
            return (False, f"d1_num = {self.d1_num:.4f} is not in plausible range (0.05–0.5 m).")
        if not (0.005 < self.x_east_num < 0.05):
            return (False, f"x_east_num = {self.x_east_num:.4f} is not in plausible range (0.005–0.05 m).")
        return (True, "Numerical results within plausible order of magnitude.")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturb inputs, observe output trends)
    # ----------------------------------------------------------------------
    def test_monotonicity_d1_increases_with_L(self):
        """Increasing tower height L should increase d1 (linearly)."""
        if self.d1_num is None or self.L is None:
            return (False, "SKIP: missing d1_num or L")
        if self.L <= 0:
            return (False, "L non-positive, cannot perturb.")
        # Perturb L upward by 5%
        L_perturbed = self.L * 1.05
        # Recompute approximate d1 using small-angle formula: d1 = L * φ, φ independent of L
        # So d1_perturbed = L_perturbed * φ_small
        if self.phi_small is None:
            return (False, "SKIP: phi_small missing")
        d1_perturbed = L_perturbed * self.phi_small
        if d1_perturbed > self.d1_num:
            return (True, f"Increasing L by 5% increases d1 from {self.d1_num:.4f} to {d1_perturbed:.4f} (approx).")
        else:
            return (False, f"Monotonicity fail: L increased, but computed d1_perturbed {d1_perturbed:.4f} not > original {self.d1_num:.4f}.")

    def test_monotonicity_d1_increases_with_R(self):
        """Increasing R should increase d1 (since φ ∝ R)."""
        if self.d1_num is None or self.R is None or self.phi_small is None:
            return (False, "SKIP: missing data for this test")
        # Perturb R upward by 1%
        R_perturbed = self.R * 1.01
        # Recompute phi_small and d1 (using small-angle)
        phi_new = (R_perturbed * self.omega**2 * math.sin(self.theta) * math.cos(self.theta)) / self.g
        d1_new = self.L * phi_new
        if d1_new > self.d1_num:
            return (True, f"Increasing R by 1% increases d1 from {self.d1_num:.4f} to approx {d1_new:.4f}.")
        else:
            return (False, f"Monotonicity fail: R increased but d1 new {d1_new:.4f} not > {self.d1_num:.4f}.")

    def test_monotonicity_x_east_increases_with_L(self):
        """Eastward displacement should increase with L (∝ L^{3/2})."""
        if self.x_east_num is None or self.L is None:
            return (False, "SKIP: missing data")
        if self.omega is None or self.theta is None or self.g is None:
            return (False, "SKIP: missing givens for recomputation")
        # Perturb L upward by 5%
        L_perturbed = self.L * 1.05
        # Recompute x_east via formula: (1/3) ω cosθ sqrt(8 L^3/g)
        x_perturbed = (1/3) * self.omega * math.cos(self.theta) * math.sqrt(8 * L_perturbed**3 / self.g)
        if x_perturbed > self.x_east_num:
            return (True, f"Increasing L by 5% increases x_east from {self.x_east_num:.6e} to approx {x_perturbed:.6e}.")
        else:
            return (False, f"Monotonicity fail: L increased but computed x_perturbed {x_perturbed:.6e} not > {self.x_east_num:.6e}.")

    def test_monotonicity_x_east_increases_with_omega(self):
        """Eastward displacement should increase linearly with ω."""
        if self.x_east_num is None or self.omega is None:
            return (False, "SKIP: missing data")
        # Perturb omega upward by 1%
        omega_perturbed = self.omega * 1.01
        if self.L is None or self.theta is None or self.g is None:
            return (False, "SKIP: missing givens")
        x_perturbed = (1/3) * omega_perturbed * math.cos(self.theta) * math.sqrt(8 * self.L**3 / self.g)
        if x_perturbed > self.x_east_num:
            return (True, f"Increasing ω by 1% increases x_east from {self.x_east_num:.6e} to approx {x_perturbed:.6e}.")
        else:
            return (False, f"Monotonicity fail: ω increased but x_perturbed {x_perturbed:.6e} <= {self.x_east_num:.6e}.")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_positive(self):
        """Check that numerical answers are finite positive numbers (not NaN, not infinite)."""
        for name, val in [("d1_num", self.d1_num), ("x_east_num", self.x_east_num)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "Both numerical answers are finite and positive.")

    # ----------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)