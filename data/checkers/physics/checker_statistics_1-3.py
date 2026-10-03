import math
import sympy
import traceback

class PhysicsChecker:
    """
    Auditor for the bimetallic strip radius of curvature problem.
    """

    def __init__(self, solution_instance):
        # Extract givens safely with None defaults
        self.x = getattr(solution_instance, 'x', None)
        self.alpha1 = getattr(solution_instance, 'alpha1', None)
        self.alpha2 = getattr(solution_instance, 'alpha2', None)
        self.dt = getattr(solution_instance, 'dt', None)
        self.R = getattr(solution_instance, 'R', None)

        # Try optional results dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            if self.x is None:
                self.x = solution_instance.results.get('x', None)
            if self.alpha1 is None:
                self.alpha1 = solution_instance.results.get('alpha1', None)
            if self.alpha2 is None:
                self.alpha2 = solution_instance.results.get('alpha2', None)
            if self.dt is None:
                self.dt = solution_instance.results.get('dt', None)
            if self.R is None:
                self.R = solution_instance.results.get('R', None)

        # Convert to float if possible
        for attr in ['x', 'alpha1', 'alpha2', 'dt', 'R']:
            val = getattr(self, attr, None)
            if val is not None and not isinstance(val, (int, float)):
                try:
                    setattr(self, attr, float(val))
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------

    def test_interface_givens_present(self):
        """All required givens (x, alpha1, alpha2, dt) must be present and numeric."""
        missing = []
        for name in ['x', 'alpha1', 'alpha2', 'dt']:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(f"{name} (non-numeric)")
        if missing:
            return (False, f"Missing or non-numeric givens: {', '.join(missing)}")
        return (True, "All givens present and numeric")

    def test_interface_result_present(self):
        """The result 'R' must be present and numeric."""
        if self.R is None:
            return (False, "Missing result 'R'")
        if not isinstance(self.R, (int, float)):
            return (False, f"Result 'R' is not numeric: {type(self.R).__name__}")
        return (True, "Result 'R' present and numeric")

    def test_interface_alpha2_greater_than_alpha1(self):
        """Constraint: alpha2 > alpha1."""
        if self.alpha2 is None or self.alpha1 is None:
            return (False, "SKIP: alpha1 or alpha2 missing")
        if not (isinstance(self.alpha2, (int, float)) and isinstance(self.alpha1, (int, float))):
            return (False, "Non-numeric alpha values")
        if self.alpha2 <= self.alpha1:
            return (False, f"alpha2 ({self.alpha2}) is not greater than alpha1 ({self.alpha1})")
        return (True, "alpha2 > alpha1 satisfied")

    def test_interface_positive_dt(self):
        """Temperature change must be strictly positive."""
        if self.dt is None:
            return (False, "SKIP: dt missing")
        if not isinstance(self.dt, (int, float)):
            return (False, "Non-numeric dt")
        if self.dt <= 0:
            return (False, f"dt must be > 0, got {self.dt}")
        return (True, "dt > 0")

    def test_interface_positive_x(self):
        """Thickness must be strictly positive."""
        if self.x is None:
            return (False, "SKIP: x missing")
        if not isinstance(self.x, (int, float)):
            return (False, "Non-numeric x")
        if self.x <= 0:
            return (False, f"x must be > 0, got {self.x}")
        return (True, "x > 0")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ------------------------------------------------------------------

    def test_invariant_geometric_thermal_balance(self):
        """
        Using the geometric constraint (difference in arc lengths = x/2 * θ)
        and free expansion difference = L*(α2-α1)*ΔT, and noting L = Rθ,
        we obtain the approximate relation: x = 2 * R * (α2-α1) * ΔT.
        We check this relation to within 2% tolerance.
        """
        if any(v is None for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "SKIP: missing givens or result")
        if not all(isinstance(v, (int, float)) for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "Non-numeric values")
        try:
            diff = self.alpha2 - self.alpha1
            if diff <= 0:
                return (False, "Non-positive alpha difference")
            expected_x = 2.0 * self.R * diff * self.dt
            if math.isclose(expected_x, self.x, rel_tol=1e-2, abs_tol=1e-12):
                return (True, f"2*R*(α2-α1)*ΔT = {expected_x:.6e} matches x = {self.x:.6e}")
            else:
                return (False, f"2*R*(α2-α1)*ΔT = {expected_x:.6e} differs from x = {self.x:.6e} "
                        f"(relative diff {abs(expected_x-self.x)/self.x:.2%})")
        except Exception as e:
            return (False, f"Exception during invariant check: {e}")

    def test_invariant_radius_much_larger_than_thickness(self):
        """Constraint: x << R, i.e., R should be at least 10 times x."""
        if self.x is None or self.R is None:
            return (False, "SKIP: x or R missing")
        if not (isinstance(self.x, (int, float)) and isinstance(self.R, (int, float))):
            return (False, "Non-numeric values")
        if self.R > 10.0 * self.x:
            return (True, f"R = {self.R:.4e} >> x = {self.x:.4e}")
        return (False, f"R = {self.R:.4e} is not sufficiently larger than x = {self.x:.4e}")

    def test_invariant_radius_positive_finite(self):
        """R must be positive, not NaN or infinity."""
        if self.R is None:
            return (False, "SKIP: R missing")
        if not isinstance(self.R, (int, float)):
            return (False, "Non-numeric R")
        if math.isnan(self.R) or math.isinf(self.R):
            return (False, f"R is non-finite: {self.R}")
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}")
        return (True, f"R = {self.R:.4e} is positive and finite")

    # ------------------------------------------------------------------
    # Modeling / trace checks (two tests)
    # ------------------------------------------------------------------

    def test_modeling_arc_length_expansion_consistency(self):
        """
        Modeling quality: Check the consistency between the difference in
        arc lengths and the difference in free thermal expansions.
        The geometric difference in arc lengths per unit interface length is
        (x/2) / R. The free expansion difference per unit length is (α2-α1)*ΔT.
        These should be equal: x/(2R) = (α2-α1)ΔT.
        We verify that the ratio of these two quantities is close to 1.
        """
        if any(v is None for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "Non-numeric values")
        try:
            diff = self.alpha2 - self.alpha1
            if diff <= 0:
                return (False, "Non-positive alpha difference")
            left = self.x / (2.0 * self.R)  # ΔL_arc per unit length
            right = diff * self.dt          # ΔL_free per unit length
            ratio = left / right
            if math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-12):
                return (True, f"Arc / free expansion ratio = {ratio:.6f} (≈1, as required)")
            else:
                return (False, f"Arc / free expansion ratio = {ratio:.6f} (should be ≈1)")
        except Exception as e:
            return (False, f"Exception during modeling check: {e}")

    def test_modeling_ratio_consistency(self):
        """
        Modeling quality: Check that the ratio of the thermal expansion lengths
        equals the ratio of the arc radii (R ± x/4).
        The free expansion length ratio is (1+α2ΔT)/(1+α1ΔT).
        The geometric arc ratio is (R + x/4)/(R - x/4).
        These two must be equal (this follows directly from the problem setup).
        """
        if any(v is None for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "Non-numeric values")
        try:
            # Thermal expansion ratio (assuming same initial length)
            thermal_ratio = (1.0 + self.alpha2 * self.dt) / (1.0 + self.alpha1 * self.dt)
            # Geometric ratio from radii to the outer and inner mid-lines
            # (thickness each x/2, so mid-line at ±x/4 from interface)
            geom_ratio = (self.R + self.x / 4.0) / (self.R - self.x / 4.0)
            if math.isclose(thermal_ratio, geom_ratio, rel_tol=1e-2, abs_tol=1e-12):
                return (True, f"Thermal ratio ({thermal_ratio:.6f}) matches geometric ratio ({geom_ratio:.6f})")
            else:
                return (False, f"Thermal ratio ({thermal_ratio:.6f}) differs from geometric ratio ({geom_ratio:.6f}) "
                        f"(relative diff {abs(thermal_ratio-geom_ratio)/thermal_ratio:.2%})")
        except ZeroDivisionError:
            return (False, "Division by zero: check alpha1*dt = -1?")
        except Exception as e:
            return (False, f"Exception during modeling check: {e}")

    # ------------------------------------------------------------------
    # Monotonicity checks – perturb an input and compare predicted output
    # ------------------------------------------------------------------

    def test_monotonicity_increase_dt_decreases_R(self):
        """If ΔT increases, the radius R should decrease (using approximate relation)."""
        if any(v is None for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "Non-numeric values")
        try:
            diff = self.alpha2 - self.alpha1
            if diff <= 0:
                return (False, "Non-positive alpha difference")
            # Perturb dt by +10%
            pert_dt = self.dt * 1.1
            # Predict new R using the approximate relation R ≈ x/(2*diff*dt)
            new_R = self.x / (2.0 * diff * pert_dt)
            if new_R < self.R:
                return (True, f"Increasing dt by 10% reduces predicted R ({new_R:.6e} < {self.R:.6e})")
            return (False, f"Monotonicity violation: new R = {new_R:.6e} >= original {self.R:.6e}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_monotonicity_increase_alpha_diff_decreases_R(self):
        """If (α2-α1) increases, R should decrease."""
        if any(v is None for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "Non-numeric values")
        try:
            diff = self.alpha2 - self.alpha1
            if diff <= 0:
                return (False, "Non-positive alpha difference")
            # Perturb diff by +10%
            new_diff = diff * 1.1
            new_R = self.x / (2.0 * new_diff * self.dt)
            if new_R < self.R:
                return (True, f"Increasing (α2-α1) by 10% reduces predicted R ({new_R:.6e} < {self.R:.6e})")
            return (False, f"Monotonicity violation: new R = {new_R:.6e} >= original {self.R:.6e}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_monotonicity_increase_x_increases_R(self):
        """If thickness x increases, R should increase."""
        if any(v is None for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.x, self.alpha1, self.alpha2, self.dt, self.R]):
            return (False, "Non-numeric values")
        try:
            diff = self.alpha2 - self.alpha1
            if diff <= 0:
                return (False, "Non-positive alpha difference")
            # Perturb x by +10%
            new_x = self.x * 1.1
            new_R = new_x / (2.0 * diff * self.dt)
            if new_R > self.R:
                return (True, f"Increasing x by 10% increases predicted R ({new_R:.6e} > {self.R:.6e})")
            return (False, f"Monotonicity violation: new R = {new_R:.6e} <= original {self.R:.6e}")
        except Exception as e:
            return (False, f"Exception: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (one test only, no precise formula matching)
    # ------------------------------------------------------------------

    def test_answer_sanity_weak(self):
        """
        Only check that R is positive, finite, and plausibly larger than thickness.
        No exact formula comparison.
        """
        if self.R is None:
            return (False, "SKIP: R missing")
        if not isinstance(self.R, (int, float)):
            return (False, "Non-numeric R")
        if math.isnan(self.R) or math.isinf(self.R):
            return (False, f"R is non-finite: {self.R}")
        if self.R <= 0:
            return (False, f"R = {self.R} is not positive (weak sanity)")
        # Check x << R if x is available
        if self.x is not None and isinstance(self.x, (int, float)) and self.x > 0:
            if self.R <= self.x:
                return (False, f"R = {self.R} is not larger than x = {self.x} (contradicts x<<R)")
        return (True, "R is positive, finite, and plausibly larger than thickness (weak sanity)")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

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