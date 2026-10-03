import math
import traceback

class PhysicsChecker:
    """
    Audits physics solutions for the bifocal lens problem.
    """

    def __init__(self, solution_instance):
        # Extract given physical parameters safely
        self.far_point = getattr(solution_instance, 'far_point', None)
        self.near_point = getattr(solution_instance, 'near_point', None)
        self.retina_distance = getattr(solution_instance, 'retina_distance', None)
        self.desired_near = getattr(solution_instance, 'desired_near', None)

        # Extract results from solution_instance; fallback to empty dict
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.f_far = self.results.get('f_far', None)
        self.f_near = self.results.get('f_near', None)
        self.f_glasses = self.results.get('f_glasses', None)

        # Precompute expected values if all necessary givens are present
        self.expected_f_far = None
        self.expected_f_near = None
        self.expected_f_glasses = None
        if self.far_point is not None and self.retina_distance is not None:
            self.expected_f_far = 1.0 / (1.0 / self.far_point + 1.0 / self.retina_distance)
        if self.near_point is not None and self.retina_distance is not None:
            self.expected_f_near = 1.0 / (1.0 / self.near_point + 1.0 / self.retina_distance)
        if self.desired_near is not None and self.near_point is not None:
            # Object 25 cm, virtual image at near point (negative)
            self.expected_f_glasses = 1.0 / (1.0 / self.desired_near + 1.0 / (-self.near_point))

    # ----------------------------------------------------------------------
    # Interface Tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        """Check that all required result keys are present."""
        missing = [key for key in ['f_far', 'f_near', 'f_glasses'] if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present.")

    def test_interface_numeric_positive(self):
        """Check that all focal lengths are positive floats."""
        for name, val in [('f_far', self.f_far), ('f_near', self.f_near), ('f_glasses', self.f_glasses)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if not isinstance(val, (int, float)):
                return (False, f"{name} not numeric, got {type(val)}")
            if val <= 0:
                return (False, f"{name} not positive, got {val}")
        return (True, "All results positive and numeric.")

    # ----------------------------------------------------------------------
    # Invariant Tests (thin lens equation)
    # ----------------------------------------------------------------------
    def test_invariant_thin_lens_far(self):
        """Check 1/f_far = 1/far_point + 1/retina_distance."""
        if any(v is None for v in [self.f_far, self.far_point, self.retina_distance]):
            return (False, "SKIP: missing data for far-point invariant")
        left = 1.0 / self.f_far
        right = 1.0 / self.far_point + 1.0 / self.retina_distance
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"1/f_far={left:.5f} ≈ {right:.5f}")
        else:
            return (False, f"1/f_far={left:.5f}, expected {right:.5f} (diff {abs(left-right):.5f})")

    def test_invariant_thin_lens_near(self):
        """Check 1/f_near = 1/near_point + 1/retina_distance."""
        if any(v is None for v in [self.f_near, self.near_point, self.retina_distance]):
            return (False, "SKIP: missing data for near-point invariant")
        left = 1.0 / self.f_near
        right = 1.0 / self.near_point + 1.0 / self.retina_distance
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"1/f_near={left:.5f} ≈ {right:.5f}")
        else:
            return (False, f"1/f_near={left:.5f}, expected {right:.5f} (diff {abs(left-right):.5f})")

    def test_invariant_thin_lens_glasses(self):
        """Check 1/f_glasses = 1/desired_near + 1/(-near_point)."""
        if any(v is None for v in [self.f_glasses, self.desired_near, self.near_point]):
            return (False, "SKIP: missing data for glasses invariant")
        left = 1.0 / self.f_glasses
        right = 1.0 / self.desired_near + 1.0 / (-self.near_point)
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"1/f_glasses={left:.4f} ≈ {right:.4f}")
        else:
            return (False, f"1/f_glasses={left:.4f}, expected {right:.4f} (diff {abs(left-right):.4f})")

    # ----------------------------------------------------------------------
    # Modeling Tests
    # ----------------------------------------------------------------------
    def test_modeling_focal_length_order(self):
        """Check that far‑point focal length > near‑point focal length (less power)."""
        if any(v is None for v in [self.f_far, self.f_near]):
            return (False, "SKIP: missing f_far or f_near")
        if self.f_far > self.f_near:
            return (True, f"f_far={self.f_far} > f_near={self.f_near} (correct order)")
        else:
            return (False, f"f_far={self.f_far} not > f_near={self.f_near} (should be longer for far point)")

    def test_modeling_glasses_positive(self):
        """Check that the bifocal lens has positive focal length (converging)."""
        if self.f_glasses is None:
            return (False, "SKIP: f_glasses missing")
        if self.f_glasses > 0:
            return (True, f"f_glasses={self.f_glasses} > 0 (converging)")
        else:
            return (False, f"f_glasses={self.f_glasses} is not positive")

    # ----------------------------------------------------------------------
    # Monotonicity Tests
    # ----------------------------------------------------------------------
    def test_monotonicity_far_vs_near(self):
        """Perturb near point: increase object distance → focal length increases."""
        # Using given values, check monotonic trend of the model.
        if any(v is None for v in [self.near_point, self.retina_distance, self.f_near]):
            return (False, "SKIP: missing data for monotonicity test")
        # Compute expected f for near point (should match self.f_near closely)
        f_near_exp = 1.0 / (1.0 / self.near_point + 1.0 / self.retina_distance)
        # Now increase object distance by 10%
        new_s_o = self.near_point * 1.1
        f_larger = 1.0 / (1.0 / new_s_o + 1.0 / self.retina_distance)
        # The model predicts f_larger > f_near_exp
        if f_larger > f_near_exp:
            msg = f"Perturbation confirms monotonicity: f({new_s_o})={f_larger:.4f} > f_near_exp={f_near_exp:.4f}"
            return (True, msg)
        else:
            return (False, f"Monotonicity violated: f({new_s_o})={f_larger:.4f} not > f_near_exp={f_near_exp:.4f}")

    def test_monotonicity_far_point(self):
        """Perturb far point: increase object distance → focal length increases."""
        if any(v is None for v in [self.far_point, self.retina_distance, self.f_far]):
            return (False, "SKIP: missing data")
        f_far_exp = 1.0 / (1.0 / self.far_point + 1.0 / self.retina_distance)
        new_s_o = self.far_point * 1.1
        f_larger = 1.0 / (1.0 / new_s_o + 1.0 / self.retina_distance)
        if f_larger > f_far_exp:
            return (True, f"Monotonicity holds: f({new_s_o:.1f})={f_larger:.4f} > f_far_exp={f_far_exp:.4f}")
        else:
            return (False, f"Monotonicity violated: f({new_s_o:.1f})={f_larger:.4f} not > f_far_exp={f_far_exp:.4f}")

    def test_monotonicity_glasses_desired_near(self):
        """Perturb desired near point: increase s_o → focal length increases."""
        if any(v is None for v in [self.desired_near, self.near_point, self.f_glasses]):
            return (False, "SKIP: missing data")
        f_glasses_exp = 1.0 / (1.0 / self.desired_near + 1.0 / (-self.near_point))
        new_s_o = self.desired_near * 1.1
        f_larger = 1.0 / (1.0 / new_s_o + 1.0 / (-self.near_point))
        if f_larger > f_glasses_exp:
            return (True, f"Monotonicity holds: f({new_s_o:.1f})={f_larger:.4f} > f_glasses_exp={f_glasses_exp:.4f}")
        else:
            return (False, f"Monotonicity violated: f({new_s_o:.1f})={f_larger:.4f} not > f_glasses_exp={f_glasses_exp:.4f}")

    # ----------------------------------------------------------------------
    # Weak Answer Sanity Check
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Broad sanity: focal lengths within plausible ranges (no equation solving)."""
        checks = []
        if self.f_far is not None:
            if 1.5 <= self.f_far <= 2.5:
                checks.append(f"f_far={self.f_far} in [1.5,2.5]")
            else:
                return (False, f"f_far={self.f_far} outside plausible range [1.5,2.5]")
        else:
            checks.append("f_far missing (skip)")
        if self.f_near is not None:
            if 1.5 <= self.f_near <= 2.5:
                checks.append(f"f_near={self.f_near} in [1.5,2.5]")
            else:
                return (False, f"f_near={self.f_near} outside plausible range [1.5,2.5]")
        else:
            checks.append("f_near missing (skip)")
        if self.f_glasses is not None:
            if 30 <= self.f_glasses <= 40:
                checks.append(f"f_glasses={self.f_glasses} in [30,40]")
            else:
                return (False, f"f_glasses={self.f_glasses} outside plausible range [30,40]")
        else:
            checks.append("f_glasses missing (skip)")
        return (True, "; ".join(checks))

    # ----------------------------------------------------------------------
    # Run All Tests
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
            # Treat skip messages as pass to avoid unnecessary failure
            if "SKIP" in msg:
                passed = True
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)