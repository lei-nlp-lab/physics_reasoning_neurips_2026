import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the pendulum-cut-string projectile problem.
    Checks invariants, trace consistency, and monotonicity trends.
    """

    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Extract inputs with safe defaults
        self.L = self._safe_get(solution_instance, 'L', 0.9)
        self.H = self._safe_get(solution_instance, 'H', 1.7)
        self.g = self._safe_get(solution_instance, 'g', 10.0)

        # Extract results (may be None if missing)
        self.v_D = self._safe_get(solution_instance, 'v_D', None)
        self.t = self._safe_get(solution_instance, 't', None)
        self.h_BC = self._safe_get(solution_instance, 'h_BC', None)
        self.h_AD = self._safe_get(solution_instance, 'h_AD', None)
        self.x_DC = self._safe_get(solution_instance, 'x_DC', None)
        self.v_B = self._safe_get(solution_instance, 'v_B', None)

        # Constants
        self.theta = 60  # degrees
        self.cos_theta = math.cos(math.radians(self.theta))

        # Log storage
        self.logs = []

    def _safe_get(self, obj, key, default):
        """
        Safely retrieve attribute or dictionary entry.
        """
        try:
            # First try direct attribute
            val = getattr(obj, key, None)
            if val is not None:
                return val
            # Then try results dictionary
            if hasattr(obj, 'results') and isinstance(obj.results, dict):
                return obj.results.get(key, default)
            return default
        except Exception:
            return default

    def add_log(self, msg):
        """Add a log message."""
        self.logs.append(msg)

    def test_invariants(self):
        """
        Verify fundamental physics laws: energy conservation and kinematics.
        """
        try:
            # 1. Energy conservation A -> D
            if self.v_D is None or self.h_AD is None:
                self.add_log("Missing v_D or h_AD for energy conservation A->D.")
                return False
            v_D_expected = math.sqrt(2 * self.g * self.h_AD)
            if not math.isclose(self.v_D, v_D_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.add_log(f"Energy A->D failed: v_D={self.v_D}, expected={v_D_expected}")
                return False
            self.add_log("Energy conservation A->D passed.")

            # 2. Energy conservation A -> B
            if self.v_B is None:
                self.add_log("Missing v_B for energy conservation A->B.")
                return False
            h_AB = self.L * (1 - self.cos_theta)
            v_B_expected = math.sqrt(2 * self.g * h_AB)
            if not math.isclose(self.v_B, v_B_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.add_log(f"Energy A->B failed: v_B={self.v_B}, expected={v_B_expected}")
                return False
            self.add_log("Energy conservation A->B passed.")

            # 3. Free-fall kinematics: h_BC = 0.5*g*t^2
            if self.h_BC is None or self.t is None:
                self.add_log("Missing h_BC or t for free-fall check.")
                return False
            h_BC_expected = 0.5 * self.g * self.t**2
            if not math.isclose(self.h_BC, h_BC_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.add_log(f"Free-fall failed: h_BC={self.h_BC}, expected={h_BC_expected}")
                return False
            self.add_log("Free-fall kinematics passed.")

            # 4. Horizontal motion: x_DC = v_B * t
            if self.x_DC is None or self.v_B is None or self.t is None:
                self.add_log("Missing x_DC, v_B, or t for horizontal motion check.")
                return False
            x_DC_expected = self.v_B * self.t
            if not math.isclose(self.x_DC, x_DC_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.add_log(f"Horizontal motion failed: x_DC={self.x_DC}, expected={x_DC_expected}")
                return False
            self.add_log("Horizontal motion passed.")

            # 5. Geometry: h_AD = H - L*cos(theta)
            if self.h_AD is None:
                self.add_log("Missing h_AD for geometry check.")
                return False
            h_AD_expected = self.H - self.L * self.cos_theta
            if not math.isclose(self.h_AD, h_AD_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.add_log(f"Geometry failed: h_AD={self.h_AD}, expected={h_AD_expected}")
                return False
            self.add_log("Geometry for h_AD passed.")

            # 6. Height of B: h_BC = H - L
            if self.h_BC is None:
                self.add_log("Missing h_BC for H-L check.")
                return False
            h_BC_geom = self.H - self.L
            if not math.isclose(self.h_BC, h_BC_geom, rel_tol=1e-2, abs_tol=1e-2):
                self.add_log(f"Height of B inconsistent: h_BC={self.h_BC}, H-L={h_BC_geom}")
                return False
            self.add_log("Height of B above ground consistent.")

            return True
        except Exception as e:
            self.add_log(f"Invariants test crashed: {e}\n{traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """
        Verify that intermediate steps are consistent with each other.
        """
        try:
            success = True

            # Step 1 & 2: v_D from h_AD and geometry
            if self.v_D is not None and self.h_AD is not None:
                v_D_calc = math.sqrt(2 * self.g * self.h_AD)
                if not math.isclose(self.v_D, v_D_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.add_log(f"Step 1 inconsistent: v_D={self.v_D}, calculated={v_D_calc}")
                    success = False
                else:
                    self.add_log("Step 1 (energy A->D) consistent.")
            else:
                self.add_log("Missing data for Step 1.")
                success = False

            # Step 2 geometry (already checked in invariants, but log again)
            if self.h_AD is not None:
                h_AD_calc = self.H - self.L * self.cos_theta
                if not math.isclose(self.h_AD, h_AD_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.add_log(f"Step 2 inconsistent: h_AD={self.h_AD}, calculated={h_AD_calc}")
                    success = False
                else:
                    self.add_log("Step 2 (geometry) consistent.")
            else:
                self.add_log("Missing data for Step 2.")
                success = False

            # Step 4: v_B from A->B energy
            if self.v_B is not None:
                v_B_calc = math.sqrt(2 * self.g * self.L * (1 - self.cos_theta))
                if not math.isclose(self.v_B, v_B_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.add_log(f"Step 4 inconsistent: v_B={self.v_B}, calculated={v_B_calc}")
                    success = False
                else:
                    self.add_log("Step 4 (energy A->B) consistent.")
            else:
                self.add_log("Missing data for Step 4.")
                success = False

            # Step 5 & 6: free-fall time
            if self.h_BC is not None and self.t is not None:
                h_BC_calc = 0.5 * self.g * self.t**2
                if not math.isclose(self.h_BC, h_BC_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.add_log(f"Step 5/6 inconsistent: h_BC={self.h_BC}, calculated={h_BC_calc}")
                    success = False
                else:
                    self.add_log("Step 5/6 (free-fall) consistent.")
            else:
                self.add_log("Missing data for Step 5/6.")
                success = False

            # Step 7: horizontal distance
            if self.x_DC is not None and self.v_B is not None and self.t is not None:
                x_DC_calc = self.v_B * self.t
                if not math.isclose(self.x_DC, x_DC_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.add_log(f"Step 7 inconsistent: x_DC={self.x_DC}, calculated={x_DC_calc}")
                    success = False
                else:
                    self.add_log("Step 7 (horizontal motion) consistent.")
            else:
                self.add_log("Missing data for Step 7.")
                success = False

            return success
        except Exception as e:
            self.add_log(f"Trace consistency test crashed: {e}\n{traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """
        Perturb length L and verify that v_D and x_DC change as expected.
        """
        try:
            # Store original L
            L_orig = self.L
            # Perturb by +1%
            delta = 0.01
            L_new = L_orig * (1 + delta)

            # Compute original values from formulas (independent of stored results)
            h_AD_orig = self.H - L_orig * self.cos_theta
            v_D_orig = math.sqrt(2 * self.g * h_AD_orig)
            x_DC_orig = math.sqrt(2 * L_orig * (self.H - L_orig))

            # Compute perturbed values
            h_AD_new = self.H - L_new * self.cos_theta
            v_D_new = math.sqrt(2 * self.g * h_AD_new)
            x_DC_new = math.sqrt(2 * L_new * (self.H - L_new))

            # Check trends: increasing L should decrease both v_D and x_DC (for given H)
            # Use a small tolerance for the decrease
            if v_D_new - v_D_orig > -1e-5:
                self.add_log(f"Monotonicity v_D failed: L↑, v_D did not decrease. v_D_orig={v_D_orig:.4f}, v_D_new={v_D_new:.4f}")
                return False
            self.add_log("Monotonicity v_D: L↑ → v_D↓ (as expected).")

            if x_DC_new - x_DC_orig > -1e-5:
                self.add_log(f"Monotonicity x_DC failed: L↑, x_DC did not decrease. x_DC_orig={x_DC_orig:.4f}, x_DC_new={x_DC_new:.4f}")
                return False
            self.add_log("Monotonicity x_DC: L↑ → x_DC↓ (as expected).")

            return True
        except Exception as e:
            self.add_log(f"Monotonicity test crashed: {e}\n{traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        """
        self.logs = []
        success = True

        self.add_log("=== Starting Physics Audit ===")

        if not self.test_invariants():
            success = False
            self.add_log("✗ Invariants test failed.")
        else:
            self.add_log("✓ Invariants test passed.")

        if not self.test_trace_consistency():
            success = False
            self.add_log("✗ Trace consistency test failed.")
        else:
            self.add_log("✓ Trace consistency test passed.")

        if not self.test_monotonicity():
            success = False
            self.add_log("✗ Monotonicity test failed.")
        else:
            self.add_log("✓ Monotonicity test passed.")

        self.add_log("=== Audit Complete ===")
        return success, self.logs