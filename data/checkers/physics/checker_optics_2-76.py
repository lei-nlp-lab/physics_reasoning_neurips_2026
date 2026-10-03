import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Givens (expected, but may be missing)
        self.lambda_XUV = getattr(solution_instance, 'lambda_XUV', None)
        self.lambda_vis = getattr(solution_instance, 'lambda_vis', None)
        self.W = getattr(solution_instance, 'W', None)
        self.D = getattr(solution_instance, 'D', None)
        self.m = getattr(solution_instance, 'm', None)
        self.D_ap = getattr(solution_instance, 'D_ap', None)

        # Extracted results (pseudocode emit keys)
        results = getattr(solution_instance, 'results', {})
        self.XUV_lens = results.get('XUV_lens', None)
        self.vis_lens = results.get('vis_lens', None)
        self.XUV_mirror = results.get('XUV_mirror', None)
        self.vis_mirror = results.get('vis_mirror', None)
        self.delta_lambda_XUV = results.get('delta_lambda_XUV', None)
        self.delta_lambda_vis = results.get('delta_lambda_vis', None)
        self.ratio = results.get('ratio', None)
        self.theta_XUV = results.get('theta_XUV', None)
        self.theta_vis = results.get('theta_vis', None)

        # Derived intermediate (if givens present)
        self.N = None
        if self.W is not None and self.D is not None:
            self.N = self.W * self.D

    # ------------------------------------------------------------------
    # Interface tests: required keys must exist
    # ------------------------------------------------------------------
    def test_interface_XUV_lens_exists(self):
        if self.XUV_lens is None:
            return (False, "Missing XUV_lens key in results")
        return (True, "XUV_lens present")

    def test_interface_vis_lens_exists(self):
        if self.vis_lens is None:
            return (False, "Missing vis_lens key in results")
        return (True, "vis_lens present")

    def test_interface_XUV_mirror_exists(self):
        if self.XUV_mirror is None:
            return (False, "Missing XUV_mirror key in results")
        return (True, "XUV_mirror present")

    def test_interface_vis_mirror_exists(self):
        if self.vis_mirror is None:
            return (False, "Missing vis_mirror key in results")
        return (True, "vis_mirror present")

    def test_interface_delta_lambda_XUV_exists(self):
        if self.delta_lambda_XUV is None:
            return (False, "Missing delta_lambda_XUV key in results")
        return (True, "delta_lambda_XUV present")

    def test_interface_delta_lambda_vis_exists(self):
        if self.delta_lambda_vis is None:
            return (False, "Missing delta_lambda_vis key in results")
        return (True, "delta_lambda_vis present")

    def test_interface_ratio_exists(self):
        if self.ratio is None:
            return (False, "Missing ratio key in results")
        return (True, "ratio present")

    def test_interface_theta_XUV_exists(self):
        if self.theta_XUV is None:
            return (False, "Missing theta_XUV key in results")
        return (True, "theta_XUV present")

    def test_interface_theta_vis_exists(self):
        if self.theta_vis is None:
            return (False, "Missing theta_vis key in results")
        return (True, "theta_vis present")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical relations
    # ------------------------------------------------------------------
    def test_invariant_delta_lambda_formula(self):
        if any(x is None for x in (self.delta_lambda_XUV, self.lambda_XUV, self.m, self.N)):
            return (True, "SKIP: insufficient data for delta_lambda_XUV numeric check")
        expected = self.lambda_XUV / (self.m * self.N)
        if math.isclose(self.delta_lambda_XUV, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_lambda_XUV matches lambda_XUV/(m*N): {self.delta_lambda_XUV} ≈ {expected}")
        else:
            return (False, f"delta_lambda_XUV {self.delta_lambda_XUV} deviates from expected {expected}")

    def test_invariant_delta_lambda_ratio(self):
        if any(x is None for x in (self.delta_lambda_XUV, self.delta_lambda_vis, self.lambda_XUV, self.lambda_vis)):
            return (True, "SKIP: insufficient data for delta_lambda ratio check")
        ratio_d = self.delta_lambda_XUV / self.delta_lambda_vis
        ratio_l = self.lambda_XUV / self.lambda_vis
        if math.isclose(ratio_d, ratio_l, rel_tol=1e-2):
            return (True, f"delta ratio {ratio_d} ≈ lambda ratio {ratio_l}")
        else:
            return (False, f"delta ratio {ratio_d} != lambda ratio {ratio_l}")

    def test_invariant_theta_formula(self):
        if any(x is None for x in (self.theta_XUV, self.lambda_XUV, self.D_ap)):
            return (True, "SKIP: insufficient data for theta_XUV numeric check")
        expected = 1.22 * self.lambda_XUV / self.D_ap
        if math.isclose(self.theta_XUV, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"theta_XUV ≈ 1.22*lambda_XUV/D_ap: {self.theta_XUV} ≈ {expected}")
        else:
            return (False, f"theta_XUV {self.theta_XUV} deviates from expected {expected}")

    def test_invariant_theta_ratio(self):
        if any(x is None for x in (self.theta_XUV, self.theta_vis, self.lambda_XUV, self.lambda_vis)):
            return (True, "SKIP: insufficient data for theta ratio check")
        ratio_t = self.theta_XUV / self.theta_vis
        ratio_l = self.lambda_XUV / self.lambda_vis
        if math.isclose(ratio_t, ratio_l, rel_tol=1e-2):
            return (True, f"theta ratio {ratio_t} ≈ lambda ratio {ratio_l}")
        else:
            return (False, f"theta ratio {ratio_t} != lambda ratio {ratio_l}")

    def test_invariant_XUV_less_than_vis(self):
        # Check monotonic ordering of XUV vs visible quantities
        if self.delta_lambda_XUV is not None and self.delta_lambda_vis is not None:
            if not self.delta_lambda_XUV < self.delta_lambda_vis:
                return (False, "delta_lambda_XUV should be less than delta_lambda_vis")
        if self.theta_XUV is not None and self.theta_vis is not None:
            if not self.theta_XUV < self.theta_vis:
                return (False, "theta_XUV should be less than theta_vis")
        if all(x is not None for x in (self.lambda_XUV, self.lambda_vis)):
            if not self.lambda_XUV < self.lambda_vis:
                return (False, "lambda_XUV should be less than lambda_vis")
        return (True, "XUV quantities are smaller than visible ones (if data present)")

    # ------------------------------------------------------------------
    # Modeling/trace quality checks
    # ------------------------------------------------------------------
    def test_modeling_qualitative_lens(self):
        if self.XUV_lens is None or self.vis_lens is None:
            return (True, "SKIP: lens strings missing")
        if 'crystal' in self.XUV_lens.lower() and 'refract' in self.vis_lens.lower():
            return (True, "Lens descriptions match expected: XUV uses crystals, visible uses refraction")
        else:
            return (False, f"Lens descriptions not as expected: XUV_lens='{self.XUV_lens}', vis_lens='{self.vis_lens}'")

    def test_modeling_qualitative_mirror(self):
        if self.XUV_mirror is None or self.vis_mirror is None:
            return (True, "SKIP: mirror strings missing")
        if 'grazing' in self.XUV_mirror.lower() and 'normal' in self.vis_mirror.lower():
            return (True, "Mirror descriptions match expected: XUV grazing, visible normal")
        else:
            return (False, f"Mirror descriptions not as expected: XUV_mirror='{self.XUV_mirror}', vis_mirror='{self.vis_mirror}'")

    def test_trace_delta_lambda_sign(self):
        for name, val in [('delta_lambda_XUV', self.delta_lambda_XUV), ('delta_lambda_vis', self.delta_lambda_vis)]:
            if val is None:
                continue
            if val <= 0:
                return (False, f"{name} should be positive, got {val}")
        return (True, "All present delta_lambda values are positive")

    def test_trace_theta_sign(self):
        for name, val in [('theta_XUV', self.theta_XUV), ('theta_vis', self.theta_vis)]:
            if val is None:
                continue
            if val <= 0:
                return (False, f"{name} should be positive, got {val}")
        return (True, "All present theta values are positive")

    def test_trace_ratio_positive(self):
        if self.ratio is None:
            return (True, "SKIP: ratio not provided")
        if self.ratio > 0:
            return (True, "Ratio is positive")
        else:
            return (False, f"Ratio should be positive, got {self.ratio}")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb input and compare baseline vs perturbed output
    # ------------------------------------------------------------------
    def test_monotonicity_theta_increases_with_wavelength(self):
        """Use the two given wavelengths as a natural perturbation: larger λ → larger θ."""
        if any(x is None for x in (self.theta_XUV, self.theta_vis, self.lambda_XUV, self.lambda_vis)):
            return (True, "SKIP: missing data for theta monotonicity")
        if not self.lambda_vis > self.lambda_XUV:
            return (True, "SKIP: lambda_vis not larger than lambda_XUV (unexpected)")
        if self.theta_vis > self.theta_XUV:
            return (True, "Monotonic: theta increases with wavelength")
        else:
            return (False, "theta_vis should be > theta_XUV since lambda_vis > lambda_XUV")

    def test_monotonicity_delta_lambda_increases_with_wavelength(self):
        """Use the two given wavelengths as a natural perturbation: larger λ → larger Δλ."""
        if any(x is None for x in (self.delta_lambda_XUV, self.delta_lambda_vis, self.lambda_XUV, self.lambda_vis)):
            return (True, "SKIP: missing data for delta_lambda monotonicity")
        if not self.lambda_vis > self.lambda_XUV:
            return (True, "SKIP: lambda_vis not larger than lambda_XUV (unexpected)")
        if self.delta_lambda_vis > self.delta_lambda_XUV:
            return (True, "Monotonic: delta_lambda increases with wavelength")
        else:
            return (False, "delta_lambda_vis should be > delta_lambda_XUV since lambda_vis > lambda_XUV")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        # Combined: check finite and order (only one method)
        errors = []
        numerics = [self.delta_lambda_XUV, self.delta_lambda_vis, self.ratio,
                    self.theta_XUV, self.theta_vis]
        for v in numerics:
            if v is None:
                continue
            if not math.isfinite(v):
                errors.append(f"Non-finite value: {v}")
        # Order check (if both present)
        if all(x is not None for x in (self.theta_XUV, self.theta_vis)):
            if not self.theta_XUV < self.theta_vis:
                errors.append("theta_XUV should be smaller than theta_vis")
        if all(x is not None for x in (self.delta_lambda_XUV, self.delta_lambda_vis)):
            if not self.delta_lambda_XUV < self.delta_lambda_vis:
                errors.append("delta_lambda_XUV should be smaller than delta_lambda_vis")
        if errors:
            return (False, "; ".join(errors))
        return (True, "All sanity checks passed")

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)