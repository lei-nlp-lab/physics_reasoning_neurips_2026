import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        self.v = getattr(solution_instance, 'v', None)
        self.phi = getattr(solution_instance, 'phi', None)
        self.u = getattr(solution_instance, 'u', None)
        self.c = getattr(solution_instance, 'c', None)

        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.vp = self.results.get('vp', None)
        self.tan_phip = self.results.get('tan_phip', None)

        # Derived gamma (only if safe)
        self.gamma = None
        if self.u is not None and self.c is not None and self.c != 0:
            beta = self.u / self.c
            if beta < 1:
                self.gamma = 1.0 / math.sqrt(1 - beta * beta)

    # --------------------------------------------------------------
    # Helper: compute v' from formula (re-derive inside checker)
    # --------------------------------------------------------------
    def _compute_vp(self, v, phi, u, c):
        """Compute v' magnitude using Lorentz transformation."""
        denom = 1 - (u * v * math.cos(phi)) / (c * c)
        if denom == 0:
            return None
        numerator = (v * v + u * u - 2 * v * u * math.cos(phi) -
                     (u * u * v * v * math.sin(phi) ** 2) / (c * c))
        if numerator < 0:
            return None
        return math.sqrt(numerator) / abs(denom)

    # --------------------------------------------------------------
    # Helper: compute tan(phi') from formula (re-derive inside checker)
    # --------------------------------------------------------------
    def _compute_tan_phip(self, v, phi, u, c):
        gamma = 1.0 / math.sqrt(1 - (u / c) ** 2) if u < c else None
        if gamma is None:
            return None
        denom = v * math.cos(phi) - u
        if denom == 0:
            return None
        return v * math.sin(phi) / (gamma * denom)

    # --------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.vp is None or self.tan_phip is None:
            return (False, "SKIP: vp or tan_phip not present in results")
        return (True, "vp and tan_phip keys present")

    def test_interface_required_givens(self):
        missing = []
        if self.v is None: missing.append('v')
        if self.phi is None: missing.append('phi')
        if self.u is None: missing.append('u')
        if self.c is None: missing.append('c')
        if missing:
            return (False, f"SKIP: missing givens: {missing}")
        return (True, "All required givens present")

    # --------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # --------------------------------------------------------------
    def test_invariant_home_frame_uzero(self):
        """If u=0, then v' = v and tan_phi' = tan(phi)."""
        if None in (self.v, self.phi, self.u, self.c, self.vp, self.tan_phip):
            return (False, "SKIP: missing input/output")
        if not math.isclose(self.u, 0.0, abs_tol=1e-12):
            return (True, "u != 0, test not applicable")
        if not math.isclose(self.vp, self.v, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"vp ({self.vp}) != v ({self.v}) when u=0")
        expected_tan = math.tan(self.phi)
        if not math.isclose(self.tan_phip, expected_tan, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tan_phi' ({self.tan_phip}) != tan(phi) ({expected_tan}) when u=0")
        return (True, "Home frame test passed")

    def test_invariant_home_frame_vzero(self):
        """If v=0, then v' = u and tan_phi' = 0."""
        if None in (self.v, self.phi, self.u, self.c, self.vp, self.tan_phip):
            return (False, "SKIP: missing input/output")
        if not math.isclose(self.v, 0.0, abs_tol=1e-12):
            return (True, "v != 0, test not applicable")
        if not math.isclose(self.vp, self.u, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"vp ({self.vp}) != u ({self.u}) when v=0")
        if not math.isclose(self.tan_phip, 0.0, abs_tol=1e-2):
            return (False, f"tan_phi' ({self.tan_phip}) != 0 when v=0")
        return (True, "v=0 test passed")

    def test_invariant_light_speed_limit(self):
        """If v is near c, then v' should be near c (c invariance)."""
        if None in (self.v, self.c, self.vp):
            return (False, "SKIP: missing input/output")
        if self.c == 0:
            return (False, "SKIP: c=0")
        if self.v > 0.99 * self.c:
            if not math.isclose(self.vp, self.c, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"vp ({self.vp}) != c ({self.c}) when v near c")
            return (True, "Light speed invariance holds")
        return (True, "v not sufficiently close to c, test skipped")

    def test_invariant_lorentz_vp(self):
        """Check that vp matches the re-derived Lorentz formula."""
        if None in (self.v, self.phi, self.u, self.c, self.vp):
            return (False, "SKIP: missing input/output")
        expected = self._compute_vp(self.v, self.phi, self.u, self.c)
        if expected is None:
            return (False, "SKIP: could not compute expected vp")
        if not math.isclose(self.vp, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"vp ({self.vp}) != expected ({expected})")
        return (True, "vp Lorentz consistency passed")

    def test_invariant_lorentz_tan_phip(self):
        """Check that tan_phi' matches the re-derived Lorentz formula."""
        if None in (self.v, self.phi, self.u, self.c, self.tan_phip):
            return (False, "SKIP: missing input/output")
        expected = self._compute_tan_phip(self.v, self.phi, self.u, self.c)
        if expected is None:
            return (False, "SKIP: could not compute expected tan_phi'")
        if not math.isclose(self.tan_phip, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tan_phi' ({self.tan_phip}) != expected ({expected})")
        return (True, "tan_phi' Lorentz consistency passed")

    # --------------------------------------------------------------
    # Modeling / trace tests (check specific physics relations)
    # --------------------------------------------------------------
    def test_modeling_aberration_formula(self):
        """Check tan_phi' = (v sin phi * sqrt(1-u^2/c^2)) / (v cos phi - u)."""
        if None in (self.v, self.phi, self.u, self.c, self.tan_phip):
            return (False, "SKIP: missing input/output")
        if self.gamma is None:
            return (False, "SKIP: gamma not computable")
        numerator = self.v * math.sin(self.phi) * math.sqrt(1 - (self.u / self.c) ** 2)
        denom = self.v * math.cos(self.phi) - self.u
        if denom == 0:
            return (False, "SKIP: denominator zero (v cos phi = u)")
        expected = numerator / denom
        if not math.isclose(self.tan_phip, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tan_phi' ({self.tan_phip}) != aberration formula ({expected})")
        return (True, "Aberration formula holds")

    def test_modeling_parallel_axis_limit(self):
        """If phi=0, then v' = (v-u)/(1-uv/c^2)."""
        if None in (self.v, self.phi, self.u, self.c, self.vp):
            return (False, "SKIP: missing input/output")
        if not math.isclose(self.phi, 0.0, abs_tol=1e-6):
            return (True, "phi not zero, test skipped")
        denom = 1 - (self.u * self.v) / (self.c * self.c)
        if denom == 0:
            return (False, "SKIP: denominator zero")
        expected = abs(self.v - self.u) / denom
        if not math.isclose(self.vp, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"vp ({self.vp}) != expected ({expected}) for phi=0")
        return (True, "Parallel axis limit holds")

    def test_modeling_orthogonal_limit(self):
        """If phi=90°, then v' = sqrt(v^2/gamma^2 + u^2)."""
        if None in (self.v, self.phi, self.u, self.c, self.vp, self.gamma):
            return (False, "SKIP: missing input/output")
        if not math.isclose(self.phi, math.pi / 2, abs_tol=1e-6):
            return (True, "phi not pi/2, test skipped")
        expected = math.sqrt(self.v * self.v / (self.gamma * self.gamma) + self.u * self.u)
        if not math.isclose(self.vp, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"vp ({self.vp}) != expected ({expected}) for phi=90°")
        return (True, "Orthogonal limit holds")

    # --------------------------------------------------------------
    # Monotonicity test (perturb an input and compare)
    # --------------------------------------------------------------
    def test_monotonicity_vp_increasing_with_v(self):
        """For fixed phi, u, c, vp should increase with v. Perturb v by +10%."""
        if None in (self.v, self.phi, self.u, self.c, self.vp):
            return (False, "SKIP: missing input/output")
        if self.v == 0 or self.c == 0:
            return (False, "SKIP: v or c zero")
        # Baseline vp
        baseline = self.vp
        # Perturb v upward by 10% (capped at c)
        v_pert = self.v * 1.1
        if v_pert > self.c:
            v_pert = self.c
        if v_pert == self.v:
            return (True, "Perturbation too small, test skipped")
        vp_pert = self._compute_vp(v_pert, self.phi, self.u, self.c)
        if vp_pert is None:
            return (False, "SKIP: could not compute perturbed vp")
        # vp should be larger (or equal if saturated)
        if vp_pert < baseline - 1e-9:
            return (False, f"vp decreased after increasing v: baseline {baseline}, perturbed {vp_pert}")
        return (True, f"Monotonicity holds: vp increased from {baseline} to {vp_pert}")

    def test_monotonicity_vp_decreasing_with_u(self):
        """For fixed v, phi, c, vp should decrease with u (since boost subtracts)."""
        if None in (self.v, self.phi, self.u, self.c, self.vp):
            return (False, "SKIP: missing input/output")
        if self.u == 0 or self.c == 0:
            return (False, "SKIP: u or c zero")
        baseline = self.vp
        # Perturb u upward by 10% (must stay below c)
        u_pert = self.u * 1.1
        if u_pert >= self.c:
            u_pert = self.c * 0.999
        if u_pert == self.u:
            return (True, "Perturbation too small, test skipped")
        vp_pert = self._compute_vp(self.v, self.phi, u_pert, self.c)
        if vp_pert is None:
            return (False, "SKIP: could not compute perturbed vp")
        # vp should be smaller (or equal if already at minimum)
        if vp_pert > baseline + 1e-9:
            return (False, f"vp increased after increasing u: baseline {baseline}, perturbed {vp_pert}")
        return (True, f"Monotonicity holds: vp decreased from {baseline} to {vp_pert}")

    # --------------------------------------------------------------
    # Weak answer sanity (max one method, no hardcoded target values)
    # --------------------------------------------------------------
    def test_answer_sanity_finite_and_positive(self):
        """Basic sanity: vp and tan_phi' should be finite and non-negative where applicable."""
        if self.vp is None and self.tan_phip is None:
            return (False, "SKIP: no results to check")
        checks = []
        if self.vp is not None:
            if not math.isfinite(self.vp) or self.vp < 0:
                checks.append(f"vp invalid: {self.vp}")
        if self.tan_phip is not None:
            if not math.isfinite(self.tan_phip):
                checks.append(f"tan_phi' not finite: {self.tan_phip}")
        if checks:
            return (False, "Sanity failed: " + "; ".join(checks))
        return (True, "Basic sanity passed")

    # --------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------
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
                passed, msg = (False, f"{name} CRASHED: {e}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)