import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Givens
        self.G = getattr(solution_instance, 'G', None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.R = getattr(solution_instance, 'R', None)
        self.omega = getattr(solution_instance, 'omega', None)

        # Results (supports both direct attributes and results dict)
        results = getattr(solution_instance, 'results', None)
        if results is not None:
            self.T1 = results.get('T1', None)
            self.T2 = results.get('T2', None)
            self.necessity = results.get('necessity', None)
        else:
            self.T1 = getattr(solution_instance, 'T1', None)
            self.T2 = getattr(solution_instance, 'T2', None)
            self.necessity = getattr(solution_instance, 'necessity', None)

        self.pi = math.pi

    # ---------- Interface tests ----------
    def test_interface_T1_exists(self):
        if self.T1 is None:
            return (False, "T1 is missing")
        return (True, "T1 present")

    def test_interface_T2_exists(self):
        if self.T2 is None:
            return (False, "T2 is missing")
        return (True, "T2 present")

    def test_interface_necessity_exists(self):
        if self.necessity is None:
            return (False, "necessity is missing")
        return (True, "necessity present")

    # ---------- Invariant tests ----------
    def test_invariant_positive_periods(self):
        if self.T1 is not None and self.T1 <= 0:
            return (False, f"T1={self.T1} must be positive")
        if self.T2 is not None and self.T2 <= 0:
            return (False, f"T2={self.T2} must be positive")
        return (True, "All periods positive")

    def test_invariant_T1_less_than_T2(self):
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: missing T1 or T2")
        if self.T1 >= self.T2:
            return (False, f"T1={self.T1} must be less than T2={self.T2} for nonzero rotation")
        return (True, "T1 < T2")

    def test_invariant_T1_from_density(self):
        if self.G is None or self.rho is None:
            return (False, "SKIP: missing G or rho")
        if self.T1 is None:
            return (False, "SKIP: missing T1")
        omega0 = math.sqrt(4 * self.pi * self.G * self.rho / 3)
        expected_T1 = 2 * self.pi / omega0
        if math.isclose(self.T1, expected_T1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T1={self.T1} matches expected {expected_T1}")
        return (False, f"T1={self.T1} differs from expected {expected_T1}")

    def test_invariant_T2_from_density(self):
        if self.G is None or self.rho is None or self.omega is None:
            return (False, "SKIP: missing G, rho, or omega")
        if self.T2 is None:
            return (False, "SKIP: missing T2")
        omega0_sq = 4 * self.pi * self.G * self.rho / 3
        omega_eff_sq = omega0_sq - self.omega**2
        if omega_eff_sq <= 0:
            return (False, "omega0^2 must be > omega^2")
        expected_T2 = 2 * self.pi / math.sqrt(omega_eff_sq)
        if math.isclose(self.T2, expected_T2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T2={self.T2} matches expected {expected_T2}")
        return (False, f"T2={self.T2} differs from expected {expected_T2}")

    def test_invariant_necessity_correct(self):
        expected = "Only Tunnel 2 must be frictionless"
        if self.necessity is None:
            return (False, "SKIP: missing necessity")
        if self.necessity == expected:
            return (True, "necessity matches expected")
        return (False, f"necessity='{self.necessity}' differs from expected '{expected}'")

    # ---------- Trace / modeling tests ----------
    def test_trace_T1_via_g0_R(self):
        """
        Re-derive T1 using intermediate steps: g0 = GM/R^2, omega0 = sqrt(g0/R).
        This checks that the modeling idea of SHM with gravitational restoring force
        is correctly translated into the period.
        """
        if self.G is None or self.rho is None or self.R is None or self.T1 is None:
            return (False, "SKIP: missing G, rho, R, or T1")
        M = (4/3) * self.pi * self.rho * self.R**3
        g0 = self.G * M / (self.R**2)          # surface gravity
        omega0 = math.sqrt(g0 / self.R)         # SHM angular frequency
        expected_T1 = 2 * self.pi / omega0
        if math.isclose(self.T1, expected_T1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T1 derivation via g0/R gives {expected_T1}, matches {self.T1}")
        return (False, f"T1 derivation via g0/R gives {expected_T1}, but got {self.T1}")

    def test_trace_T2_via_g0_R(self):
        """
        Check T2 derivation: effective omega = sqrt(g0/R - omega^2).
        """
        if self.G is None or self.rho is None or self.R is None or self.omega is None or self.T2 is None:
            return (False, "SKIP: missing data for T2 derivation")
        M = (4/3) * self.pi * self.rho * self.R**3
        g0 = self.G * M / (self.R**2)
        omega0_sq = g0 / self.R
        omega_eff_sq = omega0_sq - self.omega**2
        if omega_eff_sq <= 0:
            return (False, "omega_eff^2 <= 0; cannot derive T2")
        expected_T2 = 2 * self.pi / math.sqrt(omega_eff_sq)
        if math.isclose(self.T2, expected_T2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T2 derivation gives {expected_T2}, matches {self.T2}")
        return (False, f"T2 derivation gives {expected_T2}, but got {self.T2}")

    # ---------- Monotonicity test ----------
    def test_monotonicity_T2_increases_with_omega(self):
        """
        Perturb omega upward by 1% (if omega != 0) and verify that T2 increases.
        """
        if self.G is None or self.rho is None or self.omega is None or self.T2 is None:
            return (False, "SKIP: missing constants for monotonicity")
        omega0_sq = 4 * self.pi * self.G * self.rho / 3
        if self.omega == 0:
            return (False, "SKIP: omega=0, cannot perturb upward meaningfully")
        delta = 0.01 * self.omega  # 1% increase
        new_omega = self.omega + delta
        if new_omega**2 >= omega0_sq:
            return (False, "Perturbed omega exceeds critical value")
        T2_baseline = self.T2
        new_omega_eff = math.sqrt(omega0_sq - new_omega**2)
        T2_perturbed = 2 * self.pi / new_omega_eff
        if T2_perturbed > T2_baseline:
            return (True, f"T2 increases: {T2_baseline} -> {T2_perturbed} for 1% omega increase")
        return (False, f"T2 should increase but {T2_baseline} -> {T2_perturbed}")

    # ---------- Weak answer sanity (at most one) ----------
    def test_answer_sanity_general(self):
        """
        Very basic checks: finite, non-NaN, non-empty string.
        """
        if self.T1 is not None and (math.isnan(self.T1) or math.isinf(self.T1)):
            return (False, "T1 is NaN or inf")
        if self.T2 is not None and (math.isnan(self.T2) or math.isinf(self.T2)):
            return (False, "T2 is NaN or inf")
        if self.necessity is not None and len(self.necessity.strip()) == 0:
            return (False, "necessity is empty string")
        return (True, "Basic sanity checks passed")

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