import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution_instance for the rotating polarized cylinder problem.
    Checks: bound charge densities, currents, and magnetic field on axis.
    """

    def __init__(self, solution_instance):
        # --- Extract givens (try common formats) ---
        self.a = None
        if hasattr(solution_instance, 'a') and solution_instance.a is not None:
            self.a = solution_instance.a
        elif hasattr(solution_instance, 'givens') and 'a' in solution_instance.givens:
            self.a = solution_instance.givens['a']
        elif hasattr(solution_instance, 'params') and 'a' in solution_instance.params:
            self.a = solution_instance.params['a']

        self.P0 = None
        if hasattr(solution_instance, 'P0') and solution_instance.P0 is not None:
            self.P0 = solution_instance.P0
        elif hasattr(solution_instance, 'givens') and 'P0' in solution_instance.givens:
            self.P0 = solution_instance.givens['P0']
        elif hasattr(solution_instance, 'params') and 'P0' in solution_instance.params:
            self.P0 = solution_instance.params['P0']

        self.omega = None
        if hasattr(solution_instance, 'omega') and solution_instance.omega is not None:
            self.omega = solution_instance.omega
        elif hasattr(solution_instance, 'givens') and 'omega' in solution_instance.givens:
            self.omega = solution_instance.givens['omega']
        elif hasattr(solution_instance, 'params') and 'omega' in solution_instance.params:
            self.omega = solution_instance.params['omega']

        # --- Extract results (from pseudocode [emit] keys) ---
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.rho_b = solution_instance.results.get('rho_b', None)
            self.sigma_b = solution_instance.results.get('sigma_b', None)
            self.B_axis = solution_instance.results.get('B_axis', None)
        else:
            self.rho_b = getattr(solution_instance, 'rho_b', None)
            self.sigma_b = getattr(solution_instance, 'sigma_b', None)
            self.B_axis = getattr(solution_instance, 'B_axis', None)

    # ========== Interface tests ==========
    def test_interface_required_keys(self):
        """Check that all required result keys are present and not None."""
        missing = []
        for name, val in [('rho_b', self.rho_b), ('sigma_b', self.sigma_b), ('B_axis', self.B_axis)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required keys present")

    def test_interface_numeric_types(self):
        """Check that results are numeric (int/float) or sympy numbers."""
        for name, val in [('rho_b', self.rho_b), ('sigma_b', self.sigma_b), ('B_axis', self.B_axis)]:
            if val is None:
                continue
            if not isinstance(val, (int, float, sp.Number)):
                return (False, f"{name} is not numeric: type {type(val)}")
        return (True, "All result types are numeric")

    # ========== Invariant tests (theorem-based) ==========
    def test_invariant_divergence_rho_b(self):
        """ρ_b should equal -∇·P = -P0 when P = (1/2)P0 r e_r."""
        if None in (self.P0, self.rho_b):
            return (False, "SKIP: missing P0 or rho_b")
        rho_expected = -self.P0
        if math.isclose(self.rho_b, rho_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"rho_b = {self.rho_b} matches -P0")
        else:
            return (False, f"rho_b = {self.rho_b}, expected {rho_expected}")

    def test_invariant_surface_charge_sigma_b(self):
        """σ_b at r=a should be P·n̂ = (1/2)P0 a."""
        if None in (self.P0, self.a, self.sigma_b):
            return (False, "SKIP: missing P0, a, or sigma_b")
        sigma_expected = 0.5 * self.P0 * self.a
        if math.isclose(self.sigma_b, sigma_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_b = {self.sigma_b} matches (1/2) P0 a")
        else:
            return (False, f"sigma_b = {self.sigma_b}, expected {sigma_expected}")

    def test_invariant_net_current_zero(self):
        """Net effective linear current density (∫ J_phi dr + K) should be zero,
        causing B=0 on axis. This check derives the current from givens."""
        if None in (self.a, self.P0, self.omega):
            return (False, "SKIP: missing a, P0, or omega")
        J_integral = -self.P0 * self.omega * (self.a**2) / 2.0
        K = 0.5 * self.P0 * self.omega * (self.a**2)
        net = J_integral + K
        if math.isclose(net, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return (True, "Net effective current is zero")
        else:
            return (False, f"Net current = {net}, expected 0")

    def test_invariant_B_axis_zero(self):
        """Check that the reported B_axis is zero (theoretical result)."""
        if self.B_axis is None:
            return (False, "SKIP: B_axis not available")
        if not isinstance(self.B_axis, (int, float, sp.Number)):
            return (False, "B_axis is not numeric")
        b = float(self.B_axis)
        if math.isclose(b, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "B_axis = 0 (within tolerance)")
        else:
            return (False, f"B_axis = {b}, expected 0")

    # ========== Modeling / trace tests ==========
    def test_modeling_charge_neutrality(self):
        """Total bound charge per unit length (volume + surface) should be zero."""
        if None in (self.P0, self.a, self.rho_b, self.sigma_b):
            return (False, "SKIP: missing required data")
        vol = self.rho_b * math.pi * self.a**2
        surf = self.sigma_b * 2 * math.pi * self.a
        total = vol + surf
        if math.isclose(total, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Total bound charge per unit length is zero (neutral cylinder)")
        else:
            return (False, f"Total charge per length = {total}, expected 0")

    def test_modeling_sigma_rho_ratio(self):
        """Check the ratio sigma_b / rho_b equals -a/2 (from P field)."""
        if None in (self.sigma_b, self.rho_b, self.a):
            return (False, "SKIP: missing sigma_b, rho_b, or a")
        expected_ratio = -self.a / 2.0
        actual_ratio = self.sigma_b / self.rho_b
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sigma_b/rho_b = {actual_ratio} matches -a/2")
        else:
            return (False, f"sigma_b/rho_b = {actual_ratio}, expected {expected_ratio}")

    # ========== Monotonicity test ==========
    def test_monotonicity_surface_current_increases_with_omega(self):
        """Perturb omega slightly; the surface current K = sigma_b * omega * a should increase."""
        if None in (self.sigma_b, self.omega, self.a):
            return (False, "SKIP: missing sigma_b, omega, or a")
        K0 = self.sigma_b * self.omega * self.a
        epsilon = 0.01  # 1% increase
        omega1 = self.omega * (1 + epsilon)
        K1 = self.sigma_b * omega1 * self.a
        if K1 > K0:
            return (True, f"K increases with omega: {K1} > {K0}")
        else:
            return (False, f"K not monotonic: {K1} <= {K0}")

    # ========== Weak answer sanity ==========
    def test_answer_sanity_finite_and_sign(self):
        """B_axis should be a finite number and near zero."""
        if self.B_axis is None:
            return (False, "SKIP: B_axis not available")
        b = float(self.B_axis)
        if math.isfinite(b):
            if abs(b) < 1e-6:
                return (True, "B_axis is finite and near zero")
            else:
                return (False, f"B_axis = {b} not near zero")
        else:
            return (False, "B_axis is not finite")

    # ========== Test runner ==========
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