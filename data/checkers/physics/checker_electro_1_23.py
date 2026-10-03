import sys
import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for the dipole layer and spherical surface charge problem.
    """
    def __init__(self, solution_instance):
        # Extract givens safely – use None for missing values
        self.taugiven = getattr(solution_instance, 'taugiven', None)  # part (a) input
        self.q = getattr(solution_instance, 'q', None)
        self.a = getattr(solution_instance, 'a', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Extract result dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            res = solution_instance.results
            self.Delta_phi = res.get('Delta_phi', None)
            self.sigma = res.get('sigma', None)
            self.tau_result = res.get('tau', None)   # scalar (with sign) from part (b)
        else:
            self.Delta_phi = self.sigma = self.tau_result = None

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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all three required result keys are present."""
        missing = []
        if self.Delta_phi is None:
            missing.append('Delta_phi')
        if self.sigma is None:
            missing.append('sigma')
        if self.tau_result is None:
            missing.append('tau')
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_givens_available(self):
        """Check that the essential given parameters are present (q, a, epsilon0)."""
        missing = []
        if self.q is None:
            missing.append('q')
        if self.a is None:
            missing.append('a')
        if self.epsilon0 is None:
            missing.append('epsilon0')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "Essential givens (q, a, epsilon0) available")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_plane_dipole_jump(self):
        """
        For part (a): Δφ = τ / ε0 for an infinite plane dipole layer.
        This test requires taugiven (given τ for the plane) and Delta_phi.
        """
        if self.taugiven is None or self.Delta_phi is None or self.epsilon0 is None:
            return (False, "SKIP: taugiven, Delta_phi, or epsilon0 missing")
        expected_jump = self.taugiven / self.epsilon0
        if math.isclose(self.Delta_phi, expected_jump, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Δφ = {self.Delta_phi} ≈ τ/ε0 = {expected_jump}")
        else:
            return (False, f"Δφ = {self.Delta_phi} ≠ τ/ε0 = {expected_jump}")

    def test_invariant_sphere_surface_charge(self):
        """
        For part (b): σ = -q / (4π a^2).
        """
        if any(x is None for x in (self.q, self.a, self.sigma)):
            return (False, "SKIP: q, a, or sigma missing")
        expected_sigma = -self.q / (4.0 * math.pi * self.a**2)
        if math.isclose(self.sigma, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"σ = {self.sigma} ≈ -q/(4πa²) = {expected_sigma}")
        else:
            return (False, f"σ = {self.sigma} ≠ -q/(4πa²) = {expected_sigma}")

    def test_invariant_sphere_dipole_moment(self):
        """
        For part (b): τ (result) = -q / (4π a)  (scalar, direction e_r).
        """
        if any(x is None for x in (self.q, self.a, self.tau_result)):
            return (False, "SKIP: q, a, or tau_result missing")
        expected_tau = -self.q / (4.0 * math.pi * self.a)
        if math.isclose(self.tau_result, expected_tau, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"τ = {self.tau_result} ≈ -q/(4πa) = {expected_tau}")
        else:
            return (False, f"τ = {self.tau_result} ≠ -q/(4πa) = {expected_tau}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_derivative_jump_condition(self):
        """
        Check the normal‑derivative jump condition for the spherical surface:
        dφ_out/dr - dφ_in/dr = -σ/ε0   at r=a.
        Inside potential: φ_in = q/(4πε0 r), so dφ_in/dr = -q/(4πε0 r^2)
        Outside potential: φ_out = 0, so dφ_out/dr = 0.
        This implies σ = -q/(4π a^2).
        We verify that the result sigma satisfies this relation within tolerance.
        """
        if any(x is None for x in (self.q, self.a, self.epsilon0, self.sigma)):
            return (False, "SKIP: q, a, epsilon0, or sigma missing")
        # derivative of φ_in at r=a
        dphi_in_dr = -self.q / (4.0 * math.pi * self.epsilon0 * self.a**2)
        # left side: 0 - dphi_in_dr = q/(4π ε0 a^2)
        left_side = -dphi_in_dr   # = q/(4π ε0 a^2)
        # right side: -σ/ε0
        right_side = -self.sigma / self.epsilon0
        if math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derivative jump condition holds: {left_side} ≈ {right_side}")
        else:
            return (False, f"Derivative jump condition violated: {left_side} ≠ {right_side}")

    def test_modeling_potential_jump_condition(self):
        """
        Check the potential jump condition for the spherical dipole layer:
        φ_out(a) - φ_in(a) = τ/ε0.
        Inside potential: φ_in(a) = q/(4πε0 a)
        Outside potential: φ_out(a) = 0
        Therefore τ/ε0 = -q/(4πε0 a) => τ = -q/(4π a).
        Verify that the result tau_result satisfies this relation.
        """
        if any(x is None for x in (self.q, self.a, self.epsilon0, self.tau_result)):
            return (False, "SKIP: q, a, epsilon0, or tau_result missing")
        # Compute inside potential at r=a
        phi_in_a = self.q / (4.0 * math.pi * self.epsilon0 * self.a)
        # Outside potential is 0
        phi_out_a = 0.0
        # Expected discontinuity from given tau_result
        expected_jump = self.tau_result / self.epsilon0
        actual_jump = phi_out_a - phi_in_a
        if math.isclose(actual_jump, expected_jump, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Potential jump condition holds: Δφ = {actual_jump} ≈ τ/ε0 = {expected_jump}")
        else:
            return (False, f"Potential jump condition violated: Δφ = {actual_jump} ≠ {expected_jump}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation and sensitivity check)
    # ------------------------------------------------------------------
    def test_monotonicity_plane_dipole_jump(self):
        """
        Perturb taugiven by +5% and verify that Delta_phi changes by the same relative amount
        (since Δφ = τ/ε0 is linear in τ).
        """
        if self.taugiven is None or self.Delta_phi is None or self.epsilon0 is None:
            return (False, "SKIP: taugiven, Delta_phi, or epsilon0 missing")
        # baseline
        baseline_tau = self.taugiven
        baseline_phi = self.Delta_phi
        # perturb taugiven by +5%
        factor = 1.05
        perturbed_tau = baseline_tau * factor
        # expected perturbed phi from physics
        expected_phi = perturbed_tau / self.epsilon0
        # relative change in output (from theoretical perspective, but we only have one data point)
        # Instead, we check that the relationship is consistent with linear sensitivity:
        # If we take the baseline result as correct, then perturbed phi should be baseline_phi * factor.
        # Because Δφ ∝ τ.
        expected_phi_linear = baseline_phi * factor
        if math.isclose(expected_phi, expected_phi_linear, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Monotonicity: Δφ scales linearly with τ (factor {factor})")
        else:
            return (False, f"Monotonicity failed: expected perturbed φ = {expected_phi_linear}, "
                           f"but physics gives {expected_phi}")

    def test_monotonicity_sphere_surface_charge_with_q(self):
        """
        Perturb q by +5% and check that sigma changes by the same relative amount
        (since σ = -q/(4πa^2) is linear in q).
        """
        if any(x is None for x in (self.q, self.a, self.sigma)):
            return (False, "SKIP: q, a, or sigma missing")
        baseline_q = self.q
        baseline_sigma = self.sigma
        factor = 1.05
        perturbed_q = baseline_q * factor
        # Expected sigma from physics
        expected_sigma = -perturbed_q / (4.0 * math.pi * self.a**2)
        # Since sigma is linear in q, baseline_sigma * factor should match
        expected_sigma_linear = baseline_sigma * factor
        if math.isclose(expected_sigma, expected_sigma_linear, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Monotonicity: σ scales linearly with q (factor {factor})")
        else:
            return (False, f"Monotonicity failed: expected perturbed σ = {expected_sigma_linear}, "
                           f"but physics gives {expected_sigma}")

    def test_monotonicity_sphere_dipole_moment_with_a(self):
        """
        Perturb a by +5% and check that τ (result) changes by the same relative amount (with negative sign
        since τ ∝ 1/a). Actually τ = -q/(4πa), so if a increases by 5%, τ decreases by ~5%.
        We check that the ratio of changes is consistent.
        """
        if any(x is None for x in (self.q, self.a, self.tau_result)):
            return (False, "SKIP: q, a, or tau_result missing")
        baseline_a = self.a
        baseline_tau = self.tau_result
        factor = 1.05
        perturbed_a = baseline_a * factor
        # Expected tau from physics
        expected_tau = -self.q / (4.0 * math.pi * perturbed_a)
        # Since τ ∝ 1/a, baseline_tau should be multiplied by 1/factor
        expected_tau_linear = baseline_tau / factor
        if math.isclose(expected_tau, expected_tau_linear, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Monotonicity: τ scales inversely with a (factor {factor})")
        else:
            return (False, f"Monotonicity failed: expected perturbed τ = {expected_tau_linear}, "
                           f"but physics gives {expected_tau}")

    # ------------------------------------------------------------------
    # Weak final‑answer sanity (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs_and_magnitudes(self):
        """
        Weak sanity: check that signs of σ and τ_result are consistent with
        physical expectations and that values are finite / not NaN.
        (Do not compute expected values here, only check sign and finiteness.)
        """
        failures = []
        # Check Δφ sign consistency with taugiven if available
        if self.Delta_phi is not None:
            if not math.isfinite(self.Delta_phi):
                failures.append("Delta_phi is not finite")
            if self.taugiven is not None:
                # Δφ and taugiven should have same sign (since ε0 > 0)
                if (self.Delta_phi > 0) != (self.taugiven > 0):
                    failures.append("Sign of Delta_phi does not match sign of taugiven")
        # Check σ: should be negative (opposite sign to q)
        if self.sigma is not None:
            if not math.isfinite(self.sigma):
                failures.append("sigma is not finite")
            if self.q is not None:
                if self.q > 0 and self.sigma >= 0:
                    failures.append("sigma should be negative when q>0")
                elif self.q < 0 and self.sigma <= 0:
                    failures.append("sigma should be positive when q<0")
        # Check τ_result: should be negative when q>0 (points inward)
        if self.tau_result is not None:
            if not math.isfinite(self.tau_result):
                failures.append("tau_result is not finite")
            if self.q is not None:
                if self.q > 0 and self.tau_result >= 0:
                    failures.append("tau_result should be negative when q>0")
                elif self.q < 0 and self.tau_result <= 0:
                    failures.append("tau_result should be positive when q<0")
        if failures:
            return (False, " | ".join(failures))
        return (True, "All answer sanity checks passed")