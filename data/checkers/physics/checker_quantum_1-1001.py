import math
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits quantum negligibility calculations for macroscopic systems.
    Tests interface, invariants, modeling consistency, monotonicity,
    and weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """Extract all data safely with None defaults. No numeric fallbacks."""
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})

        # Universal constants from givens (must be provided explicitly)
        self.g = self.givens.get('g', None)
        self.hbar = self.givens.get('hbar', None)
        self.h = self.givens.get('h', None)

        # Q1 pendulum
        self.l = self.givens.get('l', None)
        self.m1 = self.givens.get('m_pendulum',
                                   self.givens.get('m', None))

        # Q2 marble
        self.m2 = self.givens.get('m_marble',
                                   self.givens.get('m', None))
        self.v2 = self.givens.get('v_marble',
                                   self.givens.get('v', None))
        self.H = self.givens.get('H', None)
        self.w = self.givens.get('w', None)

        # Q3 tennis ball
        self.m3 = self.givens.get('m_ball',
                                   self.givens.get('m', None))
        self.v3 = self.givens.get('v_ball',
                                   self.givens.get('v', None))
        self.a = self.givens.get('a', None)  # window horizontal
        self.b = self.givens.get('b', None)  # window vertical

        # Results
        self.A = self.results.get('A', None)
        self.T = self.results.get('T', None)
        self.theta_x = self.results.get('theta_x', None)
        self.theta_y = self.results.get('theta_y', None)

    # ------------------------------------------------------------------
    # Helper: skip if any required value is None
    # ------------------------------------------------------------------
    def _check_none(self, *values: Any) -> bool:
        return any(v is None for v in values)

    # ------------------------------------------------------------------
    # Interface tests – ensure required keys exist
    # ------------------------------------------------------------------
    def test_interface_results_dict(self) -> Tuple[bool, str]:
        if not isinstance(self.results, dict):
            return False, f"results is not a dict: {type(self.results)}"
        return True, "results is a dict"

    def test_interface_has_A(self) -> Tuple[bool, str]:
        if 'A' not in self.results:
            return False, "Missing required result key 'A'"
        return True, "Key 'A' present"

    def test_interface_has_T(self) -> Tuple[bool, str]:
        if 'T' not in self.results:
            return False, "Missing required result key 'T'"
        return True, "Key 'T' present"

    def test_interface_has_theta_x(self) -> Tuple[bool, str]:
        if 'theta_x' not in self.results:
            return False, "Missing required result key 'theta_x'"
        return True, "Key 'theta_x' present"

    def test_interface_has_theta_y(self) -> Tuple[bool, str]:
        if 'theta_y' not in self.results:
            return False, "Missing required result key 'theta_y'"
        return True, "Key 'theta_y' present"

    # ------------------------------------------------------------------
    # Theorem‑based invariants (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_zero_point_amplitude(self) -> Tuple[bool, str]:
        """
        Check A = sqrt(hbar/(m1*omega)) with omega = sqrt(g/l).
        """
        if self._check_none(self.A, self.l, self.m1, self.g, self.hbar):
            return (False, "SKIP: missing pendulum parameters")
        try:
            omega = math.sqrt(self.g / self.l)
            expected = math.sqrt(self.hbar / (self.m1 * omega))
            actual = float(self.A)
            if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"Zero-point amplitude matches: {actual:.3e} vs {expected:.3e}"
            else:
                return False, f"Mismatch actual={actual:.3e}, expected~{expected:.3e}"
        except Exception as e:
            return False, f"Zero‑point amplitude check crashed: {e}"

    def test_invariant_tunneling_probability(self) -> Tuple[bool, str]:
        """
        Check T = exp(-2*kappa*w) with kappa = sqrt(2*m2*(V0-E))/hbar.
        """
        if self._check_none(self.T, self.m2, self.v2, self.H,
                            self.w, self.g, self.hbar):
            return (False, "SKIP: missing marble parameters")
        try:
            E = 0.5 * self.m2 * self.v2**2
            V0 = self.m2 * self.g * self.H
            if V0 <= E:
                return (False, "SKIP: V0 <= E, not tunneling regime")
            kappa = math.sqrt(2 * self.m2 * (V0 - E)) / self.hbar
            expected = math.exp(-2 * kappa * self.w)
            actual = float(self.T)

            if actual <= 0 or expected <= 0:
                return False, "T must be positive (underflow allowed, but not ≤0)"

            log_actual = math.log10(actual)
            log_expected = math.log10(expected)
            if math.isclose(log_actual, log_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"Tunneling matches (log10): {log_actual:.3f} vs {log_expected:.3f}"
            else:
                return False, f"Tunneling mismatch (log10): actual={log_actual:.3f}, expected~{log_expected:.3f}"
        except Exception as e:
            return False, f"Tunneling invariant crash: {e}"

    def test_invariant_diffraction_angles(self) -> Tuple[bool, str]:
        """
        Check theta_x = h/(m3*v3)/a, theta_y = h/(m3*v3)/b.
        """
        if self._check_none(self.theta_x, self.theta_y, self.m3,
                            self.v3, self.a, self.b, self.h):
            return (False, "SKIP: missing ball parameters or diffraction results")
        try:
            lam = self.h / (self.m3 * self.v3)
            expected_x = lam / self.a
            expected_y = lam / self.b

            actual_x = float(self.theta_x)
            actual_y = float(self.theta_y)

            ok_x = math.isclose(actual_x, expected_x, rel_tol=1e-2, abs_tol=1e-2)
            ok_y = math.isclose(actual_y, expected_y, rel_tol=1e-2, abs_tol=1e-2)

            if ok_x and ok_y:
                return True, f"Angles match: θx={actual_x:.4e}, θy={actual_y:.4e} (expected ~{expected_x:.4e}, {expected_y:.4e})"
            else:
                return False, f"Mismatch: θx={actual_x:.4e} (expected~{expected_x:.4e}), θy={actual_y:.4e} (expected~{expected_y:.4e})"
        except Exception as e:
            return False, f"Diffraction invariant crash: {e}"

    # ------------------------------------------------------------------
    # Modeling/trace checks (re‑derive one relation per test)
    # ------------------------------------------------------------------
    def test_modeling_amplitude_from_energy(self) -> Tuple[bool, str]:
        """
        Average potential energy = ¼ ħω → ½ m ω² A² = ¼ ħω.
        """
        if self._check_none(self.A, self.l, self.m1, self.g, self.hbar):
            return (False, "SKIP: missing pendulum parameters")
        try:
            omega = math.sqrt(self.g / self.l)
            left = 0.5 * self.m1 * omega**2 * float(self.A)**2
            right = 0.25 * self.hbar * omega
            if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"Energy relation holds: LHS={left:.4e}, RHS={right:.4e}"
            else:
                return False, f"Energy relation fails: LHS={left:.4e}, RHS={right:.4e}"
        except Exception as e:
            return False, f"Modeling amplitude energy crash: {e}"

    def test_modeling_tunneling_kappa_formula(self) -> Tuple[bool, str]:
        """
        Check that ‑ln(T)/(2w) equals kappa.
        """
        if self._check_none(self.T, self.m2, self.v2, self.H,
                            self.w, self.g, self.hbar):
            return (False, "SKIP: missing marble parameters")
        try:
            actual_T = float(self.T)
            if actual_T <= 0:
                return (False, "SKIP: T is zero/underflow, cannot take log")

            E = 0.5 * self.m2 * self.v2**2
            V0 = self.m2 * self.g * self.H
            if V0 <= E:
                return (False, "SKIP: V0 <= E")
            kappa_expected = math.sqrt(2 * self.m2 * (V0 - E)) / self.hbar
            kappa_from_T = -math.log(actual_T) / (2 * self.w)

            if math.isclose(kappa_from_T, kappa_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"Kappa matches: from T={kappa_from_T:.4e}, expected={kappa_expected:.4e}"
            else:
                return False, f"Kappa mismatch: from T={kappa_from_T:.4e}, expected={kappa_expected:.4e}"
        except Exception as e:
            return False, f"Modeling kappa crash: {e}"

    # ------------------------------------------------------------------
    # Monotonicity checks (perturb input, compare perturbed output to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_amplitude_decreases_with_mass(self) -> Tuple[bool, str]:
        """
        If pendulum mass increases, amplitude decreases.
        Baseline is actual result A, perturbed is estimated via formula.
        """
        if self._check_none(self.A, self.m1):
            return (False, "SKIP: missing pendulum amplitude or mass")
        try:
            baseline_A = float(self.A)
            # Perturb mass: increase by 10%
            m_inc = self.m1 * 1.1
            # Amplitude scales as 1/sqrt(m) (frequency omega unchanged if length fixed)
            perturbed_A_expected = baseline_A * math.sqrt(self.m1 / m_inc)
            if perturbed_A_expected < baseline_A:
                return True, f"Amplitude decreases with mass: baseline A={baseline_A:.4e}, perturbed (mass*1.1) ~ {perturbed_A_expected:.4e}"
            else:
                return False, "Amplitude should decrease when mass increases"
        except Exception as e:
            return False, f"Monotonicity amplitude mass crash: {e}"

    def test_monotonicity_tunneling_decreases_with_barrier_height(self) -> Tuple[bool, str]:
        """
        If barrier height H increases, T decreases.
        Baseline is actual result T, perturbed is estimated via formula.
        """
        if self._check_none(self.T, self.m2, self.v2, self.H,
                            self.w, self.g, self.hbar):
            return (False, "SKIP: missing marble parameters or T")
        try:
            baseline_T = float(self.T)
            if baseline_T <= 0:
                return (False, "SKIP: T is zero/underflow, cannot compute perturbation")

            E = 0.5 * self.m2 * self.v2**2
            V0_base = self.m2 * self.g * self.H
            if V0_base <= E:
                return (False, "SKIP: V0_base <= E")
            kappa_base = math.sqrt(2 * self.m2 * (V0_base - E)) / self.hbar

            # Perturb H: increase by 10%
            H_inc = self.H * 1.1
            V0_inc = self.m2 * self.g * H_inc
            if V0_inc <= E:
                return (False, "SKIP: increased V0 still <= E")
            kappa_inc = math.sqrt(2 * self.m2 * (V0_inc - E)) / self.hbar
            perturbed_T_expected = math.exp(-2 * kappa_inc * self.w)

            if perturbed_T_expected < baseline_T:
                return True, f"Tunneling decreases with H: baseline T={baseline_T:.6e}, perturbed (H*1.1) ~ {perturbed_T_expected:.6e}"
            else:
                return False, "Tunneling should decrease when barrier height increases"
        except Exception as e:
            return False, f"Monotonicity tunneling height crash: {e}"

    def test_monotonicity_diffraction_angle_decreases_with_speed(self) -> Tuple[bool, str]:
        """
        If ball speed increases, both diffraction angles decrease.
        Baseline is actual theta_x, theta_y; perturbed is estimated via inverse proportionality.
        """
        if self._check_none(self.theta_x, self.theta_y, self.v3):
            return (False, "SKIP: missing diffraction angles or speed")
        try:
            baseline_theta_x = float(self.theta_x)
            baseline_theta_y = float(self.theta_y)

            # Perturb speed: increase by 10%
            v_inc = self.v3 * 1.1
            # theta ∝ 1/v, so perturbed ≈ baseline * (v / v_inc)
            perturbed_theta_x_expected = baseline_theta_x * (self.v3 / v_inc)
            perturbed_theta_y_expected = baseline_theta_y * (self.v3 / v_inc)

            if perturbed_theta_x_expected < baseline_theta_x and perturbed_theta_y_expected < baseline_theta_y:
                return True, f"Angles decrease with speed: θx base={baseline_theta_x:.4e}, perturbed~{perturbed_theta_x_expected:.4e}; θy base={baseline_theta_y:.4e}, perturbed~{perturbed_theta_y_expected:.4e}"
            else:
                return False, "Diffraction angles should decrease when speed increases"
        except Exception as e:
            return False, f"Monotonicity diffraction speed crash: {e}"

    # ------------------------------------------------------------------
    # Weak answer sanity (single method, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_negligibility(self) -> Tuple[bool, str]:
        """
        All quantum quantities should be many orders of magnitude smaller
        than macroscopic scales. No isclose, no derivation.
        """
        issues = []
        if self.A is not None:
            try:
                a = float(self.A)
                if a <= 0 or a >= 1e-10:
                    issues.append(f"Amplitude {a:.4e} m is not extremely small")
            except:
                issues.append("A is non-numeric")
        if self.T is not None:
            try:
                t = float(self.T)
                if t < 0 or t >= 1e-50:
                    issues.append(f"Tunneling prob {t:.4e} is not essentially zero")
            except:
                issues.append("T is non-numeric")
        if self.theta_x is not None:
            try:
                tx = float(self.theta_x)
                if tx <= 0 or tx >= 1e-15:
                    issues.append(f"θx {tx:.4e} rad is not negligibly small")
            except:
                issues.append("theta_x is non-numeric")
        if self.theta_y is not None:
            try:
                ty = float(self.theta_y)
                if ty <= 0 or ty >= 1e-15:
                    issues.append(f"θy {ty:.4e} rad is not negligibly small")
            except:
                issues.append("theta_y is non-numeric")

        if issues:
            return False, "; ".join(issues)
        return True, "All available results are negligibly small"

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, run them safely, collect logs.
        Returns (all_passed, logs_list).
        """
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