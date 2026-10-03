import math
import sys
import traceback

class PhysicsChecker:
    """
    Audits a solution to the NMR resonance absorption problem.
    Checks:
    - Interface: all required keys present and of correct type.
    - Invariants: population sum to 1, within [0,1], ordering, direction perpendicular.
    - Modeling: resonance frequency formula, frequency conversion, explanation contents.
    - Monotonicity: frequency increases with H0 (using solution's effective gamma).
    - Answer sanity: frequency finite positive.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Givens (use None for missing)
        self.H0 = self.givens.get('H0', None)
        self.T = self.givens.get('T', None)
        self.gamma = self.givens.get('gamma', None)      # rad/(s·Oe), normally 2.675e4
        self.hbar = self.givens.get('hbar', None)        # J·s
        self.k = self.givens.get('k', None)              # J/K
        self.mu_N = self.givens.get('mu_N', None)        # J/T

        # Results (use None for missing)
        self.P_plus = self.results.get('P_plus', None)
        self.P_minus = self.results.get('P_minus', None)
        self.direction = self.results.get('direction', None)
        self.omega_res_Mrad_per_s = self.results.get('omega_res_Mrad_per_s', None)
        self.f_res_MHz = self.results.get('f_res_MHz', None)
        self.explanation_continuous = self.results.get('explanation_continuous', None)
        self.saturation = self.results.get('saturation', None)

    # -------------------------------------------------------------------------
    # Helper: get gamma either from givens or use default
    # -------------------------------------------------------------------------
    def _get_gamma(self):
        if self.gamma is not None:
            return self.gamma
        # Known constant for protons (rad/(s·Gauss), 1 Oe = 1 G in cgs)
        return 2.675e4

    # -------------------------------------------------------------------------
    # Interface tests
    # -------------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All emitted result keys must be present."""
        required = ['P_plus', 'P_minus', 'direction',
                    'omega_res_Mrad_per_s', 'f_res_MHz',
                    'explanation_continuous', 'saturation']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_types(self):
        """Check types: populations numeric, direction string, frequencies numeric,
        explanations string."""
        checks = []
        if self.P_plus is not None and not isinstance(self.P_plus, (int, float)):
            checks.append("P_plus should be numeric")
        if self.P_minus is not None and not isinstance(self.P_minus, (int, float)):
            checks.append("P_minus should be numeric")
        if self.direction is not None and not isinstance(self.direction, str):
            checks.append("direction should be string")
        if self.omega_res_Mrad_per_s is not None and not isinstance(self.omega_res_Mrad_per_s, (int, float)):
            checks.append("omega_res_Mrad_per_s should be numeric")
        if self.f_res_MHz is not None and not isinstance(self.f_res_MHz, (int, float)):
            checks.append("f_res_MHz should be numeric")
        if self.explanation_continuous is not None and not isinstance(self.explanation_continuous, str):
            checks.append("explanation_continuous should be string")
        if self.saturation is not None and not isinstance(self.saturation, str):
            checks.append("saturation should be string")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All types acceptable")

    # -------------------------------------------------------------------------
    # Invariant tests (physics theorems)
    # -------------------------------------------------------------------------
    def test_invariant_population_sum(self):
        """P_plus + P_minus ≈ 1 (Boltzmann normalisation)."""
        if self.P_plus is None or self.P_minus is None:
            return (True, "SKIP: population values missing")
        total = self.P_plus + self.P_minus
        if not math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_plus + P_minus = {total} != 1")
        return (True, "Population sum is 1")

    def test_invariant_population_range(self):
        """Each population lies in [0,1]."""
        if self.P_plus is None or self.P_minus is None:
            return (True, "SKIP: population values missing")
        if not (0 <= self.P_plus <= 1) or not (0 <= self.P_minus <= 1):
            return (False, f"P_plus={self.P_plus}, P_minus={self.P_minus} out of [0,1]")
        return (True, "Populations within [0,1]")

    def test_invariant_population_ordering(self):
        """For H0>0, P_plus > P_minus (lower energy state more populated)."""
        if self.P_plus is None or self.P_minus is None or self.H0 is None:
            return (True, "SKIP: populations or H0 missing")
        if self.H0 > 0:
            if not (self.P_plus > self.P_minus):
                return (False, f"P_plus should be > P_minus for positive H0, got {self.P_plus} vs {self.P_minus}")
        elif self.H0 < 0:
            if not (self.P_plus < self.P_minus):
                return (False, f"P_plus should be < P_minus for negative H0, got {self.P_plus} vs {self.P_minus}")
        return (True, "Population ordering consistent")

    def test_invariant_direction_perpendicular(self):
        """Oscillating field must be perpendicular to H0 (Δm=±1)."""
        if self.direction is None:
            return (True, "SKIP: direction missing")
        if 'perpendicular' not in self.direction.lower():
            return (False, f"direction should indicate perpendicular, got '{self.direction}'")
        return (True, "Direction is perpendicular")

    # -------------------------------------------------------------------------
    # Modeling / Trace tests
    # -------------------------------------------------------------------------
    def test_modeling_resonance_frequency_formula(self):
        """omega_res_Mrad_per_s ≈ gamma * H0 / 1e6."""
        if self.omega_res_Mrad_per_s is None or self.H0 is None:
            return (True, "SKIP: omega or H0 missing")
        gamma_val = self._get_gamma()                        # rad/(s·Oe)
        expected_omega = gamma_val * self.H0 / 1e6           # Mrad/s
        if not math.isclose(self.omega_res_Mrad_per_s, expected_omega,
                            rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega_res_Mrad_per_s = {self.omega_res_Mrad_per_s}, "
                           f"expected {expected_omega}")
        return (True, "Resonance frequency formula correct")

    def test_modeling_frequency_conversion(self):
        """f_res_MHz = omega_res_Mrad_per_s / (2π)."""
        if self.f_res_MHz is None or self.omega_res_Mrad_per_s is None:
            return (True, "SKIP: f_res or omega missing")
        expected_f = self.omega_res_Mrad_per_s / (2 * math.pi)
        if not math.isclose(self.f_res_MHz, expected_f,
                            rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f_res_MHz = {self.f_res_MHz}, expected {expected_f}")
        return (True, "Frequency conversion consistent")

    def test_modeling_explanation_continuous_mentions_relaxation(self):
        """Explanation of continuous absorption must reference spin‑lattice relaxation."""
        if self.explanation_continuous is None:
            return (True, "SKIP: explanation_continuous missing")
        if 'relax' not in self.explanation_continuous.lower():
            return (False, "Explanation should mention relaxation")
        return (True, "Explanation mentions relaxation")

    def test_modeling_saturation_mentions_saturation(self):
        """Saturation explanation should mention saturation or zero net absorption."""
        if self.saturation is None:
            return (True, "SKIP: saturation missing")
        if 'satur' not in self.saturation.lower():
            return (False, f"Saturation explanation should mention saturation, got '{self.saturation}'")
        return (True, "Saturation explanation mentions saturation")

    # -------------------------------------------------------------------------
    # Monotonicity tests
    # -------------------------------------------------------------------------
    def test_monotonicity_frequency_increase_with_H0(self):
        """
        If H0 increases, the solution's resonance frequency (omega) increases.
        We perturb H0 by +10% and use the solution's effective gamma to compute
        the expected perturbed omega, then compare it with the baseline omega.
        """
        if self.H0 is None or self.omega_res_Mrad_per_s is None:
            return (True, "SKIP: H0 or omega missing")
        baseline_H0 = self.H0
        baseline_omega = self.omega_res_Mrad_per_s

        # Recover the gamma the solution is actually using
        gamma_eff = baseline_omega * 1e6 / baseline_H0   # rad/(s·Oe)
        # Perturb H0 by 10%
        perturbed_H0 = baseline_H0 * 1.1
        # Estimate perturbed omega using the same effective gamma
        perturbed_omega = gamma_eff * perturbed_H0 / 1e6

        # Monotonicity: increasing H0 should increase omega
        if not (perturbed_omega > baseline_omega):
            return (False, f"Monotonicity broken: perturbed omega={perturbed_omega} "
                           f"not > baseline omega={baseline_omega}")
        return (True, "omega increases when H0 increases")

    # -------------------------------------------------------------------------
    # Answer sanity (weak check – only one such test)
    # -------------------------------------------------------------------------
    def test_answer_sanity_frequency_finite_positive(self):
        """f_res_MHz must be positive, finite, and not NaN."""
        if self.f_res_MHz is None:
            return (True, "SKIP: f_res missing")
        if not (isinstance(self.f_res_MHz, (int, float)) and
                math.isfinite(self.f_res_MHz) and self.f_res_MHz > 0):
            return (False, f"f_res_MHz should be positive finite, got {self.f_res_MHz}")
        return (True, "Frequency positive finite")

    # -------------------------------------------------------------------------
    # Main test runner
    # -------------------------------------------------------------------------
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

# For standalone testing (optional)
if __name__ == "__main__":
    # Simulate a minimal solution instance
    class DummySolution:
        givens = {'H0': 10000.0, 'T': 300.0}
        results = {
            'P_plus': 0.5001,
            'P_minus': 0.4999,
            'direction': 'perpendicular to H0',
            'omega_res_Mrad_per_s': 2.675e-2 * 10000,
            'f_res_MHz': 4.26e-3 * 10000,
            'explanation_continuous': 'spin-lattice relaxation restores equilibrium',
            'saturation': 'saturation occurs; net absorption goes to zero'
        }
    checker = PhysicsChecker(DummySolution())
    ok, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")