import math
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits a solution instance for the dam thermal vs gravitational energy problem.
    Expects solution_instance to have attributes for givens (m, c, delta_T, g, h, cal_to_J,
    T_low, T_high) and a results dict with keys 'W_thermal_cal' and 'U_grav_cal'.
    All missing attributes default to None.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Givens (SI units unless noted)
        self.m = getattr(solution_instance, 'm', None)          # mass in kg (1 g = 0.001 kg)
        self.c = getattr(solution_instance, 'c', None)          # specific heat J/(kg·°C)
        self.delta_T = getattr(solution_instance, 'delta_T', None)  # °C
        self.g = getattr(solution_instance, 'g', None)          # m/s^2
        self.h = getattr(solution_instance, 'h', None)          # m
        self.cal_to_J = getattr(solution_instance, 'cal_to_J', None)  # J/cal
        self.T_low = getattr(solution_instance, 'T_low', None)     # K (bottom)
        self.T_high = getattr(solution_instance, 'T_high', None)   # K (surface)

        # Results – mandatory keys
        results = getattr(solution_instance, 'results', {}) or {}
        self.W_thermal_cal = results.get('W_thermal_cal', None)  # Carnot work in cal
        self.U_grav_cal = results.get('U_grav_cal', None)        # gravitational energy in cal

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_results_exist(self) -> Tuple[bool, str]:
        """Both expected result keys must be present and not None."""
        if self.W_thermal_cal is None:
            return (False, "Missing key: 'W_thermal_cal'")
        if self.U_grav_cal is None:
            return (False, "Missing key: 'U_grav_cal'")
        return (True, "Both 'W_thermal_cal' and 'U_grav_cal' present")

    def test_interface_results_numeric(self) -> Tuple[bool, str]:
        """Results must be numeric and finite."""
        for name, val in [('W_thermal_cal', self.W_thermal_cal), ('U_grav_cal', self.U_grav_cal)]:
            if not isinstance(val, (int, float)):
                return (False, f"'{name}' is not numeric: {type(val)}")
            if math.isnan(val) or math.isinf(val):
                return (False, f"'{name}' is NaN or Inf")
        return (True, "Both results are finite numbers")

    # ------------------------------------------------------------------
    # Theorem-based invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_gravitational_energy(self) -> Tuple[bool, str]:
        """
        If all gravitational givens are available, verify that
        U_grav_cal ≈ (m * g * h) / cal_to_J.
        """
        needed = ['m', 'g', 'h', 'cal_to_J']
        missing = [v for v in needed if getattr(self, v) is None]
        if missing:
            return (False, f"SKIP: missing givens {missing}")

        expected_U_J = self.m * self.g * self.h
        expected_U_cal = expected_U_J / self.cal_to_J
        if not math.isclose(self.U_grav_cal, expected_U_cal, rel_tol=0.05, abs_tol=0.01):
            return (False, f"U_grav_cal {self.U_grav_cal} != expected {expected_U_cal:.4f} cal")
        return (True, f"U_grav_cal matches gravitational energy")

    def test_invariant_thermal_work(self) -> Tuple[bool, str]:
        """
        If all thermal givens are available, verify that
        W_thermal_cal ≈ ((T_high - T_low)/T_high) * (m * c * delta_T) / cal_to_J.
        """
        needed = ['m', 'c', 'delta_T', 'T_low', 'T_high', 'cal_to_J']
        missing = [v for v in needed if getattr(self, v) is None]
        if missing:
            return (False, f"SKIP: missing givens {missing}")

        if self.cal_to_J <= 0 or self.T_high <= 0:
            return (False, "Invalid constants (cal_to_J or T_high)")

        Q_J = self.m * self.c * self.delta_T
        efficiency = (self.T_high - self.T_low) / self.T_high
        W_J = efficiency * Q_J
        expected_W_cal = W_J / self.cal_to_J
        if not math.isclose(self.W_thermal_cal, expected_W_cal, rel_tol=0.1, abs_tol=0.05):
            return (False, f"W_thermal_cal {self.W_thermal_cal} != expected {expected_W_cal:.4f} cal")
        return (True, "W_thermal_cal matches Carnot work")

    # ------------------------------------------------------------------
    # Modeling/trace tests (check internal consistency)
    # ------------------------------------------------------------------
    def test_modeling_thermal_work_equation(self) -> Tuple[bool, str]:
        """
        Re-derive W_thermal_cal from the given constants and verify it matches the recorded result.
        This is allowed because we re-derive from givens.
        """
        needed = ['m', 'c', 'delta_T', 'T_low', 'T_high', 'cal_to_J']
        missing = [v for v in needed if getattr(self, v) is None]
        if missing:
            return (False, f"SKIP: missing givens {missing}")

        if self.W_thermal_cal is None:
            return (False, "SKIP: W_thermal_cal not available")

        # Recompute exactly as the pseudocode would
        Q_J = self.m * self.c * self.delta_T
        efficiency = (self.T_high - self.T_low) / self.T_high
        W_J = efficiency * Q_J
        expected_W_cal = W_J / self.cal_to_J

        if not math.isclose(self.W_thermal_cal, expected_W_cal, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"Modeling: W_thermal_cal {self.W_thermal_cal} != recomputed {expected_W_cal:.6f}")
        return (True, "W_thermal_cal matches re-derived value")

    def test_modeling_gravitational_work_equation(self) -> Tuple[bool, str]:
        """
        Re-derive U_grav_cal from the given constants and verify it matches.
        """
        needed = ['m', 'g', 'h', 'cal_to_J']
        missing = [v for v in needed if getattr(self, v) is None]
        if missing:
            return (False, f"SKIP: missing givens {missing}")

        if self.U_grav_cal is None:
            return (False, "SKIP: U_grav_cal not available")

        U_J = self.m * self.g * self.h
        expected_U_cal = U_J / self.cal_to_J

        if not math.isclose(self.U_grav_cal, expected_U_cal, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"Modeling: U_grav_cal {self.U_grav_cal} != recomputed {expected_U_cal:.6f}")
        return (True, "U_grav_cal matches re-derived value")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_thermal_work_increases_with_delta_T(self) -> Tuple[bool, str]:
        """
        Perturb delta_T by +10% and check that the resulting W_thermal_cal increases.
        We recompute the expected perturbed value using the same Carnot formula.
        """
        needed = ['m', 'c', 'delta_T', 'T_low', 'T_high', 'cal_to_J', 'W_thermal_cal']
        missing = [v for v in needed if getattr(self, v) is None]
        if missing:
            return (False, f"SKIP: missing givens or result {missing}")

        # Baseline
        Q_J = self.m * self.c * self.delta_T
        efficiency = (self.T_high - self.T_low) / self.T_high
        baseline_W_cal = (efficiency * Q_J) / self.cal_to_J

        # Perturb: increase delta_T by 10%
        perturbed_delta_T = self.delta_T * 1.1
        Q_J_pert = self.m * self.c * perturbed_delta_T
        perturbed_W_cal = (efficiency * Q_J_pert) / self.cal_to_J

        if perturbed_W_cal <= baseline_W_cal:
            return (False, f"Perturbed W_thermal_cal {perturbed_W_cal:.6f} not > baseline {baseline_W_cal:.6f}")
        return (True, "Increasing ΔT increases thermal work (as expected)")

    def test_monotonicity_gravitational_energy_increases_with_height(self) -> Tuple[bool, str]:
        """
        Perturb h by +10% and check that U_grav_cal increases.
        """
        needed = ['m', 'g', 'h', 'cal_to_J', 'U_grav_cal']
        missing = [v for v in needed if getattr(self, v) is None]
        if missing:
            return (False, f"SKIP: missing givens or result {missing}")

        # Baseline
        baseline_U_cal = (self.m * self.g * self.h) / self.cal_to_J

        # Perturb: increase height by 10%
        perturbed_h = self.h * 1.1
        perturbed_U_cal = (self.m * self.g * perturbed_h) / self.cal_to_J

        if perturbed_U_cal <= baseline_U_cal:
            return (False, f"Perturbed U_grav_cal {perturbed_U_cal:.6f} not > baseline {baseline_U_cal:.6f}")
        return (True, "Increasing h increases gravitational energy (as expected)")

    # ------------------------------------------------------------------
    # Energy comparison invariant
    # ------------------------------------------------------------------
    def test_invariant_energy_comparison(self) -> Tuple[bool, str]:
        """For h ~ 110 m, gravitational energy should be smaller than thermal work."""
        if self.W_thermal_cal is None or self.U_grav_cal is None:
            return (False, "SKIP: results missing")
        if self.U_grav_cal < self.W_thermal_cal:
            return (True, "U_grav_cal < W_thermal_cal (as expected for ~100 m height)")
        return (False, f"U_grav_cal {self.U_grav_cal} not less than W_thermal_cal {self.W_thermal_cal}")

    # ------------------------------------------------------------------
    # Weak sanity (final-answer related, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self) -> Tuple[bool, str]:
        """Both energies should be positive and of order 0.1–1 cal for 1 g of water."""
        if self.W_thermal_cal is None or self.U_grav_cal is None:
            return (False, "SKIP: results missing")
        if not (0 < self.W_thermal_cal < 10):
            return (False, f"W_thermal_cal {self.W_thermal_cal} out of plausible range [0,10] cal")
        if not (0 < self.U_grav_cal < 10):
            return (False, f"U_grav_cal {self.U_grav_cal} out of plausible range [0,10] cal")
        return (True, "Both energies are positive and in plausible magnitude range")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
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