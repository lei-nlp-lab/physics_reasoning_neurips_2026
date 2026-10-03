import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits the solution of the ice-water mixing problem.
    Checks interface completeness, physical invariants, modeling consistency,
    and monotonic dependencies.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Access results dictionary safely
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        else:
            self.results = {}

        # Access givens dictionary safely
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.givens = solution_instance.givens
        else:
            self.givens = {}

        # Extract the final result (must be present)
        self.T_f = self.results.get('T_f', None)

        # Extract given quantities (may be in givens dict or as direct attributes)
        self.m_i = self._get_given('m_i') or getattr(solution_instance, 'm_i', None)
        self.m_w = self._get_given('m_w') or getattr(solution_instance, 'm_w', None)
        self.c = self._get_given('c') or getattr(solution_instance, 'c', None)
        self.L_f = self._get_given('L_f') or getattr(solution_instance, 'L_f', None)

        # Initial temperatures (may be given)
        self.T_i_ice = self._get_given('T_i_ice') or getattr(solution_instance, 'T_i_ice', None)
        self.T_i_water = self._get_given('T_i_water') or getattr(solution_instance, 'T_i_water', None)

    def _get_given(self, key: str) -> Optional[Any]:
        """Helper to safely extract from givens dict (if dict)."""
        if isinstance(self.givens, dict):
            return self.givens.get(key, None)
        return None

    # ------------------------------------------------------------------
    # Interface tests (presence and type of required data)
    # ------------------------------------------------------------------
    def test_interface_result_key_T_f_exists(self) -> Tuple[bool, str]:
        if self.T_f is None:
            return True, "SKIP: result key 'T_f' not found in results"
        return True, "PASS: 'T_f' key present"

    def test_interface_T_f_is_numeric(self) -> Tuple[bool, str]:
        if self.T_f is None:
            return True, "SKIP: T_f not available to check type"
        if not isinstance(self.T_f, (int, float)):
            return False, "FAIL: T_f is not numeric (int or float)"
        return True, "PASS: T_f is numeric"

    # ------------------------------------------------------------------
    # Physical invariant tests (theorem-level checks, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_temperature_range(self) -> Tuple[bool, str]:
        if self.T_f is None:
            return True, "SKIP: T_f not available, cannot assess range"
        if self.T_f <= 0.0 or self.T_f >= 70.0:
            return False, f"FAIL: T_f = {self.T_f} is outside expected range (0,70)"
        return True, "PASS: T_f within (0,70)"

    def test_invariant_ice_completely_melts(self) -> Tuple[bool, str]:
        # Check if there is enough heat from water cooling to 0 °C to melt all ice.
        if any(x is None for x in [self.m_w, self.m_i, self.c, self.L_f]):
            return True, "SKIP: missing one or more required givens (m_w, m_i, c, L_f)"
        T_water_initial = self.T_i_water if self.T_i_water is not None else 70.0
        Q_available = self.m_w * self.c * (T_water_initial - 0.0)
        Q_needed = self.m_i * self.L_f
        if Q_available < Q_needed - 1e-12:
            return False, (
                f"FAIL: insufficient heat from water to melt ice: "
                f"available {Q_available:.2f} cal vs needed {Q_needed:.2f} cal"
            )
        return True, "PASS: enough heat to completely melt the ice"

    def test_invariant_energy_conservation(self) -> Tuple[bool, str]:
        # Heat lost by warm water = heat gained by ice (melting + warming)
        if any(x is None for x in [self.T_f, self.m_w, self.m_i, self.c, self.L_f]):
            return True, "SKIP: missing required data for energy conservation check"
        T_i_water = self.T_i_water if self.T_i_water is not None else 70.0
        T_i_ice = self.T_i_ice if self.T_i_ice is not None else 0.0

        Q_lost = self.m_w * self.c * (T_i_water - self.T_f)
        Q_gain = self.m_i * self.L_f + self.m_i * self.c * (self.T_f - T_i_ice)

        if not math.isclose(Q_lost, Q_gain, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"FAIL: energy imbalance: Q_lost={Q_lost:.2f} cal, Q_gain={Q_gain:.2f} cal"
            )
        return True, "PASS: energy conservation satisfied"

    # ------------------------------------------------------------------
    # Modeling / trace tests (consistency with explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_heat_balance_equation_satisfied(self) -> Tuple[bool, str]:
        """
        Verify that the reported T_f satisfies the original heat balance equation:
        m_w * c * (T_i_water - T_f) = m_i * L_f + m_i * c * (T_f - T_i_ice)
        This checks the core modeling step without re-solving for T_f.
        """
        if any(x is None for x in [self.T_f, self.m_w, self.m_i, self.c, self.L_f]):
            return True, "SKIP: missing required data for heat balance check"
        T_i_water = self.T_i_water if self.T_i_water is not None else 70.0
        T_i_ice = self.T_i_ice if self.T_i_ice is not None else 0.0

        left = self.m_w * self.c * (T_i_water - self.T_f)
        right = self.m_i * self.L_f + self.m_i * self.c * (self.T_f - T_i_ice)

        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"FAIL: left-hand side = {left:.2f} cal, right-hand side = {right:.2f} cal"
            )
        return True, "PASS: heat balance equation holds with reported T_f"

    def test_modeling_latent_and_sensible_heat_sum(self) -> Tuple[bool, str]:
        """
        Verify that the total heat lost by the water equals the sum of the latent heat
        (to melt the ice) and the sensible heat (to warm the melted ice to T_f).
        This checks the modeling of the heat gain side.
        """
        if any(x is None for x in [self.T_f, self.m_w, self.m_i, self.c, self.L_f]):
            return True, "SKIP: missing required data for heat sum check"
        T_i_water = self.T_i_water if self.T_i_water is not None else 70.0

        Q_lost_total = self.m_w * self.c * (T_i_water - self.T_f)
        Q_latent = self.m_i * self.L_f
        Q_sensible = self.m_i * self.c * (self.T_f - 0.0)  # T_i_ice = 0

        if not math.isclose(Q_lost_total, Q_latent + Q_sensible, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"FAIL: total heat lost ({Q_lost_total:.2f} cal) "
                f"!= latent ({Q_latent:.2f} cal) + sensible ({Q_sensible:.2f} cal)"
            )
        return True, "PASS: heat lost equals latent + sensible heats"

    # ------------------------------------------------------------------
    # Monotonicity tests (trend expected from physical intuition)
    # ------------------------------------------------------------------
    def test_monotonicity_increase_ice_mass_decreases_T_f(self) -> Tuple[bool, str]:
        if any(x is None for x in [self.m_w, self.m_i, self.c, self.L_f, self.T_f]):
            return True, "SKIP: missing required data for monotonicity check"

        T_i_water = self.T_i_water if self.T_i_water is not None else 70.0
        # Compute baseline T_f from the same formula using current givens
        numerator = self.m_w * self.c * T_i_water - self.m_i * self.L_f
        denominator = self.c * (self.m_w + self.m_i)
        if denominator == 0:
            return True, "SKIP: denominator zero"
        baseline_T_f = numerator / denominator  # should be close to self.T_f

        # Perturb: increase ice mass by 10%
        m_i_new = self.m_i * 1.1
        new_numerator = self.m_w * self.c * T_i_water - m_i_new * self.L_f
        new_denominator = self.c * (self.m_w + m_i_new)
        new_T_f = new_numerator / new_denominator

        if new_T_f >= baseline_T_f - 1e-9:
            return False, (
                f"FAIL: increasing ice mass from {self.m_i} g to {m_i_new:.2f} g "
                f"did not decrease T_f (baseline {baseline_T_f:.4f}, new {new_T_f:.4f})"
            )
        return True, "PASS: increasing ice mass lowers final temperature"

    def test_monotonicity_increase_water_mass_increases_T_f(self) -> Tuple[bool, str]:
        if any(x is None for x in [self.m_w, self.m_i, self.c, self.L_f, self.T_f]):
            return True, "SKIP: missing required data for monotonicity check"

        T_i_water = self.T_i_water if self.T_i_water is not None else 70.0
        numerator = self.m_w * self.c * T_i_water - self.m_i * self.L_f
        denominator = self.c * (self.m_w + self.m_i)
        if denominator == 0:
            return True, "SKIP: denominator zero"
        baseline_T_f = numerator / denominator

        # Perturb: increase water mass by 10%
        m_w_new = self.m_w * 1.1
        new_numerator = m_w_new * self.c * T_i_water - self.m_i * self.L_f
        new_denominator = self.c * (m_w_new + self.m_i)
        new_T_f = new_numerator / new_denominator

        if new_T_f <= baseline_T_f + 1e-9:
            return False, (
                f"FAIL: increasing water mass from {self.m_w} g to {m_w_new:.2f} g "
                f"did not increase T_f (baseline {baseline_T_f:.4f}, new {new_T_f:.4f})"
            )
        return True, "PASS: increasing water mass raises final temperature"

    # ------------------------------------------------------------------
    # Master runner that discovers and executes all test_* methods
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            if passed:
                logs.append(f"PASS: {name}: {msg}")
            else:
                logs.append(f"FAIL: {name}: {msg}")
                all_pass = False
        return (all_pass, logs)