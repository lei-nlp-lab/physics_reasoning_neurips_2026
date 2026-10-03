import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Inputs (with safe access and default fallbacks)
        self.V_chamber = getattr(solution_instance, 'V_chamber', 0.13)  # m^3
        self.n_bottles = getattr(solution_instance, 'n_bottles', 10)
        self.V_bottle = getattr(solution_instance, 'V_bottle', 3.2e-2)  # m^3
        self.P0_bottle = getattr(solution_instance, 'P0_bottle', 1.5e7)  # Pa
        self.P1_bottle = getattr(solution_instance, 'P1_bottle', 2.0e6)  # Pa
        # Temperatures: often rounded to 300 K and 1500 K in such problems
        self.T0 = getattr(solution_instance, 'T0', 300.0)  # K (27°C -> 300 K)
        self.T1 = getattr(solution_instance, 'T1', 1500.0)  # K (1227°C -> 1500 K)

        # Results (from results dictionary if available)
        results = getattr(solution_instance, 'results', {})
        self.p_2 = results.get('p_2', 0.0)
        self.p_3 = results.get('p_3', 0.0)
        self.V_prime = results.get('V_prime', 0.0)  # V' from step 2

        # Constants
        self.R = 8.314462618  # ideal gas constant, though not strictly needed

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics laws: Boyle's Law and Gay-Lussac's Law.
        """
        logs = []
        success = True

        # 1. Boyle's Law for one bottle: P0 * V_bottle = P1 * V1, where V1 = V_bottle * (P0/P1)
        V1 = self.V_bottle * self.P0_bottle / self.P1_bottle
        left = self.P0_bottle * self.V_bottle
        right = self.P1_bottle * V1
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Boyle's Law violated for bottle: {left} != {right}")
            success = False
        else:
            logs.append("Boyle's Law holds for one bottle.")

        # 2. Gay-Lussac's Law for the chamber: p2 / T0 = p3 / T1
        if self.p_2 > 0 and self.p_3 > 0:
            ratio2 = self.p_2 / self.T0
            ratio3 = self.p_3 / self.T1
            if not math.isclose(ratio2, ratio3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Gay-Lussac's Law violated: p2/T0={ratio2}, p3/T1={ratio3}")
                success = False
            else:
                logs.append("Gay-Lussac's Law holds for chamber heating.")
        else:
            logs.append("Skipping Gay-Lussac test: p2 or p3 not available.")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Check consistency of intermediate results with final answers.
        """
        logs = []
        success = True

        # 1. Compute V' from bottle data and compare with stored V'
        V1 = self.V_bottle * self.P0_bottle / self.P1_bottle
        V_prime_calc = V1 - self.V_bottle
        if self.V_prime != 0.0:  # if stored
            if not math.isclose(self.V_prime, V_prime_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"V' inconsistency: stored {self.V_prime}, computed {V_prime_calc}")
                success = False
            else:
                logs.append(f"V' consistent: {self.V_prime} ≈ {V_prime_calc}")
        else:
            logs.append(f"Computed V' = {V_prime_calc} (not stored for direct check)")

        # 2. Compute p2 from V' and compare with stored p2
        p2_calc = self.n_bottles * self.P1_bottle * V_prime_calc / self.V_chamber
        if self.p_2 != 0.0:
            if not math.isclose(self.p_2, p2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"p2 inconsistency: stored {self.p_2}, computed {p2_calc}")
                success = False
            else:
                logs.append(f"p2 consistent: {self.p_2} ≈ {p2_calc}")
        else:
            logs.append(f"Computed p2 = {p2_calc} (not stored for direct check)")

        # 3. Compute p3 from p2 and temperatures
        if self.p_2 != 0.0:
            p3_calc = self.p_2 * self.T1 / self.T0
            if self.p_3 != 0.0:
                if not math.isclose(self.p_3, p3_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"p3 inconsistency: stored {self.p_3}, computed {p3_calc}")
                    success = False
                else:
                    logs.append(f"p3 consistent: {self.p_3} ≈ {p3_calc}")
            else:
                logs.append(f"Computed p3 = {p3_calc} (not stored for direct check)")
        else:
            logs.append("Skipping p3 consistency: p2 not available.")

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb an input (number of bottles) and verify the trend.
        """
        logs = []
        success = True

        # Compute baseline p2 and p3 using formulas (not stored values)
        V1 = self.V_bottle * self.P0_bottle / self.P1_bottle
        V_prime_calc = V1 - self.V_bottle
        p2_baseline = self.n_bottles * self.P1_bottle * V_prime_calc / self.V_chamber
        p3_baseline = p2_baseline * self.T1 / self.T0

        # Perturb: increase number of bottles by 1
        n_perturbed = self.n_bottles + 1
        p2_perturbed = n_perturbed * self.P1_bottle * V_prime_calc / self.V_chamber
        p3_perturbed = p2_perturbed * self.T1 / self.T0

        # Check monotonic increase (more gas -> higher pressure)
        if p2_perturbed <= p2_baseline:
            logs.append(f"Monotonicity violated: p2 did not increase with more bottles")
            success = False
        else:
            logs.append(f"p2 increases with bottle count: {p2_baseline} -> {p2_perturbed}")

        if p3_perturbed <= p3_baseline:
            logs.append(f"Monotonicity violated: p3 did not increase with more bottles")
            success = False
        else:
            logs.append(f"p3 increases with bottle count: {p3_baseline} -> {p3_perturbed}")

        # Check proportionality: p2_perturbed / p2_baseline ≈ n_perturbed / n_bottles
        expected_ratio = n_perturbed / self.n_bottles
        actual_ratio_p2 = p2_perturbed / p2_baseline
        if not math.isclose(actual_ratio_p2, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Proportionality off for p2: expected ratio {expected_ratio}, got {actual_ratio_p2}")
            success = False
        else:
            logs.append(f"p2 scales linearly with bottle count (ratio {actual_ratio_p2:.3f}).")

        # For p3, same ratio should hold because temperature scaling is independent
        actual_ratio_p3 = p3_perturbed / p3_baseline
        if not math.isclose(actual_ratio_p3, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Proportionality off for p3: expected ratio {expected_ratio}, got {actual_ratio_p3}")
            success = False
        else:
            logs.append(f"p3 scales linearly with bottle count (ratio {actual_ratio_p3:.3f}).")

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = ["=== PhysicsChecker Report ==="]
        overall_success = True

        # Test invariants
        success, logs = self.test_invariants()
        all_logs.append("\n--- test_invariants ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        # Test trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- test_trace_consistency ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        # Test monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- test_monotonicity ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs