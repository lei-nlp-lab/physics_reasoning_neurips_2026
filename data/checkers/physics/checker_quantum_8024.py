import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Physics checker for the two-particle three-level problem with Fermi, Bose and Boltzmann statistics.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all relevant inputs and results safely using None defaults.
        """
        # Givens (may be stored under solution_instance.givens or directly as attributes)
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.epsilon = solution_instance.givens.get('epsilon', None)
            self.beta = solution_instance.givens.get('beta', None)  # beta = 1/(kT)
        else:
            self.epsilon = getattr(solution_instance, 'epsilon', None)
            self.beta = getattr(solution_instance, 'beta', None)

        # Results (expected keys from pseudocode emit contract)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = getattr(solution_instance, 'results', {})

        self.Z_F = results.get('Z_F', None)
        self.E_avg_F = results.get('E_avg_F', None)
        self.Z_B = results.get('Z_B', None)
        self.E_avg_B = results.get('E_avg_B', None)
        self.Z_Boltz = results.get('Z_Boltz', None)
        self.E_avg_Boltz = results.get('E_avg_Boltz', None)
        self.avg_occupancy = results.get('avg_occupancy', None)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _compute_expected_Z_F(self, epsilon: float, beta: float) -> float:
        """Compute Fermi partition function for given epsilon and beta."""
        return 1.0 + 2.0 * math.exp(-beta * epsilon) + 2.0 * math.exp(-2.0 * beta * epsilon) + math.exp(-3.0 * beta * epsilon)

    def _compute_expected_Z_B(self, epsilon: float, beta: float) -> float:
        """Compute Bose partition function."""
        return 3.0 + 2.0 * math.exp(-beta * epsilon) + 3.0 * math.exp(-2.0 * beta * epsilon) \
               + math.exp(-3.0 * beta * epsilon) + math.exp(-4.0 * beta * epsilon)

    def _compute_expected_Z_Boltz(self, epsilon: float, beta: float) -> float:
        """Compute Boltzmann partition function."""
        Z1 = 2.0 + math.exp(-beta * epsilon) + math.exp(-2.0 * beta * epsilon)
        return Z1 * Z1

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all seven expected result keys are present."""
        required = ['Z_F', 'E_avg_F', 'Z_B', 'E_avg_B', 'Z_Boltz', 'E_avg_Boltz', 'avg_occupancy']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_results_types_numeric(self) -> Tuple[bool, str]:
        """Check that all numeric results are floats (or can be used as numbers)."""
        if self.epsilon is None or self.beta is None:
            return (True, "SKIP: no givens to evaluate numeric type")
        numeric_keys = ['Z_F', 'E_avg_F', 'Z_B', 'E_avg_B', 'Z_Boltz', 'E_avg_Boltz', 'avg_occupancy']
        bad = []
        for key in numeric_keys:
            val = getattr(self, key, None)
            if val is not None:
                try:
                    float(val)
                except (TypeError, ValueError):
                    bad.append(key)
        if bad:
            return (False, f"Non‑numeric results: {bad}")
        return (True, "All numeric results are convertible to float")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_fermi_partition(self) -> Tuple[bool, str]:
        """Check Fermi partition function matches known sum (if givens available)."""
        if self.epsilon is None or self.beta is None or self.Z_F is None:
            return (True, "SKIP: insufficient data")
        expected = self._compute_expected_Z_F(self.epsilon, self.beta)
        if not math.isclose(float(self.Z_F), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Z_F mismatch: got {self.Z_F}, expected {expected}")
        return (True, f"Z_F ≈ {expected}")

    def test_invariant_bose_partition(self) -> Tuple[bool, str]:
        """Check Bose partition function matches known sum."""
        if self.epsilon is None or self.beta is None or self.Z_B is None:
            return (True, "SKIP: insufficient data")
        expected = self._compute_expected_Z_B(self.epsilon, self.beta)
        if not math.isclose(float(self.Z_B), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Z_B mismatch: got {self.Z_B}, expected {expected}")
        return (True, f"Z_B ≈ {expected}")

    def test_invariant_boltzmann_partition(self) -> Tuple[bool, str]:
        """Check Boltzmann partition function matches square of single‑particle Z."""
        if self.epsilon is None or self.beta is None or self.Z_Boltz is None:
            return (True, "SKIP: insufficient data")
        expected = self._compute_expected_Z_Boltz(self.epsilon, self.beta)
        if not math.isclose(float(self.Z_Boltz), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Z_Boltz mismatch: got {self.Z_Boltz}, expected {expected}")
        return (True, f"Z_Boltz ≈ {expected}")

    def test_invariant_fermi_energy_consistency(self) -> Tuple[bool, str]:
        """Check that E_avg_F * Z_F equals the sum of (energy * Boltzmann weight)."""
        if self.epsilon is None or self.beta is None:
            return (True, "SKIP: no givens")
        if self.Z_F is None or self.E_avg_F is None:
            return (True, "SKIP: missing result")
        e = self.epsilon
        b = self.beta
        numerator = (0.0 * 1 + e * 2.0 * math.exp(-b * e) +
                     2.0 * e * 2.0 * math.exp(-2.0 * b * e) +
                     3.0 * e * math.exp(-3.0 * b * e))
        lhs = float(self.E_avg_F) * float(self.Z_F)
        if not math.isclose(lhs, numerator, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_avg_F * Z_F = {lhs}, expected numerator = {numerator}")
        return (True, "Fermi average energy consistent with partition")

    def test_invariant_bose_energy_consistency(self) -> Tuple[bool, str]:
        """Check that E_avg_B * Z_B equals the sum of (energy * Boltzmann weight)."""
        if self.epsilon is None or self.beta is None:
            return (True, "SKIP: no givens")
        if self.Z_B is None or self.E_avg_B is None:
            return (True, "SKIP: missing result")
        e = self.epsilon
        b = self.beta
        numerator = (0.0 * 3 + e * 2.0 * math.exp(-b * e) +
                     2.0 * e * 3.0 * math.exp(-2.0 * b * e) +
                     3.0 * e * 1.0 * math.exp(-3.0 * b * e) +
                     4.0 * e * 1.0 * math.exp(-4.0 * b * e))
        lhs = float(self.E_avg_B) * float(self.Z_B)
        if not math.isclose(lhs, numerator, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_avg_B * Z_B = {lhs}, expected numerator = {numerator}")
        return (True, "Bose average energy consistent with partition")

    def test_invariant_boltzmann_energy_consistency(self) -> Tuple[bool, str]:
        """Check that E_avg_Boltz * Z_Boltz equals the sum of (energy * Boltzmann weight)."""
        if self.epsilon is None or self.beta is None:
            return (True, "SKIP: no givens")
        if self.Z_Boltz is None or self.E_avg_Boltz is None:
            return (True, "SKIP: missing result")
        e = self.epsilon
        b = self.beta
        numerator = (0.0 * 4 + e * 4.0 * math.exp(-b * e) +
                     2.0 * e * 5.0 * math.exp(-2.0 * b * e) +
                     3.0 * e * 2.0 * math.exp(-3.0 * b * e) +
                     4.0 * e * 1.0 * math.exp(-4.0 * b * e))
        lhs = float(self.E_avg_Boltz) * float(self.Z_Boltz)
        if not math.isclose(lhs, numerator, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_avg_Boltz * Z_Boltz = {lhs}, expected numerator = {numerator}")
        return (True, "Boltzmann average energy consistent with partition")

    def test_invariant_ordering(self) -> Tuple[bool, str]:
        """Check Z_F < Z_B < Z_Boltz (should hold for any β>0, ε>0)."""
        if any(x is None for x in (self.Z_F, self.Z_B, self.Z_Boltz)):
            return (True, "SKIP: missing partition functions")
        Zf = float(self.Z_F)
        Zb = float(self.Z_B)
        Zbol = float(self.Z_Boltz)
        if not (Zf < Zb < Zbol):
            return (False, f"Ordering violated: Z_F={Zf}, Z_B={Zb}, Z_Boltz={Zbol}")
        return (True, "Order Z_F < Z_B < Z_Boltz holds")

    def test_invariant_avg_occupancy(self) -> Tuple[bool, str]:
        """Check avg_occupancy = 0.5 (derived from N=2, 4 states)."""
        if self.avg_occupancy is None:
            return (True, "SKIP: avg_occupancy not provided")
        if not math.isclose(float(self.avg_occupancy), 0.5, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"avg_occupancy = {self.avg_occupancy}, expected ≈0.5")
        return (True, "avg_occupancy ≈ 0.5")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_fermi_configurations_count(self) -> Tuple[bool, str]:
        """Fermi should have exactly 6 configurations (N=2, 4 states). Not directly testable, but Z_F ≈ 6 at high T."""
        if self.beta is None or self.epsilon is None or self.Z_F is None:
            return (True, "SKIP: insufficient data")
        # At β=0 (infinite temperature), all terms =1, Z_F =6.
        # We cannot directly verify; check that for finite β, Z_F < 6.
        if float(self.Z_F) >= 6.0:
            return (False, f"Z_F should be <6 at finite temperature; got {self.Z_F}")
        return (True, "Fermi partition function less than 6 (consistent with finite temperature)")

    def test_modeling_bose_extra_configurations(self) -> Tuple[bool, str]:
        """Bose partition function should have more terms than Fermi."""
        if self.Z_F is None or self.Z_B is None:
            return (True, "SKIP: missing Z")
        if float(self.Z_B) <= float(self.Z_F):
            return (False, "Z_B should be larger than Z_F due to extra configurations")
        return (True, "Bose partition larger than Fermi")

    def test_modeling_boltzmann_independence(self) -> Tuple[bool, str]:
        """Boltzmann partition should be the square of single‑particle Z."""
        if self.epsilon is None or self.beta is None or self.Z_Boltz is None:
            return (True, "SKIP: insufficient data")
        # This is essentially checked in the invariant; keep as a modeling note.
        return (True, "Boltzmann Z is square of single‑particle Z (verified elsewhere)")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_fermi_partition_decreases_with_epsilon(self) -> Tuple[bool, str]:
        """
        Check that Z_F is a decreasing function of epsilon.
        Perturb: compute expected Z_F at epsilon and at epsilon*1.05.
        Verify that the baseline expected Z_F > perturbed expected Z_F.
        Also verify that the solution's Z_F is close to the baseline expected value (so the test is meaningful).
        """
        if self.epsilon is None or self.beta is None or self.Z_F is None:
            return (True, "SKIP: insufficient data")
        pert_factor = 1.05
        epsilon_high = self.epsilon * pert_factor
        # Use same beta
        Z_base = self._compute_expected_Z_F(self.epsilon, self.beta)
        Z_pert = self._compute_expected_Z_F(epsilon_high, self.beta)
        # Check solution matches baseline
        if not math.isclose(float(self.Z_F), Z_base, rel_tol=1e-2, abs_tol=1e-2):
            # If solution doesn't match, we can still check monotonicity using analytic values, but it's less coupled.
            # Return skip or fail? Let's return FAIL to indicate that the solution may be incorrect.
            return (False, "Solution's Z_F does not match expected baseline; cannot verify monotonicity")
        # Monotonicity: Z_base > Z_pert
        if Z_base <= Z_pert:
            return (False, f"Fermi partition function does not decrease with epsilon: Z_base={Z_base}, Z_pert={Z_pert}")
        return (True, "Fermi partition function decreases with increasing epsilon (positive Δε)")

    def test_monotonicity_bose_partition_decreases_with_epsilon(self) -> Tuple[bool, str]:
        """Same monotonicity test for Bose partition function."""
        if self.epsilon is None or self.beta is None or self.Z_B is None:
            return (True, "SKIP: insufficient data")
        pert_factor = 1.05
        epsilon_high = self.epsilon * pert_factor
        Z_base = self._compute_expected_Z_B(self.epsilon, self.beta)
        Z_pert = self._compute_expected_Z_B(epsilon_high, self.beta)
        if not math.isclose(float(self.Z_B), Z_base, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Solution's Z_B does not match expected baseline")
        if Z_base <= Z_pert:
            return (False, f"Bose partition function does not decrease with epsilon: Z_base={Z_base}, Z_pert={Z_pert}")
        return (True, "Bose partition function decreases with increasing epsilon")

    def test_monotonicity_boltzmann_partition_decreases_with_epsilon(self) -> Tuple[bool, str]:
        """Same monotonicity test for Boltzmann partition function."""
        if self.epsilon is None or self.beta is None or self.Z_Boltz is None:
            return (True, "SKIP: insufficient data")
        pert_factor = 1.05
        epsilon_high = self.epsilon * pert_factor
        Z_base = self._compute_expected_Z_Boltz(self.epsilon, self.beta)
        Z_pert = self._compute_expected_Z_Boltz(epsilon_high, self.beta)
        if not math.isclose(float(self.Z_Boltz), Z_base, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Solution's Z_Boltz does not match expected baseline")
        if Z_base <= Z_pert:
            return (False, f"Boltzmann partition function does not decrease with epsilon: Z_base={Z_base}, Z_pert={Z_pert}")
        return (True, "Boltzmann partition function decreases with increasing epsilon")

    # ------------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_avg_occupancy_range(self) -> Tuple[bool, str]:
        """Weak sanity: avg_occupancy should be a small positive number (<1)."""
        if self.avg_occupancy is None:
            return (True, "SKIP: avg_occupancy not provided")
        occ = float(self.avg_occupancy)
        if occ <= 0 or occ >= 1:
            return (False, f"avg_occupancy={occ} out of (0,1)")
        return (True, "avg_occupancy in reasonable range")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute every test_* method, collect results."""
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)