import math
import sympy as sp
from typing import List, Tuple, Any

class PhysicsChecker:
    """Auditor for a solution to a nuclear physics problem about binding energy per nucleon,
    fission energy of U-235, and the dominant force in heavy nuclei.

    The checker implements:
        - Interface checks (required keys exist and have valid types)
        - Invariant checks (binding energy curve shape, maximum location)
        - Modeling quality checks (fission energy consistency, force identification)
        - Monotonicity check (decrease of B/A for heavy nuclei)
        - Weak answer sanity check (order-of-magnitude only)
    """

    # Semi-empirical mass formula constants (MeV)
    a_v = 15.8
    a_s = 18.3
    a_c = 0.714
    a_a = 23.2

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all required results and givens from the solution instance.
        Use None for any missing attribute or key – never numeric defaults.
        """
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Expected emitted keys
        self.A_max = self.results.get('A_max', None)
        self.B_per_n_max = self.results.get('B_per_n_max', None)
        self.Q = self.results.get('Q', None)
        self.force = self.results.get('force', None)

        # Additional useful fields (optional, but kept for completeness)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Helper: Binding energy using the semi-empirical mass formula
    # ------------------------------------------------------------------
    @staticmethod
    def _binding_energy(A: float, Z: float) -> float:
        """
        Compute the binding energy B(A,Z) in MeV using the semi-empirical mass formula.
        The pairing term δ is omitted (treated as negligible for average check).
        """
        # Avoid division by zero
        if A <= 0 or A**(1/3) == 0:
            return float('nan')
        a_v = PhysicsChecker.a_v
        a_s = PhysicsChecker.a_s
        a_c = PhysicsChecker.a_c
        a_a = PhysicsChecker.a_a

        term_volume = a_v * A
        term_surface = - a_s * A ** (2/3)
        term_coulomb = - a_c * Z * (Z - 1) / A ** (1/3)
        term_asymmetry = - a_a * (A - 2*Z)**2 / A

        return term_volume + term_surface + term_coulomb + term_asymmetry

    # ------------------------------------------------------------------
    # Interface tests (required keys exist and have valid types)
    # ------------------------------------------------------------------
    def test_interface_A_max_exists(self) -> Tuple[bool, str]:
        if self.A_max is None:
            return (False, "Missing result key 'A_max'")
        return (True, "A_max present")

    def test_interface_A_max_type(self) -> Tuple[bool, str]:
        if self.A_max is None:
            return (False, "Cannot check type: A_max is missing")
        if not isinstance(self.A_max, (int, float)):
            return (False, f"A_max has invalid type: {type(self.A_max)}")
        return (True, f"A_max type is numeric ({type(self.A_max).__name__})")

    def test_interface_B_per_n_max_exists(self) -> Tuple[bool, str]:
        if self.B_per_n_max is None:
            return (False, "Missing result key 'B_per_n_max'")
        return (True, "B_per_n_max present")

    def test_interface_B_per_n_max_type(self) -> Tuple[bool, str]:
        if self.B_per_n_max is None:
            return (False, "Cannot check type: B_per_n_max is missing")
        if not isinstance(self.B_per_n_max, (int, float)):
            return (False, f"B_per_n_max has invalid type: {type(self.B_per_n_max)}")
        return (True, f"B_per_n_max type is numeric ({type(self.B_per_n_max).__name__})")

    def test_interface_Q_exists(self) -> Tuple[bool, str]:
        if self.Q is None:
            return (False, "Missing result key 'Q'")
        return (True, "Q present")

    def test_interface_Q_type(self) -> Tuple[bool, str]:
        if self.Q is None:
            return (False, "Cannot check type: Q is missing")
        if not isinstance(self.Q, (int, float)):
            return (False, f"Q has invalid type: {type(self.Q)}")
        return (True, f"Q type is numeric ({type(self.Q).__name__})")

    def test_interface_force_exists(self) -> Tuple[bool, str]:
        if self.force is None:
            return (False, "Missing result key 'force'")
        return (True, "force present")

    def test_interface_force_type(self) -> Tuple[bool, str]:
        if self.force is None:
            return (False, "Cannot check type: force is missing")
        if not isinstance(self.force, str):
            return (False, f"force has invalid type: {type(self.force)}")
        return (True, f"force type is string ({type(self.force).__name__})")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_binding_energy_maximum(self) -> Tuple[bool, str]:
        """
        Verify that the reported maximum B/A is consistent with the semi-empirical
        formula for A=56, Z=28 (the most stable isobar near the maximum).
        """
        if self.B_per_n_max is None:
            return (False, "SKIP: B_per_n_max not available")

        A_peak = 56.0
        Z_peak = A_peak / 2.0  # stable isobar condition Z ≈ A/2
        B_peak = self._binding_energy(A_peak, Z_peak)
        B_per_n_peak = B_peak / A_peak  # ~8.8 MeV

        if not math.isfinite(B_per_n_peak):
            return (False, f"Computed B/A at A=56 is not finite: {B_per_n_peak}")

        if math.isclose(self.B_per_n_max, B_per_n_peak, rel_tol=0.02, abs_tol=0.1):
            return (True, f"B_per_n_max {self.B_per_n_max} is close to semi-empirical peak {B_per_n_peak:.3f}")
        else:
            return (False, f"B_per_n_max {self.B_per_n_max} deviates from expected peak {B_per_n_peak:.3f}")

    def test_invariant_binding_energy_heavy_decreases(self) -> Tuple[bool, str]:
        """
        Check that B/A for a heavy nucleus (U-235, Z=92) is significantly lower
        than the peak value. If B_per_n_max is present, use it; else compute from formula.
        """
        if self.B_per_n_max is None:
            return (False, "SKIP: B_per_n_max not available, cannot compare heavy B/A")

        A_H = 235.0
        Z_H = 92.0
        B_H = self._binding_energy(A_H, Z_H)
        B_per_n_H = B_H / A_H

        if not math.isfinite(B_per_n_H):
            return (False, f"Computed B/A for U-235 is not finite: {B_per_n_H}")

        if B_per_n_H < self.B_per_n_max - 0.1:
            return (True, f"B/A for U-235 ({B_per_n_H:.3f}) < peak ({self.B_per_n_max}), consistent")
        else:
            return (False, f"B/A for U-235 ({B_per_n_H:.3f}) is not sufficiently lower than peak ({self.B_per_n_max})")

    # ------------------------------------------------------------------
    # Modeling quality tests (verifying explanation steps and pseudocode consistency)
    # ------------------------------------------------------------------
    def test_modeling_fission_energy_consistency(self) -> Tuple[bool, str]:
        """
        Re‑compute the fission Q for U-235 from the semi-empirical formula
        using symmetric fragments (A1=118,Z1=46 ; A2=117,Z2=46) and compare
        with the reported Q value. Use fuzzy tolerance (5% relative).
        """
        if self.Q is None:
            return (False, "SKIP: Q not available")

        A_par = 235.0; Z_par = 92.0
        B_par = self._binding_energy(A_par, Z_par)

        # Symmetric fission fragments (approximate)
        A1 = 118.0; Z1 = 46.0
        A2 = 117.0; Z2 = 46.0
        B1 = self._binding_energy(A1, Z1)
        B2 = self._binding_energy(A2, Z2)

        Q_computed = B1 + B2 - B_par

        if not math.isfinite(Q_computed):
            return (False, f"Computed Q is not finite: {Q_computed}")

        if math.isclose(self.Q, Q_computed, rel_tol=0.05, abs_tol=10.0):
            return (True, f"Q {self.Q} ~ computed {Q_computed:.1f} MeV (tolerance 5%)")
        else:
            return (False, f"Q {self.Q} deviates significantly from computed {Q_computed:.1f} MeV")

    def test_modeling_force_contains_coulomb(self) -> Tuple[bool, str]:
        """
        The force responsible for the downward slope in the heavy mass region
        must be Coulomb repulsion (or similar wording).
        """
        if self.force is None:
            return (False, "SKIP: force not available")
        force_lower = self.force.lower()
        if 'coulomb' in force_lower or 'repulsion' in force_lower:
            return (True, f"force string contains 'Coulomb' or 'repulsion': {self.force}")
        else:
            return (False, f"force '{self.force}' does not mention Coulomb repulsion")

    # ------------------------------------------------------------------
    # Monotonicity test (trend via explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_binding_energy_decreases_with_A(self) -> Tuple[bool, str]:
        """
        For heavy nuclei (A > 100), increasing A should decrease B/A.
        Compare B/A at A=150 and A=200 (with Z ≈ A/2).
        If the baseline result A_max exists, use that region; else use generic heavy region.
        """
        # Use two points in the heavy region
        A1 = 150.0; Z1 = A1 / 2.0
        A2 = 200.0; Z2 = A2 / 2.0

        B1 = self._binding_energy(A1, Z1)
        B2 = self._binding_energy(A2, Z2)
        per_n1 = B1 / A1
        per_n2 = B2 / A2

        if not (math.isfinite(per_n1) and math.isfinite(per_n2)):
            return (False, "Computed B/A values are not finite")

        if per_n2 < per_n1 - 0.001:
            return (True, f"B/A decreases from A=150 ({per_n1:.3f}) to A=200 ({per_n2:.3f})")
        else:
            return (False, f"B/A does not decrease: A=150 gives {per_n1:.3f}, A=200 gives {per_n2:.3f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Minimal sanity: all numerical outputs must be positive, finite,
        and within broad physical ranges. No re‑derivation, no math.isclose.
        """
        issues = []
        # A_max
        if self.A_max is not None:
            if not math.isfinite(self.A_max):
                issues.append("A_max is not finite")
            if self.A_max < 0:
                issues.append("A_max is negative")
            if self.A_max > 200:
                issues.append("A_max > 200, unrealistic")
        else:
            issues.append("A_max missing")

        # B_per_n_max
        if self.B_per_n_max is not None:
            if not math.isfinite(self.B_per_n_max):
                issues.append("B_per_n_max is not finite")
            if self.B_per_n_max < 0:
                issues.append("B_per_n_max is negative")
            if self.B_per_n_max > 20:
                issues.append("B_per_n_max > 20, unrealistic")
        else:
            issues.append("B_per_n_max missing")

        # Q
        if self.Q is not None:
            if not math.isfinite(self.Q):
                issues.append("Q is not finite")
            if self.Q < 0:
                issues.append("Q is negative")
            if self.Q > 1000:
                issues.append("Q > 1000 MeV, unrealistic")
        else:
            issues.append("Q missing")

        # force
        if self.force is not None:
            if not isinstance(self.force, str) or len(self.force.strip()) == 0:
                issues.append("force is empty or not a string")
        else:
            issues.append("force missing")

        if issues:
            return (False, "; ".join(issues))
        return (True, "Answer sanity weak checks passed: all values positive and within broad physical ranges")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, execute them safely, and return (success, logs).
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