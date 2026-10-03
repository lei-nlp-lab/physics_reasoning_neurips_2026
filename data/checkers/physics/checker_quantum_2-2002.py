import math
import sympy as sp
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits a solution for the excitation energy of a quark in a cubical box.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed parameters and results from the solution instance.
        Use None defaults for missing values, no numeric fallbacks.
        """
        # Input givens: may be in solution_instance.givens dict or as direct attributes
        self.m_p = getattr(solution_instance, 'm_p', None)
        if self.m_p is None:
            givens = getattr(solution_instance, 'givens', {}) or {}
            self.m_p = givens.get('m_p', None)

        self.hbar_c = getattr(solution_instance, 'hbar_c', None)
        if self.hbar_c is None:
            givens = getattr(solution_instance, 'givens', {}) or {}
            self.hbar_c = givens.get('hbar_c', None)

        self.L = getattr(solution_instance, 'L', None)
        if self.L is None:
            givens = getattr(solution_instance, 'givens', {}) or {}
            self.L = givens.get('L', None)

        # Result: excitation energy delta_E
        self.delta_E = getattr(solution_instance, 'delta_E', None)
        if self.delta_E is None:
            results = getattr(solution_instance, 'results', {}) or {}
            self.delta_E = results.get('delta_E', None)

        # For convenience, compute derived constants if givens exist
        self.pi = math.pi

    # ------------------------------------------------------------------
    # Interface tests: check that required keys exist and are valid types
    # ------------------------------------------------------------------
    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that all necessary givens are provided."""
        missing = []
        if self.m_p is None:
            missing.append("m_p (mass of proton)")
        if self.hbar_c is None:
            missing.append("hbar_c")
        if self.L is None:
            missing.append("L (box side length)")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    def test_interface_result_delta_E_present(self) -> Tuple[bool, str]:
        """Check that the result key 'delta_E' exists and is numeric."""
        if self.delta_E is None:
            return (False, "Missing result: delta_E")
        if not isinstance(self.delta_E, (int, float)):
            return (False, f"delta_E has type {type(self.delta_E).__name__}, expected numeric")
        return (True, "Result delta_E present and numeric")

    # ------------------------------------------------------------------
    # Invariant / theorem-based sanity checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_energy_formula(self) -> Tuple[bool, str]:
        """
        Check that delta_E matches the theoretical formula:
        ΔE = 3 * π^2 * (hbar_c)^2 / (2 * (m_p/3) * L^2)
        Use the physical constants to re-derive the expected value.
        """
        if any(v is None for v in (self.m_p, self.hbar_c, self.L, self.delta_E)):
            return (False, "SKIP: missing required value for energy formula check")

        # Derive expected excitation energy
        m_quark_c2 = self.m_p / 3.0  # MeV
        E1 = (self.pi**2 * self.hbar_c**2) / (2.0 * m_quark_c2 * self.L**2)
        expected_delta_E = 3.0 * E1

        if math.isclose(self.delta_E, expected_delta_E, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_E {self.delta_E:.3f} MeV matches derived value {expected_delta_E:.3f} MeV")
        else:
            return (False, f"delta_E {self.delta_E:.3f} MeV differs from derived {expected_delta_E:.3f} MeV (rel>1%)")

    def test_invariant_excited_higher_than_ground(self) -> Tuple[bool, str]:
        """Check that the excitation energy is positive (E_2 > E_1)."""
        if self.delta_E is None:
            return (False, "SKIP: no delta_E")
        if self.delta_E > 0:
            return (True, "Excitation energy positive (E_2 > E_1)")
        return (False, f"Excitation energy {self.delta_E} is not positive")

    # ------------------------------------------------------------------
    # Modeling quality / trace checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_modeling_step_energy_unit(self) -> Tuple[bool, str]:
        """
        Verify that the intermediate 'E1' (unit energy) is computed correctly
        from the givens. This test doesn't require the pseudocode to have emitted E1,
        but we can compute it and compare with the derived delta_E / 3.
        """
        if any(v is None for v in (self.m_p, self.hbar_c, self.L, self.delta_E)):
            return (False, "SKIP: missing data to compute E1")
        m_quark_c2 = self.m_p / 3.0
        E1_from_givens = (self.pi**2 * self.hbar_c**2) / (2.0 * m_quark_c2 * self.L**2)
        E1_from_result = self.delta_E / 3.0

        if math.isclose(E1_from_givens, E1_from_result, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "E1 consistency holds")
        return (False, f"E1 from givens {E1_from_givens:.3f} vs from result {E1_from_result:.3f} disagree")

    def test_modeling_ground_state_energy_order(self) -> Tuple[bool, str]:
        """
        Check that the ground state energy E_111 = 3*E1 is a physically reasonable
        fraction of the rest mass energy of the quark.
        """
        if any(v is None for v in (self.m_p, self.delta_E)):
            return (False, "SKIP: missing data for ground state energy check")
        m_quark_c2 = self.m_p / 3.0
        E1 = self.delta_E / 3.0
        E_ground = 3.0 * E1   # = delta_E
        # Ground state should be much less than the rest mass energy (non-relativistic)
        if E_ground < 0.5 * m_quark_c2:
            return (True, f"Ground state {E_ground:.1f} MeV < rest mass {m_quark_c2:.1f} MeV (non-relativistic)")
        return (False, f"Ground state {E_ground:.1f} MeV too large compared to rest mass {m_quark_c2:.1f} MeV")

    # ------------------------------------------------------------------
    # Monotonicity / trend checks
    # ------------------------------------------------------------------
    def test_monotonicity_increase_L_decreases_delta_E(self) -> Tuple[bool, str]:
        """
        Perturb L by +10% and check that delta_E decreases (because E ~ 1/L^2).
        Simulate by recomputing with perturbed L.
        """
        if any(v is None for v in (self.m_p, self.hbar_c, self.L, self.delta_E)):
            return (False, "SKIP: missing data for monotonicity check")

        m_quark_c2 = self.m_p / 3.0
        L_perturbed = self.L * 1.10
        E1_pert = (self.pi**2 * self.hbar_c**2) / (2.0 * m_quark_c2 * L_perturbed**2)
        delta_E_pert = 3.0 * E1_pert

        if delta_E_pert < self.delta_E:
            return (True, f"Increasing L decreases ΔE ({delta_E_pert:.1f} < {self.delta_E:.1f})")
        return (False, f"Monotonicity violated: increasing L did not reduce ΔE")

    # ------------------------------------------------------------------
    # Weak answer sanity (no math.isclose, no re-derivation, no symbolic solving)
    # ------------------------------------------------------------------
    def test_answer_sanity_reasonable_range(self) -> Tuple[bool, str]:
        """
        Check that the excitation energy is positive, finite, and within a
        physically reasonable range (e.g., 100–1000 MeV for a quark in a 2 fm box).
        """
        if self.delta_E is None:
            return (False, "SKIP: no delta_E")
        if not math.isfinite(self.delta_E):
            return (False, f"delta_E is not finite: {self.delta_E}")
        if self.delta_E <= 0:
            return (False, f"delta_E not positive: {self.delta_E}")
        # Broad range based on typical strong-interaction scales
        if 100.0 < self.delta_E < 1000.0:
            return (True, f"delta_E = {self.delta_E:.1f} MeV is in [100, 1000] MeV range")
        return (False, f"delta_E = {self.delta_E:.1f} MeV outside expected range (100-1000 MeV)")

    # ------------------------------------------------------------------
    # Main runner (collects and runs all tests)
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, return overall success and logs."""
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
                passed = False
                msg = f"{name} CRASHED: {e}"
                import traceback
                msg += f"\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)