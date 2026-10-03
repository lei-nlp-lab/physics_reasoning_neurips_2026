import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    """
    Auditor for a fixed-target proton-proton collision problem.
    Checks relativistic invariant mass relation, modeling quality, and monotonicity.
    """
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.E_proj = givens.get('E_proj', None)          # projectile energy in GeV
        self.m_p = givens.get('m_p', None)                # proton rest mass (GeV), if provided

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.E_free = results.get('E_free', None)         # computed free energy in GeV

    # ------------------------------------------------------------
    # Interface tests: required keys and types
    # ------------------------------------------------------------
    def test_interface_E_free_exists(self) -> Tuple[bool, str]:
        if self.E_free is None:
            return False, "SKIP: 'E_free' not present in results"
        if not isinstance(self.E_free, (int, float)):
            return False, f"E_free is not a number: {type(self.E_free)}"
        return True, "E_free exists and is numeric"

    def test_interface_E_proj_exists(self) -> Tuple[bool, str]:
        if self.E_proj is None:
            return False, "SKIP: 'E_proj' not present in givens"
        if not isinstance(self.E_proj, (int, float)):
            return False, f"E_proj is not a number: {type(self.E_proj)}"
        return True, "E_proj exists and is numeric"

    # ------------------------------------------------------------
    # Invariant tests: theorem-based physical consistency (at least two)
    # ------------------------------------------------------------
    def test_invariant_free_energy_relation(self) -> Tuple[bool, str]:
        """Check E_free = sqrt(2*m_p*(E_proj+m_p)) - 2*m_p (fixed-target invariant)"""
        if self.E_proj is None or self.m_p is None or self.E_free is None:
            return False, "SKIP: missing E_proj, m_p, or E_free"
        # Compute expected free energy from given inputs
        expected = math.sqrt(2 * self.m_p * (self.E_proj + self.m_p)) - 2 * self.m_p
        if math.isclose(self.E_free, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_free = {self.E_free:.4f} GeV matches invariant relation ({expected:.4f} GeV)"
        else:
            return False, f"E_free = {self.E_free:.4f} GeV, expected ≈ {expected:.4f} GeV"

    def test_invariant_invariant_mass_equivalence(self) -> Tuple[bool, str]:
        """Check that s from the simplified formula equals s from the full Lorentz invariant definition."""
        if self.E_proj is None or self.m_p is None:
            return False, "SKIP: missing E_proj or m_p"
        # Compute s using the two equivalent formulas
        s_simple = 2 * self.m_p * (self.E_proj + self.m_p)
        # Full invariant: s = (E_proj + m_p)^2 - p_proj^2, where p_proj = sqrt(E_proj^2 - m_p^2)
        try:
            p_proj = math.sqrt(self.E_proj**2 - self.m_p**2)
        except ValueError:
            return False, "Cannot compute p_proj (E_proj < m_p?)"
        s_full = (self.E_proj + self.m_p)**2 - p_proj**2
        if math.isclose(s_simple, s_full, rel_tol=1e-6, abs_tol=1e-6):
            return True, f"Invariant mass s computed two ways agree: s = {s_simple:.6f} GeV^2"
        else:
            return False, f"s_simple = {s_simple:.6f} GeV^2, s_full = {s_full:.6f} GeV^2 differ"

    # ------------------------------------------------------------
    # Modeling-quality tests: physical reasonableness
    # ------------------------------------------------------------
    def test_modeling_free_energy_positive(self) -> Tuple[bool, str]:
        if self.E_free is None:
            return False, "SKIP: E_free not available"
        if self.E_free <= 0:
            return False, f"E_free should be positive, got {self.E_free:.4f} GeV"
        return True, "E_free is positive"

    def test_modeling_free_energy_less_than_E_proj(self) -> Tuple[bool, str]:
        """Free energy in fixed target is much less than projectile energy"""
        if self.E_free is None or self.E_proj is None:
            return False, "SKIP: E_free or E_proj missing"
        if self.E_free >= self.E_proj:
            return False, f"E_free ({self.E_free:.4f}) should be < E_proj ({self.E_proj:.4f}) GeV"
        return True, "E_free is less than projectile energy"

    def test_modeling_free_energy_less_than_total_cm_energy(self) -> Tuple[bool, str]:
        """E_free = sqrt(s) - 2*m_p < sqrt(s)"""
        if self.E_free is None or self.E_proj is None or self.m_p is None:
            return False, "SKIP: missing data"
        sqrt_s = math.sqrt(2 * self.m_p * (self.E_proj + self.m_p))
        if self.E_free >= sqrt_s:
            return False, f"E_free ({self.E_free:.4f}) should be < sqrt(s) = {sqrt_s:.4f} GeV"
        return True, "E_free is less than total CM energy"

    # ------------------------------------------------------------
    # Monotonicity test: response to input changes
    # ------------------------------------------------------------
    def test_monotonicity_free_energy_increases_with_E_proj(self) -> Tuple[bool, str]:
        """If E_proj increases by 5%, the expected free energy (invariant relation) increases."""
        if self.E_proj is None or self.m_p is None:
            return False, "SKIP: missing E_proj or m_p"
        # Compute baseline expected free energy using the invariant formula
        baseline_expected = math.sqrt(2 * self.m_p * (self.E_proj + self.m_p)) - 2 * self.m_p
        # Perturb E_proj by +5%
        E_proj_pert = self.E_proj * 1.05
        perturbed_expected = math.sqrt(2 * self.m_p * (E_proj_pert + self.m_p)) - 2 * self.m_p
        if perturbed_expected <= baseline_expected:
            return False, f"Expected free energy should increase when E_proj increases, but perturbed {perturbed_expected:.4f} <= baseline {baseline_expected:.4f}"
        return True, "Monotonicity holds: free energy increases with projectile energy"

    # ------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # ------------------------------------------------------------
    def test_answer_sanity_magnitude(self) -> Tuple[bool, str]:
        """Simple sanity: E_free should be in a plausible range (10–100 GeV) for 1000 GeV proton."""
        if self.E_free is None:
            return False, "SKIP: E_free not available"
        if self.E_free < 10 or self.E_free > 100:
            return False, f"E_free = {self.E_free:.2f} GeV is outside plausible range [10, 100] GeV"
        return True, f"E_free magnitude {self.E_free:.2f} GeV is plausible"

    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Discover all methods starting with 'test_'
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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)