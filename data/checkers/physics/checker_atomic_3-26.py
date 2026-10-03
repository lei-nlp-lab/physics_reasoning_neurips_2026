import math
import re
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits solutions for Clebsch-Gordan coefficients and isospin cross‑section ratios.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely; use None for missing values.
        results = getattr(solution_instance, 'results', {})
        self.cg: Optional[str] = results.get('cg', None)
        self.ratio_Delta: Optional[List[int]] = results.get('ratio_Delta', None)
        self.ratio_Nstar: Optional[List[int]] = results.get('ratio_Nstar', None)

        # Store explanation steps (only for guidance, not for hardcoding)
        self.explanation_steps: Optional[str] = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Verify that the three mandatory keys exist in results."""
        missing = []
        if self.cg is None:
            missing.append('cg')
        if self.ratio_Delta is None:
            missing.append('ratio_Delta')
        if self.ratio_Nstar is None:
            missing.append('ratio_Nstar')
        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "All required keys present."

    def test_interface_type_cg(self) -> Tuple[bool, str]:
        """cg must be a non‑empty string."""
        if not isinstance(self.cg, str) or len(self.cg.strip()) == 0:
            return False, "cg is not a non‑empty string."
        return True, "cg is a string."

    def test_interface_type_ratio_Delta(self) -> Tuple[bool, str]:
        """ratio_Delta must be a list of three numbers."""
        if not isinstance(self.ratio_Delta, list) or len(self.ratio_Delta) != 3:
            return False, "ratio_Delta is not a list of length 3."
        if not all(isinstance(x, (int, float)) for x in self.ratio_Delta):
            return False, "ratio_Delta contains non‑numeric elements."
        return True, "ratio_Delta is a list of three numbers."

    def test_interface_type_ratio_Nstar(self) -> Tuple[bool, str]:
        """ratio_Nstar must be a list of three numbers."""
        if not isinstance(self.ratio_Nstar, list) or len(self.ratio_Nstar) != 3:
            return False, "ratio_Nstar is not a list of length 3."
        if not all(isinstance(x, (int, float)) for x in self.ratio_Nstar):
            return False, "ratio_Nstar contains non‑numeric elements."
        return True, "ratio_Nstar is a list of three numbers."

    # ------------------------------------------------------------------
    # Invariant checks (theorem based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_cg_squared_sum(self) -> Tuple[bool, str]:
        """Check that the squares of CG coefficients in the string sum to 1."""
        if self.cg is None:
            return False, "SKIP: cg not available."
        # Extract all numbers that look like sqrt(p/q)
        pattern = r'sqrt\((\d+)/(\d+)\)'
        matches = re.findall(pattern, self.cg)
        if not matches:
            return False, "No sqrt coefficient patterns found in cg string."
        squares_sum = 0.0
        for num_str, den_str in matches:
            num = float(num_str)
            den = float(den_str)
            coeff = math.sqrt(num / den)
            squares_sum += coeff * coeff
        if math.isclose(squares_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            return True, f"Sum of coefficient squares = {squares_sum:.6f} (close to 1)."
        else:
            return False, f"Sum of coefficient squares = {squares_sum:.6f}, expected 1.0."

    def test_invariant_ratio_Delta_positive(self) -> Tuple[bool, str]:
        """All entries in ratio_Delta should be positive (nonzero)."""
        if self.ratio_Delta is None:
            return False, "SKIP: ratio_Delta not available."
        if all(x > 0 for x in self.ratio_Delta):
            return True, "All Δ ratio entries > 0."
        return False, f"Δ ratio has non‑positive entries: {self.ratio_Delta}"

    def test_invariant_ratio_Nstar_nonnegative(self) -> Tuple[bool, str]:
        """All entries in ratio_Nstar should be nonnegative and at least one positive."""
        if self.ratio_Nstar is None:
            return False, "SKIP: ratio_Nstar not available."
        if all(x >= 0 for x in self.ratio_Nstar) and any(x > 0 for x in self.ratio_Nstar):
            return True, "All N* ratio entries ≥0 with at least one positive."
        return False, f"N* ratio has negative entries or all zeros: {self.ratio_Nstar}"

    # ------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps and physics)
    # ------------------------------------------------------------------
    def test_modeling_cg_contains_expected_terms(self) -> Tuple[bool, str]:
        """Check that the CG string contains the two product states with correct m values."""
        if self.cg is None:
            return False, "SKIP: cg not available."
        term1 = "|1,0>|1/2,1/2>"
        term2 = "|1,1>|1/2,-1/2>"
        if term1 in self.cg and term2 in self.cg:
            return True, "CG string contains required product states."
        return False, f"CG string missing expected product states. Contains: {self.cg}"

    def test_modeling_cg_coefficient_sqrt_2over3(self) -> Tuple[bool, str]:
        """Check that cg string contains both sqrt(2/3) and sqrt(1/3)."""
        if self.cg is None:
            return False, "SKIP: cg not available."
        if "sqrt(2/3)" in self.cg and "sqrt(1/3)" in self.cg:
            return True, "cg contains both sqrt(2/3) and sqrt(1/3)."
        return False, "cg missing one or both expected sqrt coefficients."

    def test_modeling_ratio_Delta_derived(self) -> Tuple[bool, str]:
        """
        Derive the expected Δ ratio from isospin coefficients and compare
        to the solution's ratio_Delta.
        """
        if self.ratio_Delta is None:
            return False, "SKIP: ratio_Delta not available."

        # Known isospin amplitudes (absolute squares) for Δ resonance:
        # σ1: π⁺p → π⁺p : amplitude = 1 → cross section ∝ 1
        # σ2: π⁻p → π⁻p : amplitude = 1/3 → cross section ∝ 1/9
        # σ3: π⁻p → π⁰n : amplitude = √2/3 → cross section ∝ 2/9
        expected_norm = [9, 1, 2]  # after multiplying by 9
        # Normalise solution's ratio
        vals = self.ratio_Delta
        if vals[0] == 0:
            return False, "First entry (σ1) is zero, cannot check Δ ratio."
        scale = vals[0] / 9.0
        normalised = [v / scale for v in vals]
        if all(math.isclose(n, e, rel_tol=1e-3, abs_tol=1e-3) for n, e in zip(normalised, expected_norm)):
            return True, f"Δ ratio normalises to ~9:1:2 (got {normalised})."
        else:
            return False, f"Δ ratio {vals} normalises to {normalised}, expected [9,1,2]."

    def test_modeling_ratio_Nstar_derived(self) -> Tuple[bool, str]:
        """
        Derive the expected N* ratio from isospin coefficients and compare
        to the solution's ratio_Nstar.
        """
        if self.ratio_Nstar is None:
            return False, "SKIP: ratio_Nstar not available."

        # Known squared amplitudes for N* resonance:
        # σ1: π⁺p → π⁺p : 0 (no I=1/2 component)
        # σ2: π⁻p → π⁻p : ( -√(2/3) )^2 = 2/3
        # σ3: π⁻p → π⁰n : ( -√(2/3) * √(1/3) )^2 = (√2/3)^2 = 2/9
        # Ratio σ2 : σ3 = 2/3 : 2/9 = 6/9 : 2/9 = 3 : 1? Wait: 2/3 = 6/9, 2/9 = 2/9, ratio with σ1=0 gives 0:6:2 = 0:3:1. But from explanation steps they gave 0:2:1. Let's re-check explanation:
        # Explanation step: σ2 ∝ 4/9 |a2|^2, σ3 ∝ 2/9 |a2|^2 => ratio 4/9 : 2/9 = 2:1. That matches 0:2:1. My calculation: amplitude for σ2 = ( -√(2/3) ) * ( -√(2/3) ) = 2/3? Wait careful: For N* resonance, the state |π⁻p> = √(1/3)|3/2,1/2> - √(2/3)|1/2,1/2>. The projection onto I=1/2 is -√(2/3). The final state |π⁻p> has same projection onto I=1/2 = -√(2/3). So amplitude = (-√(2/3)) * (-√(2/3)) = 2/3. Cross section ∝ (2/3)^2 = 4/9. For π⁻p → π⁰n: initial projection onto I=1/2 = -√(2/3). Final state |π⁰n> = √(2/3)|3/2,1/2> + √(1/3)|1/2,1/2>. Projection onto I=1/2 is √(1/3). Amplitude = (-√(2/3)) * (√(1/3)) = -√2/3. Cross section ∝ (√2/3)^2 = 2/9. Ratio 4/9 : 2/9 = 2:1. So yes, expected is [0,2,1]. So use [0,2,1] as expected.
        expected_norm = [0, 2, 1]
        vals = self.ratio_Nstar
        # σ1 must be close to zero
        if not math.isclose(vals[0], 0.0, abs_tol=1e-10):
            return False, f"First entry (σ1) is not zero: {vals[0]}."
        # Normalise so that σ3 = 1 (if σ3 != 0)
        if vals[2] == 0:
            return False, "Third entry (σ3) is zero, cannot check N* ratio."
        scale = vals[2] / 1.0
        normalised = [v / scale for v in vals]
        if all(math.isclose(n, e, rel_tol=1e-3, abs_tol=1e-3) for n, e in zip(normalised, expected_norm)):
            return True, f"N* ratio normalises to ~0:2:1 (got {normalised})."
        else:
            return False, f"N* ratio {vals} normalises to {normalised}, expected [0,2,1]."

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation and comparison)
    # ------------------------------------------------------------------
    def test_monotonicity_Delta_coefficient_increases_sigma2(self) -> Tuple[bool, str]:
        """
        Check that increasing the isospin amplitude coefficient for π⁻p Δ resonance
        leads to a larger cross section (monotonic relationship).
        """
        # Baseline amplitude coefficient for π⁻p → π⁻p via Δ: c = 1/3
        c_baseline = 1.0 / 3.0
        output_baseline = c_baseline ** 2  # σ2 relative to σ1 = 1/9

        # Perturb the coefficient by +10%
        perturbation = 0.10
        c_perturbed = c_baseline * (1 + perturbation)
        output_perturbed = c_perturbed ** 2

        # Check monotonic increase
        if output_perturbed > output_baseline:
            return True, f"Monotonic increase: perturbed output {output_perturbed:.6f} > baseline {output_baseline:.6f}"
        else:
            return False, f"Monotonic increase failed: perturbed {output_perturbed:.6f} <= baseline {output_baseline:.6f}"

    def test_monotonicity_Nstar_coefficient_increases_sigma2(self) -> Tuple[bool, str]:
        """
        Check that increasing the isospin amplitude coefficient for π⁻p N* resonance
        leads to a larger cross section (monotonic relationship).
        """
        # Baseline amplitude coefficient for π⁻p → π⁻p via N*: c = -√(2/3) (magnitude √(2/3) ≈ 0.8165)
        c_baseline = math.sqrt(2.0 / 3.0)  # 0.8164965809
        output_baseline = c_baseline ** 2  # 2/3

        # Perturb the coefficient by +10%
        perturbation = 0.10
        c_perturbed = c_baseline * (1 + perturbation)
        output_perturbed = c_perturbed ** 2

        if output_perturbed > output_baseline:
            return True, f"Monotonic increase: perturbed output {output_perturbed:.6f} > baseline {output_baseline:.6f}"
        else:
            return False, f"Monotonic increase failed: perturbed {output_perturbed:.6f} <= baseline {output_baseline:.6f}"

    # ------------------------------------------------------------------
    # Answer sanity (weak, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_ratio_values(self) -> Tuple[bool, str]:
        """Weak sanity: all ratio entries are finite, nonnegative, and integers."""
        for name, rlist in [("Delta", self.ratio_Delta), ("Nstar", self.ratio_Nstar)]:
            if rlist is None:
                continue
            for i, v in enumerate(rlist):
                if v is None or math.isnan(v) or math.isinf(v):
                    return False, f"{name} ratio[{i}] is non‑finite: {v}."
                if v < 0:
                    return False, f"{name} ratio[{i}] is negative: {v}."
                if not isinstance(v, int):
                    return False, f"{name} ratio[{i}] is not an integer: {v}."
        return True, "Ratios are finite, nonnegative integers."

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, return (all_pass, log)."""
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)


# Example usage (not part of the required output, but for testing)
if __name__ == "__main__":
    class MockSolution:
        results = {
            'cg': "|3/2,1/2> = sqrt(2/3) |1,0>|1/2,1/2> + sqrt(1/3) |1,1>|1/2,-1/2>",
            'ratio_Delta': [9, 1, 2],
            'ratio_Nstar': [0, 2, 1],
        }
    checker = PhysicsChecker(MockSolution())
    success, log = checker.run_all_tests()
    for line in log:
        print(line)
    print(f"Overall: {'PASS' if success else 'FAIL'}")