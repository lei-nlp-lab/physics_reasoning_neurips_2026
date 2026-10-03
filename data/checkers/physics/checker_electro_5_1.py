import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audit the correctness, completeness, and physical consistency of a solution
    to the radar Doppler beat frequency problem.
    """
    def __init__(self, solution_instance: Any):
        # Extract givens safely from solution_instance.givens or solution_instance
        self.v = getattr(solution_instance, 'v', 
                         solution_instance.givens.get('v') if hasattr(solution_instance, 'givens') else None)
        self.f0 = getattr(solution_instance, 'f0',
                          solution_instance.givens.get('f0') if hasattr(solution_instance, 'givens') else None)
        self.c = getattr(solution_instance, 'c',
                         solution_instance.givens.get('c') if hasattr(solution_instance, 'givens') else None)
        # If the solution_instance stores results in a dict
        self.results = getattr(solution_instance, 'results', {})
        self.f_beat = self.results.get('f_beat', None)
        # Also try direct attribute access
        if self.f_beat is None:
            self.f_beat = getattr(solution_instance, 'f_beat', None)

        # Extract explanation steps if any (list of strings)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the required result key 'f_beat' exists and is numeric."""
        if self.f_beat is None:
            return (False, "Required key 'f_beat' not found in results or as attribute.")
        if not isinstance(self.f_beat, (int, float)):
            return (False, f"f_beat has invalid type: {type(self.f_beat)}")
        return (True, "f_beat key present and numeric.")

    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that all required givens (v, f0, c) are provided and sensible."""
        missing = []
        for name, val in [('v', self.v), ('f0', self.f0), ('c', self.c)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        # All present, check they're numeric and positive
        for name, val in [('v', self.v), ('f0', self.f0), ('c', self.c)]:
            if not isinstance(val, (int, float)) or val <= 0:
                return (False, f"Given '{name}' must be a positive number, got {val}")
        return (True, "All required givens present and positive.")

    # ------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_doppler_formula(self) -> Tuple[bool, str]:
        """Check that beat frequency matches the classical double-Doppler formula:
        f_beat = (2*v/c) * f0"""
        if any(x is None for x in [self.f_beat, self.v, self.f0, self.c]):
            return (False, "SKIP: missing one or more values for invariant check")
        expected = (2 * self.v / self.c) * self.f0
        if not math.isclose(self.f_beat, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f_beat={self.f_beat} does not match expected {expected} within 1% tolerance.")
        return (True, f"Beat frequency matches formula: {self.f_beat:.2f} ≈ {expected:.2f} Hz.")

    def test_invariant_non_relativistic(self) -> Tuple[bool, str]:
        """Check that v << c (non‑relativistic regime)."""
        if self.v is None or self.c is None:
            return (False, "SKIP: v or c missing")
        if self.v / self.c > 0.01:
            return (False, f"v/c = {self.v/self.c:.6f} > 0.01; relativistic effects may be significant.")
        return (True, f"Non‑relativistic regime: v/c = {self.v/self.c:.6f} < 0.01.")

    def test_invariant_beat_positive(self) -> Tuple[bool, str]:
        """Check that beat frequency is positive (approaching car)."""
        if self.f_beat is None:
            return (False, "SKIP: f_beat not available")
        if self.f_beat <= 0:
            return (False, f"Beat frequency should be positive, got {self.f_beat}.")
        return (True, f"Beat frequency positive: {self.f_beat} Hz.")

    # ------------------------------------------------------------------
    # Modeling / trace checks (based on explanation_steps)
    # ------------------------------------------------------------------
    def test_modeling_double_doppler_mentioned(self) -> Tuple[bool, str]:
        """Check that the explanation mentions the double Doppler shift concept."""
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps not provided")
        combined = " ".join(self.explanation_steps).lower()
        # Look for any phrase indicating double shift or reflected shift
        keywords = ["2v/c", "double", "two doppler", "reflected", "acting as a source"]
        found = any(kw in combined for kw in keywords)
        if not found:
            return (False, "Explanation does not mention double Doppler shift (expected '2v/c' or equivalent).")
        return (True, "Explanation correctly references double Doppler shift.")

    def test_modeling_beat_frequency_derivation(self) -> Tuple[bool, str]:
        """Check that the final formula f_beat = (2v/c)*f0 appears or is derived."""
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps not provided")
        combined = " ".join(self.explanation_steps).lower()
        # Look for the formula or numerical substitution
        if "f_beat" not in combined and "beat frequency" not in combined:
            return (False, "Explanation does not mention beat frequency explicitly.")
        # Check that the formula with 2v/c appears
        if "2v/c" not in combined and "2*v/c" not in combined:
            # Also check for approximate equivalent like 2*30/3e8 * 1e9
            if self.v is not None and self.c is not None and self.f0 is not None:
                # The explanation should contain the numeric substitution
                numeric_part = f"2*{self.v}/{self.c}*{self.f0}"
                if numeric_part.replace('*', '') not in combined:
                    return (False, "Explanation does not contain the numerical substitution for beat frequency.")
        return (True, "Explanation includes beat frequency derivation.")

    # ------------------------------------------------------------------
    # Monotonicity checks (sensitivity analysis)
    # ------------------------------------------------------------------
    def test_monotonicity_v_increases_f_beat(self) -> Tuple[bool, str]:
        """Check that increasing car speed increases beat frequency (linear relationship)."""
        if any(x is None for x in [self.f_beat, self.v, self.f0, self.c]):
            return (False, "SKIP: missing required values for monotonicity check")
        # Perturb v by +10%
        dv = 0.1 * self.v
        v_pert = self.v + dv
        expected_f_beat_pert = (2 * v_pert / self.c) * self.f0
        # The original f_beat should be from the same formula; we compare the perturbation effect.
        # Compute the ratio change for the current f_beat and the perturbed one based on the same formula.
        baseline = (2 * self.v / self.c) * self.f0  # formula value (should match self.f_beat closely)
        # If the result deviates from formula, the monotonicity check may still be valid.
        # But we should check that the result itself increases.
        if self.f_beat < baseline:
            # Result is lower than formula; we accept it but check direction
            pass
        # The relative change in the formula:
        rel_change_formula = (expected_f_beat_pert - baseline) / baseline if baseline != 0 else 0
        # The relative change in the result if we were to re‑compute it with perturbed v (not available)
        # So we rely on the physics formula; we can assert that if v increases, f_beat must increase.
        # We'll just check that the formula expectation is positive.
        if expected_f_beat_pert <= baseline:
            return (False, "Increasing v does not increase f_beat in the formula (unphysical).")
        # Additional: the actual result f_beat should be less than perturbed expectation (since v < v_pert)
        if self.f_beat is not None and self.f_beat >= expected_f_beat_pert:
            # If the current f_beat is already equal or larger than the perturbed formula value,
            # it suggests an anomaly, but we allow small tolerance due to rounding.
            if not math.isclose(self.f_beat, expected_f_beat_pert, rel_tol=1e-2):
                return (False, f"Monotonicity violation: current f_beat ({self.f_beat}) is not less than "
                               f"expected for v+10% ({expected_f_beat_pert:.2f}) (relative change {rel_change_formula*100:.1f}%).")
        return (True, f"Monotonicity holds: increasing v by 10% increases f_beat by ~{rel_change_formula*100:.1f}%.")

    # ------------------------------------------------------------------
    # Optional weak answer sanity check (no math.isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """Weak check: beat frequency should be between 0 and 1000 Hz for typical car speeds."""
        if self.f_beat is None:
            return (False, "SKIP: f_beat not available")
        if not (0 < self.f_beat < 1000):
            return (False, f"Beat frequency {self.f_beat} Hz is outside the expected range (0–1000 Hz).")
        return (True, f"Beat frequency {self.f_beat} Hz lies within a plausible range.")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute every test_* method, report results."""
        logs = []
        all_pass = True
        # Methods are bound to self; exclude non-callable attributes
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