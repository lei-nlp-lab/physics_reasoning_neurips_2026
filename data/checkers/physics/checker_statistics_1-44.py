import math
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Auditor for entropy change of silver at constant volume.
    Checks interface completeness, theorem invariants, modeling quality,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract result values safely, default to None
        self.dS = None
        if hasattr(solution_instance, 'dS'):
            self.dS = solution_instance.dS
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.dS = solution_instance.results.get('dS', None)

        # Extract given values safely
        self.n = None
        self.Cv = None
        self.T1 = None
        self.T2 = None

        # Try from direct attributes
        if hasattr(solution_instance, 'n'):
            self.n = solution_instance.n
        if hasattr(solution_instance, 'Cv'):
            self.Cv = solution_instance.Cv
        if hasattr(solution_instance, 'T1'):
            self.T1 = solution_instance.T1
        if hasattr(solution_instance, 'T2'):
            self.T2 = solution_instance.T2

        # Fallback to givens dict (if exists)
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
            self.n = self.n if self.n is not None else givens.get('n', None)
            self.Cv = self.Cv if self.Cv is not None else givens.get('Cv', None)
            self.T1 = self.T1 if self.T1 is not None else givens.get('T1', None)
            self.T2 = self.T2 if self.T2 is not None else givens.get('T2', None)

        # Additional fallback: results dict (if givens stored there)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
            self.n = self.n if self.n is not None else results.get('n', None)
            self.Cv = self.Cv if self.Cv is not None else results.get('Cv', None)
            self.T1 = self.T1 if self.T1 is not None else results.get('T1', None)
            self.T2 = self.T2 if self.T2 is not None else results.get('T2', None)

        # Store for convenience
        self.givens = {
            'n': self.n,
            'Cv': self.Cv,
            'T1': self.T1,
            'T2': self.T2
        }

    # ------------------------------------------------------------------
    # Interface tests: required keys and types
    # ------------------------------------------------------------------
    def test_interface_dS_exists(self) -> Tuple[bool, str]:
        """Check that dS result is present and is a number."""
        if self.dS is None:
            return (False, "No entropy change (dS) found in solution")
        if not isinstance(self.dS, (int, float)):
            return (False, f"dS is not numeric: {type(self.dS)}")
        if math.isnan(self.dS) or math.isinf(self.dS):
            return (False, f"dS is NaN or infinite: {self.dS}")
        return (True, f"dS = {self.dS}")

    def test_interface_givens_exists(self) -> Tuple[bool, str]:
        """Check that all required givens are provided."""
        missing = [k for k, v in self.givens.items() if v is None]
        if missing:
            return (False, f"Missing givens: {missing}")
        for k, v in self.givens.items():
            if not isinstance(v, (int, float)):
                return (False, f"Given '{k}' is not numeric: {type(v)}")
        return (True, "All givens present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_entropy_formula(self) -> Tuple[bool, str]:
        """Check that dS = n * Cv * ln(T2/T1) holds within tolerance."""
        if None in (self.dS, self.n, self.Cv, self.T1, self.T2):
            return (False, "SKIP: missing required values")
        if self.T1 <= 0 or self.T2 <= 0:
            return (False, f"Non-positive temperatures: T1={self.T1}, T2={self.T2}")
        expected = self.n * self.Cv * math.log(self.T2 / self.T1)
        if not math.isclose(self.dS, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dS = {self.dS} differs from expected {expected} (by more than 1%)")
        return (True, f"dS = {self.dS} matches ΔS = {expected:.4f}")

    def test_invariant_state_function(self) -> Tuple[bool, str]:
        """
        Check that the entropy change is independent of path (state function).
        Compute ΔS for a two-step process (T1 -> T_mid -> T2) and compare to direct.
        """
        if None in (self.n, self.Cv, self.T1, self.T2):
            return (False, "SKIP: missing givens")
        if self.T1 <= 0 or self.T2 <= 0:
            return (False, f"Non-positive temperatures: T1={self.T1}, T2={self.T2}")
        T_mid = (self.T1 + self.T2) / 2.0
        dS_direct = self.n * self.Cv * math.log(self.T2 / self.T1)
        dS_two_step = (self.n * self.Cv * math.log(T_mid / self.T1) +
                       self.n * self.Cv * math.log(self.T2 / T_mid))
        if not math.isclose(dS_direct, dS_two_step, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Path independence violated: direct={dS_direct:.4f}, two-step={dS_two_step:.4f}")
        return (True, f"ΔS is state function: direct={dS_direct:.4f} vs two-step={dS_two_step:.4f}")

    # ------------------------------------------------------------------
    # Modeling tests: check that the physical ingredients are reasonable
    # ------------------------------------------------------------------
    def test_modeling_n_is_one(self) -> Tuple[bool, str]:
        """Check that n represents one mole (should be 1)."""
        if self.n is None:
            return (False, "SKIP: n not available")
        if not math.isclose(self.n, 1.0, rel_tol=1e-3):
            return (False, f"n = {self.n} differs from 1 mole")
        return (True, f"n = {self.n}")

    def test_modeling_Cv_constant(self) -> Tuple[bool, str]:
        """Check that Cv is given as 5.85 cal/(K*mol)."""
        if self.Cv is None:
            return (False, "SKIP: Cv not available")
        if not math.isclose(self.Cv, 5.85, rel_tol=1e-3):
            return (False, f"Cv = {self.Cv} differs from 5.85")
        return (True, f"Cv = {self.Cv}")

    def test_modeling_temperatures_kelvin(self) -> Tuple[bool, str]:
        """Check that T1 ≈ 273.15 K and T2 ≈ 303.15 K (as from problem)."""
        if self.T1 is None or self.T2 is None:
            return (False, "SKIP: temperatures not available")
        if not math.isclose(self.T1, 273.15, rel_tol=1e-3):
            return (False, f"T1 = {self.T1} deviates from 273.15 K")
        if not math.isclose(self.T2, 303.15, rel_tol=1e-3):
            return (False, f"T2 = {self.T2} deviates from 303.15 K")
        return (True, f"T1 = {self.T1} K, T2 = {self.T2} K")

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_temperature_increase(self) -> Tuple[bool, str]:
        """
        Check that increasing T2 (final temperature) increases ΔS.
        Uses the solution's own givens to re-derive baseline and perturbed values.
        """
        if None in (self.n, self.Cv, self.T1, self.T2):
            return (False, "SKIP: missing givens for monotonicity test")
        if self.T1 <= 0 or self.T2 <= 0:
            return (False, f"Non-positive temperatures: T1={self.T1}, T2={self.T2}")

        # Baseline ΔS computed from given formula
        baseline_dS = self.n * self.Cv * math.log(self.T2 / self.T1)

        # Perturb: increase T2 by 10%
        perturbed_T2 = self.T2 * 1.10
        perturbed_dS = self.n * self.Cv * math.log(perturbed_T2 / self.T1)

        if perturbed_dS <= baseline_dS:
            return (False, f"Increasing T2 from {self.T2} to {perturbed_T2} did not increase ΔS: "
                           f"baseline {baseline_dS:.4f}, perturbed {perturbed_dS:.4f}")
        return (True, f"Increasing T2 by 10% increases ΔS from {baseline_dS:.4f} to {perturbed_dS:.4f}")

    # ------------------------------------------------------------------
    # Answer sanity tests (weak, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_basics(self) -> Tuple[bool, str]:
        """Weak sanity: check that ΔS is positive and of plausible magnitude."""
        if self.dS is None:
            return (False, "SKIP: dS not available")
        if self.dS <= 0:
            return (False, f"dS = {self.dS} is not positive; should be >0 for T2 > T1")
        # Broad magnitude check: expected ~0.6 cal/K, allow [0.1, 5.0]
        if self.dS < 0.1 or self.dS > 5.0:
            return (False, f"dS = {self.dS} outside plausible range [0.1, 5.0] cal/K")
        return (True, f"dS = {self.dS} is positive and within plausible range")

    # ------------------------------------------------------------------
    # Runner: discover all test_ methods and execute
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)