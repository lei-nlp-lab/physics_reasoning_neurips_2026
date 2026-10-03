import math
import traceback

class PhysicsChecker:
    """
    Audits a solution_instance for the Stern-Gerlach and Franck-Hertz experiments.
    Expected result keys: SG_date, FH_date, FH_threshold_E.
    """

    def __init__(self, solution_instance):
        # Safely extract results dict
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}

        # Extract required values with None defaults
        self.SG_date = results.get('SG_date', None)
        self.FH_date = results.get('FH_date', None)
        self.FH_threshold_E = results.get('FH_threshold_E', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_SG_date_present(self):
        if self.SG_date is None:
            return (False, "SG_date missing or None")
        return (True, "SG_date present")

    def test_interface_FH_date_present(self):
        if self.FH_date is None:
            return (False, "FH_date missing or None")
        return (True, "FH_date present")

    def test_interface_FH_threshold_E_present(self):
        if self.FH_threshold_E is None:
            return (False, "FH_threshold_E missing or None")
        return (True, "FH_threshold_E present")

    def test_interface_SG_date_type(self):
        if self.SG_date is None:
            return (False, "SKIP: SG_date not available")
        if not isinstance(self.SG_date, (int, float)):
            return (False, f"SG_date has type {type(self.SG_date).__name__}, expected numeric")
        return (True, "SG_date is numeric")

    def test_interface_FH_date_type(self):
        if self.FH_date is None:
            return (False, "SKIP: FH_date not available")
        if not isinstance(self.FH_date, (int, float)):
            return (False, f"FH_date has type {type(self.FH_date).__name__}, expected numeric")
        return (True, "FH_date is numeric")

    def test_interface_FH_threshold_E_type(self):
        if self.FH_threshold_E is None:
            return (False, "SKIP: FH_threshold_E not available")
        if not isinstance(self.FH_threshold_E, (int, float)):
            return (False, f"FH_threshold_E has type {type(self.FH_threshold_E).__name__}, expected numeric")
        return (True, "FH_threshold_E is numeric")

    # ------------------------------------------------------------------
    # Invariant tests (fuzzy numerical matching)
    # ------------------------------------------------------------------
    def test_invariant_SG_date_historically_plausible(self):
        """Stern-Gerlach experiment occurred in the early 20th century."""
        if self.SG_date is None:
            return (False, "SKIP: SG_date not available")
        if not isinstance(self.SG_date, (int, float)):
            return (False, "SG_date not numeric")
        plausible = (1900 <= self.SG_date <= 1950)
        if not plausible:
            return (False, f"SG_date = {self.SG_date} outside plausible range (1900‑1950)")
        return (True, f"SG_date = {self.SG_date} lies in historical range")

    def test_invariant_FH_date_historically_plausible(self):
        if self.FH_date is None:
            return (False, "SKIP: FH_date not available")
        if not isinstance(self.FH_date, (int, float)):
            return (False, "FH_date not numeric")
        plausible = (1900 <= self.FH_date <= 1950)
        if not plausible:
            return (False, f"FH_date = {self.FH_date} outside plausible range (1900‑1950)")
        return (True, f"FH_date = {self.FH_date} lies in historical range")

    # ------------------------------------------------------------------
    # Modeling/trace quality tests (at least two)
    # ------------------------------------------------------------------
    def test_modeling_SG_date_from_explanation(self):
        """Check that the Stern‑Gerlach date is close to 1921 as given in explanation steps."""
        if self.SG_date is None:
            return (False, "SKIP: SG_date not available")
        if not isinstance(self.SG_date, (int, float)):
            return (False, "SG_date not numeric")
        expected = 1921
        if not math.isclose(self.SG_date, expected, rel_tol=0.02, abs_tol=5):
            return (False, f"SG_date = {self.SG_date} is not close to 1921")
        return (True, f"SG_date = {self.SG_date} matches expected 1921")

    def test_modeling_FH_threshold_consistent_with_excitation_energy(self):
        """Franck‑Hertz threshold is the first excitation energy of mercury: 4.9 eV (from explanation steps)."""
        if self.FH_threshold_E is None:
            return (False, "SKIP: FH_threshold_E not available")
        if not isinstance(self.FH_threshold_E, (int, float)):
            return (False, "FH_threshold_E not numeric")
        expected = 4.9
        if not math.isclose(self.FH_threshold_E, expected, rel_tol=0.02, abs_tol=0.1):
            return (False, f"FH_threshold_E = {self.FH_threshold_E} is not close to 4.9 eV")
        return (True, "FH_threshold_E is approximately 4.9 eV (first excited state of Hg)")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_SG_date_after_FH_date(self):
        """Historical ordering: Stern‑Gerlach (1921) after Franck‑Hertz (1914). Perturb experiment choice (categorical input)."""
        if self.SG_date is None or self.FH_date is None:
            return (False, "SKIP: one or both dates missing")
        # baseline output: FH_date
        baseline = self.FH_date
        # perturbed output: SG_date
        perturbed = self.SG_date
        if not isinstance(baseline, (int, float)) or not isinstance(perturbed, (int, float)):
            return (False, "Dates must be numeric for comparison")
        if perturbed <= baseline:
            return (False, f"SG_date ({perturbed}) is not later than FH_date ({baseline})")
        return (True, f"SG_date ({perturbed}) > FH_date ({baseline}) — chronological ordering holds")

    # ------------------------------------------------------------------
    # Weak final-answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_threshold_positive_and_finite(self):
        if self.FH_threshold_E is None:
            return (False, "SKIP: FH_threshold_E not available")
        if not isinstance(self.FH_threshold_E, (int, float)):
            return (False, "FH_threshold_E not numeric")
        if not math.isfinite(self.FH_threshold_E):
            return (False, "FH_threshold_E is not finite")
        if self.FH_threshold_E <= 0:
            return (False, "FH_threshold_E is not positive")
        return (True, "FH_threshold_E is positive and finite")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
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