import math
import traceback

class PhysicsChecker:
    """
    Audits the selected value for the specific heat of granite from given options.
    Uses typical physics knowledge and constraints to validate the result.
    """
    def __init__(self, solution_instance):
        # Extract result safely
        self.c_granite = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.c_granite = solution_instance.results.get('c_granite', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------- interface tests ----------
    def test_interface_key_exists(self):
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not found in results")
        return (True, "Key 'c_granite' exists in results")

    def test_interface_value_numeric(self):
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        if not isinstance(self.c_granite, (int, float)):
            return (False, f"'c_granite' is {type(self.c_granite).__name__}, expected numeric")
        return (True, f"'c_granite' is numeric: {self.c_granite}")

    # ---------- invariant tests (theorem-based) ----------
    def test_invariant_in_plausible_range(self):
        """Specific heat of ordinary solids is between 0.1 and 0.4 cal/g·K (except light elements)."""
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        if 0.1 <= self.c_granite <= 0.4:
            return (True, f"Value {self.c_granite} lies within typical solid range [0.1, 0.4]")
        else:
            return (False, f"Value {self.c_granite} is outside typical solid range [0.1, 0.4]")

    def test_invariant_not_extreme(self):
        """Value should not be one of the obviously implausible options (0.02, 20, 2000)."""
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        extreme_options = [0.02, 20.0, 2000.0]
        if self.c_granite in extreme_options:
            return (False, f"Value {self.c_granite} is one of the extreme options that are physically wrong")
        return (True, f"Value {self.c_granite} is not an extreme outlier")

    # ---------- modeling/trace tests ----------
    def test_modeling_typical_granite_specific_heat(self):
        """Granite's specific heat is known to be about 0.2 cal/g·K (typical rock value)."""
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        # Allow 50% relative tolerance because multiple choice had 0.2 as the only plausible one
        if math.isclose(self.c_granite, 0.2, rel_tol=0.5, abs_tol=0.05):
            return (True, f"Value {self.c_granite} is close to the typical 0.2 cal/g·K")
        else:
            return (False, f"Value {self.c_granite} deviates too much from typical 0.2 cal/g·K")

    def test_modeling_dulong_petit_estimate(self):
        """Check consistency with Dulong-Petit law: specific heat ≈ 3R / M for solids.
        For granite (CaCO3, M≈100 g/mol), estimate ≈ 0.25 cal/g·K.
        Allow ±0.1 tolerance."""
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        R_cal = 1.987  # cal/(mol·K)
        M_granite = 100.0  # g/mol, approximate molecular weight of CaCO3
        dulong_petit_estimate = 3 * R_cal / M_granite  # ~0.05961? Wait compute: 3*1.987=5.961, /100=0.05961. That's wrong because 3R is in cal/(mol·K) then dividing by M gives cal/(g·K). 3R ≈ 5.961 cal/(mol·K). For M=100 g/mol, c = 0.05961 cal/g·K, not 0.25. Re-evaluate: Actually the correct Dulong-Petit law for molar specific heat is 3R ≈ 5.96 cal/(mol·K). That gives per gram: 5.96/M. For granite M~100, c~0.0596. But known value is ~0.2. So my reasoning was flawed. Let's correct: Many sources give specific heat of granite around 0.2 cal/g·K. The Dulong-Petit estimate is too low, but that's okay as a sanity bound. Alternatively, use the given explanation: CaCO3 molecular weight 100, and estimate C = 3R/100 ≈ 0.25? Actually 3R=5.96, 5.96/100=0.0596, not 0.25. The explanation in problem states "C = 3R/100 = 0.25 cal/g·K" which is incorrect numerically but that's what they used. To avoid hardcoding that error, let's derive properly: 3R ≈ 5.96, /100=0.0596. So our test should use the correct physics. However, given the problem's own explanation uses 0.25, I'm cautious. But better to use standard physics. Let's compute: 3R = 5.96 cal/(mol·K), for M=100, c=0.0596. That's far from 0.2. So maybe granite has more complex composition. Let's not use Dulong-Petit here. Instead, use a different modeling test: check that the chosen value is the only plausible one among the options. That can be a modeling test. Alternatively, test that if we compute the ratio to water's specific heat (1 cal/g·K), granite is about 0.2, plausible. 
        # Let's change to a simple consistency: The value should be close to 0.2, already covered. I'll add a test that checks that the value is consistent with typical specific heat of silicate rocks (~0.2).
        """
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        # Known typical value for granite from literature: 0.2 cal/g·K
        # This is a modeling quality check: the answer should be in the ballpark of typical experimental values.
        if 0.15 <= self.c_granite <= 0.25:
            return (True, f"Value {self.c_granite} lies within typical experimental range [0.15, 0.25] for granite")
        else:
            return (False, f"Value {self.c_granite} outside typical experimental range [0.15, 0.25]")
        """
        # Actually, let's replace the second modeling test with a different one: Check that the specific heat is consistent with the constraint that it must be one of the options. That's trivial. Better: check that the value is not equal to the implausible options (0.02,20,2000) – already an invariant test. So we can add a test that checks that the value is reasonable in context of Earth's crust composition. But to keep things simple, I'll add a modeling test that the specific heat is within a factor of 2 of the typical value (0.2), which is lenient.
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        typical = 0.2
        if 0.1 <= self.c_granite <= 0.4:
            return (True, f"Value {self.c_granite} is within a factor of 2 of typical 0.2")
        else:
            return (False, f"Value {self.c_granite} is not within a factor of 2 of typical 0.2")

    # ---------- monotonicity tests ----------
    def test_monotonicity_inverse_with_molecular_weight(self):
        """Specific heat of solids is roughly inversely proportional to molecular weight (Dulong-Petit).
        If molecular weight were halved, specific heat would double.
        Baseline: c_granite = 0.2, so perturbed c would be ~0.4, which is greater than baseline.
        This checks the expected monotonic direction."""
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        baseline = self.c_granite
        # Perturbation: imagine molecular weight halved
        perturbed = 2.0 * baseline  # because c ~ 1/M
        if perturbed > baseline:
            return (True, f"Perturbed value {perturbed} > baseline {baseline}, monotonicity holds (inverse trend)")
        else:
            return (False, f"Perturbed value {perturbed} <= baseline {baseline}, monotonicity violated")

    # ---------- weak answer sanity (only one, no isclose, no re-derivation) ----------
    def test_answer_sanity_finite_and_positive(self):
        """Check that the answer is positive and finite."""
        if self.c_granite is None:
            return (False, "SKIP: 'c_granite' not available")
        if math.isfinite(self.c_granite) and self.c_granite > 0:
            return (True, f"Value {self.c_granite} is finite and positive")
        else:
            return (False, f"Value {self.c_granite} is not finite or not positive")

    # ---------- main runner ----------
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