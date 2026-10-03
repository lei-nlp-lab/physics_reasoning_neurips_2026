import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the special relativity historical problem.
    """
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        # Contract keys (required)
        self.dilemma = results.get('dilemma', None)
        self.fringe_shift_expected = results.get('fringe_shift_expected', None)
        self.fringe_shift_observed = results.get('fringe_shift_observed', None)
        self.time_dilation_confirmed = results.get('time_dilation_confirmed', None)

        # Additional potential numeric inputs for monotonicity (optional)
        self.v_earth = results.get('v_earth', None)          # Earth's speed relative to aether (m/s)
        self.arm_length = results.get('arm_length', None)    # interferometer arm length (m)
        self.wavelength = results.get('wavelength', None)    # light wavelength (m)
        self.speed = results.get('speed', None)              # generic speed for time dilation

        # Try to get explanation steps if stored as attribute or in results
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = results.get('explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface tests: required keys exist with appropriate types
    # ------------------------------------------------------------------
    def test_interface_dilemma_exists(self):
        if self.dilemma is None:
            return (False, "dilemma is missing (None)")
        if not isinstance(self.dilemma, str):
            return (False, f"dilemma should be a string, got {type(self.dilemma).__name__}")
        if len(self.dilemma.strip()) == 0:
            return (False, "dilemma is empty string")
        return (True, "dilemma present and is string")

    def test_interface_fringe_shift_expected_exists(self):
        if self.fringe_shift_expected is None:
            return (False, "fringe_shift_expected is missing (None)")
        if not isinstance(self.fringe_shift_expected, str):
            return (False, f"fringe_shift_expected should be string, got {type(self.fringe_shift_expected).__name__}")
        return (True, "fringe_shift_expected present and is string")

    def test_interface_fringe_shift_observed_exists(self):
        if self.fringe_shift_observed is None:
            return (False, "fringe_shift_observed is missing (None)")
        if not isinstance(self.fringe_shift_observed, (int, float)):
            return (False, f"fringe_shift_observed should be numeric, got {type(self.fringe_shift_observed).__name__}")
        return (True, "fringe_shift_observed present and numeric")

    def test_interface_time_dilation_confirmed_exists(self):
        if self.time_dilation_confirmed is None:
            return (False, "time_dilation_confirmed is missing (None)")
        if not isinstance(self.time_dilation_confirmed, bool):
            return (False, f"time_dilation_confirmed should be bool, got {type(self.time_dilation_confirmed).__name__}")
        return (True, "time_dilation_confirmed present and bool")

    # ------------------------------------------------------------------
    # Invariant tests: physical consistency
    # ------------------------------------------------------------------
    def test_invariant_null_result_consistency(self):
        """Expected fringe shift must be 'non-zero' and observed must be 0."""
        if self.fringe_shift_expected is None or self.fringe_shift_observed is None:
            return (False, "SKIP: fringe shift data missing")
        if self.fringe_shift_expected != 'non-zero':
            return (False, f"fringe_shift_expected is '{self.fringe_shift_expected}', expected 'non-zero'")
        if not math.isclose(self.fringe_shift_observed, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"fringe_shift_observed = {self.fringe_shift_observed}, expected 0 (within tolerance)")
        return (True, "null result consistency: expected non-zero, observed zero")

    def test_invariant_time_dilation_is_true(self):
        if self.time_dilation_confirmed is None:
            return (False, "SKIP: time_dilation_confirmed missing")
        if self.time_dilation_confirmed is not True:
            return (False, f"time_dilation_confirmed = {self.time_dilation_confirmed}, expected True")
        return (True, "time dilation confirmed")

    # ------------------------------------------------------------------
    # Trace/modeling tests: explanation steps contain key concepts
    # ------------------------------------------------------------------
    def test_trace_consistency_dilemma_mention(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        has_maxwell = "maxwell" in combined
        has_newton = "newton" in combined
        has_incompatible = any(word in combined for word in ["incompat", "contrary", "conflict", "dilemma"])
        if not (has_maxwell and has_newton and has_incompatible):
            msg = "Explanation steps missing: Maxwell, Newton, and concept of incompatibility/dilemma"
            return (False, msg)
        return (True, "Explanation steps mention Maxwell, Newton, and incompatibility")

    def test_trace_consistency_aether_michelson(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        has_aether = "aether" in combined
        has_michelson = "michelson" in combined
        if not (has_aether and has_michelson):
            msg = "Explanation steps missing: 'aether' or 'Michelson'"
            return (False, msg)
        return (True, "Explanation steps mention aether and Michelson")

    def test_trace_consistency_modern_experiment(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        keywords = ["hafele", "keating", "muon", "gps", "time dilation", "atomic clock"]
        found = any(keyword in combined for keyword in keywords)
        if not found:
            msg = "Explanation steps should mention a modern experiment (Hafele-Keating, muon, etc.)"
            return (False, msg)
        return (True, "Explanation steps mention a modern experiment")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_fringe_shift_with_speed(self):
        """If v_earth, arm_length, wavelength provided, increasing v_earth increases expected fringe shift."""
        if any(x is None for x in [self.v_earth, self.arm_length, self.wavelength]):
            return (False, "SKIP: v_earth, arm_length, or wavelength missing – cannot perform test")
        v = self.v_earth
        L = self.arm_length
        lam = self.wavelength
        c = 3.0e8  # speed of light in vacuum (known constant)
        # Compute baseline expected fringe shift (using Michelson-Morley formula)
        baseline_delta = (2.0 * L / lam) * (v**2 / c**2)
        # Perturb v by +5%
        v_perturbed = v * 1.05
        perturbed_delta = (2.0 * L / lam) * (v_perturbed**2 / c**2)
        # Monotonicity: pertubed > baseline
        if perturbed_delta <= baseline_delta:
            return (False, f"Perturbed delta {perturbed_delta:.6e} is not > baseline {baseline_delta:.6e}")
        return (True, f"Monotonicity: 5% increase in v increases expected fringe shift from {baseline_delta:.6e} to {perturbed_delta:.6e}")

    def test_monotonicity_time_dilation_with_speed(self):
        """If speed provided, increasing speed increases the time dilation factor gamma."""
        if self.speed is None:
            return (False, "SKIP: no speed provided for time dilation monotonicity")
        v = self.speed
        c = 3.0e8
        if v >= c:
            return (False, f"Speed {v} >= c, cannot compute gamma")
        gamma_base = 1.0 / math.sqrt(1.0 - v**2 / c**2)
        # Perturb speed by +5%
        v_perturbed = v * 1.05
        if v_perturbed >= c:
            return (False, "Perturbed speed exceeds c, skip")
        gamma_perturbed = 1.0 / math.sqrt(1.0 - v_perturbed**2 / c**2)
        if gamma_perturbed <= gamma_base:
            return (False, f"Perturbed gamma {gamma_perturbed:.6f} not > baseline {gamma_base:.6f}")
        return (True, f"Monotonicity: 5% increase in speed increases gamma from {gamma_base:.6f} to {gamma_perturbed:.6f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (no re-derivation of final answers)
    # ------------------------------------------------------------------
    def test_answer_sanity_not_nan_not_infinite(self):
        if self.fringe_shift_observed is not None and isinstance(self.fringe_shift_observed, (int, float)):
            val = self.fringe_shift_observed
            if math.isnan(val) or math.isinf(val):
                return (False, f"fringe_shift_observed is NaN or Inf: {val}")
        # No other numeric outputs to check (time_dilation_confirmed is bool)
        return (True, "All checked values are finite/valid")

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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)