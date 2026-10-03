import math
import traceback
from typing import Any, List, Tuple, Dict, Optional

class PhysicsChecker:
    """
    Auditor for solutions to the interference visibility problem.
    Checks interface completeness, logical consistency, reasoning quality,
    monotonic behavior, and a weak answer sanity test.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract all needed attributes; use None/missing defaults
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', None) or {}
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', None) or {}
        self.explanation_steps: List[str] = getattr(solution_instance, 'explanation_steps', None) or []
        self.target_answers: List[str] = getattr(solution_instance, 'target_answers', None) or []
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)

        # Expected result
        self.interference_visible: Optional[bool] = self.results.get('interference_visible', None)

        # Input parameters from givens
        self.source_type: Optional[str] = self.givens.get('source_type', None)
        self.coherence_length: Optional[float] = self.givens.get('coherence_length', None)
        self.path_difference: Optional[float] = self.givens.get('path_difference', None)
        self.observation_duration: Optional[float] = self.givens.get('observation_duration', None)
        self.coherence_time: Optional[float] = self.givens.get('coherence_time', None)

    def _all_conditions_known(self) -> bool:
        """Check that all inputs needed for the logical test are present."""
        return (self.source_type is not None and
                self.coherence_length is not None and
                self.path_difference is not None and
                self.observation_duration is not None and
                self.coherence_time is not None)

    def _expected_interference_visible(self) -> Optional[bool]:
        """Compute expected visibility from first principles if all data are available."""
        if not self._all_conditions_known():
            return None
        if self.source_type == 'laser':
            return (self.path_difference < self.coherence_length and
                    self.observation_duration < self.coherence_time)
        else:  # 'common' or any other incoherent source
            return False

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_interference_visible_present(self) -> Tuple[bool, str]:
        if 'interference_visible' not in self.results:
            return (False, "Missing required key 'interference_visible' in results")
        vis = self.results['interference_visible']
        if not isinstance(vis, (bool,)):
            return (False, f"'interference_visible' is not boolean: {type(vis).__name__}")
        return (True, "Key 'interference_visible' present and is boolean")

    def test_interface_source_type_valid(self) -> Tuple[bool, str]:
        if self.source_type is None:
            return (False, "SKIP: source_type not provided")
        valid = ['common', 'laser']
        if self.source_type not in valid:
            return (False, f"source_type '{self.source_type}' not in allowed values: {valid}")
        return (True, f"source_type '{self.source_type}' is valid")

    # ----------------------------------------------------------------
    # Invariant tests (logic-based, not numerical)
    # ----------------------------------------------------------------
    def test_invariant_coherent_incoherent_expected(self) -> Tuple[bool, str]:
        """For known inputs, check the logical condition matches the emitted result."""
        if not self._all_conditions_known():
            return (False, "SKIP: insufficient input parameters to compute expected visibility")
        expected = self._expected_interference_visible()
        if expected is None:
            return (False, "SKIP: logic failure")
        if self.interference_visible != expected:
            return (False, f"Interference_visible is {self.interference_visible}, but expected {expected} "
                           f"based on coherence condition and path/observation thresholds")
        return (True, "Logical condition matches emitted result")

    def test_invariant_laser_implies_conditions_needed(self) -> Tuple[bool, str]:
        """For laser sources, if either condition fails, output must be False."""
        if not self._all_conditions_known():
            return (False, "SKIP: insufficient data")
        if self.source_type != 'laser':
            return (False, "SKIP: not a laser source")
        # Check that if any condition fails, the answer is False
        path_ok = self.path_difference < self.coherence_length
        time_ok = self.observation_duration < self.coherence_time
        if not (path_ok and time_ok) and self.interference_visible is True:
            return (False, "Laser source but conditions not met; interference_visible should be False")
        if (path_ok and time_ok) and self.interference_visible is False:
            return (False, "Laser source and all conditions met; interference_visible should be True")
        return (True, "Laser visibility consistent with conditions")

    # ----------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------
    def test_modeling_reasoning_mentions_coherence(self) -> Tuple[bool, str]:
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        keywords = ['coherence', 'coherent', 'phase']
        found = [kw for kw in keywords if kw in combined]
        if not found:
            return (False, "Explanation steps do not mention coherence/phase")
        return (True, f"Explanation steps mention coherence-related keywords: {found}")

    def test_modeling_reasoning_mentions_laser(self) -> Tuple[bool, str]:
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        if 'laser' not in combined:
            return (False, "Explanation steps do not mention laser as a coherent source example")
        return (True, "Explanation steps mention laser")

    def test_modeling_reasoning_mentions_path_difference(self) -> Tuple[bool, str]:
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        if 'path' not in combined:
            return (False, "Explanation steps do not mention path difference")
        return (True, "Explanation steps mention path difference")

    # ----------------------------------------------------------------
    # Monotonicity tests (perturb a parameter, check trend)
    # ----------------------------------------------------------------
    def test_monotonicity_path_difference(self) -> Tuple[bool, str]:
        """For laser with known baseline, increasing path_difference should not increase visibility."""
        if not self._all_conditions_known():
            return (False, "SKIP: insufficient data")
        if self.source_type != 'laser':
            return (False, "SKIP: not a laser source")
        if self.coherence_length is None or self.path_difference is None:
            return (False, "SKIP: missing path_difference or coherence_length")
        # Baseline
        baseline_vis = self.interference_visible if self.interference_visible is not None else False
        # Perturb: increase path_difference by 10%
        perturbed_path = self.path_difference * 1.1
        # Compute perturbed visibility using the same condition (assuming other parameters unchanged)
        perturbed_vis = (perturbed_path < self.coherence_length and
                         self.observation_duration < self.coherence_time)
        # Monotonic non-increasing: perturbed_vis <= baseline_vis
        if perturbed_vis and not baseline_vis:
            return (False, f"Path difference increased from {self.path_difference} to {perturbed_path:.3f}, "
                           f"but visibility improved from False to True. Should be non-increasing.")
        # If both False, it's fine; if baseline True and perturbed False, it's monotonic decreasing
        return (True, "Path_difference monotonicity satisfied (non-increasing visibility)")

    def test_monotonicity_observation_duration(self) -> Tuple[bool, str]:
        """For laser with known baseline, increasing observation_duration should not increase visibility."""
        if not self._all_conditions_known():
            return (False, "SKIP: insufficient data")
        if self.source_type != 'laser':
            return (False, "SKIP: not a laser source")
        if self.coherence_time is None or self.observation_duration is None:
            return (False, "SKIP: missing observation_duration or coherence_time")
        baseline_vis = self.interference_visible if self.interference_visible is not None else False
        perturbed_duration = self.observation_duration * 1.1
        perturbed_vis = (self.path_difference < self.coherence_length and
                         perturbed_duration < self.coherence_time)
        if perturbed_vis and not baseline_vis:
            return (False, f"Observation duration increased from {self.observation_duration} to {perturbed_duration:.3f}, "
                           f"but visibility improved. Should be non-increasing.")
        return (True, "Observation_duration monotonicity satisfied")

    # ----------------------------------------------------------------
    # Weak answer sanity (only one, secondary)
    # ----------------------------------------------------------------
    def test_answer_sanity_simple_content(self) -> Tuple[bool, str]:
        """Weak check: target answer mentions coherent/laser and interference and is non-empty."""
        if not self.target_answers:
            return (False, "SKIP: no target_answers provided")
        answer = " ".join(self.target_answers)
        if not answer.strip():
            return (False, "Target answer is empty string")
        # Check essential keywords
        lower = answer.lower()
        has_coherent = ('coherent' in lower or 'laser' in lower)
        has_interference = ('interference' in lower or 'visible' in lower)
        if not has_coherent:
            return (False, "Target answer does not mention 'coherent' or 'laser'")
        if not has_interference:
            return (False, "Target answer does not mention 'interference' or 'visible'")
        return (True, "Target answer contains relevant keywords")

    # ----------------------------------------------------------------
    # Main test dispatcher
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, run them, collect results."""
        logs: List[str] = []
        all_pass = True
        # Get all bound methods starting with 'test_'
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