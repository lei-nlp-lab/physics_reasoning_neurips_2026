import sympy as sp
import math
import traceback
import re

class PhysicsChecker:
    """
    Audits a solution for multiple-slit grating diffraction.
    Checks theorem invariants, modeling quality, monotonicity, and weak final-answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Extract expected keys from pseudocode [emit] contract
        self.intensity_formula = self.results.get('intensity_formula', None)
        self.num_subsidiary = self.results.get('num_subsidiary', None)

        # Extract givens if present
        self.N = getattr(solution_instance, 'N', None)
        if self.N is None:
            givens = getattr(solution_instance, 'givens', {})
            if givens and 'N' in givens:
                self.N = givens['N']
        # Also try from results
        if self.N is None and 'N' in self.results:
            self.N = self.results['N']

        self.d = getattr(solution_instance, 'd', None)
        self.I0 = getattr(solution_instance, 'I0', None)

        # Explanation steps if available
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Symbolic variable (for internal use, not as default)
        self.phi = sp.symbols('phi')

    # ---------- Interface tests ----------
    def test_interface_intensity_formula_exists(self):
        if self.intensity_formula is None:
            return (False, "SKIP: intensity_formula not provided")
        if not isinstance(self.intensity_formula, str):
            return (False, "intensity_formula must be a string")
        return (True, "intensity_formula is a non-None string")

    def test_interface_num_subsidiary_exists(self):
        if self.num_subsidiary is None:
            return (False, "SKIP: num_subsidiary not provided")
        if not isinstance(self.num_subsidiary, (int, float)):
            return (False, "num_subsidiary must be a number")
        if isinstance(self.num_subsidiary, float) and not self.num_subsidiary.is_integer():
            return (False, "num_subsidiary should be an integer")
        return (True, "num_subsidiary is an integer value")

    # ---------- Invariant (theorem-based) tests ----------
    def test_invariant_intensity_formula_structure(self):
        if self.intensity_formula is None:
            return (False, "SKIP: intensity_formula not available")
        f = self.intensity_formula.lower()
        checks = []
        if 'sin' not in f:
            checks.append("missing 'sin'")
        if 'i0' not in f and 'i_0' not in f and 'i_o' not in f:
            checks.append("missing I0")
        if 'n' not in f and 'N' not in self.intensity_formula:
            checks.append("missing N (expected variable)")
        if 'phi' not in f and 'θ' not in f and 'theta' not in f and 'φ' not in f:
            checks.append("missing phase variable (phi/theta)")
        if not re.search(r'sin\s*\([^)]*\)\s*/\s*sin\s*\(', self.intensity_formula, re.IGNORECASE):
            checks.append("missing division pattern sin(...)/sin(...)")
        if checks:
            return (False, "Structural issues: " + "; ".join(checks))
        return (True, "Intensity formula contains essential components")

    def test_invariant_intensity_symmetry(self):
        if self.intensity_formula is None:
            return (False, "SKIP: intensity_formula not available")
        if '^2' not in self.intensity_formula and '**2' not in self.intensity_formula:
            return (False, "Intensity formula likely not squared – symmetry may be broken")
        return (True, "Intensity formula appears even (contains square)")

    def test_invariant_num_subsidiary_vs_N(self):
        if self.num_subsidiary is None:
            return (False, "SKIP: num_subsidiary not available")
        if self.N is None:
            return (False, "SKIP: N not available to compare")
        expected = self.N - 2
        if self.num_subsidiary != expected:
            return (False, f"num_subsidiary = {self.num_subsidiary}, expected N-2 = {expected}")
        return (True, f"num_subsidiary equals N-2 = {expected}")

    def test_invariant_num_subsidiary_nonnegative(self):
        if self.num_subsidiary is None:
            return (False, "SKIP: num_subsidiary not available")
        if not isinstance(self.num_subsidiary, (int, float)):
            return (False, "num_subsidiary must be numeric")
        if self.num_subsidiary < 0:
            return (False, f"num_subsidiary = {self.num_subsidiary} is negative")
        return (True, "num_subsidiary is non-negative")

    def test_invariant_zeros_condition(self):
        if self.N is None or self.intensity_formula is None:
            return (False, "SKIP: need both N and intensity_formula")
        f = self.intensity_formula.lower()
        if 'sin' not in f:
            return (False, "missing sin function")
        if '/' not in f:
            return (False, "no division found; impossible to have zeros from denominator")
        return (True, "Intensity formula structure consistent with zero condition")

    # ---------- Modeling / trace consistency tests ----------
    def test_modeling_explanation_contains_intensity_formula(self):
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        text = str(self.explanation_steps)
        keywords = ['I = I0', 'intensity', 'diffraction', 'grating']
        found = [kw for kw in keywords if kw.lower() in text.lower()]
        if len(found) < 2:
            return (False, f"Explanation lacks enough intensity references: found {found}")
        return (True, f"Explanation covers intensity keywords: {found}")

    def test_modeling_explanation_contains_num_subsidiary(self):
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        text = str(self.explanation_steps)
        if 'n-2' in text.lower() or 'N-2' in text:
            return (True, "Explanation mentions N-2")
        if 'subsidiary maxima' in text.lower() or 'subsidiary' in text.lower():
            return (True, "Explanation discusses subsidiary maxima")
        return (False, "Explanation does not clearly mention number of subsidiary maxima")

    def test_modeling_emit_contract(self):
        required_emit_keys = ['intensity_formula', 'num_subsidiary']
        missing = [k for k in required_emit_keys if k not in self.results]
        extra = [k for k in self.results if k not in required_emit_keys and not k.startswith('_')]
        logs = []
        if missing:
            logs.append(f"Missing emit keys: {missing}")
        if extra:
            logs.append(f"Extra keys in results: {extra}")
        if missing:
            return (False, "; ".join(logs))
        return (True, f"All required emit keys present. Extra keys: {extra}" if extra else "All required emit keys present")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_num_subsidiary_increases_with_N(self):
        """
        Theorem: num_subsidiary = N-2. If N increases by 1,
        then num_subsidiary should increase by 1 (monotonic increase).
        """
        if self.num_subsidiary is None:
            return (False, "SKIP: num_subsidiary not available for baseline")
        if self.N is None:
            return (False, "SKIP: N not available to perturb")
        # Baseline: provided num_subsidiary
        baseline = self.num_subsidiary
        # Perturb N by +1
        perturbed_N = self.N + 1
        # Estimate perturbed output using the theorem relation
        estimated_perturbed = perturbed_N - 2  # = self.N - 1
        # Since N-2 < N-1, baseline should be less than estimated_perturbed
        if not (baseline < estimated_perturbed):
            return (False, f"Baseline num_subsidiary={baseline} (from N={self.N}) is not less than estimated perturbed value {estimated_perturbed} (for N={perturbed_N}).")
        # Additionally, the difference should be exactly 1
        if estimated_perturbed - baseline != 1:
            return (False, f"Difference perturbed - baseline = {estimated_perturbed - baseline}, expected 1.")
        return (True, f"num_subsidiary increases by 1 when N increases by 1 (baseline {baseline}, perturbed estimate {estimated_perturbed})")

    # ---------- Weak answer sanity tests (final answer, not dominant) ----------
    def test_answer_sanity_weak(self):
        """
        Combined weak sanity checks for both intensity_formula and num_subsidiary.
        Only checks non-emptiness, type, and plausible range.
        """
        issues = []
        # Check intensity_formula
        if self.intensity_formula is not None:
            if not isinstance(self.intensity_formula, str):
                issues.append("intensity_formula must be a string")
            elif len(self.intensity_formula.strip()) < 5:
                issues.append("intensity_formula is too short")
        # Check num_subsidiary
        if self.num_subsidiary is not None:
            if not isinstance(self.num_subsidiary, (int, float)):
                issues.append("num_subsidiary must be numeric")
            elif not (0 <= self.num_subsidiary <= 1e6):
                issues.append(f"num_subsidiary = {self.num_subsidiary} out of plausible range")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Weak sanity checks passed")

    # ---------- Master runner ----------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)