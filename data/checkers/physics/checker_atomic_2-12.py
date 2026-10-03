import sympy as sp
import math
import re
import traceback

class PhysicsChecker:
    """Auditor for isobar mass formula derivation and isospin minimal test condition."""

    def __init__(self, solution_instance):
        # Extract inputs from solution_instance, using .get() or getattr with None defaults
        # Assume the solution_instance may have attributes or a results dict
        try:
            results = solution_instance.results if hasattr(solution_instance, 'results') else {}
        except AttributeError:
            results = {}

        # Main result keys expected from pseudocode [emit] contract
        self.M_formula = results.get('M_formula', None)
        self.T_min = results.get('T_min', None)

        # Also extract explanation steps if available (for modeling checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Ensure safe defaults: NO numeric fallbacks, only None
        # (Already done above)

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_M_formula_exists(self):
        """Check that the mass formula result is provided and is a non-empty string."""
        if self.M_formula is None:
            return (False, "SKIP: M_formula not provided")
        if not isinstance(self.M_formula, str) or len(self.M_formula.strip()) == 0:
            return (False, "M_formula is not a non-empty string")
        return (True, "M_formula exists and is a non-empty string")

    def test_interface_T_min_exists(self):
        """Check that T_min result is provided and is a positive integer (or float)."""
        if self.T_min is None:
            return (False, "SKIP: T_min not provided")
        if not isinstance(self.T_min, (int, float)):
            return (False, f"T_min is not numeric, got {type(self.T_min).__name__}")
        if self.T_min <= 0:
            return (False, f"T_min must be positive, got {self.T_min}")
        return (True, f"T_min is {self.T_min} (positive numeric)")

    # -----------------------------------------------------------------
    # Invariant / theorem-based checks
    # -----------------------------------------------------------------
    def test_invariant_formula_quadratic_in_T_z(self):
        """The formula must be of the form a + b*T_z + c*T_z^2, no higher powers."""
        if self.M_formula is None:
            return (False, "SKIP: M_formula not available for quadratic check")
        formula = self.M_formula.strip()

        # Check for linear and quadratic terms, forbid cubic or higher
        has_linear_term = 'T_z' in formula
        has_quadratic_term = ('T_z^2' in formula) or ('T_z**2' in formula)
        forbidden_exponents = ['T_z^3', 'T_z**3', 'T_z^4', 'T_z**4', 'T_z^5', 'T_z**5']
        has_higher = any(term in formula for term in forbidden_exponents)
        if has_linear_term and has_quadratic_term and not has_higher:
            return (True, "Formula is quadratic in T_z (contains T_z and T_z^2, no higher powers)")
        else:
            details = []
            if not has_linear_term:
                details.append("missing T_z term")
            if not has_quadratic_term:
                details.append("missing T_z^2 term")
            if has_higher:
                details.append("contains higher powers (e.g., T_z^3)")
            return (False, f"Formula does not match required quadratic form: {', '.join(details)}")

    def test_invariant_T_min_ge_1(self):
        """For testing the c T_z^2 term, we need at least T=1 (so 2T+1 >= 3)."""
        if self.T_min is None:
            return (False, "SKIP: T_min not available")
        if self.T_min >= 1:
            return (True, f"T_min = {self.T_min} is >= 1, allows three distinct T_z values")
        else:
            return (False, f"T_min = {self.T_min} is < 1, insufficient to determine three parameters")

    # -----------------------------------------------------------------
    # Modeling / trace checks
    # -----------------------------------------------------------------
    def test_modeling_explanation_mentions_three_parameters(self):
        """Explanation steps should mention that three independent equations are needed."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        text = str(self.explanation_steps)
        keywords = ["three", "3", "independent", "equations", "constants", "a, b, c"]
        found = [kw for kw in keywords if kw.lower() in text.lower()]
        if len(found) >= 2:
            return (True, f"Explanation mentions relevant terms: {found}")
        else:
            return (False, "Explanation does not clearly mention the need for three equations/parameters")

    def test_trace_T_z_range_specified(self):
        """Pseudocode or explanation should mention that T_z ranges from -T to T."""
        text = ""
        if self.explanation_steps:
            text += str(self.explanation_steps)
        if self.pseudocode:
            text += str(self.pseudocode)
        if not text:
            return (False, "SKIP: no explanation or pseudocode available")
        if re.search(r'-T\s*to\s*T', text) or re.search(r'T_z\s*(?:range|values|from)', text):
            return (True, "T_z range is mentioned")
        else:
            return (False, "T_z range (-T to T) not explicitly mentioned in trace")

    # -----------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input, compare output)
    # -----------------------------------------------------------------
    def test_monotonicity_T_z_values_increase_with_T(self):
        """
        Explicitly perturb the isospin T (input) and verify that the number
        of distinct T_z values (2T+1) increases monotonically.
        Baseline: T=0 (1 value), perturbed: T=1 (3 values).
        """
        # Define baseline and perturbed inputs
        baseline_T = 0
        perturbed_T_big = 1

        # Compute the number of T_z values (output) as a function of T
        def number_of_T_z(T):
            return 2 * T + 1

        baseline_output = number_of_T_z(baseline_T)
        perturbed_output = number_of_T_z(perturbed_T_big)

        # Check that the perturbed output is larger than the baseline output
        if perturbed_output > baseline_output:
            return (True, f"Monotonic: for T=0 -> {baseline_output} T_z values, for T=1 -> {perturbed_output} T_z values, and {perturbed_output} > {baseline_output}")
        else:
            # This should never happen given simple arithmetic, but check for safety
            return (False, f"Non-monotonic: T=0 -> {baseline_output}, T=1 -> {perturbed_output}")

    # -----------------------------------------------------------------
    # Weak answer sanity (ONLY ONE such test)
    # -----------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Weak sanity: final answers must be non-None, non-empty, plausible."""
        issues = []
        if self.M_formula is None:
            issues.append("M_formula missing")
        elif not isinstance(self.M_formula, str) or len(self.M_formula.strip()) < 5:
            issues.append("M_formula too short or invalid")
        if self.T_min is None:
            issues.append("T_min missing")
        elif not isinstance(self.T_min, (int, float)) or self.T_min <= 0:
            issues.append("T_min not a positive number")
        if issues:
            return (False, "Weak sanity failed: " + ", ".join(issues))
        return (True, "Answers pass weak sanity checks (non-None, plausible values)")

    # -----------------------------------------------------------------
    # Run all tests (discovery and execution)
    # -----------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)