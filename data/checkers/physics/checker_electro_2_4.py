import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction with None defaults; no numeric fallbacks
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        if explanation_steps is None:
            explanation_steps = []
        pseudocode = getattr(solution_instance, 'pseudocode', '')
        if pseudocode is None:
            pseudocode = ''

        # Result key
        self.B = results.get('B', None)

        # Givens
        self.I = givens.get('I', None)          # current in A
        self.r = givens.get('r', None)          # distance in m

        # Physics constant (not from solution)
        self.mu0 = 4.0 * math.pi * 1e-7          # T·m/A

        # Metadata for trace checks
        self.explanation_steps = explanation_steps
        self.pseudocode = pseudocode

    # ---------- Interface Tests ----------
    def test_interface_B_present(self):
        """Check that the result key 'B' exists and is a finite number."""
        if self.B is None:
            return (False, "SKIP: 'B' not found in results")
        if not isinstance(self.B, (int, float)):
            return (False, f"B is not numeric, type={type(self.B)}")
        if math.isnan(self.B) or math.isinf(self.B):
            return (False, f"B is not finite: {self.B}")
        return (True, "B is present and numeric")

    # ---------- Invariant / Theorem-based Tests (at least two) ----------
    def test_invariant_ampere_law_structure(self):
        """Check that B satisfies Ampere's law: B = μ0 I/(2π r) within tolerance."""
        if self.B is None or self.I is None or self.r is None:
            return (False, "SKIP: insufficient data to check Ampere's law")
        if self.r <= 0:
            return (False, f"Distance r must be positive, got {self.r}")
        expected = self.mu0 * self.I / (2.0 * math.pi * self.r)
        if not math.isclose(self.B, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B = {self.B}, expected ≈ {expected}")
        return (True, "B matches Ampere's law (B = μ0 I/(2π r))")

    def test_invariant_Ampere_product(self):
        """Check that B·(2π r) equals μ0·I within tolerance."""
        if self.B is None or self.I is None or self.r is None:
            return (False, "SKIP: insufficient data for Ampere product")
        if self.r <= 0:
            return (False, f"Non-positive r: {self.r}")
        lhs = self.B * 2.0 * math.pi * self.r
        rhs = self.mu0 * self.I
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B·2πr = {lhs}, μ0 I = {rhs}")
        return (True, "Ampere's law product holds: B·2πr ≈ μ0 I")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_derivation_from_givens(self):
        """Re-derive B from given I, r using Ampere's law and compare with result."""
        if self.B is None or self.I is None or self.r is None:
            return (False, "SKIP: missing I, r, or B")
        if self.r <= 0:
            return (False, f"Non-positive r: {self.r}")
        computed = self.mu0 * self.I / (2.0 * math.pi * self.r)
        if not math.isclose(self.B, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Modeled B = {computed}, result B = {self.B}")
        return (True, "B is correctly derived from givens via Ampere's law")

    def test_trace_formula_appears_in_pseudocode(self):
        """Check that the pseudocode (if present) contains the formula B = μ0 I/(2π r)."""
        if not self.pseudocode:
            return (True, "SKIP: no pseudocode provided")
        # Normalize the pseudocode for matching
        normalized = self.pseudocode.replace(" ", "").lower()
        # List of plausible formula strings
        patterns = [
            "mu0*i/(2*pi*r)",
            "4*pi*10^(-7)*i/(2*pi*r)",
            "1.2e-2/(2*pi*r)",   # after substitution
            "b=mu0*i/(2*pi*r)",
            "b=(4*pi*1e-7*i)/(2*pi*r)"
        ]
        if not any(p in normalized for p in patterns):
            return (False, "Pseudocode does not contain expected Ampere's law formula")
        return (True, "Pseudocode references the correct formula")

    # ---------- Monotonicity Tests (must perturb input) ----------
    def test_monotonicity_B_inversely_proportional_to_r(self):
        """Perturb r upward by 10% and verify that the expected B decreases."""
        if self.B is None or self.I is None or self.r is None:
            return (False, "SKIP: cannot test monotonicity without I, r, B")
        if self.r <= 0:
            return (False, f"Invalid r: {self.r}")

        # Baseline B at given r
        baseline = self.B

        # Perturb: increase r by 10%
        r_perturbed = self.r * 1.1
        # Compute expected B at perturbed r using the same physical formula
        perturbed_B = self.mu0 * self.I / (2.0 * math.pi * r_perturbed)

        # Monotonicity check: perturbed B must be less than baseline
        if not (perturbed_B < baseline):
            return (False, f"Perturbing r up gave B_perturbed={perturbed_B} >= baseline={baseline}")
        # Additionally check the ratio is approximately inverse: baseline/perturbed ≈ r_perturbed/r
        ratio_B = baseline / perturbed_B
        ratio_r = r_perturbed / self.r
        if not math.isclose(ratio_B, ratio_r, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Inverse proportionality not preserved: B-ratio={ratio_B}, r-ratio={ratio_r}")
        return (True, "Monotonicity holds: B decreases when r increases, with inverse proportionality")

    # ---------- Weak Answer Sanity (only one allowed) ----------
    def test_answer_sanity_finite_positive_broad_range(self):
        """Check that B is positive and within plausible order of magnitude (1e-4 to 1e-1 T)."""
        if self.B is None:
            return (False, "SKIP: B not available")
        if self.B <= 0:
            return (False, f"B must be positive, got {self.B}")
        if self.B < 1e-4 or self.B > 1e-1:
            return (False, f"B = {self.B} outside plausible range [1e-4, 1e-1] T")
        return (True, "B is positive and within plausible magnitude range")

    # ---------- Test Runner ----------
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