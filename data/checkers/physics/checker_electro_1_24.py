import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the capacitor problem where charge Q is conserved
    and capacitance changes from C1 to C2 (C2 < C1). The relation V2 = (C1/C2)*V1
    must hold.
    """

    def __init__(self, solution_instance):
        # ---------- Extract givens safely ----------
        # Try accessing attributes first, then a results dict, then fallback to None
        self.C1 = getattr(solution_instance, 'C1', None)
        if self.C1 is None:
            self.C1 = solution_instance.results.get('C1', None) if hasattr(solution_instance, 'results') else None

        self.C2 = getattr(solution_instance, 'C2', None)
        if self.C2 is None:
            self.C2 = solution_instance.results.get('C2', None) if hasattr(solution_instance, 'results') else None

        self.V1 = getattr(solution_instance, 'V1', None)
        if self.V1 is None:
            self.V1 = solution_instance.results.get('V1', None) if hasattr(solution_instance, 'results') else None

        # ---------- Extract result V2 safely ----------
        self.V2 = getattr(solution_instance, 'V2', None)
        if self.V2 is None:
            self.V2 = solution_instance.results.get('V2', None) if hasattr(solution_instance, 'results') else None

        # ---------- Extract explanation steps (for modeling tests) ----------
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = solution_instance.results.get('explanation_steps', '') if hasattr(solution_instance, 'results') else ''
        # Fallback: if results contains 'explanation' key
        if self.explanation_steps is None or self.explanation_steps == '':
            tmp = solution_instance.results.get('explanation', '') if hasattr(solution_instance, 'results') else ''
            self.explanation_steps = tmp if tmp else ''

        # Also store the raw results dict if it exists (for interface checks)
        self.results = getattr(solution_instance, 'results', None) or {}

    # ==================== Interface Tests ====================

    def test_interface_results_key_V2(self):
        """Check that results dict contains key 'V2'."""
        if self.results is None:
            return (False, "results dict is None")
        if 'V2' not in self.results:
            return (False, "results missing key 'V2'")
        return (True, "results has key 'V2'")

    def test_interface_givens_numeric(self):
        """Check that C1, C2, V1 are numeric and not None."""
        for name, val in [('C1', self.C1), ('C2', self.C2), ('V1', self.V1)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
        return (True, "C1, C2, V1 are numeric")

    def test_interface_positivity(self):
        """Check that C1, C2, V1 are positive (C2 < C1 automatic later)."""
        for name, val in [('C1', self.C1), ('C2', self.C2), ('V1', self.V1)]:
            if val is not None and val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "C1, C2, V1 are positive")

    # ==================== Invariant / Theorem Tests ====================

    def test_invariant_charge_conservation(self):
        """Check Q = C1*V1 ≈ C2*V2 (charge conservation)."""
        if any(v is None for v in [self.C1, self.C2, self.V1, self.V2]):
            return (False, "SKIP: missing data for charge conservation")
        Q1 = self.C1 * self.V1
        Q2 = self.C2 * self.V2
        if math.isclose(Q1, Q2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Charge conserved: C1*V1={Q1:.3f} ≈ C2*V2={Q2:.3f}")
        return (False, f"Charge not conserved: C1*V1={Q1:.3f} != C2*V2={Q2:.3f} (rel diff {abs(Q1-Q2)/max(abs(Q1),1):.3%})")

    def test_invariant_ratio_relation(self):
        """Check V2 = (C1/C2)*V1 with tolerance."""
        if any(v is None for v in [self.C1, self.C2, self.V1, self.V2]):
            return (False, "SKIP: missing data for ratio check")
        expected = (self.C1 / self.C2) * self.V1
        if math.isclose(self.V2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V2={self.V2:.3f} ≈ (C1/C2)*V1={expected:.3f}")
        return (False, f"V2={self.V2:.3f} not close to (C1/C2)*V1={expected:.3f} (rel diff {abs(self.V2-expected)/max(abs(expected),1e-12):.3%})")

    def test_invariant_sign_consistency(self):
        """V1 and V2 must have the same sign (since Q conserved, C positive)."""
        if any(v is None for v in [self.V1, self.V2]):
            return (False, "SKIP: missing V1 or V2")
        if (self.V1 > 0 and self.V2 > 0) or (self.V1 < 0 and self.V2 < 0) or (self.V1 == 0 and self.V2 == 0):
            return (True, f"V1={self.V1} and V2={self.V2} have same sign")
        return (False, f"V1={self.V1} and V2={self.V2} have opposite signs – violates charge conservation")

    def test_invariant_inequality(self):
        """Since C2 < C1, |V2| > |V1| (assuming V1≠0)."""
        if any(v is None for v in [self.C1, self.C2, self.V1, self.V2]):
            return (False, "SKIP: missing data for inequality check")
        if self.C2 < self.C1 and abs(self.V2) < abs(self.V1) - 1e-12:
            return (False, f"C2 < C1 but |V2|={abs(self.V2)} < |V1|={abs(self.V1)} – should increase")
        return (True, f"C2={self.C2} < C1={self.C1} and |V2|={abs(self.V2)} >= |V1|={abs(self.V1)}")

    # ==================== Modeling / Trace Tests ====================

    def test_modeling_charge_conservation_mentioned(self):
        """Check that explanation_steps (if present) mention 'charge conservation' or 'Q = CV'."""
        steps = self.explanation_steps
        if not steps:
            return (True, "SKIP: no explanation_steps available")
        # Look for key phrases
        phrases = ['charge conservation', 'q = cv', 'charge is constant']
        found = [p for p in phrases if p in steps.lower()]
        if found:
            return (True, f"Explanation steps mention charge conservation: {found}")
        return (False, "Explanation steps do not mention charge conservation or Q=CV")

    def test_modeling_inverse_relationship_mentioned(self):
        """Check that explanation_steps mention the inverse relation between V and C."""
        steps = self.explanation_steps
        if not steps:
            return (True, "SKIP: no explanation_steps available")
        # Look for phrases like "inversely proportional", "increases as C decreases", etc.
        phrases = ['inversely proportional', 'increases as the capacitance decreases',
                    'decreases as the capacitance increases', 'potential increases']
        found = [p for p in phrases if p in steps.lower()]
        if found:
            return (True, f"Explanation steps mention inverse relationship: {found}")
        return (False, "Explanation steps do not mention the inverse relationship between V and C")

    def test_modeling_final_formula_present(self):
        """Check that explanation_steps contain the final formula V2 = (C1/C2) V1."""
        steps = self.explanation_steps
        if not steps:
            return (True, "SKIP: no explanation_steps available")
        # Look for the formula in various textual forms
        formula_patterns = ['v2 = (c1 / c2) * v1', 'v2 = c1/c2 v1',
                            'v2 = \\frac{c1}{c2} v1', 'v2 = (c1/c2)v1']
        found = any(p in steps.lower() for p in formula_patterns)
        if found:
            return (True, "Final formula found in explanation steps")
        return (False, "Final formula V2 = (C1/C2) V1 not found in explanation steps")

    # ==================== Monotonicity Tests ====================

    def test_monotonicity_V2_increases_when_C2_decreases(self):
        """Perturb C2 downward: if C2 were smaller (e.g., 0.9*C2), then V2 should be larger."""
        if any(v is None for v in [self.C1, self.C2, self.V1, self.V2]):
            return (False, "SKIP: missing data for monotonicity check")
        # Simulate a 10% decrease in C2
        C2_prime = 0.9 * self.C2
        if C2_prime <= 0:
            return (True, "SKIP: cannot decrease C2 further (non-positive)")
        V2_prime = (self.C1 / C2_prime) * self.V1
        if self.V2 < V2_prime:
            return (True, f"Original V2={self.V2:.3f} < V2'={V2_prime:.3f} (for 10% smaller C2) – monotonicity holds")
        return (False, f"V2={self.V2:.3f} not less than V2'={V2_prime:.3f} – monotonicity violated (should increase when C2 decreases)")

    def test_monotonicity_V2_decreases_when_C2_increases(self):
        """Perturb C2 upward: if C2 were larger (e.g., 1.1*C2), then V2 should be smaller."""
        if any(v is None for v in [self.C1, self.C2, self.V1, self.V2]):
            return (False, "SKIP: missing data for monotonicity check")
        C2_prime = 1.1 * self.C2
        V2_prime = (self.C1 / C2_prime) * self.V1
        if self.V2 > V2_prime:
            return (True, f"Original V2={self.V2:.3f} > V2'={V2_prime:.3f} (for 10% larger C2) – monotonicity holds")
        return (False, f"V2={self.V2:.3f} not greater than V2'={V2_prime:.3f} – monotonicity violated (should decrease when C2 increases)")

    # ==================== Weak Answer Sanity Test (only one) ====================

    def test_answer_sanity_V2_finite_and_non_nan(self):
        """V2 must be finite and not NaN (weak sanity)."""
        if self.V2 is None:
            return (False, "SKIP: V2 is None")
        if math.isfinite(self.V2) and not math.isnan(self.V2):
            return (True, f"V2={self.V2} is finite and not NaN")
        return (False, f"V2={self.V2} is either infinite or NaN")

    # ==================== Runner ====================

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