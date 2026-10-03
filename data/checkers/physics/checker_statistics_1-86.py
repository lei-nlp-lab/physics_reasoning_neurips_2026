import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for van der Waals gas problem.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract givens (numerical inputs)
        self.V1 = getattr(solution_instance, 'V1', None)
        if self.V1 is None:
            self.V1 = solution_instance.results.get('V1', None) if hasattr(solution_instance, 'results') else None

        self.V2 = getattr(solution_instance, 'V2', None)
        if self.V2 is None:
            self.V2 = solution_instance.results.get('V2', None) if hasattr(solution_instance, 'results') else None

        self.T = getattr(solution_instance, 'T', None)
        if self.T is None:
            self.T = solution_instance.results.get('T', None) if hasattr(solution_instance, 'results') else None

        self.a = getattr(solution_instance, 'a', None)
        if self.a is None:
            self.a = solution_instance.results.get('a', None) if hasattr(solution_instance, 'results') else None

        self.b = getattr(solution_instance, 'b', None)
        if self.b is None:
            self.b = solution_instance.results.get('b', None) if hasattr(solution_instance, 'results') else None

        self.R = getattr(solution_instance, 'R', None)
        if self.R is None:
            self.R = solution_instance.results.get('R', None) if hasattr(solution_instance, 'results') else None

        # Safe extraction of emitted results
        self.origin_a = solution_instance.results.get('origin_a', None) if hasattr(solution_instance, 'results') else None
        self.origin_b = solution_instance.results.get('origin_b', None) if hasattr(solution_instance, 'results') else None
        self.deltaF = solution_instance.results.get('deltaF', None) if hasattr(solution_instance, 'results') else None
        self.deltaU = solution_instance.results.get('deltaU', None) if hasattr(solution_instance, 'results') else None

        # Explanation steps (for modeling checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface checks: required keys and types
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required results exist."""
        required = [('origin_a', type(None), str),
                    ('origin_b', type(None), str),
                    ('deltaF', (type(None), int, float)),
                    ('deltaU', (type(None), int, float))]
        for name, allow_none, expected_types in required:
            val = getattr(self, name, None)
            if val is None:
                return (False, f"Missing result '{name}'")
            if not isinstance(val, expected_types):
                return (False, f"'{name}' has unexpected type {type(val).__name__}")
        return (True, "All required keys present with correct types")

    def test_interface_givens_present(self):
        """Check that all given parameters are available."""
        given_names = ['V1', 'V2', 'T', 'a', 'b', 'R']
        for name in given_names:
            val = getattr(self, name, None)
            if val is None:
                return (False, f"Missing given parameter '{name}'")
            if not isinstance(val, (int, float)):
                return (False, f"Given '{name}' has non-numeric type {type(val).__name__}")
        return (True, "All given parameters present and numeric")

    # ----------------------------------------------------------------------
    # Invariant checks: theorem‐based sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_deltaF_expression(self):
        """Check that deltaF matches the integral of -p dV."""
        if None in (self.V1, self.V2, self.T, self.a, self.b, self.R, self.deltaF):
            return (False, "SKIP: missing numeric data for deltaF check")
        # Compute expected deltaF
        expected = -self.R * self.T * math.log((self.V2 - self.b) / (self.V1 - self.b)) \
                   + self.a * (1.0 / self.V1 - 1.0 / self.V2)
        if math.isclose(self.deltaF, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"deltaF matches expected value within 1%")
        else:
            return (False, f"deltaF = {self.deltaF}, expected ≈ {expected}")

    def test_invariant_deltaU_expression(self):
        """Check that deltaU matches a(1/V1 - 1/V2)."""
        if None in (self.V1, self.V2, self.a, self.deltaU):
            return (False, "SKIP: missing numeric data for deltaU check")
        expected = self.a * (1.0 / self.V1 - 1.0 / self.V2)
        if math.isclose(self.deltaU, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "deltaU matches expected value within 1%")
        else:
            return (False, f"deltaU = {self.deltaU}, expected ≈ {expected}")

    def test_invariant_maxwell_relation(self):
        """Check that (dU/dV)_T = a/V^2 from Maxwell relation."""
        # We re‐derive (∂U/∂V)_T from p = RT/(V-b) - a/V^2
        if None in (self.T, self.a, self.b, self.R):
            return (False, "SKIP: missing givens for Maxwell check")
        # Use symbolic check: (∂U/∂V)_T = T*(∂p/∂T)_V - p
        V = sp.symbols('V')
        p_expr = self.R * self.T / (V - self.b) - self.a / V**2
        dpdT_expr = sp.diff(p_expr, self.T)  # ∂p/∂T at constant V
        dU_dV_expr = self.T * dpdT_expr - p_expr
        simplified = sp.simplify(dU_dV_expr)
        expected = self.a / V**2
        if simplified == expected:
            return (True, "Maxwell relation yields (dU/dV)_T = a/V^2")
        else:
            return (False, f"Maxwell gives {simplified}, expected {expected}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks: using explanation_steps and key constraints
    # ----------------------------------------------------------------------
    def test_modeling_origin_a(self):
        """Check that origin_a mentions intermolecular attraction or similar."""
        if self.origin_a is None:
            return (False, "SKIP: origin_a not provided")
        keywords = ['attract', 'intermolecular', 'cohesion', 'pairwise']
        if any(kw.lower() in self.origin_a.lower() for kw in keywords):
            return (True, "origin_a correctly describes attractive forces")
        else:
            return (False, f"origin_a ('{self.origin_a}') does not mention attraction")

    def test_modeling_origin_b(self):
        """Check that origin_b mentions excluded volume or finite size."""
        if self.origin_b is None:
            return (False, "SKIP: origin_b not provided")
        keywords = ['excluded volume', 'finite size', 'molecular volume', 'excluded']
        if any(kw.lower() in self.origin_b.lower() for kw in keywords):
            return (True, "origin_b correctly describes excluded volume")
        else:
            return (False, f"origin_b ('{self.origin_b}') does not mention excluded volume")

    def test_modeling_explanation_contains_integral(self):
        """Check that explanation_steps mention the integral for ΔF."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = str(self.explanation_steps)
        if 'integral' in text.lower() or '∫' in text:
            return (True, "Explanation steps contain integral for ΔF")
        else:
            return (False, "Explanation steps do not mention integral for ΔF")

    def test_modeling_volumes_greater_than_b(self):
        """Check that V1 > b and V2 > b (physical constraint)."""
        if None in (self.V1, self.V2, self.b):
            return (False, "SKIP: missing V1, V2, or b")
        if not (self.V1 > self.b and self.V2 > self.b):
            return (False, f"V1={self.V1} and/or V2={self.V2} not > b={self.b}")
        return (True, "Volumes satisfy V1, V2 > b")

    # ----------------------------------------------------------------------
    # Monotonicity checks: perturb an input and observe sign of change
    # ----------------------------------------------------------------------
    def test_monotonicity_deltaF_with_V2(self):
        """Increasing V2 should decrease ΔF (more negative)."""
        if None in (self.V1, self.V2, self.T, self.a, self.b, self.R, self.deltaF):
            return (False, "SKIP: missing data for monotonicity")
        # baselines
        V2_orig = self.V2
        delF_orig = self.deltaF
        # perturb V2 upward by 1%
        V2_new = V2_orig * 1.01
        # recompute expected ΔF with new V2 (same V1, etc.)
        # (We cannot modify solution, so we compute using formula)
        delF_new = -self.R * self.T * math.log((V2_new - self.b) / (self.V1 - self.b)) \
                   + self.a * (1.0 / self.V1 - 1.0 / V2_new)
        # For isothermal expansion ΔF should become more negative as V2 increases
        if delF_new < delF_orig:
            return (True, "Increasing V2 makes ΔF more negative")
        else:
            return (False, f"Monotonicity violated: ΔF increased from {delF_orig} to {delF_new}")
    
    def test_monotonicity_deltaU_with_V1(self):
        """Increasing V1 (starting volume) should decrease ΔU (since ΔU = a(1/V1 - 1/V2))."""
        if None in (self.V1, self.V2, self.a, self.deltaU):
            return (False, "SKIP: missing data for monotonicity")
        # Compute ΔU for a slightly larger V1 (keeping V2 fixed)
        V1_new = self.V1 * 1.01
        deltaU_new = self.a * (1.0 / V1_new - 1.0 / self.V2)
        # Since 1/V1 decreases, ΔU should be smaller (or less positive)
        if deltaU_new < self.deltaU:
            return (True, "Increasing V1 decreases ΔU")
        else:
            return (False, f"Monotonicity violated: ΔU increased from {self.deltaU} to {deltaU_new}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (no isclose, no re‐derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Weak sanity: deltaU not NaN/finite; deltaF not NaN/finite; origins nonempty."""
        if self.deltaU is not None:
            if not math.isfinite(self.deltaU):
                return (False, "deltaU is not finite")
        if self.deltaF is not None:
            if not math.isfinite(self.deltaF):
                return (False, "deltaF is not finite")
        if not self.origin_a or not self.origin_b:
            return (False, "origin_a or origin_b empty strings")
        return (True, "Basic sanity checks passed")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
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
                passed, msg = (False, f"{name} CRASHED: {traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)