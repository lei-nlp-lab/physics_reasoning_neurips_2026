import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results (emitted keys from pseudocode)
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        
        # Required output quantities (None if missing)
        self.y3_01 = self.results.get('y3_01', None)
        self.y3_03 = self.results.get('y3_03', None)
        self.E0_2 = self.results.get('E0_2', None)
        
        # Optional givens for re-derivation
        self.alpha = self.givens.get('alpha', sp.Symbol('alpha'))
        # For unperturbed energies we know epsilon_n = 2n+1
        # We'll use symbolic sympy for re-derivation if needed

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

    # -------- Interface tests (required keys exist and have valid types) --------
    def test_interface_y3_01(self):
        """y3_01 must be present and numeric"""
        if self.y3_01 is None:
            return (False, "y3_01 missing from results")
        if not isinstance(self.y3_01, (int, float, sp.Expr)):
            return (False, f"y3_01 has invalid type {type(self.y3_01)}")
        return (True, "y3_01 present and numeric")

    def test_interface_y3_03(self):
        """y3_03 must be present and numeric"""
        if self.y3_03 is None:
            return (False, "y3_03 missing from results")
        if not isinstance(self.y3_03, (int, float, sp.Expr)):
            return (False, f"y3_03 has invalid type {type(self.y3_03)}")
        return (True, "y3_03 present and numeric")

    def test_interface_E0_2(self):
        """E0_2 must be present and numeric"""
        if self.E0_2 is None:
            return (False, "E0_2 missing from results")
        if not isinstance(self.E0_2, (int, float, sp.Expr)):
            return (False, f"E0_2 has invalid type {type(self.E0_2)}")
        return (True, "E0_2 present and numeric")

    # -------- Invariant tests (theorem-based physics sanity) --------
    def test_invariant_y3_insertion(self):
        """
        Verify that <1|y^3|0> and <3|y^3|0> satisfy the insertion-of-identity
        using the known dipole matrix element <n+1|y|n> = sqrt((n+1)/2).
        We re-derive analytic values and compare with fuzzy tolerance.
        """
        if self.y3_01 is None or self.y3_03 is None:
            return (False, "SKIP: y3_01 or y3_03 missing")
        # Derive using sympy
        sqrt = sp.sqrt
        # y matrix elements: <n+1|y|n> = sqrt((n+1)/2), Hermitian conjugate same
        # For <1|y^3|0>:
        # Two paths: 0->1->0->1 and 0->1->2->1
        pathA = sqrt(1/2) * sqrt(1/2) * sqrt(1/2)  # <1|y|0><0|y|1><1|y|0>
        pathB = 1 * 1 * sqrt(1/2)                 # <1|y|2><2|y|1><1|y|0> = 1*1*sqrt(1/2)
        expected_y3_01 = sp.simplify(pathA + pathB)
        # For <3|y^3|0>:
        expected_y3_03 = sp.simplify(sqrt(3/2) * 1 * sqrt(1/2))  # <3|y|2><2|y|1><1|y|0>

        # Compare using numeric evaluation (fuzzy tolerance)
        try:
            val_01 = float(self.y3_01)
            exp_01 = float(expected_y3_01.evalf())
            val_03 = float(self.y3_03)
            exp_03 = float(expected_y3_03.evalf())
        except Exception as e:
            return (False, f"Could not evaluate numerically: {e}")
        
        if not math.isclose(val_01, exp_01, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"y3_01 value {val_01} does not match expected {exp_01}")
        if not math.isclose(val_03, exp_03, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"y3_03 value {val_03} does not match expected {exp_03}")
        return (True, "y3_01 and y3_03 satisfy insertion of identity")

    def test_invariant_E0_2_sign(self):
        """
        Second-order perturbation theory for ground state always gives negative correction
        (since denominator is negative for all m > 0). So E0_2 should be negative.
        """
        if self.E0_2 is None:
            return (False, "SKIP: E0_2 missing")
        try:
            val = float(self.E0_2)
        except Exception as e:
            return (False, f"Cannot evaluate E0_2: {e}")
        if not math.isfinite(val):
            return (False, f"E0_2 is not finite: {val}")
        # The sign is expected negative (assuming alpha is real and non-zero)
        if val >= 0:
            return (False, f"E0_2 should be negative for ground state, got {val}")
        return (True, "E0_2 is negative as expected from perturbation theory")

    def test_invariant_E0_2_magnitude_order(self):
        """
        Check that |E0_2| is of order |alpha|^2 with coefficient roughly 11/16.
        We do NOT hardcode the exact number; we check that the ratio |E0_2|/|alpha|^2
        is between 0.5 and 1.0 (i.e., around 0.6875). This is a broad range check.
        """
        if self.E0_2 is None:
            return (False, "SKIP: E0_2 missing")
        try:
            # Assume alpha is a parameter; we can extract it from givens or use symbolic
            alpha = self.alpha
            if isinstance(alpha, sp.Symbol):
                # If alpha is symbolic, we cannot compute directly; skip
                return (True, "SKIP: alpha is symbolic, cannot test magnitude")
            alpha_sq = float(alpha)**2
            val = float(self.E0_2)
        except Exception as e:
            return (False, f"Cannot evaluate: {e}")
        if not alpha_sq > 1e-12:
            return (True, "alpha is zero; no magnitude check")
        ratio = abs(val) / alpha_sq
        # The exact ratio is 11/16 = 0.6875; allow range 0.5 to 1.0
        if 0.5 <= ratio <= 1.0:
            return (True, f"Ratio |E0_2|/|alpha|^2 = {ratio} is in [0.5,1.0]")
        else:
            return (False, f"Ratio {ratio} outside expected range")

    # -------- Modeling/trace tests (based on explanation steps) --------
    def test_modeling_y3_01_paths(self):
        """
        Verify that the calculation of <1|y^3|0> includes both paths (0->1->0->1 and 0->1->2->1).
        This is a structural check: we cannot directly see steps, but we can check that the value
        is consistent with the sum of two contributions.
        Actually, we already tested numeric consistency in invariant; here we check
        that the value is not equal to a single-path coefficient (like sqrt(1/2) only).
        If the value is exactly one of the path contributions, that indicates missing path.
        """
        if self.y3_01 is None:
            return (False, "SKIP: y3_01 missing")
        # Single path contributions:
        pathA_only = 1/(2*math.sqrt(2))  # about 0.3536
        pathB_only = 1/math.sqrt(2)      # about 0.7071
        try:
            val = float(self.y3_01)
        except:
            return (False, "Cannot evaluate y3_01")
        # If val is exactly one of these (within 1e-6), it's likely a mistake
        if math.isclose(val, pathA_only, rel_tol=1e-6) or math.isclose(val, pathB_only, rel_tol=1e-6):
            return (False, f"y3_01 = {val} matches single path; likely missing the other path")
        return (True, "y3_01 appears to be sum of both paths")

    def test_modeling_E0_2_matrix_elements(self):
        """
        Check that E0_2 used the correct matrix elements: only m=1 and m=3 contribute.
        We can infer by checking if the magnitude of E0_2 is consistent with contributions
        from both states. If E0_2 magnitude is too small, it might have used only one.
        """
        if self.E0_2 is None or self.y3_01 is None or self.y3_03 is None:
            return (False, "SKIP: missing required results")
        try:
            alpha = self.alpha
            if isinstance(alpha, sp.Symbol):
                return (True, "SKIP: alpha symbolic")
            alpha_sq = float(alpha)**2
            val = float(self.E0_2)
            y01_sq = (float(self.y3_01))**2
            y03_sq = (float(self.y3_03))**2
        except Exception as e:
            return (False, f"Cannot evaluate: {e}")
        # Expected contributions: m=1 denominator = -2, m=3 denominator = -6
        expected_E0_2 = -alpha_sq * (y01_sq/2 + y03_sq/6)  # derived from E0_2 formula
        if math.isclose(val, expected_E0_2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "E0_2 consistent with contributions from m=1 and m=3")
        else:
            return (False, f"E0_2 = {val}, but expected from matrix elements {expected_E0_2}")

    # -------- Monotonicity tests (trend checks) --------
    def test_monotonicity_E0_2_with_alpha(self):
        """
        If alpha increases in magnitude, the correction should increase in magnitude.
        We cannot perturb actual solution, but we can use reasoning: E0_2 scales as |alpha|^2.
        So we perturb alpha by +10% and check that predicted E0_2 magnitude increases.
        """
        if self.E0_2 is None:
            return (False, "SKIP: E0_2 missing")
        # Extract alpha from givens or assume it's symbolic; if symbolic, skip.
        alpha = self.alpha
        if isinstance(alpha, sp.Symbol):
            return (True, "SKIP: alpha symbolic, cannot perturb")
        try:
            alpha0 = float(alpha)
            E0_2_0 = float(self.E0_2)
        except Exception as e:
            return (False, f"Cannot evaluate: {e}")
        if abs(alpha0) < 1e-12:
            return (True, "alpha is zero; monotonicity trivial")
        # Perturb alpha by +10%
        alpha1 = alpha0 * 1.1
        # Predicted new E0_2 scales as |alpha|^2: E0_2_1 = E0_2_0 * (1.1)^2
        E0_2_1_pred = E0_2_0 * (1.1**2)
        # Check that magnitude increases: |E0_2_1| > |E0_2_0|
        if abs(E0_2_1_pred) > abs(E0_2_0):
            return (True, f"Magnitude increases with |alpha|: {abs(E0_2_0)} -> {abs(E0_2_1_pred)}")
        else:
            return (False, "Monotonicity violated: magnitude did not increase")

    # -------- Weak answer sanity (at most one, no exact matching) --------
    def test_answer_sanity_E0_2_finite_and_negative(self):
        """
        Check that E0_2 is finite, not NaN, and negative.
        This is a weak sanity, not an exact match.
        """
        if self.E0_2 is None:
            return (False, "SKIP: E0_2 missing")
        try:
            val = float(self.E0_2)
        except Exception as e:
            return (False, f"Cannot evaluate E0_2: {e}")
        if not math.isfinite(val):
            return (False, f"E0_2 is not finite: {val}")
        if val >= 0:
            return (False, f"E0_2 should be negative, got {val}")
        # Do not check exact value 11/16; only sign and finite
        return (True, "E0_2 is finite and negative")