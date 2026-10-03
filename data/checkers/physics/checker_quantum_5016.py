import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audit the rigid rotator in weak electric field solution.
    Checks interface completeness, physical invariants, modeling quality,
    and monotonicity via explicit input perturbation.
    """
    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', {})

        # Emitted strings (from pseudocode contract)
        self.H = self.results.get('H', None)
        self.H0 = self.results.get('H0', None)
        self.Hprime = self.results.get('Hprime', None)
        self.E0_lm = self.results.get('E0_lm', None)
        self.degeneracy = self.results.get('degeneracy', None)
        self.E2_lm_str = self.results.get('E2_lm', None)

        # Numerical parameters (givens). Try both direct attribute and results dict.
        self.I = getattr(solution_instance, 'I', self.results.get('I', None))
        self.d = getattr(solution_instance, 'd', self.results.get('d', None))
        self.E = getattr(solution_instance, 'E', self.results.get('E', None))
        self.hbar = getattr(solution_instance, 'hbar', self.results.get('hbar', None))
        self.l = getattr(solution_instance, 'l', self.results.get('l', None))
        self.m = getattr(solution_instance, 'm', self.results.get('m', None))

        # Explanation steps (if present) for trace consistency
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks: required keys and types
    # ------------------------------------------------------------------
    def test_interface_H_key(self):
        if self.results is None:
            return (False, "results dict missing")
        if 'H' not in self.results:
            return (False, "missing key 'H'")
        return (True, "key 'H' present")

    def test_interface_H0_key(self):
        if self.results is None:
            return (False, "results dict missing")
        if 'H0' not in self.results:
            return (False, "missing key 'H0'")
        return (True, "key 'H0' present")

    def test_interface_Hprime_key(self):
        if self.results is None:
            return (False, "results dict missing")
        if 'Hprime' not in self.results:
            return (False, "missing key 'Hprime'")
        return (True, "key 'Hprime' present")

    def test_interface_E0_lm_key(self):
        if self.results is None:
            return (False, "results dict missing")
        if 'E0_lm' not in self.results:
            return (False, "missing key 'E0_lm'")
        return (True, "key 'E0_lm' present")

    def test_interface_degeneracy_key(self):
        if self.results is None:
            return (False, "results dict missing")
        if 'degeneracy' not in self.results:
            return (False, "missing key 'degeneracy'")
        return (True, "key 'degeneracy' present")

    def test_interface_E2_lm_key(self):
        if self.results is None:
            return (False, "results dict missing")
        if 'E2_lm' not in self.results:
            return (False, "missing key 'E2_lm'")
        return (True, "key 'E2_lm' present")

    def test_interface_H_type(self):
        if self.H is None:
            return (False, "H is missing, cannot check type")
        if not isinstance(self.H, str):
            return (False, f"H should be a string, got {type(self.H)}")
        return (True, "H is string")

    def test_interface_H0_type(self):
        if self.H0 is None:
            return (False, "H0 is missing")
        if not isinstance(self.H0, str):
            return (False, f"H0 should be string, got {type(self.H0)}")
        return (True, "H0 is string")

    def test_interface_Hprime_type(self):
        if self.Hprime is None:
            return (False, "Hprime is missing")
        if not isinstance(self.Hprime, str):
            return (False, f"Hprime should be string, got {type(self.Hprime)}")
        return (True, "Hprime is string")

    def test_interface_E0_lm_type(self):
        if self.E0_lm is None:
            return (False, "E0_lm is missing")
        if not isinstance(self.E0_lm, str):
            return (False, f"E0_lm should be string, got {type(self.E0_lm)}")
        return (True, "E0_lm is string")

    def test_interface_degeneracy_type(self):
        if self.degeneracy is None:
            return (False, "degeneracy is missing")
        if not isinstance(self.degeneracy, str):
            return (False, f"degeneracy should be string, got {type(self.degeneracy)}")
        return (True, "degeneracy is string")

    def test_interface_E2_lm_type(self):
        if self.E2_lm_str is None:
            return (False, "E2_lm is missing")
        if not isinstance(self.E2_lm_str, str):
            return (False, f"E2_lm should be string, got {type(self.E2_lm_str)}")
        return (True, "E2_lm is string")

    # ------------------------------------------------------------------
    # String content checks (basic sanity of emitted formulas)
    # ------------------------------------------------------------------
    def test_invariant_H_contains_L2(self):
        if self.H is None:
            return (False, "SKIP: H not available")
        # Accept various notations: L^2, L^{2}, mathbf{L}^2, etc.
        if 'L^2' in self.H or 'L^{2}' in self.H or 'mathbf{L}^2' in self.H:
            return (True, "H contains L^2 term")
        return (False, "H missing L^2 term")

    def test_invariant_H_contains_dE_cos(self):
        if self.H is None:
            return (False, "SKIP: H not available")
        if 'dE' in self.H and ('cos' in self.H or '\cos' in self.H):
            return (True, "H contains dE cos theta")
        return (False, "H missing dE cos theta term")

    def test_invariant_H0_contains_L2(self):
        if self.H0 is None:
            return (False, "SKIP: H0 not available")
        if 'L^2' in self.H0 or 'L^{2}' in self.H0 or 'mathbf{L}^2' in self.H0:
            return (True, "H0 contains L^2")
        return (False, "H0 missing L^2")

    def test_invariant_Hprime_contains_dE_cos(self):
        if self.Hprime is None:
            return (False, "SKIP: Hprime not available")
        if 'dE' in self.Hprime and ('cos' in self.Hprime or '\cos' in self.Hprime):
            return (True, "Hprime contains dE cos theta")
        return (False, "Hprime missing dE cos theta")

    def test_invariant_E0_lm_contains_hbar_l(self):
        if self.E0_lm is None:
            return (False, "SKIP: E0_lm not available")
        if 'hbar' in self.E0_lm and ('l' in self.E0_lm or 'L' in self.E0_lm):
            return (True, "E0_lm mentions hbar and l")
        return (False, "E0_lm missing expected variables")

    def test_invariant_degeneracy_contains_2l_plus_1(self):
        if self.degeneracy is None:
            return (False, "SKIP: degeneracy not available")
        if '2l+1' in self.degeneracy or '2l + 1' in self.degeneracy or '(2l+1)' in self.degeneracy:
            return (True, "degeneracy mentions 2l+1")
        return (False, "degeneracy does not express 2l+1")

    def test_invariant_E2_lm_formula_skeleton(self):
        """Check that the E2_lm string contains key factors: I, d, E, hbar, l(l+1), m^2."""
        if self.E2_lm_str is None:
            return (False, "SKIP: E2_lm not available")
        s = self.E2_lm_str
        # Must have I, d, E, hbar
        if 'I' not in s or 'd' not in s or 'E' not in s or 'hbar' not in s:
            return (False, "E2_lm missing one of I,d,E,hbar")
        if 'l(l+1)' not in s and 'l(l + 1)' not in s:
            return (False, "E2_lm missing l(l+1) in numerator or denominator")
        if 'm^2' not in s and 'm**2' not in s:
            return (False, "E2_lm missing m^2 term")
        if '(2l-1)' in s and '(2l+3)' in s:
            return (True, "E2_lm contains correct denominator structure")
        if '(2*l-1)' in s and '(2*l+3)' in s:
            return (True, "E2_lm contains correct denominator structure")
        return (False, "E2_lm denominator does not match expected (2l-1)(2l+3)")

    # ------------------------------------------------------------------
    # Invariant: first-order correction is zero (by parity)
    # ------------------------------------------------------------------
    def test_invariant_first_order_zero_parity(self):
        """Check that the perturbation H' has no diagonal matrix elements because cosθ couples different l."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps to check")
        text = str(self.explanation_steps).lower()
        if 'first order' in text and ('zero' in text or '0' in text):
            return (True, "First-order correction mentioned as zero")
        return (False, "First-order correction not indicated as zero")

    # ------------------------------------------------------------------
    # Modeling check: second-order correction formula numerically correct for a sample (l,m)
    # ------------------------------------------------------------------
    def test_modeling_e2_lm_numerical_sample(self):
        """Use a specific (l,m) to compare student's symbolic formula with derived formula."""
        # Need numerical parameters
        if any(v is None for v in [self.I, self.d, self.E, self.hbar, self.l, self.m]):
            return (False, "SKIP: one or more numerical parameters missing")
        if self.E2_lm_str is None:
            return (False, "SKIP: E2_lm string missing")

        # Define symbols and the derived expression
        I_s, d_s, E_s, hbar_s, l_s, m_s = sp.symbols('I d E hbar l m')
        # Derived formula from standard result
        E2_derived = (I_s * d_s**2 * E_s**2 * (l_s*(l_s+1) - 3*m_s**2) /
                      (hbar_s**2 * l_s * (l_s+1) * (2*l_s-1) * (2*l_s+3)))
        # Subs numerical values
        subs_dict = {I_s: self.I, d_s: self.d, E_s: self.E, hbar_s: self.hbar,
                     l_s: self.l, m_s: self.m}
        expected_num = float(E2_derived.subs(subs_dict).evalf())

        # Parse student string
        try:
            # Replace common notation: 'hbar'->'hbar_s', etc. Use parse_expr.
            student_expr = sp.parse_expr(self.E2_lm_str)
            # Assume student uses same variable names, substitute
            student_num = float(student_expr.subs(subs_dict).evalf())
        except Exception as e:
            return (False, f"Cannot parse/evaluate student E2_lm: {e}")

        # Compare with fuzzy tolerance
        if math.isclose(expected_num, student_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E2_lm numerical match: student={student_num}, expected={expected_num}")
        else:
            return (False, f"E2_lm numerical mismatch: student={student_num}, expected={expected_num}")

    # ------------------------------------------------------------------
    # Modeling check: verify that for l=0, the second‑order correction is zero
    # ------------------------------------------------------------------
    def test_modeling_e2_l0_m0_zero(self):
        """For l=0,m=0 the correction should vanish. The student's formula may be singular, but they should handle it."""
        if self.E2_lm_str is None:
            return (False, "SKIP: E2_lm string missing")
        if self.l is not None and self.l != 0:
            return (False, "SKIP: l is not 0 for this test")
        if self.l is None:
            return (False, "SKIP: l not provided")
        # Try to evaluate the student expression at l=0, m=0 using symbolic limit.
        try:
            I_s, d_s, E_s, hbar_s, l_s, m_s = sp.symbols('I d E hbar l m')
            student_expr = sp.parse_expr(self.E2_lm_str)
            limit_val = sp.limit(student_expr.subs({I_s: 1, d_s: 1, E_s: 1, hbar_s: 1, m_s: 0}), l_s, 0)
            if sp.simplify(limit_val) == 0:
                return (True, "E2_lm for l=0,m=0 gives 0 (limit)")
            else:
                return (False, f"Limit of E2_lm at l=0,m=0 is {limit_val}, expected 0")
        except Exception as e:
            return (False, f"Cannot compute limit: {e}. Accept if student explicitly states zero.")

    # ------------------------------------------------------------------
    # Monotonicity: perturb electric field E and verify E2 scales quadratically.
    # ------------------------------------------------------------------
    def test_monotonicity_e2_scales_with_E_squared(self):
        """Perturb the electric field strength E by +10% and check that E2 changes by factor ~1.21."""
        if self.E2_lm_str is None:
            return (False, "SKIP: E2_lm string missing")
        # We need numerical parameters to get a baseline numeric. Use arbitrary defaults if not provided.
        I_val = self.I if self.I is not None else 1.0
        d_val = self.d if self.d is not None else 1.0
        E_val = self.E if self.E is not None else 1.0
        hbar_val = self.hbar if self.hbar is not None else 1.0
        l_val = self.l if self.l is not None else 2
        m_val = self.m if self.m is not None else 0

        try:
            I_s, d_s, E_s, hbar_s, l_s, m_s = sp.symbols('I d E hbar l m')
            student_expr = sp.parse_expr(self.E2_lm_str)
            subs_baseline = {I_s: I_val, d_s: d_val, E_s: E_val,
                             hbar_s: hbar_val, l_s: l_val, m_s: m_val}
            baseline = float(student_expr.subs(subs_baseline).evalf())

            # Perturb E by +10%
            E_pert = 1.1 * E_val
            subs_pert = {I_s: I_val, d_s: d_val, E_s: E_pert,
                         hbar_s: hbar_val, l_s: l_val, m_s: m_val}
            perturbed = float(student_expr.subs(subs_pert).evalf())

            # Expect E2 scales as E^2, so perturbed/baseline ≈ (1.1)^2 = 1.21
            # Use relative tolerance 5% to allow rounding
            expected_ratio = 1.21
            if abs(baseline) < 1e-12:
                # If baseline is zero, perturbed should also be zero (or very small)
                if abs(perturbed) < 1e-12:
                    return (True, "Both baseline and perturbed E2 are zero; scale holds trivially")
                else:
                    return (False, "Baseline E2 is zero but perturbed is non-zero")
            actual_ratio = perturbed / baseline
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E2 scales as E^2: baseline={baseline}, perturbed={perturbed}, ratio={actual_ratio} ~ 1.21")
            else:
                return (False, f"E2 does not scale as E^2: baseline={baseline}, perturbed={perturbed}, ratio={actual_ratio} != 1.21")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ------------------------------------------------------------------
    # Monotonicity: perturb moment of inertia I and check that E0 decreases.
    # ------------------------------------------------------------------
    def test_monotonicity_unperturbed_energy_decreases_with_I(self):
        """Perturb I by +10% and verify unperturbed energy E0 decreases."""
        if self.E0_lm is None:
            return (False, "SKIP: E0_lm string missing")
        # Use default values if not provided
        hbar_val = self.hbar if self.hbar is not None else 1.0
        I_val = self.I if self.I is not None else 2.0
        l_val = self.l if self.l is not None else 1

        try:
            hbar_s, I_s, l_s = sp.symbols('hbar I l')
            expr = sp.parse_expr(self.E0_lm)
            subs_baseline = {hbar_s: hbar_val, I_s: I_val, l_s: l_val}
            baseline = float(expr.subs(subs_baseline).evalf())

            # Perturb I by +10%
            I_pert = 1.1 * I_val
            subs_pert = {hbar_s: hbar_val, I_s: I_pert, l_s: l_val}
            perturbed = float(expr.subs(subs_pert).evalf())

            if baseline < 0:
                # If baseline is negative, perturbed should be less negative (i.e., larger)
                # Energy decreases in magnitude? Actually E0 is always positive.
                # But handle general case: check that perturbed < baseline if both positive.
                pass
            if perturbed < baseline:
                return (True, f"E0 decreases with increasing I: baseline={baseline}, perturbed={perturbed}")
            else:
                return (False, f"E0 does not decrease: baseline={baseline}, perturbed={perturbed}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ------------------------------------------------------------------
    # Optional weak answer‑sanity check (not dominant)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_non_nan(self):
        """Check that final answer strings are finite length (not huge) and not empty."""
        checks = [
            ("H", self.H, 5, 200),
            ("H0", self.H0, 5, 200),
            ("Hprime", self.Hprime, 5, 200),
            ("E0_lm", self.E0_lm, 5, 200),
            ("degeneracy", self.degeneracy, 5, 100),
            ("E2_lm", self.E2_lm_str, 5, 300)
        ]
        for key, val, min_len, max_len in checks:
            if val is None:
                return (False, f"{key} is missing")
            if len(val) < min_len or len(val) > max_len:
                return (False, f"{key} length {len(val)} out of range [{min_len},{max_len}]")
        return (True, "All answer strings have plausible lengths")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test methods
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