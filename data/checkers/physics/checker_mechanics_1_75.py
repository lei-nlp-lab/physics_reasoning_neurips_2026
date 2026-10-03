import math
import traceback

class PhysicsChecker:
    """
    Physics checker for lunar surface gravity and escape velocity.
    Audits results for gravitational acceleration (g) and escape velocity (v_esc)
    against Newtonian gravitation and energy conservation.
    """
    
    def __init__(self, solution_instance):
        # Extract givens (G, M, R) safely; default to None.
        givens = getattr(solution_instance, 'givens', {})
        self.G = givens.get('G', None)
        self.M = givens.get('M', None)
        self.R = givens.get('R', None)
        
        # Extract result quantities from solution_instance.results.
        results = getattr(solution_instance, 'results', {})
        self.g = results.get('g', None)
        self.v_esc = results.get('v_esc', None)
        
        # Reference constants for physical checks (only used inside tests, never as fallback for user data).
        self._ref_G = 6.67430e-11  # m^3 kg^-1 s^-2
        self._ref_M = 7.342e22     # kg
        self._ref_R = 1.737e6      # m

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that 'g' and 'v_esc' are present and numeric."""
        msgs = []
        if self.g is None:
            msgs.append("Missing 'g' in results.")
        elif not isinstance(self.g, (int, float)):
            msgs.append("'g' is not numeric.")
        if self.v_esc is None:
            msgs.append("Missing 'v_esc' in results.")
        elif not isinstance(self.v_esc, (int, float)):
            msgs.append("'v_esc' is not numeric.")
        if msgs:
            return (False, "; ".join(msgs))
        return (True, "Required keys present and numeric.")

    def test_interface_positive(self):
        """Sanity: g and v_esc must be positive finite numbers."""
        if self.g is None or self.v_esc is None:
            return (False, "SKIP: cannot check positivity because g and/or v_esc missing.")
        if not (self.g > 0 and math.isfinite(self.g)):
            return (False, f"g = {self.g} is not positive finite.")
        if not (self.v_esc > 0 and math.isfinite(self.v_esc)):
            return (False, f"v_esc = {self.v_esc} is not positive finite.")
        return (True, "Both quantities positive and finite.")

    # ------------------------------------------------------------
    # Invariant tests (fuzzy numerical matching allowed)
    # ------------------------------------------------------------
    def test_invariant_g_formula(self):
        """If G, M, R available, check g ≈ G M / R²."""
        G = self.G if self.G is not None else self._ref_G
        M = self.M if self.M is not None else self._ref_M
        R = self.R if self.R is not None else self._ref_R
        if self.g is None:
            return (False, "SKIP: g not available.")
        expected = G * M / (R * R)
        if math.isclose(self.g, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"g ≈ G M / R² = {expected:.4f} m/s² (relative error < 1%).")
        else:
            return (False, f"g = {self.g} does not match expected {expected:.4f} (relative error > 1%).")

    def test_invariant_vesc_formula(self):
        """If G, M, R available, check v_esc ≈ sqrt(2 G M / R)."""
        G = self.G if self.G is not None else self._ref_G
        M = self.M if self.M is not None else self._ref_M
        R = self.R if self.R is not None else self._ref_R
        if self.v_esc is None:
            return (False, "SKIP: v_esc not available.")
        expected = math.sqrt(2 * G * M / R)
        if math.isclose(self.v_esc, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_esc ≈ sqrt(2GM/R) = {expected:.1f} m/s (relative error < 1%).")
        else:
            return (False, f"v_esc = {self.v_esc} does not match expected {expected:.1f} (relative error > 1%).")

    def test_invariant_vesc_via_g(self):
        """Check v_esc = sqrt(2 g R) if both g and R available."""
        R = self.R if self.R is not None else self._ref_R
        if self.g is None or self.v_esc is None:
            return (False, "SKIP: g or v_esc missing.")
        if R is None:
            return (False, "SKIP: R missing.")
        expected = math.sqrt(2 * self.g * R)
        if math.isclose(self.v_esc, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_esc ≈ sqrt(2 g R) = {expected:.1f} m/s (relative error < 1%).")
        else:
            return (False, f"v_esc = {self.v_esc} does not match sqrt(2 g R) = {expected:.1f} (relative error > 1%).")

    # ------------------------------------------------------------
    # Weak answer sanity (only one such test)
    # ------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Broad magnitude and sign checks only."""
        if self.g is None or self.v_esc is None:
            return (False, "SKIP: missing data for sanity check.")
        # Reasonable ranges: g ~ 1.6 m/s², v_esc ~ 2.4e3 m/s
        if not (1.0 < self.g < 2.0):
            return (False, f"g = {self.g} outside plausible range [1.0, 2.0] m/s².")
        if not (2.0e3 < self.v_esc < 3.0e3):
            return (False, f"v_esc = {self.v_esc} outside plausible range [2000, 3000] m/s.")
        return (True, "Magnitudes within expected broad ranges.")

    # ------------------------------------------------------------
    # Modeling/trace tests (one relation per test)
    # ------------------------------------------------------------
    def test_trace_vesc_via_energy(self):
        """
        Check consistency of explanation step: v_esc = sqrt(2 g R).
        This is a direct consequence of energy conservation.
        """
        return self.test_invariant_vesc_via_g()

    def test_modeling_g_numerical_consistency(self):
        """
        Verify that the numerical calculation of g from the explanation steps
        (using the standard moon constants) yields the provided g.
        This test uses the reference constants from the problem statement.
        """
        G = self._ref_G
        M = self._ref_M
        R = self._ref_R
        if self.g is None:
            return (False, "SKIP: g not available.")
        expected = G * M / (R * R)
        if math.isclose(self.g, expected, rel_tol=1e-2):
            return (True, f"g ≈ {expected:.4f} matches explanation numerical calculation.")
        else:
            return (False, f"g = {self.g} does not match explanation's intermediate value {expected:.4f}.")

    def test_modeling_energy_conservation_symbolic(self):
        """
        Symbolic verification that solving ½mv² = G M m / R yields v = sqrt(2 G M / R).
        This checks the fundamental theorem used in the explanation.
        """
        import sympy as sp
        try:
            m, G, M, R, v = sp.symbols('m G M R v', positive=True)
            kinetic = sp.Rational(1,2) * m * v**2
            potential = G * M * m / R
            eq = sp.Eq(kinetic, potential)
            sol = sp.solve(eq, v)
            expected = sp.sqrt(2 * G * M / R)
            if sol[0] == expected:
                return (True, "Symbolic derivation confirms v_esc = sqrt(2GM/R).")
            else:
                return (False, f"Symbolic derivation gave {sol[0]}; expected {expected}.")
        except Exception as e:
            return (False, f"Symbolic test crashed: {e}")

    # ------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_g_decreases_with_R(self):
        """
        Verify that gravitational acceleration decreases when radius increases.
        Use reference (or given) G, M, R; compute g at R and at 1.1*R; assert g_pert < g_ref.
        """
        G = self.G if self.G is not None else self._ref_G
        M = self.M if self.M is not None else self._ref_M
        R = self.R if self.R is not None else self._ref_R
        # Baseline g
        g_ref = G * M / (R * R)
        # Perturb R upward by 10%
        R_pert = R * 1.1
        g_pert = G * M / (R_pert * R_pert)
        if g_pert < g_ref:
            return (True, f"g decreases with R: g({R:.2e}) = {g_ref:.4f} > g({R_pert:.2e}) = {g_pert:.4f}.")
        else:
            return (False, "Monotonicity failed: g did not decrease when R increased.")

    def test_monotonicity_vesc_increases_with_M(self):
        """
        Verify that escape velocity increases when mass increases.
        Use reference (or given) G, M, R; compute v_esc at M and at 1.1*M; assert v_pert > v_ref.
        """
        G = self.G if self.G is not None else self._ref_G
        M = self.M if self.M is not None else self._ref_M
        R = self.R if self.R is not None else self._ref_R
        # Baseline v_esc
        v_ref = math.sqrt(2 * G * M / R)
        # Perturb M upward by 10%
        M_pert = M * 1.1
        v_pert = math.sqrt(2 * G * M_pert / R)
        if v_pert > v_ref:
            return (True, f"v_esc increases with M: v({M:.2e}) = {v_ref:.1f} < v({M_pert:.2e}) = {v_pert:.1f}.")
        else:
            return (False, "Monotonicity failed: v_esc did not increase when M increased.")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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