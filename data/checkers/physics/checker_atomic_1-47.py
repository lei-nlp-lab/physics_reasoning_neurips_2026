import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results dictionary
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = getattr(solution_instance, 'data', {}) if getattr(solution_instance, 'data', None) else {}
        
        # Extract individual probabilities with None defaults
        self.P_1s = self.results.get('P_1s', None) if isinstance(self.results, dict) else None
        self.P_2s = self.results.get('P_2s', None) if isinstance(self.results, dict) else None
        self.P_2p = self.results.get('P_2p', None) if isinstance(self.results, dict) else None

    # ---------- Interface Tests ----------
    def test_interface_required_keys(self):
        """Check that all expected result keys are present."""
        missing = []
        if self.P_1s is None:
            missing.append('P_1s')
        if self.P_2s is None:
            missing.append('P_2s')
        if self.P_2p is None:
            missing.append('P_2p')
        if missing:
            return (False, f"Missing required keys: {', '.join(missing)}")
        return (True, "All required keys present")

    # ---------- Invariant Tests (theorem-based) ----------
    def test_invariant_nonnegativity(self):
        """Check that all probabilities are non-negative."""
        for name, val in [('P_1s', self.P_1s), ('P_2s', self.P_2s), ('P_2p', self.P_2p)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if val < -1e-9:
                return (False, f"{name} = {val} is negative")
        return (True, "All probabilities non-negative")

    def test_invariant_bounds(self):
        """Check that probabilities do not exceed 1."""
        for name, val in [('P_1s', self.P_1s), ('P_2s', self.P_2s), ('P_2p', self.P_2p)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if val > 1 + 1e-9:
                return (False, f"{name} = {val} exceeds 1")
        return (True, "All probabilities ≤ 1")

    def test_invariant_2p_zero(self):
        """Check that P_2p is zero due to angular orthogonality."""
        if self.P_2p is None:
            return (False, "SKIP: P_2p missing")
        if math.isclose(self.P_2p, 0, abs_tol=1e-9):
            return (True, f"P_2p = {self.P_2p} is consistent with zero")
        else:
            return (False, f"P_2p = {self.P_2p} should be zero")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_angular_selection(self):
        """Only s-states have nonzero overlap with initial spherically symmetric state."""
        if self.P_2p is None or self.P_1s is None or self.P_2s is None:
            return (False, "SKIP: one or more probabilities missing")
        if not math.isclose(self.P_2p, 0, abs_tol=1e-9):
            return (False, f"P_2p = {self.P_2p} should be zero due to orthogonality")
        if self.P_1s <= 0:
            return (False, f"P_1s = {self.P_1s} should be positive")
        if self.P_2s <= 0:
            return (False, f"P_2s = {self.P_2s} should be positive")
        return (True, "Angular selection rule satisfied: s-state overlaps positive, p-state zero")

    def test_modeling_P_1s_correct(self):
        """Verify P_1s by recomputing the overlap integral symbolically (numerical check)."""
        if self.P_1s is None:
            return (False, "SKIP: P_1s missing")
        # Define symbols
        r, a0 = sp.symbols('r a0', positive=True)
        # Radial wavefunctions (initial Z=1, final Z=2)
        R_i = 2 * (1 / a0) ** (sp.Rational(3, 2)) * sp.exp(-r / a0)
        R_1s = 2 * (2 / a0) ** (sp.Rational(3, 2)) * sp.exp(-2 * r / a0)
        integrand = R_1s * R_i * r ** 2
        integral = sp.integrate(integrand, (r, 0, sp.oo))
        probability = sp.simplify(sp.Abs(integral) ** 2)
        expected = float(probability.subs(a0, 1))  # a0 cancels, any value works
        if math.isclose(self.P_1s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_1s = {self.P_1s} matches expected {expected}")
        else:
            return (False, f"P_1s = {self.P_1s} does not match expected {expected}")

    def test_modeling_P_2s_correct(self):
        """Verify P_2s by recomputing the overlap integral symbolically."""
        if self.P_2s is None:
            return (False, "SKIP: P_2s missing")
        r, a0 = sp.symbols('r a0', positive=True)
        R_i = 2 * (1 / a0) ** (sp.Rational(3, 2)) * sp.exp(-r / a0)
        # He+ 2s radial: (1/sqrt(8)) * (2/a0)^{3/2} * (2 - 2r/a0) * exp(-r/a0)
        R_2s = (1 / sp.sqrt(8)) * (2 / a0) ** (sp.Rational(3, 2)) * (2 - 2 * r / a0) * sp.exp(-r / a0)
        integrand = R_2s * R_i * r ** 2
        integral = sp.integrate(integrand, (r, 0, sp.oo))
        probability = sp.simplify(sp.Abs(integral) ** 2)
        expected = float(probability.subs(a0, 1))
        if math.isclose(self.P_2s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_2s = {self.P_2s} matches expected {expected}")
        else:
            return (False, f"P_2s = {self.P_2s} does not match expected {expected}")

    # ---------- Monotonicity Tests (explicit perturbation) ----------
    def test_monotonicity_P_1s_decreases_with_Zf(self):
        """Check that P_1s decreases when He+ nuclear charge Z is increased (sudden approximation)."""
        # Baseline: Zf = 2.0
        r, a0, Z = sp.symbols('r a0 Z', positive=True)
        # Initial wavefunction with Z=1
        R_i = 2 * (1 / a0) ** (sp.Rational(3, 2)) * sp.exp(-r / a0)
        # Final 1s with charge Z
        R_f = 2 * (Z / a0) ** (sp.Rational(3, 2)) * sp.exp(-Z * r / a0)
        integrand = R_f * R_i * r ** 2
        integral = sp.integrate(integrand, (r, 0, sp.oo)).simplify()
        prob_expr = sp.Abs(integral) ** 2

        # Evaluate at Z=2.0 and Z=2.1
        prob_base = float(prob_expr.subs({Z: 2.0, a0: 1}))
        prob_pert = float(prob_expr.subs({Z: 2.1, a0: 1}))

        if prob_pert < prob_base:
            return (True, f"P_1s decreases from {prob_base:.6f} (Z=2) to {prob_pert:.6f} (Z=2.1)")
        else:
            return (False, f"P_1s did not decrease: baseline {prob_base:.6f}, perturbed {prob_pert:.6f}")

    def test_monotonicity_P_2s_decreases_with_Zf(self):
        """Check that P_2s decreases when He+ nuclear charge Z is increased."""
        r, a0, Z = sp.symbols('r a0 Z', positive=True)
        # Initial wavefunction with Z=1
        R_i = 2 * (1 / a0) ** (sp.Rational(3, 2)) * sp.exp(-r / a0)
        # Final 2s radial: (1/sqrt(8)) * (Z/a0)^{3/2} * (2 - Z*r/a0) * exp(-Z*r/(2*a0))
        R_2s = (1 / sp.sqrt(8)) * (Z / a0) ** (sp.Rational(3, 2)) * (2 - Z * r / a0) * sp.exp(-Z * r / (2 * a0))
        integrand = R_2s * R_i * r ** 2
        integral = sp.integrate(integrand, (r, 0, sp.oo)).simplify()
        prob_expr = sp.Abs(integral) ** 2

        prob_base = float(prob_expr.subs({Z: 2.0, a0: 1}))
        prob_pert = float(prob_expr.subs({Z: 2.1, a0: 1}))

        if prob_pert < prob_base:
            return (True, f"P_2s decreases from {prob_base:.6f} (Z=2) to {prob_pert:.6f} (Z=2.1)")
        else:
            return (False, f"P_2s did not decrease: baseline {prob_base:.6f}, perturbed {prob_pert:.6f}")

    # ---------- Weak Answer Sanity (no math.isclose, no re-derivation) ----------
    def test_answer_sanity_magnitude(self):
        """Weak ordering check: P_1s > P_2s > P_2p."""
        if self.P_1s is None or self.P_2s is None or self.P_2p is None:
            return (False, "SKIP: one or more probabilities missing")
        if not (self.P_1s > self.P_2s):
            return (False, f"Expected P_1s > P_2s, got {self.P_1s} and {self.P_2s}")
        if not (self.P_2s > self.P_2p):
            return (False, f"Expected P_2s > P_2p, got {self.P_2s} and {self.P_2p}")
        return (True, "Ordering P_1s > P_2s > P_2p satisfied")

    # ---------- Main Runner ----------
    def run_all_tests(self):
        """Discover and run all test_* methods, collect logs."""
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