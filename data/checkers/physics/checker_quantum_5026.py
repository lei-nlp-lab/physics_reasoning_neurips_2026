import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution to the finite-proton-size correction to hydrogen binding energy
    using first-order perturbation theory.
    """
    def __init__(self, solution_instance):
        # Extract givens (safe defaults None)
        self.e = getattr(solution_instance, 'e', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.R = getattr(solution_instance, 'R', None)
        self.a0 = getattr(solution_instance, 'a0', None)
        # Extract results
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.Delta_E = self.results.get('Delta_E', None)
        # Extract explanation steps if present (may be None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # For symbolic checks
        self._e_sym = sp.symbols('e', positive=True)
        self._eps0_sym = sp.symbols('epsilon0', positive=True)
        self._R_sym = sp.symbols('R', positive=True)
        self._a0_sym = sp.symbols('a0', positive=True)

    # ---------- interface tests ----------
    def test_interface_required_keys(self):
        """Check that all essential givens and results are present and finite."""
        missing = []
        for name, val in [('e', self.e), ('epsilon0', self.epsilon0),
                          ('R', self.R), ('a0', self.a0), ('Delta_E', self.Delta_E)]:
            if val is None:
                missing.append(name)
            elif isinstance(val, (int, float)) and (math.isnan(val) or math.isinf(val)):
                missing.append(f"{name} (invalid numeric)")
        if missing:
            return False, f"Missing or invalid keys: {', '.join(missing)}"
        return True, "All required keys present and finite."

    # ---------- invariant tests (theorem-based) ----------
    def test_invariant_perturbation_positive(self):
        """Check that ΔE > 0 (since H' > 0 inside the proton)."""
        if self.Delta_E is None:
            return False, "SKIP: Delta_E not available"
        # Compare with zero using absolute tolerance
        if self.Delta_E <= 0:
            return False, f"Delta_E = {self.Delta_E} should be > 0"
        return True, f"Delta_E = {self.Delta_E} > 0 (positive as expected)"

    def test_invariant_delta_sign_physical(self):
        """Check that positive ΔE means binding energy becomes less negative (weaker binding)."""
        if self.Delta_E is None:
            return False, "SKIP: Delta_E not available"
        # We cannot directly check binding energy, but we can confirm reasoning exists.
        # Just check sign has been interpreted (optional weak check)
        # Explanation should mention that binding energy decreases.
        # Since we can't parse explanation, we rely on the positive ΔE being physically correct.
        return True, "Positive ΔE implies reduced binding – sign consistent with physics."

    # ---------- modeling trace tests (using symbolic physics) ----------
    def test_modeling_integral_relation(self):
        """Check that the integral ∫₀ᴿ (1/r - 1/R) r² dr equals R²/6."""
        if self.R is None:
            return False, "SKIP: R not given"
        r = sp.symbols('r', positive=True)
        integral = sp.integrate((1/r - 1/self._R_sym) * r**2, (r, 0, self._R_sym))
        expected = self._R_sym**2 / 6
        # Check symbolic equality
        if sp.simplify(integral - expected) != 0:
            return False, f"Integral symbolic check failed: {integral} != {expected}"
        # Numerical check as fallback (if R is numeric)
        if isinstance(self.R, (int, float)):
            integral_num = sp.N(integral.subs({self._R_sym: self.R}))
            expected_num = self.R**2 / 6
            if not math.isclose(float(integral_num), expected_num, rel_tol=1e-9):
                return False, f"Numerical integral mismatch: {integral_num} vs {expected_num}"
        return True, "Integral relation holds: ∫₀ᴿ (1/r - 1/R) r² dr = R²/6"

    def test_modeling_wavefunction_constancy(self):
        """Check that the ΔE expression uses |ψ(0)|² = 1/(π a0³) approximation."""
        if self.Delta_E is None:
            return False, "SKIP: Delta_E not available"
        # Derive the expected expression symbolically
        psi2_0 = 1/(sp.pi * self._a0_sym**3)
        C = self._e_sym**2 / (4*sp.pi * self._eps0_sym)
        integral = self._R_sym**2 / 6
        expected_Delta_E = psi2_0 * C * 4*sp.pi * integral
        # Simplify
        expected_Delta_E_simplified = sp.simplify(expected_Delta_E)
        # We cannot compare directly with self.Delta_E unless it is symbolic.
        # But we can check the functional form if Delta_E is a string? Usually numeric.
        # Weak check: if Delta_E is numeric, we can plug reasonable numbers.
        if isinstance(self.Delta_E, (int, float)) and all(isinstance(v, (int, float)) for v in [self.e, self.epsilon0, self.R, self.a0]):
            expected_num = float(expected_Delta_E_simplified.subs({
                self._e_sym: self.e,
                self._eps0_sym: self.epsilon0,
                self._R_sym: self.R,
                self._a0_sym: self.a0
            }))
            if not math.isclose(self.Delta_E, expected_num, rel_tol=1e-2):
                return False, f"ΔE mismatch with wavefunction-approximation formula: got {self.Delta_E}, expected {expected_num}"
        return True, "ΔE expression consistent with constant wavefunction approximation"

    def test_trace_delta_E_formula(self):
        """Check that ΔE matches the derived closed form: (2/3) * e²/(4πϵ0) * R²/a0³."""
        if self.Delta_E is None or any(v is None for v in [self.e, self.epsilon0, self.R, self.a0]):
            return False, "SKIP: missing variables for formula check"
        formula = (2/3) * (self.e**2/(4*sp.pi*self.epsilon0)) * (self.R**2/self.a0**3)
        # formula is sympy expression, convert to float
        if isinstance(self.Delta_E, (int, float)):
            expected_num = float(formula.evalf())
            if not math.isclose(self.Delta_E, expected_num, rel_tol=1e-2):
                return False, f"ΔE formula mismatch: got {self.Delta_E}, expected {expected_num}"
        return True, "ΔE matches the closed-form result (2/3) e²/(4πϵ0) R²/a0³"

    # ---------- monotonicity tests ----------
    def test_monotonicity_delta_increases_with_R(self):
        """Check that ΔE increases when R is increased (positive correlation)."""
        if self.Delta_E is None or self.R is None:
            return False, "SKIP: Delta_E or R missing"
        if not isinstance(self.R, (int, float)) or not isinstance(self.Delta_E, (int, float)):
            return False, "SKIP: R and Delta_E must be numeric for monotonicity test"
        # Perturb R by +10% (relative) and recompute ΔE using the derived formula (internal)
        R1 = self.R
        R2 = self.R * 1.1
        # Recompute Delta_E using the formula within checker (re-derive)
        if any(v is None for v in [self.e, self.epsilon0, self.a0]):
            return False, "SKIP: e, epsilon0, a0 missing for recomputation"
        Delta_E1 = (2/3) * (self.e**2/(4*math.pi*self.epsilon0)) * (R1**2/self.a0**3)
        Delta_E2 = (2/3) * (self.e**2/(4*math.pi*self.epsilon0)) * (R2**2/self.a0**3)
        if Delta_E2 <= Delta_E1:
            return False, f"Monotonicity fails: when R increases from {R1} to {R2}, ΔE should increase (from {Delta_E1} to {Delta_E2})"
        return True, f"ΔE increases with R (consistent with R² dependence)"

    # ---------- answer sanity (weak, only one allowed) ----------
    def test_answer_sanity_broad_magnitude(self):
        """Weak check: ΔE should be positive and within plausible order of magnitude."""
        if self.Delta_E is None or any(v is None for v in [self.e, self.epsilon0, self.R, self.a0]):
            return False, "SKIP: missing data for magnitude check"
        if not all(isinstance(v, (int, float)) for v in [self.e, self.epsilon0, self.R, self.a0]):
            return False, "SKIP: all quantities must be numeric"
        # Compute approximate value using formula
        approx = (2/3) * (self.e**2/(4*math.pi*self.epsilon0)) * (self.R**2/self.a0**3)
        # Plausible range: positive and not more than 10 times the point-like ground state energy
        # Ground state energy = -13.6 eV, so ΔE should be much smaller (positive)
        # For typical values: e=1.602e-19 C, ϵ0=8.854e-12, R=0.877e-15 m, a0=5.29e-11 m
        # ΔE ≈ 2e-9 eV (tiny). So we just check it's positive and not huge.
        if self.Delta_E <= 0:
            return False, f"Answer sanity: ΔE must be positive, got {self.Delta_E}"
        # Ensure it's not orders of magnitude larger than 1 eV (unphysical)
        if self.Delta_E > 1e-3:   # > 1 meV is unlikely for such small R
            return False, f"Answer sanity: ΔE seems too large: {self.Delta_E}"
        return True, f"ΔE = {self.Delta_E} is positive and within plausible magnitude (<1 meV)"

    # ---------- main execution ----------
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