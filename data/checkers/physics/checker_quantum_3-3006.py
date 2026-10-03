import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the angular momentum problem with a spinless particle
    wave function ψ = K (x + y + 2z) e^{-αr}.
    """
    def __init__(self, solution_instance):
        # Extract results safely, using None for missing keys
        results = getattr(solution_instance, 'results', {})
        self.L_mag = results.get('L_mag', None)
        self.expect_Lz = results.get('expect_Lz', None)
        self.prob_plus = results.get('prob_plus', None)
        self.dP = results.get('dP', None)
        # Optional angular coefficients (from solution_instance or results)
        self.coeffs = getattr(solution_instance, 'angular_coeffs', None)
        if self.coeffs is None:
            self.coeffs = results.get('angular_coeffs', None)

    # ========== Interface Tests ==========
    def test_interface_required_keys(self):
        """Check that all four required result keys exist."""
        missing = []
        if self.L_mag is None:
            missing.append('L_mag')
        if self.expect_Lz is None:
            missing.append('expect_Lz')
        if self.prob_plus is None:
            missing.append('prob_plus')
        if self.dP is None:
            missing.append('dP')
        if missing:
            return (False, f"Missing required keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Check that result values have plausible types (not None, not empty)."""
        checks = []
        for name, val in [('L_mag', self.L_mag), ('expect_Lz', self.expect_Lz),
                          ('prob_plus', self.prob_plus), ('dP', self.dP)]:
            if val is None:
                checks.append(f"{name} is None")
            elif isinstance(val, str) and len(val.strip()) == 0:
                checks.append(f"{name} is empty string")
        if checks:
            return (False, " | ".join(checks))
        return (True, "All values have acceptable types")

    # ========== Invariant (theorem-based) Tests ==========
    def test_invariant_l_mag_squared(self):
        """Check that L_mag^2 = 2 ħ² (only l=1 contributes)."""
        if self.L_mag is None:
            return (False, "SKIP: L_mag not available")
        try:
            L = sp.sympify(self.L_mag) if not isinstance(self.L_mag, sp.Basic) else self.L_mag
            hbar = sp.Symbol('hbar', positive=True)
            expected_sq = 2 * hbar**2
            if L.is_number:
                if not math.isclose(float(L), math.sqrt(2), rel_tol=1e-5, abs_tol=1e-5):
                    return (False, f"L_mag = {float(L):.5f}, expected sqrt(2) ≈ 1.41421")
                return (True, "L_mag numeric value matches sqrt(2)*ħ with ħ=1")
            diff = sp.simplify(L**2 - expected_sq)
            if diff == 0:
                return (True, "L_mag^2 = 2 ħ² symbolically satisfied")
            else:
                ratio = sp.simplify(L**2 / hbar**2)
                if ratio == 2:
                    return (True, "L_mag^2 / ħ² = 2 symbolically")
                return (False, f"L_mag^2 = {sp.simplify(L**2)}, expected 2ħ²")
        except Exception as e:
            return (False, f"Error checking L_mag: {e}")

    def test_invariant_expectation_Lz(self):
        """Check that ⟨L_z⟩ = 0 (symmetry of expansion coefficients)."""
        if self.expect_Lz is None:
            return (False, "SKIP: expect_Lz not available")
        try:
            val = sp.sympify(self.expect_Lz) if not isinstance(self.expect_Lz, sp.Basic) else self.expect_Lz
            if val.is_number:
                if math.isclose(float(val), 0.0, rel_tol=1e-5, abs_tol=1e-5):
                    return (True, "⟨L_z⟩ = 0")
                return (False, f"⟨L_z⟩ = {float(val)}, expected 0")
            if sp.simplify(val) == 0:
                return (True, "⟨L_z⟩ = 0 symbolically")
            return (False, f"⟨L_z⟩ simplifies to {sp.simplify(val)}, not 0")
        except Exception as e:
            return (False, f"Error checking expect_Lz: {e}")

    def test_invariant_dP_normalization(self):
        """Check that ∫ dP = 1 (proper angular normalization)."""
        if self.dP is None:
            return (False, "SKIP: dP not available")
        try:
            expr = sp.sympify(self.dP) if not isinstance(self.dP, sp.Basic) else self.dP
            theta, phi = sp.symbols('theta phi', real=True)
            dOmega_sym = None
            for atom in sp.preorder_traversal(expr):
                if isinstance(atom, sp.Symbol) and 'omega' in str(atom).lower():
                    dOmega_sym = atom
                    break
            if dOmega_sym is not None:
                f = sp.simplify(expr / dOmega_sym)
            else:
                f = expr
            integral = sp.integrate(f * sp.sin(theta), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
            if sp.simplify(integral - 1) == 0:
                return (True, "∫ dP = 1 exactly")
            if integral.is_number and math.isclose(float(integral), 1.0, rel_tol=1e-3, abs_tol=1e-3):
                return (True, "∫ dP ≈ 1 (numerical check)")
            return (False, f"∫ dP = {integral}, expected 1")
        except Exception as e:
            return (False, f"Error checking dP normalization: {e}")

    # ========== Modeling / Trace Tests ==========
    def test_modeling_dP_angular_dependence(self):
        """Check that dP has the form (1/(8π)) * [sinθ(cosφ+sinφ)+2cosθ]^2 dΩ."""
        if self.dP is None:
            return (False, "SKIP: dP not available")
        try:
            expr = sp.sympify(self.dP) if not isinstance(self.dP, sp.Basic) else self.dP
            theta, phi = sp.symbols('theta phi', real=True)
            expected_density = (1/(8*sp.pi)) * (sp.sin(theta)*(sp.cos(phi)+sp.sin(phi)) + 2*sp.cos(theta))**2
            dOmega_sym = None
            for atom in sp.preorder_traversal(expr):
                if isinstance(atom, sp.Symbol) and 'omega' in str(atom).lower():
                    dOmega_sym = atom
                    break
            if dOmega_sym is not None:
                factor = sp.simplify(expr / dOmega_sym)
            else:
                factor = expr
            diff = sp.simplify(factor - expected_density)
            if diff == 0:
                return (True, "dP matches expected angular form exactly")
            # Allow proportional if constant factor (e.g., without the 1/(8π))
            ratio = sp.simplify(factor / expected_density)
            if ratio == 1:
                return (True, "dP matches expected form")
            if ratio.is_constant() and sp.simplify(ratio).is_number:
                return (True, f"dP proportional to expected form (ratio = {ratio})")
            return (False, f"dP does not match expected angular dependence, diff = {diff}")
        except Exception as e:
            return (False, f"Error checking dP form: {e}")

    def test_modeling_dP_positivity(self):
        """
        Check that the angular probability density is non-negative for all θ,φ.
        The density (without dΩ) should be ≥ 0 everywhere.
        """
        if self.dP is None:
            return (False, "SKIP: dP not available")
        try:
            expr = sp.sympify(self.dP) if not isinstance(self.dP, sp.Basic) else self.dP
            theta, phi = sp.symbols('theta phi', real=True)
            # Extract the angular function f(θ,φ) (without dΩ)
            dOmega_sym = None
            for atom in sp.preorder_traversal(expr):
                if isinstance(atom, sp.Symbol) and 'omega' in str(atom).lower():
                    dOmega_sym = atom
                    break
            if dOmega_sym is not None:
                f = sp.simplify(expr / dOmega_sym)
            else:
                f = expr
            f_simplified = sp.simplify(f)
            # Test a few sampled points over the sphere
            test_points = [
                (sp.pi/2, 0),
                (sp.pi/2, sp.pi/2),
                (0, 0),
                (sp.pi, 0),
                (sp.pi/2, sp.pi/4),
                (sp.pi/4, sp.pi/6)
            ]
            negative_at = []
            for t, p in test_points:
                val = sp.N(f_simplified.subs({theta: t, phi: p}))
                if not val.is_number:
                    continue
                if float(val) < -1e-12:  # allow tiny negative due to rounding
                    negative_at.append(f"θ={float(t):.2f}, φ={float(p):.2f} => {float(val):.5f}")
            if negative_at:
                return (False, f"dP density negative at: {', '.join(negative_at)}")
            return (True, "dP angular density is non-negative at sampled points")
        except Exception as e:
            return (False, f"Error checking dP positivity: {e}")

    def test_modeling_symmetric_coefficients(self):
        """
        Check that the coefficients for Lz=+ħ and Lz=-ħ have equal magnitudes,
        leading to ⟨L_z⟩=0. Only if angular coefficients are provided.
        """
        if self.coeffs is None:
            return (False, "SKIP: angular_coeffs not provided")
        try:
            # Find keys for m=1 and m=-1
            keys = list(self.coeffs.keys())
            c1_key = None
            c_neg1_key = None
            for k in keys:
                if k in ['1', 'c1', 'C1', 'c_plus']:
                    c1_key = k
                elif k in ['-1', 'c_neg1', 'C_neg1', 'c_minus']:
                    c_neg1_key = k
            if c1_key is None or c_neg1_key is None:
                return (False, "SKIP: cannot identify m=±1 coefficients")
            c1_val = self.coeffs[c1_key]
            c_neg1_val = self.coeffs[c_neg1_key]
            # Get absolute squares
            if isinstance(c1_val, (sp.Basic, complex)):
                sq1 = abs(c1_val)**2
            else:
                sq1 = float(c1_val)**2
            if isinstance(c_neg1_val, (sp.Basic, complex)):
                sq_neg1 = abs(c_neg1_val)**2
            else:
                sq_neg1 = float(c_neg1_val)**2
            # They should be equal
            if math.isclose(float(sq1), float(sq_neg1), rel_tol=1e-5, abs_tol=1e-5):
                return (True, f"|c1|^2 = {float(sq1):.4f}, |c_{-1}|^2 = {float(sq_neg1):.4f} — symmetric")
            else:
                return (False, f"|c1|^2 = {float(sq1):.4f}, |c_{-1}|^2 = {float(sq_neg1):.4f} — not symmetric")
        except Exception as e:
            return (False, f"Error checking coefficient symmetry: {e}")

    # ========== Monotonicity Tests ==========
    def test_monotonicity_prob_plus_increase_with_c1(self):
        """
        If the solution provides angular coefficients, check that increasing |c1|^2
        (Lz=+ħ coefficient) increases the probability for Lz=+ħ.
        """
        if self.coeffs is None or self.prob_plus is None:
            return (False, "SKIP: angular_coeffs not provided")
        try:
            # Find the c1 coefficient
            c1_key = None
            for k in ['1', 'c1', 'C1', 'c_plus']:
                if k in self.coeffs:
                    c1_key = k
                    break
            if c1_key is None:
                return (False, "SKIP: no c1 coefficient found")
            c1_val = self.coeffs[c1_key]
            if isinstance(c1_val, (sp.Basic, complex)):
                c1_sq = abs(c1_val)**2
            else:
                c1_sq = float(c1_val)**2
            # Total sum of squares from coefficients
            sum_sq = 0
            for key, val in self.coeffs.items():
                if key == 'sum_sq':
                    sum_sq = float(val)
                    break
                if isinstance(val, (sp.Basic, complex)):
                    sum_sq += float(abs(val)**2)
                else:
                    sum_sq += float(val)**2
            if sum_sq == 0:
                return (False, "SKIP: total sum of squares zero")
            baseline = c1_sq / sum_sq
            # Perturb: increase c1_sq by 10%
            new_c1_sq = c1_sq * 1.1
            new_sum_sq = sum_sq - c1_sq + new_c1_sq
            new_prob = new_c1_sq / new_sum_sq
            if new_prob > baseline:
                return (True, f"Increasing |c1|^2 from {c1_sq:.4f} to {new_c1_sq:.4f} increases prob from {baseline:.4f} to {new_prob:.4f}")
            else:
                return (False, f"Monotonicity violated: increasing |c1|^2 did not increase prob (baseline {baseline:.4f}, new {new_prob:.4f})")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    # ========== Answer Sanity (weak) ==========
    def test_answer_sanity_prob_plus(self):
        """Check that prob_plus is between 0 and 1 and roughly 1/6."""
        if self.prob_plus is None:
            return (False, "SKIP: prob_plus not available")
        try:
            p = float(sp.sympify(self.prob_plus) if not isinstance(self.prob_plus, sp.Basic) else self.prob_plus)
            if not (0 <= p <= 1):
                return (False, f"prob_plus = {p:.4f} out of [0,1]")
            if abs(p - 1/6) > 0.1:
                return (False, f"prob_plus = {p:.4f} too far from 1/6 (0.1667)")
            return (True, f"prob_plus = {p:.4f} within plausible range")
        except Exception as e:
            return (False, f"Error checking prob_plus: {e}")

    # ========== Runner ==========
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