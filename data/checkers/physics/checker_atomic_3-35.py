import sympy as sp
import math
from sympy.physics.wigner import clebsch_gordan

class PhysicsChecker:
    """
    Audits the isospin-based ΔI = 1/2 analysis for hyperon nonleptonic decays.
    Checks that the ratios x, y, z (as defined in the problem) are correctly
    derived from Clebsch–Gordan coefficients and the ΔI = 1/2 rule.
    """

    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', {})
        # Never assign numeric defaults; keep None for missing values
        self.x = results.get('x', None)
        self.y = results.get('y', None)
        self.z = results.get('z', None)
        self.results = results

    # --------------------------------------------------------------
    # Interface checks
    # --------------------------------------------------------------
    def test_interface_keys_present(self):
        """Check that all three required keys exist in results."""
        missing = []
        for key in ['x', 'y', 'z']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present."

    def test_interface_numeric_types(self):
        """Check that x, y, z are numeric (int or float)."""
        for name, val in [('x', self.x), ('y', self.y), ('z', self.z)]:
            if val is None:
                return False, f"SKIP: {name} is None"
            if not isinstance(val, (int, float)):
                return False, f"{name} is not numeric (got {type(val).__name__})"
        return True, "All values are numeric."

    # --------------------------------------------------------------
    # Invariant tests – re‑derive ratios from Clebsch–Gordan coefficients
    # --------------------------------------------------------------
    def test_invariant_lambda_ratio(self):
        """Check that x = A(Λ→pπ⁻)/A(Λ→nπ⁰) equals -√2 from CG coefficients."""
        if self.x is None:
            return False, "SKIP: x not available"
        try:
            # Λ → pπ⁻ : CG(½, 1, ½; ½, -1, -½)
            cg1 = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(1,2),
                                  sp.Rational(1,2), -1, -sp.Rational(1,2))
            # Λ → nπ⁰ : CG(½, 1, ½; -½, 0, -½)
            cg2 = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(1,2),
                                  -sp.Rational(1,2), 0, -sp.Rational(1,2))
            expected_x = float(sp.N(cg1 / cg2))   # should be -√2 ≈ -1.4142
        except Exception as e:
            return False, f"CG computation failed: {e}"
        if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"x = {self.x}, expected ≈ {expected_x:.5f}"
        return True, f"x matches CG prediction ({expected_x:.5f})."

    def test_invariant_sigma_y_ratio(self):
        """
        Check that y = (A(Σ⁺→π⁺n) - A(Σ⁻→π⁻n)) / A(Σ⁺→π⁰p) equals -√2.
        Uses the symbolic expressions for the three amplitudes in terms of
        reduced matrix elements M_{1/2} and M_{3/2}.
        """
        if self.y is None:
            return False, "SKIP: y not available"
        try:
            M12, M32 = sp.symbols('M12 M32')
            # CG coefficients for Σ initial states with spurion
            # Σ⁺(I=1, I3=1) + spurion(½, -½) → I=3/2 component
            c_3half = clebsch_gordan(1, sp.Rational(1,2), sp.Rational(3,2),
                                      1, -sp.Rational(1,2), sp.Rational(1,2))
            # Σ⁺ + spurion → I=1/2 component
            c_half = clebsch_gordan(1, sp.Rational(1,2), sp.Rational(1,2),
                                     1, -sp.Rational(1,2), sp.Rational(1,2))

            # Final |π⁰ p> = |1,0>|½,½> → I=1/2 component CG
            final_cg_pi0_p = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(1,2),
                                              sp.Rational(1,2), 0, sp.Rational(1,2))

            # Final |π⁺ n> = |1,1>|½,-½> → I=3/2 component CG
            final_cg_pi_plus_n_3half = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(3,2),
                                                        -sp.Rational(1,2), 1, sp.Rational(1,2))
            # Final |π⁺ n> → I=1/2 component CG
            final_cg_pi_plus_n_half = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(1,2),
                                                       -sp.Rational(1,2), 1, sp.Rational(1,2))

            # A(Σ⁺→π⁰p) = c_half * M12 * final_cg_pi0_p   (only I=1/2 contributes)
            A_pi0p = c_half * M12 * final_cg_pi0_p

            # A(Σ⁺→π⁺n) = c_3half * M32 * final_cg_pi_plus_n_3half + c_half * M12 * final_cg_pi_plus_n_half
            A_pi_plus_n = c_3half * M32 * final_cg_pi_plus_n_3half + c_half * M12 * final_cg_pi_plus_n_half

            # A(Σ⁻→π⁻n) : initial |Σ⁻,a> = |3/2, -3/2>  (coefficient 1)
            # Final |π⁻ n> = |1,-1>|½,-½> → |3/2, -3/2> (CG = 1)
            # So A = M32 * 1
            A_pi_minus_n = M32 * 1

            y_expr = sp.simplify((A_pi_plus_n - A_pi_minus_n) / A_pi0p)
            expected_y = float(sp.N(y_expr))   # should be -√2 independent of M12, M32
        except Exception as e:
            return False, f"Symbolic derivation of y failed: {e}"
        if not math.isclose(self.y, expected_y, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"y = {self.y}, expected ≈ {expected_y:.5f}"
        return True, f"y matches isospin prediction ({expected_y:.5f})."

    def test_invariant_xi_z_ratio(self):
        """Check that z = A(Ξ⁰→Λπ⁰)/A(Ξ⁻→Λπ⁻) equals 1/√2."""
        if self.z is None:
            return False, "SKIP: z not available"
        try:
            # The initial |Ξ⁰, a> = |½,½>|½,-½> contains |1,0> component with CG = 1/√2
            # (positive sign in standard Condon-Shortley phase)
            # The amplitude for Ξ⁰→Λπ⁰ is proportional to that CG = 1/√2.
            # A(Ξ⁻→Λπ⁻) has CG = 1.
            # So z = |CG| = 1/√2 ≈ 0.7071.
            # Use sympy CG to get exact value
            cg_xi0 = clebsch_gordan(sp.Rational(1,2), sp.Rational(1,2), 1,
                                      sp.Rational(1,2), -sp.Rational(1,2), 0)
            expected_z = float(sp.N(cg_xi0))   # = 1/√2 ≈ 0.7071
        except Exception as e:
            return False, f"CG for Ξ decays failed: {e}"
        if not math.isclose(self.z, expected_z, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"z = {self.z}, expected ≈ {expected_z:.5f}"
        return True, f"z matches CG prediction ({expected_z:.5f})."

    # --------------------------------------------------------------
    # Modeling / trace tests – specific relations from explanation
    # --------------------------------------------------------------
    def test_modeling_lambda_amplitudes_sign(self):
        """Check that the two Λ amplitudes have opposite signs (x negative)."""
        if self.x is None:
            return False, "SKIP: x not available"
        if self.x >= 0:
            return False, f"x = {self.x} is not negative as expected."
        return True, "x is negative, consistent with CG sign."

    def test_modeling_sigma_y_constant(self):
        """Check that y is a constant independent of reduced matrix elements."""
        # Already verified in invariant test; just a sanity.
        if self.y is None:
            return False, "SKIP: y not available"
        expected = -math.sqrt(2)
        if not math.isclose(self.y, expected, rel_tol=1e-2):
            return False, f"y = {self.y} not close to -√2."
        return True, "y is near -√2."

    def test_trace_xi_initial_state(self):
        """Check that z is positive (coefficient magnitude)."""
        if self.z is None:
            return False, "SKIP: z not available"
        if self.z <= 0:
            return False, f"z = {self.z} should be positive."
        return True, "z is positive."

    # --------------------------------------------------------------
    # Monotonicity test – perturb ratio M32/M12, y remains constant
    # --------------------------------------------------------------
    def test_monotonicity_y_independent_of_reduced_matrix_ratio(self):
        """
        Perturb the ratio r = M32 / M12 and verify that y remains constant.
        The isospin prediction for y is purely from CG coefficients and does
        not depend on the actual values of the reduced matrix elements.
        """
        if self.y is None:
            return False, "SKIP: y not available"
        try:
            M12, M32 = sp.symbols('M12 M32')
            # Reuse the same expression from invariant test
            c_3half = clebsch_gordan(1, sp.Rational(1,2), sp.Rational(3,2),
                                      1, -sp.Rational(1,2), sp.Rational(1,2))
            c_half = clebsch_gordan(1, sp.Rational(1,2), sp.Rational(1,2),
                                     1, -sp.Rational(1,2), sp.Rational(1,2))
            final_cg_pi0_p = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(1,2),
                                              sp.Rational(1,2), 0, sp.Rational(1,2))
            final_cg_pi_plus_n_3half = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(3,2),
                                                        -sp.Rational(1,2), 1, sp.Rational(1,2))
            final_cg_pi_plus_n_half = clebsch_gordan(sp.Rational(1,2), 1, sp.Rational(1,2),
                                                       -sp.Rational(1,2), 1, sp.Rational(1,2))
            A_pi0p = c_half * M12 * final_cg_pi0_p
            A_pi_plus_n = c_3half * M32 * final_cg_pi_plus_n_3half + c_half * M12 * final_cg_pi_plus_n_half
            A_pi_minus_n = M32 * 1
            y_expr = sp.simplify((A_pi_plus_n - A_pi_minus_n) / A_pi0p)

            # Baseline: r = 1
            y_baseline = float(sp.N(y_expr.subs({M32:1, M12:1})))
            # Perturb: r = 2
            y_perturbed = float(sp.N(y_expr.subs({M32:2, M12:1})))
            # Also try r = 0.5
            y_perturbed2 = float(sp.N(y_expr.subs({M32:0.5, M12:1})))

            if not (math.isclose(y_baseline, y_perturbed, rel_tol=1e-10) and
                    math.isclose(y_baseline, y_perturbed2, rel_tol=1e-10)):
                return False, "y changed when M32/M12 was perturbed (should be invariant)."

            # Also ensure the provided y matches the predicted constant
            if not math.isclose(self.y, y_baseline, rel_tol=1e-2):
                return False, f"Provided y = {self.y} does not match predicted constant {y_baseline:.5f}."
        except Exception as e:
            return False, f"Monotonicity test crashed: {e}"
        return True, "y is invariant under perturbation of M32/M12 ratio (monotonicity satisfied)."

    # --------------------------------------------------------------
    # Answer sanity (weak check – only broad magnitude and sign)
    # --------------------------------------------------------------
    def test_answer_sanity_magnitudes(self):
        """Weak sanity: all ratios should have finite magnitude less than 10."""
        for name, val in [('x', self.x), ('y', self.y), ('z', self.z)]:
            if val is None:
                return False, f"SKIP: {name} is None"
            if not math.isfinite(val):
                return False, f"{name} is not finite."
            if abs(val) > 10:
                return False, f"{name} magnitude {abs(val)} > 10, unreasonable."
        return True, "All ratios are finite and have plausible magnitude."

    # --------------------------------------------------------------
    # run_all_tests – discover and execute every test_ method
    # --------------------------------------------------------------
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