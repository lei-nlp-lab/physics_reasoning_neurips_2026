import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Auditor for the heat capacity relation problem:
        C_p - C_v = 3α M p / ρ
    Checks thermodynamic identities, volume expansion, and monotonic trends.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (could be attributes or in results dict)
        self.M = getattr(solution_instance, 'M', None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.alpha = getattr(solution_instance, 'alpha', None)
        self.p = getattr(solution_instance, 'p', None)

        # If not found, try from results dict
        if self.M is None:
            self.M = solution_instance.results.get('M', None)
        if self.rho is None:
            self.rho = solution_instance.results.get('rho', None)
        if self.alpha is None:
            self.alpha = solution_instance.results.get('alpha', None)
        if self.p is None:
            self.p = solution_instance.results.get('p', None)

        # Extract emitted results
        results = solution_instance.results
        self.V = results.get('V', None)
        self.dvdT = results.get('dvdT', None)
        self.Cp_minus_Cv = results.get('Cp_minus_Cv', None)

        # Optional: store explanation steps for trace checks (if available)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        """Check that all three required result keys exist and are numeric."""
        if self.V is None:
            return (False, "SKIP: V not found in results")
        if self.dvdT is None:
            return (False, "SKIP: dvdT not found in results")
        if self.Cp_minus_Cv is None:
            return (False, "SKIP: Cp_minus_Cv not found in results")
        # Check numeric types
        for name, val in [('V', self.V), ('dvdT', self.dvdT), ('Cp_minus_Cv', self.Cp_minus_Cv)]:
            if not isinstance(val, (int, float)):
                return (False, f"Key {name} is not numeric (got {type(val).__name__})")
        return (True, "All required keys present and numeric")

    def test_interface_givens_exist(self):
        """Check that all given parameters are available (optional but helpful)."""
        missing = []
        for name in ['M', 'rho', 'alpha', 'p']:
            if getattr(self, name, None) is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: Given parameters missing: {', '.join(missing)}")
        return (True, "All givens available")

    # ---------- Invariant (physical reality) tests ----------
    def test_invariant_thermodynamic_identity(self):
        """C_p - C_v = p * (dV/dT)_p  (under (∂U/∂V)_T ≈ 0)"""
        if self.Cp_minus_Cv is None or self.p is None or self.dvdT is None:
            return (False, "SKIP: required data missing")
        expected = self.p * self.dvdT
        if math.isclose(self.Cp_minus_Cv, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Cp_minus_Cv ≈ p * dvdT ({self.Cp_minus_Cv} ≈ {expected})")
        else:
            return (False, f"Identity mismatch: Cp_minus_Cv={self.Cp_minus_Cv}, p*dvdT={expected}")

    def test_invariant_volume_expansion(self):
        """dV/dT = 3α V"""
        if self.dvdT is None or self.alpha is None or self.V is None:
            return (False, "SKIP: required data missing")
        expected = 3 * self.alpha * self.V
        if math.isclose(self.dvdT, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dvdT ≈ 3αV ({self.dvdT} ≈ {expected})")
        else:
            return (False, f"Expansion mismatch: dvdT={self.dvdT}, 3αV={expected}")

    def test_invariant_volume_from_mass(self):
        """V = M/ρ"""
        if self.V is None or self.M is None or self.rho is None:
            return (False, "SKIP: required data missing")
        expected = self.M / self.rho
        if math.isclose(self.V, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V ≈ M/ρ ({self.V} ≈ {expected})")
        else:
            return (False, f"Volume mismatch: V={self.V}, M/ρ={expected}")

    def test_invariant_full_expression(self):
        """C_p - C_v = 3α M p / ρ  (combined identity)"""
        if None in (self.Cp_minus_Cv, self.alpha, self.M, self.p, self.rho):
            return (False, "SKIP: required data missing")
        expected = 3 * self.alpha * self.M * self.p / self.rho
        if math.isclose(self.Cp_minus_Cv, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Full formula holds: {self.Cp_minus_Cv} ≈ {expected}")
        else:
            return (False, f"Full formula mismatch: result={self.Cp_minus_Cv}, expected={expected}")

    def test_invariant_positivity(self):
        """With positive givens, all quantities should be non-negative."""
        if self.V is not None and self.V < 0:
            return (False, f"Negative volume: V={self.V}")
        if self.dvdT is not None and self.dvdT < 0:
            return (False, f"Negative dV/dT: dvdT={self.dvdT}")
        # Cp_minus_Cv should be positive if p>0, α>0, etc.
        if self.p is not None and self.alpha is not None and self.M is not None and self.rho is not None:
            expected_sign = 3 * self.alpha * self.M * self.p / self.rho
            if expected_sign > 0 and self.Cp_minus_Cv is not None and self.Cp_minus_Cv <= 0:
                return (False, f"Expected positive Cp_minus_Cv but got {self.Cp_minus_Cv}")
        return (True, "All available quantities are non-negative")

    # ---------- Modeling / trace quality tests ----------
    def test_modeling_step_consistency(self):
        """
        Combined consistency: verify that V = M/ρ, dvdT = 3αV, and Cp‑Cv = p·dvdT.
        This checks the logical chain prescribed in the pseudocode.
        """
        if None in (self.V, self.dvdT, self.alpha, self.p, self.Cp_minus_Cv):
            return (False, "SKIP: required data missing")
        # V from mass
        if self.M is not None and self.rho is not None:
            V_from_M = self.M / self.rho
            if not math.isclose(self.V, V_from_M, rel_tol=1e-2, abs_tol=1e-2):
                return (False, "Modeling: V does not match M/ρ")
        # dvdT from alpha and V
        dvdT_expected = 3 * self.alpha * self.V
        if not math.isclose(self.dvdT, dvdT_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Modeling: dvdT does not equal 3αV")
        # Final identity
        final_expected = self.p * self.dvdT
        if not math.isclose(self.Cp_minus_Cv, final_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Modeling: Cp_minus_Cv does not equal p * dvdT")
        return (True, "Step‑by‑step modeling consistent")

    def test_modeling_volume_calculation(self):
        """Logical step: volume V is computed as M/ρ."""
        if self.V is None or self.M is None or self.rho is None:
            return (False, "SKIP: required data missing")
        rederived_V = self.M / self.rho
        if math.isclose(self.V, rederived_V, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Volume step correct: V = M/ρ ({self.V} ≈ {rederived_V})")
        else:
            return (False, f"Volume step mismatch: V={self.V}, computed M/ρ={rederived_V}")

    def test_modeling_dvdT_calculation(self):
        """Logical step: dV/dT = 3αV (volumetric expansion)."""
        if self.dvdT is None or self.alpha is None or self.V is None:
            return (False, "SKIP: required data missing")
        rederived_dvdT = 3 * self.alpha * self.V
        if math.isclose(self.dvdT, rederived_dvdT, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dV/dT step correct: dvdT = 3αV ({self.dvdT} ≈ {rederived_dvdT})")
        else:
            return (False, f"dV/dT step mismatch: dvdT={self.dvdT}, computed 3αV={rederived_dvdT}")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_Cp_minus_Cv_with_alpha(self):
        """Increasing alpha should increase Cp_minus_Cv (assuming p>0)."""
        if None in (self.alpha, self.Cp_minus_Cv, self.M, self.p, self.rho):
            return (False, "SKIP: required data missing")
        # Perturb alpha by +10%
        alpha_pert = self.alpha * 1.1
        Cp_pert = 3 * alpha_pert * self.M * self.p / self.rho
        if Cp_pert > self.Cp_minus_Cv:
            return (True, "Monotonic: Cp_minus_Cv increases with alpha")
        else:
            return (False, f"Monotonic fail: alpha increased, but Cp_minus_Cv did not increase "
                           f"(baseline={self.Cp_minus_Cv}, perturbed={Cp_pert})")

    def test_monotonicity_Cp_minus_Cv_with_M(self):
        """Increasing M should increase Cp_minus_Cv."""
        if None in (self.M, self.Cp_minus_Cv, self.alpha, self.p, self.rho):
            return (False, "SKIP: required data missing")
        M_pert = self.M * 1.1
        Cp_pert = 3 * self.alpha * M_pert * self.p / self.rho
        if Cp_pert > self.Cp_minus_Cv:
            return (True, "Monotonic: Cp_minus_Cv increases with M")
        else:
            return (False, f"Monotonic fail: M increased, but Cp_minus_Cv did not increase")

    def test_monotonicity_Cp_minus_Cv_with_rho(self):
        """Increasing rho should decrease Cp_minus_Cv (since it is in denominator)."""
        if None in (self.rho, self.Cp_minus_Cv, self.alpha, self.M, self.p):
            return (False, "SKIP: required data missing")
        rho_pert = self.rho * 1.1
        Cp_pert = 3 * self.alpha * self.M * self.p / rho_pert
        if Cp_pert < self.Cp_minus_Cv:
            return (True, "Monotonic: Cp_minus_Cv decreases with rho")
        else:
            return (False, f"Monotonic fail: rho increased, but Cp_minus_Cv did not decrease")

    # ---------- Answer sanity (weak, one only) ----------
    def test_answer_sanity_broad_range(self):
        """
        Weak sanity: Cp_minus_Cv should be finite and plausible.
        No formula derivation or isclose allowed.
        """
        if self.Cp_minus_Cv is None:
            return (False, "SKIP: Cp_minus_Cv not available")
        if not math.isfinite(self.Cp_minus_Cv):
            return (False, f"Cp_minus_Cv is not finite: {self.Cp_minus_Cv}")
        # Broad range: typical solids have Cp-Cv ~ a few J/K per kg, but keep it very loose.
        if abs(self.Cp_minus_Cv) > 1e9:
            return (False, f"Cp_minus_Cv magnitude suspiciously large: {self.Cp_minus_Cv}")
        return (True, "Sanity: finite and plausible magnitude")

    # ---------- Runner ----------
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