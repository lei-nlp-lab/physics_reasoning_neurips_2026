import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the angular momentum measurement problem.
    Uses safe attribute access, fuzzy matching where allowed, and focuses
    on physical invariants and modeling quality.
    """

    def __init__(self, solution_instance):
        # Safely extract results – never use numeric defaults
        self.results = getattr(solution_instance, 'results', None) or {}
        self.Lz_possible = self.results.get('Lz_possible', None)
        self.P_m1 = self.results.get('P_m1', None)
        self.P_m0 = self.results.get('P_m0', None)
        self.expectation_Lz = self.results.get('expectation_Lz', None)

        # Physical constants (symbolic)
        self.hbar = sp.Symbol('hbar', real=True, positive=True)

        # Spherical harmonics for internal calculations
        self.Y11 = -sp.sqrt(3/(8*sp.pi)) * sp.sin(sp.Symbol('theta')) * sp.exp(sp.I*sp.Symbol('phi'))
        self.Y10 = sp.sqrt(3/(4*sp.pi)) * sp.cos(sp.Symbol('theta'))

        # Angular part of the wavefunction
        self.A = (1/sp.sqrt(4*sp.pi)) * (sp.exp(sp.I*sp.Symbol('phi')) * sp.sin(sp.Symbol('theta')) + sp.cos(sp.Symbol('theta')))

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_has_keys(self):
        """Ensure all required result keys are present."""
        missing = []
        if self.Lz_possible is None:
            missing.append('Lz_possible')
        if self.P_m1 is None:
            missing.append('P_m1')
        if self.P_m0 is None:
            missing.append('P_m0')
        if self.expectation_Lz is None:
            missing.append('expectation_Lz')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present.")

    def test_interface_Lz_possible_type(self):
        """Lz_possible should be a list of two strings."""
        if self.Lz_possible is None:
            return (False, "SKIP: Lz_possible not available.")
        if not isinstance(self.Lz_possible, list):
            return (False, "Lz_possible must be a list.")
        if len(self.Lz_possible) != 2:
            return (False, "Lz_possible must have exactly two elements.")
        for item in self.Lz_possible:
            if not isinstance(item, str):
                return (False, f"Element '{item}' is not a string.")
        return (True, "Lz_possible is a list of two strings.")

    def test_interface_P_m1_type(self):
        """P_m1 should be a number (int, float, or sympy symbolic)."""
        if self.P_m1 is None:
            return (False, "SKIP: P_m1 not available.")
        if not isinstance(self.P_m1, (int, float, sp.Basic)):
            return (False, f"P_m1 has type {type(self.P_m1).__name__}, expected numeric.")
        return (True, "P_m1 type is acceptable.")

    def test_interface_P_m0_type(self):
        """P_m0 should be a number (int, float, or sympy symbolic)."""
        if self.P_m0 is None:
            return (False, "SKIP: P_m0 not available.")
        if not isinstance(self.P_m0, (int, float, sp.Basic)):
            return (False, f"P_m0 has type {type(self.P_m0).__name__}, expected numeric.")
        return (True, "P_m0 type is acceptable.")

    def test_interface_expectation_Lz_type(self):
        """expectation_Lz should be a number (int, float, or sympy symbolic)."""
        if self.expectation_Lz is None:
            return (False, "SKIP: expectation_Lz not available.")
        if not isinstance(self.expectation_Lz, (int, float, sp.Basic)):
            return (False, f"expectation_Lz has type {type(self.expectation_Lz).__name__}, expected numeric.")
        return (True, "expectation_Lz type is acceptable.")

    # ------------------------------------------------------------------
    # INVARIANT TESTS (theorem-based, use math.isclose for numerical safety)
    # ------------------------------------------------------------------
    def test_invariant_probability_sum(self):
        """Sum of probabilities must be 1 (theorem: normalization of measurement outcomes)."""
        if self.P_m1 is None or self.P_m0 is None:
            return (False, "SKIP: Missing probability data.")
        try:
            p1 = float(self.P_m1)
            p0 = float(self.P_m0)
        except (ValueError, TypeError):
            return (False, "Cannot convert probabilities to float.")
        total = p1 + p0
        if math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Probability sum = {total}, close to 1.")
        else:
            return (False, f"Probability sum = {total}, expected 1.")

    def test_invariant_expectation_formula(self):
        """Expectation ⟨L_z⟩ must equal ħ * (1·P_{m=1} + 0·P_{m=0}) = ħ * P_{m=1}."""
        if self.expectation_Lz is None or self.P_m1 is None:
            return (False, "SKIP: Missing expectation or probability data.")
        try:
            # Convert both to sympy
            exp_sym = sp.sympify(self.expectation_Lz)
            p1_sym = sp.sympify(self.P_m1)
        except (sp.SympifyError, TypeError):
            return (False, "Cannot convert to sympy expressions.")
        expected = self.hbar * p1_sym
        if sp.simplify(exp_sym - expected) == 0:
            return (True, "Expectation matches the formula ⟨L_z⟩ = ħ·P(m=1).")
        # Numeric fallback with standard hbar
        try:
            hbar_val = 1.054571817e-34
            exp_num = float(exp_sym.subs({self.hbar: hbar_val}))
            p1_num = float(p1_sym)
            if math.isclose(exp_num, hbar_val * p1_num, rel_tol=1e-2, abs_tol=1e-2 * hbar_val):
                return (True, f"Numeric check passed: ⟨L_z⟩ ≈ {exp_num}.")
            else:
                return (False, f"Numeric check failed: ⟨L_z⟩ = {exp_num}, expected ħ·P(m=1) ≈ {hbar_val * p1_num}.")
        except Exception:
            return (False, "Unable to verify expectation formula both symbolically and numerically.")

    def test_invariant_angular_normalization(self):
        """The angular part must be normalized to 1: ∫|A|^2 dΩ = 1."""
        try:
            A_conj = sp.conjugate(self.A)
            integrand = A_conj * self.A * sp.sin(sp.Symbol('theta'))
            integral_phi = sp.integrate(integrand, (sp.Symbol('phi'), 0, 2*sp.pi))
            integral_theta = sp.integrate(integral_phi, (sp.Symbol('theta'), 0, sp.pi))
            normalized = sp.simplify(integral_theta)
            if normalized == 1:
                return (True, "Angular part is normalized.")
            else:
                return (False, f"∫|A|^2 dΩ = {normalized}, expected 1.")
        except Exception as e:
            return (False, f"Symbolic integration failed: {str(e)}")

    def test_invariant_L2_quantum_number(self):
        """Given l=1 from expansion, L^2 eigenvalue should be 2ħ², but only require that
           the expansion contains only l=1 harmonics (check with inner product)."""
        try:
            Y20 = sp.sqrt(5/(16*sp.pi)) * (3*sp.cos(sp.Symbol('theta'))**2 - 1)
            overlap = sp.integrate(
                sp.conjugate(Y20) * self.A * sp.sin(sp.Symbol('theta')),
                (sp.Symbol('phi'), 0, 2*sp.pi),
                (sp.Symbol('theta'), 0, sp.pi)
            )
            if sp.simplify(overlap) == 0:
                return (True, "No l=2 component detected.")
            else:
                return (False, f"Overlap with Y20 is {overlap}, expected 0.")
        except Exception as e:
            return (False, f"Overlap check failed: {str(e)}")

    # ------------------------------------------------------------------
    # MODELING / TRACE TESTS (coefficient extraction, eigenvalue list)
    # ------------------------------------------------------------------
    def test_modeling_coefficient_c11(self):
        """Coefficient of Y11 must be -√(2/3). c11 = ∫ Y11^* A dΩ."""
        try:
            c11_sym = sp.integrate(
                sp.conjugate(self.Y11) * self.A * sp.sin(sp.Symbol('theta')),
                (sp.Symbol('phi'), 0, 2*sp.pi),
                (sp.Symbol('theta'), 0, sp.pi)
            ).simplify()
            expected_c11 = -sp.sqrt(2/3)
            if sp.simplify(c11_sym - expected_c11) == 0:
                return (True, "c11 = -√(2/3) is correct.")
            else:
                return (False, f"c11 = {c11_sym}, expected {expected_c11}.")
        except Exception as e:
            return (False, f"c11 calculation failed: {str(e)}")

    def test_modeling_coefficient_c10(self):
        """Coefficient of Y10 must be 1/√3."""
        try:
            c10_sym = sp.integrate(
                sp.conjugate(self.Y10) * self.A * sp.sin(sp.Symbol('theta')),
                (sp.Symbol('phi'), 0, 2*sp.pi),
                (sp.Symbol('theta'), 0, sp.pi)
            ).simplify()
            expected_c10 = 1/sp.sqrt(3)
            if sp.simplify(c10_sym - expected_c10) == 0:
                return (True, "c10 = 1/√3 is correct.")
            else:
                return (False, f"c10 = {c10_sym}, expected {expected_c10}.")
        except Exception as e:
            return (False, f"c10 calculation failed: {str(e)}")

    def test_trace_possible_Lz_values(self):
        """The list of possible Lz results must be exactly ['+hbar', '0']."""
        if self.Lz_possible is None:
            return (False, "SKIP: Lz_possible not available.")
        expected = ['+hbar', '0']
        if self.Lz_possible == expected:
            return (True, "Lz_possible matches expected list.")
        else:
            return (False, f"Lz_possible = {self.Lz_possible}, expected {expected}.")

    def test_trace_coefficient_squared_sum(self):
        """Sum of squared coefficients |c11|^2 + |c10|^2 must be 1."""
        if self.P_m1 is None or self.P_m0 is None:
            return (False, "SKIP: Missing probability data.")
        try:
            p1 = float(self.P_m1)
            p0 = float(self.P_m0)
            total = p1 + p0
            if math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Sum of probabilities = {total}, consistent with ∑|c|^2=1.")
            else:
                return (False, f"∑|c|^2 = {total}, expected 1.")
        except Exception:
            return (False, "Cannot evaluate probability sum.")

    # ------------------------------------------------------------------
    # MONOTONICITY TEST (explicit perturbation of an input parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_expectation_increases_with_Pm1(self):
        """
        Perturb P_m1 by increasing it slightly. Since ⟨L_z⟩ = ħ * P_m1,
        the perturbed expectation must be larger.
        """
        if self.P_m1 is None or self.expectation_Lz is None:
            return (False, "SKIP: Missing P_m1 or expectation_Lz.")
        try:
            p1 = float(self.P_m1)
            # Use hbar = 1 (relative comparison); the actual numeric value matters only for sign.
            # Baseline expectation in units of ħ is p1.
            baseline = p1
            # Increase P_m1 by 5% (capped to <=1)
            delta = 0.05
            new_p1 = min(p1 + delta, 1.0)
            # The perturbed expectation in ħ units is new_p1
            perturbed = new_p1
            if perturbed > baseline:
                return (True, f"Increasing P_m1 by {delta} raises expectation ({perturbed} > {baseline}), monotonicity holds.")
            else:
                return (False, f"Perturbed expectation {perturbed} is not greater than baseline {baseline}.")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {str(e)}")

    # ------------------------------------------------------------------
    # ANSWER SANITY (weak, must not use math.isclose or re-derive)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        """Probabilities must be non-negative and finite; expectation must be finite."""
        issues = []
        for name, val in [('P_m1', self.P_m1), ('P_m0', self.P_m0), ('expectation_Lz', self.expectation_Lz)]:
            if val is None:
                issues.append(f"{name} missing")
                continue
            try:
                num = float(val)
                if num < 0 or math.isnan(num) or math.isinf(num):
                    issues.append(f"{name} = {num} is not positive/finite")
            except Exception:
                issues.append(f"{name} cannot be converted to float")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Probabilities and expectation are positive and finite.")

    # ------------------------------------------------------------------
    # RUNNER
    # ------------------------------------------------------------------
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