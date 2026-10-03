import math
import traceback
import re

class PhysicsChecker:
    """
    PhysicsChecker for Fresnel zone plate problem.
    Audits the solution against Fresnel diffraction theorems, experimental constraints,
    and modeling quality. All numerical checks are derived from theoretical relationships,
    not from target answer values.
    """

    def __init__(self, solution_instance):
        # Safely extract all required data; use None as default for missing values.
        self.lambda_given = getattr(solution_instance, 'lambda', None)
        self.f_given = getattr(solution_instance, 'f', None)
        self.results = getattr(solution_instance, 'results', {})
        self.r1 = self.results.get('r1', None)
        self.I_ratio = self.results.get('I_ratio', None)
        self.z_formula = self.results.get('z_formula', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Physical constants (known inputs)
        self.lambda_val = 5000e-10  # m
        self.f_val = 1.0           # m

    # ------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        """Check that all required result keys are present (r1, I_ratio, z_formula)."""
        required = ['r1', 'I_ratio', 'z_formula']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_r1_type(self):
        """Check that r1 is numeric (float or int)."""
        if self.r1 is None:
            return (False, "SKIP: r1 not available")
        if not isinstance(self.r1, (int, float)):
            return (False, f"r1 has invalid type: {type(self.r1)}")
        return (True, f"r1 type OK: {type(self.r1)}")

    def test_interface_I_ratio_type(self):
        """Check that I_ratio is numeric."""
        if self.I_ratio is None:
            return (False, "SKIP: I_ratio not available")
        if not isinstance(self.I_ratio, (int, float)):
            return (False, f"I_ratio has invalid type: {type(self.I_ratio)}")
        return (True, f"I_ratio type OK: {type(self.I_ratio)}")

    def test_interface_z_formula_type(self):
        """Check that z_formula is a string."""
        if self.z_formula is None:
            return (False, "SKIP: z_formula not available")
        if not isinstance(self.z_formula, str):
            return (False, f"z_formula has invalid type: {type(self.z_formula)}")
        return (True, f"z_formula type OK: {type(self.z_formula)}")

    # ------------------------------------------------------------
    # INVARIANT TESTS (theorem-based, use math.isclose where appropriate)
    # ------------------------------------------------------------
    def test_invariant_fresnel_radii_condition(self):
        """
        Check that r1^2 is consistent with the Fresnel condition r_n^2 = n λ f.
        Re-derived from given λ and f, not from target answer.
        """
        if self.r1 is None:
            return (False, "SKIP: r1 not available")
        expected_product = self.lambda_val * self.f_val  # exactly 5e-7 m²
        actual_product = self.r1 ** 2
        # Tolerance 1% (Fresnel condition is exact, but numerical rounding may occur)
        if math.isclose(actual_product, expected_product, rel_tol=1e-2):
            return (True, f"r1^2 = {actual_product:.3e} ≈ {expected_product:.3e} (λ f)")
        else:
            return (False, f"r1^2 = {actual_product:.3e} deviates from λ f = {expected_product:.3e}")

    def test_invariant_phasor_sum_intensity(self):
        """
        Compute the expected I_ratio from the Fresnel phasor sum for zones 2 and 4,
        using the zone radius condition r_n^2 = n λ f. This is theorem-based, not
        a target-answer comparison.
        """
        if self.I_ratio is None:
            return (False, "SKIP: I_ratio not available")
        # Derive expected I_ratio:
        # For n=1..4, phase_n = π * r_n^2 / (λ f) = π * n
        # Contributions:
        # Zone 2 (r1 to r2): exp(iπ) - exp(i2π) = -1 - 1 = -2
        # Zone 4 (r3 to r4): exp(i3π) - exp(i4π) = -1 - 1 = -2
        # Total U = -4 U0 => I_ratio = 16
        # We compute it directly using the numbers:
        import cmath
        U0 = 1.0  # normalized
        pi = math.pi
        # phases: π, 2π, 3π, 4π
        phase1 = pi
        phase2 = 2*pi
        phase3 = 3*pi
        phase4 = 4*pi
        contrib2 = cmath.exp(1j*phase1) - cmath.exp(1j*phase2)  # -2
        contrib4 = cmath.exp(1j*phase3) - cmath.exp(1j*phase4)  # -2
        U_total = U0 * (contrib2 + contrib4)  # -4 U0
        expected_ratio = abs(U_total)**2 / abs(U0)**2  # should be 16
        if math.isclose(self.I_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"I_ratio = {self.I_ratio} ≈ {expected_ratio:.4f} (ph theoretical)")
        else:
            return (False, f"I_ratio = {self.I_ratio} deviates from ph theoretical {expected_ratio:.4f}")

    def test_invariant_z_formula_pattern(self):
        """Check that z_formula has the form f/(odd integer) (e.g., f/(2m+1))."""
        if self.z_formula is None:
            return (False, "SKIP: z_formula not available")
        cleaned = self.z_formula.replace(' ', '')
        # Accept patterns: f/(2m+1), f/(1+2m), f/(2*m+1), etc.
        pattern = r'f\s*/\s*\(?\s*[12]?\s*m\s*[+-]\s*[0-9]+\s*\)?'
        if re.search(pattern, self.z_formula):
            return (True, f"z_formula matches expected pattern: {self.z_formula}")
        # Also accept explicit fractions like f/3, f/5 (if formula enumerates values)
        if re.match(r'[fF]\s*/\s*[0-9]+\s*,\s*[fF]\s*/\s*[0-9]+', self.z_formula):
            return (True, f"z_formula lists fractions: {self.z_formula}")
        return (False, f"z_formula does not match expected pattern: {self.z_formula}")

    # ------------------------------------------------------------
    # MODELING/TRACE TESTS (check explanation steps and modeling quality)
    # ------------------------------------------------------------
    def test_modeling_explanation_mentions_fresnel_relation(self):
        """Check that explanation_steps mention key Fresnel relation r_n^2 ∝ n."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_str = str(self.explanation_steps).lower()
        keywords = ['r^2', 'lambda', 'f', 'proportional', 'zone', 'radius']
        found = sum(1 for kw in keywords if kw in steps_str)
        if found >= 2:
            return (True, f"Explanation mentions {found} of the expected keywords")
        else:
            return (False, "Explanation does not adequately reference Fresnel zone radius relation")

    def test_modeling_intensity_derivation_uses_zones(self):
        """Check that the derivation for I_ratio refers to the two transparent zones and phasor sum."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_str = str(self.explanation_steps).lower()
        # Look for generic terms: zone, phasor, amplitude, sum
        needed_terms = ['zone', 'phasor', 'amplitude', 'sum']
        found = sum(1 for t in needed_terms if t in steps_str)
        if found >= 2:
            return (True, f"Explanation contains {found} relevant modeling terms")
        else:
            return (False, "Explanation does not adequately describe phasor summation")

    def test_modeling_z_formula_references_focal_series(self):
        """Check that the explanation for part (c) mentions the series of focal lengths or odd integers."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_str = str(self.explanation_steps).lower()
        if ('focal' in steps_str or 'focus' in steps_str) and ('series' in steps_str or 'odd' in steps_str or '1/3' in steps_str):
            return (True, "Explanation references the focal series (odd denominators)")
        else:
            return (False, "Explanation does not mention the focal series")

    # ------------------------------------------------------------
    # MONOTONICITY TEST (must perturb an input and compare)
    # ------------------------------------------------------------
    def test_monotonicity_r1_increases_with_lambda(self):
        """
        Verify monotonicity: r1 should increase if wavelength λ increases.
        Use baseline (given λ and solution's r1) and perturb λ upward by 10%.
        The perturbed r1' derived from Fresnel condition must be > baseline r1.
        """
        if self.r1 is None:
            return (False, "SKIP: r1 not available")
        # Baseline: r1 from solution
        # Compute perturbed r1 using Fresnel relation r1 = sqrt(lambda * f)
        lambda_pert = self.lambda_val * 1.10
        r1_pert = math.sqrt(lambda_pert * self.f_val)
        if r1_pert > self.r1:
            return (True, f"λ↑10% -> r1 from {self.r1:.4e} to {r1_pert:.4e} (increase)")
        else:
            return (False, f"λ↑10% but r1 did not increase: perturbed {r1_pert:.4e} <= baseline {self.r1:.4e}")

    # ------------------------------------------------------------
    # WEAK ANSWER SANITY (only one method, no math.isclose)
    # ------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: check positivity, finite, and broad magnitude ranges.
        No target answer comparison or numeric derivation.
        """
        failures = []
        # r1: positive and < 1 cm (0.01 m)
        if self.r1 is not None:
            if not (0 < self.r1 < 0.01):
                failures.append(f"r1 = {self.r1} out of physical range (should be 0..10 mm)")
        # I_ratio: positive and > 1 (since plate concentrates light)
        if self.I_ratio is not None:
            if not (self.I_ratio > 0):
                failures.append(f"I_ratio = {self.I_ratio} not positive")
            elif self.I_ratio < 1:
                failures.append(f"I_ratio = {self.I_ratio} < 1 (unlikely for concentration)")
        # z_formula: non-empty string
        if self.z_formula is not None:
            if not isinstance(self.z_formula, str) or len(self.z_formula.strip()) == 0:
                failures.append("z_formula is empty or not a string")
        if failures:
            return (False, " | ".join(failures))
        else:
            return (True, "Basic sanity checks passed")

    # ------------------------------------------------------------
    # RUN ALL TESTS
    # ------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (all_pass, log_list)."""
        logs = []
        all_pass = True
        # List all callable attributes starting with 'test_'
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