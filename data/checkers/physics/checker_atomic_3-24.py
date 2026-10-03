import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely using None defaults
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.r1 = results.get('r1', None)
        self.r2 = results.get('r2', None)
        self.r3 = results.get('r3', None)
        # Optional extraction of explanation steps for trace checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # Known physics constants (Clebsch-Gordan squared amplitudes)
    # K^-p: |<1,0|1/2,-1/2;1/2,1/2>|^2 = 1/2
    # bar{K}^0 n: |<1,0|1/2,1/2;1/2,-1/2>|^2 = 1/2
    # pi^- p: no I=1,I3=0 component -> 0
    _cg_sq_Kp = 0.5
    _cg_sq_barK0n = 0.5
    _cg_sq_pip = 0.0

    # Interface tests
    def test_interface_required_keys(self):
        if self.r1 is None:
            return (False, "r1 (Gamma_barK0n / Gamma_K-p) not found")
        if self.r2 is None:
            return (False, "r2 (Gamma_pi-p / Gamma_K-p) not found")
        if self.r3 is None:
            return (False, "r3 (Gamma_pi+pi- / Gamma_K-p) not found")
        if not isinstance(self.r1, (int, float)):
            return (False, f"r1 has invalid type: {type(self.r1)}")
        if not (isinstance(self.r2, (int, float)) or isinstance(self.r2, str)):
            return (False, f"r2 has invalid type: {type(self.r2)}")
        if not isinstance(self.r3, (int, float)):
            return (False, f"r3 has invalid type: {type(self.r3)}")
        return (True, "All required result keys present with appropriate types")

    # Invariant tests (theorem-based physical sanity)
    def test_invariant_isospin_KN_ratio(self):
        """Isospin invariance: K-p and bar{K}^0 n have same CG squared amplitude → ratio=1."""
        if self.r1 is None:
            return (False, "SKIP: r1 not available")
        # Both have same squared amplitude 0.5
        if not math.isclose(self._cg_sq_Kp, self._cg_sq_barK0n, rel_tol=1e-12):
            return (False, "Internal CG constants mismatch (should not happen)")
        expected = 1.0
        actual = float(self.r1)
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r1 ≈ {expected}: equal CG amplitudes")
        else:
            return (False, f"r1 = {actual}, expected ≈ {expected}")

    def test_invariant_isospin_forbidden_pi_p(self):
        """Pi-p decay has zero I=1,I3=0 component → r2 must indicate suppression."""
        if self.r2 is None:
            return (False, "SKIP: r2 not available")
        # The squared CG is zero; consequently the ratio should be <<1
        if isinstance(self.r2, str):
            if self.r2.lower() in ['suppressed', '<<1', 'small', 'forbidden']:
                return (True, f"r2 = '{self.r2}' correctly indicates suppression")
            else:
                return (False, f"r2 = '{self.r2}', expected suppression indication")
        elif isinstance(self.r2, (int, float)):
            if float(self.r2) < 0.01:
                return (True, f"r2 = {self.r2} is small, consistent with isospin-forbidden decay")
            else:
                return (False, f"r2 = {self.r2} is not small; expected <<1")
        else:
            return (False, "r2 has unexpected type")

    def test_invariant_baryon_number_forbidden(self):
        """Baryon number conservation forbids pi+pi- decay → r3=0."""
        if self.r3 is None:
            return (False, "SKIP: r3 not available")
        if not math.isclose(float(self.r3), 0.0, abs_tol=1e-12):
            return (False, f"r3 = {self.r3}, but must be 0 (baryon number violation)")
        return (True, "r3 = 0, baryon number conservation satisfied")

    # Modeling / trace tests
    def test_modeling_cg_symmetry(self):
        """Trace check: verify that the CG coefficients for KN decays are symmetric (modeling quality)."""
        # The squared amplitudes are both 0.5; check that they are equal
        if not math.isclose(self._cg_sq_Kp, self._cg_sq_barK0n, rel_tol=1e-12):
            return (False, "Internal CG constants mismatch")
        return (True, "CG squared amplitudes symmetric: both 0.5")

    def test_modeling_explanation_steps_keywords(self):
        """Trace check: if explanation_steps exist, verify key physics concepts are mentioned."""
        if self.explanation_steps is None:
            return (True, "SKIP: no explanation_steps provided")
        steps_text = str(self.explanation_steps).lower()
        required_terms = ['isospin', 'baryon number', 'clebsch']
        missing = [t for t in required_terms if t not in steps_text]
        if missing:
            return (False, f"Explanation steps missing terms: {missing}")
        return (True, "Explanation steps contain required physics concepts")

    # Monotonicity test (must perturb an input and compare to baseline)
    def test_monotonicity_r1_under_cg_perturbation(self):
        """
        Monotonicity: perturb the squared CG coefficient for K-p decay by +20%
        and recompute the ratio. The new ratio should be less than the baseline (r1=1).
        """
        if self.r1 is None:
            return (False, "SKIP: r1 not available")
        # Baseline ratio from solution
        baseline = float(self.r1)
        # Perturbation: increase K-p squared CG by 20%
        perturbed_cg_sq_Kp = self._cg_sq_Kp * 1.2  # 0.6
        # bar{K}^0 n squared CG remains 0.5
        new_ratio = self._cg_sq_barK0n / perturbed_cg_sq_Kp  # 0.5/0.6 = 0.8333...
        # Check monotonicity: new ratio should be less than baseline
        if new_ratio >= baseline:
            return (False, f"After increasing denominator CG, ratio increased from {baseline} to {new_ratio}, expected decrease")
        # Also check that the change is non-zero
        if math.isclose(new_ratio, baseline, rel_tol=1e-6):
            return (False, "Perturbation did not affect ratio")
        return (True, f"Monotonicity satisfied: perturbed ratio={new_ratio:.4f} < baseline={baseline}")

    def test_monotonicity_r1_under_cg_perturbation_reverse(self):
        """
        Monotonicity: perturb the squared CG coefficient for bar{K}^0 n decay by +20%
        and recompute the ratio. The new ratio should be greater than the baseline (r1=1).
        """
        if self.r1 is None:
            return (False, "SKIP: r1 not available")
        baseline = float(self.r1)
        # Perturbation: increase bar{K}^0 n squared CG by 20%
        perturbed_cg_sq_barK0n = self._cg_sq_barK0n * 1.2  # 0.6
        # K-p squared CG remains 0.5
        new_ratio = perturbed_cg_sq_barK0n / self._cg_sq_Kp  # 0.6/0.5 = 1.2
        if new_ratio <= baseline:
            return (False, f"After increasing numerator CG, ratio decreased from {baseline} to {new_ratio}, expected increase")
        if math.isclose(new_ratio, baseline, rel_tol=1e-6):
            return (False, "Perturbation did not affect ratio")
        return (True, f"Monotonicity satisfied: perturbed ratio={new_ratio:.4f} > baseline={baseline}")

    # Weak answer sanity (secondary)
    def test_answer_sanity_nonnegativity(self):
        """All ratios must be non-negative (weak sanity)."""
        values = [(self.r1, 'r1'), (self.r2 if isinstance(self.r2, (int,float)) else None, 'r2'), (self.r3, 'r3')]
        for val, name in values:
            if val is None:
                continue
            if float(val) < -1e-12:
                return (False, f"{name} is negative")
        return (True, "All reported ratios are non-negative")

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