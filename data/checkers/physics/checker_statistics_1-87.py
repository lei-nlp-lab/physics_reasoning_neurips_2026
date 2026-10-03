import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.R = givens.get('R', None)
        self.T = givens.get('T', None)
        self.I = givens.get('I', None)
        self.t = givens.get('t', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.dS_res = results.get('dS_res', None)
        self.W = results.get('W', None)
        self.dS_univ = results.get('dS_univ', None)
        self.dU_univ = results.get('dU_univ', None)
        self.dF_univ = results.get('dF_univ', None)

    # ==================== Interface tests ====================
    def test_interface_dS_res_present(self):
        if self.dS_res is None:
            return (False, "dS_res missing")
        return (True, "dS_res is present")

    def test_interface_W_present(self):
        if self.W is None:
            return (False, "W missing")
        return (True, "W is present")

    def test_interface_dS_univ_present(self):
        if self.dS_univ is None:
            return (False, "dS_univ missing")
        return (True, "dS_univ is present")

    def test_interface_dU_univ_present(self):
        if self.dU_univ is None:
            return (False, "dU_univ missing")
        return (True, "dU_univ is present")

    def test_interface_dF_univ_present(self):
        if self.dF_univ is None:
            return (False, "dF_univ missing")
        return (True, "dF_univ is present")

    def test_interface_givens_present(self):
        missing = []
        for name, val in [('R', self.R), ('T', self.T), ('I', self.I), ('t', self.t)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens present")

    # ==================== Invariant tests (physical theorem sanity) ====================
    def test_invariant_dS_res_zero(self):
        if self.dS_res is None:
            return (False, "SKIP: dS_res not available")
        if math.isclose(self.dS_res, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return (True, "dS_res = 0 as expected for cyclic process")
        return (False, f"dS_res = {self.dS_res}, expected 0")

    def test_invariant_dS_univ_equals_W_over_T(self):
        if any(x is None for x in [self.dS_univ, self.W, self.T]):
            return (False, "SKIP: missing dS_univ, W, or T")
        expected = self.W / self.T
        if math.isclose(self.dS_univ, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "dS_univ = W/T holds within tolerance")
        return (False, f"dS_univ = {self.dS_univ}, expected {expected}")

    def test_invariant_dU_univ_equals_W(self):
        if any(x is None for x in [self.dU_univ, self.W]):
            return (False, "SKIP: missing dU_univ or W")
        if math.isclose(self.dU_univ, self.W, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "dU_univ = W holds within tolerance")
        return (False, f"dU_univ = {self.dU_univ}, expected {self.W}")

    def test_invariant_dF_univ_zero_derived(self):
        if any(x is None for x in [self.dF_univ, self.dU_univ, self.dS_univ, self.T]):
            return (False, "SKIP: missing dF_univ, dU_univ, dS_univ, or T")
        predicted = self.dU_univ - self.T * self.dS_univ
        if math.isclose(self.dF_univ, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "dF_univ = dU_univ - T dS_univ ≈ 0")
        return (False, f"dF_univ = {self.dF_univ}, expected ≈0 (predicted {predicted})")

    # ==================== Modeling / trace quality tests ====================
    def test_modeling_W_positive_and_finite(self):
        if self.W is None:
            return (False, "SKIP: W not available")
        if self.W > 0 and math.isfinite(self.W):
            return (True, f"W = {self.W} is positive and finite")
        return (False, f"W = {self.W} is not positive/finite")

    def test_modeling_dS_univ_positive(self):
        if self.dS_univ is None:
            return (False, "SKIP: dS_univ not available")
        if self.dS_univ > 0 and math.isfinite(self.dS_univ):
            return (True, f"dS_univ = {self.dS_univ} is positive (second law)")
        return (False, f"dS_univ = {self.dS_univ} should be positive")

    # ==================== Monotonicity tests (input perturbation) ====================
    def test_monotonicity_dS_univ_increases_with_I(self):
        if any(x is None for x in [self.I, self.R, self.t, self.T, self.dS_univ]):
            return (False, "SKIP: missing data for monotonicity")
        I_pert = self.I * 1.01
        W_pert = I_pert**2 * self.R * self.t
        dS_pert = W_pert / self.T
        if dS_pert > self.dS_univ:
            return (True, "dS_univ increases when I increases")
        return (False, f"Perturbed dS_univ={dS_pert} ≤ baseline={self.dS_univ}")

    def test_monotonicity_W_increases_with_I(self):
        if any(x is None for x in [self.I, self.R, self.t, self.W]):
            return (False, "SKIP: missing data for monotonicity")
        I_pert = self.I * 1.01
        W_pert = I_pert**2 * self.R * self.t
        if W_pert > self.W:
            return (True, "W increases when I increases")
        return (False, f"Perturbed W={W_pert} ≤ baseline={self.W}")

    def test_monotonicity_dU_univ_increases_with_I(self):
        if any(x is None for x in [self.I, self.R, self.t, self.dU_univ]):
            return (False, "SKIP: missing data for monotonicity")
        I_pert = self.I * 1.01
        W_pert = I_pert**2 * self.R * self.t
        # dU_univ should equal W, so after perturbation dU_univ would be W_pert
        if W_pert > self.dU_univ:
            return (True, "dU_univ (equal to W) increases when I increases")
        return (False, f"Perturbed dU_univ (W_pert)={W_pert} ≤ baseline dU_univ={self.dU_univ}")

    # ==================== Answer sanity (weak, at most one) ====================
    def test_answer_sanity_magnitudes(self):
        failures = []
        if self.dU_univ is not None:
            if not (0 < self.dU_univ < 1e7):
                failures.append(f"dU_univ = {self.dU_univ} out of range (0, 1e7)")
        else:
            failures.append("dU_univ missing")
        if self.dS_univ is not None:
            if not (0 < self.dS_univ < 2e4):
                failures.append(f"dS_univ = {self.dS_univ} out of range (0, 2e4)")
        else:
            failures.append("dS_univ missing")
        if self.dF_univ is not None:
            if abs(self.dF_univ) > 1.0:
                failures.append(f"dF_univ = {self.dF_univ} not near zero")
        else:
            failures.append("dF_univ missing")
        if failures:
            return (False, "; ".join(failures))
        return (True, "Magnitudes within broad expected ranges")

    # ==================== Main runner ====================
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)