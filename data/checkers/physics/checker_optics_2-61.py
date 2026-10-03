import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction from solution_instance (assume it might have attributes or results dict)
        self.lambda_ = getattr(solution_instance, 'lambda', None) or solution_instance.results.get('lambda', None)
        self.d = getattr(solution_instance, 'd', None) or solution_instance.results.get('d', None)
        self.r = getattr(solution_instance, 'r', None) or solution_instance.results.get('r', None)
        self.costheta = getattr(solution_instance, 'costheta', None) or solution_instance.results.get('costheta', None)
        self.m = getattr(solution_instance, 'm', None) or solution_instance.results.get('m', None)

        self.results = getattr(solution_instance, 'results', {})
        self.path_diff = self.results.get('path_diff', None)
        self.fringe_order = self.results.get('fringe_order', None)
        self.finesse = self.results.get('finesse', None)
        self.resolving_power = self.results.get('resolving_power', None)

        # Explanation steps (for reference, not used directly)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

    # --- INTERFACE TESTS ---
    def test_interface_path_diff_present(self):
        if self.path_diff is None:
            return (False, "SKIP: path_diff not provided")
        return (True, "path_diff present")

    def test_interface_fringe_order_present(self):
        if self.fringe_order is None:
            return (False, "SKIP: fringe_order not provided")
        return (True, "fringe_order present")

    def test_interface_finesse_present(self):
        if self.finesse is None:
            return (False, "SKIP: finesse not provided")
        return (True, "finesse present")

    def test_interface_resolving_power_present(self):
        if self.resolving_power is None:
            return (False, "SKIP: resolving_power not provided")
        return (True, "resolving_power present")

    def test_interface_results_dict(self):
        if not isinstance(self.results, dict):
            return (False, "SKIP: results is not a dict")
        return (True, "results is a dict")

    # --- INVARIANT TESTS (theorem-based physical sanity, use math.isclose) ---
    def test_invariant_maxima_condition(self):
        # Check 2*d*costheta = m*lambda
        if any(v is None for v in [self.d, self.costheta, self.m, self.lambda_]):
            return (False, "SKIP: missing d, costheta, m, or lambda for maxima condition")
        LHS = 2 * self.d * self.costheta
        RHS = self.m * self.lambda_
        if math.isclose(LHS, RHS, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Maxima condition holds: 2*d*costheta = {LHS:.4f} ≈ m*lambda = {RHS:.4f}")
        else:
            return (False, f"Maxima condition violated: 2*d*costheta = {LHS:.4f}, m*lambda = {RHS:.4f}")

    def test_invariant_finesse_formula(self):
        # Check F = pi*sqrt(R)/(1-R) where R = r^2
        if self.r is None or self.finesse is None:
            return (False, "SKIP: missing r or finesse")
        R = self.r ** 2
        if math.isclose(R, 1.0):  # avoid division by zero
            pass
        expected_finesse = math.pi * math.sqrt(R) / (1 - R)
        if math.isclose(self.finesse, expected_finesse, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Finesse formula holds: F = {self.finesse:.4f} ≈ pi*sqrt(R)/(1-R) = {expected_finesse:.4f}")
        else:
            return (False, f"Finesse mismatch: got {self.finesse:.4f}, expected {expected_finesse:.4f}")

    def test_invariant_resolving_power(self):
        # Check resolving power = m * F
        if any(v is None for v in [self.m, self.finesse, self.resolving_power]):
            return (False, "SKIP: missing m, finesse, or resolving_power")
        expected_R = self.m * self.finesse
        if math.isclose(self.resolving_power, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Resolving power = m*F = {self.resolving_power:.4f} ≈ {expected_R:.4f}")
        else:
            return (False, f"Resolving power mismatch: got {self.resolving_power:.4f}, expected {expected_R:.4f}")

    def test_invariant_fringe_order_integer(self):
        # m should be integer (or at least close)
        if self.m is None:
            return (False, "SKIP: m not provided")
        if math.isclose(self.m, round(self.m), rel_tol=1e-3, abs_tol=1e-2):
            return (True, f"Fringe order m = {self.m} is close to integer {round(self.m)}")
        else:
            return (False, f"Fringe order m = {self.m} is not near an integer")

    # --- MODELING QUALITY TESTS (based on explanation steps) ---
    def test_modeling_path_diff_sign(self):
        # path_diff should be positive (d, costheta >0)
        if self.path_diff is None or self.path_diff <= 0:
            return (False, "SKIP: path_diff missing or non-positive")
        return (True, f"path_diff = {self.path_diff:.4f} > 0")

    def test_modeling_finesse_bounds(self):
        # Finesse should be > 0 for 0<r<1
        if self.finesse is None or self.finesse <= 0:
            return (False, "SKIP: finesse missing or non-positive")
        if math.isfinite(self.finesse):
            return (True, f"Finesse = {self.finesse:.4f} positive finite")
        else:
            return (False, "Finesse is not finite")

    def test_modeling_resolving_power_positive(self):
        if self.resolving_power is None or self.resolving_power <= 0:
            return (False, "SKIP: resolving_power missing or non-positive")
        if math.isfinite(self.resolving_power):
            return (True, f"Resolving power = {self.resolving_power:.4f} positive finite")
        else:
            return (False, "Resolving power is not finite")

    # --- MONOTONICITY TESTS (perturb one parameter and check trend) ---
    def test_monotonicity_finesse_increases_with_r(self):
        # Finesse increases as r increases (for 0<r<1)
        if self.r is None or self.finesse is None:
            return (False, "SKIP: missing r or finesse")
        # Perturb r upward by 10%
        r_pert = self.r * 1.1
        if r_pert >= 1.0:
            return (False, "SKIP: perturbed r exceeds 1")
        R_pert = r_pert ** 2
        expected_finesse_pert = math.pi * math.sqrt(R_pert) / (1 - R_pert)
        if expected_finesse_pert > self.finesse:
            return (True, f"Finesse increases with r: original F={self.finesse:.4f}, perturbed F={expected_finesse_pert:.4f}")
        else:
            return (False, f"Finesse did not increase: original F={self.finesse:.4f}, perturbed F={expected_finesse_pert:.4f}")

    def test_monotonicity_resolving_power_increases_with_m(self):
        # Resolving power = m * F, so increases with m (assuming F constant)
        if self.m is None or self.finesse is None or self.resolving_power is None:
            return (False, "SKIP: missing m, finesse, or resolving_power")
        # Perturb m upward by 10%
        m_pert = self.m * 1.1
        expected_R_pert = m_pert * self.finesse
        if expected_R_pert > self.resolving_power:
            return (True, f"Resolving power increases with m: original R={self.resolving_power:.4f}, perturbed R={expected_R_pert:.4f}")
        else:
            return (False, f"Resolving power did not increase: original R={self.resolving_power:.4f}, perturbed R={expected_R_pert:.4f}")

    # --- ANSWER SANITY TEST (weak, only one allowed) ---
    def test_answer_sanity_resolving_power_range(self):
        # Weak check: resolving power should be > 0 and not astronomically large
        if self.resolving_power is None:
            return (False, "SKIP: resolving_power missing")
        if self.resolving_power <= 0 or not math.isfinite(self.resolving_power):
            return (False, "Resolving power non-positive or infinite")
        if self.resolving_power > 1e12:   # plausible upper bound
            return (False, f"Resolving power too large: {self.resolving_power:.2e}")
        return (True, f"Resolving power = {self.resolving_power:.4e} within plausible range")

    # --- RUN ALL TESTS ---
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Sort methods to get deterministic order
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