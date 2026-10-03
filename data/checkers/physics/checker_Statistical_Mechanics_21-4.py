import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract output results safely (None if missing)
        self.L = getattr(solution_instance, 'results', {}).get('L', None)
        self.H_w = getattr(solution_instance, 'results', {}).get('H_w', None)
        self.G_w = getattr(solution_instance, 'results', {}).get('G_w', None)
        self.G_s = getattr(solution_instance, 'results', {}).get('G_s', None)

        # Known physical constants from problem statement (ground truth)
        self.T = 373.15          # 100°C
        self.S_w = 0.31          # cal/g·K
        self.S_s = 1.76          # cal/g·K
        self.H_s = 640.0         # cal/g

        # Try to extract given values from solution (if present) for informative messages
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Explanation steps (list of strings or string)
        self.explanation = getattr(solution_instance, 'explanation_steps', [])
        if not isinstance(self.explanation, list):
            self.explanation = [str(self.explanation)]

    # ------------------------------------------------------------------
    # Interface tests: required output keys exist
    # ------------------------------------------------------------------
    def test_interface_L_key_present(self):
        if self.L is None:
            return (False, "Missing result key 'L'")
        return (True, "'L' present")

    def test_interface_H_w_key_present(self):
        if self.H_w is None:
            return (False, "Missing result key 'H_w'")
        return (True, "'H_w' present")

    def test_interface_G_w_key_present(self):
        if self.G_w is None:
            return (False, "Missing result key 'G_w'")
        return (True, "'G_w' present")

    def test_interface_G_s_key_present(self):
        if self.G_s is None:
            return (False, "Missing result key 'G_s'")
        return (True, "'G_s' present")

    # ------------------------------------------------------------------
    # Invariant tests: physics relationships (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_heat_of_vaporization(self):
        needed = [self.L, self.T, self.S_w, self.S_s]
        if any(v is None for v in needed):
            return (False, "SKIP: missing L, T, S_w, or S_s")
        delta_S = self.S_s - self.S_w
        expected_L = self.T * delta_S
        if math.isclose(self.L, expected_L, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L = {self.L} ≈ T*ΔS = {expected_L}")
        return (False, f"L = {self.L} differs from expected {expected_L} (T*ΔS)")

    def test_invariant_water_enthalpy(self):
        needed = [self.H_w, self.H_s, self.L]
        if any(v is None for v in needed):
            return (False, "SKIP: missing H_w, H_s, or L")
        expected_H_w = self.H_s - self.L
        if math.isclose(self.H_w, expected_H_w, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"H_w = {self.H_w} ≈ H_s - L = {expected_H_w}")
        return (False, f"H_w = {self.H_w} differs from expected {expected_H_w} (H_s - L)")

    def test_invariant_gibbs_water(self):
        needed = [self.G_w, self.H_w, self.T, self.S_w]
        if any(v is None for v in needed):
            return (False, "SKIP: missing G_w, H_w, T, or S_w")
        expected_G_w = self.H_w - self.T * self.S_w
        if math.isclose(self.G_w, expected_G_w, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"G_w = {self.G_w} ≈ H_w - T*S_w = {expected_G_w}")
        return (False, f"G_w = {self.G_w} differs from expected {expected_G_w} (H_w - T*S_w)")

    def test_invariant_gibbs_steam(self):
        needed = [self.G_s, self.H_s, self.T, self.S_s]
        if any(v is None for v in needed):
            return (False, "SKIP: missing G_s, H_s, T, or S_s")
        expected_G_s = self.H_s - self.T * self.S_s
        if math.isclose(self.G_s, expected_G_s, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"G_s = {self.G_s} ≈ H_s - T*S_s = {expected_G_s}")
        return (False, f"G_s = {self.G_s} differs from expected {expected_G_s} (H_s - T*S_s)")

    def test_invariant_gibbs_equality(self):
        needed = [self.G_w, self.G_s]
        if any(v is None for v in needed):
            return (False, "SKIP: missing G_w or G_s")
        if math.isclose(self.G_w, self.G_s, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"G_w = {self.G_w} ≈ G_s = {self.G_s} (equilibrium)")
        return (False, f"G_w = {self.G_w} and G_s = {self.G_s} are not equal (ΔG should be 0)")

    # ------------------------------------------------------------------
    # Modeling / reasoning quality checks
    # ------------------------------------------------------------------
    def test_modeling_phase_equilibrium_mentioned(self):
        for step in self.explanation:
            text = step.lower() if isinstance(step, str) else str(step).lower()
            if "phase equilibrium" in text or "Δg" in text or "∆g" in text or "delta g" in text:
                return (True, "Phase equilibrium reasoning found")
        return (False, "Missing reference to phase equilibrium in explanation steps")

    def test_modeling_relation_L_T_deltaS(self):
        for step in self.explanation:
            text = step if isinstance(step, str) else str(step)
            if "L = T * (S_s - S_w)" in text or "L = T * ΔS" in text or "L = T \Delta S" in text:
                return (True, "Vaporization relation present")
        return (False, "Missing L = T*ΔS relation in explanation")

    # ------------------------------------------------------------------
    # Monotonicity test: perturb S_s and check L increases
    # ------------------------------------------------------------------
    def test_monotonicity_heat_of_vaporization_with_entropy(self):
        needed = [self.T, self.S_w, self.S_s, self.L]
        if any(v is None for v in needed):
            return (False, "SKIP: missing T, S_w, S_s, or L")
        # First verify baseline consistency (self.L should match T*(S_s - S_w))
        baseline = self.T * (self.S_s - self.S_w)
        if not math.isclose(self.L, baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline L mismatch: {self.L} vs {baseline}")
        # Perturb S_s upward by 0.1 cal/g·K
        S_s_perturbed = self.S_s + 0.1
        L_perturbed = self.T * (S_s_perturbed - self.S_w)
        if L_perturbed > self.L:
            return (True, f"Increasing S_s increases L (L_pert={L_perturbed} > L={self.L})")
        return (False, f"Monotonicity violated: increasing S_s produced L_pert={L_perturbed} ≤ L={self.L}")

    # ------------------------------------------------------------------
    # Weak answer sanity check (only one, limited)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_and_finite(self):
        values = [self.L, self.H_w, self.G_w, self.G_s]
        if any(v is None for v in values):
            return (False, "SKIP: missing some result values")
        for name, val in zip(['L','H_w','G_w','G_s'], values):
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                return (False, f"{name} is not a finite number: {val}")
        if self.L <= 0:
            return (False, f"L should be positive, got {self.L}")
        return (True, "All output values are finite, L > 0")

    # ------------------------------------------------------------------
    # Master runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)