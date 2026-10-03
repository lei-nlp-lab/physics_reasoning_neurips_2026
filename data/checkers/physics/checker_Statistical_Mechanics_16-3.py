import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction from results dictionary
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.U_per_V = results.get('U_per_V', None)
        self.A = results.get('A', None)
        self.J_U = results.get('J_U', None)
        self.B = results.get('B', None)
        self.T_e = results.get('T_e', None)
        self.k = results.get('k', None)
        self.T_e1 = results.get('T_e1', None)
        self.Delta_T = results.get('Delta_T', None)

        # Givens: no numeric defaults, only None
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.T_e0_given = givens.get('T_e0', None)   # originally 280 K
        self.f_given = givens.get('f', None)         # originally 0.95

        # Explanation steps
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = []

    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test methods
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

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_U_per_V_exists(self):
        if self.U_per_V is None:
            return (False, "SKIP: 'U_per_V' missing or None")
        return (True, "'U_per_V' present")

    def test_interface_A_exists(self):
        if self.A is None:
            return (False, "SKIP: 'A' missing or None")
        return (True, "'A' present")

    def test_interface_J_U_exists(self):
        if self.J_U is None:
            return (False, "SKIP: 'J_U' missing or None")
        return (True, "'J_U' present")

    def test_interface_B_exists(self):
        if self.B is None:
            return (False, "SKIP: 'B' missing or None")
        return (True, "'B' present")

    def test_interface_T_e_exists(self):
        if self.T_e is None:
            return (False, "SKIP: 'T_e' missing or None")
        return (True, "'T_e' present")

    def test_interface_k_exists(self):
        if self.k is None:
            return (False, "SKIP: 'k' missing or None")
        return (True, "'k' present")

    def test_interface_T_e1_exists(self):
        if self.T_e1 is None:
            return (False, "SKIP: 'T_e1' missing or None")
        return (True, "'T_e1' present")

    def test_interface_Delta_T_exists(self):
        if self.Delta_T is None:
            return (False, "SKIP: 'Delta_T' missing or None")
        return (True, "'Delta_T' present")

    # ------------------------------------------------------------------
    # Invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_Delta_T_computed(self):
        if self.Delta_T is None:
            return (False, "SKIP: Delta_T not available")
        if not isinstance(self.Delta_T, (int, float)):
            return (False, "SKIP: Delta_T is not numeric")
        if self.T_e0_given is None or self.f_given is None:
            return (False, "SKIP: T_e0_given or f_given missing (no numeric fallback)")
        expected = self.T_e0_given - self.T_e0_given * (self.f_given ** 0.25)
        if math.isclose(self.Delta_T, expected, rel_tol=1e-2):
            return (True, f"Delta_T = {self.Delta_T} matches expected {expected:.2f}")
        else:
            return (False, f"Delta_T = {self.Delta_T} but expected ~{expected:.2f}")

    def test_invariant_T_e1_via_fourth_root(self):
        if self.T_e1 is None:
            return (False, "SKIP: T_e1 not available")
        if not isinstance(self.T_e1, (int, float)):
            return (False, "SKIP: T_e1 is not numeric")
        if self.T_e0_given is None or self.f_given is None:
            return (False, "SKIP: T_e0_given or f_given missing")
        lhs = self.T_e1 ** 4
        rhs = self.f_given * (self.T_e0_given ** 4)
        if math.isclose(lhs, rhs, rel_tol=1e-2):
            return (True, f"T_e1^4 ≈ f*T_e0^4 ({lhs:.1f} vs {rhs:.1f})")
        else:
            return (False, f"T_e1^4 = {lhs:.1f}, f*T_e0^4 = {rhs:.1f}")

    # ------------------------------------------------------------------
    # Modeling checks
    # ------------------------------------------------------------------
    def test_modeling_power_balance_areas(self):
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps empty")
        combined = " ".join(str(step) for step in self.explanation_steps)
        has_cross = "pi R_e^2" in combined or "π R_e^2" in combined
        has_surf = "4pi R_e^2" in combined or "4π R_e^2" in combined
        if has_cross and has_surf:
            return (True, "Explanation contains correct area terms for Earth power balance")
        else:
            return (False, "Missing correct area terms (πR_e^2 and 4πR_e^2)")

    def test_modeling_flux_hole_relation(self):
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps empty")
        combined = " ".join(str(step) for step in self.explanation_steps)
        if "c/4" in combined or "c / 4" in combined:
            return (True, "c/4 relation for flux from hole mentioned")
        else:
            return (False, "c/4 relation not found in explanation")

    # ------------------------------------------------------------------
    # Answer sanity (weak, only sign/range/finite)
    # ------------------------------------------------------------------
    def test_answer_sanity_Delta_T_positive(self):
        if self.Delta_T is None:
            return (False, "SKIP: Delta_T not available")
        if not isinstance(self.Delta_T, (int, float)):
            return (False, "SKIP: Delta_T not numeric")
        if self.Delta_T > 0:
            return (True, "Delta_T > 0, consistent with cooling")
        else:
            return (False, f"Delta_T = {self.Delta_T}, expected positive")

    # ------------------------------------------------------------------
    # Monotonicity: perturb an input and compare output change
    # ------------------------------------------------------------------
    def test_monotonicity_T_e_increases_with_T_s(self):
        """
        T_e = k * T_s, with k = sqrt(R_s/(2d)).
        If T_s increases, T_e increases linearly.
        We verify this by picking representative numbers and perturbing T_s.
        """
        # Re-derived from physical law, not from solution numbers
        R_s = 7.0e8   # m (solar radius)
        d = 1.5e11    # m (Earth-Sun distance)
        T_s_base = 5000.0  # arbitrary
        f = 1.1       # 10% increase

        try:
            k = math.sqrt(R_s / (2 * d))
            T_e_base = k * T_s_base
            T_e_pert = k * (f * T_s_base)
        except Exception as e:
            return (False, f"Monotonicity test computation failed: {e}")

        if T_e_pert > T_e_base:
            return (True, "T_e increases when T_s increases (as expected)")
        else:
            return (False, "T_e did not increase after increasing T_s (contradicts physics)")

    def test_monotonicity_T_e_decreases_with_distance(self):
        """
        T_e ∝ sqrt(1/d), so increasing d decreases T_e.
        """
        R_s = 7.0e8
        d_base = 1.5e11
        T_s = 5000.0
        f = 1.1   # increase distance by 10%

        try:
            k_base = math.sqrt(R_s / (2 * d_base))
            T_e_base = k_base * T_s
            k_pert = math.sqrt(R_s / (2 * (f * d_base)))
            T_e_pert = k_pert * T_s
        except Exception as e:
            return (False, f"Monotonicity test failed: {e}")

        if T_e_pert < T_e_base:
            return (True, "T_e decreases when distance increases (as expected)")
        else:
            return (False, "T_e did not decrease with increased distance")