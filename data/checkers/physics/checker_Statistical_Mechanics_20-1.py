import math

class PhysicsChecker:
    """
    Audits solutions for the thermal vs gravitational energy extraction problem.
    """

    def __init__(self, solution_instance):
        # ------------------------------------------------------------------
        # Extract result quantities with safe defaults (None)
        # ------------------------------------------------------------------
        results = getattr(solution_instance, 'results', None) or {}
        self.W_th = results.get('W_th', None)   # thermal extraction work
        self.E_g  = results.get('E_g', None)    # gravitational potential work

        # ------------------------------------------------------------------
        # Input parameters (use same‑pattern safe extraction)
        # ------------------------------------------------------------------
        givens = getattr(solution_instance, 'givens', None) or {}
        self.m        = givens.get('m', None)
        self.c_V      = givens.get('c_V', None)
        self.delta_T  = givens.get('delta_T', None)
        self.h        = givens.get('h', None)
        self.g        = givens.get('g', None)
        self.T_h      = givens.get('T_h', None)
        self.T_c      = givens.get('T_c', None)

        # ------------------------------------------------------------------
        # Meta‑information (strings, not numerical)
        # ------------------------------------------------------------------
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.solving_pseudocode = getattr(solution_instance, 'solving_pseudocode', None)

    # ------------------------------------------------------------------
    # Helper: make safe numeric check
    # ------------------------------------------------------------------
    @staticmethod
    def _is_close(a, b, rel_tol=0.01, abs_tol=0.01):
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    # ==================================================================
    # INTERFACE TESTS – required keys and types
    # ==================================================================
    def test_interface_results_keys(self):
        """Both 'W_th' and 'E_g' must be present and numeric."""
        if self.W_th is None:
            return (False, "SKIP: 'W_th' not found in results")
        if self.E_g is None:
            return (False, "SKIP: 'E_g' not found in results")
        if not isinstance(self.W_th, (int, float)):
            return (False, "'W_th' is not numeric")
        if not isinstance(self.E_g, (int, float)):
            return (False, "'E_g' is not numeric")
        return (True, "Both required keys present and numeric")

    # ==================================================================
    # INVARIANT TESTS – theorem‑based physical sanity (use math.isclose)
    # ==================================================================
    def test_invariant_positivity(self):
        """Both energies must be positive."""
        if self.W_th is None or self.E_g is None:
            return (False, "SKIP: one or both energies missing")
        if self.W_th <= 0:
            return (False, f"W_th = {self.W_th} is not positive")
        if self.E_g <= 0:
            return (False, f"E_g = {self.E_g} is not positive")
        return (True, "Both energies positive")

    def test_invariant_energy_relation(self):
        """
        Check internal consistency: W_th = eta * Q, with Q = m c_V ΔT,
        eta = 1 - T_c/T_h.  (Uses extracted parameters if available.)
        """
        # Compute from parameters if present; otherwise skip
        if any(x is None for x in [self.m, self.c_V, self.delta_T, self.T_h, self.T_c]):
            return (False, "SKIP: insufficient thermodynamic givens")
        Q = self.m * self.c_V * self.delta_T
        eta = 1 - self.T_c / self.T_h
        expected_W_th = eta * Q
        if self.W_th is None:
            return (False, "W_th missing but givens present")
        if not self._is_close(self.W_th, expected_W_th):
            return (False, f"W_th = {self.W_th} does not match expected {expected_W_th}")
        return (True, "W_th consistent with thermodynamic relation")

    def test_invariant_gravitational_energy(self):
        """E_g must be m * g * h."""
        if any(x is None for x in [self.m, self.g, self.h]):
            return (False, "SKIP: missing gravitational parameters")
        expected_Eg = self.m * self.g * self.h
        if self.E_g is None:
            return (False, "E_g missing but givens present")
        if not self._is_close(self.E_g, expected_Eg):
            return (False, f"E_g = {self.E_g} does not match expected {expected_Eg}")
        return (True, "E_g consistent with gravitational PE")

    def test_invariant_carnot_efficiency_range(self):
        """Efficiency η = 1 - T_c/T_h must be between 0 and 1."""
        if self.T_h is None or self.T_c is None:
            return (False, "SKIP: missing temperatures")
        if self.T_h <= 0 or self.T_c <= 0:
            return (False, "Temperatures must be positive")
        if self.T_c >= self.T_h:
            return (False, "T_c must be < T_h")
        eta = 1 - self.T_c / self.T_h
        if not (0 < eta < 1):
            return (False, f"η = {eta} out of range (0,1)")
        return (True, f"η = {eta} is within (0,1)")

    def test_invariant_comparison(self):
        """
        For the canonical givens (ΔT=10 K, h=110 m, T_h≈293 K) thermal
        work should exceed gravitational work.
        """
        if self.W_th is None or self.E_g is None:
            return (False, "SKIP: missing energies for comparison")
        if self.W_th <= self.E_g:
            return (False, "W_th should be > E_g (for usual parameters)")
        return (True, "W_th > E_g as expected")

    # ==================================================================
    # MODELING / TRACE TESTS – check explanation & pseudocode
    # ==================================================================
    def test_modeling_explanation_formulas(self):
        """Explanation steps must contain key formulas: Carnot efficiency,
        Q = m c_v ΔT, W_th = η Q, W' = mgh."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = self.explanation_steps.lower()
        checks = [
            "carnot",
            "efficiency",
            "q = m c",
            "w = eta q",
            "w' = mgh",
        ]
        missing = [c for c in checks if c not in text]
        if missing:
            return (False, f"Missing key phrases: {missing}")
        return (True, "Explanation contains required formulas")

    def test_modeling_pseudocode_steps(self):
        """Pseudocode must compute Q, η, W_th, and E_g explicitly."""
        if self.solving_pseudocode is None:
            return (False, "SKIP: no solving_pseudocode provided")
        text = self.solving_pseudocode.lower()
        checks = [
            "q = m * c_v * delta_t",
            "eta = 1 - t_c/t_h",
            "w_th = eta * q",
            "e_g = m * g * h",
        ]
        missing = [c for c in checks if c not in text]
        if missing:
            return (False, f"Missing pseudocode steps: {missing}")
        return (True, "Pseudocode contains all required computation steps")

    # ==================================================================
    # MONOTONICITY TESTS – perturb a parameter and verify trend
    # ==================================================================
    def test_monotonicity_W_th_with_delta_T(self):
        """
        W_th = m c_v (ΔT) * (1 - (T_h-ΔT)/T_h) = m c_v ΔT² / T_h.
        A 10 % increase in ΔT must increase W_th (quadratic).
        """
        if any(x is None for x in [self.m, self.c_V, self.delta_T, self.T_h]):
            return (False, "SKIP: missing parameters for monotonicity test")
        T_h = self.T_h
        delta_T = self.delta_T
        # baseline
        Q0 = self.m * self.c_V * delta_T
        eta0 = 1 - (T_h - delta_T) / T_h
        baseline = eta0 * Q0
        # perturb ΔT +10%
        new_delta = delta_T * 1.1
        Q1 = self.m * self.c_V * new_delta
        eta1 = 1 - (T_h - new_delta) / T_h
        perturbed = eta1 * Q1
        if perturbed <= baseline:
            return (False, f"Perturbed W_th = {perturbed} not > baseline {baseline}")
        return (True, "W_th increases with ΔT (quadratic)")

    def test_monotonicity_E_g_with_height(self):
        """E_g = m g h; a 10 % increase in h must increase E_g."""
        if any(x is None for x in [self.m, self.g, self.h]):
            return (False, "SKIP: missing gravitational givens")
        baseline = self.m * self.g * self.h
        new_h = self.h * 1.1
        perturbed = self.m * self.g * new_h
        if perturbed <= baseline:
            return (False, f"Perturbed E_g = {perturbed} not > baseline {baseline}")
        return (True, "E_g increases with height")

    # ==================================================================
    # WEAK ANSWER SANITY (only one, non‑dominant)
    # ==================================================================
    def test_answer_sanity_magnitude(self):
        """
        Both energies must be within [0.1 J, 10 J] for physically
        reasonable parameters (1 g water, 110 m height, 10 K ΔT).
        """
        if self.W_th is None or self.E_g is None:
            return (False, "SKIP: one or both energies missing")
        if not (0.1 <= self.W_th <= 10.0):
            return (False, f"W_th = {self.W_th} out of plausible range [0.1, 10.0] J")
        if not (0.1 <= self.E_g <= 10.0):
            return (False, f"E_g = {self.E_g} out of plausible range [0.1, 10.0] J")
        return (True, "Energies within expected magnitude")
    # ==================================================================
    # RUN ALL TESTS
    # ==================================================================
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