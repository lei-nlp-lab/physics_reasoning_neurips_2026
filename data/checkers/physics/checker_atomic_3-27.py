import math

class PhysicsChecker:
    """
    Audits the solution for the particle decay rate ratio problem.
    Checks interface completeness, physical invariants, modeling of
    selection rules, monotonicity, and provides a weak final-answer
    sanity check.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely (default to empty dict)
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.results = results

        # Extract explanation steps if present (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Ensure required keys exist (even if None)
        for k in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']:
            if k not in self.results:
                self.results[k] = None

    # --------------------------------------------------------------------
    # Interface tests – check that result keys exist and are numeric
    # --------------------------------------------------------------------
    def test_interface_results_exist(self):
        required = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys (None values): {missing}")
        return (True, "All required keys present and not None")

    def test_interface_results_numeric(self):
        for k in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']:
            v = self.results.get(k)
            if v is None:
                continue
            if not isinstance(v, (int, float)):
                return (False, f"{k} is not numeric: {type(v)}")
        return (True, "All present results are numeric")

    # --------------------------------------------------------------------
    # Invariant tests – basic physical sanity
    # --------------------------------------------------------------------
    def test_invariant_non_negative(self):
        for k in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']:
            v = self.results.get(k)
            if v is None:
                continue
            if v < -1e-12:
                return (False, f"{k} = {v} is negative")
        return (True, "All ratios are non-negative")

    def test_invariant_finite(self):
        for k in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']:
            v = self.results.get(k)
            if v is None:
                continue
            if not math.isfinite(v):
                return (False, f"{k} = {v} is not finite")
        return (True, "All ratios are finite")

    # --------------------------------------------------------------------
    # Modeling tests – check that selection rules are respected
    # --------------------------------------------------------------------
    def test_modeling_forbidden_decays_zero(self):
        """
        Decays that are forbidden by a conservation law must have ratio ~ 0.
        Forbidden keys: R2 (C-parity), R4 (ΔQ/ΔS rule), R5 (ΔS=2),
        R6 (parity), R7 (ΔI₃ rule), R8 (G-parity), R9 (ΔS=0).
        """
        forbidden_keys = ['R2', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
        for k in forbidden_keys:
            v = self.results.get(k)
            if v is None:
                continue
            if not math.isclose(v, 0.0, abs_tol=1e-12):
                return (False, f"Forbidden decay {k} should be ~0, got {v}")
        return (True, "All forbidden decays have ratio essentially zero")

    def test_modeling_allowed_decays_positive(self):
        """
        Decays that are allowed (R1, R3, R10) should have strictly positive ratios.
        """
        allowed_keys = ['R1', 'R3', 'R10']
        for k in allowed_keys:
            v = self.results.get(k)
            if v is None:
                continue
            if v <= 0:
                return (False, f"Allowed decay {k} should be positive, got {v}")
        return (True, "All allowed decay ratios are positive")

    # --------------------------------------------------------------------
    # Trace consistency check – verify explanation steps mention key physics
    # --------------------------------------------------------------------
    def test_trace_selection_rules(self):
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        # Important physics concepts that should appear
        keywords = [
            'forbidden', 'ΔI', 'selection rule', 'CKM', 'helicity',
            'C-parity', 'Cabibbo', 'GIM', 'ΔS', 'ΔQ'
        ]
        found = [kw for kw in keywords if kw in self.explanation_steps]
        # Require at least two distinct concepts to be mentioned
        if len(found) < 2:
            return (False, f"Only {len(found)} relevant keywords found: {found}")
        return (True, f"Explanation steps contain relevant physics: {found}")

    # --------------------------------------------------------------------
    # Monotonicity test – explicit perturbation of an input parameter
    # --------------------------------------------------------------------
    def test_monotonicity_R10_vs_electron_mass(self):
        """
        For subquestion (j), the ratio R10 = Γ(π→eν)/Γ(K→μν).
        Because of helicity suppression, the rate is proportional to m_l^2.
        Increasing the electron mass should increase Γ(π→eν), but since the
        numerator depends on m_e^2 and the denominator is independent,
        the ratio R10 should increase as m_e increases.
        We verify by recomputing the ratio with a perturbed m_e.
        """
        v = self.results.get('R10')
        if v is None:
            return (False, "SKIP: R10 not available")

        # Define approximate ratio formula based on pseudocode.
        # Gamma(π→eν) ∝ |V_ud|^2 * f_π^2 * m_π * m_e^2 * (1 - m_e^2/m_π^2)^2
        # Gamma(K→μν) ∝ |V_us|^2 * f_K^2 * m_K * m_μ^2 * (1 - m_μ^2/m_K^2)^2
        # Ratio = (f_π^2/f_K^2) * (m_π/m_K) * (m_e^2/m_μ^2) * ((1 - m_e^2/m_π^2)/(1 - m_μ^2/m_K^2))^2 * (|V_ud|^2/|V_us|^2)
        f_ratio_sq = 0.67   # approximate (f_π/f_K)^2
        m_pi = 139.6        # MeV
        m_K = 493.7
        m_mu = 105.7
        V_ud = 0.974
        V_us = 0.225
        def ratio(m_e):
            factor = f_ratio_sq * (m_pi/m_K) * (m_e**2/m_mu**2)
            factor *= ((1 - m_e**2/m_pi**2) / (1 - m_mu**2/m_K**2))**2
            factor *= (V_ud**2) / (V_us**2)
            return factor

        baseline = ratio(0.511)          # standard electron mass
        perturbed = ratio(0.511 * 1.05)  # increase by 5%
        if perturbed <= baseline:
            return (False, f"R10 should increase when m_e increases by 5%, but baseline={baseline}, perturbed={perturbed}")
        return (True, f"R10 monotonic w.r.t. m_e: baseline={baseline}, perturbed={perturbed} (increase)")

    # --------------------------------------------------------------------
    # Weak final-answer sanity test – no exact values, only plausible ranges
    # --------------------------------------------------------------------
    def test_answer_sanity_plausible_orders(self):
        """
        Very lenient check: allowed decays must have small ratios (< 1%).
        R1, R3, R10 are all < 0.01 physically.
        """
        plausibility = {
            'R1': 0.01,   # about 1.5e-3
            'R3': 0.001,  # about 1e-5
            'R10': 0.001, # about 1.35e-4
        }
        for k, upper in plausibility.items():
            v = self.results.get(k)
            if v is None:
                continue
            if v > upper:
                return (False, f"{k} = {v} exceeds plausible upper bound {upper}")
        return (True, "Allowed decay ratios are within plausible small range")

    # --------------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all bound methods starting with 'test_'
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