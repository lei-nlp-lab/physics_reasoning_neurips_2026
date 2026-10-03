import math
import traceback

class PhysicsChecker:
    """
    Auditor for the Carnot refrigerator problem.
    
    Problem: An ideal Carnot refrigerator freezes ice at 5 g/s, 
    with L_f = 320 J/g, Tc = 273 K, Th = 303 K. Expected results: 
    Qh_dot (J/s), W_dot (J/s), W_kW (kW), COP.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}  # ensure it's a dict for key checking

        # Problem constants (ground truth, not from solution)
        self.m_dot = 5.0       # g/s
        self.L_f = 320.0       # J/g
        self.Tc = 273.0        # K
        self.Th = 303.0        # K

        # Derived constant that appears in many checks
        self.Qc_dot = self.m_dot * self.L_f   # 1600 J/s

    # ------------------------------------------------------------
    #  Helper: safe extraction of a single result
    # ------------------------------------------------------------
    def _get(self, key):
        """Return result value or None if missing."""
        if not isinstance(self.results, dict):
            return None
        return self.results.get(key, None)

    # ------------------------------------------------------------
    #  Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Ensure the results dict contains all four required keys."""
        if not isinstance(self.results, dict):
            return False, "results is not a dict"
        needed = ['Qh_dot', 'W_dot', 'W_kW', 'COP']
        missing = [k for k in needed if k not in self.results]
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present"

    # ------------------------------------------------------------
    #  Invariant tests (physics laws, use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_carnot_relation(self):
        """Check Qh_dot/Th ≈ Qc_dot/Tc (reversible Carnot)."""
        Qh = self._get('Qh_dot')
        if Qh is None:
            return False, "Qh_dot not available"
        left = Qh / self.Th
        right = self.Qc_dot / self.Tc
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Carnot relation holds: {left:.4f} ≈ {right:.4f}"
        else:
            return False, f"Carnot relation violated: left={left:.4f}, right={right:.4f}"

    def test_invariant_energy_conservation(self):
        """Check W_dot = Qh_dot - Qc_dot (first law)."""
        Qh = self._get('Qh_dot')
        W = self._get('W_dot')
        if Qh is None or W is None:
            return False, "Qh_dot or W_dot missing"
        expected = Qh - self.Qc_dot
        if math.isclose(W, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Energy balance: W_dot = {W:.4f} ≈ {expected:.4f}"
        else:
            return False, f"Energy balance violation: W_dot={W:.4f}, expected {expected:.4f}"

    def test_invariant_cop_definition(self):
        """Check COP = Qc_dot / W_dot (refrigerator COP)."""
        W = self._get('W_dot')
        cop = self._get('COP')
        if W is None or cop is None:
            return False, "W_dot or COP missing"
        if W == 0:
            return False, "W_dot is zero, COP undefined"
        expected = self.Qc_dot / W
        if math.isclose(cop, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"COP = {cop:.4f} ≈ {expected:.4f}"
        else:
            return False, f"COP mismatch: {cop:.4f} vs expected {expected:.4f}"

    def test_invariant_cop_carnot(self):
        """Check COP ≈ Tc/(Th-Tc) (Carnot refrigerator ideal)."""
        cop = self._get('COP')
        if cop is None:
            return False, "COP not available"
        ideal = self.Tc / (self.Th - self.Tc)   # 273/30 = 9.1
        if math.isclose(cop, ideal, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"COP matches Carnot ideal: {cop:.4f} ≈ {ideal:.4f}"
        else:
            return False, f"COP {cop:.4f} deviates from Carnot ideal {ideal:.4f}"

    # ------------------------------------------------------------
    #  Modeling / sanity tests (no isclose, but weak)
    # ------------------------------------------------------------
    def test_modeling_positive_Qh_dot(self):
        """Qh_dot must be positive."""
        Qh = self._get('Qh_dot')
        if Qh is None:
            return False, "Qh_dot missing"
        if Qh > 0:
            return True, f"Qh_dot = {Qh:.2f} > 0"
        else:
            return False, f"Qh_dot should be positive, got {Qh:.2f}"

    def test_modeling_positive_W_dot(self):
        """W_dot must be positive (engine needs work input)."""
        W = self._get('W_dot')
        if W is None:
            return False, "W_dot missing"
        if W > 0:
            return True, f"W_dot = {W:.2f} > 0"
        else:
            return False, f"W_dot should be positive, got {W:.2f}"

    def test_modeling_conversion_to_kW(self):
        """W_kW should be W_dot / 1000 within 1%."""
        W = self._get('W_dot')
        W_kW = self._get('W_kW')
        if W is None or W_kW is None:
            return False, "W_dot or W_kW missing"
        expected = W / 1000.0
        if math.isclose(W_kW, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"W_kW = {W_kW:.6f} ≈ expected {expected:.6f}"
        else:
            return False, f"W_kW = {W_kW:.6f}, expected {expected:.6f}"

    # ------------------------------------------------------------
    #  Monotonicity test (perturb Th, check COP direction)
    # ------------------------------------------------------------
    def test_monotonicity_cop_decreases_with_increasing_Th(self):
        """
        For fixed Tc, as Th increases, Carnot COP = Tc/(Th-Tc) decreases.
        Perturb Th upward by 1 K; predicted COP should be lower than baseline COP.
        """
        cop = self._get('COP')
        if cop is None:
            return False, "COP not available"
        # Baseline Th = 303, Th_pert = 304
        Th_pert = self.Th + 1.0
        cop_pert = self.Tc / (Th_pert - self.Tc)   # 273/31 ≈ 8.8065
        # Direction test: perturbed COP should be less than baseline
        if cop_pert < cop - 1e-2:   # allow small tolerance
            return True, f"COP decreases from {cop:.4f} to predicted {cop_pert:.4f} when Th increases by 1 K"
        else:
            return False, f"COP did not decrease: baseline {cop:.4f}, predicted {cop_pert:.4f} (should be smaller)"

    # ------------------------------------------------------------
    #  Weak answer sanity (broad ranges only, no isclose)
    # ------------------------------------------------------------
    def test_answer_sanity_broad_ranges(self):
        """
        Weak sanity: all quantities should be finite, positive, and within
        broad physically plausible ranges.
        """
        Qh = self._get('Qh_dot')
        W = self._get('W_dot')
        W_kW = self._get('W_kW')
        cop = self._get('COP')

        # Check existence and finiteness
        for name, val in [('Qh_dot', Qh), ('W_dot', W), ('W_kW', W_kW), ('COP', cop)]:
            if val is None:
                return False, f"{name} is missing"
            if math.isnan(val) or math.isinf(val):
                return False, f"{name} is NaN or Inf"

        # Broad magnitude checks (not matching exact answer)
        if not (1000 < Qh < 3000):
            return False, f"Qh_dot {Qh} not in plausible range (1000,3000)"
        if not (100 < W < 500):
            return False, f"W_dot {W} not in plausible range (100,500)"
        if not (0.1 < W_kW < 0.5):
            return False, f"W_kW {W_kW} not in plausible range (0.1,0.5)"
        if not (5 < cop < 15):
            return False, f"COP {cop} not in plausible range (5,15)"
        return True, "All quantities within broad plausible ranges"

    # ------------------------------------------------------------
    #  Runner: discovers all test_* methods and executes them safely
    # ------------------------------------------------------------
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