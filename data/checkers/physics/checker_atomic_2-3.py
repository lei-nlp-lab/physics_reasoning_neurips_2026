import math
import sympy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Fundamental constants (if provided as givens)
        self.h = self.givens.get('h', None)
        self.d_n = self.givens.get('d_n', None)

        # Results from pseudocode emit contract
        self.p_min = self.results.get('p_min', None)
        self.advantage_e = self.results.get('advantage_e', None)
        self.disadvantage_e = self.results.get('disadvantage_e', None)
        self.advantage_p = self.results.get('advantage_p', None)
        self.disadvantage_p = self.results.get('disadvantage_p', None)
        self.advantage_n = self.results.get('advantage_n', None)
        self.disadvantage_n = self.results.get('disadvantage_n', None)
        self.opinion_photon = self.results.get('opinion_photon', None)

    # ------------------------------------------------------------------
    # Interface checks: presence and type of required keys
    # ------------------------------------------------------------------
    def test_interface_p_min_present(self):
        if self.p_min is None:
            return (False, "p_min missing from results")
        if not isinstance(self.p_min, (int, float)):
            return (False, f"p_min should be numeric, got {type(self.p_min)}")
        return (True, "p_min present and numeric")

    def test_interface_p_min_positive(self):
        if self.p_min is None:
            return (False, "SKIP: p_min not available")
        if self.p_min <= 0:
            return (False, f"p_min must be positive, got {self.p_min}")
        return (True, f"p_min positive: {self.p_min}")

    def test_interface_advantage_e_present(self):
        if self.advantage_e is None:
            return (False, "advantage_e missing")
        if not isinstance(self.advantage_e, str):
            return (False, f"advantage_e should be string, got {type(self.advantage_e)}")
        return (True, "advantage_e present and string")

    def test_interface_disadvantage_e_present(self):
        if self.disadvantage_e is None:
            return (False, "disadvantage_e missing")
        if not isinstance(self.disadvantage_e, str):
            return (False, f"disadvantage_e should be string, got {type(self.disadvantage_e)}")
        return (True, "disadvantage_e present and string")

    def test_interface_advantage_p_present(self):
        if self.advantage_p is None:
            return (False, "advantage_p missing")
        if not isinstance(self.advantage_p, str):
            return (False, f"advantage_p should be string, got {type(self.advantage_p)}")
        return (True, "advantage_p present and string")

    def test_interface_disadvantage_p_present(self):
        if self.disadvantage_p is None:
            return (False, "disadvantage_p missing")
        if not isinstance(self.disadvantage_p, str):
            return (False, f"disadvantage_p should be string, got {type(self.disadvantage_p)}")
        return (True, "disadvantage_p present and string")

    def test_interface_advantage_n_present(self):
        if self.advantage_n is None:
            return (False, "advantage_n missing")
        if not isinstance(self.advantage_n, str):
            return (False, f"advantage_n should be string, got {type(self.advantage_n)}")
        return (True, "advantage_n present and string")

    def test_interface_disadvantage_n_present(self):
        if self.disadvantage_n is None:
            return (False, "disadvantage_n missing")
        if not isinstance(self.disadvantage_n, str):
            return (False, f"disadvantage_n should be string, got {type(self.disadvantage_n)}")
        return (True, "disadvantage_n present and string")

    def test_interface_opinion_photon_present(self):
        if self.opinion_photon is None:
            return (False, "opinion_photon missing")
        if not isinstance(self.opinion_photon, str):
            return (False, f"opinion_photon should be string, got {type(self.opinion_photon)}")
        return (True, "opinion_photon present and string")

    # ------------------------------------------------------------------
    # Invariant checks : physics relations (fuzzy numerical)
    # ------------------------------------------------------------------
    def test_invariant_de_broglie_relation(self):
        """Check p_min ≈ h / d_n (de Broglie wavelength equal to nuclear size)."""
        if self.p_min is None or self.h is None or self.d_n is None:
            return (False, "SKIP: missing p_min, h, or d_n")
        expected_p = self.h / self.d_n
        if math.isclose(self.p_min, expected_p, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_min ≈ h/d_n: {self.p_min} vs {expected_p}")
        else:
            return (False, f"p_min deviates from h/d_n: got {self.p_min}, expected ~{expected_p}")

    def test_invariant_resolution_condition(self):
        """Check that λ = h/p ≤ d_n is satisfied."""
        if self.p_min is None or self.h is None or self.d_n is None:
            return (False, "SKIP: missing p_min, h, or d_n")
        wavelength = self.h / self.p_min
        if wavelength <= self.d_n * (1 + 1e-2):  # allow 1% tolerance due to numeric rounding
            return (True, f"λ = {wavelength} ≤ d_n = {self.d_n}")
        else:
            return (False, f"λ = {wavelength} > d_n = {self.d_n} (resolution not satisfied)")

    # ------------------------------------------------------------------
    # Modeling / trace checks : qualitative statements consistency
    # ------------------------------------------------------------------
    def test_modeling_advantage_e_content(self):
        """Check that advantage_e mentions electromagnetic interaction."""
        if self.advantage_e is None:
            return (False, "SKIP: advantage_e not available")
        keywords = ['electromagnetic', 'easy analysis', 'point-like']
        for kw in keywords:
            if kw.lower() in self.advantage_e.lower():
                return (True, f"advantage_e contains '{kw}'")
        return (False, f"advantage_e missing expected keywords: {self.advantage_e}")

    def test_modeling_disadvantage_e_content(self):
        """Check that disadvantage_e mentions high energy or beam."""
        if self.disadvantage_e is None:
            return (False, "SKIP: disadvantage_e not available")
        keywords = ['high energy', 'energy', 'beam', 'Coulomb']
        for kw in keywords:
            if kw.lower() in self.disadvantage_e.lower():
                return (True, f"disadvantage_e contains '{kw}'")
        return (False, f"disadvantage_e missing expected keywords: {self.disadvantage_e}")

    def test_modeling_advantage_p_content(self):
        """Check that advantage_p mentions high flux or beam availability."""
        if self.advantage_p is None:
            return (False, "SKIP: advantage_p not available")
        keywords = ['high flux', 'beam', 'proton', 'available']
        for kw in keywords:
            if kw.lower() in self.advantage_p.lower():
                return (True, f"advantage_p contains '{kw}'")
        return (False, f"advantage_p missing expected keywords: {self.advantage_p}")

    def test_modeling_disadvantage_p_content(self):
        """Check that disadvantage_p mentions complex analysis or multiple interactions."""
        if self.disadvantage_p is None:
            return (False, "SKIP: disadvantage_p not available")
        keywords = ['complex', 'strong interaction', 'both', 'electromagnetic']
        for kw in keywords:
            if kw.lower() in self.disadvantage_p.lower():
                return (True, f"disadvantage_p contains '{kw}'")
        return (False, f"disadvantage_p missing expected keywords: {self.disadvantage_p}")

    def test_modeling_advantage_n_content(self):
        """Check that advantage_n mentions lack of charge or neat."""
        if self.advantage_n is None:
            return (False, "SKIP: advantage_n not available")
        keywords = ['neat', 'no charge', 'lack of charge', 'strong force only']
        for kw in keywords:
            if kw.lower() in self.advantage_n.lower():
                return (True, f"advantage_n contains '{kw}'")
        return (False, f"advantage_n missing expected keywords: {self.advantage_n}")

    def test_modeling_disadvantage_n_content(self):
        """Check that disadvantage_n mentions difficulty in generation or detection."""
        if self.disadvantage_n is None:
            return (False, "SKIP: disadvantage_n not available")
        keywords = ['difficulty', 'generation', 'detection', 'low cross']
        for kw in keywords:
            if kw.lower() in self.disadvantage_n.lower():
                return (True, f"disadvantage_n contains '{kw}'")
        return (False, f"disadvantage_n missing expected keywords: {self.disadvantage_n}")

    def test_modeling_opinion_photon_content(self):
        """Check that opinion_photon mentions hadron-like or complicated."""
        if self.opinion_photon is None:
            return (False, "SKIP: opinion_photon not available")
        keywords = ['hadron', 'complicated', 'high energy', 'photon']
        for kw in keywords:
            if kw.lower() in self.opinion_photon.lower():
                return (True, f"opinion_photon contains '{kw}'")
        return (False, f"opinion_photon missing expected keywords: {self.opinion_photon}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_p_min_magnitude(self):
        """Check that p_min is in a physically plausible range (10^-20 to 10^-17 kg·m/s)."""
        if self.p_min is None:
            return (False, "SKIP: p_min not available")
        if 1e-20 < self.p_min < 1e-17:
            return (True, f"p_min magnitude plausible: {self.p_min}")
        else:
            return (False, f"p_min magnitude unlikely: {self.p_min} (expected ~10^-19)")

    # ------------------------------------------------------------------
    # Monotonicity test (one example)
    # ------------------------------------------------------------------
    def test_monotonicity_increase_dn_decreases_p_min(self):
        """If nuclear size increases, required minimum momentum should decrease (λ fixed)."""
        if self.p_min is None or self.h is None or self.d_n is None:
            return (False, "SKIP: missing p_min, h, or d_n")
        # Increase d_n by 10% and compute expected new p_min
        d_n_perturbed = self.d_n * 1.1
        expected_p_new = self.h / d_n_perturbed
        # The original p_min should be larger than the new expected value (since inverse relation)
        if self.p_min > expected_p_new:
            return (True, f"Monotonicity holds: original p_min={self.p_min} > new p_min~{expected_p_new} after d_n increase")
        else:
            return (False, f"Monotonicity violated: original p_min={self.p_min} <= new p_min~{expected_p_new} after d_n increase")

    # ------------------------------------------------------------------
    # Main test runner
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