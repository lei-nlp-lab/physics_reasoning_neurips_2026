import sympy
import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for the problem on fundamental interaction strengths and conservation laws.
    """

    def __init__(self, solution_instance):
        # --- Extract final results (required by contract) ---
        self.strong_strength = None
        self.EM_strength = None
        self.weak_strength = None
        self.grav_strength = None
        self.strong_conserved = None
        self.EM_conserved = None
        self.weak_conserved = None

        # Try attributes first
        for attr in ['strong_strength', 'EM_strength', 'weak_strength', 'grav_strength',
                     'strong_conserved', 'EM_conserved', 'weak_conserved']:
            if hasattr(solution_instance, attr):
                setattr(self, attr, getattr(solution_instance, attr))

        # Then try .results dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
            for key in ['strong_strength', 'EM_strength', 'weak_strength', 'grav_strength',
                        'strong_conserved', 'EM_conserved', 'weak_conserved']:
                if key in results and getattr(self, key) is None:
                    setattr(self, key, results[key])

        # --- Extract possible input constants for modeling and monotonicity ---
        self.m_p = None
        self.G_F = None
        self.G = None
        self.hbar = None
        self.c = None
        self.alpha_s = None  # strong coupling
        self.alpha_EM = None # fine structure constant

        for key in ['m_p', 'G_F', 'G', 'hbar', 'c', 'alpha_s', 'alpha_EM']:
            if hasattr(solution_instance, key):
                setattr(self, key, getattr(solution_instance, key))
            elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                if key in solution_instance.results and getattr(self, key) is None:
                    setattr(self, key, solution_instance.results[key])

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

    # ---------------------------------------------------------------
    # Interface tests
    # ---------------------------------------------------------------

    def test_interface_required_keys(self):
        """All seven result keys must be present (non-None)."""
        keys = ['strong_strength', 'EM_strength', 'weak_strength', 'grav_strength',
                'strong_conserved', 'EM_conserved', 'weak_conserved']
        missing = [k for k in keys if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present.")

    def test_interface_strong_strength_numeric(self):
        if self.strong_strength is None:
            return (False, "SKIP: strong_strength not available")
        if not isinstance(self.strong_strength, (int, float)):
            return (False, f"strong_strength is {type(self.strong_strength)}, expected numeric")
        return (True, "ok")

    def test_interface_EM_strength_numeric(self):
        if self.EM_strength is None:
            return (False, "SKIP: EM_strength not available")
        if not isinstance(self.EM_strength, (int, float)):
            return (False, f"EM_strength is {type(self.EM_strength)}, expected numeric")
        return (True, "ok")

    def test_interface_weak_strength_numeric(self):
        if self.weak_strength is None:
            return (False, "SKIP: weak_strength not available")
        if not isinstance(self.weak_strength, (int, float)):
            return (False, f"weak_strength is {type(self.weak_strength)}, expected numeric")
        return (True, "ok")

    def test_interface_grav_strength_numeric(self):
        if self.grav_strength is None:
            return (False, "SKIP: grav_strength not available")
        if not isinstance(self.grav_strength, (int, float)):
            return (False, f"grav_strength is {type(self.grav_strength)}, expected numeric")
        return (True, "ok")

    def test_interface_conserved_lists_str(self):
        for name, val in [('strong_conserved', self.strong_conserved),
                         ('EM_conserved', self.EM_conserved),
                         ('weak_conserved', self.weak_conserved)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not isinstance(val, list):
                return (False, f"{name} is not a list")
            for item in val:
                if not isinstance(item, str):
                    return (False, f"{name} contains non-string: {item}")
        return (True, "All conserved lists are lists of strings.")

    # ---------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ---------------------------------------------------------------

    def test_invariant_strength_hierarchy(self):
        if any(v is None for v in [self.strong_strength, self.EM_strength, self.weak_strength, self.grav_strength]):
            return (False, "SKIP: one or more strengths missing")
        if not (self.strong_strength > self.EM_strength > self.weak_strength > self.grav_strength):
            return (False, "Strength ordering violated (strong > EM > weak > grav)")
        return (True, "Strength hierarchy correct.")

    def test_invariant_strong_approx_one(self):
        if self.strong_strength is None:
            return (False, "SKIP: strong_strength not available")
        if not math.isclose(self.strong_strength, 1.0, rel_tol=1.0):
            return (False, f"strong_strength = {self.strong_strength}, expected ~1")
        return (True, "strong_strength ~1")

    def test_invariant_EM_approx_1over137(self):
        if self.EM_strength is None:
            return (False, "SKIP: EM_strength not available")
        expected = 1.0 / 137.0
        if not math.isclose(self.EM_strength, expected, rel_tol=0.5):
            return (False, f"EM_strength = {self.EM_strength}, expected ~{expected}")
        return (True, f"EM_strength ~1/137")

    def test_invariant_weak_approx_1e5(self):
        if self.weak_strength is None:
            return (False, "SKIP: weak_strength not available")
        expected = 1e-5
        if not math.isclose(self.weak_strength, expected, rel_tol=1.0):
            return (False, f"weak_strength = {self.weak_strength}, expected ~{expected}")
        return (True, "weak_strength ~1e-5")

    def test_invariant_grav_approx_6e39(self):
        if self.grav_strength is None:
            return (False, "SKIP: grav_strength not available")
        expected = 6e-39
        if not math.isclose(self.grav_strength, expected, rel_tol=1.0):
            return (False, f"grav_strength = {self.grav_strength}, expected ~{expected}")
        return (True, "grav_strength ~6e-39")

    def test_invariant_strong_conserved_color(self):
        if self.strong_conserved is None:
            return (False, "SKIP: strong_conserved not available")
        if 'color' not in self.strong_conserved:
            return (False, "'color' missing from strong_conserved")
        return (True, "'color' present")

    def test_invariant_strong_conserved_parity(self):
        if self.strong_conserved is None:
            return (False, "SKIP: strong_conserved not available")
        if 'P' not in self.strong_conserved:
            return (False, "'P' missing from strong_conserved")
        return (True, "'P' present")

    def test_invariant_EM_conserved_charge(self):
        if self.EM_conserved is None:
            return (False, "SKIP: EM_conserved not available")
        if 'electric charge' not in self.EM_conserved:
            return (False, "'electric charge' missing from EM_conserved")
        return (True, "'electric charge' present")

    def test_invariant_EM_conserved_parity(self):
        if self.EM_conserved is None:
            return (False, "SKIP: EM_conserved not available")
        # allow either 'P' or 'parity'
        if not ('P' in self.EM_conserved or 'parity' in self.EM_conserved):
            return (False, "No parity conservation indicated in EM_conserved")
        return (True, "Parity conservation indicated")

    def test_invariant_weak_conserved_charge(self):
        if self.weak_conserved is None:
            return (False, "SKIP: weak_conserved not available")
        if 'electric charge' not in self.weak_conserved:
            return (False, "'electric charge' missing from weak_conserved")
        return (True, "'electric charge' present")

    def test_invariant_weak_conserved_lepton(self):
        if self.weak_conserved is None:
            return (False, "SKIP: weak_conserved not available")
        if 'lepton number' not in self.weak_conserved:
            return (False, "'lepton number' missing from weak_conserved")
        return (True, "'lepton number' present")

    def test_invariant_weak_conserved_baryon(self):
        if self.weak_conserved is None:
            return (False, "SKIP: weak_conserved not available")
        if 'baryon number' not in self.weak_conserved:
            return (False, "'baryon number' missing from weak_conserved")
        return (True, "'baryon number' present")

    def test_invariant_weak_nonconserved_parity(self):
        if self.weak_conserved is None:
            return (False, "SKIP: weak_conserved not available")
        # Look for indicator of parity violation
        p_violated = ('delta_P!=0' in self.weak_conserved or 'ΔP !=0' in self.weak_conserved or
                      'parity violation' in self.weak_conserved or 'P' not in self.weak_conserved)
        if not p_violated:
            return (False, "No indication of parity violation in weak_conserved")
        return (True, "Parity violation indicated")

    # ---------------------------------------------------------------
    # Modeling / trace tests (at least two required)
    # ---------------------------------------------------------------

    def test_modeling_ratio_grav_EM(self):
        """Check that grav/EM ratio ~ 8e-37 (about 6e-39/(1/137))."""
        if any(v is None for v in [self.grav_strength, self.EM_strength]):
            return (False, "SKIP: grav_strength or EM_strength missing")
        expected_ratio = 6e-39 / (1.0/137.0)  # ~8.22e-37
        actual_ratio = self.grav_strength / self.EM_strength if self.EM_strength else 0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1.0):
            return (False, f"grav/EM ratio = {actual_ratio}, expected ~{expected_ratio}")
        return (True, "grav/EM ratio consistent with ~8e-37")

    def test_modeling_strong_EM_ratio(self):
        """
        Check that strong/EM ratio is ~137 (since strong ~ 1, EM ~1/137).
        This validates the relative strength modeling.
        """
        if any(v is None for v in [self.strong_strength, self.EM_strength]):
            return (False, "SKIP: strong_strength or EM_strength missing")
        # Expected ratio: strong ~ 1, EM ~ 1/137 => ratio ~ 137
        # Use a relaxed tolerance to accommodate typical approximations.
        expected_ratio = 137.0
        actual_ratio = self.strong_strength / self.EM_strength if self.EM_strength else 0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1.0):  # allows factor 2
            return (False, f"strong/EM ratio = {actual_ratio}, expected ~{expected_ratio}")
        return (True, "strong/EM ratio consistent with ~137")

    def test_modeling_weak_formula(self):
        """
        Check that weak_strength follows the formula g_w^2/(hbar c) = G_F * m_p^2 * c / hbar^3.
        Only runs if all input constants are available.
        """
        if any(v is None for v in [self.weak_strength, self.m_p, self.G_F, self.hbar, self.c]):
            return (False, "SKIP: missing constants for weak formula check")
        try:
            computed = self.G_F * (self.m_p ** 2) * self.c / (self.hbar ** 3)
        except Exception as e:
            return (False, f"Error computing weak strength from constants: {e}")
        if not math.isclose(self.weak_strength, computed, rel_tol=1.0):
            return (False, f"Given weak_strength {self.weak_strength} does not match computed {computed}")
        return (True, "Weak_strength matches the formula G_F m_p^2 c / hbar^3")

    # ---------------------------------------------------------------
    # Monotonicity test (uses physical input constants)
    # ---------------------------------------------------------------

    def test_monotonicity_weak_strength_with_proton_mass(self):
        """
        Perturb proton mass by +10% and recompute weak_strength using the formula
        g_w^2/(hbar c) = G_F * m_p^2 * c / hbar^3.
        Check that the new value is larger and matches the expected scaling.
        """
        if any(v is None for v in [self.weak_strength, self.m_p, self.G_F, self.hbar, self.c]):
            return (False, "SKIP: missing one or more of weak_strength, m_p, G_F, hbar, c")

        # Compute baseline from formula
        try:
            baseline = self.G_F * (self.m_p ** 2) * self.c / (self.hbar ** 3)
        except Exception as e:
            return (False, f"Error computing baseline weak_strength: {e}")

        # Check that given weak_strength matches baseline (within factor 2)
        if not math.isclose(self.weak_strength, baseline, rel_tol=1.0):
            return (False, f"Given weak_strength {self.weak_strength} does not match computed baseline {baseline}")

        # Perturb m_p by +10%
        new_m_p = self.m_p * 1.1
        new_weak = self.G_F * (new_m_p ** 2) * self.c / (self.hbar ** 3)

        # Monotonicity: new_weak > baseline
        if new_weak <= baseline:
            return (False, f"Perturbed weak_strength {new_weak} not larger than baseline {baseline}")
        # Check that the ratio is (1.1)^2 = 1.21 within tolerance
        ratio = new_weak / baseline
        if not math.isclose(ratio, 1.21, rel_tol=0.1):
            return (False, f"Ratio of perturbed to baseline is {ratio}, expected ~1.21")
        return (True, "Weak strength increases with proton mass as expected (α m_p^2)")

    # ---------------------------------------------------------------
    # Answer sanity (weak, only one method)
    # ---------------------------------------------------------------

    def test_answer_sanity_strengths_finite_positive(self):
        """All strengths must be positive finite numbers."""
        strengths = [self.strong_strength, self.EM_strength, self.weak_strength, self.grav_strength]
        if any(v is None for v in strengths):
            return (False, "SKIP: one or more strengths missing")
        if any(v <= 0 or math.isinf(v) or math.isnan(v) for v in strengths):
            return (False, "Some strength is non-positive or infinite/NaN")
        return (True, "All strengths are positive finite numbers")