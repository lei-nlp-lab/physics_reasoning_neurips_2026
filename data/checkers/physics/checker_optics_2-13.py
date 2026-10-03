import math
import traceback

class PhysicsChecker:
    """
    Checks physics consistency for thin film interference problem:
    maximum reflected intensity for normal incidence on a film with n=1.5, t=5000 Å.
    Expected result key: 'wavelengths' (list of float numbers).
    """
    def __init__(self, solution_instance):
        # Extract results safely, no numeric defaults
        self.results = getattr(solution_instance, 'results', None)
        if isinstance(self.results, dict):
            self.wavelengths = self.results.get('wavelengths', None)
        else:
            self.wavelengths = None

        # Physics constants from problem statement
        self.n = 1.5
        self.t = 5000.0          # angstrom
        self.visible_low = 4000.0
        self.visible_high = 7000.0
        self.const_prod = 2.0 * self.n * self.t   # 15000 angstrom

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_wavelengths_key(self):
        """Check that 'wavelengths' key exists in results."""
        if self.wavelengths is None:
            return (False, "No 'wavelengths' key in results dict.")
        return (True, "'wavelengths' key is present.")

    def test_interface_wavelengths_list(self):
        """Check that 'wavelengths' is a list."""
        if not isinstance(self.wavelengths, list):
            return (False, "'wavelengths' is not a list.")
        return (True, "'wavelengths' is a list.")

    def test_interface_wavelengths_numeric(self):
        """Check that every element in wavelengths is numeric."""
        if not self.wavelengths:
            return (False, "wavelengths list is empty or None")
        for w in self.wavelengths:
            if not isinstance(w, (int, float)):
                return (False, f"Non-numeric wavelength: {w}")
        return (True, "All wavelengths are numeric.")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_positive(self):
        """All wavelengths must be positive."""
        if not self.wavelengths:
            return (False, "No wavelengths to check.")
        for w in self.wavelengths:
            if w <= 0:
                return (False, f"Wavelength {w} is not positive.")
        return (True, "All wavelengths are positive.")

    def test_invariant_visible_range(self):
        """All wavelengths must lie between 4000 and 7000 Å."""
        if not self.wavelengths:
            return (False, "No wavelengths to check.")
        for w in self.wavelengths:
            if w < self.visible_low or w > self.visible_high:
                return (False, f"Wavelength {w} is outside visible range.")
        return (True, "All wavelengths lie within the visible spectrum.")

    def test_invariant_constructive_interference(self):
        """
        For each wavelength, there must exist an integer m such that
        2 n t = (m + 0.5) λ, within 1% relative / absolute tolerance.
        """
        if not self.wavelengths:
            return (False, "No wavelengths to check.")
        for w in self.wavelengths:
            m_float = self.const_prod / w - 0.5
            m_int = round(m_float)
            expected = self.const_prod / (m_int + 0.5)
            if not math.isclose(expected, w, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Wavelength {w}: nearest integer m={m_int} gives λ={expected}, mismatch.")
        return (True, "All wavelengths satisfy 2 n t = (m+0.5) λ.")

    def test_invariant_count(self):
        """Weak invariant: there should be exactly 2 wavelengths (m=2,3 yield two visible)."""
        if not self.wavelengths:
            return (False, "No wavelengths found.")
        if len(self.wavelengths) != 2:
            return (False, f"Expected exactly 2 wavelengths, found {len(self.wavelengths)}.")
        return (True, "Exactly 2 wavelengths present.")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_wavelengths_derived(self):
        """Re‑derive wavelengths from givens and compare with emitted list."""
        m_min = math.ceil(self.const_prod / self.visible_high - 0.5)
        m_max = math.floor(self.const_prod / self.visible_low - 0.5)
        derived = sorted([self.const_prod / (m + 0.5) for m in range(m_min, m_max + 1)])
        emitted = sorted(self.wavelengths) if self.wavelengths else []
        if len(emitted) != len(derived):
            return (False, f"Count mismatch: emitted {len(emitted)}, derived {len(derived)}.")
        for ew, dw in zip(emitted, derived):
            if not math.isclose(ew, dw, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"λ mismatch: emitted {ew}, derived {dw}")
        return (True, "Derived wavelengths match emitted within tolerance.")

    def test_modeling_m_range(self):
        """Each wavelength should correspond to m = 2 or m = 3 (the only integers giving visible λ)."""
        if not self.wavelengths:
            return (False, "No wavelengths to check.")
        for w in self.wavelengths:
            m_float = self.const_prod / w - 0.5
            m_int = round(m_float)
            if not math.isclose(m_float, m_int, abs_tol=1e-2):
                return (False, f"Wavelength {w}: m={m_float} is not integer.")
            if m_int not in (2, 3):
                return (False, f"Wavelength {w} gives m={m_int}, expected 2 or 3.")
        return (True, "All wavelengths correspond to m=2 or m=3.")

    # ------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare perturbed output)
    # ------------------------------------------------------------------
    def test_monotonicity_thickness_increases_wavelength(self):
        """
        Perturb film thickness t by +1% and verify that, for each interference order m,
        the new wavelength is larger than the original (since λ ∝ t).
        """
        if not self.wavelengths:
            return (False, "No wavelengths to check.")
        try:
            # For each wavelength, determine its integer m
            m_values = []
            for w in self.wavelengths:
                m_float = self.const_prod / w - 0.5
                m_int = round(m_float)
                if not math.isclose(m_float, m_int, abs_tol=1e-2):
                    return (False, f"Cannot reliably determine m for wavelength {w}.")
                m_values.append(m_int)

            # Perturb thickness: +1%
            t_perturbed = self.t * 1.01
            const_prod_pert = 2.0 * self.n * t_perturbed

            all_increased = True
            for w, m in zip(self.wavelengths, m_values):
                new_lambda = const_prod_pert / (m + 0.5)
                if not (new_lambda > w):   # should be strictly larger
                    all_increased = False
                    # Note: small numerical noise might cause isclose for equality;
                    # but with +1% and same m, it should be clearly larger.
            if not all_increased:
                return (False, "Wavelengths did not all increase when thickness increased by 1%.")
            return (True, "All wavelengths increase when thickness increases by 1% (consistent with λ ∝ t).")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (no math.isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Very weak checks: finite, distinct, within a broad range."""
        if not self.wavelengths:
            return (False, "No wavelengths to check.")
        for w in self.wavelengths:
            if w is None or not math.isfinite(w):
                return (False, f"Wavelength {w} is not finite.")
        if len(set(self.wavelengths)) != len(self.wavelengths):
            return (False, "Duplicate wavelengths found.")
        # Broad range (3000-8000 Å) – not the precise visible band
        for w in self.wavelengths:
            if w < 3000 or w > 8000:
                return (False, f"Wavelength {w} is far outside the visible range.")
        return (True, "Weak sanity passed.")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, return (success, logs)."""
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