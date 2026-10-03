import math
import traceback

class PhysicsChecker:
    """
    Robust checker for camera lens aperture problem:
    Find aperture D such that diffraction blur equals defocus blur for a star (object at infinity).
    """
    
    def __init__(self, solution_instance):
        # ---------- Extract givens ----------
        # Try direct attribute access first, then dict-like access
        self.f = None
        self.s = None
        self.lam = None
        self.D_mm = None
        self.D_cm = None
        self.explanation_steps = None
        self.pseudocode = None
        
        # Use a helper for safe extraction
        def safe_get(obj, key, default=None):
            """Try attribute access then dict get."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            try:
                return getattr(obj, key, default)
            except:
                pass
            try:
                return obj[key]
            except:
                return default
        
        # Try to get givens from solution_instance.givens (dict) or directly
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.f = solution_instance.givens.get('f', None)
            self.s = solution_instance.givens.get('s', None)
            self.lam = solution_instance.givens.get('lam', None)
        else:
            self.f = safe_get(solution_instance, 'f', None)
            self.s = safe_get(solution_instance, 's', None)
            self.lam = safe_get(solution_instance, 'lam', None)
        
        # Try to get results
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.D_mm = solution_instance.results.get('D_mm', None)
            self.D_cm = solution_instance.results.get('D_cm', None)
        else:
            self.D_mm = safe_get(solution_instance, 'D_mm', None)
            self.D_cm = safe_get(solution_instance, 'D_cm', None)
        
        # Explanation steps
        if hasattr(solution_instance, 'explanation_steps'):
            self.explanation_steps = solution_instance.explanation_steps
        else:
            self.explanation_steps = safe_get(solution_instance, 'explanation_steps', None)
        
        if hasattr(solution_instance, 'pseudocode'):
            self.pseudocode = solution_instance.pseudocode
        else:
            self.pseudocode = safe_get(solution_instance, 'pseudocode', None)
        
        # Derived quantities (computed in checker)
        self.v = None
        self.delta_v = None
        if self.f is not None and self.s is not None and self.f != 0 and self.s != self.f:
            self.v = (self.f * self.s) / (self.s - self.f)
            self.delta_v = abs(self.v - self.f)   # for star: object at infinity so focus difference
            # Note: For star at infinity, defocus: image distance v (for finite object) vs f.
            # Since s is finite, v > f, so delta_v = v - f.
            # But we keep abs for safety.
    
    def _compute_predicted_D_mm(self, f=None, s=None, lam=None):
        """Compute D in mm from given or stored parameters."""
        if f is None:
            f = self.f
        if s is None:
            s = self.s
        if lam is None:
            lam = self.lam
        if any(x is None for x in (f, s, lam)):
            return None
        if s <= f:
            return None
        v = f * s / (s - f)
        delta_v = v - f
        if delta_v <= 0:
            return None
        D_mm = math.sqrt(1.22 * lam * f * f / delta_v)
        return D_mm
    
    # ===================== Interface Tests =====================
    def test_interface_givens_present(self):
        """Check that required givens (f, s, lam) are present and positive."""
        for name, val in [('f', self.f), ('s', self.s), ('lam', self.lam)]:
            if val is None:
                return (False, f"Missing givens: {name} is None")
            if val <= 0:
                return (False, f"Invalid givens: {name} must be positive, got {val}")
        return (True, "All givens present and positive")
    
    def test_interface_results_present(self):
        """Check that both D_mm and D_cm are present and positive."""
        for name, val in [('D_mm', self.D_mm), ('D_cm', self.D_cm)]:
            if val is None:
                return (False, f"Missing result: {name} is None")
            if val <= 0:
                return (False, f"Invalid result: {name} must be positive, got {val}")
        return (True, "Both results present and positive")
    
    def test_interface_results_unit_consistency(self):
        """Check D_cm = D_mm / 10."""
        if self.D_mm is None or self.D_cm is None:
            return (False, "SKIP: missing D_mm or D_cm")
        expected_cm = self.D_mm / 10.0
        if not math.isclose(self.D_cm, expected_cm, rel_tol=1e-9):
            return (False, f"D_cm ({self.D_cm}) not equal D_mm/10 ({expected_cm})")
        return (True, f"D_cm = D_mm/10 holds: {self.D_cm} ≈ {expected_cm}")
    
    # ===================== Invariant / Theorem Checks =====================
    def test_invariant_lens_equation(self):
        """Check 1/s + 1/v = 1/f with computed v."""
        if self.f is None or self.s is None:
            return (False, "SKIP: missing f or s")
        if self.v is None:
            return (False, "SKIP: cannot compute v (s <= f?)")
        lhs = 1.0/self.s + 1.0/self.v
        rhs = 1.0/self.f
        if math.isclose(lhs, rhs, rel_tol=1e-3):
            return (True, f"Lens equation satisfied: 1/s + 1/v = {lhs} ≈ 1/f = {rhs}")
        else:
            return (False, f"Lens equation violated: 1/s + 1/v = {lhs}, 1/f = {rhs}")
    
    def test_invariant_equality_condition(self):
        """Check c_def = c_diff using results D_mm."""
        if any(x is None for x in (self.f, self.s, self.lam, self.D_mm)):
            return (False, "SKIP: missing required values")
        if self.v is None:
            return (False, "SKIP: cannot compute v")
        # Defocus blur: c_def = D * |v - f| / f
        c_def = self.D_mm * abs(self.v - self.f) / self.f
        # Diffraction blur: c_diff = 2.44 * lam * f / D
        c_diff = 2.44 * self.lam * self.f / self.D_mm
        if math.isclose(c_def, c_diff, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Equality holds: c_def={c_def} ≈ c_diff={c_diff}")
        else:
            return (False, f"Equality violated: c_def={c_def}, c_diff={c_diff}")
    
    def test_invariant_positivity(self):
        """Check all physical quantities are >0."""
        vals = {'f': self.f, 's': self.s, 'lam': self.lam, 'D_mm': self.D_mm, 'D_cm': self.D_cm}
        for name, v in vals.items():
            if v is not None and v <= 0:
                return (False, f"Non-positive {name}: {v}")
        if self.v is not None and self.v <= 0:
            return (False, f"Non-positive v: {self.v}")
        return (True, "All examined quantities are positive")
    
    # ===================== Modeling / Trace Checks =====================
    def test_modeling_defocus_formula(self):
        """Check that computed D matches defocus-diffraction equality."""
        if any(x is None for x in (self.f, self.s, self.lam, self.D_mm)):
            return (False, "SKIP: missing values")
        if self.v is None:
            return (False, "SKIP: v not available")
        delta = abs(self.v - self.f)
        if delta == 0:
            return (False, "delta_v=0 leads to infinite D")
        expected_D_mm = math.sqrt(1.22 * self.lam * self.f * self.f / delta)
        if math.isclose(self.D_mm, expected_D_mm, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Predicted D_mm={expected_D_mm} matches result {self.D_mm}")
        else:
            return (False, f"Formula D_mm mismatch: expected {expected_D_mm}, got {self.D_mm}")
    
    def test_modeling_pseudocode_emits(self):
        """Check that pseudocode emit contract is satisfied if pseudocode available."""
        if self.pseudocode is None:
            return (False, "SKIP: no pseudocode provided")
        # Attempt to check that results keys match emit contracts
        # We just verify we have the keys; actual values checked elsewhere.
        if self.D_mm is None or self.D_cm is None:
            return (False, "Results missing required emit keys")
        # The emit contract says D_mm = D, D_cm = D/10.
        # We already have that check, but we can double-check unit consistency.
        if not math.isclose(self.D_cm * 10.0, self.D_mm, rel_tol=1e-9):
            return (False, "Pseudocode emit contract violated: D_cm * 10 != D_mm")
        return (True, "Pseudocode emit keys present and units consistent")
    
    # ===================== Monotonicity Tests =====================
    def test_monotonicity_aperture_increases_with_wavelength(self):
        """If lambda increases, D increases (D ∝ sqrt(lambda))."""
        if any(x is None for x in (self.f, self.s, self.lam, self.D_mm)):
            return (False, "SKIP: missing data")
        # Perturb lambda by +10%
        lam_pert = self.lam * 1.10
        D_pert = self._compute_predicted_D_mm(lam=lam_pert)
        if D_pert is None:
            return (False, "Cannot compute for perturbed lambda")
        if D_pert >= self.D_mm:
            return (True, f"Monotonicity holds: lambda+10% -> D={D_pert} > baseline {self.D_mm}")
        else:
            return (False, f"Monotonicity fails: lambda+10% gave D={D_pert} <= baseline {self.D_mm}")
    
    def test_monotonicity_aperture_increases_with_distance(self):
        """If s increases, D increases (D ∝ sqrt(s) for large s)."""
        if any(x is None for x in (self.f, self.s, self.lam, self.D_mm)):
            return (False, "SKIP: missing data")
        s_pert = self.s * 1.10
        D_pert = self._compute_predicted_D_mm(s=s_pert)
        if D_pert is None:
            return (False, "Cannot compute for perturbed s")
        if D_pert >= self.D_mm:
            return (True, f"Monotonicity holds: s+10% -> D={D_pert} > baseline {self.D_mm}")
        else:
            return (False, f"Monotonicity fails: s+10% gave D={D_pert} <= baseline {self.D_mm}")
    
    # ===================== Weak Answer Sanity (max one method) =====================
    def test_answer_sanity_range(self):
        """Sanity check: D_cm should be in a reasonable range (0.01 to 1 cm)."""
        if self.D_cm is None:
            return (False, "SKIP: missing D_cm")
        if 0.01 < self.D_cm < 1.0:
            return (True, f"D_cm={self.D_cm} within plausible range (0.01, 1.0) cm")
        else:
            return (False, f"D_cm={self.D_cm} outside plausible range (0.01, 1.0) cm")
    
    # ===================== Main test runner =====================
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test_ methods
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)