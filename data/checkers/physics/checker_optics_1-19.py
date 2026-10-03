import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary if present
        self.results = getattr(solution_instance, 'results', {})
        
        # Main result: magnification m (emitted by pseudocode)
        self.m = self.results.get('m', None)
        
        # Givens: p (object distance) and R (radius of curvature)
        # They might be stored as attributes of solution_instance or in results
        self.p = self._safe_get_attr_or_results(solution_instance, 'p')
        self.R = self._safe_get_attr_or_results(solution_instance, 'R')
        
        # Optional intermediate values that the solver might store
        self.f = self._safe_get_attr_or_results(solution_instance, 'f')
        self.q = self._safe_get_attr_or_results(solution_instance, 'q')
        
        # Derived focal length (only if R available)
        self.f_derived = self.R / 2.0 if self.R is not None else None
    
    def _safe_get_attr_or_results(self, instance, attr_name):
        """Try to get attribute from instance, then from results dict, return None if not found."""
        if hasattr(instance, attr_name):
            return getattr(instance, attr_name)
        return self.results.get(attr_name, None)
    
    # ------------------------------------------------------------
    # Interface tests: check that required keys exist and have valid types
    # ------------------------------------------------------------
    def test_interface_magnification_key(self):
        if self.m is None:
            return (False, "Missing m in results.")
        if not isinstance(self.m, (int, float)):
            return (False, f"m has invalid type: {type(self.m)}. Expected numeric.")
        return (True, "m key present and numeric.")
    
    def test_interface_givens_present(self):
        # Not strictly required, but useful for other tests.
        if self.p is None:
            return (False, "Missing p (object distance) in solution_instance or results.")
        if self.R is None:
            return (False, "Missing R (radius of curvature) in solution_instance or results.")
        return (True, "p and R are available.")
    
    # ------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity
    # ------------------------------------------------------------
    def test_invariant_focal_length_relation(self):
        if self.R is None:
            return (False, "SKIP: R not available for focal length check.")
        f_expected = self.R / 2.0
        # If solver provided explicit f, check it
        if self.f is not None:
            if math.isclose(self.f, f_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"f = {self.f} matches R/2 = {f_expected}.")
            else:
                return (False, f"f = {self.f} differs from R/2 = {f_expected}.")
        else:
            # f not provided, check the derived value
            return (True, f"f_derived = {f_expected}. No explicit f to compare.")
    
    def test_invariant_mirror_equation_via_m(self):
        """Use m = -q/p and mirror equation to verify consistency of m, p, and R."""
        if self.p is None or self.R is None or self.m is None:
            return (False, "SKIP: p, R, or m not available for mirror equation check.")
        f = self.R / 2.0
        # Derive expected magnification from mirror equation:
        # 1/q = 1/f - 1/p => q = 1/(1/f - 1/p)
        # Handle division carefully
        try:
            q_expected = 1.0 / (1.0 / f - 1.0 / self.p)
        except ZeroDivisionError:
            return (False, "Division by zero when computing q from mirror equation.")
        m_expected = -q_expected / self.p
        if math.isclose(self.m, m_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"m = {self.m} matches mirror equation prediction {m_expected}.")
        else:
            return (False, f"m = {self.m} differs from mirror equation prediction {m_expected}.")
    
    def test_invariant_sign_convention(self):
        """Check that m is positive (virtual image) and p < f."""
        if self.p is None or self.R is None or self.m is None:
            return (False, "SKIP: p, R, or m not available for sign convention check.")
        f = self.R / 2.0
        if self.p >= f:
            return (False, f"p = {self.p} >= f = {f}. For a shaving mirror p should be < f to get virtual image.")
        if self.m <= 0:
            return (False, f"m = {self.m}. For virtual image with p<f, m should be positive.")
        # If q is available, check that it's negative
        if self.q is not None:
            if self.q >= 0:
                return (False, f"q = {self.q}. For virtual image, q should be negative.")
            return (True, "Sign conventions satisfied: p<f, m>0, q<0.")
        return (True, "Sign conventions satisfied: p<f, m>0.")
    
    # ------------------------------------------------------------
    # Modeling tests: correct use of formulas and derivation quality
    # ------------------------------------------------------------
    def test_modeling_magnification_formula(self):
        """Check that m is correctly derived from mirror equation using givens."""
        if self.p is None or self.R is None or self.m is None:
            return (False, "SKIP: missing givens for magnification derivation check.")
        f = self.R / 2.0
        # Derive q from mirror equation
        try:
            q = 1.0 / (1.0 / f - 1.0 / self.p)
        except ZeroDivisionError:
            return (False, "Division error when computing q.")
        m_calculated = -q / self.p
        if math.isclose(self.m, m_calculated, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"m = {self.m} matches the calculated m = {m_calculated} from mirror equation.")
        else:
            return (False, f"m = {self.m} does not match calculated m = {m_calculated}.")
    
    def test_modeling_magnification_positive(self):
        """Magnification for a virtual image should be positive and greater than 1."""
        if self.m is None:
            return (False, "SKIP: m not available.")
        if self.m <= 0:
            return (False, f"m = {self.m}. For concave mirror with p<f, m should be positive and >1.")
        # In this specific case: p=4, f=6, m=3. But we only check >1 as a qualitative sanity.
        if self.m <= 1.0:
            return (False, f"m = {self.m}. For shaving mirror (p<f), magnification should be > 1.")
        return (True, f"m = {self.m} > 1, consistent with virtual, magnified image.")
    
    # ------------------------------------------------------------
    # Monotonicity tests: trend check via perturbed inputs
    # ------------------------------------------------------------
    def test_monotonicity_magnification_increases_with_p(self):
        """
        For a concave mirror with p<f, as p increases towards f,
        magnification should increase (image moves further behind mirror).
        Perturb p upward by 1% and check that m increases.
        """
        if self.p is None or self.R is None:
            return (False, "SKIP: p or R not available for monotonicity check.")
        f = self.R / 2.0
        if self.p >= f:
            return (False, "Monotonicity check only valid for p < f.")
        # Baseline magnification from formula
        m_baseline = f / (f - self.p)
        # Perturb p upward by 1%
        delta = self.p * 0.01
        p_perturbed = self.p + delta
        if p_perturbed >= f:
            return (False, "Perturbed p exceeds f, cannot test monotonicity.")
        m_perturbed = f / (f - p_perturbed)
        if m_perturbed > m_baseline:
            return (True, f"Monotonicity holds: m increases from {m_baseline} to {m_perturbed} when p increases by 1%.")
        else:
            return (False, f"Monotonicity violated: m_perturbed = {m_perturbed} <= m_baseline = {m_baseline}.")
    
    def test_monotonicity_magnification_increases_with_decreasing_f(self):
        """
        For fixed p < f, decreasing focal length (stronger curvature)
        should increase magnification (since m = f/(f-p) decreases as f increases).
        Perturb f downward by 1% and check that m increases.
        """
        if self.p is None or self.R is None:
            return (False, "SKIP: p or R not available for monotonicity check with f.")
        f = self.R / 2.0
        if self.p >= f:
            return (False, "Monotonicity check requires p < f.")
        # Baseline
        m_baseline = f / (f - self.p)
        # Perturb f downward by 1%
        f_perturbed = f * 0.99
        if self.p >= f_perturbed:
            return (False, "Perturbed f too small (p >= f_perturbed).")
        m_perturbed = f_perturbed / (f_perturbed - self.p)
        if m_perturbed > m_baseline:
            return (True, f"Monotonicity holds: m increases from {m_baseline} to {m_perturbed} when f decreases by 1%.")
        else:
            return (False, f"Monotonicity violated: m_perturbed = {m_perturbed} <= m_baseline = {m_baseline}.")
    
    # ------------------------------------------------------------
    # Weak answer sanity: only sign/magnitude/finite checks, no derivation
    # ------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        if self.m is None:
            return (False, "SKIP: m not available.")
        if math.isnan(self.m) or math.isinf(self.m):
            return (False, f"m is NaN or Inf: {self.m}")
        if self.m <= 0:
            return (False, f"m should be positive, got {self.m}.")
        if self.m > 10:
            return (False, f"m is unexpectedly large (>10): {self.m}. A shaving mirror usually gives moderate magnification.")
        return (True, f"m = {self.m} is finite, positive, and within plausible range.")
    
    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all methods starting with test_
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