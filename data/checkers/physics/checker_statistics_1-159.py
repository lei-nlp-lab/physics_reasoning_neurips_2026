import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for local entropy generation rate per unit volume in a heat conductor.
    Checks interface completeness, physical invariants (Fourier's law + entropy balance),
    modeling quality, monotonicity, and a weak final-answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract givens: thermal conductivity k, temperature T, temperature gradient vector gradT
        self.k = None
        self.T = None
        self.gradT = None
        # Try from .givens dict first, then direct attributes
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            giv = solution_instance.givens
            self.k = giv.get('k', None)
            self.T = giv.get('T', None)
            self.gradT = giv.get('gradT', None)
        if self.k is None:
            self.k = getattr(solution_instance, 'k', None)
        if self.T is None:
            self.T = getattr(solution_instance, 'T', None)
        if self.gradT is None:
            self.gradT = getattr(solution_instance, 'gradT', None)

        # Extract result: local entropy generation rate per volume
        self.sdot_gen = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.sdot_gen = solution_instance.results.get('sdot_gen', None)
        if self.sdot_gen is None:
            self.sdot_gen = getattr(solution_instance, 'sdot_gen', None)

        # Store raw solution instance for potential fallback
        self.si = solution_instance

    # ------------------------------------------------------------------
    # Helper: compute expected sdot_gen = k * |gradT|^2 / T^2
    # Works for numeric or sympy symbolic inputs.
    # Returns None if required givens are missing.
    # ------------------------------------------------------------------
    def _expected_sdot_gen(self, k, T, gradT):
        if k is None or T is None or gradT is None:
            return None
        # Compute norm squared of gradT
        def norm_sq(vec):
            if isinstance(vec, (list, tuple)):
                return sum(x**2 for x in vec)
            elif isinstance(vec, sp.Matrix):
                return (vec.dot(vec))  # dot yields scalar
            elif isinstance(vec, sp.Expr):
                # Assume scalar? Not expected.
                return vec**2
            else:
                return None
        nsq = norm_sq(gradT)
        if nsq is None:
            return None
        try:
            return k * nsq / (T**2)
        except:
            return None

    # ------------------------------------------------------------------
    # Helper: numeric isclose check with tolerances
    # ------------------------------------------------------------------
    def _isclose(self, a, b):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
        except:
            return False

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_sdot_gen_key(self):
        """Required result key 'sdot_gen' must exist and be accessible."""
        if self.sdot_gen is None:
            return (False, "Missing result: 'sdot_gen' not found in results or as attribute")
        return (True, "Result 'sdot_gen' is present")

    def test_interface_givens_availability(self):
        """Required givens k, T, gradT should exist (not None)."""
        missing = []
        if self.k is None:
            missing.append("k")
        if self.T is None:
            missing.append("T")
        if self.gradT is None:
            missing.append("gradT")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_entropy_generation_formula(self):
        """sdot_gen must equal k * |gradT|^2 / T^2."""
        if self.sdot_gen is None:
            return (False, "SKIP: sdot_gen not available")
        if any(x is None for x in (self.k, self.T, self.gradT)):
            return (False, "SKIP: insufficient givens")
        expected = self._expected_sdot_gen(self.k, self.T, self.gradT)
        if expected is None:
            return (False, "Cannot compute expected sdot_gen")
        if self._isclose(self.sdot_gen, expected):
            return (True, "sdot_gen matches theoretical formula")
        return (False, f"sdot_gen {self.sdot_gen} != expected {expected}")

    def test_invariant_positivity(self):
        """Entropy generation rate must be non-negative (within numerical tolerance)."""
        if self.sdot_gen is None:
            return (False, "SKIP: sdot_gen not available")
        try:
            val = float(self.sdot_gen)
            if val >= -1e-10:  # allow tiny negative due to rounding
                return (True, f"sdot_gen = {val} >= 0 (within tolerance)")
            return (False, f"sdot_gen is negative: {val}")
        except:
            return (False, "Cannot convert sdot_gen to float for positivity check")

    def test_invariant_zero_gradient_yields_zero(self):
        """If gradT is zero vector, sdot_gen must be ~0."""
        if self.sdot_gen is None or self.gradT is None:
            return (False, "SKIP: missing data")
        # Check if gradT is effectively zero
        def is_zero_vec(vec):
            if isinstance(vec, (list, tuple)):
                return all(abs(x) < 1e-12 for x in vec)
            elif isinstance(vec, sp.Matrix):
                return all(sp.simplify(x) == 0 for x in vec)
            return False
        if is_zero_vec(self.gradT):
            expected_zero = 0.0
            if self._isclose(self.sdot_gen, expected_zero):
                return (True, "Zero gradient yields correct zero entropy generation")
            return (False, f"Zero gradient but sdot_gen = {self.sdot_gen} not zero")
        return (False, "SKIP: gradT is not zero vector")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_sign_consistent_with_fourier(self):
        """If gradT is nonzero, sdot_gen must be positive (since k>0, T>0)."""
        if self.sdot_gen is None or self.gradT is None:
            return (False, "SKIP: missing data")
        try:
            sdot = float(self.sdot_gen)
            if sdot <= 0:
                # Could be zero if gradT is zero, but that's covered in another test
                return (False, f"sdot_gen = {sdot} should be >0 for nonzero gradient")
            return (True, "sdot_gen positive as expected")
        except:
            return (False, "Cannot convert sdot_gen to float")

    def test_modeling_fourier_law_consistency(self):
        """Compute q from Fourier's law (q = -k gradT) and verify entropy generation formula.
           This is a consistency check using the explanation steps.
        """
        if any(x is None for x in (self.k, self.T, self.gradT)):
            return (False, "SKIP: missing givens")
        # Compute q vector: -k * gradT (component-wise)
        gradT = self.gradT
        if isinstance(gradT, (list, tuple)):
            q = [-self.k * x for x in gradT]
        elif isinstance(gradT, sp.Matrix):
            q = -self.k * gradT
        else:
            return (False, "SKIP: gradT has unsupported type")
        # Compute q · ∇(1/T) = q · (-gradT / T^2)
        # That equals k * |gradT|^2 / T^2 = sdot_gen
        # So the test is equivalent to invariant; we can just check that sdot_gen matches.
        # But for modeling, we can check that sdot_gen equals q·(-gradT/T^2).
        # Since we already have invariant, this is redundant but adds a cross-check.
        expected = self._expected_sdot_gen(self.k, self.T, self.gradT)
        if expected is None:
            return (False, "Cannot compute expected")
        # Compute q·(-gradT/T^2) numerically or symbolically
        if all(hasattr(x, 'evalf') for x in [self.k, self.T] + (list(gradT) if isinstance(gradT, (list,tuple)) else [gradT])):
            # symbolic: simplify difference
            try:
                import sympy as sp
                q_vec = [-self.k * x for x in gradT]
                grad_inv_T = [-x / (self.T**2) for x in gradT]
                dot_product = sum(q * gi for q, gi in zip(q_vec, grad_inv_T))
                diff = sp.simplify(dot_product - self.sdot_gen)
                if diff == 0:
                    return (True, "Modeling: sdot_gen = q·∇(1/T) holds symbolically")
                return (False, f"Symbolic difference: {diff}")
            except:
                return (False, "Symbolic modeling check failed")
        else:
            # numeric
            try:
                q = [-float(self.k) * float(x) for x in gradT]
                grad_inv_T = [-float(x) / (float(self.T)**2) for x in gradT]
                dot_product = sum(qi * gti for qi, gti in zip(q, grad_inv_T))
                if self._isclose(self.sdot_gen, dot_product):
                    return (True, "Modeling: sdot_gen ≈ q·∇(1/T) numerically")
                return (False, f"Numerical mismatch: sdot_gen {self.sdot_gen} vs q·∇(1/T) {dot_product}")
            except:
                return (False, "Numeric modeling check failed")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb one input, recompute expected, compare)
    # ------------------------------------------------------------------
    def test_monotonicity_k_increase(self):
        """If thermal conductivity increases, sdot_gen should increase (for fixed T, gradT)."""
        if any(x is None for x in (self.k, self.T, self.gradT)):
            return (False, "SKIP: missing givens")
        baseline = self._expected_sdot_gen(self.k, self.T, self.gradT)
        if baseline is None:
            return (False, "Cannot compute baseline")
        # perturb k by +10%
        k_pert = float(self.k) * 1.1 if not isinstance(self.k, sp.Basic) else self.k * 1.1
        perturbed = self._expected_sdot_gen(k_pert, self.T, self.gradT)
        if perturbed is None:
            return (False, "Cannot compute perturbed")
        try:
            if float(perturbed) > float(baseline):
                return (True, f"k increase → sdot_gen increases (baseline {baseline}, perturbed {perturbed})")
            return (False, f"k increase did not increase sdot_gen: baseline {baseline}, perturbed {perturbed}")
        except:
            return (False, "Cannot compare numeric values")

    def test_monotonicity_gradient_increase(self):
        """If |gradT| increases, sdot_gen should increase (for fixed k, T)."""
        if any(x is None for x in (self.k, self.T, self.gradT)):
            return (False, "SKIP: missing givens")
        baseline = self._expected_sdot_gen(self.k, self.T, self.gradT)
        if baseline is None:
            return (False, "Cannot compute baseline")
        # scale gradT by 1.1
        gradT = self.gradT
        if isinstance(gradT, (list, tuple)):
            new_gradT = [float(x)*1.1 if not isinstance(x, sp.Basic) else x*1.1 for x in gradT]
        elif isinstance(gradT, sp.Matrix):
            new_gradT = gradT * 1.1
        else:
            return (False, "Unsupported gradT type")
        perturbed = self._expected_sdot_gen(self.k, self.T, new_gradT)
        if perturbed is None:
            return (False, "Cannot compute perturbed")
        try:
            if float(perturbed) > float(baseline):
                return (True, f"gradT increase → sdot_gen increases (baseline {baseline}, perturbed {perturbed})")
            return (False, f"gradT increase did not increase sdot_gen: baseline {baseline}, perturbed {perturbed}")
        except:
            return (False, "Cannot compare numeric values")

    def test_monotonicity_temperature_increase(self):
        """If temperature increases (T larger), sdot_gen should decrease (for fixed k, gradT)."""
        if any(x is None for x in (self.k, self.T, self.gradT)):
            return (False, "SKIP: missing givens")
        baseline = self._expected_sdot_gen(self.k, self.T, self.gradT)
        if baseline is None:
            return (False, "Cannot compute baseline")
        # perturb T by +10%
        T_pert = float(self.T) * 1.1 if not isinstance(self.T, sp.Basic) else self.T * 1.1
        perturbed = self._expected_sdot_gen(self.k, T_pert, self.gradT)
        if perturbed is None:
            return (False, "Cannot compute perturbed")
        try:
            if float(perturbed) < float(baseline):
                return (True, f"T increase → sdot_gen decreases (baseline {baseline}, perturbed {perturbed})")
            return (False, f"T increase did not decrease sdot_gen: baseline {baseline}, perturbed {perturbed}")
        except:
            return (False, "Cannot compare numeric values")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one, no isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitude(self):
        """sdot_gen should be finite and if inputs are positive, sdot_gen should be nonnegative."""
        if self.sdot_gen is None:
            return (False, "SKIP: sdot_gen not available")
        try:
            val = float(self.sdot_gen)
            if math.isfinite(val) and val >= -1e-10:
                return (True, f"sdot_gen = {val} is finite and nonnegative")
            return (False, f"sdot_gen is not finite or negative: {val}")
        except:
            return (False, "Cannot convert sdot_gen to float")

    # ------------------------------------------------------------------
    # Main runner: discover all test_ methods and execute safely.
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)