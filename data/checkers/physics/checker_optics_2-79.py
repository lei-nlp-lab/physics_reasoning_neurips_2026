import math
import sympy
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the polarising plates problem.
    Checks interface, invariant fraction, modeling quality, monotonicity, and basic answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None) or {}
        # Final transmitted intensity
        self.I4 = self.results.get('I4', None)
        # Incident intensity – try multiple sources
        self.I0 = (
            self.results.get('I0', None) or
            getattr(solution_instance, 'I0', None) or
            getattr(solution_instance, 'givens', {}).get('I0', None)
        )
        # Explanation steps (table or plain text)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', '')
        # Pseudocode
        self.pseudocode = getattr(solution_instance, 'pseudocode', '')
        # Relative angle (known from problem statement)
        self.theta_deg = 30.0

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_I4_exists(self):
        """Check that results['I4'] is present and not None."""
        if 'I4' not in self.results:
            return (False, "Key 'I4' missing from results")
        if self.I4 is None:
            return (False, "I4 is None")
        return (True, "I4 exists")

    def test_interface_I4_type(self):
        """Check that I4 is either a number or a sympy expression."""
        if self.I4 is None:
            return (False, "SKIP: I4 not available")
        if isinstance(self.I4, (int, float, complex)):
            return (True, "I4 is a numeric type")
        if isinstance(self.I4, sympy.Basic):
            return (True, "I4 is a sympy expression")
        return (False, f"I4 has unexpected type: {type(self.I4)}")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_fraction(self):
        """
        Check that the transmitted fraction I4/I0 equals 27/128.
        Uses math.isclose for numeric/symbolic evaluation.
        """
        if self.I4 is None:
            return (False, "SKIP: I4 not available")
        I0_sym = sympy.Symbol('I0')
        expected_frac = sympy.Rational(27, 128)
        try:
            if isinstance(self.I4, sympy.Basic):
                # If expression contains I0, simplify the fraction
                if self.I4.has(I0_sym):
                    frac = sympy.simplify(self.I4 / I0_sym)
                    if frac.equals(expected_frac):
                        return (True, "I4/I0 = 27/128")
                    else:
                        return (False, f"I4/I0 = {frac}, expected {expected_frac}")
                else:
                    # I4 is a pure number (no I0 symbol)
                    frac = float(self.I4)
                    if math.isclose(frac, 27/128, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"I4 ≈ {frac}")
                    else:
                        return (False, f"I4 = {frac}, expected ≈0.2109")
            else:
                # plain numeric
                frac = float(self.I4)
                if math.isclose(frac, 27/128, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"I4 ≈ {frac}")
                else:
                    return (False, f"I4 = {frac}, expected ≈0.2109")
        except Exception as e:
            return (False, f"Error during fraction check: {e}")

    def test_invariant_positivity(self):
        """Check that I4 is non-negative (only for numeric I4)."""
        if self.I4 is None:
            return (False, "SKIP: I4 not available")
        if isinstance(self.I4, sympy.Basic):
            return (True, "SKIP: symbolic expression")
        try:
            val = float(self.I4)
            if val < 0:
                return (False, f"I4 = {val} is negative")
            return (True, f"I4 = {val} >= 0")
        except:
            return (False, "Could not convert I4 to float")

    def test_invariant_finite(self):
        """Check that I4 is finite (not NaN or inf) for numeric I4."""
        if self.I4 is None:
            return (False, "SKIP: I4 not available")
        if isinstance(self.I4, sympy.Basic):
            return (True, "SKIP: symbolic expression")
        try:
            val = float(self.I4)
            if math.isnan(val) or math.isinf(val):
                return (False, f"I4 is NaN or Inf")
            return (True, "I4 is finite")
        except:
            return (False, "Could not convert I4 to float")

    # ------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------
    def test_modeling_malus_mentioned(self):
        """Check that explanation_steps mentions Malus' law."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps")
        text = self.explanation_steps.lower()
        if 'malus' in text:
            return (True, "Malus' law mentioned")
        else:
            return (False, "Malus' law not found in explanation")

    def test_modeling_cos2_factor(self):
        """Check that explanation_steps mentions cos^2 factor."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps")
        text = self.explanation_steps
        if 'cos^2' in text or 'cos²' in text:
            return (True, "cos^2 factor mentioned")
        else:
            return (False, "cos^2 factor not found in explanation")

    def test_modeling_unpolarised_first_plate(self):
        """Check that the explanation mentions I0/2 or half for first plate."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps")
        text = self.explanation_steps.lower()
        if 'i0/2' in text or 'half' in text or 'i0/2' in text:
            return (True, "Unpolarised → half intensity mentioned")
        else:
            return (False, "First-plate halving not mentioned")

    def test_modeling_sequential_mult(self):
        """Check that the pseudocode or explanation indicates multiplying by (3/4)^3."""
        combined = (self.explanation_steps + ' ' + self.pseudocode).lower()
        # Look for indications of three multiplications or factor 3/4
        if '(3/4)^3' in combined or '27/64' in combined or 'cos^6' in combined:
            return (True, "Sequential multiplication by 3/4 three times mentioned")
        else:
            return (False, "No clear mention of three cos^2 multiplications")

    # ------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------
    def _compute_transmitted_intensity(self, I0, theta_deg, num_plates):
        """Compute transmitted intensity for N polarising plates with sequential 30° rotation.
        Returns I_out = I0/2 * (cos²(theta_deg))^(num_plates-1)
        """
        cos2 = math.cos(math.radians(theta_deg)) ** 2
        return I0 * 0.5 * (cos2 ** (num_plates - 1))

    def test_monotonicity_angle_increase(self):
        """Increasing relative angle (e.g., from 30° to 31°) reduces transmitted intensity."""
        if self.I4 is None or self.I0 is None:
            return (False, "SKIP: I4 or I0 not available")
        # Number of plates = 4, angle reference = 30°
        baseline = self._compute_transmitted_intensity(self.I0, self.theta_deg, 4)
        perturbed = self._compute_transmitted_intensity(self.I0, self.theta_deg + 1, 4)
        if not math.isclose(baseline, float(self.I4), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline recomputed {baseline} does not match user's I4 {self.I4}")
        if perturbed < baseline - 1e-12:
            return (True, f"Perturbed (31°) = {perturbed} < baseline = {baseline}")
        else:
            return (False, f"Perturbed (31°) = {perturbed} is not less than baseline = {baseline}")

    def test_monotonicity_angle_decrease(self):
        """Decreasing relative angle (e.g., from 30° to 29°) increases transmitted intensity."""
        if self.I4 is None or self.I0 is None:
            return (False, "SKIP: I4 or I0 not available")
        baseline = self._compute_transmitted_intensity(self.I0, self.theta_deg, 4)
        perturbed = self._compute_transmitted_intensity(self.I0, self.theta_deg - 1, 4)
        if not math.isclose(baseline, float(self.I4), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline recomputed {baseline} does not match user's I4 {self.I4}")
        if perturbed > baseline + 1e-12:
            return (True, f"Perturbed (29°) = {perturbed} > baseline = {baseline}")
        else:
            return (False, f"Perturbed (29°) = {perturbed} is not greater than baseline = {baseline}")

    def test_monotonicity_plates_increase(self):
        """Increasing number of polarising plates (N=5 vs N=4) reduces transmitted intensity."""
        if self.I4 is None or self.I0 is None:
            return (False, "SKIP: I4 or I0 not available")
        baseline = self._compute_transmitted_intensity(self.I0, self.theta_deg, 4)  # n=4
        perturbed = self._compute_transmitted_intensity(self.I0, self.theta_deg, 5)  # n=5
        if not math.isclose(baseline, float(self.I4), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline recomputed {baseline} does not match user's I4 {self.I4}")
        if perturbed < baseline - 1e-12:
            return (True, f"Perturbed (N=5) = {perturbed} < baseline (N=4) = {baseline}")
        else:
            return (False, f"Perturbed (N=5) = {perturbed} is not less than baseline = {baseline}")

    # ------------------------------------------------------------
    # Answer sanity (weak sanity only)
    # ------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: I4 exists, is not NaN/Inf, and lies in [0,1] (if numeric).
        No re-derivation of expected answer, no isclose.
        """
        if self.I4 is None:
            return (False, "SKIP: I4 not available")
        # For symbolic expressions, accept as is (no numeric range check)
        if isinstance(self.I4, sympy.Basic):
            return (True, "Symbolic expression, accepted (no numeric bounds applied)")
        # Numeric check
        try:
            val = float(self.I4)
            if math.isnan(val) or math.isinf(val):
                return (False, "I4 is NaN or Inf")
            if val < 0 or val > 1:      # transmitted fraction cannot exceed incident
                return (False, f"I4 = {val} is outside expected [0,1]")
            return (True, f"I4 = {val} is in [0,1] and finite")
        except:
            return (False, "Could not evaluate I4 numerically")

    # ------------------------------------------------------------
    # Run all tests (discovery and execution)
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with "test_"
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