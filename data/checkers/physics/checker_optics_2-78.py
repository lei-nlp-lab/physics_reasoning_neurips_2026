import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract I0 (given constant)
        if isinstance(solution_instance, dict):
            self.I0 = solution_instance.get('I0', None)
            results = solution_instance.get('results', {})
        else:
            self.I0 = getattr(solution_instance, 'I0', None)
            results = getattr(solution_instance, 'results', {}) or {}
        self.results = results

        # Required result keys
        self.I_theta = results.get('I_theta', None)
        self.I_max   = results.get('I_max', None)
        self.f_unpol = results.get('f_unpol', None)

        # Symbol for evaluation of I_theta (if it is a symbolic expression)
        self.theta = sp.Symbol('theta')

    # ------------------------------------------------------------------
    # Helper: evaluate I_theta at a numerical angle (radians)
    # ------------------------------------------------------------------
    def _eval_I_theta(self, theta_val):
        if self.I_theta is None:
            return None
        try:
            if isinstance(self.I_theta, sp.Expr):
                val = self.I_theta.subs(self.theta, theta_val)
                return float(sp.N(val))
            elif isinstance(self.I_theta, (int, float)):
                return float(self.I_theta)   # ignore angle
            else:
                return float(self.I_theta)   # try to convert
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All three required result keys must exist."""
        if self.I_theta is None:
            return False, "Missing key: I_theta"
        if self.I_max is None:
            return False, "Missing key: I_max"
        if self.f_unpol is None:
            return False, "Missing key: f_unpol"
        return True, "All required keys are present"

    def test_interface_I_theta_type(self):
        """I_theta should be a type that can be evaluated (sympy or numeric)."""
        if self.I_theta is None:
            return False, "I_theta not available"
        if isinstance(self.I_theta, (sp.Expr, int, float)):
            return True, f"I_theta type: {type(self.I_theta).__name__}"
        else:
            return False, f"I_theta type {type(self.I_theta).__name__} not supported"

    # ------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ------------------------------------------------------------------
    def test_invariant_extrema(self):
        """I(0) = 1.5 I0, I(π/2) = I0."""
        if self.I0 is None:
            return False, "SKIP: I0 not available"
        if self.I_theta is None:
            return False, "SKIP: I_theta not available"
        at0  = self._eval_I_theta(0.0)
        at90 = self._eval_I_theta(math.pi / 2.0)
        if at0 is None or at90 is None:
            return False, "Cannot evaluate I_theta at required angles"
        if not math.isclose(at0, 1.5 * self.I0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"I(0)={at0} is not close to 1.5 I0 = {1.5*self.I0}"
        if not math.isclose(at90, self.I0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"I(π/2)={at90} is not close to I0 = {self.I0}"
        return True, "Extrema match the problem statement"

    def test_invariant_part_b_max(self):
        """Maximum after QWP should be 1.75 I0."""
        if self.I0 is None:
            return False, "SKIP: I0 not available"
        if self.I_max is None:
            return False, "SKIP: I_max not available"
        expected = 1.75 * self.I0
        if not math.isclose(self.I_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"I_max={self.I_max} is not close to 1.75 I0 = {expected}"
        return True, "I_max matches expected value"

    def test_invariant_f_unpol(self):
        """Unpolarised fraction should be 0.6."""
        if self.f_unpol is None:
            return False, "SKIP: f_unpol not available"
        if not math.isclose(self.f_unpol, 0.6, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"f_unpol={self.f_unpol} is not close to 0.6"
        return True, "Unpolarised fraction matches expected value"

    # ------------------------------------------------------------------
    # Modeling / trace quality checks
    # ------------------------------------------------------------------
    def test_modeling_I_theta_at_45(self):
        """Check that I(π/4) = (I(0) + I(π/2))/2."""
        if self.I0 is None:
            return False, "SKIP: I0 not available"
        if self.I_theta is None:
            return False, "SKIP: I_theta not available"
        at0  = self._eval_I_theta(0.0)
        at90 = self._eval_I_theta(math.pi / 2.0)
        at45 = self._eval_I_theta(math.pi / 4.0)
        if None in (at0, at90, at45):
            return False, "Cannot evaluate I_theta at required angles"
        expected = (at0 + at90) / 2.0
        if not math.isclose(at45, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"I(π/4)={at45} is not close to ({at0}+{at90})/2 = {expected}"
        return True, "I(π/4) satisfies the linear combination form"

    def test_modeling_unpolarized_fraction_derived(self):
        """
        Use total incident intensity (I_x + I_y = 2.5 I0) and I_max to
        re‑derive the unpolarised fraction and compare with f_unpol.
        """
        if self.I0 is None or self.I_max is None or self.f_unpol is None:
            return False, "SKIP: required results missing"
        # Given: I_x = 1.5 I0, I_y = I0  (from problem statement)
        I_x = 1.5 * self.I0
        I_y = self.I0
        total_incident = I_x + I_y   # = 2.5 I0
        # After QWP: I_max = total_incident - a  (since a = unpolarized part)
        a = total_incident - self.I_max
        # Unpolarized fraction = 2*a / total_incident
        derived_fraction = 2.0 * a / total_incident
        if not math.isclose(derived_fraction, self.f_unpol, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Derived fraction {derived_fraction} does not match f_unpol = {self.f_unpol}"
        return True, "Unpolarised fraction is consistent with total intensity and I_max"

    def test_modeling_decreasing_behavior(self):
        """Check that I(θ) is decreasing on [0, π/2] (modeling sanity)."""
        if self.I_theta is None:
            return False, "SKIP: I_theta not available"
        at0  = self._eval_I_theta(0.0)
        at45 = self._eval_I_theta(math.pi / 4.0)
        at90 = self._eval_I_theta(math.pi / 2.0)
        if None in (at0, at45, at90):
            return False, "Cannot evaluate I_theta at required angles"
        if at0 < at45 - 1e-9 or at45 < at90 - 1e-9:
            return False, "I(θ) is not strictly decreasing on the interval"
        return True, "I(θ) is decreasing on [0, π/2]"

    # ------------------------------------------------------------------
    # Monotonicity test (must involve explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_intensity_decreases_with_theta(self):
        """
        Perturb θ by a small positive amount Δθ = 0.1 rad from 0,
        verify that the intensity decreases.
        """
        if self.I_theta is None:
            return False, "SKIP: I_theta not available"
        baseline = self._eval_I_theta(0.0)
        delta = 0.1  # rad
        perturbed = self._eval_I_theta(delta)
        if baseline is None or perturbed is None:
            return False, "Cannot evaluate I_theta at required angles"
        if perturbed >= baseline + 1e-9:
            return False, f"Intensity did not decrease after perturbing θ by +{delta} rad: baseline={baseline}, perturbed={perturbed}"
        return True, f"Intensity decreases when θ is increased from 0 to {delta} rad"

    # ------------------------------------------------------------------
    # Weak answer‑sanity (only one such method allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_f_unpol_range(self):
        """Unpolarised fraction must be between 0 and 1."""
        if self.f_unpol is None:
            return False, "SKIP: f_unpol not available"
        if self.f_unpol < 0 or self.f_unpol > 1:
            return False, f"f_unpol={self.f_unpol} is outside [0,1]"
        return True, "f_unpol lies within the physical range"

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
        return all_pass, logs