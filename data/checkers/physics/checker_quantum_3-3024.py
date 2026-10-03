import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.P_plus = results.get('P_plus', None)
        self.P_minus = results.get('P_minus', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.t = getattr(solution_instance, 't', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------- INTERFACE TESTS ----------
    def test_interface_has_P_plus(self):
        if self.P_plus is None:
            return (False, "Missing 'P_plus' in results")
        return (True, "P_plus present")

    def test_interface_has_P_minus(self):
        if self.P_minus is None:
            return (False, "Missing 'P_minus' in results")
        return (True, "P_minus present")

    def test_interface_P_plus_is_real(self):
        if self.P_plus is None:
            return (False, "SKIP: P_plus not available")
        if isinstance(self.P_plus, complex):
            return (False, f"P_plus is complex: {self.P_plus}")
        return (True, "P_plus is real")

    def test_interface_P_minus_is_real(self):
        if self.P_minus is None:
            return (False, "SKIP: P_minus not available")
        if isinstance(self.P_minus, complex):
            return (False, f"P_minus is complex: {self.P_minus}")
        return (True, "P_minus is real")

    # ---------- INVARIANT TESTS (theorem-based physical sanity) ----------
    def test_invariant_normalization(self):
        if self.P_plus is None or self.P_minus is None:
            return (False, "SKIP: probabilities missing")
        try:
            total = self.P_plus + self.P_minus
        except TypeError:
            return (False, "Cannot sum probabilities (non-numeric)")
        if math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_plus + P_minus = {total} ≈ 1")
        else:
            return (False, f"P_plus + P_minus = {total}, expected ~1")

    def test_invariant_symmetry_around_half(self):
        if self.P_plus is None or self.P_minus is None:
            return (False, "SKIP: probabilities missing")
        try:
            dev_plus = self.P_plus - 0.5
            dev_minus = self.P_minus - 0.5
        except TypeError:
            return (False, "Non-numeric probabilities")
        if math.isclose(dev_plus, -dev_minus, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"(P_plus-0.5)={dev_plus}, -(P_minus-0.5)={-dev_minus}, symmetric")
        else:
            return (False, f"Symmetry broken: (P_plus-0.5)={dev_plus}, -(P_minus-0.5)={-dev_minus}")

    # ---------- MODELING / TRACE QUALITY TESTS (at least two) ----------
    def test_modeling_sum_to_one(self):
        """Check fundamental probability conservation, a direct consequence of unitary evolution."""
        if self.P_plus is None or self.P_minus is None:
            return (False, "SKIP: probabilities missing")
        try:
            total = self.P_plus + self.P_minus
        except TypeError:
            return (False, "Non-numeric probabilities")
        if math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_plus + P_minus = {total} ≈ 1 (conservation of probability)")
        else:
            return (False, f"Sum = {total}, expected 1")

    def test_modeling_probability_bounds(self):
        """Each probability must lie in [0,1] to be physically meaningful."""
        if self.P_plus is None or self.P_minus is None:
            return (False, "SKIP: probabilities missing")
        if not (0 <= self.P_plus <= 1):
            return (False, f"P_plus = {self.P_plus} outside [0,1]")
        if not (0 <= self.P_minus <= 1):
            return (False, f"P_minus = {self.P_minus} outside [0,1]")
        return (True, f"Both probabilities in [0,1]")

    # ---------- MONOTONICITY TEST ----------
    def test_monotonicity_P_plus_with_t(self):
        """Perturb t by +10% and verify that the change in P_plus has the same sign as the derivative."""
        if self.omega is None or self.t is None or self.P_plus is None:
            return (False, "SKIP: omega, t, or P_plus missing")
        try:
            omega = float(self.omega)
            t = float(self.t)
            P_plus_original = float(self.P_plus)
        except (TypeError, ValueError):
            return (False, "Cannot convert to float")

        # Baseline using the derived formula: P_plus = 0.5*(1 + sin(2ωt)) with ω = μ0B0/(2ħ)
        # This matches the explanation steps final expression.
        baseline = 0.5 * (1 + math.sin(2 * omega * t))
        if not math.isclose(P_plus_original, baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Original P_plus={P_plus_original} deviates from baseline formula {baseline}")

        # Perturb t by +10%
        t_perturbed = t * 1.10
        perturbed = 0.5 * (1 + math.sin(2 * omega * t_perturbed))

        # Derivative sign: d(P_plus)/dt = ω * cos(2ωt)
        derivative = omega * math.cos(2 * omega * t)
        if abs(derivative) < 1e-9:
            return (False, "SKIP: derivative zero – test inconclusive")

        delta = perturbed - baseline
        if (derivative > 0 and delta > 0) or (derivative < 0 and delta < 0):
            return (True, f"Monotonicity consistent: derivative={derivative:.4f}, change={delta:.4f}")
        else:
            return (False, f"Monotonicity violation: derivative={derivative:.4f}, change={delta:.4f}")

    # ---------- WEAK ANSWER SANITY (at most one) ----------
    def test_answer_sanity_bounds(self):
        if self.P_plus is None or self.P_minus is None:
            return (False, "SKIP: probabilities missing")
        if math.isnan(self.P_plus) or math.isnan(self.P_minus):
            return (False, "Probability is NaN")
        if math.isinf(self.P_plus) or math.isinf(self.P_minus):
            return (False, "Probability is infinite")
        return (True, "Probabilities are finite numbers")

    # ---------- RUN ALL ----------
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