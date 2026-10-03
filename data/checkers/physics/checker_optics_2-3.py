import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Always defaults to None
        self.lambda_ = None
        self.n = None
        self.delta_m = None
        self.t = None
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Extract givens from dictionary (most reliable, avoids keyword conflict)
        givens = getattr(solution_instance, 'givens', {})
        if isinstance(givens, dict):
            # wavelength
            if 'lambda' in givens and givens['lambda'] is not None:
                self.lambda_ = float(givens['lambda'])
            elif 'wavelength' in givens and givens['wavelength'] is not None:
                self.lambda_ = float(givens['wavelength'])

            # refractive index
            if 'n' in givens and givens['n'] is not None:
                self.n = float(givens['n'])

            # fringe shift (delta m)
            if 'delta_m' in givens and givens['delta_m'] is not None:
                self.delta_m = int(givens['delta_m'])
            elif 'fringe_shift' in givens and givens['fringe_shift'] is not None:
                self.delta_m = int(givens['fringe_shift'])

        # Extract results from dictionary
        results = getattr(solution_instance, 'results', {})
        if isinstance(results, dict):
            if 't' in results and results['t'] is not None:
                self.t = float(results['t'])

        # Pre‑compute derived thickness from givens if possible
        self.derived_t = None
        if (self.lambda_ is not None and self.n is not None and self.delta_m is not None):
            try:
                self.derived_t = (self.delta_m * self.lambda_) / (self.n - 1.0)
            except (ZeroDivisionError, TypeError):
                self.derived_t = None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_t_in_results(self):
        if self.t is None:
            return (False, "Missing 't' in results")
        if not isinstance(self.t, (int, float)):
            return (False, f"'t' is not numeric, got {type(self.t)}")
        return (True, "Film thickness t is present and numeric")

    def test_interface_givens_available(self):
        if self.lambda_ is None:
            return (False, "Missing wavelength data (lambda). Cannot verify physics.")
        if self.n is None:
            return (False, "Missing refractive index n. Cannot verify physics.")
        return (True, "All essential givens present")

    # ------------------------------------------------------------------
    # Physics invariants (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_optical_path_condition(self):
        if self.lambda_ is None or self.n is None or self.t is None or self.delta_m is None:
            return (False, "SKIP: insufficient data to check optical path condition")
        left = (self.n - 1.0) * self.t
        right = self.delta_m * self.lambda_
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"(n-1)*t = {left:.2e}, δm*λ = {right:.2e} (within 1%)")
        else:
            return (False, f"Mismatch: (n-1)*t = {left:.2e}, expected {right:.2e}")

    def test_invariant_t_consistency_with_givens(self):
        if self.derived_t is None or self.t is None:
            return (False, "SKIP: cannot derive t from givens or t missing")
        if math.isclose(self.derived_t, self.t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t from results ({self.t:.4e}) matches derived t ({self.derived_t:.4e})")
        else:
            return (False, f"t mismatch: results t={self.t:.4e}, derived t={self.derived_t:.4e}")

    # ------------------------------------------------------------------
    # Modeling / trace checks (at least two)
    # ------------------------------------------------------------------
    def test_modeling_explanation_contains_relation(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = str(self.explanation_steps).lower()
        patterns = [
            "t(n-1)=4λ",
            "t = 4λ/(n-1)",
            "(n-1)t = 4λ",
            "t(n-1)=4lambda",
            "delta m = 4"
        ]
        for pat in patterns:
            if pat.lower() in text:
                return (True, "Explanation steps contain the key optical path relation")
        return (False, "Explanation steps do not mention the required equation")

    def test_modeling_pseudocode_contains_equation(self):
        if self.pseudocode is None:
            return (False, "SKIP: no pseudocode provided")
        text = str(self.pseudocode).lower()
        # Look for the core equation (n-1)*t = 4*lambda or similar
        key_phrases = ["(n-1)*t = 4*lambda", "t = 4*lambda/(n-1)", "n-1)*t = 4*lambda"]
        for phrase in key_phrases:
            if phrase in text:
                return (True, "Pseudocode includes the essential equation for t")
        return (False, "Pseudocode does not contain the key equation")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_t_with_n(self):
        if self.lambda_ is None or self.n is None or self.delta_m is None:
            return (False, "SKIP: missing givens for monotonicity")
        baseline_t = self.derived_t
        if baseline_t is None:
            return (False, "SKIP: cannot compute baseline t")
        perturbed_n = self.n + 0.1
        try:
            perturbed_t = (self.delta_m * self.lambda_) / (perturbed_n - 1.0)
        except ZeroDivisionError:
            return (False, "Perturbed n would make n-1=0, skip")
        if perturbed_t < baseline_t - 1e-12:
            return (True, f"Increase of n by 0.1 reduces t from {baseline_t:.4e} to {perturbed_t:.4e}")
        else:
            return (False, f"Monotonicity violated: t did not decrease after n increase (original {baseline_t:.4e}, perturbed {perturbed_t:.4e})")

    def test_monotonicity_t_with_lambda(self):
        if self.lambda_ is None or self.n is None or self.delta_m is None:
            return (False, "SKIP: missing givens for monotonicity")
        baseline_t = self.derived_t
        if baseline_t is None:
            return (False, "SKIP: cannot compute baseline t")
        perturbed_lambda = self.lambda_ * 1.1
        try:
            perturbed_t = (self.delta_m * perturbed_lambda) / (self.n - 1.0)
        except ZeroDivisionError:
            return (False, "Division by zero in perturbed t calculation")
        if perturbed_t > baseline_t + 1e-12:
            return (True, f"Increase of λ by 10% increases t from {baseline_t:.4e} to {perturbed_t:.4e}")
        else:
            return (False, f"Monotonicity violated: t did not increase with λ (original {baseline_t:.4e}, perturbed {perturbed_t:.4e})")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        if self.t is None:
            return (False, "Cannot check answer sanity: t missing")
        if not math.isfinite(self.t):
            return (False, f"Thickness is not finite: {self.t}")
        if self.t <= 0:
            return (False, f"Thickness is not positive: {self.t}")
        # Broad plausible range: 0.1 µm to 1 mm
        if self.t < 1e-7 or self.t > 1e-3:
            return (False, f"Thickness out of plausible range: {self.t:.4e}")
        return (True, "Thickness is positive, finite, and within plausible range")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        method_names = [name for name in dir(self) if name.startswith("test_")]
        method_names.sort()
        for name in method_names:
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED with exception: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)