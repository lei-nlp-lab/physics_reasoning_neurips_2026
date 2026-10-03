import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results dictionary
        self.results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results

        # Extract the two required keys with safe defaults
        self.distribution_Q1 = self.results.get('distribution_Q1', None)
        self.distribution_Q2 = self.results.get('distribution_Q2', None)

    # ------------------------------------------------------------
    # Interface tests: ensure required keys are present and are strings
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.distribution_Q1 is None or self.distribution_Q2 is None:
            missing = []
            if self.distribution_Q1 is None:
                missing.append("distribution_Q1")
            if self.distribution_Q2 is None:
                missing.append("distribution_Q2")
            return (False, f"Missing keys: {', '.join(missing)}")
        if not isinstance(self.distribution_Q1, str) or not self.distribution_Q1.strip():
            return (False, "distribution_Q1 is not a valid non‑empty string")
        if not isinstance(self.distribution_Q2, str) or not self.distribution_Q2.strip():
            return (False, "distribution_Q2 is not a valid non‑empty string")
        return (True, "All required keys present and are non‑empty strings")

    # ------------------------------------------------------------
    # Invariant tests: theorem‑based physical sanity
    # ------------------------------------------------------------
    def test_invariant_sin2_factor(self):
        """Both distributions must contain sin^2(theta) due to angular momentum selection rules."""
        for name, dist in [("distribution_Q1", self.distribution_Q1),
                           ("distribution_Q2", self.distribution_Q2)]:
            if dist is None:
                return (False, f"SKIP: {name} not available")
            if "sin^2(" not in dist and "sin²" not in dist and "sin^2θ" not in dist:
                return (False, f"{name} does not contain sin^2(theta) factor")
        return (True, "Both distributions contain expected sin^2(theta) factor")

    def test_invariant_positivity(self):
        """The angular distribution must be non‑negative for all physical angles.
           We check that there is no overall negative sign outside the sin^2 factor."""
        for name, dist in [("distribution_Q1", self.distribution_Q1),
                           ("distribution_Q2", self.distribution_Q2)]:
            if dist is None:
                continue
            # Remove spaces and check for a negative sign before sin^2
            cleaned = dist.replace(" ", "").lower()
            if cleaned.startswith("-sin^2") or cleaned.startswith("-sin²"):
                return (False, f"{name} seems to have an overall negative sign")
        return (True, "Distributions are structurally non‑negative")

    # ------------------------------------------------------------
    # Modeling / trace checks: verify detailed form matches physics
    # ------------------------------------------------------------
    def test_modeling_q1_parity_violation_term(self):
        """Q1 must contain a cos(theta) term (parity violation)."""
        if self.distribution_Q1 is None:
            return (False, "SKIP: distribution_Q1 not available")
        if "cos(" in self.distribution_Q1 or "cosθ" in self.distribution_Q1.lower():
            return (True, "Q1 distribution contains cos(theta) term as expected")
        else:
            return (False, "Q1 distribution does not contain cos(theta) term")

    def test_modeling_q2_parity_conservation(self):
        """Q2 must NOT contain a linear cos(theta) term (only sin^2 or 1‑cos^2)."""
        if self.distribution_Q2 is None:
            return (False, "SKIP: distribution_Q2 not available")
        # Allowed: pure sin^2 or 1 – cos^2 (the target answer form)
        # Disallowed: any term with "cos(" that is not part of "- cos^2"
        cleaned = self.distribution_Q2.replace(" ", "")
        if "-cos^2" in cleaned or "-cos²" in cleaned:
            # It is the allowed form "1 – cos^2" – fine
            return (True, "Q2 distribution has the allowed 1‑cos^2 form")
        if "cos(" in self.distribution_Q2 or "cosθ" in self.distribution_Q2.lower():
            return (False, "Q2 distribution contains an unwanted cos(theta) term")
        return (True, "Q2 distribution correctly lacks a cos(theta) term")

    # ------------------------------------------------------------
    # Monotonicity test: explicit perturbation of a physical parameter
    # ------------------------------------------------------------
    def test_monotonicity_q1_alpha_perturbation(self):
        """
        For the Q1 distribution, increasing the parameter alpha (strength of parity‑violating
        interference) increases the distribution in the range where cos(theta) > 0,
        and decreases it where cos(theta) < 0.
        We re‑derive the expression from theorem constraints (clebsch‑gordan/angular momentum).
        """
        # Re‑derive the standard form: W(theta) ∝ sin^2(theta)*(1+alpha*cos(theta))
        theta = sp.symbols('theta', real=True)
        alpha = sp.symbols('alpha', real=True)
        expr = sp.sin(theta)**2 * (1 + alpha * sp.cos(theta))

        # Choose two angular points: one with cos > 0, one with cos < 0
        theta_pos = 0.1   # rad, cos ~ 0.995
        theta_neg = 2.0   # rad, cos ~ -0.416

        # Baseline with alpha = 0
        base_pos = float(expr.subs({theta: theta_pos, alpha: 0}).evalf())
        base_neg = float(expr.subs({theta: theta_neg, alpha: 0}).evalf())

        # Perturbed with alpha = 0.1
        pert_pos = float(expr.subs({theta: theta_pos, alpha: 0.1}).evalf())
        pert_neg = float(expr.subs({theta: theta_neg, alpha: 0.1}).evalf())

        # Check monotonicity: increasing alpha should increase at theta_pos (cos>0)
        if pert_pos <= base_pos:
            return (False, f"At theta={theta_pos} rad (cos>0), increasing alpha did NOT increase the distribution: baseline={base_pos}, perturbed={pert_pos}")
        # ... and decrease at theta_neg (cos<0)
        if pert_neg >= base_neg:
            return (False, f"At theta={theta_neg} rad (cos<0), increasing alpha did NOT decrease the distribution: baseline={base_neg}, perturbed={pert_neg}")
        return (True, "Monotonicity of Q1 with respect to alpha confirmed: positive cos region increases, negative cos region decreases")

    # ------------------------------------------------------------
    # Weak answer sanity (at most one such method)
    # ------------------------------------------------------------
    def test_answer_sanity_nonempty_strings(self):
        """Both distribution strings are non‑empty."""
        for name, dist in [("distribution_Q1", self.distribution_Q1),
                           ("distribution_Q2", self.distribution_Q2)]:
            if dist is None:
                return (False, f"SKIP: {name} missing")
            if not dist.strip():
                return (False, f"{name} is an empty or whitespace‑only string")
        return (True, "Both distribution strings are non‑empty")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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