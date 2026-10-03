import math
from typing import Any, Tuple, List

class PhysicsChecker:
    """Audits the physics solution for the critical length L0 of a uranium cube."""

    def __init__(self, solution_instance: Any):
        # Extract givens with None defaults
        givens = getattr(solution_instance, 'givens', {})
        self.A = givens.get('A', None)
        self.B = givens.get('B', None)

        # Extract results with None defaults
        results = getattr(solution_instance, 'results', {})
        self.L0 = results.get('L0', None)

        # Extract text for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    def _theoretical_L0(self, A: float, B: float) -> float:
        """Compute L0 from the formula L0 = π sqrt(3A/B)."""
        return math.pi * math.sqrt(3.0 * A / B)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        missing = []
        if self.A is None:
            missing.append("A")
        if self.B is None:
            missing.append("B")
        if self.L0 is None:
            missing.append("L0")
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_A_positive(self) -> Tuple[bool, str]:
        if self.A is None:
            return (False, "SKIP: A not available")
        if not isinstance(self.A, (int, float)):
            return (False, f"A has invalid type {type(self.A)}")
        if self.A <= 0:
            return (False, f"A = {self.A} is not positive")
        return (True, f"A = {self.A} > 0")

    def test_interface_B_positive(self) -> Tuple[bool, str]:
        if self.B is None:
            return (False, "SKIP: B not available")
        if not isinstance(self.B, (int, float)):
            return (False, f"B has invalid type {type(self.B)}")
        if self.B <= 0:
            return (False, f"B = {self.B} is not positive")
        return (True, f"B = {self.B} > 0")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_critical_condition(self) -> Tuple[bool, str]:
        """Check λ = B - A*(π/L0)^2 * 3 ≈ 0."""
        if None in (self.A, self.B, self.L0):
            return (False, "SKIP: A, B, or L0 missing")
        if self.L0 <= 0:
            return (False, f"L0 = {self.L0} is not positive")
        k2 = (math.pi / self.L0) ** 2 * 3.0
        lam = self.B - self.A * k2
        if not math.isfinite(lam):
            return (False, f"lambda is not finite: {lam}")
        if math.isclose(lam, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Critical condition holds: λ = {lam:.3e} ≈ 0")
        return (False, f"Critical condition fails: λ = {lam:.3e}")

    def test_invariant_eigenvalue_relation(self) -> Tuple[bool, str]:
        """Check B/A = 3*(π/L0)^2."""
        if None in (self.A, self.B, self.L0):
            return (False, "SKIP: A, B, or L0 missing")
        if self.A <= 0 or self.L0 <= 0:
            return (False, "A or L0 not positive")
        expected = self.B / self.A
        computed = 3.0 * (math.pi / self.L0) ** 2
        if not math.isfinite(expected) or not math.isfinite(computed):
            return (False, "Non‑finite eigenvalue component")
        if math.isclose(expected, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Eigenvalue relation holds: B/A = {expected:.4f}, 3(π/L0)^2 = {computed:.4f}")
        return (False, f"Eigenvalue mismatch: B/A = {expected:.4f}, 3(π/L0)^2 = {computed:.4f}")

    # ------------------------------------------------------------------
    # Modeling / trace checks (quality of explanation)
    # ------------------------------------------------------------------
    def test_modeling_separation_of_variables(self) -> Tuple[bool, str]:
        """Check that separation of variables is mentioned in explanation/pseudocode."""
        text = str(self.explanation_steps or '') + str(self.pseudocode or '')
        if not text:
            return (False, "SKIP: No explanation steps or pseudocode")
        keywords = ["separation", "variable", "X(x)", "sin", "eigenfunction"]
        found = [kw for kw in keywords if kw.lower() in text.lower()]
        if found:
            return (True, f"Found modeling keywords: {found}")
        return (False, "No evidence of separation of variables")

    def test_modeling_fundamental_mode(self) -> Tuple[bool, str]:
        """Check that the solution uses the fundamental mode (1,1,1)."""
        text = str(self.explanation_steps or '') + str(self.pseudocode or '')
        if not text:
            return (False, "SKIP: No explanation steps or pseudocode")
        # Look for explicit mention of n_x = 1 or sum of squares = 3
        patterns = ["n_x = 1", "n_y = 1", "n_z = 1", "sum of squares = 3",
                    "n_x^2 + n_y^2 + n_z^2 = 3", "n_x=n_y=n_z=1"]
        found = [p for p in patterns if p.lower() in text.lower()]
        if found:
            return (True, f"Fundamental mode referenced: {found}")
        return (False, "No explicit reference to fundamental mode (1,1,1)")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of input, compare outputs)
    # ------------------------------------------------------------------
    def test_monotonicity_L0_increases_with_A(self) -> Tuple[bool, str]:
        """Perturb A by +1%, check that theoretical L0 increases."""
        if self.A is None or self.B is None:
            return (False, "SKIP: A or B not available")
        if self.A <= 0 or self.B <= 0:
            return (False, "A and B must be positive")
        baseline = self._theoretical_L0(self.A, self.B)
        # Perturb A by +1%
        perturbed = self._theoretical_L0(self.A * 1.01, self.B)
        if perturbed > baseline:
            return (True, f"Perturbed A by +1%: L0 goes from {baseline:.4f} to {perturbed:.4f} (increase)")
        return (False, f"L0 did not increase: baseline {baseline:.4f}, perturbed {perturbed:.4f}")

    def test_monotonicity_L0_decreases_with_B(self) -> Tuple[bool, str]:
        """Perturb B by +1%, check that theoretical L0 decreases."""
        if self.A is None or self.B is None:
            return (False, "SKIP: A or B not available")
        if self.A <= 0 or self.B <= 0:
            return (False, "A and B must be positive")
        baseline = self._theoretical_L0(self.A, self.B)
        # Perturb B by +1%
        perturbed = self._theoretical_L0(self.A, self.B * 1.01)
        if perturbed < baseline:
            return (True, f"Perturbed B by +1%: L0 goes from {baseline:.4f} to {perturbed:.4f} (decrease)")
        return (False, f"L0 did not decrease: baseline {baseline:.4f}, perturbed {perturbed:.4f}")

    # ------------------------------------------------------------------
    # Single weak answer sanity (only sign and finiteness)
    # ------------------------------------------------------------------
    def test_answer_sanity_L0_positive_finite(self) -> Tuple[bool, str]:
        """Check L0 is positive and finite. No numeric target."""
        if self.L0 is None:
            return (False, "SKIP: L0 not available")
        if not isinstance(self.L0, (int, float)):
            return (False, f"L0 has invalid type {type(self.L0)}")
        if math.isnan(self.L0) or math.isinf(self.L0):
            return (False, "L0 is NaN or infinite")
        if self.L0 <= 0:
            return (False, f"L0 = {self.L0} is not positive")
        return (True, f"L0 = {self.L0} is positive and finite")

    # ------------------------------------------------------------------
    # Runner (discover all test_ methods, wrap in try/except)
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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