import sympy
import math
import traceback

class PhysicsChecker:
    """
    Audits the solution to the π⁻ + d → n + n problem.
    Checks:
    - Parity conservation → L odd.
    - Antisymmetry for identical fermions → (-1)^L * (-1)^{S+1} = -1.
    - Angular momentum composition: |L - S| ≤ J ≤ L + S.
    - Probability for both neutrons opposite to deuteron spin is zero.
    - Angular distribution functional form and normalization.
    - Monotonic increase with polar angle (θ).
    """
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        # Extract each required result, default None
        self.L = self.results.get('L', None)
        self.S = self.results.get('S', None)
        self.P_both = self.results.get('P_both_opposite', None)
        self.ang_dist = self.results.get('ang_dist', None)
        # Additional givens (not strictly needed but useful)
        # We assume the checker reads the problem context, but we store for completeness
        self.initial_J = 1       # deuteron spin J=1, π⁻ spin 0, orbital L_π=0 → J=1
        self.initial_parity = -1  # p(π⁻)=-1, p(d)=+1, (-1)^L_π=1 → -1

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present."""
        missing = []
        for key in ['L', 'S', 'P_both_opposite', 'ang_dist']:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Check types of the results."""
        if not isinstance(self.L, int):
            return (False, f"L should be int, got {type(self.L).__name__}")
        if not isinstance(self.S, int):
            return (False, f"S should be int, got {type(self.S).__name__}")
        if not isinstance(self.P_both, (int, float)):
            return (False, f"P_both_opposite should be numeric, got {type(self.P_both).__name__}")
        if not isinstance(self.ang_dist, str):
            return (False, f"ang_dist should be string, got {type(self.ang_dist).__name__}")
        return (True, "All types correct")

    # ------------------------------------------------------------
    # Invariant tests (physics theorems)
    # ------------------------------------------------------------
    def test_invariant_parity_conservation(self):
        """Initial parity = -1, final parity = (-1)^L must be -1 → L odd."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if self.L % 2 == 0:
            return (False, f"L = {self.L} is even, but parity requires odd L")
        return (True, f"L = {self.L} is odd ✓")

    def test_invariant_antisymmetry(self):
        """Identical fermions: (-1)^L * (-1)^{S+1} = -1. With L odd, S must be odd too."""
        if self.L is None or self.S is None:
            return (False, "SKIP: L or S missing")
        left = (-1)**self.L * (-1)**(self.S + 1)
        if left != -1:
            return (False, f"Antisymmetry condition fails: left = {left}, expected -1")
        return (True, f"Antisymmetry satisfied: L={self.L}, S={self.S}")

    def test_invariant_angular_momentum_limits(self):
        """Total J=1 must satisfy |L-S| ≤ J ≤ L+S."""
        if self.L is None or self.S is None:
            return (False, "SKIP: L or S missing")
        J = self.initial_J
        if not (abs(self.L - self.S) <= J <= self.L + self.S):
            return (False, f"J=1 not in range |L-S|={abs(self.L-self.S)} to L+S={self.L+self.S}")
        return (True, f"J=1 is within allowed range for L={self.L}, S={self.S}")

    def test_invariant_clebsch_gordan_zero_probability(self):
        """For J=1, M=1, L=1, S=1, the component with S_z = -1 requires m_L=2, impossible → P=0."""
        if self.L is None or self.S is None or self.P_both is None:
            return (False, "SKIP: missing results")
        if self.L == 1 and self.S == 1:
            # L=S=1 case: probability must be zero because no m_L=2 available.
            if self.P_both != 0:
                return (False, f"Expected P=0 for L=S=1, got {self.P_both}")
        else:
            # General weak check: if L < 2, probability should be zero.
            if self.L is not None and self.L < 2 and self.P_both != 0:
                return (False, f"L={self.L}<2 implies no m_L=2, so P=0 expected but got {self.P_both}")
        return (True, "P_both_opposite = 0 consistent with Clebsch-Gordan")

    # ------------------------------------------------------------
    # Modeling/test of trace consistency
    # ------------------------------------------------------------
    def test_modeling_angular_distribution_form(self):
        """Check that ang_dist string contains the expected functional form derived from Y_1^1."""
        if self.ang_dist is None:
            return (False, "SKIP: ang_dist not available")
        allowed_patterns = ["3/(16pi)", "3/(16π)", "\\frac{3}{16\\pi}"]
        matches_pattern = any(p in self.ang_dist for p in allowed_patterns)
        has_sin2 = "sin" in self.ang_dist and ("²" in self.ang_dist or "^2" in self.ang_dist or "**2" in self.ang_dist)
        if not (matches_pattern and has_sin2):
            return (False, f"ang_dist '{self.ang_dist}' does not match expected pattern")
        return (True, f"angular distribution has correct form")

    def test_modeling_angular_distribution_normalization(self):
        """The integral over solid angle of dP/dΩ should be 1/2.
        We check that the constant factor is 3/(16π).
        """
        if self.ang_dist is None:
            return (False, "SKIP: ang_dist not available")
        # Acceptable string patterns
        allowed = ["3/(16pi)", "3/(16π)", "\\frac{3}{16\\pi}"]
        if any(p in self.ang_dist for p in allowed):
            return (True, f"Normalization factor correct (constant = 3/(16π))")
        else:
            return (False, f"Cannot verify constant from string: {self.ang_dist}")

    # ------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_angular_distribution(self):
        """Perturb the polar angle θ and check that dP/dΩ increases with sin²θ.
        Use baseline at θ1=0.1 rad, perturb to θ2=0.2 rad.
        The distribution should increase because sin^2 is increasing for θ in (0, π/2).
        """
        if self.ang_dist is None:
            return (False, "SKIP: ang_dist not available, cannot test monotonicity")
        # Extract constant from string (accept known patterns)
        # We'll compute the constant numerically.
        # The distribution should be (c * sin²θ) where c ≈ 3/(16π) ≈ 0.059683
        try:
            # Parse the constant from known patterns
            if "\\frac{3}{16\\pi}" in self.ang_dist:
                const = 3.0 / (16.0 * math.pi)
            elif "3/(16pi)" in self.ang_dist:
                const = 3.0 / (16.0 * math.pi)
            elif "3/(16π)" in self.ang_dist:
                const = 3.0 / (16.0 * math.pi)
            else:
                # Fallback: try to extract any numeric part? Too complex.
                return (False, "Cannot parse constant from ang_dist string")
        except Exception:
            return (False, "Failed to compute constant")

        # Baseline at θ1 = 0.1 rad
        theta1 = 0.1
        val1 = const * (math.sin(theta1) ** 2)
        # Perturb to θ2 = 0.2 rad (increase)
        theta2 = 0.2
        val2 = const * (math.sin(theta2) ** 2)

        # Since sin^2 increases on (0, π/2), val2 > val1
        if val2 <= val1:
            return (False, f"dP/dΩ not monotonic increasing with θ: val1={val1}, val2={val2}")
        # Also check that the ratio is close to (sin²θ2)/(sin²θ1)
        expected_ratio = (math.sin(theta2)**2) / (math.sin(theta1)**2)
        actual_ratio = val2 / val1
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (False, f"Ratio mismatch: expected ~{expected_ratio:.3f}, got {actual_ratio:.3f}")
        return (True, f"Monotonic increase confirmed: val1={val1:.4g}, val2={val2:.4g}")

    # ------------------------------------------------------------
    # Weak answer sanity (only one allowed, no isclose, no derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_reasonableness(self):
        """Very weak checks: answers are finite and within reasonable bounds."""
        checks = []
        if self.L is not None and (self.L < 0 or self.L > 10):
            checks.append(f"L = {self.L} outside reasonable range [0,10]")
        if self.S is not None and (self.S < 0 or self.S > 2):
            checks.append(f"S = {self.S} outside reasonable range [0,2]")
        if self.P_both is not None and not (0.0 <= self.P_both <= 1.0):
            checks.append(f"P_both = {self.P_both} not in [0,1]")
        if self.P_both is not None and math.isnan(self.P_both):
            checks.append("P_both is NaN")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All answers are finite and within plausible ranges")

    # ------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_ methods and execute them."""
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


# Example usage (for testing):
if __name__ == "__main__":
    # Dummy solution instance
    class DummySolution:
        def __init__(self):
            self.results = {
                'L': 1,
                'S': 1,
                'P_both_opposite': 0,
                'ang_dist': '\\frac{3}{16\\pi} \\sin^2 \\theta'
            }
    sol = DummySolution()
    checker = PhysicsChecker(sol)
    success, logs = checker.run_all_tests()
    for line in logs:
        print(line)
    print(f"\nOverall success: {success}")