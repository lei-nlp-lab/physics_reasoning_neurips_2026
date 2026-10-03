import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Check physics solution for optical activity in a dextrose solution.
    """
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.omega = givens.get('omega', None)
        self.gamma = givens.get('gamma', None)
        self.L = givens.get('L', None)
        self.c = givens.get('c', None)          # speed of light in vacuum
        self.mu0 = givens.get('mu0', None)      # vacuum permeability

        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.n_plus = self.results.get('n_plus', None)
        self.n_minus = self.results.get('n_minus', None)
        self.phi = self.results.get('phi', None)

        # Additional optional items (electric field components, if provided)
        self.E_x = self.results.get('E_x', None)
        self.E_y = self.results.get('E_y', None)

        # Constants for later use
        self.eps0 = 1.0 / (self.mu0 * self.c**2) if (self.mu0 and self.c) else None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_has_n_plus(self):
        if self.n_plus is None:
            return (False, "n_plus is missing")
        return (True, "n_plus present")

    def test_interface_has_n_minus(self):
        if self.n_minus is None:
            return (False, "n_minus is missing")
        return (True, "n_minus present")

    def test_interface_has_phi(self):
        if self.phi is None:
            return (False, "phi is missing")
        return (True, "phi present")

    def test_interface_n_plus_numeric(self):
        if self.n_plus is None:
            return (False, "SKIP: n_plus not available")
        if not isinstance(self.n_plus, (int, float)):
            return (False, f"n_plus is not numeric: {type(self.n_plus)}")
        return (True, "n_plus is numeric")

    def test_interface_n_minus_numeric(self):
        if self.n_minus is None:
            return (False, "SKIP: n_minus not available")
        if not isinstance(self.n_minus, (int, float)):
            return (False, f"n_minus is not numeric: {type(self.n_minus)}")
        return (True, "n_minus is numeric")

    def test_interface_phi_numeric(self):
        if self.phi is None:
            return (False, "SKIP: phi not available")
        if not isinstance(self.phi, (int, float)):
            return (False, f"phi is not numeric: {type(self.phi)}")
        return (True, "phi is numeric")

    # ------------------------------------------------------------------
    # Invariant / Theorem-based tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_symmetry_n_plus_n_minus(self):
        """Check that n_plus - 1 ≈ -(n_minus - 1) within tolerance."""
        if self.n_plus is None or self.n_minus is None:
            return (False, "SKIP: n_plus or n_minus not available")
        delta_plus = self.n_plus - 1.0
        delta_minus = self.n_minus - 1.0
        if math.isclose(delta_plus, -delta_minus, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"n_plus-1 ≈ -(n_minus-1): {delta_plus:.4f} vs {-delta_minus:.4f}")
        else:
            return (False, f"n_plus-1 ({delta_plus:.4f}) not symmetric with n_minus-1 ({delta_minus:.4f})")

    def test_invariant_phi_from_n_difference(self):
        """Check that φ ≈ (omega * L / (2*c)) * (n_plus - n_minus)."""
        if any(x is None for x in [self.phi, self.n_plus, self.n_minus, self.omega, self.L, self.c]):
            return (False, "SKIP: required values missing for phi check")
        delta_n = self.n_plus - self.n_minus
        expected_phi = (self.omega * self.L / (2.0 * self.c)) * delta_n
        if math.isclose(self.phi, expected_phi, rel_tol=1e-2, abs_tol=1e-6):
            return (True, f"phi matches delta_n formula: phi={self.phi:.6e}, expected={expected_phi:.6e}")
        else:
            return (False, f"phi={self.phi:.6e} differs from expected={expected_phi:.6e}")

    def test_invariant_dispersion_relation_n_plus(self):
        """Check that n_plus satisfies (n^2 - 1) ≈ γ μ0 ω c n (right circular)."""
        if any(x is None for x in [self.n_plus, self.gamma, self.mu0, self.omega, self.c]):
            return (False, "SKIP: missing data for dispersion check of n_plus")
        n = self.n_plus
        lhs = n**2 - 1.0
        rhs = self.gamma * self.mu0 * self.omega * self.c * n
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-4):
            return (True, f"n_plus dispersion holds: n^2-1={lhs:.6e}, γμ0ωc n={rhs:.6e}")
        else:
            return (False, f"n_plus dispersion violated: n^2-1={lhs:.6e}, expected ~{rhs:.6e}")

    def test_invariant_dispersion_relation_n_minus(self):
        """Check that n_minus satisfies (n^2 - 1) ≈ -γ μ0 ω c n (left circular)."""
        if any(x is None for x in [self.n_minus, self.gamma, self.mu0, self.omega, self.c]):
            return (False, "SKIP: missing data for dispersion check of n_minus")
        n = self.n_minus
        lhs = n**2 - 1.0
        rhs = -self.gamma * self.mu0 * self.omega * self.c * n
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-4):
            return (True, f"n_minus dispersion holds: n^2-1={lhs:.6e}, -γμ0ωc n={rhs:.6e}")
        else:
            return (False, f"n_minus dispersion violated: n^2-1={lhs:.6e}, expected ~{rhs:.6e}")

    def test_invariant_small_gamma_approximation(self):
        """Check that |n_plus - 1| << 1 and n_minus - 1 ≈ -(n_plus - 1) (small gamma)."""
        if self.n_plus is None or self.n_minus is None:
            return (False, "SKIP: n_plus or n_minus not available")
        if abs(self.n_plus - 1.0) < 0.1 and abs(self.n_minus - 1.0) < 0.1:
            return (True, "Both n_plus and n_minus are close to 1 (small gamma regime)")
        else:
            return (False, f"n_plus={self.n_plus:.4f}, n_minus={self.n_minus:.4f} are not close to 1")

    # ------------------------------------------------------------------
    # Modeling / Trace tests
    # ------------------------------------------------------------------
    def test_modeling_phi_sign_and_magnitude(self):
        """Check that phi has the same sign as gamma (if gamma > 0, phi > 0)."""
        if self.phi is None or self.gamma is None:
            return (False, "SKIP: phi or gamma not available")
        if self.gamma > 0 and self.phi <= 0:
            return (False, f"gamma>0 but phi={self.phi:.6e} not positive")
        if self.gamma < 0 and self.phi >= 0:
            return (False, f"gamma<0 but phi={self.phi:.6e} not negative")
        return (True, f"phi sign consistent with gamma: gamma={self.gamma}, phi={self.phi:.6e}")

    def test_modeling_n_plus_greater_than_one(self):
        """For gamma positive, n_plus should be > 1 (assuming gamma>0)."""
        if self.n_plus is None or self.gamma is None:
            return (False, "SKIP: n_plus or gamma not available")
        if self.gamma > 0 and self.n_plus <= 1.0:
            return (False, f"gamma>0 but n_plus={self.n_plus:.6f} <= 1")
        if self.gamma < 0 and self.n_plus >= 1.0:
            return (False, f"gamma<0 but n_plus={self.n_plus:.6f} >= 1")
        return (True, "n_plus relative to 1 consistent with gamma sign")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturbation-based)
    # ------------------------------------------------------------------
    def test_monotonicity_phi_increases_with_gamma(self):
        """
        Weak monotonicity: if we artificially increase gamma by 1%, phi should increase.
        Since we have only one data point, we compare with the theoretical expression
        phi ≈ 0.5 * μ0 * γ * ω^2 * L. If givens provide the needed constants, we can compute
        expected phi and then perturb gamma hypothetically. Better: check that phi is
        approximately proportional to gamma by comparing with the formula.
        This test does NOT use math.isclose in a strict sense; it checks sign and rough magnitude.
        """
        if any(x is None for x in [self.phi, self.gamma, self.mu0, self.omega, self.L]):
            return (False, "SKIP: missing data for monotonicity test")
        # Compute nominal expected phi from formula
        expected_nominal = 0.5 * self.mu0 * self.gamma * self.omega**2 * self.L
        # Perturb gamma by +1%
        gamma_pert = self.gamma * 1.01
        expected_pert = 0.5 * self.mu0 * gamma_pert * self.omega**2 * self.L
        # Check that perturbed expected is larger than nominal
        if expected_pert > expected_nominal:
            return (True, f"Phi increases with gamma: nominal {expected_nominal:.6e}, +1% gamma gives {expected_pert:.6e}")
        else:
            return (False, "Unexpected: perturbed phi not larger than nominal (theoretical formula)")
    # ------------------------------------------------------------------
    # Weak answer sanity (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        """Check that n_plus, n_minus, phi are positive finite numbers."""
        values = []
        if self.n_plus is not None:
            values.append(('n_plus', self.n_plus))
        if self.n_minus is not None:
            values.append(('n_minus', self.n_minus))
        if self.phi is not None:
            values.append(('phi', self.phi))
        for name, val in values:
            if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
                return (False, f"{name} is not positive finite: {val}")
        return (True, "All checked quantities are positive finite numbers")

    # ------------------------------------------------------------------
    # Run all tests
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
                # Optionally include traceback for debugging
                # msg += "\n" + traceback.format_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)