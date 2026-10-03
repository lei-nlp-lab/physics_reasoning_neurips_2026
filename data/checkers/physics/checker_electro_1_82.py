import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (use None defaults)
        givens = getattr(solution_instance, 'givens', {})
        self.Q = givens.get('Q', None)
        self.d = givens.get('d', None)
        self.V = givens.get('V', None)
        self.epsilon0 = givens.get('epsilon0', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.R = results.get('R', None)
        self.F = results.get('F', None)

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        """Check that all required result keys are present and numeric."""
        if self.R is None:
            return (False, "R is missing (None)")
        if self.F is None:
            return (False, "F is missing (None)")
        if self.d is None:
            return (False, "d is missing (None)")
        if not isinstance(self.R, (int, float)):
            return (False, f"R is not numeric: {type(self.R)}")
        if not isinstance(self.F, (int, float)):
            return (False, f"F is not numeric: {type(self.F)}")
        return (True, "All required keys present and numeric")

    # ---------- Invariant / theorem tests ----------
    def test_invariant_radius_positive_and_less_than_d(self):
        """R must be positive and lie inside the half‑separation d."""
        if self.R is None or self.d is None:
            return (False, "SKIP: R or d missing")
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}")
        if self.d <= 0:
            return (False, f"d must be positive, got {self.d}")
        if self.R >= self.d:
            return (False, f"R must be less than d, got R={self.R}, d={self.d}")
        return (True, f"R={self.R} is positive and less than d={self.d}")

    def test_invariant_force_sign(self):
        """Force sign should match the sign of Q*V (positive outward for Q,V>0)."""
        if self.F is None or self.Q is None or self.V is None:
            return (False, "SKIP: F, Q, or V missing")
        prod = self.Q * self.V
        if prod > 0:
            if self.F > 0:
                return (True, f"F={self.F} positive as expected for Q*V>0")
            else:
                return (False, f"F should be positive but got {self.F}")
        elif prod < 0:
            if self.F < 0:
                return (True, f"F={self.F} negative as expected for Q*V<0")
            else:
                return (False, f"F should be negative but got {self.F}")
        else:  # product zero => V=0 or Q=0 => force must be zero
            if math.isclose(self.F, 0, abs_tol=1e-12):
                return (True, "F=0 as expected when Q*V=0")
            else:
                return (False, f"F should be zero but got {self.F}")

    # ---------- Modeling / trace tests ----------
    def test_modeling_radius_fraction(self):
        """R/d must be close to 1/8 (derived from zero‑force condition)."""
        if self.R is None or self.d is None:
            return (False, "SKIP: R or d missing")
        if self.d == 0:
            return (False, "d is zero")
        ratio = self.R / self.d
        expected = 1/8
        if math.isclose(ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R/d={ratio:.4f} is close to {expected}")
        else:
            return (False, f"R/d={ratio:.4f} not close to expected {expected}")

    def test_modeling_force_relation(self):
        """Check F = Q * R * V / d^2, the correct relation from method of images."""
        if any(x is None for x in [self.F, self.Q, self.R, self.V, self.d]):
            return (False, "SKIP: one or more required values missing")
        if self.d == 0:
            return (False, "d is zero")
        expected_F = self.Q * self.R * self.V / (self.d**2)
        if math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F={self.F:.6e} matches derived formula Q*R*V/d^2 = {expected_F:.6e}")
        else:
            return (False, f"F={self.F:.6e} does not match expected {expected_F:.6e}")

    # ---------- Monotonicity test ----------
    def test_monotonicity_force_with_potential(self):
        """If potential V is increased by 10%, force increases proportionally."""
        if any(x is None for x in [self.Q, self.R, self.V, self.d, self.F]):
            return (False, "SKIP: missing data")
        if self.d == 0 or self.V == 0:
            return (False, "d or V is zero – cannot test proportional increase")
        # Baseline: original F
        baseline_F = self.F
        # Perturb V upward by 10%
        V_pert = 1.1 * self.V
        # Compute expected F_pert using the derived formula F = Q*R*V/d^2
        F_pert = self.Q * self.R * V_pert / (self.d**2)
        # Check monotonic increase
        if F_pert > baseline_F:
            return (True, f"Increasing V by 10% increases F from {baseline_F:.6e} to {F_pert:.6e} (monotonic increase)")
        else:
            return (False, f"Monotonicity violated: V increase should increase F, got F_pert={F_pert:.6e} <= baseline={baseline_F:.6e}")

    # ---------- Weak answer sanity ----------
    def test_answer_sanity_finite(self):
        """Basic sanity: R and F are finite and not pathological."""
        if self.R is None or self.F is None:
            return (False, "SKIP: R or F missing")
        if not (math.isfinite(self.R) and math.isfinite(self.F)):
            return (False, "R or F is not finite or is NaN")
        if self.R == 0.0:
            return (False, "R is zero, which is physically impossible")
        return (True, "R and F are finite and non‑pathological")

    # ---------- Test runner ----------
    def run_all_tests(self):
        """Discover all test_* methods and execute them, returning (all_pass, logs)."""
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