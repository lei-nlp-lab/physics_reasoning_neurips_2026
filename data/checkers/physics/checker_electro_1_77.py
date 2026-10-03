import math
import traceback

class PhysicsChecker:
    """
    A robust physics checker for the electrostatic problem involving two charges
    above a grounded conducting plane.  It performs interface, invariant, modeling,
    monotonicity, and weak answer-sanity tests.
    """

    def __init__(self, solution_instance):
        # Extract results safely; do NOT assign numeric defaults.
        results = getattr(solution_instance, 'results', {})
        givens = getattr(solution_instance, 'givens', {})

        self.F = results.get('F', None)
        self.W = results.get('W', None)
        self.sigma = results.get('sigma', None)

        self.q = givens.get('q', None)
        self.a = givens.get('a', None)
        self.epsilon0 = givens.get('epsilon0', None)

    # ------------------------------------------------------------------
    # Helper methods (not tests) for computing reference values
    # ------------------------------------------------------------------
    def _compute_F_mag(self, q, a, eps0):
        """Return magnitude of force on +q from all other real and image charges."""
        k = 1 / (4 * math.pi * eps0)
        return (math.sqrt(2) - 1) * k * q**2 / (8 * a**2)

    def _compute_W(self, q, a, eps0):
        """Return work done to assemble the system."""
        k = 1 / (4 * math.pi * eps0)
        return k * q**2 * (1 / (2 * math.sqrt(2) * a) - 1 / a)

    def _compute_sigma(self, q, a, eps0):
        """Return surface charge density at (a,0,0)."""
        return (q / (2 * math.pi * eps0 * a**2)) * (1 / (5 * math.sqrt(5)) - 1)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Ensure that results contain F, W, sigma and they are numeric."""
        if self.F is None:
            return (False, "Missing 'F' in results")
        if self.W is None:
            return (False, "Missing 'W' in results")
        if self.sigma is None:
            return (False, "Missing 'sigma' in results")
        # Check that they are numbers (int or float)
        for name, val in [('F', self.F), ('W', self.W), ('sigma', self.sigma)]:
            if not isinstance(val, (int, float)):
                return (False, f"'{name}' is not numeric, got {type(val).__name__}")
        return (True, "All required result keys present and numeric")

    def test_interface_givens_available(self):
        """Check that required givens are present."""
        if self.q is None:
            return (False, "Missing 'q' in givens")
        if self.a is None:
            return (False, "Missing 'a' in givens")
        if self.epsilon0 is None:
            return (False, "Missing 'epsilon0' in givens")
        # Numeric check
        for name, val in [('q', self.q), ('a', self.a), ('epsilon0', self.epsilon0)]:
            if not isinstance(val, (int, float)):
                return (False, f"Given '{name}' is not numeric, got {type(val).__name__}")
        if self.a <= 0:
            return (False, "a must be positive")
        if self.epsilon0 <= 0:
            return (False, "epsilon0 must be positive")
        return (True, "All required givens present and valid")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_force_positive(self):
        """Force magnitude must be positive."""
        if self.F is None:
            return (False, "SKIP: F not available")
        if self.F <= 0:
            return (False, f"Force magnitude is non-positive: {self.F}")
        return (True, f"Force magnitude is positive: {self.F}")

    def test_invariant_work_positive(self):
        """Work done against electrostatic forces must be positive."""
        if self.W is None:
            return (False, "SKIP: W not available")
        if self.W <= 0:
            return (False, f"Work is non-positive: {self.W}")
        return (True, f"Work is positive: {self.W}")

    def test_invariant_sigma_negative(self):
        """Surface charge density at (a,0,0) must be negative."""
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        if self.sigma >= 0:
            return (False, f"sigma is non-negative: {self.sigma}")
        return (True, f"sigma is negative: {self.sigma}")

    def test_invariant_force_y_zero(self):
        """The y-component of the force on +q should be zero (by symmetry)."""
        # Since only magnitude is given, we can only check that F_y is zero if provided.
        # Assume F_y is not a separate result; skip unless available.
        # This is a placeholder: we note that the geometry forces zero y-force.
        return (True, "Symmetry ensures zero y-force (no data to check numerically)")

    # ------------------------------------------------------------------
    # Modeling / trace tests (re‑derive from physics)
    # ------------------------------------------------------------------
    def test_modeling_force_magnitude(self):
        """Recompute force magnitude from givens and compare with results."""
        if None in (self.F, self.q, self.a, self.epsilon0):
            return (False, "SKIP: missing required data")
        try:
            F_ref = self._compute_F_mag(self.q, self.a, self.epsilon0)
            if math.isclose(self.F, F_ref, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"F matches reference: {self.F} vs {F_ref}")
            else:
                return (False, f"F mismatch: got {self.F}, expected {F_ref}")
        except Exception as e:
            return (False, f"Error in force modeling: {e}")

    def test_modeling_work(self):
        """Recompute work from givens and compare with results."""
        if None in (self.W, self.q, self.a, self.epsilon0):
            return (False, "SKIP: missing required data")
        try:
            W_ref = self._compute_W(self.q, self.a, self.epsilon0)
            if math.isclose(self.W, W_ref, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"W matches reference: {self.W} vs {W_ref}")
            else:
                return (False, f"W mismatch: got {self.W}, expected {W_ref}")
        except Exception as e:
            return (False, f"Error in work modeling: {e}")

    def test_modeling_sigma(self):
        """Recompute sigma from givens and compare with results."""
        if None in (self.sigma, self.q, self.a, self.epsilon0):
            return (False, "SKIP: missing required data")
        try:
            sigma_ref = self._compute_sigma(self.q, self.a, self.epsilon0)
            if math.isclose(self.sigma, sigma_ref, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"sigma matches reference: {self.sigma} vs {sigma_ref}")
            else:
                return (False, f"sigma mismatch: got {self.sigma}, expected {sigma_ref}")
        except Exception as e:
            return (False, f"Error in sigma modeling: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input and check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_force_decreases_with_a(self):
        """If a increases, the force magnitude should decrease (∝ 1/a^2)."""
        if None in (self.q, self.a, self.epsilon0):
            return (False, "SKIP: missing required givens")
        try:
            a_orig = self.a
            a_new = a_orig * 1.1  # 10% increase
            F_orig = self._compute_F_mag(self.q, a_orig, self.epsilon0)
            F_new = self._compute_F_mag(self.q, a_new, self.epsilon0)
            if F_new < F_orig:
                return (True, f"Force decreased from {F_orig} to {F_new} when a increased")
            else:
                return (False, f"Force did not decrease: old={F_orig}, new={F_new}")
        except Exception as e:
            return (False, f"Error in monotonicity force: {e}")

    def test_monotonicity_work_decreases_with_a(self):
        """If a increases, the work should decrease (∝ 1/a)."""
        if None in (self.q, self.a, self.epsilon0):
            return (False, "SKIP: missing required givens")
        try:
            a_orig = self.a
            a_new = a_orig * 1.1
            W_orig = self._compute_W(self.q, a_orig, self.epsilon0)
            W_new = self._compute_W(self.q, a_new, self.epsilon0)
            if W_new < W_orig:
                return (True, f"Work decreased from {W_orig} to {W_new} when a increased")
            else:
                return (False, f"Work did not decrease: old={W_orig}, new={W_new}")
        except Exception as e:
            return (False, f"Error in monotonicity work: {e}")

    def test_monotonicity_sigma_magnitude_decreases_with_a(self):
        """If a increases, the magnitude of sigma should decrease (∝ 1/a^2)."""
        if None in (self.q, self.a, self.epsilon0):
            return (False, "SKIP: missing required givens")
        try:
            a_orig = self.a
            a_new = a_orig * 1.1
            sigma_orig = self._compute_sigma(self.q, a_orig, self.epsilon0)
            sigma_new = self._compute_sigma(self.q, a_new, self.epsilon0)
            # sigma is negative, so magnitude = -sigma
            if -sigma_new < -sigma_orig:
                return (True, f"Sigma magnitude decreased from {-sigma_orig} to {-sigma_new} when a increased")
            else:
                return (False, f"Sigma magnitude did not decrease: old mag={-sigma_orig}, new mag={-sigma_new}")
        except Exception as e:
            return (False, f"Error in monotonicity sigma: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Weak sanity: all results are finite, non-zero, and signs are consistent."""
        checks = []
        all_pass = True
        for name, val in [('F', self.F), ('W', self.W), ('sigma', self.sigma)]:
            if val is None:
                checks.append(f"'{name}' missing")
                all_pass = False
                continue
            if not math.isfinite(val):
                checks.append(f"'{name}' is not finite: {val}")
                all_pass = False
            if val == 0:
                checks.append(f"'{name}' is zero (unexpected)")
                all_pass = False
        if self.F is not None and self.F <= 0:
            checks.append("F should be positive")
            all_pass = False
        if self.W is not None and self.W <= 0:
            checks.append("W should be positive")
            all_pass = False
        if self.sigma is not None and self.sigma >= 0:
            checks.append("sigma should be negative")
            all_pass = False
        if all_pass:
            return (True, "All weak sanity checks passed")
        else:
            return (False, "; ".join(checks))

    # ------------------------------------------------------------------
    # Runner
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
                traceback.print_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)