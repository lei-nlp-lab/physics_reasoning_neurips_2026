import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.E_out = results.get('E_out', None)
        self.E_in = results.get('E_in', None)

        # Extract givens safely (from results or directly from solution_instance)
        # Assume solution_instance may have attributes: Q, R, epsilon_0, r
        self.Q = getattr(solution_instance, 'Q', None)
        self.R = getattr(solution_instance, 'R', None)
        self.epsilon_0 = getattr(solution_instance, 'epsilon_0', None)
        self.r = getattr(solution_instance, 'r', None)

        # If givens are inside results, try that too
        if self.Q is None:
            self.Q = results.get('Q', None)
        if self.R is None:
            self.R = results.get('R', None)
        if self.epsilon_0 is None:
            self.epsilon_0 = results.get('epsilon_0', None)
        if self.r is None:
            self.r = results.get('r', None)

    # ------------------------------------------------------------------
    # INTERFACE CHECKS
    # ------------------------------------------------------------------
    def test_interface_has_results_keys(self):
        """Check that expected keys exist in results."""
        if self.E_out is None:
            return (False, "Missing key 'E_out' in results")
        if self.E_in is None:
            return (False, "Missing key 'E_in' in results")
        return (True, "Both E_out and E_in present")

    def test_interface_givens_available(self):
        """Check that essential givens are provided and positive."""
        missing = []
        for name, val in [('Q', self.Q), ('R', self.R),
                          ('epsilon_0', self.epsilon_0), ('r', self.r)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        # Now check positivity (allow Q=0? but that's trivial; here Q>0 for typical)
        if self.Q <= 0:
            return (False, f"Q must be positive, got {self.Q}")
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}")
        if self.epsilon_0 <= 0:
            return (False, f"epsilon_0 must be positive, got {self.epsilon_0}")
        if self.r <= 0:
            return (False, f"r must be positive, got {self.r}")
        return (True, "All essential givens present and positive")

    # ------------------------------------------------------------------
    # INVARIANT CHECKS (theorem-based, use isclose)
    # ------------------------------------------------------------------
    def test_invariant_gauss_outside(self):
        """For r > R, check Gauss's law: E_out * 4πr² = Q/ε₀."""
        if any(v is None for v in [self.E_out, self.Q, self.epsilon_0, self.r, self.R]):
            return (False, "SKIP: missing data for outside Gauss check")
        if self.r <= self.R:
            return (False, "SKIP: r not > R for outside check")
        expected_flux = self.Q / self.epsilon_0
        actual_flux = self.E_out * 4 * math.pi * self.r**2
        if math.isclose(actual_flux, expected_flux, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Outside Gauss law satisfied: flux ~ {actual_flux:.3e}")
        else:
            return (False, f"Outside Gauss law violated: flux={actual_flux:.3e}, expected~{expected_flux:.3e}")

    def test_invariant_gauss_inside(self):
        """For r < R, check Gauss's law: E_in * 4πr² = (Q * r³/R³)/ε₀."""
        if any(v is None for v in [self.E_in, self.Q, self.R, self.epsilon_0, self.r]):
            return (False, "SKIP: missing data for inside Gauss check")
        if self.r >= self.R:
            return (False, "SKIP: r not < R for inside check")
        enclosed_charge = self.Q * (self.r**3) / (self.R**3)
        expected_flux = enclosed_charge / self.epsilon_0
        actual_flux = self.E_in * 4 * math.pi * self.r**2
        if math.isclose(actual_flux, expected_flux, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Inside Gauss law satisfied: flux ~ {actual_flux:.3e}")
        else:
            return (False, f"Inside Gauss law violated: flux={actual_flux:.3e}, expected~{expected_flux:.3e}")

    def test_invariant_continuity_at_R(self):
        """Check continuity of E at r=R: E_out(R) should equal E_in(R)."""
        if any(v is None for v in [self.Q, self.R, self.epsilon_0]):
            return (False, "SKIP: missing data for continuity check")
        # Compute expected E at R from both formulas
        E_at_R_out = self.Q / (4 * math.pi * self.epsilon_0 * self.R**2)
        E_at_R_in = self.Q * self.R / (4 * math.pi * self.epsilon_0 * self.R**3)
        # They are identical; compare if they match
        if math.isclose(E_at_R_out, E_at_R_in, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Continuity holds: E(R)~{E_at_R_out:.3e}")
        else:
            return (False, f"Continuity violated: E_out(R)={E_at_R_out:.3e}, E_in(R)={E_at_R_in:.3e}")

    def test_invariant_field_sign(self):
        """Check that E_out and E_in have the same sign as Q (for nonzero Q)."""
        if any(v is None for v in [self.E_out, self.E_in, self.Q]):
            return (False, "SKIP: missing data for sign check")
        # Q could be positive or negative; we require same sign for both fields
        if (self.E_out * self.Q) <= 0:
            return (False, f"E_out sign inconsistent with Q: E_out={self.E_out}, Q={self.Q}")
        if (self.E_in * self.Q) <= 0:
            return (False, f"E_in sign inconsistent with Q: E_in={self.E_in}, Q={self.Q}")
        return (True, "Both fields have same sign as Q (radial direction consistent)")

    # ------------------------------------------------------------------
    # MODELING / TRACE CHECKS (using explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_outside_form(self):
        """Check that E_out follows the expected functional form: E_out = Q/(4πε₀r²)."""
        if any(v is None for v in [self.E_out, self.Q, self.epsilon_0, self.r]):
            return (False, "SKIP: missing data for outside form check")
        expected = self.Q / (4 * math.pi * self.epsilon_0 * self.r**2)
        if math.isclose(self.E_out, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_out matches expected form: {expected:.3e}")
        else:
            return (False, f"E_out={self.E_out:.3e} does not match expected {expected:.3e}")

    def test_modeling_inside_form(self):
        """Check that E_in follows expected form: E_in = Qr/(4πε₀R³)."""
        if any(v is None for v in [self.E_in, self.Q, self.R, self.epsilon_0, self.r]):
            return (False, "SKIP: missing data for inside form check")
        expected = self.Q * self.r / (4 * math.pi * self.epsilon_0 * self.R**3)
        if math.isclose(self.E_in, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_in matches expected form: {expected:.3e}")
        else:
            return (False, f"E_in={self.E_in:.3e} does not match expected {expected:.3e}")

    # ------------------------------------------------------------------
    # MONOTONICITY CHECKS (by perturbing r)
    # ------------------------------------------------------------------
    def _compute_expected_E_out(self, r_val):
        """Helper: compute E_out for a given r (assuming r >= R)."""
        return self.Q / (4 * math.pi * self.epsilon_0 * r_val**2)

    def _compute_expected_E_in(self, r_val):
        """Helper: compute E_in for a given r (assuming r <= R)."""
        return self.Q * r_val / (4 * math.pi * self.epsilon_0 * self.R**3)

    def test_monotonicity_E_out_decreases_with_r(self):
        """Check that E_out decreases as r increases (for r>R)."""
        if any(v is None for v in [self.Q, self.R, self.epsilon_0, self.r, self.E_out]):
            return (False, "SKIP: missing data for E_out monotonicity")
        if self.r <= self.R:
            return (False, "SKIP: r not > R for outside monotonicity")
        # Perturb r upward by 1%
        r_pert = self.r * 1.01
        E_pert = self._compute_expected_E_out(r_pert)
        if E_pert < self.E_out:
            return (True, f"E_out decreases with r: original={self.E_out:.3e}, perturbed={E_pert:.3e}")
        else:
            return (False, f"Monotonicity violated: E_out not decreasing, orig={self.E_out:.3e}, pert={E_pert:.3e}")

    def test_monotonicity_E_in_increases_with_r(self):
        """Check that E_in increases as r increases (for r<R)."""
        if any(v is None for v in [self.Q, self.R, self.epsilon_0, self.r, self.E_in]):
            return (False, "SKIP: missing data for E_in monotonicity")
        if self.r >= self.R:
            return (False, "SKIP: r not < R for inside monotonicity")
        # Perturb r upward by 1%
        r_pert = self.r * 1.01
        if r_pert > self.R:
            # Keep within inside region: cap at R
            r_pert = (self.r + self.R) / 2.0
        E_pert = self._compute_expected_E_in(r_pert)
        if E_pert > self.E_in:
            return (True, f"E_in increases with r: original={self.E_in:.3e}, perturbed={E_pert:.3e}")
        else:
            return (False, f"Monotonicity violated: E_in not increasing, orig={self.E_in:.3e}, pert={E_pert:.3e}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (exact value not used; no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_non_nan(self):
        """Check that E_out and E_in are finite and not NaN."""
        for name, val in [('E_out', self.E_out), ('E_in', self.E_in)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if math.isnan(val):
                return (False, f"{name} is NaN")
            if math.isinf(val):
                return (False, f"{name} is infinite")
        return (True, "Both fields are finite and not NaN")

    # ------------------------------------------------------------------
    # RUN ALL TESTS
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)