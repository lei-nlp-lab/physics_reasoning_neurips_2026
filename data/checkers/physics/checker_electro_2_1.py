import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for magnetic fields inside and outside a cylindrical wire
    carrying steady current I, with permeability μ and radius R.
    Checks Ampere's law, constitutive relations, boundary conditions, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Safely extract results
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.H_out = results.get('H_out', None)
        self.B_out = results.get('B_out', None)
        self.H_in = results.get('H_in', None)
        self.B_in = results.get('B_in', None)

        # Safely extract givens (if available)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.I = givens.get('I', None)
        self.R = givens.get('R', None)
        self.mu = givens.get('mu', None)        # permeability inside wire
        self.mu0 = givens.get('mu0', None)      # permeability of free space
        self.r = givens.get('r', None)          # radial coordinate (may be symbolic)

        # Also try to get from solution_instance directly
        if self.I is None:
            self.I = getattr(solution_instance, 'I', None)
        if self.R is None:
            self.R = getattr(solution_instance, 'R', None)
        if self.mu is None:
            self.mu = getattr(solution_instance, 'mu', None)
        if self.mu0 is None:
            self.mu0 = getattr(solution_instance, 'mu0', None)
        if self.r is None:
            self.r = getattr(solution_instance, 'r', None)

        # Detect symbolic content
        self._symbolic = False
        if any(isinstance(x, sp.Basic) for x in [self.H_out, self.B_out, self.H_in, self.B_in, self.I, self.R, self.mu, self.mu0, self.r]):
            self._symbolic = True

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _to_numeric(self, expr):
        """Convert an expression to float if possible. Return None otherwise."""
        if expr is None:
            return None
        try:
            if isinstance(expr, sp.Basic):
                if expr.is_number:
                    return float(expr.evalf())
                else:
                    return None
            return float(expr)
        except (ValueError, TypeError, AttributeError):
            return None

    def _skip_if_missing(self, *names):
        """Return True if any of the given attribute names have None value."""
        for name in names:
            if getattr(self, name, None) is None:
                return True
        return False

    def _has_symbolic_r(self, expr):
        """Check if expr is a SymPy expression containing a free symbol 'r'."""
        if not isinstance(expr, sp.Basic):
            return False
        return any(s.name == 'r' for s in expr.free_symbols)

    # ----------------------------------------------------------------------
    # Interface Tests
    # ----------------------------------------------------------------------
    def test_interface_results_keys(self):
        """All four expected result keys must exist."""
        if self.H_out is None:
            return (False, "Missing key 'H_out' in results")
        if self.B_out is None:
            return (False, "Missing key 'B_out' in results")
        if self.H_in is None:
            return (False, "Missing key 'H_in' in results")
        if self.B_in is None:
            return (False, "Missing key 'B_in' in results")
        return (True, "All four result keys present")

    def test_interface_givens_available(self):
        """Essential givens (I, R, mu, mu0) should be provided for full checking."""
        missing = []
        if self.I is None: missing.append("I")
        if self.R is None: missing.append("R")
        if self.mu is None: missing.append("mu")
        if self.mu0 is None: missing.append("mu0")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All essential givens available")

    # ----------------------------------------------------------------------
    # Invariant Tests (theorem-based, using isclose for numeric)
    # ----------------------------------------------------------------------
    def test_invariant_ampere_outside(self):
        """H_out must satisfy ∮ H·dl = I_enc, yielding H_out = I/(2πr) for r≥R."""
        if self._skip_if_missing('H_out', 'I'):
            return (False, "SKIP: Missing H_out or I")
        if self._symbolic and self._has_symbolic_r(self.H_out):
            # Derive expected H from Ampere's law
            expected_H = self.I / (2 * sp.pi * self.r)
            try:
                diff = sp.simplify(self.H_out - expected_H)
                if diff == 0:
                    return (True, "Symbolic Ampere law outside holds exactly")
                else:
                    return (False, f"Symbolic discrepancy: H_out - I/(2πr) simplified to {diff}")
            except Exception as e:
                return (False, f"Symbolic check failed: {e}")
        # Numeric fallback
        if self._skip_if_missing('r'):
            return (False, "SKIP: Numeric check requires r")
        H_val = self._to_numeric(self.H_out)
        I_val = self._to_numeric(self.I)
        r_val = self._to_numeric(self.r)
        if None in (H_val, I_val, r_val):
            return (False, "Could not convert to numeric values")
        expected_H = I_val / (2 * math.pi * r_val)
        if math.isclose(H_val, expected_H, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Ampere law outside: H ≈ {expected_H:.4e}, got {H_val:.4e}")
        else:
            return (False, f"Ampere law outside fails: expected H ~ {expected_H:.4e}, got {H_val:.4e}")

    def test_invariant_ampere_inside(self):
        """H_in must satisfy ∮ H·dl = I_enc = I r²/R², yielding H_in = I r/(2πR²) for r<R."""
        if self._skip_if_missing('H_in', 'I', 'R'):
            return (False, "SKIP: Missing H_in, I, or R")
        if self._symbolic and self._has_symbolic_r(self.H_in):
            expected_H = (self.I * self.r) / (2 * sp.pi * self.R**2)
            try:
                diff = sp.simplify(self.H_in - expected_H)
                if diff == 0:
                    return (True, "Symbolic Ampere law inside holds exactly")
                else:
                    return (False, f"Symbolic discrepancy: H_in - I r/(2πR²) simplified to {diff}")
            except Exception as e:
                return (False, f"Symbolic check failed: {e}")
        if self._skip_if_missing('r'):
            return (False, "SKIP: Numeric check requires r")
        H_val = self._to_numeric(self.H_in)
        I_val = self._to_numeric(self.I)
        R_val = self._to_numeric(self.R)
        r_val = self._to_numeric(self.r)
        if None in (H_val, I_val, R_val, r_val):
            return (False, "Could not convert to numeric values")
        expected_H = (I_val * r_val) / (2 * math.pi * R_val**2)
        if math.isclose(H_val, expected_H, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Ampere law inside: H ≈ {expected_H:.4e}, got {H_val:.4e}")
        else:
            return (False, f"Ampere law inside fails: expected H ~ {expected_H:.4e}, got {H_val:.4e}")

    def test_invariant_constitutive_outside(self):
        """Outside wire: B_out = μ₀ H_out."""
        if self._skip_if_missing('B_out', 'H_out', 'mu0'):
            return (False, "SKIP: Missing B_out, H_out, or mu0")
        if self._symbolic:
            expected_B = self.mu0 * self.H_out
            try:
                diff = sp.simplify(self.B_out - expected_B)
                if diff == 0:
                    return (True, "Symbolic B_out = μ₀ H_out holds exactly")
                else:
                    return (False, f"Symbolic discrepancy: B_out - μ₀ H_out simplified to {diff}")
            except Exception as e:
                return (False, f"Symbolic check failed: {e}")
        B_val = self._to_numeric(self.B_out)
        H_val = self._to_numeric(self.H_out)
        mu0_val = self._to_numeric(self.mu0)
        if None in (B_val, H_val, mu0_val):
            return (False, "Could not convert to numeric values")
        expected_B = mu0_val * H_val
        if math.isclose(B_val, expected_B, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Constitutive outside: B ≈ {expected_B:.4e}, got {B_val:.4e}")
        else:
            return (False, f"Constitutive outside fails: expected B ~ {expected_B:.4e}, got {B_val:.4e}")

    def test_invariant_constitutive_inside(self):
        """Inside wire: B_in = μ H_in."""
        if self._skip_if_missing('B_in', 'H_in', 'mu'):
            return (False, "SKIP: Missing B_in, H_in, or mu")
        if self._symbolic:
            expected_B = self.mu * self.H_in
            try:
                diff = sp.simplify(self.B_in - expected_B)
                if diff == 0:
                    return (True, "Symbolic B_in = μ H_in holds exactly")
                else:
                    return (False, f"Symbolic discrepancy: B_in - μ H_in simplified to {diff}")
            except Exception as e:
                return (False, f"Symbolic check failed: {e}")
        B_val = self._to_numeric(self.B_in)
        H_val = self._to_numeric(self.H_in)
        mu_val = self._to_numeric(self.mu)
        if None in (B_val, H_val, mu_val):
            return (False, "Could not convert to numeric values")
        expected_B = mu_val * H_val
        if math.isclose(B_val, expected_B, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Constitutive inside: B ≈ {expected_B:.4e}, got {B_val:.4e}")
        else:
            return (False, f"Constitutive inside fails: expected B ~ {expected_B:.4e}, got {B_val:.4e}")

    def test_invariant_boundary_condition_H(self):
        """At r = R, both H_in and H_out must be equal (continuity of tangential H)."""
        if self._skip_if_missing('H_in', 'H_out', 'R'):
            return (False, "SKIP: Missing H_in, H_out, or R")
        if self._symbolic and self._has_symbolic_r(self.H_in) and self._has_symbolic_r(self.H_out):
            try:
                r_sym = sp.Symbol('r', positive=True)
                H_in_at_R = self.H_in.subs(r_sym, self.R)
                H_out_at_R = self.H_out.subs(r_sym, self.R)
                diff = sp.simplify(H_in_at_R - H_out_at_R)
                if diff == 0:
                    return (True, "Symbolic H continuous at r=R")
                else:
                    return (False, f"H discontinuity at r=R: difference = {diff}")
            except Exception as e:
                return (False, f"Symbolic boundary check failed: {e}")
        # Numeric: we need numeric values of H at r=R; we can compute from Ampere's law if we have I, R.
        # But the stored H_in, H_out may be for a different r. So we skip.
        return (False, "SKIP: Numeric boundary test requires H values at r=R, which is not provided.")

    # ----------------------------------------------------------------------
    # Monotonicity Tests (explicit perturbation of radial coordinate r)
    # ----------------------------------------------------------------------
    def test_monotonicity_H_out_decreases_with_r(self):
        """For r ≥ R, H_out ∝ 1/r, so increasing r must decrease H_out.
        Perturb r by +10% and compare baseline vs perturbed values."""
        if not self._symbolic or not self._has_symbolic_r(self.H_out):
            return (False, "SKIP: H_out is not a symbolic expression with r; cannot perturb")
        if self._skip_if_missing('H_out', 'I', 'R'):
            return (False, "SKIP: Missing H_out, I, or R")
        r_sym = sp.Symbol('r', positive=True)
        # Choose baseline r0 outside the wire (e.g., 1.5*R)
        try:
            R_num = self._to_numeric(self.R)
            if R_num is None:
                return (False, "SKIP: R is not numeric")
        except:
            return (False, "SKIP: Could not get numeric R")
        r0 = 1.5 * R_num
        r1 = 1.1 * r0   # perturbed: increase by 10%
        # Evaluate H_out at r0 and r1
        try:
            H0 = float(self.H_out.subs(r_sym, r0).evalf())
            H1 = float(self.H_out.subs(r_sym, r1).evalf())
        except Exception as e:
            return (False, f"Monotonicity evaluation failed: {e}")
        if H0 > H1:
            return (True, f"H_out decreased as r increased: baseline {H0:.4e} > perturbed {H1:.4e}")
        else:
            return (False, f"H_out did not decrease: baseline {H0:.4e} <= perturbed {H1:.4e}")

    def test_monotonicity_H_in_increases_with_r(self):
        """For 0 ≤ r < R, H_in ∝ r, so increasing r must increase H_in.
        Perturb r by +10% and compare baseline vs perturbed values."""
        if not self._symbolic or not self._has_symbolic_r(self.H_in):
            return (False, "SKIP: H_in is not a symbolic expression with r; cannot perturb")
        if self._skip_if_missing('H_in', 'I', 'R'):
            return (False, "SKIP: Missing H_in, I, or R")
        r_sym = sp.Symbol('r', positive=True)
        try:
            R_num = self._to_numeric(self.R)
            if R_num is None:
                return (False, "SKIP: R is not numeric")
        except:
            return (False, "SKIP: Could not get numeric R")
        r0 = 0.5 * R_num   # inside wire
        r1 = 1.1 * r0     # perturbed: increase by 10%
        try:
            H0 = float(self.H_in.subs(r_sym, r0).evalf())
            H1 = float(self.H_in.subs(r_sym, r1).evalf())
        except Exception as e:
            return (False, f"Monotonicity evaluation failed: {e}")
        if H0 < H1:
            return (True, f"H_in increased as r increased: baseline {H0:.4e} < perturbed {H1:.4e}")
        else:
            return (False, f"H_in did not increase: baseline {H0:.4e} >= perturbed {H1:.4e}")

    # ----------------------------------------------------------------------
    # Modeling / Trace Tests (check that results follow from physics; single relation per test)
    # ----------------------------------------------------------------------
    def test_trace_H_out_derived_from_ampere(self):
        """H_out must equal I/(2πr) as derived from Ampere's law outside."""
        # This is essentially the same as test_invariant_ampere_outside; reuse.
        return self.test_invariant_ampere_outside()

    def test_trace_B_out_derived_from_constitutive(self):
        """B_out must equal μ₀ H_out."""
        return self.test_invariant_constitutive_outside()

    def test_trace_H_in_derived_from_ampere(self):
        """H_in must equal I r/(2πR²) as derived from Ampere's law inside."""
        return self.test_invariant_ampere_inside()

    def test_trace_B_in_derived_from_constitutive(self):
        """B_in must equal μ H_in."""
        return self.test_invariant_constitutive_inside()

    # ----------------------------------------------------------------------
    # Weak Answer Sanity (only one, no hardcoded target values)
    # ----------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """All field results must be positive (for positive I) and finite."""
        fields = {
            'H_out': self.H_out,
            'B_out': self.B_out,
            'H_in': self.H_in,
            'B_in': self.B_in
        }
        issues = []
        for name, val in fields.items():
            if val is None:
                continue
            num = self._to_numeric(val)
            if num is not None:
                if num < 0:
                    issues.append(f"{name} is negative ({num})")
                if not math.isfinite(num):
                    issues.append(f"{name} is non-finite ({num})")
            else:
                # For symbolic, we cannot check sign numerically here.
                pass
        if issues:
            return (False, "; ".join(issues))
        return (True, "No negative or non-finite numeric field values detected")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
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