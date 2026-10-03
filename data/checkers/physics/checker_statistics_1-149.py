import math
import traceback

# Try to import sympy; if not available, we'll still be able to run numeric tests
try:
    import sympy as sp
except ImportError:
    sp = None

class PhysicsChecker:
    """
    Audits a solution for the adiabatic sound speed in a degenerate Fermi gas.
    Checks interface completeness, physical invariants, modeling consistency,
    monotonicity, and performs a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract results and givens
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', getattr(solution_instance, 'given', {}))
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Extract input parameters (use None if missing)
        self.n = self.givens.get('n', None)
        self.m = self.givens.get('m', None)
        self.N = self.givens.get('N', None)
        self.V = self.givens.get('V', None)

        # Physical constants (SI)
        self.hbar_val = 1.054571817e-34
        self.h_val = 6.62607015e-34

        # Extracted results (symbolic or numeric)
        self.v_sq = self.results.get('v_sq', None)
        self.v = self.results.get('v', None)
        self.v_h = self.results.get('v_h', None)

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    def _is_numeric(self, val):
        """Check if a value is a numeric type or can be converted to float."""
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False

    def _get_expected_v_numeric(self):
        """
        Compute the expected sound speed for a degenerate Fermi gas at T=0,
        using the standard formula v = (hbar/(m*sqrt(3))) * (3*pi^2 * n)^(1/3).
        Returns None if n or m are not numeric.
        """
        if self.n is None or self.m is None:
            return None
        if not self._is_numeric(self.n) or not self._is_numeric(self.m):
            return None
        n_num = float(self.n)
        m_num = float(self.m)
        return (self.hbar_val / (m_num * math.sqrt(3))) * \
               math.pow(3 * math.pi**2 * n_num, 1.0/3.0)

    # --------------------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present."""
        missing = []
        if self.v_sq is None:
            missing.append('v_sq')
        if self.v is None:
            missing.append('v')
        if self.v_h is None:
            missing.append('v_h')
        if missing:
            return (False, f"Missing results keys: {missing}")
        return (True, "All required keys present")

    # --------------------------------------------------------------------------
    # Invariant (theorem-based) tests
    # --------------------------------------------------------------------------
    def test_invariant_adiabatic_definition(self):
        """
        Validate that v_sq is non‑negative (if numeric) and generally consistent
        with the adiabatic definition v^2 = (∂p/∂ρ)_S.
        """
        if self.v_sq is None:
            return (False, "SKIP: v_sq not available")
        if self._is_numeric(self.v_sq):
            if float(self.v_sq) <= 0:
                return (False, f"v_sq must be positive, got {self.v_sq}")
        return (True, "v_sq is present and positive (if numeric)")

    def test_invariant_v_positive(self):
        """Check that the sound speed v is positive (if numeric)."""
        if self.v is None:
            return (False, "SKIP: v not available")
        if self._is_numeric(self.v):
            if float(self.v) <= 0:
                return (False, f"v must be positive, got {self.v}")
        return (True, "v is positive (if numeric)")

    # --------------------------------------------------------------------------
    # Modeling / trace consistency tests
    # --------------------------------------------------------------------------
    def test_modeling_v_formula_numeric(self):
        """
        If n and m are numeric, compare the solution's v against the expected
        formula for a degenerate Fermi gas at T=0.
        """
        if self.n is None or self.m is None:
            return (False, "SKIP: n or m not available")
        if self.v is None:
            return (False, "SKIP: v not available")
        if not self._is_numeric(self.n) or not self._is_numeric(self.m):
            return (False, "SKIP: n or m not numeric")
        expected = self._get_expected_v_numeric()
        if expected is None:
            return (False, "SKIP: cannot compute expected v")
        if not self._is_numeric(self.v):
            return (False, "SKIP: solution v is not numeric")
        v_sol = float(self.v)
        if not math.isclose(v_sol, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"v mismatch: solution {v_sol}, expected {expected}")
        return (True, "v matches expected form within tolerance")

    def test_modeling_v_h_consistency(self):
        """
        Verify that v and v_h represent the same physical quantity.
        For numerical values, they should be equal.
        For symbolic expressions, they should be equivalent.
        """
        if self.v is None or self.v_h is None:
            return (False, "SKIP: v or v_h not available")
        # Numeric check
        if self._is_numeric(self.v) and self._is_numeric(self.v_h):
            v_num = float(self.v)
            vh_num = float(self.v_h)
            if not math.isclose(v_num, vh_num, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"v and v_h differ numerically: v={v_num}, v_h={vh_num}")
            return (True, "v and v_h are consistent (equal numerically)")
        # Symbolic check (if sympy is available)
        if sp is not None:
            if isinstance(self.v, sp.Basic) or isinstance(self.v_h, sp.Basic):
                try:
                    diff = sp.simplify(self.v - self.v_h)
                    if diff == 0:
                        return (True, "Symbolic expressions are equivalent")
                    else:
                        return (False, f"Symbolic expressions differ: diff = {diff}")
                except Exception as e:
                    return (False, f"Symbolic comparison crashed: {e}")
        return (False, "SKIP: cannot compare non‑numeric, non‑symbolic values")

    # --------------------------------------------------------------------------
    # Monotonicity tests
    # --------------------------------------------------------------------------
    def test_monotonicity_v_increases_with_n(self):
        """
        For a degenerate Fermi gas, v ∝ n^(1/3). Perturb n upward by 10%
        and check that the expected v increases.
        """
        if self.n is None or self.m is None:
            return (False, "SKIP: n or m not available")
        if not self._is_numeric(self.n) or not self._is_numeric(self.m):
            return (False, "SKIP: n or m not numeric")
        n_baseline = float(self.n)
        m_baseline = float(self.m)

        def v_from_n(n):
            return (self.hbar_val / (m_baseline * math.sqrt(3))) * \
                   math.pow(3 * math.pi**2 * n, 1.0/3.0)

        v_baseline = v_from_n(n_baseline)
        n_perturbed = n_baseline * 1.10
        v_perturbed = v_from_n(n_perturbed)
        if v_perturbed <= v_baseline:
            return (False, f"v with increased n ({v_perturbed}) not greater than baseline ({v_baseline})")
        return (True, "v increases with n (monotonic increase)")

    # --------------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # --------------------------------------------------------------------------
    def test_answer_sanity_v_finite_nonnegative(self):
        """Ensure v is finite and non‑negative."""
        if self.v is None:
            return (False, "SKIP: v not available")
        if self._is_numeric(self.v):
            v_val = float(self.v)
            if v_val < 0:
                return (False, f"v is negative: {v_val}")
            if not math.isfinite(v_val):
                return (False, f"v is not finite: {v_val}")
            # Also check v_h if numeric
            if self.v_h is not None and self._is_numeric(self.v_h):
                vh_val = float(self.v_h)
                if vh_val < 0:
                    return (False, f"v_h is negative: {vh_val}")
        return (True, "v (and v_h) are finite and non‑negative")

    # --------------------------------------------------------------------------
    # Test runner
    # --------------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all test methods (names starting with 'test_'),
        execute them, collect results, and return (all_pass, logs).
        """
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