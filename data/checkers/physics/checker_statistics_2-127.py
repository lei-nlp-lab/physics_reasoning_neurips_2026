import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for ideal gas mixing entropy problem.
    Extracts givens and results from solution_instance and runs tests.
    """

    def __init__(self, solution_instance):
        # Safely extract givens - use None if missing
        self.N = getattr(solution_instance, 'N', None)
        self.T = getattr(solution_instance, 'T', None)
        self.m = getattr(solution_instance, 'm', None)
        self.h = getattr(solution_instance, 'h', None)
        self.k = getattr(solution_instance, 'k', None)
        self.P1 = getattr(solution_instance, 'P1', None)
        self.P2 = getattr(solution_instance, 'P2', None)
        self.V1 = getattr(solution_instance, 'V1', None)
        self.V2 = getattr(solution_instance, 'V2', None)
        self.z0 = getattr(solution_instance, 'z0', None)
        self.S_int = getattr(solution_instance, 'S_int', None)

        # Extract results from pseudocode emit keys
        results = getattr(solution_instance, 'results', {})
        self.F = results.get('F', None)
        self.S_total = results.get('S_total', None)
        self.Delta_S = results.get('Delta_S', None)

    # ---------- Interface Tests ----------
    def test_interface_F_present(self):
        """Check that results['F'] exists."""
        if self.F is None:
            return (False, "SKIP: results['F'] missing")
        return (True, "results['F'] present")

    def test_interface_S_total_present(self):
        """Check that results['S_total'] exists."""
        if self.S_total is None:
            return (False, "SKIP: results['S_total'] missing")
        return (True, "results['S_total'] present")

    def test_interface_Delta_S_present(self):
        """Check that results['Delta_S'] exists."""
        if self.Delta_S is None:
            return (False, "SKIP: results['Delta_S'] missing")
        return (True, "results['Delta_S'] present")

    def test_interface_F_type(self):
        """F should be a numeric value (not a string)."""
        if self.F is None:
            return (False, "SKIP: F not available")
        if not isinstance(self.F, (int, float)):
            return (False, f"F has unsupported type {type(self.F)}")
        return (True, "F is numeric")

    def test_interface_S_total_type(self):
        """S_total should be numeric."""
        if self.S_total is None:
            return (False, "SKIP: S_total not available")
        if not isinstance(self.S_total, (int, float)):
            return (False, f"S_total has unsupported type {type(self.S_total)}")
        return (True, "S_total is numeric")

    def test_interface_Delta_S_type(self):
        """Delta_S should be numeric."""
        if self.Delta_S is None:
            return (False, "SKIP: Delta_S not available")
        if not isinstance(self.Delta_S, (int, float)):
            return (False, f"Delta_S has unsupported type {type(self.Delta_S)}")
        return (True, "Delta_S is numeric")

    # ---------- Invariant / Theorem Tests ----------
    def test_invariant_ideal_gas_law_P1V1(self):
        """Check P1 * V1 ≈ N * k * T (ideal gas law for first vessel)."""
        if any(v is None for v in [self.P1, self.V1, self.N, self.k, self.T]):
            return (False, "SKIP: missing givens for ideal gas law (P1,V1,N,k,T)")
        expected = self.N * self.k * self.T
        if not math.isclose(self.P1 * self.V1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P1*V1 = {self.P1*self.V1} not close to NkT = {expected}")
        return (True, "P1*V1 matches NkT (rel/abs tol 1e-2)")

    def test_invariant_ideal_gas_law_P2V2(self):
        """Check P2 * V2 ≈ N * k * T."""
        if any(v is None for v in [self.P2, self.V2, self.N, self.k, self.T]):
            return (False, "SKIP: missing givens for ideal gas law (P2,V2,N,k,T)")
        expected = self.N * self.k * self.T
        if not math.isclose(self.P2 * self.V2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P2*V2 = {self.P2*self.V2} not close to NkT = {expected}")
        return (True, "P2*V2 matches NkT (rel/abs tol 1e-2)")

    def test_invariant_lambda_computation(self):
        """Check that if m,h,k,T are given, we can compute lambda and compare if V1/present."""
        if any(v is None for v in [self.m, self.k, self.T, self.h]):
            return (False, "SKIP: missing m,k,T,h for lambda")
        lambda_ = self.h / math.sqrt(2 * math.pi * self.m * self.k * self.T)
        return (True, f"lambda computed = {lambda_:.6e} (no reference check)")

    def test_invariant_F_expression_check(self):
        """
        Rederive free energy F from givens and compare with result.
        Uses: F = -NkT ln( V1 / lambda^3 * z0 ) + kT ln(N!)
        Uses math.lgamma for ln(N!).
        Only if all necessary quantities are present.
        """
        if any(v is None for v in [self.N, self.T, self.k, self.V1, self.m, self.h, self.z0, self.F]):
            return (False, "SKIP: missing givens or result F for re-derivation")
        lambda_ = self.h / math.sqrt(2 * math.pi * self.m * self.k * self.T)
        ln_Nfact = math.lgamma(self.N + 1)  # ln(N!)
        expected_F = -self.N * self.k * self.T * math.log(self.V1 / (lambda_**3) * self.z0) + self.k * self.T * ln_Nfact
        if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"F = {self.F} not close to rederived {expected_F} (rel/abs tol 1e-2)")
        return (True, "F consistent with partition function re-derivation")

    def test_invariant_Delta_S_symmetry(self):
        """Delta_S should be symmetric under exchange of P1 and P2: ΔS = 2Nk ln((P1+P2)/(2√(P1P2)))."""
        if any(v is None for v in [self.N, self.k, self.P1, self.P2, self.Delta_S]):
            return (False, "SKIP: missing data for Delta_S")
        expected = 2 * self.N * self.k * math.log((self.P1 + self.P2) / (2 * math.sqrt(self.P1 * self.P2)))
        if not math.isclose(self.Delta_S, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Delta_S = {self.Delta_S} not close to symmetric expression {expected}")
        return (True, "Delta_S matches symmetric formula")

    def test_invariant_Delta_S_nonnegative(self):
        """Delta_S should be nonnegative (entropy never decreases)."""
        if self.Delta_S is None:
            return (False, "SKIP: Delta_S not available")
        if self.Delta_S < -1e-12:  # allow rounding
            return (False, f"Delta_S = {self.Delta_S} is negative")
        return (True, "Delta_S >= 0 (nonnegative)")

    def test_invariant_Delta_S_zero_if_equal_pressures(self):
        """If P1 == P2, Delta_S should be zero."""
        if any(v is None for v in [self.P1, self.P2, self.Delta_S]):
            return (False, "SKIP: missing pressures or Delta_S")
        if not math.isclose(self.P1, self.P2, rel_tol=1e-6, abs_tol=1e-6):
            return (False, "SKIP: P1 != P2, cannot test zero limit")
        if not math.isclose(self.Delta_S, 0.0, abs_tol=1e-10):
            return (False, f"Delta_S = {self.Delta_S} should be 0 when P1=P2")
        return (True, "Delta_S is zero when P1 = P2")

    def test_invariant_S_total_additivity(self):
        """
        Check that S_total = S1 + S2, where each Si is given by Sackur-Tetrode-like formula.
        Derive S1 and S2 from givens and compare to S_total.
        """
        if any(v is None for v in [self.N, self.k, self.T, self.V1, self.V2, self.m, self.h, self.S_int, self.S_total]):
            return (False, "SKIP: missing givens or S_total")
        lambda_ = self.h / math.sqrt(2 * math.pi * self.m * self.k * self.T)
        ln_term1 = math.log(self.V1 / (self.N * lambda_**3))
        S1 = self.N * self.k * (ln_term1 + 2.5 + self.S_int)
        ln_term2 = math.log(self.V2 / (self.N * lambda_**3))
        S2 = self.N * self.k * (ln_term2 + 2.5 + self.S_int)
        expected_S_total = S1 + S2
        if not math.isclose(self.S_total, expected_S_total, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S_total = {self.S_total} not close to sum S1+S2 = {expected_S_total}")
        return (True, "S_total consistent with additive Sackur-Tetrode entropies")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_F_logarithmic_structure(self):
        """
        Quick check that the free energy expression contains -NkT ln V1.
        We already verified exact value in invariant test, so skip.
        """
        return (True, "SKIP: covered by test_invariant_F_expression_check")

    def test_modeling_S_total_structure(self):
        """
        Check that S_total contains terms 2Nk ln(sqrt(V1V2)/N) + constant.
        Already verified in test_invariant_S_total_additivity.
        """
        return (True, "SKIP: covered by test_invariant_S_total_additivity")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_Delta_S_increases_with_ratio(self):
        """
        Perturb P1 by +5% and recompute expected Delta_S using formula.
        Compare with baseline (original expected Delta_S).
        For P1 > P2, increasing P1 should increase ΔS.
        For P1 < P2, decreasing P1 should increase ΔS.
        """
        if any(v is None for v in [self.N, self.k, self.P1, self.P2]):
            return (False, "SKIP: missing N, k, P1, P2 for monotonicity")
        baseline = 2 * self.N * self.k * math.log((self.P1 + self.P2) / (2 * math.sqrt(self.P1 * self.P2)))
        # Perturb P1 by +5%
        P1_pert = self.P1 * 1.05
        perturbed = 2 * self.N * self.k * math.log((P1_pert + self.P2) / (2 * math.sqrt(P1_pert * self.P2)))
        if self.P1 > self.P2:
            if perturbed < baseline - 1e-12:
                return (False, f"Monotonicity violation: increasing P1 decreased expected Delta_S from {baseline} to {perturbed}")
            return (True, "Delta_S increases when P1 increased (P1 > P2 case)")
        elif self.P1 < self.P2:
            # Decreasing P1 (ratio moves away from 1) should increase Delta_S
            P1_pert_down = self.P1 * 0.95
            perturbed_down = 2 * self.N * self.k * math.log((P1_pert_down + self.P2) / (2 * math.sqrt(P1_pert_down * self.P2)))
            if perturbed_down < baseline - 1e-12:
                return (False, f"Monotonicity violation: decreasing P1 decreased expected Delta_S from {baseline} to {perturbed_down}")
            return (True, "Delta_S increases when P1 decreased (P1 < P2 case)")
        else:
            return (True, "SKIP: P1 = P2, derivative is zero")

    # ---------- Answer Sanity Tests (single weak check) ----------
    def test_answer_sanity_basic_checks(self):
        """
        Weak sanity: all three result quantities should be finite and nonnegative (where applicable).
        No re-derivation allowed.
        """
        checks = []
        # F
        if self.F is not None:
            if math.isnan(self.F) or math.isinf(self.F):
                checks.append("F is NaN or Inf")
        # S_total
        if self.S_total is not None:
            if math.isnan(self.S_total) or math.isinf(self.S_total):
                checks.append("S_total is NaN or Inf")
            elif self.S_total < 0:
                checks.append("S_total is negative")
        # Delta_S
        if self.Delta_S is not None:
            if math.isnan(self.Delta_S) or math.isinf(self.Delta_S):
                checks.append("Delta_S is NaN or Inf")
            elif self.Delta_S < -1e-12:
                checks.append("Delta_S is negative")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All available results are finite and nonnegative")

    # ---------- Main Runner ----------
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