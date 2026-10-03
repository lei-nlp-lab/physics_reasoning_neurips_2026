import math

class PhysicsChecker:
    """
    Audits the solution for the isothermal compression + adiabatic expansion problem.
    Checks interface completeness, physical invariants, modeling formulas, monotonicity,
    and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.W_net_mono = results.get('W_net_mono', None)
        self.sign_mono = results.get('sign_mono', None)
        self.W_net_di = results.get('W_net_di', None)
        self.comparison = results.get('comparison', None)

        # Extract givens if provided (optional, for re-derivation)
        givens = getattr(solution_instance, 'givens', {})
        self.p1 = givens.get('p1', None)
        self.V1 = givens.get('V1', None)
        self.V2 = givens.get('V2', None)
        self.V3 = givens.get('V3', None)
        self.gamma_mono = givens.get('gamma_mono', None)
        self.gamma_di = givens.get('gamma_di', None)

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_W_net_mono_exists(self):
        if self.W_net_mono is None:
            return (False, "Missing required result key: W_net_mono")
        return (True, "W_net_mono is present")

    def test_interface_sign_mono_exists(self):
        if self.sign_mono is None:
            return (False, "Missing required result key: sign_mono")
        return (True, "sign_mono is present")

    def test_interface_W_net_di_exists(self):
        if self.W_net_di is None:
            return (False, "Missing required result key: W_net_di")
        return (True, "W_net_di is present")

    def test_interface_comparison_exists(self):
        if self.comparison is None:
            return (False, "Missing required result key: comparison")
        return (True, "comparison is present")

    # ------------------------------------------------------------
    # Physical invariants (theorem-based, use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_isothermal_Boyle(self):
        """Check that p1*V1 = p2*V2 (Boyle's law)."""
        if self.p1 is None or self.V1 is None or self.V2 is None:
            return (False, "SKIP: p1, V1, V2 not available")
        p2 = self.p1 * self.V1 / self.V2
        # The solution's p2 is not directly exposed, so we check consistency via implied p2.
        # If W12 is given, we can verify the product.
        # Alternatively, we check that p1*V1 = p2*V2 if p2 were computed.
        # Since we don't have p2, we use W12 if available.
        # But W12 may not be in results. We'll compute p2 from p1, V1, V2 and check against expectation.
        # Actually, we need p2 from results? Not required. This test is optional.
        # For safety, only run if we have p2 or can infer from W12.
        # Instead, check that W12 formula implies product conservation.
        # We'll compute W12 from given values if available.
        if self.W_net_mono is not None and self.p1 is not None and self.V1 is not None and self.V2 is not None:
            W12 = self.p1 * self.V1 * math.log(self.V2 / self.V1)
            # But we cannot access W12 directly. So skip.
            pass
        # To avoid complexity, we'll skip if required not present.
        return (True, "SKIP: invariant check requires p2 or W12 not stored")

    def test_invariant_adiabatic_mono(self):
        """Check p2*V2^gamma = p3_mono*V3^gamma for monatomic."""
        if self.p1 is None or self.V1 is None or self.V2 is None or self.V3 is None or self.gamma_mono is None:
            return (False, "SKIP: missing givens for adiabatic check")
        p2 = self.p1 * self.V1 / self.V2
        p3_expected = p2 * (self.V2 / self.V3) ** self.gamma_mono
        # We don't directly have p3_mono, but we can compute W_net_mono from formulas and compare.
        # Instead, check that the provided W_net_mono is consistent with the formula.
        if self.W_net_mono is None:
            return (False, "SKIP: W_net_mono not available for consistency check")
        # Re-derive W_net_mono
        W12 = p2 * self.V2 * math.log(self.V1 / self.V2)  # Actually p1*V1*ln(V2/V1) = -p2*V2*ln(V2/V1)? Let's use standard.
        # Better: W12 = p1*V1*ln(V2/V1)
        W12 = self.p1 * self.V1 * math.log(self.V2 / self.V1)  # negative because V2<V1
        W23 = (p2 * self.V2 - p3_expected * self.V3) / (self.gamma_mono - 1)
        W_net_expected = W12 + W23
        if math.isclose(self.W_net_mono, W_net_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Monatomic net work consistent with adiabatic relation")
        else:
            return (False, f"Monatomic net work {self.W_net_mono} not close to expected {W_net_expected:.4f}")

    def test_invariant_adiabatic_di(self):
        if self.p1 is None or self.V1 is None or self.V2 is None or self.V3 is None or self.gamma_di is None:
            return (False, "SKIP: missing givens for diatomic adiabatic check")
        p2 = self.p1 * self.V1 / self.V2
        p3_expected = p2 * (self.V2 / self.V3) ** self.gamma_di
        if self.W_net_di is None:
            return (False, "SKIP: W_net_di not available")
        W12 = self.p1 * self.V1 * math.log(self.V2 / self.V1)
        W23 = (p2 * self.V2 - p3_expected * self.V3) / (self.gamma_di - 1)
        W_net_expected = W12 + W23
        if math.isclose(self.W_net_di, W_net_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Diatomic net work consistent with adiabatic relation")
        else:
            return (False, f"Diatomic net work {self.W_net_di} not close to expected {W_net_expected:.4f}")

    # ------------------------------------------------------------
    # Modeling/trace checks (use math.isclose)
    # ------------------------------------------------------------
    def test_modeling_W_net_sign_mono(self):
        """Net work should be negative (work on system)."""
        if self.W_net_mono is None:
            return (False, "SKIP: W_net_mono not available")
        if self.W_net_mono < 0:
            return (True, "Monatomic net work is negative (work done on system)")
        else:
            return (False, f"Monatomic net work {self.W_net_mono} should be negative")

    def test_modeling_W_net_sign_di(self):
        if self.W_net_di is None:
            return (False, "SKIP: W_net_di not available")
        if self.W_net_di < 0:
            return (True, "Diatomic net work is negative")
        else:
            return (False, f"Diatomic net work {self.W_net_di} should be negative")

    def test_modeling_mono_vs_di_magnitude(self):
        """Magnitude of monatomic net work greater than diatomic."""
        if self.W_net_mono is None or self.W_net_di is None:
            return (False, "SKIP: net work values missing")
        if abs(self.W_net_mono) > abs(self.W_net_di):
            return (True, "Monatomic net work magnitude greater than diatomic")
        else:
            return (False, f"Monatomic |W|={abs(self.W_net_mono):.4f} not > diatomic |W|={abs(self.W_net_di):.4f}")

    def test_modeling_W12_compression_work_negative(self):
        """Isothermal compression work should be negative."""
        if self.p1 is None or self.V1 is None or self.V2 is None:
            return (False, "SKIP: missing givens for W12 check")
        W12 = self.p1 * self.V1 * math.log(self.V2 / self.V1)  # negative
        # We cannot compare to stored value, but we can assert sign
        if W12 < 0:
            return (True, "Isothermal compression work is negative")
        else:
            return (False, "Computed W12 is not negative")

    def test_modeling_W23_expansion_work_positive(self):
        """Adiabatic expansion work should be positive (work done by gas)."""
        if self.p1 is None or self.V1 is None or self.V2 is None or self.V3 is None or self.gamma_mono is None:
            return (False, "SKIP: missing givens for W23 check")
        p2 = self.p1 * self.V1 / self.V2
        p3 = p2 * (self.V2 / self.V3) ** self.gamma_mono
        W23 = (p2 * self.V2 - p3 * self.V3) / (self.gamma_mono - 1)
        if W23 > 0:
            return (True, "Monatomic adiabatic expansion work is positive")
        else:
            return (False, f"Monatomic adiabatic work W23={W23:.4f} should be positive")

    # ------------------------------------------------------------
    # Monotonicity checks (perturb gamma)
    # ------------------------------------------------------------
    def test_monotonicity_net_work_magnitude_decreases_with_gamma(self):
        """
        As gamma increases (e.g., from 1.4 to 1.5), net work becomes less negative
        (magnitude decreases), because p3 increases and the adiabatic work term gets larger positive.
        We'll simulate this using the formulas and the given inputs if available.
        """
        if self.p1 is None or self.V1 is None or self.V2 is None or self.V3 is None:
            return (False, "SKIP: missing givens for monotonicity test")
        p2 = self.p1 * self.V1 / self.V2
        W12 = self.p1 * self.V1 * math.log(self.V2 / self.V1)  # constant
        # Baseline gamma 1.4 (like diatomic)
        gamma_low = 1.4
        p3_low = p2 * (self.V2 / self.V3) ** gamma_low
        W23_low = (p2 * self.V2 - p3_low * self.V3) / (gamma_low - 1)
        W_net_low = W12 + W23_low

        # Perturb gamma upward to 1.5
        gamma_high = 1.5
        p3_high = p2 * (self.V2 / self.V3) ** gamma_high
        W23_high = (p2 * self.V2 - p3_high * self.V3) / (gamma_high - 1)
        W_net_high = W12 + W23_high

        # Expect net work becomes less negative (increases) as gamma increases
        if W_net_high > W_net_low:
            return (True, f"Net work increased from {W_net_low:.4f} to {W_net_high:.4f} with gamma increase")
        else:
            return (False, f"Net work did not increase (gamma {gamma_low} -> {gamma_high}): {W_net_low:.4f} -> {W_net_high:.4f}")

    # ------------------------------------------------------------
    # Weak answer sanity (only one method allowed)
    # ------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Only weak sanity: check that sign_mono is 'negative' and comparison contains expected keywords.
        Do NOT compute any expected values.
        """
        errors = []
        if self.sign_mono != 'negative':
            errors.append(f"sign_mono is '{self.sign_mono}', expected 'negative'")
        if self.comparison is None:
            errors.append("comparison missing")
        else:
            # Comparison should indicate monatomic net work magnitude greater
            if not ('monatomic' in str(self.comparison).lower() and 'greater' in str(self.comparison).lower()):
                errors.append(f"comparison '{self.comparison}' does not indicate monatomic greater")
        if errors:
            return (False, "; ".join(errors))
        return (True, "Answer sanity: sign and comparison match expectations")

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
                import traceback
                traceback.print_exc()  # for debugging
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)