import math
import traceback

class PhysicsChecker:
    """
    Auditor for cylindrical capacitor problem.
    Checks Gauss's law, potential boundary conditions, energy consistency,
    and monotonicity of physical quantities.
    """

    def __init__(self, solution_instance):
        # Extract given inputs (safe defaults: None)
        self.r1 = getattr(solution_instance, 'r1', None)
        self.r2 = getattr(solution_instance, 'r2', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.Q = getattr(solution_instance, 'Q', None)
        self.r = getattr(solution_instance, 'r', None)  # point where Q is initially

        # Extract results (safe defaults: None)
        results = getattr(solution_instance, 'results', {})
        self.E_r = results.get('E_r', None)
        self.V_r = results.get('V_r', None)
        self.Delta_Q = results.get('Delta_Q', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required givens and results are present and numeric."""
        missing = []
        for name in ['r1', 'r2', 'V0', 'Q', 'r']:
            if getattr(self, name) is None:
                missing.append(name)
        for name in ['E_r', 'V_r', 'Delta_Q']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing required data: {', '.join(missing)}")
        # Check numeric types (allow int or float)
        for name in ['r1', 'r2', 'V0', 'Q', 'r', 'E_r', 'V_r', 'Delta_Q']:
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (type: {type(val).__name__})")
        return (True, "All required keys present and numeric.")

    # ------------------------------------------------------------------
    # Invariant / theorem tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_gauss_law_product(self):
        """
        Gauss's law for cylindrical symmetry: E(r) * r = constant.
        Constant = V0 / ln(r2/r1). We check |E_r * r - V0/ln(r2/r1)| < 1e-2.
        """
        if None in (self.E_r, self.r, self.V0, self.r1, self.r2):
            return (False, "SKIP: missing data for Gauss law product")
        try:
            expected_const = self.V0 / math.log(self.r2 / self.r1)
            actual_product = self.E_r * self.r
            if math.isclose(actual_product, expected_const, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E_r * r ≈ {expected_const:.4f} (actual {actual_product:.4f})")
            else:
                return (False, f"E_r * r = {actual_product:.4f}, expected ≈ {expected_const:.4f}")
        except Exception as e:
            return (False, f"Gauss law product check crashed: {e}")

    def test_invariant_potential_difference(self):
        """
        Potential difference from r1 to r should satisfy:
        V0 - V(r) = integral_{r1}^{r} E(r') dr' = (V0/ln(r2/r1)) * ln(r/r1)
        So V(r) = V0 * ln(r2/r) / ln(r2/r1).
        We check that V_r is close to this expression.
        """
        if None in (self.V_r, self.V0, self.r, self.r1, self.r2):
            return (False, "SKIP: missing data for potential difference")
        try:
            expected_V = self.V0 * math.log(self.r2 / self.r) / math.log(self.r2 / self.r1)
            if math.isclose(self.V_r, expected_V, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"V(r) ≈ {expected_V:.4f} (actual {self.V_r:.4f})")
            else:
                return (False, f"V(r) = {self.V_r:.4f}, expected ≈ {expected_V:.4f}")
        except Exception as e:
            return (False, f"Potential difference check crashed: {e}")

    def test_invariant_charge_change_energy(self):
        """
        Energy conservation: work done by battery = V0 * ΔQ
        = work done on moving charge Q: Q*(V0 - V(r)).
        So ΔQ = Q*(V0 - V(r))/V0.
        Checks that Delta_Q ≈ Q * (V0 - V_r) / V0.
        """
        if None in (self.Delta_Q, self.Q, self.V0, self.V_r):
            return (False, "SKIP: missing data for energy check")
        try:
            expected_DQ = self.Q * (self.V0 - self.V_r) / self.V0
            if math.isclose(self.Delta_Q, expected_DQ, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"ΔQ ≈ {expected_DQ:.4f} (actual {self.Delta_Q:.4f})")
            else:
                return (False, f"ΔQ = {self.Delta_Q:.4f}, expected ≈ {expected_DQ:.4f}")
        except Exception as e:
            return (False, f"Energy consistency check crashed: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (use math.isclose, check structural relations)
    # ------------------------------------------------------------------
    def test_modeling_electric_field_structure(self):
        """
        E(r) should be positive and decreasing with r.
        We check that E_r > 0 and that derivative (E_r - E(r+δ)) would be negative.
        Since we only have one point, we test monotonicity of the expected function:
        The formula implies E_r * r = constant, so E_r decreases as r increases.
        A minimal check: if we had a second point, E would be smaller.
        Here we just check positivity and the product invariant (already done).
        """
        if self.E_r is None:
            return (False, "SKIP: E_r not available")
        if self.E_r <= 0:
            return (False, f"E_r should be positive, got {self.E_r}")
        return (True, "E_r > 0 (consistent with positive inner potential)")

    def test_modeling_potential_bounds(self):
        """
        Potential should be between 0 and V0, and monotonic.
        V(r1)=V0, V(r2)=0, and for r1<r<r2, V(r) should be positive and less than V0.
        """
        if None in (self.V_r, self.V0, self.r1, self.r2):
            return (False, "SKIP: missing data for bound checks")
        if not (0 < self.V_r < self.V0):
            return (False, f"V_r = {self.V_r} is not between 0 and V0 = {self.V0}")
        return (True, f"V_r = {self.V_r} lies in (0, V0)")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_Vr_decreases_with_r(self):
        """
        As r increases (moves toward outer conductor), potential V(r) should decrease.
        We simulate perturbing r by a small positive delta and recompute V using the
        formula derived from physics (V0, r1, r2 known). Then check that the perturbed
        V is less than the solution's V_r.
        """
        if None in (self.V_r, self.V0, self.r1, self.r2, self.r):
            return (False, "SKIP: missing data for monotonicity test")
        # Choose a small perturbation (1% of (r2-r1) but at least 0.1)
        delta = max(0.01 * (self.r2 - self.r1), 0.1)
        r_perturbed = self.r + delta
        # Ensure perturbed r is still within (r1, r2)
        if r_perturbed >= self.r2:
            # If too close to outer, use a smaller or inward perturbation
            r_perturbed = self.r - delta
            if r_perturbed <= self.r1:
                return (False, "Unable to find valid perturbation; r too close to boundaries")
        try:
            # Compute V at perturbed position using the same physics formula
            V_perturbed = self.V0 * math.log(self.r2 / r_perturbed) / math.log(self.r2 / self.r1)
            # For increasing r, V should decrease
            if r_perturbed > self.r:
                if V_perturbed < self.V_r:
                    return (True, f"V decreases when r increases: V({r_perturbed:.3f})={V_perturbed:.4f} < V({self.r})={self.V_r:.4f}")
                else:
                    return (False, f"V did not decrease when r increased: V_perturbed={V_perturbed:.4f} >= V={self.V_r:.4f}")
            else:  # perturbed inward
                if V_perturbed > self.V_r:
                    return (True, f"V increases when r decreases: V({r_perturbed:.3f})={V_perturbed:.4f} > V({self.r})={self.V_r:.4f}")
                else:
                    return (False, f"V did not increase when r decreased: V_perturbed={V_perturbed:.4f} <= V={self.V_r:.4f}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, no math.isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_charge_sign(self):
        """
        Since Q is negative and r>r1 → ln(r/r1)>0 → ΔQ should be negative.
        Also ΔQ should be finite (not NaN).
        """
        if self.Delta_Q is None:
            return (False, "SKIP: Delta_Q not available")
        if not math.isfinite(self.Delta_Q):
            return (False, f"Delta_Q is not finite: {self.Delta_Q}")
        if self.Delta_Q >= 0:
            return (False, f"Delta_Q should be negative (Q negative), got {self.Delta_Q}")
        return (True, f"Delta_Q negative as expected: {self.Delta_Q}")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Run all test_ methods and return (all_pass, logs)."""
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

# Example usage (uncomment to test with a dummy instance):
# class DummySolution:
#     def __init__(self):
#         self.r1 = 1.0
#         self.r2 = 3.0
#         self.V0 = 10.0
#         self.Q = -1e-6
#         self.r = 2.0
#         self.results = {
#             'E_r': 10.0/(2.0*math.log(3.0/1.0)),  # approx 4.551
#             'V_r': 10.0*math.log(3.0/2.0)/math.log(3.0/1.0),  # approx 3.69
#             'Delta_Q': -1e-6 * math.log(2.0/1.0)/math.log(3.0/1.0)  # approx -6.30e-7
#         }
# checker = PhysicsChecker(DummySolution())
# all_pass, logs = checker.run_all_tests()
# for log in logs:
#     print(log)