import math
import traceback

class PhysicsChecker:
    """
    Auditor for heat-transfer solution: steady-state convection+conduction through a window pane.
    Checks interface completeness, invariant flux relations, modeling consistency, and
    monotonic trends. Uses safe defaults and lenient tolerance for invariant/modeling tests.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        results = {}
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        elif isinstance(solution_instance, dict):
            results = solution_instance

        # Required result keys (from pseudocode emit contract)
        self.T1 = results.get('T1', None)

        # Extract givens from results or solution_instance attributes
        self.h = results.get('h', None) if results.get('h') is not None else getattr(solution_instance, 'h', None)
        self.k = results.get('k', None) if results.get('k') is not None else getattr(solution_instance, 'k', None)
        self.L = results.get('L', None) if results.get('L') is not None else getattr(solution_instance, 'L', None)
        self.T_in = results.get('T_in', None) if results.get('T_in') is not None else getattr(solution_instance, 'T_in', None)
        self.T_out = results.get('T_out', None) if results.get('T_out') is not None else getattr(solution_instance, 'T_out', None)

        # Compute derived quantities only if givens are available
        self.alpha = None
        if self.k is not None and self.L is not None and self.L != 0:
            self.alpha = self.k / self.L

    def run_all_tests(self):
        """Discover and run all test_ methods, collect logs."""
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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_T1_exists(self):
        if self.T1 is None:
            return False, "T1 not found in results"
        if not isinstance(self.T1, (int, float)):
            return False, f"T1 has non-numeric type {type(self.T1)}"
        return True, "T1 is present and numeric"

    def test_interface_givens_complete(self):
        missing = []
        for name in ['h', 'k', 'L', 'T_in', 'T_out']:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
        if missing:
            return False, f"Missing givens: {missing}"
        return True, "All givens present"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_inner_convection_vs_conduction(self):
        """Check that heat flux from inner convection equals conduction through pane."""
        if any(v is None for v in (self.T1, self.h, self.alpha, self.T_in)):
            return False, "SKIP: missing data"
        # Compute T2 from the equality of convection rates (temperature differences equal)
        T2 = self.T_in + self.T_out - self.T1  # from 25 - T1 = T2 + 15
        # Convection inside: q_conv_in = h * (T_in - T1)^(5/4)
        dt_in = self.T_in - self.T1
        if dt_in < 0:
            return False, f"T_in - T1 negative ({dt_in}), invalid"
        q_conv_in = self.h * (dt_in ** 1.25)
        # Conduction: q_cond = alpha * (T1 - T2)
        q_cond = self.alpha * (self.T1 - T2)
        if not math.isclose(q_conv_in, q_cond, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"q_conv_in={q_conv_in:.6e} vs q_cond={q_cond:.6e} mismatch"
        return True, f"q_conv_in ≈ q_cond within tolerance"

    def test_invariant_outer_convection_vs_conduction(self):
        """Check heat flux from outer convection matches conduction."""
        if any(v is None for v in (self.T1, self.h, self.alpha, self.T_in, self.T_out)):
            return False, "SKIP: missing data"
        T2 = self.T_in + self.T_out - self.T1
        dt_out = T2 - self.T_out
        if dt_out < 0:
            return False, f"T2 - T_out negative ({dt_out}), invalid"
        q_conv_out = self.h * (dt_out ** 1.25)
        q_cond = self.alpha * (self.T1 - T2)
        if not math.isclose(q_conv_out, q_cond, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"q_conv_out={q_conv_out:.6e} vs q_cond={q_cond:.6e} mismatch"
        return True, f"q_conv_out ≈ q_cond within tolerance"

    def test_invariant_all_fluxes_equal(self):
        """All three heat flux expressions should be equal."""
        if any(v is None for v in (self.T1, self.h, self.alpha, self.T_in, self.T_out)):
            return False, "SKIP: missing data"
        T2 = self.T_in + self.T_out - self.T1
        dt_in = self.T_in - self.T1
        dt_out = T2 - self.T_out
        if dt_in < 0 or dt_out < 0 or (self.T1 - T2) < 0:
            return False, "Temperature differences inconsistent"
        q1 = self.h * (dt_in ** 1.25)
        q2 = self.alpha * (self.T1 - T2)
        q3 = self.h * (dt_out ** 1.25)
        if not (math.isclose(q1, q2, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(q2, q3, rel_tol=1e-2, abs_tol=1e-2)):
            return False, f"Fluxes differ: q1={q1:.6e}, q2={q2:.6e}, q3={q3:.6e}"
        return True, "All three fluxes equal within tolerance"

    # ------------------------------------------------------------------
    # Modeling consistency tests (derived from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_equal_temperature_differences(self):
        """Check that (T_in - T1) and (T2 - T_out) are equal (because same convection law and coefficient)."""
        if any(v is None for v in (self.T1, self.T_in, self.T_out)):
            return False, "SKIP: missing temperature data"
        T2 = self.T_in + self.T_out - self.T1
        diff_in = self.T_in - self.T1
        diff_out = T2 - self.T_out
        if not math.isclose(diff_in, diff_out, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Temperature differences not equal: {diff_in} vs {diff_out}"
        return True, "Temperature differences equal (consistent with equal convection fluxes)"

    def test_modeling_positivity_and_bounds(self):
        """Check that T1 is between T_out and T_in, and heat flux positive."""
        if any(v is None for v in (self.T1, self.T_in, self.T_out)):
            return False, "SKIP: missing temperatures"
        if self.T1 < self.T_out:
            return False, f"T1 ({self.T1}) < T_out ({self.T_out})"
        if self.T1 > self.T_in:
            return False, f"T1 ({self.T1}) > T_in ({self.T_in})"
        # Check that T_in - T1 and T1 - T_out are positive (since T_out < T2 < T1 < T_in)
        if self.T_in - self.T1 <= 0:
            return False, "T_in - T1 not positive"
        if self.T1 - self.T_out <= 0:
            return False, "T1 - T_out not positive"
        return True, "T1 is within expected bounds and temperature differences positive"

    def test_modeling_flux_sign(self):
        """Heat flux must be positive."""
        if any(v is None for v in (self.T1, self.h, self.alpha, self.T_in)):
            return False, "SKIP: missing data"
        T2 = self.T_in + self.T_out - self.T1
        q_cond = self.alpha * (self.T1 - T2) if self.alpha else None
        if q_cond is not None and q_cond <= 0:
            return False, f"Heat flux is zero or negative: {q_cond}"
        return True, "Heat flux positive"

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks via parameter perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_T1_increases_with_internal_temp(self):
        """
        Increase T_in by a small amount and verify that the residual equation
        at the given T1 becomes negative, indicating that equilibrium would require a higher T1.
        """
        if any(v is None for v in (self.T1, self.h, self.alpha, self.T_in, self.T_out)):
            return False, "SKIP: missing data"
        # Residual at original T_in: R(T1) = alpha*(2*T1-10) - h*(T_in - T1)^(5/4)
        # For the given T1, compute residual
        T1 = self.T1
        dt_in = self.T_in - T1
        if dt_in < 0:
            return False, "Original dt_in negative"
        R_orig = self.alpha * (2 * T1 - 10) - self.h * (dt_in ** 1.25)
        # Increase T_in by 1°C
        T_in_new = self.T_in + 1.0
        dt_in_new = T_in_new - T1
        if dt_in_new < 0:
            return False, "Perturbed dt_in negative, cannot test"
        R_new = self.alpha * (2 * T1 - 10) - self.h * (dt_in_new ** 1.25)
        # Expect R_new < R_orig (R becomes more negative) because right side increased
        # For an increase in T_in, the equilibrium T1 should increase; at the original T1,
        # the new R is more negative if the function f(T1) = alpha*(2T1-10) - h*(T_in - T1)^(5/4)
        # is decreasing in T_in? Actually, with T1 fixed, increasing T_in makes (T_in - T1) larger,
        # so (T_in - T1)^(5/4) larger, so subtracting a larger number makes R smaller (more negative).
        # So we check R_new < R_orig.
        if not (R_new < R_orig):
            return False, f"R_new ({R_new:.4f}) not less than R_orig ({R_orig:.4f}), monotonicity fails"
        return True, f"T1 increases with T_in: R_orig={R_orig:.4f}, R_new={R_new:.4f}"

    def test_monotonicity_T1_increases_with_conductivity(self):
        """
        Increase k (thermal conductivity) by 10% and check that the conduction flux
        at the given T1 becomes larger, causing the residual to become positive,
        indicating that T1 must increase to rebalance.
        """
        if any(v is None for v in (self.T1, self.h, self.k, self.L, self.T_in, self.T_out)):
            return False, "SKIP: missing data"
        # Original alpha = k / L
        alpha_orig = self.k / self.L
        # new k = k * 1.1
        k_new = self.k * 1.1
        alpha_new = k_new / self.L
        # Residual at original T1 and original alpha: R_orig = alpha*(2T1-10) - h*(T_in-T1)^(5/4)
        # At new alpha (keeping T1 same), R_new = alpha_new*(2T1-10) - h*(T_in-T1)^(5/4)
        dt_in = self.T_in - self.T1
        if dt_in < 0:
            return False, "dt_in negative"
        R_orig = alpha_orig * (2 * self.T1 - 10) - self.h * (dt_in ** 1.25)
        R_new = alpha_new * (2 * self.T1 - 10) - self.h * (dt_in ** 1.25)
        # For increased k, conduction flux increases, so at fixed T1, R becomes more positive.
        # Then to bring R back to zero, T1 must increase. So R_new > R_orig.
        if not (R_new > R_orig):
            return False, f"R_new ({R_new:.4f}) not greater than R_orig ({R_orig:.4f})"
        return True, f"T1 increases with k: R_orig={R_orig:.4f}, R_new={R_new:.4f}"

    # ------------------------------------------------------------------
    # Weak answer sanity test (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_simple_ranges(self):
        """Broad sign/finite check on T1."""
        if self.T1 is None:
            return False, "SKIP: T1 missing"
        # Must be a finite number
        if not math.isfinite(self.T1):
            return False, f"T1 is not finite: {self.T1}"
        # Within physically plausible range: between -273°C and 100°C
        if not (-273 <= self.T1 <= 100):
            return False, f"T1 = {self.T1} outside plausible range [-273, 100]"
        return True, "T1 is finite and within physically plausible range"