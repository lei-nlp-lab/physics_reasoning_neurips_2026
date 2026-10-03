import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for the paramagnetic adiabatic demagnetization problem.
    """

    def __init__(self, solution_instance):
        # Safely extract givens and results; always use None for missing values
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        if isinstance(self.givens, dict):
            self.T_i = self.givens.get('T_i', None)
            self.H_i_given = self.givens.get('H_i', None)
            self.a = self.givens.get('a', None)
            self.b = self.givens.get('b', None)
        else:
            self.T_i = getattr(solution_instance, 'T_i', None)
            self.H_i_given = getattr(solution_instance, 'H_i', None)
            self.a = getattr(solution_instance, 'a', None)
            self.b = getattr(solution_instance, 'b', None)

        if isinstance(self.results, dict):
            self.T_f = self.results.get('T_f', None)
            self.H_i_result = self.results.get('H_i', None)
            self.T_i_relation = self.results.get('T_i_relation', None)
        else:
            self.T_f = getattr(solution_instance, 'T_f', None)
            self.H_i_result = getattr(solution_instance, 'H_i', None)
            self.T_i_relation = getattr(solution_instance, 'T_i_relation', None)

    def _data_present(self, *args):
        """Return True only if all arguments are not None."""
        return all(arg is not None for arg in args)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Verify that all required result keys exist in self.results."""
        if not isinstance(self.results, dict):
            return (False, "SKIP: results is not a dictionary")
        required = {'T_f', 'T_i_relation', 'H_i'}
        missing = required - self.results.keys()
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present in results")

    def test_interface_givens_types(self):
        """Check that numeric givens are numeric (int or float)."""
        numeric_vars = [('T_i', self.T_i), ('H_i_given', self.H_i_given),
                        ('a', self.a), ('b', self.b)]
        for name, val in numeric_vars:
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (type {type(val).__name__})")
        return (True, "Numeric givens have correct types")

    # ------------------------------------------------------------------
    # Theorem‑based invariant tests (use math.isclose with tolerances)
    # ------------------------------------------------------------------
    def test_invariant_adiabatic_constant(self):
        """
        Adiabatic invariance: b*ln(T) - (a/2)*H^2 is constant
        along the process. Verify at initial and final states.
        """
        if not self._data_present(self.b, self.a, self.T_i, self.H_i_given, self.T_f):
            return (False, "SKIP: missing required values")
        try:
            C_initial = self.b * math.log(self.T_i) - 0.5 * self.a * (self.H_i_given ** 2)
            C_final   = self.b * math.log(self.T_f)   # H_f = 0
        except Exception as e:
            return (False, f"Computation error: {e}")
        if math.isclose(C_initial, C_final, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Adiabatic constant holds: C_initial={C_initial:.4e}, C_final={C_final:.4e}")
        else:
            return (False, f"Invariant violated: C_initial={C_initial:.4e}, C_final={C_final:.4e}")

    def test_invariant_temperature_ratio(self):
        """
        From the integrated condition: b*ln(T_f/T_i) = -(a/2)*H_i^2.
        This is a direct consequence of the adiabatic invariant.
        """
        if not self._data_present(self.b, self.a, self.T_i, self.H_i_given, self.T_f):
            return (False, "SKIP: missing required values")
        try:
            lhs = self.b * math.log(self.T_f / self.T_i)
            rhs = -0.5 * self.a * (self.H_i_given ** 2)
        except Exception as e:
            return (False, f"Computation error: {e}")
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Ratio invariant holds: lhs={lhs:.4e}, rhs={rhs:.4e}")
        else:
            return (False, f"Ratio invariant violated: lhs={lhs:.4e}, rhs={rhs:.4e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (also use math.isclose)
    # ------------------------------------------------------------------
    def test_modeling_T_f_expression(self):
        """Verify T_f = T_i * exp( -a/(2b) * H_i_given^2 )."""
        if not self._data_present(self.b, self.a, self.T_i, self.H_i_given, self.T_f):
            return (False, "SKIP: missing required values")
        try:
            expected = self.T_i * math.exp(-0.5 * self.a / self.b * self.H_i_given ** 2)
        except Exception as e:
            return (False, f"Computation error: {e}")
        if math.isclose(self.T_f, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T_f matches expected expression within tolerance")
        else:
            return (False, f"T_f mismatch: expected {expected:.4e}, got {self.T_f:.4e}")

    def test_modeling_T_i_relation_consistency(self):
        """
        Check that T_i_relation is consistent with the adiabatic relation:
        T_i = T_f * exp( a/(2b) * H_i_result^2 ).
        """
        if not self._data_present(self.b, self.a, self.T_i, self.H_i_result, self.T_f):
            return (False, "SKIP: missing required values")
        try:
            expected_T_i = self.T_f * math.exp(0.5 * self.a / self.b * self.H_i_result ** 2)
        except Exception as e:
            return (False, f"Computation error: {e}")
        if math.isclose(self.T_i, expected_T_i, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "T_i relation is consistent")
        else:
            return (False, f"T_i relation inconsistent: expected {expected_T_i:.4e}, got {self.T_i:.4e}")

    def test_modeling_H_i_from_T_f_ratio(self):
        """For the second part (T_f = T_i/2), check H_i_result satisfies H_i^2 = (2b/a)*ln(2)."""
        if not self._data_present(self.b, self.a, self.T_i, self.T_f, self.H_i_result):
            return (False, "SKIP: missing required values")
        ratio = self.T_f / self.T_i if self.T_i != 0 else None
        if ratio is None:
            return (False, "T_i is zero")
        # This test only applies when the ratio is close to 0.5
        if not math.isclose(ratio, 0.5, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"SKIP: T_f/T_i = {ratio:.4f} not ~0.5, part 2 not applicable")
        try:
            expected_sq = (2.0 * self.b / self.a) * math.log(2)
            actual_sq = self.H_i_result ** 2
        except Exception as e:
            return (False, f"Computation error: {e}")
        if math.isclose(actual_sq, expected_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"H_i^2 matches expected value for T_f = T_i/2")
        else:
            return (False, f"H_i^2 mismatch: expected {expected_sq:.4e}, got {actual_sq:.4e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (must explicitly perturb an input)
    # ------------------------------------------------------------------
    def test_monotonicity_final_temp_decreases_with_initial_field(self):
        """
        If H_i is increased by 5%, the final temperature should decrease.
        Compute T_f for the perturbed H_i using the re‑derived expression.
        """
        if not self._data_present(self.T_i, self.H_i_given, self.T_f, self.a, self.b):
            return (False, "SKIP: missing required values")
        try:
            # Baseline using given H_i
            T_f_baseline = self.T_i * math.exp(-0.5 * self.a / self.b * self.H_i_given ** 2)
            # Perturb H_i upward by 5%
            H_i_pert = self.H_i_given * 1.05
            T_f_pert = self.T_i * math.exp(-0.5 * self.a / self.b * H_i_pert ** 2)
        except Exception as e:
            return (False, f"Computation error: {e}")
        if T_f_pert < T_f_baseline:
            return (True, "Increasing H_i decreases T_f (monotonicity holds)")
        else:
            return (False, f"Monotonicity violated: T_f_baseline={T_f_baseline:.4e}, T_f_pert={T_f_pert:.4e}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, no math.isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_H_i_positive_finite(self):
        """Check that H_i_result is positive and finite."""
        if self.H_i_result is None:
            return (False, "SKIP: H_i result not present")
        if not math.isfinite(self.H_i_result):
            return (False, f"H_i result is not finite: {self.H_i_result}")
        if self.H_i_result <= 0:
            return (False, f"H_i result should be positive, got {self.H_i_result}")
        return (True, "H_i result is positive and finite")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all bound methods with names starting with test_
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                tb = traceback.format_exc()
                msg = f"{attr_name} CRASHED: {e}\n{tb}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)