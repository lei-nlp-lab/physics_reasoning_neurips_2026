import math
import traceback
import sympy

class PhysicsChecker:
    """
    Physics checker for the problem: Show that p/eta ≈ collision frequency,
    and compute collision frequency at STP.
    """

    def __init__(self, solution_instance):
        # Safely extract all needed data with None defaults
        self.p_STP = getattr(solution_instance, 'p_STP', None)
        self.eta = getattr(solution_instance, 'eta', None)
        self.v_bar = getattr(solution_instance, 'v_bar', None)
        self.lambda_ = getattr(solution_instance, 'lambda_', None)
        self.n = getattr(solution_instance, 'n', None)
        self.m = getattr(solution_instance, 'm', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # From results dictionary
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.ratio = self.results.get('ratio', None)
        self.nu = self.results.get('nu', None)

    # ------------------------------------------------------------------
    # Interface tests: verify expected keys exist and have valid types
    # ------------------------------------------------------------------

    def test_interface_required_keys(self):
        """Check that required result keys are present."""
        missing = []
        if self.ratio is None:
            missing.append('ratio')
        if self.nu is None:
            missing.append('nu')
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_ratio_expression(self):
        """Check that ratio is a valid expression (non-empty string or sympy)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if isinstance(self.ratio, str):
            if not self.ratio.strip():
                return (False, "ratio string is empty")
            low = self.ratio.lower()
            if 'v' not in low and 'lambda' not in low and 'λ' not in low:
                return (False, "ratio expression does not contain expected symbols")
        elif not hasattr(self.ratio, 'atoms'):
            return (False, "ratio has unexpected type")
        return (True, "ratio expression appears valid")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical checks with math.isclose
    # ------------------------------------------------------------------

    def test_invariant_ratio_equals_nu(self):
        """
        Verify that the ratio (v̄/λ) is approximately equal to ν,
        using the provided ratio and nu if both are numeric.
        """
        if self.ratio is None or self.nu is None:
            return (False, "SKIP: ratio or nu missing")
        ratio_val = None
        if isinstance(self.ratio, (int, float)):
            ratio_val = self.ratio
        elif hasattr(self.ratio, 'subs'):
            # Try to evaluate with dummy values (just to check form)
            try:
                # If expression is v/lambda, substitution with 1,1 gives 1
                ratio_val = float(self.ratio.subs({'v_bar': 1, 'lambda': 1}))
            except:
                pass
        if ratio_val is not None and self.nu is not None:
            if math.isclose(ratio_val, self.nu, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"ratio ≈ nu within tolerance: {ratio_val} vs {self.nu}")
            else:
                return (False, f"ratio ({ratio_val}) not close to nu ({self.nu})")
        return (False, "SKIP: cannot compare ratio and nu numerically")

    def test_invariant_nu_formula(self):
        """
        Check that nu satisfies the derived relation: nu = p/eta.
        Use the provided p_STP and eta if given, compute expected nu,
        and compare to the reported nu.
        """
        if self.p_STP is None or self.eta is None or self.nu is None:
            return (False, "SKIP: missing p_STP, eta or nu")
        if self.eta == 0:
            return (False, "eta is zero, cannot compute")
        expected_nu = self.p_STP / self.eta
        if math.isclose(self.nu, expected_nu, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"nu ≈ p/eta: {self.nu} vs {expected_nu}")
        return (False, f"nu ({self.nu}) mismatch with p/eta ({expected_nu})")

    def test_invariant_mean_free_path_relation(self):
        """
        Check that if lambda and v_bar are provided, then v_bar/lambda ≈ nu.
        """
        if self.v_bar is None or self.lambda_ is None or self.nu is None:
            return (False, "SKIP: missing v_bar, lambda or nu")
        if self.lambda_ == 0:
            return (False, "lambda is zero")
        computed_nu = self.v_bar / self.lambda_
        if math.isclose(computed_nu, self.nu, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"nu ≈ v_bar/lambda: {self.nu} vs {computed_nu}")
        return (False, f"nu mismatch with v_bar/lambda: {self.nu} vs {computed_nu}")

    # ------------------------------------------------------------------
    # Modeling / trace quality tests
    # ------------------------------------------------------------------

    def test_modeling_ratio_derivation_consistency(self):
        """
        Check that the ratio expression is consistent with the kinetic theory
        formulas. The ratio should be of the form v̄/λ.
        """
        if self.ratio is None:
            return (False, "SKIP: ratio not provided")
        ratio_str = str(self.ratio).replace(' ', '').lower()
        # Check that it involves v and lambda (or λ)
        has_v = 'v' in ratio_str
        has_lambda = 'λ' in ratio_str or 'lambda' in ratio_str
        # Check that it's a quotient
        is_quotient = '/' in ratio_str
        if has_v and has_lambda and is_quotient:
            return (True, "Ratio expressed as fraction of v and lambda")
        return (False, "Ratio does not appear as v/lambda form")

    def test_modeling_explanation_contains_key_relations(self):
        """
        Verify that the explanation_steps mention the crucial kinetic theory
        relations: pressure formula, viscosity formula, mean free path.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        # Combine if list
        text = ''
        if isinstance(self.explanation_steps, list):
            text = ' '.join(self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        else:
            return (False, "SKIP: explanation_steps has unexpected type")

        # Normalize for checking
        text_lower = text.lower()
        checks = {
            'pressure formula': 'p = (1/3) n m v^2' in text_lower or 'pressure' in text_lower and '1/3' in text,
            'viscosity formula': 'eta = (1/3) n m v lambda' in text_lower or 'viscosity' in text_lower and '1/3' in text,
            'mean free path': 'lambda = v / nu' in text_lower or 'mean free path' in text_lower,
        }
        passed_checks = [key for key, val in checks.items() if val]
        if len(passed_checks) >= 2:
            return (True, f"Explanation contains key relations: {', '.join(passed_checks)}")
        return (False, f"Only {len(passed_checks)}/3 key relations found in explanation: {passed_checks}")

    def test_trace_substitution_path(self):
        """
        Trace that the derivation p/eta → ν follows from substituting
        the kinetic theory expressions. This test checks intermediate
        reasoning consistency without hardcoding numbers.
        """
        # We assume the solution has explanation steps. If not, skip.
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps for trace")
        text = ''
        if isinstance(self.explanation_steps, list):
            text = ' '.join(self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        else:
            return (False, "SKIP: explanation_steps has unexpected type")
        # Check that the derivation mentions substitution of pressure and viscosity
        text_lower = text.lower()
        keywords = ['substitute', 'ratio', 'p/eta', 'η', 'ν', 'collision frequency']
        found = sum(1 for kw in keywords if kw in text_lower)
        if found >= 3:
            return (True, f"Explanation mentions substitution path ({found} keywords)")
        return (False, f"Substitution path not well described (only {found} keywords found)")

    # ------------------------------------------------------------------
    # Monotonicity tests: explicitly perturb an input and compare outputs
    # ------------------------------------------------------------------

    def test_monotonicity_nu_pressure(self):
        """
        Increase pressure by 1% and verify that expected nu increases proportionally.
        Use the relation nu = p/eta to compute perturbed expectation.
        """
        if self.p_STP is None or self.eta is None or self.nu is None:
            return (False, "SKIP: missing p_STP, eta or nu")
        if self.eta == 0:
            return (False, "eta is zero")
        # Baseline expected nu from given formula
        baseline_nu = self.p_STP / self.eta
        # Perturb: increase p by 1%
        perturbed_p = self.p_STP * 1.01
        perturbed_nu = perturbed_p / self.eta
        # Check monotonic direction
        if perturbed_nu > baseline_nu:
            return (True, f"Perturbed nu ({perturbed_nu:.3e}) > baseline nu ({baseline_nu:.3e}) as expected")
        return (False, f"Monotonicity violation: perturbed nu ({perturbed_nu:.3e}) <= baseline nu ({baseline_nu:.3e})")

    def test_monotonicity_nu_viscosity(self):
        """
        Increase viscosity by 1% and verify that expected nu decreases.
        Use the relation nu = p/eta.
        """
        if self.p_STP is None or self.eta is None or self.nu is None:
            return (False, "SKIP: missing p_STP, eta or nu")
        if self.eta == 0:
            return (False, "eta is zero")
        baseline_nu = self.p_STP / self.eta
        # Perturb: increase eta by 1%
        perturbed_eta = self.eta * 1.01
        perturbed_nu = self.p_STP / perturbed_eta
        if perturbed_nu < baseline_nu:
            return (True, f"Perturbed nu ({perturbed_nu:.3e}) < baseline nu ({baseline_nu:.3e}) as expected")
        return (False, f"Monotonicity violation: perturbed nu ({perturbed_nu:.3e}) >= baseline nu ({baseline_nu:.3e})")

    # ------------------------------------------------------------------
    # Weak answer sanity: no exact matching, no re-derivation
    # ------------------------------------------------------------------

    def test_answer_sanity_nu_magnitude(self):
        """
        Check that nu is finite, positive, and of plausible order (~10^9-10^10).
        """
        if self.nu is None:
            return (False, "SKIP: nu not provided")
        if not isinstance(self.nu, (int, float)):
            return (False, "nu is not numeric")
        if not math.isfinite(self.nu):
            return (False, "nu is not finite")
        if self.nu <= 0:
            return (False, "nu is not positive")
        # Broad magnitude check
        if self.nu < 1e8 or self.nu > 1e11:
            return (False, f"nu out of plausible range (1e8-1e11): {self.nu}")
        return (True, f"nu is plausible: {self.nu}")

    # ------------------------------------------------------------------
    # Run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)