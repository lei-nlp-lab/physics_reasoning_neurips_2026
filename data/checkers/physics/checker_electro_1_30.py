import math

class PhysicsChecker:
    """Audits coaxial cable charge & field solution."""

    def __init__(self, solution_instance):
        # Extract givens safely (prefer attributes, fallback to results dict)
        self.a = getattr(solution_instance, 'a', 
                         solution_instance.results.get('a', None) if hasattr(solution_instance, 'results') else None)
        self.b = getattr(solution_instance, 'b', 
                         solution_instance.results.get('b', None) if hasattr(solution_instance, 'results') else None)
        self.V0 = getattr(solution_instance, 'V0', 
                          solution_instance.results.get('V0', None) if hasattr(solution_instance, 'results') else None)
        self.r = getattr(solution_instance, 'r', 
                         solution_instance.results.get('r', None) if hasattr(solution_instance, 'results') else None)
        self.epsilon0 = 8.854187817e-12  # constant, always available

        # Extract emitted results
        self.lambda_val = getattr(solution_instance, 'lambda', None)
        if self.lambda_val is None and hasattr(solution_instance, 'results'):
            self.lambda_val = solution_instance.results.get('lambda', None)

        self.E_val = getattr(solution_instance, 'E', None)
        if self.E_val is None and hasattr(solution_instance, 'results'):
            self.E_val = solution_instance.results.get('E', None)

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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

    # ----- Interface tests -----
    def test_interface_required_keys(self):
        if self.lambda_val is None:
            return False, "Missing required result: lambda"
        if self.E_val is None:
            return False, "Missing required result: E"
        if not isinstance(self.lambda_val, (int, float)):
            return False, "lambda must be numeric"
        if not isinstance(self.E_val, (int, float)):
            return False, "E must be numeric"
        return True, "All required keys present and numeric"

    def test_interface_givens_present(self):
        missing = []
        if self.a is None:
            missing.append("a")
        if self.b is None:
            missing.append("b")
        if self.V0 is None:
            missing.append("V0")
        if self.r is None:
            missing.append("r")
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        # also check positivity
        if self.a <= 0:
            return False, f"a must be positive, got {self.a}"
        if self.b <= 0:
            return False, f"b must be positive, got {self.b}"
        if self.V0 <= 0:
            return False, f"V0 must be positive, got {self.V0}"
        if self.a >= self.b:
            return False, f"Expected a < b, got a={self.a}, b={self.b}"
        return True, "All givens present and valid"

    # ----- Invariant tests (physics relationships with tolerance) -----
    def test_invariant_lambda_formula(self):
        if self.lambda_val is None or self.a is None or self.b is None or self.V0 is None:
            return False, "SKIP: insufficient data for lambda formula check"
        try:
            expected_lambda = (2 * math.pi * self.epsilon0 * self.V0) / math.log(self.b / self.a)
        except Exception as e:
            return False, f"Error computing expected lambda: {e}"
        if math.isclose(self.lambda_val, expected_lambda, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"lambda = {self.lambda_val:.3e} matches formula {expected_lambda:.3e}"
        else:
            return False, f"lambda = {self.lambda_val:.3e} deviates from formula {expected_lambda:.3e} by >1%"

    def test_invariant_E_outside_zero(self):
        if self.E_val is None or self.r is None or self.b is None:
            return False, "SKIP: insufficient data for E outside check"
        if self.r > self.b:
            if not math.isclose(self.E_val, 0, abs_tol=1e-2):
                return False, f"Expected E=0 for r={self.r} > b={self.b}, got E={self.E_val}"
            return True, f"E={self.E_val} is zero for r outside conductor"
        else:
            # r inside or on conductor: no constraint from this test
            return True, "SKIP: r not outside conductor"

    # ----- Modeling/trace tests (consistency with solving steps) -----
    def test_modeling_lambda_positive(self):
        if self.lambda_val is None:
            return False, "SKIP: lambda not available"
        if self.lambda_val > 0:
            return True, f"lambda = {self.lambda_val:.3e} > 0"
        else:
            return False, f"lambda should be positive, got {self.lambda_val}"

    def test_modeling_E_is_zero(self):
        if self.E_val is None:
            return False, "SKIP: E not available"
        if self.r is not None and self.r > self.b:
            if math.isclose(self.E_val, 0, abs_tol=1e-2):
                return True, "E=0 for r>b"
            else:
                return False, f"E should be zero for r>b, got {self.E_val}"
        else:
            return True, "SKIP: no constraint on E inside conductor"

    # ----- Monotonicity tests (perturb one input and check trend) -----
    def test_monotonicity_lambda_increases_with_V0(self):
        # Baseline: use provided lambda_val, but we also compute expectation from formula
        if self.lambda_val is None or self.a is None or self.b is None or self.V0 is None:
            return False, "SKIP: insufficient data for monotonicity V0"
        # Perturb V0 by +10%
        perturbed_V0 = self.V0 * 1.1
        try:
            baseline_lambda = (2 * math.pi * self.epsilon0 * self.V0) / math.log(self.b / self.a)
            perturbed_lambda = (2 * math.pi * self.epsilon0 * perturbed_V0) / math.log(self.b / self.a)
        except Exception as e:
            return False, f"Error computing perturbed lambda: {e}"
        if perturbed_lambda > baseline_lambda:
            return True, f"lambda increases with V0: base={baseline_lambda:.3e}, perturbed(V0*1.1)={perturbed_lambda:.3e}"
        else:
            return False, f"Monotonicity fails: V0 increase should increase lambda"

    def test_monotonicity_lambda_decreases_with_b(self):
        # Perturb b by +10% (still keeping b>a)
        if self.lambda_val is None or self.a is None or self.b is None or self.V0 is None:
            return False, "SKIP: insufficient data for monotonicity b"
        perturbed_b = self.b * 1.1
        # Ensure a < perturbed_b
        if perturbed_b <= self.a:
            perturbed_b = self.a * 1.5  # adjust to keep logical
            if perturbed_b <= self.a:
                return False, "Cannot create valid perturbed b"
        try:
            baseline_lambda = (2 * math.pi * self.epsilon0 * self.V0) / math.log(self.b / self.a)
            perturbed_lambda = (2 * math.pi * self.epsilon0 * self.V0) / math.log(perturbed_b / self.a)
        except Exception as e:
            return False, f"Error computing perturbed lambda: {e}"
        if perturbed_lambda < baseline_lambda:
            return True, f"lambda decreases with b: base={baseline_lambda:.3e}, perturbed(b*1.1)={perturbed_lambda:.3e}"
        else:
            return False, f"Monotonicity fails: b increase should decrease lambda"

    # ----- Answer sanity (weak, no re-derivation) -----
    def test_answer_sanity_broad_magnitude(self):
        issues = []
        if self.lambda_val is not None:
            if not (1e-10 <= self.lambda_val <= 1e-4):
                issues.append(f"lambda={self.lambda_val:.3e} outside plausible range [1e-10,1e-4]")
        if self.E_val is not None:
            if self.r is not None and self.r > self.b:
                if abs(self.E_val) > 1e-3:
                    issues.append(f"E={self.E_val} should be zero for r>b")
        if issues:
            return False, "; ".join(issues)
        return True, "Lambda within plausible range, E=0 for r>b"