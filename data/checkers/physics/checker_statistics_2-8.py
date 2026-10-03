import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of all inputs and outputs with None defaults
        # Parameters
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        self.k = getattr(solution_instance, 'k', None)

        # Results keys
        self.E_per_N_max_noneq = None
        self.E_per_N_max_eq = None
        self.S_per_N = None

        # Try to get from results dict (preferred)
        results_dict = getattr(solution_instance, 'results', {})
        if isinstance(results_dict, dict):
            self.E_per_N_max_noneq = results_dict.get('E_per_N_max_noneq', None)
            self.E_per_N_max_eq = results_dict.get('E_per_N_max_eq', None)
            self.S_per_N = results_dict.get('S_per_N', None)

        # If not found, try attribute access
        if self.E_per_N_max_noneq is None:
            self.E_per_N_max_noneq = getattr(solution_instance, 'E_per_N_max_noneq', None)
        if self.E_per_N_max_eq is None:
            self.E_per_N_max_eq = getattr(solution_instance, 'E_per_N_max_eq', None)
        if self.S_per_N is None:
            self.S_per_N = getattr(solution_instance, 'S_per_N', None)

        # Define symbols for sympy operations
        self.x = sp.Symbol('x', real=True, positive=True)
        self.eps_sym = sp.Symbol('epsilon', positive=True)
        self.k_sym = sp.Symbol('k', positive=True)

    # ---------- Helper for numeric entropy evaluation ----------
    def _eval_entropy_numeric(self, x_val):
        """Try to evaluate S_per_N at a given x value (0<=x<=1). Returns float or None."""
        if self.S_per_N is None:
            return None
        if isinstance(self.S_per_N, (int, float)):
            # Cannot map to arbitrary x, return None
            return None
        try:
            if isinstance(self.S_per_N, str):
                expr = sp.sympify(self.S_per_N, locals={'x': self.x, 'epsilon': self.eps_sym, 'k': self.k_sym})
            elif isinstance(self.S_per_N, sp.Expr):
                expr = self.S_per_N
            else:
                return None
            subs_dict = {self.x: x_val, self.k_sym: 1.0, self.eps_sym: 1.0}
            numeric_val = float(expr.subs(subs_dict))
            return numeric_val
        except:
            return None

    # ---------- Interface Tests ----------
    def test_interface_required_keys(self):
        """Check that all expected result keys are present."""
        missing = []
        if self.E_per_N_max_noneq is None:
            missing.append('E_per_N_max_noneq')
        if self.E_per_N_max_eq is None:
            missing.append('E_per_N_max_eq')
        if self.S_per_N is None:
            missing.append('S_per_N')
        if missing:
            return False, f"Missing required result keys: {', '.join(missing)}"
        return True, "All required result keys present"

    # ---------- Invariant Tests (use math.isclose) ----------
    def test_invariant_max_noneq(self):
        """Maximum E/N without equilibrium must equal epsilon (all atoms excited)."""
        if self.E_per_N_max_noneq is None:
            return False, "SKIP: E_per_N_max_noneq not available"
        if self.epsilon is None:
            return False, "SKIP: epsilon not available"
        if not isinstance(self.E_per_N_max_noneq, (int, float)):
            return False, "E_per_N_max_noneq is not numeric"
        if not isinstance(self.epsilon, (int, float)):
            return False, "epsilon is not numeric"
        if math.isclose(self.E_per_N_max_noneq, self.epsilon, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_per_N_max_noneq = {self.E_per_N_max_noneq} ≈ epsilon = {self.epsilon}"
        return False, f"E_per_N_max_noneq = {self.E_per_N_max_noneq} != epsilon = {self.epsilon}"

    def test_invariant_max_eq(self):
        """Maximum E/N in equilibrium (positive T) must equal epsilon/2."""
        if self.E_per_N_max_eq is None:
            return False, "SKIP: E_per_N_max_eq not available"
        if self.epsilon is None:
            return False, "SKIP: epsilon not available"
        if not isinstance(self.E_per_N_max_eq, (int, float)):
            return False, "E_per_N_max_eq is not numeric"
        if not isinstance(self.epsilon, (int, float)):
            return False, "epsilon is not numeric"
        target = self.epsilon / 2.0
        if math.isclose(self.E_per_N_max_eq, target, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_per_N_max_eq = {self.E_per_N_max_eq} ≈ epsilon/2 = {target}"
        return False, f"E_per_N_max_eq = {self.E_per_N_max_eq} != epsilon/2 = {target}"

    def test_invariant_entropy_nonnegative(self):
        """Entropy per atom must be nonnegative for all allowed x."""
        if self.S_per_N is None:
            return False, "SKIP: S_per_N not available"
        x_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
        passed = True
        details = []
        for xv in x_vals:
            numeric_val = self._eval_entropy_numeric(xv)
            if numeric_val is None:
                return False, f"SKIP: Cannot evaluate entropy for x={xv}"
            if numeric_val < -1e-9:
                passed = False
                details.append(f"x={xv}: S/N = {numeric_val} < 0")
            else:
                details.append(f"x={xv}: S/N = {numeric_val} >= 0")
        if passed:
            return True, "Entropy nonnegative for sample x: " + "; ".join(details)
        return False, "Entropy negative at some x: " + "; ".join(details)

    def test_invariant_entropy_zero_endpoints(self):
        """Entropy per atom must be zero at x=0 and x=1 (all in ground or all excited)."""
        if self.S_per_N is None:
            return False, "SKIP: S_per_N not available"
        s0 = self._eval_entropy_numeric(0.0)
        s1 = self._eval_entropy_numeric(1.0)
        if s0 is None or s1 is None:
            return False, "SKIP: Cannot evaluate entropy at endpoints"
        if math.isclose(s0, 0.0, abs_tol=1e-9) and math.isclose(s1, 0.0, abs_tol=1e-9):
            return True, f"Entropy at endpoints: S/N(0)={s0}, S/N(1)={s1} both ~0"
        return False, f"Entropy not zero at endpoints: S/N(0)={s0}, S/N(1)={s1}"

    def test_invariant_entropy_max_at_half(self):
        """Entropy per atom should be maximal at x=0.5 (equal populations)."""
        if self.S_per_N is None:
            return False, "SKIP: S_per_N not available"
        s_half = self._eval_entropy_numeric(0.5)
        if s_half is None:
            return False, "SKIP: Cannot evaluate entropy at x=0.5"
        s_low = self._eval_entropy_numeric(0.4)
        s_high = self._eval_entropy_numeric(0.6)
        if s_low is None or s_high is None:
            return False, "SKIP: Cannot evaluate entropy at comparison points"
        if s_half > s_low and s_half > s_high:
            return True, f"Entropy maximal at x=0.5: S/N(0.5)={s_half} > S/N(0.4)={s_low}, S/N(0.6)={s_high}"
        return False, f"Entropy not maximal at x=0.5: S/N(0.5)={s_half}, S/N(0.4)={s_low}, S/N(0.6)={s_high}"

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_entropy_expression_sympy(self):
        """Verify that S_per_N symbolic expression matches -k*(x*ln(x)+(1-x)*ln(1-x))."""
        if self.S_per_N is None:
            return False, "SKIP: S_per_N not available"
        try:
            if isinstance(self.S_per_N, str):
                expr_user = sp.sympify(self.S_per_N, locals={'x': self.x, 'epsilon': self.eps_sym, 'k': self.k_sym})
            elif isinstance(self.S_per_N, sp.Expr):
                expr_user = self.S_per_N
            else:
                return False, "S_per_N is not a string or sympy expression"
            expected_expr = -self.k_sym * (self.x * sp.log(self.x) + (1 - self.x) * sp.log(1 - self.x))
            diff = sp.simplify(expr_user - expected_expr)
            if diff == 0:
                return True, "S_per_N expression matches expected symbolic form"
            subs = {self.x: 0.3, self.k_sym: 1.0, self.eps_sym: 1.0}
            num_user = float(expr_user.subs(subs))
            num_exp = float(expected_expr.subs(subs))
            if math.isclose(num_user, num_exp, rel_tol=1e-3):
                return True, f"S_per_N numerically matches expected: user={num_user}, expected={num_exp}"
            return False, f"Expression mismatch: diff = {diff}; numerical: user={num_user}, expected={num_exp}"
        except Exception as e:
            return False, f"Error in symbolic comparison: {e}"

    def test_modeling_entropy_derivative(self):
        """Check that derivative of S_per_N at x=0.5 is zero (maximum) and sign around."""
        if self.S_per_N is None:
            return False, "SKIP: S_per_N not available"
        try:
            if isinstance(self.S_per_N, str):
                expr = sp.sympify(self.S_per_N, locals={'x': self.x, 'epsilon': self.eps_sym, 'k': self.k_sym})
            else:
                expr = self.S_per_N
            dS = sp.diff(expr, self.x)
            subs = {self.x: 0.5, self.k_sym: 1.0, self.eps_sym: 1.0}
            val = float(dS.subs(subs))
            if math.isclose(val, 0.0, abs_tol=1e-3):
                return True, f"Derivative at x=0.5 is zero (max): dS/dx = {val}"
            return False, f"Derivative at x=0.5 is not zero: dS/dx = {val}"
        except Exception as e:
            return False, f"Error in derivative check: {e}"

    # ---------- Monotonicity Tests (explicit input perturbation) ----------
    def test_monotonicity_entropy_near_zero(self):
        """For small x, increasing x should increase entropy (derivative positive)."""
        if self.S_per_N is None:
            return False, "SKIP: S_per_N not available"
        baseline_x = 0.1
        perturbed_x = baseline_x * 1.1  # 10% increase
        s_baseline = self._eval_entropy_numeric(baseline_x)
        s_perturbed = self._eval_entropy_numeric(perturbed_x)
        if s_baseline is None or s_perturbed is None:
            return False, "SKIP: Cannot evaluate entropy for required x values"
        # Expect: s_perturbed > s_baseline since derivative positive near zero
        if s_perturbed > s_baseline:
            return True, f"Entropy increases when x goes from {baseline_x} to {perturbed_x}: {s_baseline} -> {s_perturbed}"
        return False, f"Entropy did not increase: baseline={s_baseline}, perturbed={s_perturbed}"

    def test_monotonicity_entropy_near_one(self):
        """For x close to 1, increasing x should decrease entropy (derivative negative)."""
        if self.S_per_N is None:
            return False, "SKIP: S_per_N not available"
        baseline_x = 0.9
        perturbed_x = baseline_x * 1.05  # 5% increase (must stay <=1)
        if perturbed_x > 1.0:
            perturbed_x = 0.95
        s_baseline = self._eval_entropy_numeric(baseline_x)
        s_perturbed = self._eval_entropy_numeric(perturbed_x)
        if s_baseline is None or s_perturbed is None:
            return False, "SKIP: Cannot evaluate entropy for required x values"
        # Expect: s_perturbed < s_baseline since derivative negative near one
        if s_perturbed < s_baseline:
            return True, f"Entropy decreases when x goes from {baseline_x} to {perturbed_x}: {s_baseline} -> {s_perturbed}"
        return False, f"Entropy did not decrease: baseline={s_baseline}, perturbed={s_perturbed}"

    # ---------- Weak Answer Sanity Test ----------
    def test_answer_sanity_bounds(self):
        """Weak sanity: maximum energies should be positive and finite, and non-eq > eq."""
        if None in (self.E_per_N_max_noneq, self.E_per_N_max_eq, self.epsilon):
            return False, "SKIP: missing required values"
        if not isinstance(self.E_per_N_max_noneq, (int, float)):
            return False, "E_per_N_max_noneq not numeric"
        if not isinstance(self.E_per_N_max_eq, (int, float)):
            return False, "E_per_N_max_eq not numeric"
        if self.E_per_N_max_noneq <= 0 or self.E_per_N_max_eq <= 0:
            return False, "Maximum energies must be positive"
        if math.isnan(self.E_per_N_max_noneq) or math.isinf(self.E_per_N_max_noneq):
            return False, "E_per_N_max_noneq is NaN or Inf"
        if math.isnan(self.E_per_N_max_eq) or math.isinf(self.E_per_N_max_eq):
            return False, "E_per_N_max_eq is NaN or Inf"
        if not (self.E_per_N_max_noneq > self.E_per_N_max_eq):
            return False, "Non-equilibrium max should be strictly larger than equilibrium max"
        return True, "Basic sanity passed: positive finite energies and non-eq > eq"

    # ---------- Main runner ----------
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