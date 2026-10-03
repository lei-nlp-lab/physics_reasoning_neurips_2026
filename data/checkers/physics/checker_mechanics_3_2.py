import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution to the relativistic clock synchronization problem.
    Checks: interface, invariants (time dilation, light travel), modeling consistency,
    monotonicity, and a weak final answer sanity.
    """

    def __init__(self, solution_instance):
        self.solution_instance = solution_instance
        # Extract results
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.t = self.results.get('t', None)

        # Extract givens
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        self.v = self.givens.get('v', None)
        self.c = self.givens.get('c', None)
        self.t_obs = self.givens.get('t_obs', None)

        # Sympy symbols for symbolic checks
        self.v_sym, self.c_sym, self.t_obs_sym = sp.symbols('v c t_obs', positive=True, real=True)

    def _evaluate_t(self, v_val=None, c_val=None, t_obs_val=None):
        """Return float value of self.t or None if impossible."""
        if v_val is None:
            v_val = self.v
        if c_val is None:
            c_val = self.c
        if t_obs_val is None:
            t_obs_val = self.t_obs

        if v_val is None or c_val is None or t_obs_val is None:
            return None

        try:
            v_val = float(v_val)
            c_val = float(c_val)
            t_obs_val = float(t_obs_val)
        except Exception:
            return None

        if isinstance(self.t, sp.Basic):
            try:
                val = float(self.t.subs({
                    self.v_sym: v_val,
                    self.c_sym: c_val,
                    self.t_obs_sym: t_obs_val
                }))
            except Exception:
                val = float('nan')
        elif self.t is not None:
            val = float(self.t)  # assumes numeric
        else:
            val = None
        return val

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_result_key_exists(self):
        if self.t is None:
            return (False, "FAIL: result['t'] is missing or None")
        return (True, "PASS: result['t'] is present")

    def test_interface_inputs_valid(self):
        if self.v is None or self.c is None or self.t_obs is None:
            return (True, "SKIP: not all inputs provided (v/c/t_obs)")
        try:
            v = float(self.v)
            c = float(self.c)
            t_obs = float(self.t_obs)
        except Exception:
            return (False, "FAIL: inputs not numeric")
        if v < 0 or c <= 0 or t_obs <= 0:
            return (False, "FAIL: v>=0, c>0, t_obs>0 expected")
        if v >= c:
            return (False, "FAIL: v must be < c for causality")
        return (True, "PASS: inputs valid")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_zero_velocity_returns_t_obs(self):
        """If v = 0, t should equal t_obs (no dilation, no delay)."""
        if self.v is None or self.t_obs is None:
            return (True, "SKIP: v or t_obs missing")
        if float(self.v) != 0:
            return (True, "SKIP: v is non‑zero")
        c_val = self.c if self.c is not None else 1.0
        actual = self._evaluate_t(v_val=0.0, c_val=c_val, t_obs_val=float(self.t_obs))
        if actual is None:
            return (False, "FAIL: cannot evaluate t with v=0")
        if math.isclose(actual, float(self.t_obs), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: v=0 gives t={actual} ≈ t_obs={self.t_obs}")
        else:
            return (False, f"FAIL: v=0 gives t={actual}, expected {self.t_obs}")

    def test_invariant_positive_velocity_t_gt_t_obs(self):
        """For v>0, t must be strictly greater than t_obs."""
        v_val = self.v
        t_obs_val = self.t_obs
        if v_val is None or t_obs_val is None:
            return (True, "SKIP: v or t_obs missing")
        try:
            v_val = float(v_val)
            t_obs_val = float(t_obs_val)
        except:
            return (False, "FAIL: cannot convert to float")
        if v_val <= 0:
            return (True, "SKIP: v not positive")
        c_val = self.c if self.c is not None else 1.0
        actual = self._evaluate_t(v_val=v_val, c_val=c_val, t_obs_val=t_obs_val)
        if actual is None:
            return (False, "FAIL: cannot evaluate t")
        if actual > t_obs_val + 1e-12:
            return (True, f"PASS: t={actual} > t_obs={t_obs_val}")
        else:
            return (False, f"FAIL: t={actual} ≤ t_obs={t_obs_val}")

    def test_invariant_near_light_speed_diverges(self):
        """As v → c, t should become much larger than t_obs."""
        if self.t is None:
            return (False, "FAIL: t missing")
        c_val = self.c if self.c is not None else 1.0
        t_obs_val = self.t_obs if self.t_obs is not None else 1.0
        v_near = 0.99 * c_val
        actual = self._evaluate_t(v_val=v_near, c_val=c_val, t_obs_val=t_obs_val)
        if actual is None:
            return (True, "SKIP: cannot evaluate t near c")
        # For v=0.99c, t/t_obs ≈ sqrt((1+0.99)/(1-0.99)) = sqrt(199) ≈ 14.1
        if actual > 10 * t_obs_val:
            return (True, f"PASS: t ≈ {actual} >> t_obs for v near c")
        else:
            return (False, f"FAIL: t ≈ {actual} is not large enough for v=0.99c")

    # ------------------------------------------------------------------
    # Modeling/trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_correct_formula(self):
        """The result should be equivalent to sqrt((c+v)/(c-v)) * t_obs."""
        if self.t is None:
            return (False, "FAIL: t missing")
        # symbolic check
        if isinstance(self.t, sp.Basic):
            expected = sp.sqrt((self.c_sym + self.v_sym) / (self.c_sym - self.v_sym)) * self.t_obs_sym
            diff = sp.simplify(self.t - expected)
            if diff == 0:
                return (True, "PASS: symbolic equality with expected formula")
            # Maybe t_obs is treated as 1 (problem default)
            alt_expected = sp.sqrt((self.c_sym + self.v_sym) / (self.c_sym - self.v_sym))
            if sp.simplify(self.t - alt_expected) == 0:
                return (True, "PASS: symbolic equality with formula under t_obs=1")
        # numeric fallback
        v_val = self.v if self.v is not None else 0.5
        c_val = self.c if self.c is not None else 1.0
        t_obs_val = self.t_obs if self.t_obs is not None else 1.0
        actual = self._evaluate_t(v_val=v_val, c_val=c_val, t_obs_val=t_obs_val)
        if actual is None:
            return (False, "FAIL: cannot evaluate t")
        expected_num = math.sqrt((c_val + v_val) / (c_val - v_val)) * t_obs_val
        if math.isclose(actual, expected_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: numeric match: t={actual} ≈ {expected_num}")
        else:
            return (False, f"FAIL: numeric mismatch: t={actual}, expected {expected_num}")

    def test_modeling_gamma_factor(self):
        """Check that t / t_obs = gamma * (1 + v/c)."""
        if self.t is None:
            return (False, "FAIL: t missing")
        v_val = self.v if self.v is not None else 0.5
        c_val = self.c if self.c is not None else 1.0
        t_obs_val = self.t_obs if self.t_obs is not None else 1.0
        actual = self._evaluate_t(v_val=v_val, c_val=c_val, t_obs_val=t_obs_val)
        if actual is None:
            return (False, "FAIL: cannot evaluate t")
        gamma = 1.0 / math.sqrt(1 - v_val**2 / c_val**2)
        expected = gamma * t_obs_val * (1 + v_val / c_val)
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: t consistent with gamma: {actual} ≈ {expected}")
        else:
            return (False, f"FAIL: t inconsistent: {actual} vs {expected}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_velocity_increase(self):
        """Increasing v (while v < c) should increase t."""
        if self.t is None:
            return (False, "FAIL: t missing")
        c_val = self.c if self.c is not None else 1.0
        t_obs_val = self.t_obs if self.t_obs is not None else 1.0
        v_base = self.v if self.v is not None else 0.5
        try:
            v_base = float(v_base)
            c_val = float(c_val)
            t_obs_val = float(t_obs_val)
        except:
            return (False, "FAIL: non‑numeric inputs")
        if v_base >= c_val - 1e-10:
            return (True, "SKIP: v too close to c")
        delta = 0.02 * c_val
        v_high = v_base + delta
        if v_high >= c_val:
            return (True, "SKIP: perturbed v would exceed c")
        t_base = self._evaluate_t(v_val=v_base, c_val=c_val, t_obs_val=t_obs_val)
        t_high = self._evaluate_t(v_val=v_high, c_val=c_val, t_obs_val=t_obs_val)
        if t_base is None or t_high is None:
            return (False, "FAIL: cannot evaluate t for perturbation")
        if t_high > t_base:
            return (True, f"PASS: t({v_high:.4f}) = {t_high} > t({v_base:.4f}) = {t_base}")
        else:
            return (False, f"FAIL: t did not increase: base={t_base}, high={t_high}")

    def test_monotonicity_light_speed_increase(self):
        """Increasing c (while v fixed) should decrease t (less relativistic)."""
        if self.t is None:
            return (False, "FAIL: t missing")
        # Use symbolic derivative if possible, else numeric perturbation
        if isinstance(self.t, sp.Basic):
            try:
                dt_dc = sp.diff(self.t, self.c_sym)
                # Evaluate at a representative point
                v_val = self.v if self.v is not None else 0.5
                c_val = self.c if self.c is not None else 1.0
                t_obs_val = self.t_obs if self.t_obs is not None else 1.0
                deriv = float(dt_dc.subs({
                    self.v_sym: v_val,
                    self.c_sym: c_val,
                    self.t_obs_sym: t_obs_val
                }))
            except Exception:
                deriv = None
            if deriv is not None and deriv < 0:
                return (True, f"PASS: dt/dc = {deriv} < 0")
        # Numeric perturbation
        v_val = self.v if self.v is not None else 0.5
        c_base = self.c if self.c is not None else 1.0
        t_obs_val = self.t_obs if self.t_obs is not None else 1.0
        try:
            v_val = float(v_val)
            c_base = float(c_base)
            t_obs_val = float(t_obs_val)
        except:
            return (False, "FAIL: non‑numeric inputs")
        delta = 0.05 * c_base
        c_high = c_base + delta
        t_base = self._evaluate_t(v_val=v_val, c_val=c_base, t_obs_val=t_obs_val)
        t_high = self._evaluate_t(v_val=v_val, c_val=c_high, t_obs_val=t_obs_val)
        if t_base is None or t_high is None:
            return (False, "FAIL: cannot evaluate t for perturbation")
        if t_high < t_base:
            return (True, f"PASS: t({c_high:.4f}) = {t_high} < t({c_base:.4f}) = {t_base}")
        else:
            return (False, f"FAIL: t did not decrease: base={t_base}, high={t_high}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausible(self):
        """
        Weak check: t must be positive, not extremely tiny/huge for reasonable inputs.
        Do NOT re‑derive expected value.
        """
        if self.t is None:
            return (False, "FAIL: t missing")
        # Use typical default values if missing (but we do NOT hardcode expected answer)
        v_val = self.v if self.v is not None else 0.5
        c_val = self.c if self.c is not None else 1.0
        t_obs_val = self.t_obs if self.t_obs is not None else 1.0
        actual = self._evaluate_t(v_val=v_val, c_val=c_val, t_obs_val=t_obs_val)
        if actual is None:
            return (False, "FAIL: cannot evaluate t")
        # Reasonable bounds: for v in [0, 0.9c], t in [t_obs, ~4.36*t_obs]
        if actual <= 0:
            return (False, f"FAIL: t={actual} not positive")
        if actual > 100 * t_obs_val:
            return (False, f"FAIL: t={actual} implausibly large")
        return (True, f"PASS: t={actual} positive and plausible")

    # ------------------------------------------------------------------
    # run_all_tests
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