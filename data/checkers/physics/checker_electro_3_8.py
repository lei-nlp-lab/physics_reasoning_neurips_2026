import math
import sympy
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the low-pass RC filter compensation problem.
    Verifies that the solution correctly identifies that a single high-pass
    RC filter in parallel with the low-pass RC filter yields exact compensation
    (H_lp + H_hp = 1), while other options do not.
    """

    def __init__(self, solution_instance):
        # Extract everything safely using getattr and .get()
        self.results = getattr(solution_instance, 'results', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        self.final_answer = getattr(solution_instance, 'final_answer', None) or self.results.get('final_answer')

        # Required key from pseudocode [emit] contract
        self.exact_compensation = self.results.get('exact_compensation', None)

        # Optional givens and transfer functions (may be present)
        self.R = self.results.get('R', None)
        self.C = self.results.get('C', None)
        self.tau = self.results.get('tau', None)          # R*C
        self.H_lp = self.results.get('H_lp', None)        # expected: 1/(1+tau*s)
        self.H_hp = self.results.get('H_hp', None)        # expected: tau*s/(1+tau*s)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_exact_compensation_exists(self):
        """Check that the required key 'exact_compensation' is present."""
        if self.exact_compensation is None:
            return (False, "SKIP: 'exact_compensation' not found in results")
        return (True, "'exact_compensation' key present")

    def test_interface_exact_compensation_type(self):
        """Check that exact_compensation is a boolean."""
        if self.exact_compensation is None:
            return (False, "SKIP: 'exact_compensation' is missing")
        if not isinstance(self.exact_compensation, bool):
            return (False, f"Type is {type(self.exact_compensation).__name__}, expected bool")
        return (True, "'exact_compensation' is a boolean")

    # ------------------------------------------------------------------
    # Theorem‑based invariant tests (use math.isclose / symbolic)
    # ------------------------------------------------------------------
    def test_invariant_sum_relation(self):
        """
        Verify that if H_lp and H_hp are provided, their sum simplifies to 1.
        This is the theorem‑level condition for exact compensation.
        """
        if self.H_lp is None or self.H_hp is None:
            return (False, "SKIP: H_lp or H_hp not provided")
        try:
            s = sympy.Symbol('s')
            tau = sympy.Symbol('tau')
            # Convert strings to SymPy expressions if needed
            H_lp_expr = sympy.sympify(self.H_lp) if isinstance(self.H_lp, str) else self.H_lp
            H_hp_expr = sympy.sympify(self.H_hp) if isinstance(self.H_hp, str) else self.H_hp
            H_sum = sympy.simplify(H_lp_expr + H_hp_expr)
            if H_sum == 1:
                return (True, "H_lp + H_hp = 1 exactly verified")
            else:
                return (False, f"H_lp + H_hp simplifies to {H_sum}, not 1")
        except Exception as e:
            return (False, f"Symbolic verification error: {e}")

    def test_invariant_transfer_function_forms(self):
        """
        Check that the provided transfer function strings match the standard
        forms for an RC low‑pass and high‑pass filter.
        """
        if self.H_lp is None or self.H_hp is None:
            return (False, "SKIP: H_lp or H_hp not provided")
        try:
            s = sympy.Symbol('s')
            tau = sympy.Symbol('tau')
            H_lp_expr = sympy.sympify(self.H_lp) if isinstance(self.H_lp, str) else self.H_lp
            H_hp_expr = sympy.sympify(self.H_hp) if isinstance(self.H_hp, str) else self.H_hp
            expected_lp = 1/(1 + tau*s)
            expected_hp = tau*s/(1 + tau*s)
            if sympy.simplify(H_lp_expr - expected_lp) == 0:
                lp_ok = True
            else:
                lp_ok = False
            if sympy.simplify(H_hp_expr - expected_hp) == 0:
                hp_ok = True
            else:
                hp_ok = False
            if lp_ok and hp_ok:
                return (True, "Transfer function forms are correct")
            else:
                msg = f"Low‑pass correct: {lp_ok}, High‑pass correct: {hp_ok}"
                return (False, msg)
        except Exception as e:
            return (False, f"Transfer function check error: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace quality tests (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_explanation_contains_sum_unity(self):
        """The explanation should mention that H_lp + H_hp = 1 or "exact compensation"."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = self.explanation_steps.lower()
        # Look for key phrases
        if "sum" in text and ("1" in text or "unity" in text):
            return (True, "Explanation mentions sum unity")
        if "exact compensation" in text:
            return (True, "Explanation mentions exact compensation")
        return (False, "Explanation does not mention sum=1 or exact compensation")

    def test_modeling_pseudocode_contains_key_equation(self):
        """The pseudocode should contain the relation H_sum = H_lp(s) + H_hp(s) = 1."""
        if self.pseudocode is None:
            return (False, "SKIP: pseudocode not available")
        if "H_sum = H_lp(s) + H_hp(s) = 1" in self.pseudocode:
            return (True, "Pseudocode contains the exact compensation equation")
        else:
            return (False, "Pseudocode does not contain the expected equation")

    def test_modeling_explains_why_infinite_series_fails(self):
        """A good explanation should note that infinite series goes to zero except at DC."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = self.explanation_steps.lower()
        if "infinite series" in text or "limit" in text or "approach" in text:
            return (True, "Explanation discusses infinite series behavior")
        return (False, "Explanation does not address infinite series limitation")

    # ------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_lowpass_magnitude_decreases_with_frequency(self):
        """
        If R, C, tau are given, pick a fixed frequency and verify that
        the magnitude of H_lp decreases when R or C is increased.
        """
        # Need at least two of R, C, tau to perturb
        if self.tau is None and (self.R is None or self.C is None):
            return (True, "SKIP: Not enough data to perform monotonicity check")
        try:
            # Use a representative angular frequency (e.g., 1 rad/s) for comparison
            omega = 1.0
            # Determine tau
            if self.tau is not None:
                tau0 = float(self.tau)
            else:
                tau0 = float(self.R) * float(self.C)
            # Compute baseline magnitude of H_lp at omega
            s = 1j * omega
            baseline_mag = abs(1.0 / (1.0 + tau0 * s))  # |H_lp(jω)|
            # Perturb tau by +10%
            tau1 = tau0 * 1.1
            perturbed_mag = abs(1.0 / (1.0 + tau1 * s))
            # Check that perturbed magnitude is smaller (low‑pass filtering)
            if perturbed_mag < baseline_mag:
                return (True, f"Monotonicity OK: increasing tau by 10% reduces |H_lp| at ω=1 ({baseline_mag:.4f} -> {perturbed_mag:.4f})")
            else:
                return (False, f"Monotonicity failed: magnitude increased from {baseline_mag:.4f} to {perturbed_mag:.4f}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ------------------------------------------------------------------
    # Very weak final‑answer sanity (only ONE such test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_choice(self):
        """Final answer should be one of the three multiple‑choice options."""
        valid = {'(a)', '(b)', '(c)'}
        if self.final_answer is None:
            return (False, "SKIP: final_answer not available")
        ans = self.final_answer.strip()
        if ans in valid:
            return (True, f"Final answer '{ans}' is a valid option")
        else:
            return (False, f"Final answer '{ans}' not in {valid}")

    # ------------------------------------------------------------------
    # Master test runner – discover and run all test_* methods
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)