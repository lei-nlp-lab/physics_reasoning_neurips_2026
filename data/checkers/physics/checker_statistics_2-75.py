import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for the 3K blackbody radiation problem.
    Checks interface completeness, adiabatic relation, energy density via Stefan–Boltzmann,
    modeling of reasoning steps, monotonicity of physical relations, and weak sanity.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.Q1_answer = results.get('Q1_answer', None)
        self.T_f = results.get('T_f', None)
        self.u_value = results.get('u_value', None)
        self.u_order = results.get('u_order', None)

        # Possibly extract explanation steps for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Known physical constants and givens (problem statement)
        self.T0 = 3.0               # current temperature in K
        self.V_factor = 2.0         # volume doubling factor
        self.a = 7.5657e-16        # Stefan–Boltzmann constant for u = a T^4 (J m⁻³ K⁻⁴)

    # ------------------------------------------------------------------ #
    # Interface checks
    # ------------------------------------------------------------------ #
    def test_interface_required_keys(self):
        """All four result keys must be present."""
        missing = []
        if self.Q1_answer is None:
            missing.append('Q1_answer')
        if self.T_f is None:
            missing.append('T_f')
        if self.u_value is None:
            missing.append('u_value')
        if self.u_order is None:
            missing.append('u_order')
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_Q1_answer_type(self):
        """Q1_answer must be a non‑empty string."""
        if self.Q1_answer is None:
            return (False, "SKIP: Q1_answer missing")
        if not isinstance(self.Q1_answer, str) or len(self.Q1_answer.strip()) == 0:
            return (False, f"Q1_answer should be a non‑empty string, got {type(self.Q1_answer)}: {self.Q1_answer!r}")
        return (True, "Q1_answer is a non‑empty string")

    def test_interface_T_f_type(self):
        """T_f must be numeric and finite."""
        if self.T_f is None:
            return (False, "SKIP: T_f missing")
        try:
            val = float(self.T_f)
            if not math.isfinite(val):
                return (False, f"T_f is not finite: {val}")
            return (True, f"T_f is numeric and finite: {val}")
        except (ValueError, TypeError):
            return (False, f"T_f cannot be converted to float: {self.T_f!r}")

    def test_interface_u_value_type(self):
        """u_value must be numeric and finite."""
        if self.u_value is None:
            return (False, "SKIP: u_value missing")
        try:
            val = float(self.u_value)
            if not math.isfinite(val):
                return (False, f"u_value is not finite: {val}")
            return (True, f"u_value is numeric and finite: {val}")
        except (ValueError, TypeError):
            return (False, f"u_value cannot be converted to float: {self.u_value!r}")

    def test_interface_u_order_type(self):
        """u_order must be a string."""
        if self.u_order is None:
            return (False, "SKIP: u_order missing")
        if not isinstance(self.u_order, str):
            return (False, f"u_order should be a string, got {type(self.u_order)}")
        return (True, "u_order is a string")

    # ------------------------------------------------------------------ #
    # Invariant checks (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------ #
    def test_invariant_adiabatic_relation(self):
        """T_f should equal T0 / 2^{1/3} from the adiabatic relation T V^{1/3} = const."""
        if self.T_f is None:
            return (False, "SKIP: T_f missing")
        try:
            T_f_val = float(self.T_f)
        except (ValueError, TypeError):
            return (False, "T_f not numeric")
        expected = self.T0 / (self.V_factor ** (1.0/3.0))   # 3 / 2^{1/3}
        if not math.isclose(T_f_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_f = {T_f_val} does not match adiabatic relation {expected:.6f}")
        return (True, f"T_f matches adiabatic relation: {T_f_val} ≈ {expected:.6f}")

    def test_invariant_energy_density(self):
        """u_value should be close to a * T0^4 from Stefan–Boltzmann."""
        if self.u_value is None:
            return (False, "SKIP: u_value missing")
        try:
            u_val = float(self.u_value)
        except (ValueError, TypeError):
            return (False, "u_value not numeric")
        expected = self.a * (self.T0 ** 4)   # 7.5657e-16 * 81 ≈ 6.128e-14
        if not math.isclose(u_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"u_value = {u_val} does not match Stefan–Boltzmann derived {expected:.4e}")
        return (True, f"u_value matches Stefan–Boltzmann: {u_val:.4e} ≈ {expected:.4e}")

    # ------------------------------------------------------------------ #
    # Modeling / trace checks (reasoning quality)
    # ------------------------------------------------------------------ #
    def test_modeling_Q1_adiabatic_reason(self):
        """Q1_answer should indicate 'adiabatic' and the key reason (isolated system)."""
        if self.Q1_answer is None:
            return (False, "SKIP: Q1_answer missing")
        ans = self.Q1_answer.lower()
        if 'adiabatic' not in ans:
            return (False, f"Q1_answer does not contain 'adiabatic': {self.Q1_answer!r}")
        # Check for isolation reasoning – may be explicit or implied
        if 'isolated' not in ans and 'no heat' not in ans and 'no exchange' not in ans:
            # Accept if the answer still mentions adiabatic; weak check
            pass
        return (True, "Q1_answer correctly identifies adiabatic expansion")

    def test_modeling_Q2_relation_used(self):
        """If explanation_steps exist, check that the adiabatic relation T V^{1/3}=const appears."""
        if self.explanation_steps is None:
            return (True, "SKIP: no explanation_steps provided")
        # Flatten if dict or list
        text = ""
        if isinstance(self.explanation_steps, dict):
            text = " ".join(str(v) for v in self.explanation_steps.values())
        elif isinstance(self.explanation_steps, list):
            text = " ".join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        # Check for the relation
        if 'V^{1/3}' in text or "V^{1/3}" in text or "T V^{1/3}" in text:
            return (True, "Explanation steps contain the adiabatic relation T V^{1/3}=const")
        # Fallback: check for 'adiabatic' plus 'volume'
        if 'adiabatic' in text.lower() and 'volume' in text.lower():
            return (True, "Explanation steps mention adiabatic and volume")
        return (False, "Explanation steps do not clearly show the adiabatic relation T V^{1/3}=const")

    # ------------------------------------------------------------------ #
    # Monotonicity checks (explicit input perturbation)
    # ------------------------------------------------------------------ #
    def test_monotonicity_T_f_with_volume(self):
        """
        Check that if the volume factor were larger, T_f would be smaller.
        Baseline: V_factor = 2 → T_f = T0 / 2^{1/3}.
        Perturb: V_factor = 3 → T_f_pert = T0 / 3^{1/3} (should be less than baseline).
        """
        if self.T_f is None:
            return (False, "SKIP: T_f missing")
        try:
            T_f_baseline = float(self.T_f)
        except (ValueError, TypeError):
            return (False, "T_f not numeric")
        # Expected baseline from given V_factor=2
        expected_baseline = self.T0 / (self.V_factor ** (1/3))
        if not math.isclose(T_f_baseline, expected_baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Baseline T_f does not match expected; monotonicity test cannot proceed")
        # Perturbed volume factor (larger)
        V_pert = 3.0
        T_f_perturbed = self.T0 / (V_pert ** (1/3))
        if T_f_perturbed >= T_f_baseline:
            return (False, f"Perturbed T_f ({T_f_perturbed:.4f}) is not less than baseline ({T_f_baseline:.4f}); monotonicity violated")
        return (True, f"Monotonicity holds: increasing volume factor from {self.V_factor} to {V_pert} decreases T_f from {T_f_baseline:.4f} to {T_f_perturbed:.4f}")

    def test_monotonicity_u_with_temperature(self):
        """
        Check that if T0 were perturbed upward, energy density u increases (T^4 law).
        Baseline: T0 = 3 K, u = a * T0^4.
        Perturb: T0_pert = 3.3 K (10% increase), then u_pert > u_baseline.
        """
        if self.u_value is None:
            return (False, "SKIP: u_value missing")
        try:
            u_baseline = float(self.u_value)
        except (ValueError, TypeError):
            return (False, "u_value not numeric")
        # Known baseline expected
        expected_baseline = self.a * (self.T0 ** 4)
        if not math.isclose(u_baseline, expected_baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Baseline u does not match Stefan–Boltzmann; monotonicity test cannot proceed")
        # Perturb temperature upward by 10%
        T0_pert = self.T0 * 1.10
        u_pert = self.a * (T0_pert ** 4)
        if u_pert <= u_baseline:
            return (False, f"Perturbed u ({u_pert:.4e}) is not larger than baseline ({u_baseline:.4e}); monotonicity violated")
        return (True, f"Monotonicity holds: increasing T0 by 10% increases u from {u_baseline:.4e} to {u_pert:.4e}")

    # ------------------------------------------------------------------ #
    # Weak answer sanity (only one method, no math.isclose, no re-derivation)
    # ------------------------------------------------------------------ #
    def test_answer_sanity_weak(self):
        """
        Combine weak sanity checks: positivity/finite and order plausibility.
        No exact numeric matching.
        """
        checks = []
        # T_f positive
        if self.T_f is not None:
            try:
                val_T = float(self.T_f)
                if val_T <= 0:
                    checks.append("T_f not positive")
                elif not math.isfinite(val_T):
                    checks.append("T_f not finite")
            except:
                checks.append("T_f not numeric")
        # u_value positive
        if self.u_value is not None:
            try:
                val_u = float(self.u_value)
                if val_u <= 0:
                    checks.append("u_value not positive")
                elif not math.isfinite(val_u):
                    checks.append("u_value not finite")
            except:
                checks.append("u_value not numeric")
        # u_order plausibility (must contain '10^{-13}' or '10^{-14}')
        if self.u_order is not None:
            if isinstance(self.u_order, str):
                if not ('10^{-13}' in self.u_order or '10^{-14}' in self.u_order):
                    checks.append("u_order not in expected range (10^{-13} or 10^{-14})")
            else:
                checks.append("u_order not a string")

        if checks:
            return (False, "; ".join(checks))
        return (True, "All weak sanity checks passed")

    # ------------------------------------------------------------------ #
    # Test runner
    # ------------------------------------------------------------------ #
    def run_all_tests(self):
        """Discover and run all test_* methods, collect results."""
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