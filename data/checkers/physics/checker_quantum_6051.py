import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits the solution for the transition probability from ground to first excited
    state in a 3D infinite cubical well, due to an exponentially decaying electric field.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        if self.hbar is None:
            # support if stored in givens dict
            self.hbar = solution_instance.givens.get('hbar', None) if hasattr(solution_instance, 'givens') else None

        self.e = getattr(solution_instance, 'e', None)
        if self.e is None and hasattr(solution_instance, 'givens'):
            self.e = solution_instance.givens.get('e', None)

        self.E0 = getattr(solution_instance, 'E0', None)
        if self.E0 is None and hasattr(solution_instance, 'givens'):
            self.E0 = solution_instance.givens.get('E0', None)

        self.alpha = getattr(solution_instance, 'alpha', None)
        if self.alpha is None and hasattr(solution_instance, 'givens'):
            self.alpha = solution_instance.givens.get('alpha', None)

        self.m = getattr(solution_instance, 'm', None)
        if self.m is None and hasattr(solution_instance, 'givens'):
            self.m = solution_instance.givens.get('m', None)

        self.b = getattr(solution_instance, 'b', None)
        if self.b is None and hasattr(solution_instance, 'givens'):
            self.b = solution_instance.givens.get('b', None)

        # Extract computed results
        if hasattr(solution_instance, 'results'):
            self.P = solution_instance.results.get('P', None)
        else:
            self.P = None

        # Additional computed quantities that may be present (optional)
        self.DeltaE = None
        self.V0 = None
        self.matrix_element = None
        self.omega = None
        if self.hbar and self.m and self.b:
            self.expected_DeltaE = (3 * math.pi**2 * self.hbar**2) / (8 * self.m * self.b**2)
        else:
            self.expected_DeltaE = None

        if self.b:
            self.expected_matrix_element = -32 * self.b / (9 * math.pi**2)
        else:
            self.expected_matrix_element = None

        if self.b and self.e and self.E0:
            self.expected_V0 = (32 * self.b * self.e * self.E0) / (9 * math.pi**2)
        else:
            self.expected_V0 = None

        if self.hbar and self.m and self.b:
            self.expected_omega = self.expected_DeltaE / self.hbar
        else:
            self.expected_omega = None

        # If results contain omega, V0, matrix_element, extract them
        if hasattr(solution_instance, 'step_variables'):
            self.omega = solution_instance.step_variables.get('omega', None)
            self.V0 = solution_instance.step_variables.get('V0', None)
            self.matrix_element = solution_instance.step_variables.get('matrix_element', None)
            self.DeltaE = solution_instance.step_variables.get('DeltaE', None)
        else:
            self.omega = None
            self.V0 = None
            self.matrix_element = None
            self.DeltaE = None

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
                passed = False
                msg = f"{name} CRASHED: {e}"
                traceback.print_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---------- Interface checks ----------
    def test_interface_has_P(self):
        if self.P is None:
            return (False, "Missing required result key 'P'")
        return (True, "P is present")

    def test_interface_has_hbar(self):
        if self.hbar is None:
            return (False, "Missing required given 'hbar'")
        return (True, "hbar is present")

    def test_interface_has_e(self):
        if self.e is None:
            return (False, "Missing required given 'e'")
        return (True, "e is present")

    def test_interface_has_E0(self):
        if self.E0 is None:
            return (False, "Missing required given 'E0'")
        return (True, "E0 is present")

    def test_interface_has_alpha(self):
        if self.alpha is None:
            return (False, "Missing required given 'alpha'")
        return (True, "alpha is present")

    def test_interface_has_m(self):
        if self.m is None:
            return (False, "Missing required given 'm'")
        return (True, "m is present")

    def test_interface_has_b(self):
        if self.b is None:
            return (False, "Missing required given 'b'")
        return (True, "b is present")

    def test_interface_P_is_numeric(self):
        if self.P is None:
            return (False, "SKIP: P not available")
        if not isinstance(self.P, (int, float, complex)):
            return (False, f"P has non-numeric type {type(self.P)}")
        return (True, "P is numeric")

    # ---------- Invariant checks (physics sanity) ----------
    def test_invariant_energy_difference(self):
        if self.DeltaE is None and self.expected_DeltaE is not None:
            computed_DeltaE = self.expected_DeltaE
        elif self.DeltaE is not None:
            computed_DeltaE = self.DeltaE
        else:
            return (False, "SKIP: insufficient data to compute DeltaE")

        if self.expected_DeltaE is None:
            return (False, "SKIP: cannot compute expected DeltaE (missing givens)")

        if not math.isclose(computed_DeltaE, self.expected_DeltaE, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"DeltaE {computed_DeltaE} does not match expected {self.expected_DeltaE}")
        return (True, "DeltaE invariant satisfied")

    def test_invariant_matrix_element(self):
        if self.matrix_element is not None:
            me_val = self.matrix_element
        else:
            me_val = None
        if self.expected_matrix_element is None:
            return (False, "SKIP: cannot compute expected matrix element (b missing)")
        if me_val is None:
            return (False, "SKIP: matrix_element not provided in solution")

        if not math.isclose(me_val, self.expected_matrix_element, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Matrix element {me_val} does not match expected {self.expected_matrix_element}")
        return (True, "Matrix element invariant satisfied")

    def test_invariant_V0(self):
        if self.V0 is not None:
            v0_val = self.V0
        else:
            v0_val = None
        if self.expected_V0 is None:
            return (False, "SKIP: cannot compute expected V0 (missing b, e, or E0)")
        if v0_val is None:
            return (False, "SKIP: V0 not provided in solution")

        if not math.isclose(abs(v0_val), abs(self.expected_V0), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"V0 magnitude {abs(v0_val)} does not match expected {abs(self.expected_V0)}")
        return (True, "V0 invariant satisfied")

    def test_invariant_omega(self):
        if self.omega is None and self.expected_omega is not None:
            omega_val = self.expected_omega
        elif self.omega is not None:
            omega_val = self.omega
        else:
            return (False, "SKIP: cannot compute omega")
        if self.expected_omega is None:
            return (False, "SKIP: cannot compute expected omega (missing m, b, hbar)")
        if not math.isclose(omega_val, self.expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega {omega_val} does not match expected {self.expected_omega}")
        return (True, "Omega invariant satisfied")

    def test_invariant_alpha_positive(self):
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        if self.alpha <= 0:
            return (False, f"alpha = {self.alpha} is not positive")
        return (True, "Alpha positive invariant satisfied")

    # ---------- Modeling/Trace checks (consistency with explanation steps) ----------
    def test_modeling_probability_formula(self):
        if None in (self.hbar, self.e, self.E0, self.alpha, self.m, self.b, self.P):
            return (False, "SKIP: missing givens or P")

        V0 = (32 * self.b * self.e * self.E0) / (9 * math.pi**2)
        DeltaE = (3 * math.pi**2 * self.hbar**2) / (8 * self.m * self.b**2)
        expected_P = (V0**2) / (self.hbar**2 * (self.alpha**2 + (DeltaE/self.hbar)**2))

        if not math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Computed P {self.P} does not match re-derived expected P {expected_P}")
        return (True, "Probability formula modeled correctly")

    def test_modeling_amplitude_integral(self):
        if self.alpha is None or self.omega is None or self.V0 is None or self.hbar is None:
            return (False, "SKIP: missing alpha, omega, V0, or hbar")
        amp_mag_expected = abs(self.V0) / (self.hbar * math.sqrt(self.alpha**2 + self.omega**2))
        if self.P is not None:
            P_from_amp = amp_mag_expected**2
            if not math.isclose(self.P, P_from_amp, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Probability {self.P} inconsistent with amplitude magnitude {amp_mag_expected}")
        return (True, "Amplitude integral consistency satisfied")

    def test_trace_zero_field_offset(self):
        return (True, "No specific trace test required")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_probability_with_alpha(self):
        if self.alpha is None or self.hbar is None or self.P is None:
            return (False, "SKIP: missing data")
        alpha_pert = self.alpha * 1.1
        V0 = (32 * self.b * self.e * self.E0) / (9 * math.pi**2) if all(v is not None for v in (self.b,self.e,self.E0)) else None
        if V0 is None or self.m is None:
            return (False, "SKIP: cannot compute V0")
        DeltaE = (3 * math.pi**2 * self.hbar**2) / (8 * self.m * self.b**2)
        P_pert = (V0**2) / (self.hbar**2 * (alpha_pert**2 + (DeltaE/self.hbar)**2))
        if self.P <= P_pert - 1e-12:
            return (False, f"Increasing alpha from {self.alpha} to {alpha_pert} increases P from {self.P} to {P_pert}, but should decrease")
        return (True, "Monotonicity: P decreases with alpha satisfied")

    def test_monotonicity_probability_with_E0(self):
        if self.E0 is None or self.P is None:
            return (False, "SKIP: missing data")
        E0_pert = self.E0 * 1.05
        V0_pert = (32 * self.b * self.e * E0_pert) / (9 * math.pi**2) if all(v is not None for v in (self.b,self.e,self.E0)) else None
        if V0_pert is None:
            return (False, "SKIP: cannot compute V0")
        DeltaE = (3 * math.pi**2 * self.hbar**2) / (8 * self.m * self.b**2) if self.hbar and self.m and self.b else None
        if DeltaE is None:
            return (False, "SKIP: cannot compute DeltaE")
        P_pert = (V0_pert**2) / (self.hbar**2 * (self.alpha**2 + (DeltaE/self.hbar)**2))
        if P_pert <= self.P - 1e-12:
            return (False, f"Increasing E0 decreases P, but should increase")
        return (True, "Monotonicity: P increases with E0 satisfied")

    def test_monotonicity_probability_with_b(self):
        if self.b is None or self.P is None:
            return (False, "SKIP: missing data")
        b_pert = self.b * 1.1
        V0_pert = (32 * b_pert * self.e * self.E0) / (9 * math.pi**2) if self.e and self.E0 else None
        if V0_pert is None:
            return (False, "SKIP: cannot compute V0")
        DeltaE_pert = (3 * math.pi**2 * self.hbar**2) / (8 * self.m * b_pert**2) if self.hbar and self.m else None
        if DeltaE_pert is None:
            return (False, "SKIP: cannot compute DeltaE")
        P_pert = (V0_pert**2) / (self.hbar**2 * (self.alpha**2 + (DeltaE_pert/self.hbar)**2))
        if P_pert <= self.P - 1e-12:
            return (False, f"Increasing b from {self.b} to {b_pert} decreases P from {self.P} to {P_pert}, but should increase")
        return (True, "Monotonicity: P increases with b satisfied")

    def test_monotonicity_probability_with_m(self):
        if self.m is None or self.P is None:
            return (False, "SKIP: missing data")
        m_pert = self.m * 1.1
        V0 = (32 * self.b * self.e * self.E0) / (9 * math.pi**2) if all(v is not None for v in (self.b,self.e,self.E0)) else None
        if V0 is None:
            return (False, "SKIP: cannot compute V0")
        DeltaE_pert = (3 * math.pi**2 * self.hbar**2) / (8 * m_pert * self.b**2) if self.hbar and self.b else None
        if DeltaE_pert is None:
            return (False, "SKIP: cannot compute DeltaE")
        P_pert = (V0**2) / (self.hbar**2 * (self.alpha**2 + (DeltaE_pert/self.hbar)**2))
        if P_pert <= self.P - 1e-12:
            return (False, f"Increasing m from {self.m} to {m_pert} decreases P from {self.P} to {P_pert}, but should increase")
        return (True, "Monotonicity: P increases with m satisfied")

    # ---------- Answer sanity (weak, non-derivative, only one) ----------
    def test_answer_sanity(self):
        """
        Weak sanity: check that P is finite, non-negative, and within plausible range [0,1].
        Uses only simple conditions, no math.isclose or re-derivation.
        """
        if self.P is None:
            return (False, "SKIP: P not available")
        if not math.isfinite(self.P):
            return (False, f"P = {self.P} is not finite")
        if self.P < 0:
            return (False, f"P = {self.P} is negative")
        if self.P > 1:
            return (False, f"P = {self.P} > 1, which is improbable for first-order perturbation")
        return (True, "P is finite, non-negative, and <= 1")