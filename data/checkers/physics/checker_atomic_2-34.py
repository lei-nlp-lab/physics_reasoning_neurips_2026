import math

class PhysicsChecker:
    """
    Auditor for solutions comparing alpha and proton radioactivity.
    Checks consistency with semi-empirical mass formula, Q-value sign, threshold, and monotonicity.
    """

    # Semi-empirical mass formula coefficients (MeV)
    a_V = 15.75
    a_S = 17.8
    a_C = 0.711
    a_A = 23.7
    B_alpha = 28.3  # MeV

    def __init__(self, solution_instance):
        # Safely extract expected results (None if missing)
        results = getattr(solution_instance, 'results', None) or {}
        self.Q_alpha_threshold = results.get('Q_alpha_threshold', None)
        self.Q_p_sign = results.get('Q_p_sign', None)

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------

    def B(self, A, Z):
        """Binding energy (MeV) from semi-empirical mass formula."""
        if A <= 0 or Z < 0 or Z > A:
            return None
        try:
            term_V = self.a_V * A
            term_S = self.a_S * (A ** (2.0 / 3.0))
            term_C = self.a_C * Z * (Z - 1) / (A ** (1.0 / 3.0))
            term_A = self.a_A * ((A - 2 * Z) ** 2) / A
            return term_V - term_S - term_C - term_A
        except (OverflowError, ZeroDivisionError):
            return None

    def Q_alpha(self, A, Z):
        """Q-value for alpha decay (MeV)."""
        if A < 4 or Z < 2:
            return None
        B_parent = self.B(A, Z)
        B_daughter = self.B(A - 4, Z - 2)
        if B_parent is None or B_daughter is None:
            return None
        return self.B_alpha + B_daughter - B_parent

    def Q_p(self, A, Z):
        """Q-value for proton decay (MeV)."""
        if A < 1 or Z < 1:
            return None
        B_parent = self.B(A, Z)
        B_daughter = self.B(A - 1, Z - 1)
        if B_parent is None or B_daughter is None:
            return None
        return B_daughter - B_parent  # B(p) = 0 by convention

    @staticmethod
    def beta_stable_Z(A):
        """Approximate Z for beta-stable heavy nuclei (from explanation)."""
        if A <= 0:
            return None
        return A / (2 + 0.0154 * (A ** (2.0 / 3.0)))

    # Approximate Q_alpha expression from explanation steps
    def Q_alpha_approx(self, A, Z):
        """Approximate formula from explanation steps."""
        if A <= 0 or Z < 0:
            return None
        term1 = 48.88 * (A ** (-1.0 / 3.0))
        term2 = 2.856 * Z * (A ** (-1.0 / 3.0)) * (1 - Z / (3 * A))
        term3 = 92.80 * ((1 - 2 * Z / A) ** 2)
        term4 = 35.04
        return term1 + term2 - term3 - term4

    # -----------------------------------------------------------------
    # Interface tests – verify that required result keys exist and have correct type
    # -----------------------------------------------------------------

    def test_interface_Q_alpha_threshold_present(self):
        if self.Q_alpha_threshold is None:
            return (False, "SKIP: Q_alpha_threshold not provided")
        return (True, "Q_alpha_threshold is present")

    def test_interface_Q_p_sign_present(self):
        if self.Q_p_sign is None:
            return (False, "SKIP: Q_p_sign not provided")
        return (True, "Q_p_sign is present")

    def test_interface_Q_alpha_threshold_type_string(self):
        if self.Q_alpha_threshold is None:
            return (False, "SKIP: Q_alpha_threshold not available")
        if not isinstance(self.Q_alpha_threshold, str):
            return (False, f"Expected string, got {type(self.Q_alpha_threshold)}")
        return (True, "Q_alpha_threshold is a string")

    def test_interface_Q_p_sign_type_string(self):
        if self.Q_p_sign is None:
            return (False, "SKIP: Q_p_sign not available")
        if not isinstance(self.Q_p_sign, str):
            return (False, f"Expected string, got {type(self.Q_p_sign)}")
        return (True, "Q_p_sign is a string")

    # -----------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use math.isclose)
    # -----------------------------------------------------------------

    def test_invariant_Q_alpha_positive_heavy_nuclei(self):
        # Check several heavy, beta‑stable nuclei
        test_cases = [(150, 62), (200, 82), (238, 92)]
        for A, Z in test_cases:
            Qa = self.Q_alpha(A, Z)
            if Qa is None:
                return (False, f"Could not compute Q_alpha for A={A}, Z={Z}")
            if not math.isclose(Qa, 0.0, abs_tol=1e-2) and Qa <= 0:
                return (False, f"Q_alpha({A},{Z}) = {Qa:.2f} MeV, expected > 0")
        return (True, "Q_alpha positive for heavy nuclei")

    def test_invariant_Q_p_negative_heavy_nuclei(self):
        test_cases = [(150, 62), (200, 82), (238, 92)]
        for A, Z in test_cases:
            Qp = self.Q_p(A, Z)
            if Qp is None:
                return (False, f"Could not compute Q_p for A={A}, Z={Z}")
            if not math.isclose(Qp, 0.0, abs_tol=1e-2) and Qp >= 0:
                return (False, f"Q_p({A},{Z}) = {Qp:.2f} MeV, expected < 0")
        return (True, "Q_p negative for heavy nuclei")

    def test_invariant_Q_alpha_threshold_numeric(self):
        # Find A where Q_alpha crosses zero using beta‑stable Z
        found = False
        threshold_A = None
        for A in range(130, 170):
            Z = self.beta_stable_Z(A)
            if Z is None:
                continue
            Z_int = int(round(Z))
            Qa = self.Q_alpha(A, Z_int)
            if Qa is None:
                continue
            if Qa > 0:
                found = True
                threshold_A = A
                break
        if not found:
            return (False, "Could not find positive Q_alpha in range A=130..170")
        # Check that threshold_A is in the plausible range 140-160
        if 140 <= threshold_A <= 160:
            return (True, f"Q_alpha threshold A ≈ {threshold_A} (expected ~150)")
        else:
            return (False, f"Threshold A = {threshold_A}, expected near 150")

    def test_invariant_approximate_expression_quality(self):
        # Check that the exact Q_alpha agrees with the approximate formula to within 10%
        A, Z = 200, 82
        Q_exact = self.Q_alpha(A, Z)
        Q_approx = self.Q_alpha_approx(A, Z)
        if Q_exact is None or Q_approx is None:
            return (False, "Cannot compute Q_alpha for test nucleus")
        if abs(Q_exact) < 1e-6:
            return (False, "Q_alpha near zero, comparison not meaningful")
        rel_diff = abs(Q_exact - Q_approx) / abs(Q_exact)
        if rel_diff > 0.1:
            return (False, f"Relative difference {rel_diff:.3f}, expected ≤ 0.1")
        return (True, f"Exact vs approximate Q_alpha agree within {rel_diff:.2%}")

    # -----------------------------------------------------------------
    # Modeling / trace tests – consistency with explanation steps
    # -----------------------------------------------------------------

    def test_modeling_energy_balance_alpha(self):
        # Verify that Q_alpha is derived from binding energies as stated
        A, Z = 200, 82
        Q = self.Q_alpha(A, Z)
        B_parent = self.B(A, Z)
        B_daughter = self.B(A-4, Z-2)
        expected = self.B_alpha + B_daughter - B_parent
        if Q is None or expected is None:
            return (False, "Cannot compute Q_alpha and expected")
        if not math.isclose(Q, expected, rel_tol=1e-9):
            return (False, f"Q_alpha = {Q} not equal to {expected}")
        return (True, "Q_alpha = B_alpha + B_daughter - B_parent")

    def test_modeling_energy_balance_proton(self):
        A, Z = 200, 82
        Q = self.Q_p(A, Z)
        B_parent = self.B(A, Z)
        B_daughter = self.B(A-1, Z-1)
        expected = B_daughter - B_parent  # B(p)=0
        if Q is None or expected is None:
            return (False, "Cannot compute Q_p and expected")
        if not math.isclose(Q, expected, rel_tol=1e-9):
            return (False, f"Q_p = {Q} not equal to {expected}")
        return (True, "Q_p = B_daughter - B_parent")

    # -----------------------------------------------------------------
    # Monotonicity tests – explicitly perturb inputs and compare perturbed vs baseline
    # -----------------------------------------------------------------

    def test_monotonicity_Q_alpha_increases_with_A(self):
        # Baseline: A=200, Z=82 (beta-stable)
        A_base = 200
        Z_base = 82
        Q_base = self.Q_alpha(A_base, Z_base)
        if Q_base is None:
            return (False, "Cannot compute baseline Q_alpha")
        # Perturb A by +5% (A = 210)
        A_pert = 210
        Q_pert = self.Q_alpha(A_pert, Z_base)
        if Q_pert is None:
            return (False, "Cannot compute perturbed Q_alpha")
        if not math.isclose(Q_pert, 0.0, abs_tol=1e-2) and Q_pert <= Q_base:
            return (False, f"Perturbed Q_alpha({A_pert},{Z_base}) = {Q_pert:.2f} ≤ baseline Q_alpha({A_base},{Z_base}) = {Q_base:.2f}")
        return (True, f"Q_alpha increases from {Q_base:.2f} to {Q_pert:.2f} when A increases by 5%")

    def test_monotonicity_Q_alpha_increases_with_Z(self):
        # Fixed A=200, baseline Z=80, perturb by +5% (Z=84)
        A = 200
        Z_base = 80
        Z_pert = 84
        Q_base = self.Q_alpha(A, Z_base)
        Q_pert = self.Q_alpha(A, Z_pert)
        if Q_base is None or Q_pert is None:
            return (False, "Cannot compute Q_alpha for test nuclei")
        if not math.isclose(Q_pert, 0.0, abs_tol=1e-2) and Q_pert <= Q_base:
            return (False, f"Perturbed Q_alpha({A},{Z_pert}) = {Q_pert:.2f} ≤ baseline Q_alpha({A},{Z_base}) = {Q_base:.2f}")
        return (True, f"Q_alpha increases from {Q_base:.2f} to {Q_pert:.2f} when Z increases by 5%")

    def test_monotonicity_Q_p_stays_negative_with_A(self):
        # Check that perturbing A does not make Q_p positive
        A_base = 200
        Z_base = 82
        Q_base = self.Q_p(A_base, Z_base)
        if Q_base is None:
            return (False, "Cannot compute baseline Q_p")
        # Perturb A by +10% (A=220, Z unchanged)
        A_pert = 220
        Q_pert = self.Q_p(A_pert, Z_base)
        if Q_pert is None:
            return (False, "Cannot compute perturbed Q_p")
        if Q_pert >= 0:
            return (False, f"Perturbed Q_p({A_pert},{Z_base}) = {Q_pert:.2f} ≥ 0")
        return (True, f"Q_p remains negative after increasing A by 10% (baseline {Q_base:.2f}, perturbed {Q_pert:.2f})")

    # -----------------------------------------------------------------
    # Weak answer-sanity test – at most one method; no math.isclose, no rederivation
    # -----------------------------------------------------------------

    def test_answer_sanity_Q_alpha_threshold_range(self):
        if self.Q_alpha_threshold is None:
            return (False, "SKIP: Q_alpha_threshold not provided")
        import re
        numbers = re.findall(r'\d+', self.Q_alpha_threshold)
        if not numbers:
            return (False, "No number found in Q_alpha_threshold string")
        # Allow a wide range (100–200) because the exact threshold depends on model
        if any(100 <= int(num) <= 200 for num in numbers):
            return (True, "Q_alpha_threshold contains a plausible number")
        return (False, f"Numbers in threshold: {numbers}, none in 100–200")

    # -----------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------

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