import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics Checker for the tritium beta decay problem (sudden approximation).
    Checks the overlap integral and probability results.
    """

    def __init__(self, solution_instance):
        # Safely extract results from the instance
        self.results = getattr(solution_instance, 'results', {})

        # Required keys: I, P, P_numeric (as per pseudocode emit)
        self.I = self.results.get('I', None)
        self.P = self.results.get('P', None)
        self.P_numeric = self.results.get('P_numeric', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_keys_exist(self):
        """Check that required keys (I, P, P_numeric) are present in results."""
        missing = []
        for key in ['I', 'P', 'P_numeric']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return (False, f"Missing key(s): {', '.join(missing)}")
        return (True, "All required keys present.")

    def test_interface_I_type(self):
        """I should be a number or sympy expression (not None)."""
        if self.I is None:
            return (False, "I is None.")
        if not isinstance(self.I, (int, float, sp.Expr, sp.Number)):
            return (False, f"I has unexpected type: {type(self.I)}.")
        return (True, "I type is acceptable.")

    def test_interface_P_type(self):
        """P should be a number or sympy expression (not None)."""
        if self.P is None:
            return (False, "P is None.")
        if not isinstance(self.P, (int, float, sp.Expr, sp.Number)):
            return (False, f"P has unexpected type: {type(self.P)}.")
        return (True, "P type is acceptable.")

    def test_interface_P_numeric_type(self):
        """P_numeric should be a float (not None)."""
        if self.P_numeric is None:
            return (False, "P_numeric is None.")
        if not isinstance(self.P_numeric, (int, float)):
            return (False, f"P_numeric has unexpected type: {type(self.P_numeric)}.")
        return (True, "P_numeric is a number.")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_probability_range(self):
        """Probability must be between 0 and 1."""
        if self.P_numeric is None:
            return (False, "SKIP: P_numeric not available.")
        p = float(self.P_numeric)
        if not (0.0 <= p <= 1.0):
            return (False, f"Probability {p} outside [0,1].")
        return (True, f"Probability {p} in [0,1].")

    def test_invariant_non_negative_I_squared(self):
        """P should be the square magnitude of I (if both numeric)."""
        if self.I is None or self.P is None:
            return (False, "SKIP: I or P not available.")
        try:
            I_val = complex(sp.N(self.I) if isinstance(self.I, sp.Expr) else self.I)
            P_val = float(sp.N(self.P) if isinstance(self.P, sp.Expr) else self.P)
        except Exception:
            return (False, "SKIP: Could not evaluate I or P numerically.")
        expected_P = abs(I_val)**2
        if not math.isclose(P_val, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P ({P_val}) differs from |I|^2 ({expected_P}).")
        return (True, f"P ({P_val}) matches |I|^2 ({expected_P}).")

    def test_invariant_numeric_consistency(self):
        """P_numeric should be close to the numerical value of P (if P symbolic)."""
        if self.P is None or self.P_numeric is None:
            return (False, "SKIP: P or P_numeric not available.")
        try:
            P_val = float(sp.N(self.P) if isinstance(self.P, sp.Expr) else self.P)
        except Exception:
            return (False, "SKIP: Could not evaluate P numerically.")
        if not math.isclose(P_val, self.P_numeric, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P numeric ({P_val}) differs from P_numeric ({self.P_numeric}).")
        return (True, f"P numeric ({P_val}) and P_numeric ({self.P_numeric}) agree.")

    # ------------------------------------------------------------------
    # Modeling/trace checks
    # ------------------------------------------------------------------
    def test_modeling_overlap_squared(self):
        """Check that P is computed as |I|^2 symbolically (if symbolic)."""
        if self.I is None or self.P is None:
            return (False, "SKIP: I or P not available.")
        if isinstance(self.I, sp.Expr) and isinstance(self.P, sp.Expr):
            I_sym = sp.simplify(self.I)
            try:
                derived_P = sp.simplify(I_sym**2)
                if sp.simplify(derived_P - self.P) != 0:
                    # Maybe I is not real? Check with conjugate
                    derived_P2 = sp.simplify(I_sym * sp.conjugate(I_sym))
                    if sp.simplify(derived_P2 - self.P) != 0:
                        return (False, "P is not |I|^2 symbolically.")
                return (True, "P = |I|^2 symbolically.")
            except Exception:
                return (False, "SKIP: symbolic simplification failed.")
        else:
            # Numeric check already done in invariant
            return (False, "SKIP: expression not symbolic.")

    def test_modeling_rational_number(self):
        """P should equal 512/729 exactly (if rational form present)."""
        if self.P is None:
            return (False, "SKIP: P not available.")
        try:
            P_val = float(sp.N(self.P) if isinstance(self.P, sp.Expr) else self.P)
        except Exception:
            return (False, "SKIP: P not numeric.")
        expected = 512/729.0
        if not math.isclose(P_val, expected, rel_tol=1e-6):
            return (False, f"P ({P_val}) differs from exact value 512/729.")
        return (True, f"P ({P_val}) matches 512/729.")

    # ------------------------------------------------------------------
    # Monotonicity check
    # ------------------------------------------------------------------
    def test_monotonicity_probability_decreases_with_final_charge(self):
        """
        Perturb Z_f from 2 to 2.1; the overlap probability should decrease.
        Baseline P from given result, perturbed P re-derived inside checker.
        """
        if self.P is None:
            return (False, "SKIP: P not available, cannot define baseline.")
        # Baseline probability from given result
        try:
            P_baseline = float(sp.N(self.P) if isinstance(self.P, sp.Expr) else self.P)
        except Exception:
            return (False, "SKIP: could not evaluate baseline P.")

        # Re-derive probability formula as function of Z_f, with Z_i=1.
        # In atomic units: ψ_i = (Z_i³ᐟ²/√π) e^{-Z_i r}, ψ_f = (Z_f³ᐟ²/√π) e^{-Z_f r}
        # Overlap: I = (Z_i Z_f)^{3/2}/π * 4π ∫_0^∞ r² e^{-(Z_i+Z_f)r} dr
        # = 4 (Z_i Z_f)^{3/2} ∫_0^∞ r² e^{-(Z_i+Z_f)r} dr
        # ∫_0^∞ r² e^{-αr} dr = 2/α³
        # So I = 8 (Z_i Z_f)^{3/2} / (Z_i+Z_f)³
        # P = I² = 64 (Z_i Z_f)³ / (Z_i+Z_f)⁶
        Z_i = 1.0
        Z_f_original = 2.0
        Z_f_perturbed = 2.1  # +5% perturbation

        def probability(Zf):
            return 64 * (Z_i * Zf)**3 / (Z_i + Zf)**6

        P_original = probability(Z_f_original)
        P_perturbed = probability(Z_f_perturbed)

        # The original probability from formula should match baseline
        if not math.isclose(P_baseline, P_original, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline P ({P_baseline}) does not match re-derived P_original ({P_original}).")

        if P_perturbed >= P_baseline:
            return (False, f"Perturbed P ({P_perturbed}) not less than baseline ({P_baseline}).")
        return (True, f"Perturbing Z_f from {Z_f_original} to {Z_f_perturbed} decreases P ({P_original} -> {P_perturbed}).")

    # ------------------------------------------------------------------
    # Weak answer sanity (one test only)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """Weak sanity: P_numeric should be roughly 0.7 (between 0.6 and 0.8)."""
        if self.P_numeric is None:
            return (False, "SKIP: P_numeric not available.")
        p = float(self.P_numeric)
        if p < 0.6 or p > 0.8:
            return (False, f"P_numeric {p} is not in reasonable range [0.6, 0.8].")
        return (True, f"P_numeric {p} in reasonable range.")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_ methods, collect results."""
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
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)