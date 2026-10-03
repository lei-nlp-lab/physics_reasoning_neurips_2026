import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the solenoid self/mutual inductance and charge flow problem.
    Extracts required results and optional inputs from the solution instance.
    Implements unit-test-style checks for interface, invariants, modeling, monotonicity,
    and a single weak sanity check on the final answers.
    """

    def __init__(self, solution_instance):
        # --- Results (must be present) ---
        self.L = getattr(solution_instance, 'L', None)
        self.M = getattr(solution_instance, 'M', None)
        self.Q = getattr(solution_instance, 'Q', None)

        # Fallback to results dict if attributes not found
        if self.L is None and hasattr(solution_instance, 'results'):
            res = solution_instance.results
            self.L = res.get('L', None)
            self.M = res.get('M', None)
            self.Q = res.get('Q', None)

        # --- Inputs (may be missing, use None. No numeric defaults!) ---
        self.N1 = getattr(solution_instance, 'N1', None)
        self.N2 = getattr(solution_instance, 'N2', None)
        self.l = getattr(solution_instance, 'l', None)
        self.A = getattr(solution_instance, 'A', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.R = getattr(solution_instance, 'R', None)
        self.I2 = getattr(solution_instance, 'I2', None)

        # We will precompute expected values only when all needed inputs are available.
        self.expected_L = None
        self.expected_M = None
        self.expected_Q = None
        if all(x is not None for x in (self.N1, self.l, self.A, self.mu0)):
            self.expected_L = self.mu0 * (self.N1 ** 2) * self.A / self.l
        if all(x is not None for x in (self.N1, self.N2, self.l, self.A, self.mu0)):
            self.expected_M = self.mu0 * self.N1 * self.N2 * self.A / self.l
        if self.expected_M is not None and self.R is not None and self.I2 is not None:
            self.expected_Q = self.expected_M * self.I2 / self.R

    # ---------- Interface checks ----------
    def test_interface_L_present(self):
        if self.L is None:
            return (False, "L is missing")
        return (True, "L is present")

    def test_interface_M_present(self):
        if self.M is None:
            return (False, "M is missing")
        return (True, "M is present")

    def test_interface_Q_present(self):
        if self.Q is None:
            return (False, "Q is missing")
        return (True, "Q is present")

    def test_interface_L_numeric(self):
        if self.L is None:
            return (False, "SKIP: L not available")
        if isinstance(self.L, (int, float)):
            return (True, "L is numeric")
        return (False, "L is not numeric")

    def test_interface_M_numeric(self):
        if self.M is None:
            return (False, "SKIP: M not available")
        if isinstance(self.M, (int, float)):
            return (True, "M is numeric")
        return (False, "M is not numeric")

    def test_interface_Q_numeric(self):
        if self.Q is None:
            return (False, "SKIP: Q not available")
        if isinstance(self.Q, (int, float)):
            return (True, "Q is numeric")
        return (False, "Q is not numeric")

    # ---------- Invariant checks (theorem-based, using isclose) ----------
    def test_invariant_L_self_inductance_formula(self):
        """Check L ≈ mu0 * N1^2 * A / l"""
        if any(x is None for x in (self.L, self.expected_L)):
            return (False, "SKIP: missing L or input parameters")
        if math.isclose(self.L, self.expected_L, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L ≈ mu0*N1^2*A/l: L={self.L:.6e}, expected={self.expected_L:.6e}")
        return (False, f"L does not match formula: L={self.L:.6e}, expected={self.expected_L:.6e}")

    def test_invariant_M_mutual_inductance_formula(self):
        """Check M ≈ mu0 * N1 * N2 * A / l"""
        if any(x is None for x in (self.M, self.expected_M)):
            return (False, "SKIP: missing M or input parameters")
        if math.isclose(self.M, self.expected_M, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M ≈ mu0*N1*N2*A/l: M={self.M:.6e}, expected={self.expected_M:.6e}")
        return (False, f"M does not match formula: M={self.M:.6e}, expected={self.expected_M:.6e}")

    def test_invariant_M_to_L_ratio(self):
        """Check M/L ≈ N2/N1"""
        if any(x is None for x in (self.L, self.M, self.N1, self.N2)):
            return (False, "SKIP: missing L, M, N1, or N2")
        if self.L == 0:
            return (False, "SKIP: L is zero")
        ratio = self.M / self.L
        expected_ratio = self.N2 / self.N1
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M/L={ratio:.6f} ≈ N2/N1={expected_ratio:.6f}")
        return (False, f"M/L={ratio:.6f} does not equal N2/N1={expected_ratio:.6f}")

    def test_invariant_Q_from_M_and_R(self):
        """Check Q ≈ M * I2 / R"""
        if any(x is None for x in (self.Q, self.M, self.R, self.I2)):
            return (False, "SKIP: missing Q, M, R, or I2")
        expected = self.M * self.I2 / self.R
        if math.isclose(self.Q, expected, rel_tol=1e-2, abs_tol=1e-9):
            return (True, f"Q ≈ M*I2/R: Q={self.Q:.6e}, expected={expected:.6e}")
        return (False, f"Q does not match M*I2/R: Q={self.Q:.6e}, expected={expected:.6e}")

    # ---------- Modeling / trace checks (quality of solution structure) ----------
    def test_modeling_L_typical_order(self):
        """L should be in the range 1e-6 to 1e-2 H for these parameters"""
        if self.L is None:
            return (False, "SKIP: L not available")
        if 1e-6 < self.L < 1e-2:
            return (True, f"L={self.L:.6e} has plausible magnitude")
        return (False, f"L={self.L:.6e} outside expected range [1e-6, 1e-2]")

    def test_modeling_M_typical_order(self):
        """M should be in the range 1e-7 to 1e-4 H"""
        if self.M is None:
            return (False, "SKIP: M not available")
        if 1e-7 < self.M < 1e-4:
            return (True, f"M={self.M:.6e} has plausible magnitude")
        return (False, f"M={self.M:.6e} outside expected range [1e-7, 1e-4]")

    def test_modeling_Q_typical_order(self):
        """Q should be in the range 1e-9 to 1e-6 C"""
        if self.Q is None:
            return (False, "SKIP: Q not available")
        if 1e-9 < self.Q < 1e-6:
            return (True, f"Q={self.Q:.6e} has plausible magnitude")
        return (False, f"Q={self.Q:.6e} outside expected range [1e-9, 1e-6]")

    def test_modeling_N1_N2_ratio_small(self):
        """M should be smaller than L (since N2 < N1)"""
        if any(x is None for x in (self.L, self.M)):
            return (False, "SKIP: L or M missing")
        if self.L is not None and self.M is not None and self.M < self.L:
            return (True, "M < L as expected")
        return (False, f"M >= L, but expected M < L (N2 < N1)")

    # ---------- Monotonicity checks (explicit perturbation of inputs) ----------
    def test_monotonicity_L_with_N1(self):
        """If N1 increases by 10%, L should increase by factor ~1.21"""
        if any(x is None for x in (self.L, self.N1, self.l, self.A, self.mu0)):
            return (False, "SKIP: missing inputs")
        N1_new = self.N1 * 1.1
        L_perturbed = self.mu0 * (N1_new ** 2) * self.A / self.l
        expected_ratio = (N1_new / self.N1) ** 2  # 1.21
        predicted = self.L * expected_ratio
        if math.isclose(L_perturbed, predicted, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L scales with N1^2: predicted L'={predicted:.6e}, actual L_perturbed={L_perturbed:.6e}")
        return (False, f"L does not scale as N1^2: predicted L'={predicted:.6e}, actual L_perturbed={L_perturbed:.6e}")

    def test_monotonicity_L_with_l(self):
        """If l increases by 10%, L should decrease by factor ~0.909"""
        if any(x is None for x in (self.L, self.N1, self.l, self.A, self.mu0)):
            return (False, "SKIP: missing inputs")
        l_new = self.l * 1.1
        L_perturbed = self.mu0 * (self.N1 ** 2) * self.A / l_new
        predicted = self.L * (self.l / l_new)
        if math.isclose(L_perturbed, predicted, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L scales with 1/l: predicted L'={predicted:.6e}, actual L_perturbed={L_perturbed:.6e}")
        return (False, f"L does not scale as 1/l: predicted L'={predicted:.6e}, actual L_perturbed={L_perturbed:.6e}")

    def test_monotonicity_M_with_N2(self):
        """If N2 increases by 10%, M should increase by factor 1.1"""
        if any(x is None for x in (self.M, self.N1, self.N2, self.l, self.A, self.mu0)):
            return (False, "SKIP: missing inputs")
        N2_new = self.N2 * 1.1
        M_perturbed = self.mu0 * self.N1 * N2_new * self.A / self.l
        predicted = self.M * (N2_new / self.N2)
        if math.isclose(M_perturbed, predicted, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M scales with N2: predicted M'={predicted:.6e}, actual M_perturbed={M_perturbed:.6e}")
        return (False, f"M does not scale as N2: predicted M'={predicted:.6e}, actual M_perturbed={M_perturbed:.6e}")

    def test_monotonicity_Q_with_R(self):
        """If R increases by 20%, Q should decrease by factor 1/1.2"""
        if any(x is None for x in (self.Q, self.M, self.R, self.I2)):
            return (False, "SKIP: missing Q, M, R, I2")
        R_new = self.R * 1.2
        Q_perturbed = self.M * self.I2 / R_new
        predicted = self.Q * (self.R / R_new)
        if math.isclose(Q_perturbed, predicted, rel_tol=1e-2, abs_tol=1e-9):
            return (True, f"Q scales with 1/R: predicted Q'={predicted:.6e}, actual Q_perturbed={Q_perturbed:.6e}")
        return (False, f"Q does not scale as 1/R: predicted Q'={predicted:.6e}, actual Q_perturbed={Q_perturbed:.6e}")

    # ---------- Weak answer sanity checks (single method, no isclose, no re-derivation) ----------
    def test_answer_sanity_general(self):
        """All results must be positive and finite."""
        ok = True
        msgs = []
        for name, val in [("L", self.L), ("M", self.M), ("Q", self.Q)]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                ok = False
                msgs.append(f"{name} is not numeric")
                continue
            if not math.isfinite(val):
                ok = False
                msgs.append(f"{name} is not finite (value={val})")
            if val <= 0:
                ok = False
                msgs.append(f"{name} is not positive")
        if ok:
            return (True, "All available results are positive and finite")
        return (False, "; ".join(msgs))

    # ---------- Runner ----------
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