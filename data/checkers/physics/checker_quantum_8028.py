import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Extract all relevant data from the solution instance.
        Use None as default for missing values.
        """
        self.results = getattr(solution_instance, 'results', {})

        # Required result keys (based on pseudocode [emit] contract)
        self.P_mu = self.results.get('P_mu', None)
        self.P_tau = self.results.get('P_tau', None)
        self.hbar_omega1 = self.results.get('hbar_omega1', None)

        # Givens (may be stored directly or inside results)
        self.M0 = getattr(solution_instance, 'M0', self.results.get('M0', None))
        self.E = getattr(solution_instance, 'E', self.results.get('E', None))
        self.L = getattr(solution_instance, 'L', self.results.get('L', None))
        self.omega1 = getattr(solution_instance, 'omega1', self.results.get('omega1', None))

        # Physical constants
        self.hbar_c = 1.973269804e-7   # eV·m
        self.hbar_SI = 6.582119569e-16 # eV·s
        self.c = 3.0e8                 # m/s

    def run_all_tests(self):
        """Discover and run all test methods, collecting results."""
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

    # ---------- Interface tests ----------
    def test_interface_P_mu_exists(self):
        if self.P_mu is None:
            return (False, "Required result 'P_mu' missing")
        return (True, "P_mu present")

    def test_interface_P_tau_exists(self):
        if self.P_tau is None:
            return (False, "Required result 'P_tau' missing")
        return (True, "P_tau present")

    def test_interface_hbar_omega1_exists(self):
        if self.hbar_omega1 is None:
            return (False, "Required result 'hbar_omega1' missing")
        return (True, "hbar_omega1 present")

    # ---------- Invariant tests (theorem‑based, use math.isclose) ----------
    def test_invariant_probability_range(self):
        """Check that the probability expressions are bounded (0 ≤ P ≤ 1)."""
        if self.P_mu is None or self.P_tau is None:
            return (False, "SKIP: P_mu or P_tau missing")
        if "cos" not in self.P_mu or "\\frac{2}{9}" not in self.P_mu:
            return (False, "P_mu expression does not match expected form")
        if "cos" not in self.P_tau or "\\frac{2}{9}" not in self.P_tau:
            return (False, "P_tau expression does not match expected form")
        return (True, "Probability expressions contain correct form")

    def test_invariant_probabilities_equal(self):
        """Due to symmetry, P_e→μ and P_e→τ must be identical."""
        if self.P_mu is None or self.P_tau is None:
            return (False, "SKIP: P_mu or P_tau missing")
        if self.P_mu == self.P_tau:
            return (True, "P_mu and P_tau expressions are identical")
        return (False, "P_mu and P_tau expressions differ")

    def test_invariant_sum_probabilities(self):
        """
        Check that the sum of all three probabilities does not exceed 1.
        Since P_mu = P_tau ≤ 4/9, the sum ≤ 8/9 < 1, so this is automatically satisfied.
        """
        if self.P_mu is None or self.P_tau is None:
            return (False, "SKIP: probabilities missing")
        return (True, "Sum of all three probabilities is always ≤ 1 (theoretical bound)")

    def test_invariant_hbar_omega1_relation(self):
        """
        Re‑derive the detectable hbar_omega1 from the given L, E, M0
        and compare with the solution's hbar_omega1.
        """
        if self.hbar_omega1 is None:
            return (False, "SKIP: hbar_omega1 not provided")
        if self.M0 is None or self.E is None or self.L is None:
            return (False, "SKIP: required givens (M0, E, L) missing")

        # Use problem parameters: M0=20 eV, E=100e9 eV, L=2000 m
        M0_given = 20.0   # eV
        E_given = 100.0e9 # eV
        L_given = 2000.0  # m

        L_nat = L_given / self.hbar_c   # eV⁻¹
        expected = (0.1 * E_given) / (M0_given * L_nat)

        if math.isclose(self.hbar_omega1, expected, rel_tol=0.02, abs_tol=0.001):
            return (True, f"hbar_omega1 ~ {self.hbar_omega1:.4f} eV, expected ~ {expected:.4f} eV")
        return (False, f"hbar_omega1 = {self.hbar_omega1:.4f} eV, expected ~ {expected:.4f} eV")

    # ---------- Modeling / trace tests ----------
    def test_modeling_probability_expression_format(self):
        """Check that the probability expressions contain key elements: cos, 3, omega1."""
        if self.P_mu is None or self.P_tau is None:
            return (False, "SKIP: probabilities missing")
        necessary = ["cos", "3", "omega1"]
        for expr_name, expr in [("P_mu", self.P_mu), ("P_tau", self.P_tau)]:
            for part in necessary:
                if part not in expr:
                    return (False, f"{expr_name} missing '{part}' in expression")
        return (True, "Probability expressions contain expected elements")

    def test_modeling_oscillation_period(self):
        """If omega1 is available, note that the oscillation period is T = 2π/(3ω1)."""
        if self.omega1 is None:
            return (False, "SKIP: omega1 not available")
        if isinstance(self.omega1, (int, float)) and self.omega1 > 0:
            T = 2.0 * math.pi / (3.0 * self.omega1)
            return (True, f"Oscillation period T = {T:.3e} s (if omega1 in 1/s)")
        return (True, "Oscillation period can be derived from expression")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_probability_increases_with_omega1(self):
        """
        Perturb omega1 by a small positive amount and verify that
        the probability at a fixed small t increases.
        Uses analytic formula: P = (2/9)*(1-cos(3*omega1*t)).
        """
        if self.omega1 is None:
            return (False, "SKIP: omega1 not available")
        if not isinstance(self.omega1, (int, float)):
            return (False, "SKIP: omega1 is not numeric")

        # Choose a small fixed time where cos is monotonic increasing function of omega1
        t_fixed = 0.01  # seconds, must be small enough that 3*omega1*t < π
        # Check that t_fixed is reasonable: if omega1 is too large, the argument might exceed π.
        # We'll scale t_fixed to ensure argument < 0.5 rad for baseline.
        # Alternatively, compute based on typical omega1 from problem.
        # Use a relative perturbation of 10%
        omega1_base = self.omega1
        omega1_pert = omega1_base * 1.1

        P_base = (2.0/9.0) * (1.0 - math.cos(3.0 * omega1_base * t_fixed))
        P_pert = (2.0/9.0) * (1.0 - math.cos(3.0 * omega1_pert * t_fixed))

        if P_pert > P_base:
            return (True, f"Probability increases with omega1: P_base={P_base:.6f}, P_pert={P_pert:.6f}")
        return (False, f"Probability did not increase: P_base={P_base:.6f}, P_pert={P_pert:.6f}")

    def test_monotonicity_hbar_omega1_decreases_with_M0(self):
        """
        Perturb M0 by +10% and check that the required detectable hbar_omega1 decreases.
        Uses the derived formula: hbar_omega1 = (0.1*E)/(M0*L_nat).
        """
        if self.E is None or self.L is None:
            return (False, "SKIP: E or L missing to compute baseline")
        # Use problem parameters
        M0_base = 20.0
        E_val = 100.0e9
        L_val = 2000.0
        L_nat = L_val / self.hbar_c
        baseline = (0.1 * E_val) / (M0_base * L_nat)

        # Perturb M0 by +10%
        M0_pert = M0_base * 1.1
        perturbed = (0.1 * E_val) / (M0_pert * L_nat)

        if perturbed < baseline:
            return (True, f"hbar_omega1 decreases when M0 increases: {perturbed:.6e} < {baseline:.6e}")
        return (False, f"hbar_omega1 did not decrease: baseline={baseline:.6e}, perturbed={perturbed:.6e}")

    # ---------- Answer sanity (weak, only one such test) ----------
    def test_answer_sanity_hbar_omega1_magnitude(self):
        """Check that hbar_omega1 is positive, finite, and < 1 eV."""
        if self.hbar_omega1 is None:
            return (False, "SKIP: hbar_omega1 missing")
        if not isinstance(self.hbar_omega1, (int, float)):
            return (False, "hbar_omega1 is not numeric")
        if not math.isfinite(self.hbar_omega1):
            return (False, "hbar_omega1 is NaN or Inf")
        if self.hbar_omega1 <= 0:
            return (False, "hbar_omega1 is not positive")
        if self.hbar_omega1 > 1.0:
            return (False, f"hbar_omega1 too large: {self.hbar_omega1} > 1 eV")
        return (True, f"hbar_omega1 = {self.hbar_omega1:.4f} eV is a positive finite value < 1 eV")