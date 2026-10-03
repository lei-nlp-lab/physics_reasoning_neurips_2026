import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the spin-1/2 problem:
    A particle with magnetic moment μ = μ0 s and spin s=1/2
    in a constant magnetic field B along the x-axis.
    Initial state: s_z = +1/2.
    Find probabilities for s_y = ±1/2 later.

    The checker focuses on theorem-based invariants, modeling quality,
    and monotonicity. It uses the physics mapping as ground truth.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely with None defaults
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}

        # Result quantities (probabilities)
        self.P_minus = self.results.get('P_minus', None)
        self.P_plus = self.results.get('P_plus', None)

        # Givens (needed for formula checks)
        self.mu0 = self.givens.get('mu0', None)
        self.B = self.givens.get('B', None)
        self.hbar = self.givens.get('hbar', None)
        self.t = self.givens.get('t', None)

        # Also try accessing as direct attributes if givens is empty
        if self.mu0 is None:
            self.mu0 = getattr(solution_instance, 'mu0', None)
        if self.B is None:
            self.B = getattr(solution_instance, 'B', None)
        if self.hbar is None:
            self.hbar = getattr(solution_instance, 'hbar', None)
        if self.t is None:
            self.t = getattr(solution_instance, 't', None)

    # ------------------------------------------------------------------
    # Interface tests – required keys and types
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Results must contain P_minus and P_plus."""
        if 'P_minus' not in self.results:
            return (False, "Missing key 'P_minus' in results")
        if 'P_plus' not in self.results:
            return (False, "Missing key 'P_plus' in results")
        return (True, "Required keys present")

    def test_interface_numeric(self):
        """Both probabilities must be numeric (float or int), not None."""
        if self.P_minus is None or self.P_plus is None:
            return (True, "SKIP: one or both probabilities are None")
        if not isinstance(self.P_minus, (float, int)):
            return (False, f"P_minus is not numeric: {type(self.P_minus)}")
        if not isinstance(self.P_plus, (float, int)):
            return (False, f"P_plus is not numeric: {type(self.P_plus)}")
        return (True, "Both probabilities are numeric")

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (lenient matching)
    # ------------------------------------------------------------------
    def test_invariant_probability_sum(self):
        """P_minus + P_plus must equal 1 (total probability)."""
        if self.P_minus is None or self.P_plus is None:
            return (True, "SKIP: probability missing")
        if not math.isclose(self.P_minus + self.P_plus, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Sum {self.P_minus}+{self.P_plus} = {self.P_minus+self.P_plus} not close to 1")
        return (True, "Probability sum equals 1")

    def test_invariant_range(self):
        """Both probabilities must be between 0 and 1."""
        if self.P_minus is None or self.P_plus is None:
            return (True, "SKIP: probability missing")
        if not (0.0 <= self.P_minus <= 1.0):
            return (False, f"P_minus = {self.P_minus} outside [0,1]")
        if not (0.0 <= self.P_plus <= 1.0):
            return (False, f"P_plus = {self.P_plus} outside [0,1]")
        return (True, "Both probabilities in [0,1]")

    def test_invariant_initial_condition(self):
        """At t=0, P_plus should be 1, P_minus should be 0."""
        if self.t is None:
            return (True, "SKIP: t not provided")
        try:
            t_float = float(self.t)
        except (ValueError, TypeError):
            return (True, "SKIP: t not numeric")
        if not math.isclose(t_float, 0.0, abs_tol=1e-9):
            return (True, f"t = {t_float} not 0, skip initial condition check")
        if self.P_plus is None or self.P_minus is None:
            return (True, "SKIP: probability missing")
        if not math.isclose(self.P_plus, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"At t=0, P_plus = {self.P_plus} should be 1")
        if not math.isclose(self.P_minus, 0.0, abs_tol=1e-2):
            return (False, f"At t=0, P_minus = {self.P_minus} should be 0")
        return (True, "Initial condition satisfied")

    # ------------------------------------------------------------------
    # Modeling/trace tests – consistency with the predicted formulas
    # ------------------------------------------------------------------
    def test_modeling_probability_formula(self):
        """
        If all givens (mu0, B, hbar, t) are provided, check that
        P_minus = sin^2(phi) and P_plus = cos^2(phi) with phi = mu0*B*t/(2*hbar).
        """
        if any(x is None for x in [self.mu0, self.B, self.hbar, self.t]):
            return (True, "SKIP: missing givens")
        try:
            mu0 = float(self.mu0)
            B = float(self.B)
            hbar = float(self.hbar)
            t = float(self.t)
        except (ValueError, TypeError):
            return (True, "SKIP: givens not numeric")
        phi = mu0 * B * t / (2.0 * hbar)
        expected_minus = math.sin(phi)**2
        expected_plus = math.cos(phi)**2

        if self.P_minus is None or self.P_plus is None:
            return (True, "SKIP: one or both probabilities missing")

        if not math.isclose(self.P_minus, expected_minus, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_minus = {self.P_minus}, expected {expected_minus}")
        if not math.isclose(self.P_plus, expected_plus, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_plus = {self.P_plus}, expected {expected_plus}")
        return (True, "Probability formulas correct")

    def test_modeling_sy_eigenstate_overlap(self):
        """
        Re-derive the probabilities by explicitly computing the overlap
        of the time-evolved state (from initial |+z> under H = -μ0 B s_x)
        with the eigenstates of s_y. Compare to the reported probabilities.
        """
        if any(x is None for x in [self.mu0, self.B, self.hbar, self.t]):
            return (True, "SKIP: missing givens")
        try:
            mu0 = float(self.mu0)
            B = float(self.B)
            hbar = float(self.hbar)
            t = float(self.t)
        except (ValueError, TypeError):
            return (True, "SKIP: givens not numeric")

        # Time-evolved state in the standard basis (|+z>, |-z>)
        # |psi(t)> = cos(phi) |+z> + i sin(phi) |-z>    with phi = mu0 B t / (2 hbar)
        phi = mu0 * B * t / (2.0 * hbar)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        # state vector (psi_plus, psi_minus) in z-basis
        psi_plus = cos_phi
        psi_minus = 1j * sin_phi

        # Eigenstates of s_y in the z-basis:
        # |s_y = +1/2> = 1/sqrt(2) ( |+z> + i |-z> )
        # |s_y = -1/2> = 1/sqrt(2) ( |+z> - i |-z> )
        sqrt2 = math.sqrt(2.0)
        sy_plus = (1.0 / sqrt2) * (1 + 0j, 1j)      # (coeff for |+z>, coeff for |-z>)
        sy_minus = (1.0 / sqrt2) * (1 + 0j, -1j)

        # Overlaps
        overlap_plus = (sy_plus[0].conjugate() * psi_plus +
                        sy_plus[1].conjugate() * psi_minus)
        overlap_minus = (sy_minus[0].conjugate() * psi_plus +
                         sy_minus[1].conjugate() * psi_minus)

        prob_plus_calc = abs(overlap_plus)**2
        prob_minus_calc = abs(overlap_minus)**2

        if self.P_plus is None or self.P_minus is None:
            return (True, "SKIP: solution probabilities missing")

        if not math.isclose(self.P_plus, prob_plus_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_plus = {self.P_plus}, re-derived = {prob_plus_calc}")
        if not math.isclose(self.P_minus, prob_minus_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_minus = {self.P_minus}, re-derived = {prob_minus_calc}")
        return (True, "Probabilities consistent with overlap calculation")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb an input and compare
    # ------------------------------------------------------------------
    def test_monotonicity_P_plus_decreases_with_t(self):
        """
        Check that for a small positive increase in t, P_plus decreases
        when sin(2*phi) > 0 (phi in (0, pi/2) mod pi). Use the
        re‑derived P_plus formula to compute the perturbed value.
        """
        if any(x is None for x in [self.mu0, self.B, self.hbar, self.t]):
            return (True, "SKIP: missing givens")
        try:
            mu0 = float(self.mu0)
            B = float(self.B)
            hbar = float(self.hbar)
            t = float(self.t)
        except (ValueError, TypeError):
            return (True, "SKIP: givens not numeric")

        phi = mu0 * B * t / (2.0 * hbar)
        sin_2phi = math.sin(2.0 * phi)
        if abs(sin_2phi) < 1e-6:
            return (True, "SKIP: derivative near zero")

        # Choose a small perturbation (10% of t, but at least 0.1 if t=0)
        if abs(t) > 1e-9:
            delta = 0.1 * t
        else:
            delta = 0.1
        t_pert = t + delta
        phi_pert = mu0 * B * t_pert / (2.0 * hbar)
        P_plus_pert = math.cos(phi_pert)**2

        # Baseline from solution
        if self.P_plus is None:
            return (True, "SKIP: P_plus not available")
        diff = self.P_plus - P_plus_pert
        # Derivative dP_plus/dt = - (mu0 B / (2 hbar)) sin(2 phi)
        # So increasing t decreases P_plus when sin(2 phi) > 0.
        # Thus diff > 0 when sin(2 phi) > 0.
        if diff * sin_2phi <= 0:
            return (False,
                    f"Monotonicity violated: sin(2*phi)={sin_2phi:.4f}, "
                    f"diff P_plus(t)-P_plus(t+delta) = {diff:.4e}")
        return (True, f"P_plus decreases with t (sin(2*phi)={sin_2phi:.3f}, diff={diff:.4e})")

    def test_monotonicity_P_minus_increases_with_t(self):
        """
        Similarly check that P_minus increases with t when sin(2*phi) > 0.
        """
        if any(x is None for x in [self.mu0, self.B, self.hbar, self.t]):
            return (True, "SKIP: missing givens")
        try:
            mu0 = float(self.mu0)
            B = float(self.B)
            hbar = float(self.hbar)
            t = float(self.t)
        except (ValueError, TypeError):
            return (True, "SKIP: givens not numeric")

        phi = mu0 * B * t / (2.0 * hbar)
        sin_2phi = math.sin(2.0 * phi)
        if abs(sin_2phi) < 1e-6:
            return (True, "SKIP: derivative near zero")

        if abs(t) > 1e-9:
            delta = 0.1 * t
        else:
            delta = 0.1
        t_pert = t + delta
        phi_pert = mu0 * B * t_pert / (2.0 * hbar)
        P_minus_pert = math.sin(phi_pert)**2

        if self.P_minus is None:
            return (True, "SKIP: P_minus not available")
        diff = P_minus_pert - self.P_minus
        # dP_minus/dt = (mu0 B / (2 hbar)) sin(2 phi)
        # So increasing t increases P_minus when sin(2 phi) > 0.
        if diff * sin_2phi <= 0:
            return (False,
                    f"Monotonicity violated: sin(2*phi)={sin_2phi:.4f}, "
                    f"diff P_minus(t+delta)-P_minus(t) = {diff:.4e}")
        return (True, f"P_minus increases with t (sin(2*phi)={sin_2phi:.3f}, diff={diff:.4e})")

    # ------------------------------------------------------------------
    # Weak answer sanity (non‑dominant, no isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Very weak: probabilities must be real, non‑negative, and finite.
        """
        if self.P_minus is None or self.P_plus is None:
            return (True, "SKIP: probabilities not available")
        for name, val in [('P_minus', self.P_minus), ('P_plus', self.P_plus)]:
            if not isinstance(val, (int, float)):
                return (False, f"{name} not a number")
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or infinite")
            if val < 0:
                return (False, f"{name} is negative ({val})")
        return (True, "Probabilities are real, finite, non‑negative")

    # ------------------------------------------------------------------
    # Runner: discover all test_* methods and execute them safely
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
                traceback.print_exc()
                passed = False
                msg = f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)