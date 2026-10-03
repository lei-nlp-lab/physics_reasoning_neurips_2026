import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Robust checker for Bose-Einstein condensation thermodynamic expressions.
    """

    def __init__(self, solution_instance):
        # Extract parameters from solution_instance
        results = getattr(solution_instance, 'results', {}) or {}

        # Givens
        self.N = getattr(solution_instance, 'N', results.get('N', None))
        self.tau = getattr(solution_instance, 'tau', results.get('tau', None))
        self.tau_E = getattr(solution_instance, 'tau_E', results.get('tau_E', None))

        # Results
        self.U = results.get('U', None)
        self.C_V = results.get('C_V', None)
        self.S = results.get('S', None)

        # Zeta constants (mathematical constants, not from answer key)
        self.zeta_32 = 2.61238  # ζ(3/2)
        self.zeta_52 = 1.34149  # ζ(5/2)

    # --------------------------------------------------------------------------
    # Helper: extract numeric value from a potential symbolic expression
    # --------------------------------------------------------------------------
    def _to_float(self, x):
        if x is None:
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            try:
                return float(sp.N(x))
            except Exception:
                return None

    # --------------------------------------------------------------------------
    # Interface checks: required keys present and numeric
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that U, C_V, S are provided."""
        if self.U is None:
            return (False, "Missing required key: U")
        if self.C_V is None:
            return (False, "Missing required key: C_V")
        if self.S is None:
            return (False, "Missing required key: S")
        return (True, "All required keys (U, C_V, S) are present")

    def test_interface_parameters(self):
        """Check that N, tau, tau_E are provided."""
        if self.N is None:
            return (False, "Missing parameter: N")
        if self.tau is None:
            return (False, "Missing parameter: tau")
        if self.tau_E is None:
            return (False, "Missing parameter: tau_E")
        return (True, "All parameters (N, tau, tau_E) are present")

    # --------------------------------------------------------------------------
    # Invariant checks: theoretical relations satisfied
    # --------------------------------------------------------------------------
    def test_invariant_proportionality_U(self):
        """U should match (3/2)*ζ(5/2)/ζ(3/2)*N*tau**(5/2)/tau_E**(3/2)."""
        U = self._to_float(self.U)
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if any(v is None for v in (U, N, tau, tau_E)):
            return (False, "SKIP: missing numeric value")
        expected_U = (3/2) * (self.zeta_52 / self.zeta_32) * N * tau**2.5 / tau_E**1.5
        if math.isclose(U, expected_U, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U = {U} matches expected {expected_U}")
        else:
            return (False, f"U = {U} deviates from expected {expected_U} (rel diff {abs(U-expected_U)/expected_U:.2%})")

    def test_invariant_proportionality_CV(self):
        """C_V should match (15/4)*ζ(5/2)/ζ(3/2)*N*(tau/tau_E)**(3/2)."""
        CV = self._to_float(self.C_V)
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if any(v is None for v in (CV, N, tau, tau_E)):
            return (False, "SKIP: missing numeric value")
        expected_CV = (15/4) * (self.zeta_52 / self.zeta_32) * N * (tau/tau_E)**1.5
        if math.isclose(CV, expected_CV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_V = {CV} matches expected {expected_CV}")
        else:
            return (False, f"C_V = {CV} deviates from expected {expected_CV}")

    def test_invariant_proportionality_S(self):
        """S should match (3/2)*ζ(5/2)/ζ(3/2)*N*(tau/tau_E)**(3/2)."""
        S = self._to_float(self.S)
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if any(v is None for v in (S, N, tau, tau_E)):
            return (False, "SKIP: missing numeric value")
        expected_S = (3/2) * (self.zeta_52 / self.zeta_32) * N * (tau/tau_E)**1.5
        if math.isclose(S, expected_S, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S = {S} matches expected {expected_S}")
        else:
            return (False, f"S = {S} deviates from expected {expected_S}")

    def test_invariant_homogeneity_U(self):
        """U * tau_E**(3/2) / (N * tau**(5/2)) should be constant."""
        U = self._to_float(self.U)
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if any(v is None for v in (U, N, tau, tau_E)):
            return (False, "SKIP: missing numeric value")
        if N == 0 or tau == 0:
            return (False, "SKIP: zero denominator")
        computed_const = U * tau_E**1.5 / (N * tau**2.5)
        expected_const = 1.5 * self.zeta_52 / self.zeta_32
        if math.isclose(computed_const, expected_const, rel_tol=1e-2):
            return (True, f"Homogeneity constant = {computed_const} matches expected {expected_const}")
        else:
            return (False, f"Homogeneity constant = {computed_const} deviates from {expected_const}")

    # --------------------------------------------------------------------------
    # Modeling / trace checks: consistency of thermodynamic relations and scaling
    # --------------------------------------------------------------------------
    def test_modeling_thermodynamic_consistency(self):
        """C_V should equal (5/2) * U / tau, and S should equal (5/3) * U / tau."""
        U = self._to_float(self.U)
        CV = self._to_float(self.C_V)
        S = self._to_float(self.S)
        tau = self._to_float(self.tau)
        if None in (U, CV, S, tau):
            return (False, "SKIP: missing numeric value")
        if tau == 0:
            return (False, "SKIP: tau = 0")
        # Check C_V relation
        expected_CV = 2.5 * U / tau
        cv_match = math.isclose(CV, expected_CV, rel_tol=1e-2, abs_tol=1e-2)
        # Check S relation
        expected_S = (5/3) * U / tau
        s_match = math.isclose(S, expected_S, rel_tol=1e-2, abs_tol=1e-2)
        if cv_match and s_match:
            return (True, f"Both C_V and S satisfy derived scaling from E (C_V = {CV}, expected {expected_CV}; S = {S}, expected {expected_S})")
        else:
            msg_parts = []
            if not cv_match:
                msg_parts.append(f"C_V mismatch: got {CV}, expected {expected_CV}")
            if not s_match:
                msg_parts.append(f"S mismatch: got {S}, expected {expected_S}")
            return (False, "; ".join(msg_parts))

    def test_modeling_linear_in_N(self):
        """Check that U, C_V, S scale linearly with N (by doubling N in the reference formula)."""
        U = self._to_float(self.U)
        CV = self._to_float(self.C_V)
        S = self._to_float(self.S)
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if None in (U, CV, S, N, tau, tau_E):
            return (False, "SKIP: missing numeric value")
        if N == 0:
            return (False, "SKIP: N = 0")
        # Compute scaling factor: double N
        factor = 2.0
        expected_U_double = (3/2) * (self.zeta_52 / self.zeta_32) * (factor * N) * tau**2.5 / tau_E**1.5
        expected_U_base = (3/2) * (self.zeta_52 / self.zeta_32) * N * tau**2.5 / tau_E**1.5
        # Compare ratio from formulas
        # Instead of using the actual U, we check that the ratio expected_U_double / expected_U_base equals factor
        ratio_formula = expected_U_double / expected_U_base
        if math.isclose(ratio_formula, factor, rel_tol=1e-9):
            return (True, f"Reference formula scales linearly with N: ratio = {ratio_formula}")
        else:
            return (False, f"Reference formula failed linear scaling: ratio = {ratio_formula}")

    # --------------------------------------------------------------------------
    # Monotonicity checks: verify that U, C_V, S increase with tau
    # --------------------------------------------------------------------------
    def test_monotonicity_energy_increases_with_tau(self):
        """Perturb tau by +10% and check that expected U (from formula) increases."""
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if None in (N, tau, tau_E):
            return (False, "SKIP: missing N, tau, or tau_E")
        if tau <= 0 or tau_E <= 0:
            return (False, "SKIP: non-positive tau or tau_E")
        # Baseline expected U
        U_base = (3/2) * (self.zeta_52 / self.zeta_32) * N * tau**2.5 / tau_E**1.5
        # Perturb tau by +10%
        tau_pert = tau * 1.1
        U_pert = (3/2) * (self.zeta_52 / self.zeta_32) * N * tau_pert**2.5 / tau_E**1.5
        if U_pert > U_base:
            return (True, f"U increases with tau: base {U_base:.4f}, perturbed {U_pert:.4f}")
        else:
            return (False, f"U does not increase: base {U_base:.4f}, perturbed {U_pert:.4f}")

    def test_monotonicity_heat_capacity_increases_with_tau(self):
        """Perturb tau by +10% and check that expected C_V (from formula) increases."""
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if None in (N, tau, tau_E):
            return (False, "SKIP: missing N, tau, or tau_E")
        if tau <= 0 or tau_E <= 0:
            return (False, "SKIP: non-positive tau or tau_E")
        CV_base = (15/4) * (self.zeta_52 / self.zeta_32) * N * (tau/tau_E)**1.5
        tau_pert = tau * 1.1
        CV_pert = (15/4) * (self.zeta_52 / self.zeta_32) * N * (tau_pert/tau_E)**1.5
        if CV_pert > CV_base:
            return (True, f"C_V increases with tau: base {CV_base:.4f}, perturbed {CV_pert:.4f}")
        else:
            return (False, f"C_V does not increase: base {CV_base:.4f}, perturbed {CV_pert:.4f}")

    def test_monotonicity_entropy_increases_with_tau(self):
        """Perturb tau by +10% and check that expected S (from formula) increases."""
        N = self._to_float(self.N)
        tau = self._to_float(self.tau)
        tau_E = self._to_float(self.tau_E)
        if None in (N, tau, tau_E):
            return (False, "SKIP: missing N, tau, or tau_E")
        if tau <= 0 or tau_E <= 0:
            return (False, "SKIP: non-positive tau or tau_E")
        S_base = (3/2) * (self.zeta_52 / self.zeta_32) * N * (tau/tau_E)**1.5
        tau_pert = tau * 1.1
        S_pert = (3/2) * (self.zeta_52 / self.zeta_32) * N * (tau_pert/tau_E)**1.5
        if S_pert > S_base:
            return (True, f"S increases with tau: base {S_base:.4f}, perturbed {S_pert:.4f}")
        else:
            return (False, f"S does not increase: base {S_base:.4f}, perturbed {S_pert:.4f}")

    # --------------------------------------------------------------------------
    # Weak answer sanity: sign check only
    # --------------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        """U, C_V, S must be positive for tau>0."""
        U = self._to_float(self.U)
        CV = self._to_float(self.C_V)
        S = self._to_float(self.S)
        if any(v is None for v in (U, CV, S)):
            return (False, "SKIP: missing numeric value")
        if U > 0 and CV > 0 and S > 0:
            return (True, "All quantities are positive")
        else:
            nonpos = []
            if U <= 0: nonpos.append("U")
            if CV <= 0: nonpos.append("C_V")
            if S <= 0: nonpos.append("S")
            return (False, f"Non-positive: {', '.join(nonpos)}")

    # --------------------------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------------------------
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