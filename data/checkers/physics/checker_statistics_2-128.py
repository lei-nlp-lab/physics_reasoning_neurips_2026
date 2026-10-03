import math
import traceback

class PhysicsChecker:
    """
    Physics checker for ideal gas of spinless atoms with internal two-level system.
    Checks partition function, free energy, chemical potential, entropy, pressure,
    and heat capacity at constant pressure.
    """
    def __init__(self, solution_instance):
        # Extract input parameters safely
        self.M = getattr(solution_instance, 'M', None)
        self.V = getattr(solution_instance, 'V', None)
        self.T = getattr(solution_instance, 'T', None)
        self.N = getattr(solution_instance, 'N', None)
        self.Delta = getattr(solution_instance, 'Delta', None)
        self.h = getattr(solution_instance, 'h', None)
        self.k = getattr(solution_instance, 'k', None)
        # Allow fallback for k_B
        if self.k is None:
            self.k = getattr(solution_instance, 'k_B', None)
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.z = results.get('z', None)
        self.mu = results.get('mu', None)
        self.F = results.get('F', None)
        self.S = results.get('S', None)
        self.P = results.get('P', None)
        self.C_P = results.get('C_P', None)

        # Store for easy access
        self._have_givens = all([
            self.M is not None,
            self.V is not None,
            self.T is not None,
            self.N is not None,
            self.Delta is not None,
            self.h is not None,
            self.k is not None
        ])

    def _compute_expected(self):
        """Compute expected results from givens using standard statistical mechanics."""
        if not self._have_givens:
            return None
        beta = 1.0 / (self.k * self.T)
        # Translational partition function per atom: V / lambda^3
        # lambda = h / sqrt(2 pi M k T)
        # So z_trans = V * (2 pi M k T / h^2)^(3/2)
        z_trans = self.V * (2.0 * math.pi * self.M * self.k * self.T / (self.h ** 2)) ** 1.5
        # Internal partition function (two-level, ground state energy = 0)
        z_int = 1.0 + math.exp(-beta * self.Delta)
        z = z_trans * z_int

        # Chemical potential: mu = -kT ln(z/N)
        mu = -self.k * self.T * math.log(z / self.N)

        # ln Z_N using Stirling: ln(z^N/N!) ≈ N ln z - N ln N + N
        lnZ = self.N * math.log(z) - self.N * math.log(self.N) + self.N
        F = -self.k * self.T * lnZ

        # Internal energy: translational (3/2 NkT) + internal
        U_trans = 1.5 * self.N * self.k * self.T
        U_int = self.N * self.Delta / (math.exp(beta * self.Delta) + 1.0)
        U = U_trans + U_int

        # Entropy: S = (U - F) / T
        S = (U - F) / self.T

        # Pressure: P = NkT / V
        P = self.N * self.k * self.T / self.V

        # Enthalpy: H = U + PV, then C_P = dH/dT (constant P)
        # For ideal gas, C_P = dH/dT = (5/2)Nk + Nk (βΔ)^2 e^{-βΔ} / (1+e^{-βΔ})^2
        C_P = 2.5 * self.N * self.k + self.N * self.k * (self.Delta / (self.k * self.T)) ** 2 * math.exp(-beta * self.Delta) / (1.0 + math.exp(-beta * self.Delta)) ** 2

        return {
            'z': z,
            'mu': mu,
            'F': F,
            'S': S,
            'P': P,
            'C_P': C_P
        }

    def _compute_expected_for_params(self, M, V, T, N, Delta, h, k):
        """Compute expected results for given parameters (used in monotonicity tests)."""
        beta = 1.0 / (k * T)
        z_trans = V * (2.0 * math.pi * M * k * T / (h ** 2)) ** 1.5
        z_int = 1.0 + math.exp(-beta * Delta)
        z = z_trans * z_int
        mu = -k * T * math.log(z / N)
        lnZ = N * math.log(z) - N * math.log(N) + N
        F = -k * T * lnZ
        U_trans = 1.5 * N * k * T
        U_int = N * Delta / (math.exp(beta * Delta) + 1.0)
        U = U_trans + U_int
        S = (U - F) / T
        P = N * k * T / V
        C_P = 2.5 * N * k + N * k * (Delta / (k * T)) ** 2 * math.exp(-beta * Delta) / (1.0 + math.exp(-beta * Delta)) ** 2
        return {'z': z, 'mu': mu, 'F': F, 'S': S, 'P': P, 'C_P': C_P}

    # ------------------------------------------------------------------ #
    # Interface tests
    # ------------------------------------------------------------------ #
    def test_interface_z_exists(self):
        if self.z is None:
            return (False, "SKIP: z not provided in results")
        return (True, "z exists")

    def test_interface_mu_exists(self):
        if self.mu is None:
            return (False, "SKIP: mu not provided in results")
        return (True, "mu exists")

    def test_interface_F_exists(self):
        if self.F is None:
            return (False, "SKIP: F not provided in results")
        return (True, "F exists")

    def test_interface_S_exists(self):
        if self.S is None:
            return (False, "SKIP: S not provided in results")
        return (True, "S exists")

    def test_interface_P_exists(self):
        if self.P is None:
            return (False, "SKIP: P not provided in results")
        return (True, "P exists")

    def test_interface_C_P_exists(self):
        if self.C_P is None:
            return (False, "SKIP: C_P not provided in results")
        return (True, "C_P exists")

    # ------------------------------------------------------------------ #
    # Invariant tests (fuzzy numerical matching, rel_tol=1e-2)
    # ------------------------------------------------------------------ #
    def test_invariant_z_consistency(self):
        if self.z is None:
            return (False, "SKIP: z not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens for expected computation")
        exp = self._compute_expected()
        if math.isclose(self.z, exp['z'], rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"z = {self.z:.6e} matches expected {exp['z']:.6e}")
        else:
            return (False, f"z = {self.z:.6e} differs from expected {exp['z']:.6e}")

    def test_invariant_mu_consistency(self):
        if self.mu is None:
            return (False, "SKIP: mu not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens for expected computation")
        exp = self._compute_expected()
        if math.isclose(self.mu, exp['mu'], rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"mu = {self.mu:.6e} matches expected {exp['mu']:.6e}")
        else:
            return (False, f"mu = {self.mu:.6e} differs from expected {exp['mu']:.6e}")

    def test_invariant_F_consistency(self):
        if self.F is None:
            return (False, "SKIP: F not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens for expected computation")
        exp = self._compute_expected()
        if math.isclose(self.F, exp['F'], rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F = {self.F:.6e} matches expected {exp['F']:.6e}")
        else:
            return (False, f"F = {self.F:.6e} differs from expected {exp['F']:.6e}")

    def test_invariant_S_consistency(self):
        if self.S is None:
            return (False, "SKIP: S not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens for expected computation")
        exp = self._compute_expected()
        if math.isclose(self.S, exp['S'], rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S = {self.S:.6e} matches expected {exp['S']:.6e}")
        else:
            return (False, f"S = {self.S:.6e} differs from expected {exp['S']:.6e}")

    def test_invariant_P_consistency(self):
        if self.P is None:
            return (False, "SKIP: P not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens for expected computation")
        exp = self._compute_expected()
        if math.isclose(self.P, exp['P'], rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P = {self.P:.6e} matches expected {exp['P']:.6e}")
        else:
            return (False, f"P = {self.P:.6e} differs from expected {exp['P']:.6e}")

    def test_invariant_C_P_consistency(self):
        if self.C_P is None:
            return (False, "SKIP: C_P not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens for expected computation")
        exp = self._compute_expected()
        if math.isclose(self.C_P, exp['C_P'], rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_P = {self.C_P:.6e} matches expected {exp['C_P']:.6e}")
        else:
            return (False, f"C_P = {self.C_P:.6e} differs from expected {exp['C_P']:.6e}")

    # ------------------------------------------------------------------ #
    # Modeling / trace tests (at least two)
    # ------------------------------------------------------------------ #
    def test_modeling_z_trans_expression(self):
        """Check that z_trans = V * (2 pi M k T / h^2)^(3/2) is used."""
        if self.z is None:
            return (False, "SKIP: z not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens")
        # Compute expected translational partition function directly
        z_trans_expected = self.V * (2.0 * math.pi * self.M * self.k * self.T / (self.h ** 2)) ** 1.5
        # From z, infer z_trans = z / (1 + exp(-beta*Delta))
        beta = 1.0 / (self.k * self.T)
        z_int = 1.0 + math.exp(-beta * self.Delta)
        if abs(z_int) < 1e-300:
            return (False, "z_int near zero, cannot check")
        z_trans_inferred = self.z / z_int
        if math.isclose(z_trans_inferred, z_trans_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Inferred z_trans = {z_trans_inferred:.6e} matches expected {z_trans_expected:.6e}")
        else:
            return (False, f"Inferred z_trans = {z_trans_inferred:.6e} differs from expected {z_trans_expected:.6e}")

    def test_modeling_mu_expression(self):
        """Check that mu = -kT ln(z/N) is satisfied."""
        if self.mu is None or self.z is None:
            return (False, "SKIP: mu or z not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens")
        expected_mu = -self.k * self.T * math.log(self.z / self.N)
        if math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"mu = {self.mu:.6e} matches expected {expected_mu:.6e} from formula")
        else:
            return (False, f"mu = {self.mu:.6e} differs from expected {expected_mu:.6e}")

    def test_modeling_P_expression(self):
        """Check that P = NkT/V is satisfied."""
        if self.P is None:
            return (False, "SKIP: P not available")
        if not self._have_givens:
            return (False, "SKIP: missing givens")
        expected_P = self.N * self.k * self.T / self.V
        if math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P = {self.P:.6e} matches expected {expected_P:.6e} from ideal gas law")
        else:
            return (False, f"P = {self.P:.6e} differs from expected {expected_P:.6e}")

    # ------------------------------------------------------------------ #
    # Monotonicity tests (perturb an input and compare output)
    # ------------------------------------------------------------------ #
    def test_monotonicity_z_increases_with_V(self):
        """z should increase when V increases."""
        if not self._have_givens or self.z is None:
            return (False, "SKIP: missing data")
        # Perturb V by +5%
        V_new = self.V * 1.05
        exp_base = self._compute_expected()
        exp_pert = self._compute_expected_for_params(self.M, V_new, self.T, self.N, self.Delta, self.h, self.k)
        if exp_base['z'] < exp_pert['z']:
            return (True, f"z increases when V increases from {self.V} to {V_new}")
        else:
            return (False, f"z did not increase: base {exp_base['z']}, perturbed {exp_pert['z']}")

    def test_monotonicity_z_increases_with_T(self):
        """z should increase when T increases (because lambda decreases)."""
        if not self._have_givens or self.z is None:
            return (False, "SKIP: missing data")
        T_new = self.T * 1.05
        exp_base = self._compute_expected()
        exp_pert = self._compute_expected_for_params(self.M, self.V, T_new, self.N, self.Delta, self.h, self.k)
        if exp_base['z'] < exp_pert['z']:
            return (True, f"z increases when T increases from {self.T} to {T_new}")
        else:
            return (False, f"z did not increase: base {exp_base['z']}, perturbed {exp_pert['z']}")

    def test_monotonicity_P_increases_with_N(self):
        """P should increase proportionally with N (ideal gas law)."""
        if not self._have_givens or self.P is None:
            return (False, "SKIP: missing data")
        N_new = self.N * 1.10
        exp_base = self._compute_expected()
        exp_pert = self._compute_expected_for_params(self.M, self.V, self.T, N_new, self.Delta, self.h, self.k)
        if exp_base['P'] < exp_pert['P']:
            return (True, f"P increases when N increases from {self.N} to {N_new}")
        else:
            return (False, f"P did not increase: base {exp_base['P']}, perturbed {exp_pert['P']}")

    def test_monotonicity_mu_decreases_with_N(self):
        """mu should become more negative as N increases (log term)."""
        if not self._have_givens or self.mu is None:
            return (False, "SKIP: missing data")
        N_new = self.N * 1.10
        exp_base = self._compute_expected()
        exp_pert = self._compute_expected_for_params(self.M, self.V, self.T, N_new, self.Delta, self.h, self.k)
        # mu becomes more negative (less than)
        if exp_base['mu'] > exp_pert['mu']:
            return (True, f"mu decreases (more negative) when N increases from {self.N} to {N_new}")
        else:
            return (False, f"mu did not decrease: base {exp_base['mu']}, perturbed {exp_pert['mu']}")

    # ------------------------------------------------------------------ #
    # Weak answer sanity test (only sign/finite/range checks)
    # ------------------------------------------------------------------ #
    def test_answer_sanity_positivity(self):
        """Sanity: partition function and pressure must be positive; free energy negative or positive? 
        For ideal gas, F should be negative. Mu should be negative. S positive."""
        errors = []
        if self.z is not None and self.z <= 0:
            errors.append("z should be positive")
        if self.mu is not None and self.mu > 0:
            errors.append("mu should be negative for an ideal gas")
        if self.F is not None and self.F > 0:
            errors.append("F should be negative (classical ideal gas)")
        if self.S is not None and self.S < 0:
            errors.append("S should be positive")
        if self.P is not None and self.P <= 0:
            errors.append("P should be positive")
        if self.C_P is not None and self.C_P <= 0:
            errors.append("C_P should be positive")
        if errors:
            return (False, "; ".join(errors))
        else:
            return (True, "Basic sign/range checks passed")

    # ------------------------------------------------------------------ #
    # Main runner
    # ------------------------------------------------------------------ #
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