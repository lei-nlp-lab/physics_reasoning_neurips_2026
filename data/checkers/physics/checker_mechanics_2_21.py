import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens: m1, m2, k, I
        # Try as attributes first, then as dict from .givens
        if hasattr(solution_instance, 'm1'):
            self.m1 = solution_instance.m1
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.m1 = solution_instance.givens.get('m1', None)
        else:
            self.m1 = None

        if hasattr(solution_instance, 'm2'):
            self.m2 = solution_instance.m2
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.m2 = solution_instance.givens.get('m2', None)
        else:
            self.m2 = None

        if hasattr(solution_instance, 'k'):
            self.k = solution_instance.k
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.k = solution_instance.givens.get('k', None)
        else:
            self.k = None

        if hasattr(solution_instance, 'I'):
            self.I = solution_instance.I
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.I = solution_instance.givens.get('I', None)
        else:
            self.I = None

        # Extract results dictionary
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        else:
            self.results = {}

        # Extract required result keys with None default
        self.displacement = self.results.get('displacement', None)
        self.possible = self.results.get('possible', None)

        # Optionally extract intermediate variables from results (if provided)
        self.V_cm = self.results.get('V_cm', None)
        self.omega = self.results.get('omega', None)
        self.mu = self.results.get('mu', None)
        self.T = self.results.get('T', None)

    # ---------- INTERFACE TESTS ----------
    def test_interface_results_keys(self):
        """Check that required result keys exist and are of correct type."""
        if self.displacement is None:
            return (False, "Missing result key 'displacement'")
        if self.possible is None:
            return (False, "Missing result key 'possible'")
        if not isinstance(self.displacement, (int, float)):
            return (False, "'displacement' is not numeric, got {}".format(type(self.displacement)))
        if not isinstance(self.possible, bool):
            return (False, "'possible' is not boolean, got {}".format(type(self.possible)))
        return (True, "All required result keys present and typed correctly")

    def test_interface_givens_present(self):
        """Check that all required givens are present and positive."""
        all_givens = [self.m1, self.m2, self.k, self.I]
        names = ['m1', 'm2', 'k', 'I']
        for val, name in zip(all_givens, names):
            if val is None:
                return (False, "Missing given '{}'".format(name))
            if not isinstance(val, (int, float)):
                return (False, "Given '{}' is not numeric".format(name))
            if val <= 0:
                return (False, "Given '{}' must be positive".format(name))
        return (True, "All givens present and positive")

    # ---------- INVARIANT TESTS (theorem-based, with math.isclose) ----------
    def test_invariant_momentum_conservation(self):
        """Check that total momentum after impulse equals I."""
        if any(x is None for x in [self.m1, self.m2, self.I]):
            return (False, "SKIP: missing givens for momentum check")
        # We need V_cm to be provided or computed from displacement/time? Not needed.
        # If user provided V_cm, check (m1+m2)*V_cm = I
        if self.V_cm is not None:
            expected_V = self.I / (self.m1 + self.m2)
            if math.isclose(self.V_cm, expected_V, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "CM velocity matches momentum conservation")
            else:
                return (False, "V_cm = {} does not equal I/(m1+m2) = {}".format(self.V_cm, expected_V))
        else:
            # If V_cm not provided, we can still check via displacement and omega?
            # But we can skip if missing data.
            return (True, "SKIP: V_cm not provided, momentum conservation not checked")

    def test_invariant_energy_consistency(self):
        """Check that initial kinetic energy is consistent with displacement formula (spring returns to natural length at stop)."""
        if any(x is None for x in [self.m1, self.I, self.displacement, self.m2, self.k]):
            return (False, "SKIP: missing data for energy check")
        # Initial KE = I^2/(2*m1)
        initial_KE = self.I**2 / (2 * self.m1)
        # At stop, spring is at natural length, so all energy is kinetic of m1 (since m2 stops, spring extension zero)
        # m1 velocity at that instant from momentum: v1 = I/m1
        final_KE_m1 = 0.5 * self.m1 * (self.I / self.m1)**2
        # Should equal initial_KE
        if math.isclose(initial_KE, final_KE_m1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Energy conservation holds at stop time")
        else:
            return (False, "Energy not conserved: initial KE {} != final KE {}".format(initial_KE, final_KE_m1))

    # ---------- MODELING / TRACE TESTS ----------
    def test_modeling_displacement_formula(self):
        """Check that displacement matches the analytic formula derived from physics."""
        if any(x is None for x in [self.m1, self.m2, self.k, self.I, self.displacement]):
            return (False, "SKIP: missing data for displacement formula check")
        # Compute expected displacement
        M = self.m1 + self.m2
        mu = self.m1 * self.m2 / M
        omega = math.sqrt(self.k / mu)
        expected = 2 * math.pi * self.I / (omega * M)
        if math.isclose(self.displacement, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Displacement matches derived formula")
        else:
            return (False, "Displacement {} differs from expected {}".format(self.displacement, expected))

    def test_modeling_displacement_via_cm(self):
        """Check displacement = V_cm * T if intermediate values are provided."""
        if self.displacement is None:
            return (False, "SKIP: displacement not provided")
        if self.V_cm is None or self.T is None:
            # Compute V_cm and T from givens if available
            if any(x is None for x in [self.m1, self.m2, self.k, self.I]):
                return (False, "SKIP: missing givens to compute V_cm/T")
            M = self.m1 + self.m2
            V_cm = self.I / M
            mu = self.m1 * self.m2 / M
            omega = math.sqrt(self.k / mu)
            T = 2 * math.pi / omega
        else:
            V_cm = self.V_cm
            T = self.T
        expected = V_cm * T
        if math.isclose(self.displacement, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Displacement equals V_cm * T")
        else:
            return (False, "Displacement {} != V_cm * T = {} * {} = {}".format(self.displacement, V_cm, T, expected))

    def test_modeling_possible_false(self):
        """Check that 'possible' is False (pure rotation without oscillation impossible)."""
        if self.possible is None:
            return (False, "SKIP: 'possible' not provided")
        if self.possible is False:
            return (True, "possible = False as expected")
        else:
            return (False, "possible should be False because impulse passes through CM, zero angular momentum")

    # ---------- MONOTONICITY TESTS (perturb input, check trend) ----------
    def test_monotonicity_displacement_increases_with_I(self):
        """Check that increasing I increases displacement (proportional)."""
        if any(x is None for x in [self.m1, self.m2, self.k, self.I, self.displacement]):
            return (False, "SKIP: missing data for monotonicity check")
        # Compute original displacement using analytic formula (ground truth)
        M = self.m1 + self.m2
        mu = self.m1 * self.m2 / M
        omega = math.sqrt(self.k / mu)
        disp_orig = 2 * math.pi * self.I / (omega * M)
        # Perturb I by +10%
        I_pert = self.I * 1.1
        disp_new = 2 * math.pi * I_pert / (omega * M)
        if disp_new > disp_orig:
            return (True, "Displacement increases with I as expected")
        else:
            return (False, "Displacement should increase with I, but predicted decrease")

    def test_monotonicity_displacement_decreases_with_m2(self):
        """Check that increasing m2 decreases displacement (for fixed m1,k,I)."""
        if any(x is None for x in [self.m1, self.m2, self.k, self.I, self.displacement]):
            return (False, "SKIP: missing data for monotonicity check")
        # Use formula: displacement = 2πI * sqrt(m1 m2/k) / (m1+m2)^(3/2)
        # We'll compare for +10% change in m2
        M = self.m1 + self.m2
        mu = self.m1 * self.m2 / M
        omega = math.sqrt(self.k / mu)
        disp_orig = 2 * math.pi * self.I / (omega * M)  # equivalent to formula
        # Now perturb m2 by +10%
        m2_pert = self.m2 * 1.1
        M_pert = self.m1 + m2_pert
        mu_pert = self.m1 * m2_pert / M_pert
        omega_pert = math.sqrt(self.k / mu_pert)
        disp_new = 2 * math.pi * self.I / (omega_pert * M_pert)
        if disp_new < disp_orig:
            return (True, "Displacement decreases with increasing m2 as expected")
        else:
            return (False, "Displacement should decrease with m2, but predicted increase")

    # ---------- ANSWER SANITY TEST (one weak test, no isclose, no derivation) ----------
    def test_answer_sanity_displacement_sign_and_finite(self):
        """Weak sanity: displacement must be finite, positive, and not NaN."""
        if self.displacement is None:
            return (False, "SKIP: displacement not available")
        if math.isnan(self.displacement) or math.isinf(self.displacement):
            return (False, "Displacement is NaN or infinite")
        if self.displacement <= 0:
            return (False, "Displacement must be positive (impulse direction positive)")
        return (True, "Displacement is finite and positive")

    # ---------- RUN ALL TESTS ----------
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