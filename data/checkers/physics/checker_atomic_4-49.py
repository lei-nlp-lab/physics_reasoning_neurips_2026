import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits the solution for the pion decay problem:
    - Mean rest lifetime τ = 2.6e-8 s
    - Distance L = 10 m
    - 10% of pions decay (N/N0 = 0.9)
    - Required outputs: momentum p, kinetic energy T
    Physics law: exponential decay with time dilation and relativistic kinematics.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.p = results.get('p', None)
        self.T = results.get('T', None)

        # Extract givens from solution_instance if available
        self.L = getattr(solution_instance, 'L', None)
        self.tau = getattr(solution_instance, 'tau', None)
        self.c = getattr(solution_instance, 'c', None)
        self.m_pi_c2 = getattr(solution_instance, 'm_pi_c2', None)
        self.frac = getattr(solution_instance, 'frac', None)

        # Fallback: try to get from results as well
        if self.L is None:
            self.L = results.get('L', None)
        if self.tau is None:
            self.tau = results.get('tau', None)
        if self.c is None:
            self.c = results.get('c', None)
        if self.m_pi_c2 is None:
            self.m_pi_c2 = results.get('m_pi_c2', None)
        if self.frac is None:
            self.frac = results.get('frac', None)

        # Derived intermediate (if we have enough givens)
        self.gamma_beta = None
        self.gamma = None
        if all(x is not None for x in [self.L, self.c, self.tau, self.frac]):
            try:
                self.gamma_beta = self.L / (self.c * self.tau * (-math.log(self.frac)))
                self.gamma = math.sqrt(1 + self.gamma_beta**2)
            except (ValueError, ZeroDivisionError):
                pass

        # Explanation steps (for reference, not hardcoded)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface tests: check that required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_p_exists(self) -> Tuple[bool, str]:
        """Check that results['p'] is present and numeric."""
        if self.p is None:
            return (False, "Missing required output: p")
        if not isinstance(self.p, (int, float)):
            return (False, f"p must be numeric, got {type(self.p).__name__}")
        return (True, "p exists and is numeric")

    def test_interface_T_exists(self) -> Tuple[bool, str]:
        """Check that results['T'] is present and numeric."""
        if self.T is None:
            return (False, "Missing required output: T")
        if not isinstance(self.T, (int, float)):
            return (False, f"T must be numeric, got {type(self.T).__name__}")
        return (True, "T exists and is numeric")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that at least the necessary givens are present (L, tau, c, frac)."""
        missing = []
        if self.L is None: missing.append('L')
        if self.tau is None: missing.append('tau')
        if self.c is None: missing.append('c')
        if self.frac is None: missing.append('frac')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ----------------------------------------------------------------------
    # Invariant tests: physics-based sanity checks (use math.isclose if needed)
    # ----------------------------------------------------------------------
    def test_invariant_gamma_beta_positive(self) -> Tuple[bool, str]:
        """Check that gamma_beta (if derivable) is positive."""
        if self.gamma_beta is None:
            return (False, "Cannot compute gamma_beta – missing givens")
        if self.gamma_beta <= 0:
            return (False, f"gamma_beta = {self.gamma_beta} is not positive")
        return (True, f"gamma_beta = {self.gamma_beta:.6f} > 0")

    def test_invariant_gamma_greater_one(self) -> Tuple[bool, str]:
        """Check that gamma (if derivable) > 1."""
        if self.gamma is None:
            return (False, "Cannot compute gamma – missing givens")
        if self.gamma <= 1.0:
            return (False, f"gamma = {self.gamma} must be > 1")
        return (True, f"gamma = {self.gamma:.6f} > 1")

    def test_invariant_lifetime_dilation(self) -> Tuple[bool, str]:
        """
        Check consistency: t_lab = L/v and t_proper = tau * N/N0 relation.
        We derive N/N0 = 0.9 from given fraction.
        Actually we check that gamma_beta matches L/(c tau (-ln frac))
        using the given numbers, with 1% tolerance.
        This test is only run if all givens are present.
        """
        if any(x is None for x in [self.L, self.c, self.tau, self.frac]):
            return (False, "SKIP: missing givens to compute theoretical gamma_beta")
        if self.gamma_beta is None:
            return (False, "Cannot compute gamma_beta internally")
        # Compute expected using the same formula (should already be stored, but double-check)
        expected_gb = self.L / (self.c * self.tau * (-math.log(self.frac)))
        # Use isclose for fuzzy matching
        if not math.isclose(self.gamma_beta, expected_gb, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"gamma_beta computed = {self.gamma_beta}, expected = {expected_gb:.6f}")
        return (True, f"gamma_beta consistent: {self.gamma_beta:.6f}")

    def test_invariant_momentum_positive(self) -> Tuple[bool, str]:
        """Check that momentum p is positive (if available)."""
        if self.p is None:
            return (False, "SKIP: p not available")
        if self.p <= 0:
            return (False, f"p = {self.p} must be positive")
        return (True, f"p = {self.p:.6f} GeV/c > 0")

    def test_invariant_kinetic_energy_positive(self) -> Tuple[bool, str]:
        """Check that kinetic energy T is positive (if available)."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if self.T <= 0:
            return (False, f"T = {self.T} must be positive")
        return (True, f"T = {self.T:.6f} GeV > 0")

    def test_invariant_gamma_consistency(self) -> Tuple[bool, str]:
        """
        Check that gamma computed from p and m satisfies the relativistic relation.
        p = m_pi*c * gamma_beta => p^2 = (m_pi*c)^2 * (gamma^2 - 1).
        So gamma = sqrt(1 + (p/(m_pi*c))^2). If m_pi_c2 is known, compare.
        """
        if self.p is None or self.m_pi_c2 is None:
            return (False, "SKIP: needed p or m_pi_c2")
        # Compute expected gamma from p (if we had m_pi*c – note m_pi_c2 is rest energy)
        # p is in GeV/c, m_pi_c2 is in GeV, so p/m_pi_c2 is dimensionless.
        if self.m_pi_c2 <= 0:
            return (False, "Invalid m_pi_c2")
        expected_gamma = math.sqrt(1 + (self.p / self.m_pi_c2)**2)
        if self.gamma is not None:
            if not math.isclose(self.gamma, expected_gamma, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"gamma from solution = {self.gamma:.6f}, from p = {expected_gamma:.6f}")
            return (True, f"gamma consistent: {self.gamma:.6f}")
        # If gamma not computed, just check that expected_gamma >= 1
        if expected_gamma < 1.0:
            return (False, f"Derived gamma from p and mass is < 1")
        return (True, f"Derived gamma from p = {expected_gamma:.6f}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests: check one specific relation from explanation steps
    # ----------------------------------------------------------------------
    def test_trace_energy_momentum_relation(self) -> Tuple[bool, str]:
        """
        Check that T = sqrt(p^2 + m^2) - m (natural units) is satisfied.
        Since both p and T are given, and we have m_pi_c2, verify with 1% tolerance.
        This is a modeling consistency check.
        """
        if None in (self.p, self.T, self.m_pi_c2):
            return (False, "SKIP: missing p, T, or m_pi_c2")
        if self.m_pi_c2 <= 0:
            return (False, "Invalid m_pi_c2")
        total_energy = self.T + self.m_pi_c2
        expected_total = math.sqrt(self.p**2 + self.m_pi_c2**2)
        if not math.isclose(total_energy, expected_total, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T + m = {total_energy:.6f}, sqrt(p^2+m^2) = {expected_total:.6f}")
        return (True, f"Energy-momentum relation holds: T+m = sqrt(p^2+m^2)")

    def test_trace_decay_fraction_self_consistency(self) -> Tuple[bool, str]:
        """
        If we have givens, compute the fraction N/N0 from the output momentum and compare to 0.9.
        N/N0 = exp( -L / (gamma * v * tau) ) = exp( -L / (gamma_beta * c * tau) )
        where gamma_beta can be derived from p/m_pi_c2.
        """
        if any(x is None for x in [self.p, self.m_pi_c2, self.L, self.c, self.tau]):
            return (False, "SKIP: insufficient data to compute fraction")
        gamma_beta_from_p = self.p / self.m_pi_c2
        if gamma_beta_from_p <= 0:
            return (False, "gamma_beta from p is non-positive")
        computed_frac = math.exp(-self.L / (gamma_beta_from_p * self.c * self.tau))
        expected_frac = 0.9
        if not math.isclose(computed_frac, expected_frac, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Computed survival fraction = {computed_frac:.6f}, expected {expected_frac}")
        return (True, f"Survival fraction consistent: {computed_frac:.6f}")

    # ----------------------------------------------------------------------
    # Monotonicity tests: perturb one input and check direction of output change
    # ----------------------------------------------------------------------
    def test_monotonicity_distance_increases_momentum(self) -> Tuple[bool, str]:
        """
        If L increases, gamma_beta increases, so p increases.
        Use symbolic re-computation with a +10% perturbation.
        This test only runs if all givens are available.
        """
        if any(x is None for x in [self.L, self.c, self.tau, self.frac, self.m_pi_c2]):
            return (False, "SKIP: missing givens")
        # Baseline p from internal gamma_beta (we'll compute it)
        gamma_beta0 = self.L / (self.c * self.tau * (-math.log(self.frac)))
        p0 = self.m_pi_c2 * gamma_beta0
        # Perturb L by +10%
        L_pert = self.L * 1.1
        gamma_beta_pert = L_pert / (self.c * self.tau * (-math.log(self.frac)))
        p_pert = self.m_pi_c2 * gamma_beta_pert
        if p_pert <= p0:
            return (False, f"p after +10% L = {p_pert:.6f} <= baseline p = {p0:.6f}")
        return (True, f"Monotonicity holds: p increases with L (p_base={p0:.3f}, p_pert={p_pert:.3f})")

    def test_monotonicity_lifetime_decreases_momentum(self) -> Tuple[bool, str]:
        """
        If tau decreases, gamma_beta increases (since denominator smaller), so p increases.
        Use +5% increase in tau (so denominator larger) => gamma_beta decreases => p decreases.
        """
        if any(x is None for x in [self.L, self.c, self.tau, self.frac, self.m_pi_c2]):
            return (False, "SKIP: missing givens")
        gamma_beta0 = self.L / (self.c * self.tau * (-math.log(self.frac)))
        p0 = self.m_pi_c2 * gamma_beta0
        # Increase tau by 5% (makes gamma_beta smaller)
        tau_pert = self.tau * 1.05
        gamma_beta_pert = self.L / (self.c * tau_pert * (-math.log(self.frac)))
        p_pert = self.m_pi_c2 * gamma_beta_pert
        if p_pert >= p0:
            return (False, f"p after +5% tau = {p_pert:.6f} >= baseline p = {p0:.6f} (should be smaller)")
        return (True, f"Monotonicity holds: p decreases with increased tau (p_base={p0:.3f}, p_pert={p_pert:.3f})")

    # ----------------------------------------------------------------------
    # Answer sanity test (weak, only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_reasonable_range(self) -> Tuple[bool, str]:
        """
        Check that p and T are within plausible orders of magnitude.
        Expected p ~ few GeV/c, T ~ few GeV for these parameters.
        We do NOT compute expected values; just check that they are positive and
        not astronomically large or small.
        """
        if self.p is None or self.T is None:
            return (False, "SKIP: p or T missing")
        # p: should be between 0.1 and 10 GeV/c
        if not (0.1 < self.p < 10.0):
            return (False, f"p = {self.p} is outside plausible range (0.1, 10) GeV/c")
        # T: should be between 0.1 and 10 GeV
        if not (0.1 < self.T < 10.0):
            return (False, f"T = {self.T} is outside plausible range (0.1, 10) GeV")
        return (True, "p and T are in reasonable range")

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True

        # Collect all test methods
        test_methods = [name for name in dir(self) if name.startswith("test_")]
        test_methods.sort()

        for name in test_methods:
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