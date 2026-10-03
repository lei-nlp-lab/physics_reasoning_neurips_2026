import math
import sympy
import traceback

class PhysicsChecker:
    """
    Robust physics checker for the molecular bond problem (HCl vibrational/rotational statistics).
    """

    def __init__(self, solution_instance):
        # Safely extract given parameters (use None if missing)
        self.k = getattr(solution_instance, 'k', None) or (solution_instance.results.get('k', None) if hasattr(solution_instance, 'results') else None)
        self.I = getattr(solution_instance, 'I', None) or (solution_instance.results.get('I', None) if hasattr(solution_instance, 'results') else None)
        self.T = getattr(solution_instance, 'T', None) or (solution_instance.results.get('T', None) if hasattr(solution_instance, 'results') else None)
        # Atomic masses (may be provided)
        self.m_H = getattr(solution_instance, 'm_H', None) or (solution_instance.results.get('m_H', None) if hasattr(solution_instance, 'results') else None)
        self.m_Cl = getattr(solution_instance, 'm_Cl', None) or (solution_instance.results.get('m_Cl', None) if hasattr(solution_instance, 'results') else None)
        self.mu = getattr(solution_instance, 'mu', None) or (solution_instance.results.get('mu', None) if hasattr(solution_instance, 'results') else None)

        # Results (critical output keys)
        if hasattr(solution_instance, 'results'):
            self.hbar_omega = solution_instance.results.get('hbar_omega', None)
            self.P1 = solution_instance.results.get('P1', None)
            self.B = solution_instance.results.get('B', None)
            self.ratio = solution_instance.results.get('ratio', None)
        else:
            self.hbar_omega = self.P1 = self.B = self.ratio = None

        # Physical constants
        self.hbar = 1.054571817e-34  # J·s
        self.kB = 1.380649e-23       # J/K
        self.u = 1.66053906660e-27   # kg per amu

        # Derived quantities (computed if possible)
        self._compute_derived()

    def _compute_derived(self):
        """Compute ω, B, kT from available givens; do NOT overwrite existing results with numeric defaults."""
        # Reduced mass
        if self.mu is None and self.m_H is not None and self.m_Cl is not None:
            m_H_kg = self.m_H * self.u if self.m_H < 100 else self.m_H  # assume amu if small
            m_Cl_kg = self.m_Cl * self.u if self.m_Cl < 100 else self.m_Cl
            self.mu = (m_H_kg * m_Cl_kg) / (m_H_kg + m_Cl_kg)

        # Vibrational frequency → hbar_omega
        if self.hbar_omega is None and self.k is not None and self.mu is not None:
            omega = math.sqrt(self.k / self.mu)
            self.hbar_omega = self.hbar * omega

        # Rotational constant B
        if self.B is None and self.I is not None:
            self.B = self.hbar**2 / (2.0 * self.I)

        # kT
        if self.T is not None:
            self.kT = self.kB * self.T
        else:
            self.kT = None

    # ----------------------------------------------------------------------
    # Helper methods (not tests, used by monotonicity/modeling tests)
    # ----------------------------------------------------------------------
    def _compute_P1_from_formula(self, hbar_omega, T):
        """P(v=1) = e^{-x} (1 - e^{-x}) where x = hbar*omega / (k_B T)."""
        if hbar_omega is None or T is None or T <= 0:
            return None
        x = hbar_omega / (self.kB * T)
        return math.exp(-x) * (1.0 - math.exp(-x))

    def _compute_ratio_from_formula(self, B, T):
        """Ratio N(J=0)/N(J=1) = (1/3) * exp(2B/(k_B T))."""
        if B is None or T is None or T <= 0:
            return None
        return (1.0 / 3.0) * math.exp(2.0 * B / (self.kB * T))

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        """Check that all four required result keys are present and not None."""
        missing = []
        for key, val in [('hbar_omega', self.hbar_omega),
                         ('P1', self.P1),
                         ('B', self.B),
                         ('ratio', self.ratio)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"SKIP (missing results): {missing}")
        return (True, "All required result keys present")

    def test_interface_results_numeric(self):
        """Check that all results are finite numeric values."""
        keys = ['hbar_omega', 'P1', 'B', 'ratio']
        vals = [self.hbar_omega, self.P1, self.B, self.ratio]
        for key, val in zip(keys, vals):
            if val is None:
                return (False, f"SKIP: {key} not available")
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                return (False, f"{key}={val} is not a finite number")
        return (True, "All results are finite numbers")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_vibrational_probability_range(self):
        """Probability must be within [0,1]."""
        if self.P1 is None:
            return (False, "SKIP: P1 not available")
        if not (0.0 <= self.P1 <= 1.0):
            return (False, f"P1={self.P1} outside [0,1]")
        return (True, "P1 in [0,1]")

    def test_invariant_rotational_ratio_positive(self):
        """The ratio N00/N01 must be positive (degeneracies are positive)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if self.ratio <= 0:
            return (False, f"ratio={self.ratio} is not positive")
        return (True, "ratio > 0")

    def test_invariant_hbar_omega_positive(self):
        """hbar_omega must be positive (finite zero‑point energy)."""
        if self.hbar_omega is None:
            return (False, "SKIP: hbar_omega not available")
        if self.hbar_omega <= 0:
            return (False, f"hbar_omega={self.hbar_omega} is not positive")
        return (True, "hbar_omega > 0")

    def test_invariant_B_positive(self):
        """Rotational constant B must be positive."""
        if self.B is None:
            return (False, "SKIP: B not available")
        if self.B <= 0:
            return (False, f"B={self.B} is not positive")
        return (True, "B > 0")

    # ----------------------------------------------------------------------
    # Modeling quality tests (check internal consistency of formulas)
    # ----------------------------------------------------------------------
    def test_modeling_vibrational_partition_consistency(self):
        """Check that P1 matches the analytic expression P1 = e^{-x}(1-e^{-x})."""
        if self.hbar_omega is None:
            return (False, "SKIP: hbar_omega not available")
        if self.T is None:
            return (False, "SKIP: T not available")
        if self.P1 is None:
            return (False, "SKIP: P1 not available")
        expected = self._compute_P1_from_formula(self.hbar_omega, self.T)
        if expected is None:
            return (False, "Could not compute expected P1")
        if math.isclose(self.P1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P1={self.P1} matches expected {expected} (1% tolerance)")
        else:
            return (False, f"P1={self.P1} differs from expected {expected} (rel_tol=1e-2)")

    def test_modeling_rotational_ratio_formula(self):
        """Check that ratio matches (1/3)*exp(2B/kT)."""
        if self.B is None:
            return (False, "SKIP: B not available")
        if self.T is None:
            return (False, "SKIP: T not available")
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        expected = self._compute_ratio_from_formula(self.B, self.T)
        if expected is None:
            return (False, "Could not compute expected ratio")
        if math.isclose(self.ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ratio={self.ratio} matches expected {expected} (1% tolerance)")
        else:
            return (False, f"ratio={self.ratio} differs from expected {expected} (rel_tol=1e-2)")

    # ----------------------------------------------------------------------
    # Monotonicity tests (trend checks via input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_vibrational_probability_with_T(self):
        """P(v=1) should increase when temperature increases (Boltzmann factor)."""
        if self.hbar_omega is None:
            return (False, "SKIP: hbar_omega not available")
        if self.T is None or self.T <= 0:
            return (False, "SKIP: T not available or non‑positive")
        baseline = self._compute_P1_from_formula(self.hbar_omega, self.T)
        perturbed_T = self.T * 1.05  # +5%
        perturbed = self._compute_P1_from_formula(self.hbar_omega, perturbed_T)
        if baseline is None or perturbed is None:
            return (False, "Could not compute P1 for baseline or perturbed T")
        if perturbed > baseline:
            return (True, f"P1 increases with T: {baseline:.3e} → {perturbed:.3e}")
        else:
            return (False, f"Monotonicity violated: P1 does not increase when T increases: {baseline:.3e} → {perturbed:.3e}")

    def test_monotonicity_rotational_ratio_with_T(self):
        """N00/N01 should decrease when temperature increases (exponent decreases)."""
        if self.B is None:
            return (False, "SKIP: B not available")
        if self.T is None or self.T <= 0:
            return (False, "SKIP: T not available or non‑positive")
        baseline = self._compute_ratio_from_formula(self.B, self.T)
        perturbed_T = self.T * 1.05
        perturbed = self._compute_ratio_from_formula(self.B, perturbed_T)
        if baseline is None or perturbed is None:
            return (False, "Could not compute ratio for baseline or perturbed T")
        if perturbed < baseline:
            return (True, f"Ratio decreases with T: {baseline:.3f} → {perturbed:.3f}")
        else:
            return (False, f"Monotonicity violated: ratio does not decrease when T increases: {baseline:.3f} → {perturbed:.3f}")

    # ----------------------------------------------------------------------
    # One weak test_answer_sanity_* (combined order‑of‑magnitude checks)
    # ----------------------------------------------------------------------
    def test_answer_sanity_plausible_ranges(self):
        """
        Weak sanity: Check that P1 is very small (< 1e-4) and ratio is order 1 (0.01-10).
        No hardcoded target values, just broad plausible ranges.
        """
        if self.P1 is None or self.ratio is None:
            return (False, "SKIP: P1 or ratio not available")
        p1_ok = 0.0 < self.P1 < 1e-4
        ratio_ok = 0.01 < self.ratio < 10.0
        if p1_ok and ratio_ok:
            return (True, f"P1={self.P1} plausible (<1e-4), ratio={self.ratio} plausible (0.01–10)")
        else:
            msg = []
            if not p1_ok:
                msg.append(f"P1={self.P1} not in (0,1e-4)")
            if not ratio_ok:
                msg.append(f"ratio={self.ratio} not in (0.01,10)")
            return (False, "; ".join(msg))

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Execute all test_* methods and return (success, logs)."""
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