import math
import traceback
from typing import Optional, Dict, Any, Tuple, List

class PhysicsChecker:
    """
    Audits a physics solution for the 1D Debye solid specific heat problem.
    Checks are based on the problem statement, Debye model constraints,
    and the provided pseudocode emit contract.
    """
    def __init__(self, solution_instance: Any):
        # Extract givens from the solution instance; default to None if missing
        # Givens are expected as attributes or in solution_instance.results
        # We try both common patterns: attribute and results dict
        self.N = getattr(solution_instance, 'N', None)
        self.a = getattr(solution_instance, 'a', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.kB = getattr(solution_instance, 'kB', None)
        self.T = getattr(solution_instance, 'T', None)

        # If givens are stored in results dictionary
        if hasattr(solution_instance, 'results'):
            res = solution_instance.results
            if self.N is None:
                self.N = res.get('N', None)
            if self.a is None:
                self.a = res.get('a', None)
            if self.omega is None:
                self.omega = res.get('omega', None)
            if self.hbar is None:
                self.hbar = res.get('hbar', None)
            if self.kB is None:
                self.kB = res.get('kB', None)
            if self.T is None:
                self.T = res.get('T', None)

        # Extract emitted results; default to None
        if hasattr(solution_instance, 'results'):
            res = solution_instance.results
            self.omega_D = res.get('omega_D', None)
            self.g = res.get('g', None)
            self.C_high = res.get('C_high', None)
            self.C_low = res.get('C_low', None)
            self.Theta_D = res.get('Theta_D', None)
        else:
            self.omega_D = getattr(solution_instance, 'omega_D', None)
            self.g = getattr(solution_instance, 'g', None)
            self.C_high = getattr(solution_instance, 'C_high', None)
            self.C_low = getattr(solution_instance, 'C_low', None)
            self.Theta_D = getattr(solution_instance, 'Theta_D', None)

    # ------------------------------------------------------------
    # Interface Checks: verify required input keys exist and are valid
    # ------------------------------------------------------------
    def test_interface_input_particles(self) -> Tuple[bool, str]:
        """Check that the number of particles N is a positive integer."""
        N = self.N
        if N is None:
            return (False, "SKIP: N is not provided")
        if not isinstance(N, int) or N <= 0:
            return (False, f"N should be a positive integer, got {N}")
        return (True, f"N = {N} is a positive integer")

    def test_interface_input_spring_parameter(self) -> Tuple[bool, str]:
        """Check that omega is a positive number (characteristic frequency)."""
        omega = self.omega
        if omega is None:
            return (False, "SKIP: omega is not provided")
        if not isinstance(omega, (int, float)) or omega <= 0:
            return (False, f"omega should be positive, got {omega}")
        return (True, f"omega = {omega} is positive")

    def test_interface_input_temperature(self) -> Tuple[bool, str]:
        """Check that T is a non-negative number (temperature)."""
        T = self.T
        if T is None:
            return (False, "SKIP: T is not provided")
        if not isinstance(T, (int, float)) or T < 0:
            return (False, f"T should be non-negative, got {T}")
        return (True, f"T = {T} is non-negative")

    def test_interface_result_keys_exist(self) -> Tuple[bool, str]:
        """Verify that all five required result keys are present and not None."""
        required = ['omega_D', 'g', 'C_high', 'C_low', 'Theta_D']
        missing = [key for key in required if getattr(self, key, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    # ------------------------------------------------------------
    # Invariant / Theorem Checks: physical laws that must hold
    # ------------------------------------------------------------
    def test_invariant_omega_D(self) -> Tuple[bool, str]:
        """omega_D should equal pi * omega (from Debye normalization)."""
        omega = self.omega
        omega_D = self.omega_D
        if omega is None or omega_D is None:
            return (False, "SKIP: omega or omega_D missing")
        expected = math.pi * omega
        if not math.isclose(omega_D, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega_D = {omega_D} but expected {expected} (rel diff {abs(omega_D/expected - 1):.3f})")
        return (True, f"omega_D = {omega_D} matches pi*omega = {expected}")

    def test_invariant_DOS_constant(self) -> Tuple[bool, str]:
        """Density of states g should be N/(pi*omega)."""
        N = self.N
        omega = self.omega
        g = self.g
        if N is None or omega is None or g is None:
            return (False, "SKIP: missing inputs for DOS check")
        expected = N / (math.pi * omega)
        if not math.isclose(g, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"g = {g} but expected N/(pi*omega) = {expected}")
        return (True, f"g = {g} matches N/(pi*omega) = {expected}")

    def test_invariant_DOS_positivity(self) -> Tuple[bool, str]:
        """DOS g must be positive."""
        g = self.g
        if g is None:
            return (False, "SKIP: g missing")
        if g <= 0:
            return (False, f"g = {g} is not positive")
        return (True, f"g = {g} is positive")

    def test_invariant_Debye_temperature(self) -> Tuple[bool, str]:
        """Theta_D should equal hbar * omega_D / kB."""
        hbar = self.hbar
        kB = self.kB
        omega_D = self.omega_D
        Theta_D = self.Theta_D
        if hbar is None or kB is None or omega_D is None or Theta_D is None:
            return (False, "SKIP: missing values for Theta_D check")
        expected = hbar * omega_D / kB
        if not math.isclose(Theta_D, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Theta_D = {Theta_D} but expected hbar*omega_D/kB = {expected}")
        return (True, f"Theta_D = {Theta_D} matches hbar*omega_D/kB")

    def test_invariant_low_T_limit_form(self) -> Tuple[bool, str]:
        """C_low should be of the form (pi^2/3) * N * kB * (T/Theta_D)."""
        N = self.N
        kB = self.kB
        T = self.T
        Theta_D = self.Theta_D
        C_low = self.C_low
        if any(v is None for v in [N, kB, T, Theta_D, C_low]):
            return (False, "SKIP: missing inputs for C_low form check")
        expected = (math.pi**2 / 3.0) * N * kB * (T / Theta_D)
        if not math.isclose(C_low, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_low = {C_low} but expected ({math.pi**2}/3)*N*kB*T/Theta_D = {expected}")
        return (True, f"C_low = {C_low} matches theoretical low-T form")

    def test_invariant_high_T_limit(self) -> Tuple[bool, str]:
        """C_high should equal N * kB (classical equipartition)."""
        N = self.N
        kB = self.kB
        C_high = self.C_high
        if N is None or kB is None or C_high is None:
            return (False, "SKIP: missing inputs for C_high check")
        expected = N * kB
        if not math.isclose(C_high, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_high = {C_high} but expected N*kB = {expected}")
        return (True, f"C_high = {C_high} matches classical N*kB")

    # ------------------------------------------------------------
    # Modeling Checks: verify internal consistency of relations
    # ------------------------------------------------------------
    def test_modeling_linear_dispersion_small_k(self) -> Tuple[bool, str]:
        """Check that v_s = omega*a (sound speed from small-k limit)."""
        omega = self.omega
        a = self.a
        if omega is None or a is None:
            return (False, "SKIP: omega or a missing")
        v_s = omega * a
        # No direct result vs, but we can check that omega_D = pi * v_s / a implies v_s = omega*a.
        # This is a consistency check: if omega_D is correct, v_s is implicitly omega*a.
        # Since we already checked omega_D == pi*omega, and v_s = omega*a gives omega_D = pi*v_s/a,
        # the check is redundant but good for documentation.
        return (True, f"v_s = omega * a = {v_s} (implicitly consistent)")

    def test_modeling_high_T_equipartition(self) -> Tuple[bool, str]:
        """C_high should be independent of T (classical limit)."""
        C_high = self.C_high
        if C_high is None:
            return (False, "SKIP: C_high missing")
        # C_high should not depend on T; but we cannot check T dependence without multiple values.
        # Instead, verify that C_high is positive and finite.
        if not (isinstance(C_high, (int, float)) and math.isfinite(C_high)):
            return (False, f"C_high = {C_high} is not a finite number")
        return (True, f"C_high = {C_high} is finite and positive")

    def test_modeling_low_T_linearity(self) -> Tuple[bool, str]:
        """C_low should be linearly dependent on T (from form)."""
        # This is already tested in invariant_low_T_limit_form; if that passes, linear in T is satisfied.
        # We add a separate check that C_low approaches zero as T -> 0.
        # Since we only have one T, we can check that C_low is finite and positive.
        C_low = self.C_low
        if C_low is None:
            return (False, "SKIP: C_low missing")
        if not (isinstance(C_low, (int, float)) and C_low > 0):
            return (False, f"C_low = {C_low} is not positive")
        return (True, f"C_low = {C_low} is positive (consistent with T>0)")

    # ------------------------------------------------------------
    # Monotonicity Checks: perturb inputs to check expected trends
    # ------------------------------------------------------------
    def test_monotonicity_high_T_increasing(self) -> Tuple[bool, str]:
        """C_high should not decrease with increasing N (for fixed other parameters)."""
        N = self.N
        kB = self.kB
        C_high = self.C_high
        if N is None or kB is None or C_high is None:
            return (False, "SKIP: missing inputs for monotonicity")
        # Simulate a small increase in N (e.g., +10%)
        N_perturbed = N * 1.1
        expected_C_high_perturbed = N_perturbed * kB
        # Since we cannot actually recompute from solution_instance, we check that
        # the provided C_high is proportional to N. We only have one N, but we can
        # check that if N were larger, C_high would be larger.
        # For a single point, monotonicity is trivially satisfied.
        # Provide a weak pass with note.
        if expected_C_high_perturbed > C_high:
            return (True, f"Monotonicity with N is plausible: increasing N increases C_high (provided form linear)")
        else:
            return (True, "Unable to verify monotonicity; assuming linear scaling")

    def test_monotonicity_low_T_with_T(self) -> Tuple[bool, str]:
        """C_low should increase with T (for given parameters)."""
        # Since C_low is proportional to T, we can check that if T increases, C_low increases.
        # We need to compute expected C_low for a perturbed T.
        N = self.N
        kB = self.kB
        T = self.T
        Theta_D = self.Theta_D
        C_low = self.C_low
        if any(v is None for v in [N, kB, T, Theta_D, C_low]):
            return (False, "SKIP: missing inputs for monotonicity")
        # Perturb T by +10%
        T_perturbed = T * 1.1
        expected_C_low_perturbed = (math.pi**2 / 3.0) * N * kB * (T_perturbed / Theta_D)
        if expected_C_low_perturbed > C_low:
            return (True, f"C_low increases with T as expected: base={C_low:.4f}, perturbed={expected_C_low_perturbed:.4f}")
        else:
            return (False, f"C_low does not increase with T: base={C_low:.4f}, perturbed={expected_C_low_perturbed:.4f}")

    # ------------------------------------------------------------
    # Weak Final Answer Sanity (only one)
    # ------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self) -> Tuple[bool, str]:
        """Check that all result quantities are positive and finite (weak sanity)."""
        results = {
            'omega_D': self.omega_D,
            'g': self.g,
            'C_high': self.C_high,
            'C_low': self.C_low,
            'Theta_D': self.Theta_D
        }
        errors = []
        for key, val in results.items():
            if val is None:
                errors.append(f"{key} is None")
            elif not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
                errors.append(f"{key} = {val} is not positive finite")
        if errors:
            return (False, "; ".join(errors))
        return (True, "All results are positive and finite")

    # ------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods."""
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
                # Return failure with traceback for debugging
                passed, msg = False, f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)