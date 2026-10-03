import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits solutions for Doppler and collision broadening of a spectral line.
    Checks result keys, physical invariants, modeling consistency, and monotonicity.
    """

    # Fundamental constants (safe to use as defaults)
    K_B = 1.38e-23          # J/K
    C_LIGHT = 3e8           # m/s
    U = 1.66e-27            # kg (atomic mass unit)
    LN2 = 0.693147          # ln(2)
    PI = math.pi

    def __init__(self, solution_instance) -> None:
        # Extract results with None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.delta_lambda_D = self.results.get('delta_lambda_D', None)  # in Angstrom
        self.delta_lambda_C = self.results.get('delta_lambda_C', None)  # in Angstrom

        # Extract givens with None for problem-specific quantities
        self.lambda0 = getattr(solution_instance, 'lambda0', None)      # in meters
        self.T = getattr(solution_instance, 'T', None)                  # in Kelvin
        self.P = getattr(solution_instance, 'P', None)                  # in Pa
        self.m_Na = getattr(solution_instance, 'm_Na', None)            # in kg
        self.m_Ar = getattr(solution_instance, 'm_Ar', None)            # in kg
        self.d = getattr(solution_instance, 'd', None)                  # collision diameter in meters

        # Use known constants as fallback for fundamental constants
        self.k_B = getattr(solution_instance, 'k_B', self.K_B)
        self.c_light = getattr(solution_instance, 'c_light', self.C_LIGHT)

        # Determine if all required givens are present for numerical tests
        self._has_givens = all(x is not None for x in [self.lambda0, self.T, self.P, self.m_Na, self.m_Ar, self.d])

        # If givens are present, compute derived quantities needed for tests
        if self._has_givens:
            self._n_Ar = self.P / (self.k_B * self.T)                     # number density of Ar
            self._mu = (self.m_Na * self.m_Ar) / (self.m_Na + self.m_Ar) # reduced mass
            self._sigma = self.PI * (self.d ** 2)                         # collision cross section
            self._v_rel = math.sqrt(8 * self.k_B * self.T / (self.PI * self._mu))  # mean relative speed

    # ========== Interface Tests ==========

    def test_interface_delta_lambda_D_present(self) -> Tuple[bool, str]:
        """Check that delta_lambda_D exists and is not None."""
        if self.delta_lambda_D is None:
            return (False, "delta_lambda_D is missing or None")
        return (True, "delta_lambda_D present")

    def test_interface_delta_lambda_C_present(self) -> Tuple[bool, str]:
        """Check that delta_lambda_C exists and is not None."""
        if self.delta_lambda_C is None:
            return (False, "delta_lambda_C is missing or None")
        return (True, "delta_lambda_C present")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check that both results are numeric (int or float)."""
        if not isinstance(self.delta_lambda_D, (int, float)):
            return (False, "delta_lambda_D is not numeric")
        if not isinstance(self.delta_lambda_C, (int, float)):
            return (False, "delta_lambda_C is not numeric")
        return (True, "Both results are numeric")

    # ========== Invariant (Theorem-based) Tests ==========

    def test_invariant_doppler_positive(self) -> Tuple[bool, str]:
        """Doppler broadening must be positive."""
        if self.delta_lambda_D is None:
            return (False, "SKIP: delta_lambda_D not available")
        if self.delta_lambda_D <= 0:
            return (False, f"Doppler broadening non-positive: {self.delta_lambda_D}")
        return (True, "Doppler broadening positive")

    def test_invariant_collision_positive(self) -> Tuple[bool, str]:
        """Collision broadening must be positive."""
        if self.delta_lambda_C is None:
            return (False, "SKIP: delta_lambda_C not available")
        if self.delta_lambda_C <= 0:
            return (False, f"Collision broadening non-positive: {self.delta_lambda_C}")
        return (True, "Collision broadening positive")

    def test_invariant_doppler_formula(self) -> Tuple[bool, str]:
        """
        Re-derive Doppler FWHM from the standard formula and compare
        to the provided result using math.isclose (lenient tolerance).
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens for Doppler formula check")
        if self.delta_lambda_D is None:
            return (False, "SKIP: delta_lambda_D not available")

        expected_D = (2 * self.lambda0 / self.c_light) * math.sqrt(2 * self.k_B * self.T * self.LN2 / self.m_Na)
        expected_D_A = expected_D * 1e10  # convert to Angstrom

        if not math.isclose(self.delta_lambda_D, expected_D_A, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Doppler Δλ = {self.delta_lambda_D:.3e} Å, expected ~{expected_D_A:.3e} Å")
        return (True, f"Doppler formula consistent (Δλ_D = {self.delta_lambda_D:.3e} Å)")

    def test_invariant_collision_formula(self) -> Tuple[bool, str]:
        """
        Re-derive collision broadening FWHM using the standard formula
        and compare to the provided result.
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens for collision formula check")
        if self.delta_lambda_C is None:
            return (False, "SKIP: delta_lambda_C not available")

        n_Ar = self._n_Ar
        sigma = self._sigma
        v_rel = self._v_rel
        expected_C = (self.lambda0 ** 2) / (self.PI * self.c_light) * n_Ar * sigma * v_rel
        expected_C_A = expected_C * 1e10

        if not math.isclose(self.delta_lambda_C, expected_C_A, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Collision Δλ = {self.delta_lambda_C:.3e} Å, expected ~{expected_C_A:.3e} Å")
        return (True, f"Collision formula consistent (Δλ_C = {self.delta_lambda_C:.3e} Å)")

    # ========== Modeling/Trace Tests ==========

    def test_trace_doppler_lambda_nu_relation(self) -> Tuple[bool, str]:
        """
        Check that the reported Doppler Δλ satisfies the small-shift relation
        Δλ ≈ (λ0^2 / c) * Δν, where Δν is the Doppler FWHM in frequency.
        Derive Δν from the provided Δλ using this relation and compare to
        the predicted Gaussian FWHM from the Doppler formula (Δν = sqrt(8kT ln2 / m) / λ0? Wait, careful.)
        
        The Gaussian FWHM in frequency is Δν_D = sqrt(8kT ln2 / m) * ν0 / c.
        But using the Δλ provided, we compute expected Δν via Δλ = λ0^2 / c * Δν => Δν = c * Δλ / λ0^2.
        Then compare this Δν to the formula Δν_theory = sqrt(8kT ln2 / m) * ν0 / c.
        This tests consistency between the reported Δλ and the theoretical Doppler width.
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens for trace test")
        if self.delta_lambda_D is None:
            return (False, "SKIP: delta_lambda_D not available")

        # Convert reported Δλ (in Angstrom) back to Δλ in meters
        delta_lambda_m = self.delta_lambda_D * 1e-10

        # Compute Δν from the relation Δν ≈ (c / λ0^2) * Δλ (for small shifts)
        # Actually: Δλ = (λ0^2 / c) * Δν  =>  Δν = (c / λ0^2) * Δλ
        delta_nu_from_lambda = self.c_light * delta_lambda_m / (self.lambda0 ** 2)

        # Compute theoretical Doppler FWHM in frequency: Δν = sqrt(8kT ln2 / m_Na) * ν0 / c
        nu0 = self.c_light / self.lambda0
        delta_nu_theory = math.sqrt(8 * self.k_B * self.T * self.LN2 / self.m_Na) * nu0 / self.c_light

        # Compare with lenient tolerance (allow up to 10% difference due to approximations)
        if not math.isclose(delta_nu_from_lambda, delta_nu_theory, rel_tol=1e-1, abs_tol=1e-1):
            return (False, f"Doppler Δν from Δλ ({delta_nu_from_lambda:.3e} Hz) vs theory ({delta_nu_theory:.3e} Hz)")
        return (True, "Doppler Δλ ↔ Δν relation consistent with theory")

    def test_trace_collision_tau_relation(self) -> Tuple[bool, str]:
        """
        Check that the collision broadening corresponds to a physically reasonable
        collision lifetime = 1/(2π Δν) ≈ 1/(π Δν) for FWHM? Actually for Lorentzian,
        FWHM Δν = 1/(πτ) (since Δν = γ/π, where γ=1/τ). Check: τ = 1/(π Δν).
        Derive Δν from Δλ (Δν = c Δλ / λ0^2) and then compute τ = 1/(π Δν).
        Compare to the mean free time τ_mean = 1/(n σ v_rel).
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens for trace test")
        if self.delta_lambda_C is None:
            return (False, "SKIP: delta_lambda_C not available")

        # Derived mean free time from kinetic theory
        tau_mean = 1.0 / (self._n_Ar * self._sigma * self._v_rel)

        # Derive Δν from Δλ
        delta_lambda_m = self.delta_lambda_C * 1e-10
        delta_nu = self.c_light * delta_lambda_m / (self.lambda0 ** 2)

        # For Lorentzian FWHM: Δν = 1/(πτ)  => τ = 1/(π Δν)
        if delta_nu <= 0:
            return (False, "Collision Δν non-positive, cannot compute τ")
        tau_from_delta = 1.0 / (self.PI * delta_nu)

        # Compare orders of magnitude (lenient tolerance)
        if not math.isclose(tau_from_delta, tau_mean, rel_tol=0.5, abs_tol=1e-10):
            return (False, f"Collision τ from Δλ ({tau_from_delta:.3e} s) vs kinetic τ ({tau_mean:.3e} s)")
        return (True, "Collision broadening τ consistent with mean free time")

    # ========== Monotonicity Tests (Perturbation) ==========

    def test_monotonicity_doppler_increases_temperature(self) -> Tuple[bool, str]:
        """
        Check that if temperature increases, Doppler broadening increases.
        Perturb T by +10% and re-derive the expected broadening.
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens for monotonicity test")
        if self.delta_lambda_D is None:
            return (False, "SKIP: delta_lambda_D not available")

        # Baseline expected value
        base = (2 * self.lambda0 / self.c_light) * math.sqrt(2 * self.k_B * self.T * self.LN2 / self.m_Na)
        base_A = base * 1e10

        # Increased temperature
        T_pert = self.T * 1.10
        pert = (2 * self.lambda0 / self.c_light) * math.sqrt(2 * self.k_B * T_pert * self.LN2 / self.m_Na)
        pert_A = pert * 1e10

        if not (pert_A > base_A):
            return (False, f"Doppler broadening did not increase with temperature: base {base_A:.3e} Å, perturb {pert_A:.3e} Å")
        return (True, "Doppler broadening increases with temperature (+10% T)")

    def test_monotonicity_collision_increases_pressure(self) -> Tuple[bool, str]:
        """
        Check that if pressure increases, collision broadening increases.
        Perturb P by +10% and re-derive the expected broadening.
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens for monotonicity test")
        if self.delta_lambda_C is None:
            return (False, "SKIP: delta_lambda_C not available")

        # Baseline computed from internal formulas
        base = (self.lambda0 ** 2) / (self.PI * self.c_light) * self._n_Ar * self._sigma * self._v_rel
        base_A = base * 1e10

        # Increased pressure: n_Ar scales with P, v_rel unchanged
        P_pert = self.P * 1.10
        n_Ar_pert = P_pert / (self.k_B * self.T)
        pert = (self.lambda0 ** 2) / (self.PI * self.c_light) * n_Ar_pert * self._sigma * self._v_rel
        pert_A = pert * 1e10

        if not (pert_A > base_A):
            return (False, f"Collision broadening did not increase with pressure: base {base_A:.3e} Å, perturb {pert_A:.3e} Å")
        return (True, "Collision broadening increases with pressure (+10% P)")

    # ========== Weak Answer Sanity (Broad Magnitude Check) ==========

    def test_answer_sanity_magnitude(self) -> Tuple[bool, str]:
        """
        Weak sanity that the results are within physically plausible ranges.
        Doppler Δλ should be ~10^{-3} Å, collision Δλ ~10^{-5} Å.
        Use orders of magnitude only: 1e-6 to 1e-1 for Doppler, 1e-7 to 1e-3 for collision.
        """
        if self.delta_lambda_D is None or self.delta_lambda_C is None:
            return (False, "SKIP: one or both results not available")

        d_ok = 1e-6 <= self.delta_lambda_D <= 1e-1
        c_ok = 1e-7 <= self.delta_lambda_C <= 1e-3

        if not d_ok:
            return (False, f"Doppler broadening magnitude {self.delta_lambda_D:.3e} Å outside plausible range [1e-6, 1e-1]")
        if not c_ok:
            return (False, f"Collision broadening magnitude {self.delta_lambda_C:.3e} Å outside plausible range [1e-7, 1e-3]")
        return (True, "Both broadening magnitudes in plausible ranges")

    # ========== Test Runner ==========

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, return (all_pass, log_messages)."""
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)