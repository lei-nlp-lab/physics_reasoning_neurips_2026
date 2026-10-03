import math
import traceback
import sympy as sp
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Robust checker for the solar temperature problem using Stefan-Boltzmann law,
    inverse square law, and Wien's displacement law (blackbody modeling).
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all relevant quantities from the solution instance.
        Use None as default for missing attributes/keys. No numeric fallbacks.
        """
        # Givens
        self.S = None
        if hasattr(solution_instance, 'S'):
            self.S = solution_instance.S
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.S = solution_instance.givens.get('S', None)

        self.sigma = None
        if hasattr(solution_instance, 'sigma'):
            self.sigma = solution_instance.sigma
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.sigma = solution_instance.givens.get('sigma', None)

        self.R_s = None
        if hasattr(solution_instance, 'R_s'):
            self.R_s = solution_instance.R_s
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.R_s = solution_instance.givens.get('R_s', None)

        self.d = None
        if hasattr(solution_instance, 'd'):
            self.d = solution_instance.d
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.d = solution_instance.givens.get('d', None)

        # Result
        self.T_s = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.T_s = solution_instance.results.get('T_s', None)
        elif hasattr(solution_instance, 'T_s'):
            self.T_s = solution_instance.T_s

        # Symbolic placeholders (not used in numerical checks, but available)
        self.sym = sp.symbols('S sigma R_s d T_s')

    # ------- Interface tests ------- #
    def test_interface_has_T_s(self) -> Tuple[bool, str]:
        """Check that T_s is present and numeric."""
        if self.T_s is None:
            return (False, "T_s is missing (None)")
        if not isinstance(self.T_s, (int, float, sp.core.numbers.Number)):
            return (False, f"T_s has invalid type: {type(self.T_s)}")
        return (True, "T_s is present and numeric")

    def test_interface_has_givens(self) -> Tuple[bool, str]:
        """Check all required givens are present and numeric."""
        missing = []
        for name, val in [('S', self.S), ('sigma', self.sigma),
                          ('R_s', self.R_s), ('d', self.d)]:
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float, sp.core.numbers.Number)):
                missing.append(f"{name} (invalid type)")
        if missing:
            return (False, f"Missing/invalid givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------- Invariant tests (theorem-based, use math.isclose) ------- #
    def test_invariant_stefan_boltzmann_inverse_square(self) -> Tuple[bool, str]:
        """
        Invariant: S = sigma * T_s^4 * (R_s/d)^2
        This is the combined Stefan-Boltzmann + inverse square law.
        """
        if None in (self.S, self.sigma, self.R_s, self.d, self.T_s):
            return (False, "SKIP: missing data")
        try:
            lhs = float(self.S)
            rhs = float(self.sigma) * (float(self.T_s)**4) * (float(self.R_s)/float(self.d))**2
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"S ≈ sigma*T^4*(R_s/d)^2: {lhs:.3e} vs {rhs:.3e}")
            else:
                return (False, f"Invariant violated: lhs={lhs:.3e}, rhs={rhs:.3e}, "
                        f"rel diff={abs(lhs-rhs)/max(abs(lhs),abs(rhs)):.3%}")
        except (TypeError, ValueError) as e:
            return (False, f"Computation error: {e}")

    def test_invariant_sun_surface_flux(self) -> Tuple[bool, str]:
        """
        Invariant: Flux at Sun's surface J_sun = S*(d/R_s)^2 equals sigma * T_s^4.
        """
        if None in (self.S, self.sigma, self.R_s, self.d, self.T_s):
            return (False, "SKIP: missing data")
        try:
            J_sun = float(self.S) * (float(self.d)/float(self.R_s))**2
            sigma_T4 = float(self.sigma) * (float(self.T_s)**4)
            if math.isclose(J_sun, sigma_T4, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"J_sun ≈ sigma*T^4: {J_sun:.3e} vs {sigma_T4:.3e}")
            else:
                return (False, f"Sun surface flux mismatch: J_sun={J_sun:.3e}, sigma*T^4={sigma_T4:.3e}, "
                        f"rel diff={abs(J_sun-sigma_T4)/max(abs(J_sun),abs(sigma_T4)):.3%}")
        except (TypeError, ValueError) as e:
            return (False, f"Computation error: {e}")

    # ------- Modeling / Trace tests ------- #
    def test_modeling_rearranged_equation(self) -> Tuple[bool, str]:
        """
        Verify that T_s^4 computed from givens equals (S*d^2)/(sigma*R_s^2).
        This checks the algebraic step used in the solution.
        """
        if None in (self.S, self.sigma, self.R_s, self.d, self.T_s):
            return (False, "SKIP: missing data")
        try:
            computed_T4 = (float(self.S) * float(self.d)**2) / (float(self.sigma) * float(self.R_s)**2)
            T4_from_result = float(self.T_s)**4
            if math.isclose(computed_T4, T4_from_result, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"(S*d^2)/(sigma*R_s^2) ≈ T_s^4: {computed_T4:.3e} vs {T4_from_result:.3e}")
            else:
                return (False, f"Mismatch: computed T^4 = {computed_T4:.3e}, result T^4 = {T4_from_result:.3e}")
        except (TypeError, ValueError) as e:
            return (False, f"Computation error: {e}")

    def test_modeling_wien_displacement(self) -> Tuple[bool, str]:
        """
        Modeling quality: a blackbody at temperature T_s should have
        peak wavelength λ_max = b / T_s in the visible range.
        (b = 2.898e-3 m·K = 2.898e6 nm·K)
        This checks the consistency of the blackbody model assumption.
        """
        if self.T_s is None:
            return (False, "SKIP: T_s not available")
        try:
            T = float(self.T_s)
        except (TypeError, ValueError):
            return (False, "T_s cannot be converted to float")
        if T <= 0:
            return (False, f"T_s is not positive: {T}")
        # Wien's displacement constant in nm·K
        b_nm_K = 2.898e6  # nm·K
        lambda_peak = b_nm_K / T  # nanometers
        # Typical visible range: 400-700 nm
        if 400 <= lambda_peak <= 700:
            return (True, f"λ_peak = {lambda_peak:.0f} nm is in visible range (400-700 nm) for T={T:.0f} K")
        else:
            return (False, f"λ_peak = {lambda_peak:.0f} nm is outside visible range (400-700 nm) for T={T:.0f} K")

    # ------- Monotonicity tests (explicit perturbation) ------- #
    def test_monotonicity_solar_constant_increases_temperature(self) -> Tuple[bool, str]:
        """
        Increasing S should increase T_s (from the equation).
        Perturb S up by 10%, recompute T_s from givens, compare.
        """
        if None in (self.S, self.sigma, self.R_s, self.d, self.T_s):
            return (False, "SKIP: missing data")
        try:
            S_orig = float(self.S)
            sigma = float(self.sigma)
            R_s = float(self.R_s)
            d = float(self.d)
            # Baseline T_s computed from the inversion formula
            T_orig = (S_orig * d**2 / (sigma * R_s**2)) ** 0.25
            # Perturb: increase S by 10%
            S_new = S_orig * 1.1
            T_new = (S_new * d**2 / (sigma * R_s**2)) ** 0.25
            if T_new > T_orig:
                return (True, f"+10% S raises T_s from {T_orig:.2f} to {T_new:.2f}")
            else:
                return (False, f"Monotonicity violated: T_new ({T_new:.2f}) ≤ T_orig ({T_orig:.2f})")
        except (TypeError, ValueError) as e:
            return (False, f"Computation error: {e}")

    def test_monotonicity_distance_increases_decreases_temperature(self) -> Tuple[bool, str]:
        """
        Increasing d should decrease T_s (inverse square effect).
        Perturb d up by 10%, recompute.
        """
        if None in (self.S, self.sigma, self.R_s, self.d, self.T_s):
            return (False, "SKIP: missing data")
        try:
            S = float(self.S)
            sigma = float(self.sigma)
            R_s = float(self.R_s)
            d_orig = float(self.d)
            T_orig = (S * d_orig**2 / (sigma * R_s**2)) ** 0.25
            # Perturb: increase d by 10%
            d_new = d_orig * 1.1
            T_new = (S * d_new**2 / (sigma * R_s**2)) ** 0.25
            if T_new < T_orig:
                return (True, f"+10% d decreases T_s from {T_orig:.2f} to {T_new:.2f}")
            else:
                return (False, f"Monotonicity violated: T_new ({T_new:.2f}) ≥ T_orig ({T_orig:.2f})")
        except (TypeError, ValueError) as e:
            return (False, f"Computation error: {e}")

    # ------- Weak answer sanity (at most one, no isclose, no re-derivation) ------- #
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """
        Weak sanity: T_s must be finite, positive, and in a plausible range (~1000–10000 K).
        No hardcoded expected value, no math.isclose, no re-derivation.
        """
        if self.T_s is None:
            return (False, "T_s is missing")
        try:
            T = float(self.T_s)
        except (TypeError, ValueError):
            return (False, "T_s cannot be converted to float")
        if not math.isfinite(T):
            return (False, f"T_s is not finite: {T}")
        if T <= 0:
            return (False, f"T_s is not positive: {T}")
        # Plausible range for a star like the Sun: 5000–7000 K, but enlarge to 1000–10000
        if 1000 <= T <= 10000:
            return (True, f"T_s = {T:.2f} K is finite, positive, and in plausible range")
        else:
            return (False, f"T_s = {T:.2f} K is outside 1000–10000 K range")

    # ------- Test runner ------- #
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all methods starting with 'test_', run them with try/except,
        collect logs, return (all_pass, logs).
        """
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)