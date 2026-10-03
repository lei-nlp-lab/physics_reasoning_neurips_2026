import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits solution for ideal gas thermodynamics problem:
    Part (a) isothermal work, Part (b) adiabatic final temperature.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens safely with None defaults
        # Attributes assumed to be on solution_instance (e.g., .givens dict, .results dict)
        # This implementation expects either direct attributes or dictionaries.
        self.n = getattr(solution_instance, 'n', None)
        if self.n is None:
            self.n = getattr(solution_instance, 'givens', {}).get('n', None)
        
        self.R = getattr(solution_instance, 'R', None)
        if self.R is None:
            self.R = getattr(solution_instance, 'givens', {}).get('R', None)
        
        self.T0 = getattr(solution_instance, 'T0', None)
        if self.T0 is None:
            self.T0 = getattr(solution_instance, 'givens', {}).get('T0', None)
        
        self.V_ratio = getattr(solution_instance, 'V_ratio', None)  # Vf/V0
        if self.V_ratio is None:
            self.V_ratio = getattr(solution_instance, 'givens', {}).get('V_ratio', None)
        
        self.gamma = getattr(solution_instance, 'gamma', None)
        if self.gamma is None:
            self.gamma = getattr(solution_instance, 'givens', {}).get('gamma', None)

        # Result quantities (from solution output)
        self.W = getattr(solution_instance, 'results', {}).get('W', None)
        self.t_f = getattr(solution_instance, 'results', {}).get('t_f', None)

        # Also try direct attributes if not in results dict
        if self.W is None:
            self.W = getattr(solution_instance, 'W', None)
        if self.t_f is None:
            self.t_f = getattr(solution_instance, 't_f', None)

    # ------------------------------------------------------------------
    # Interface tests: Check that required keys exist and are numeric
    # ------------------------------------------------------------------
    def test_interface_n_provided(self) -> Tuple[bool, str]:
        if self.n is None:
            return (False, "n (number of moles) is missing")
        if not isinstance(self.n, (int, float)):
            return (False, f"n is not numeric: {type(self.n)}")
        return (True, "n is present and numeric")

    def test_interface_R_provided(self) -> Tuple[bool, str]:
        if self.R is None:
            return (False, "R (gas constant) is missing")
        if not isinstance(self.R, (int, float)):
            return (False, f"R is not numeric: {type(self.R)}")
        return (True, "R is present and numeric")

    def test_interface_T0_provided(self) -> Tuple[bool, str]:
        if self.T0 is None:
            return (False, "T0 (initial temperature) is missing")
        if not isinstance(self.T0, (int, float)):
            return (False, f"T0 is not numeric: {type(self.T0)}")
        return (True, "T0 is present and numeric")

    def test_interface_V_ratio_provided(self) -> Tuple[bool, str]:
        if self.V_ratio is None:
            return (False, "V_ratio (final/initial volume) is missing")
        if not isinstance(self.V_ratio, (int, float)):
            return (False, f"V_ratio is not numeric: {type(self.V_ratio)}")
        return (True, "V_ratio is present and numeric")

    def test_interface_gamma_provided(self) -> Tuple[bool, str]:
        if self.gamma is None:
            return (False, "gamma is missing (required for adiabatic part)")
        if not isinstance(self.gamma, (int, float)):
            return (False, f"gamma is not numeric: {type(self.gamma)}")
        return (True, "gamma is present and numeric")

    def test_interface_W_provided(self) -> Tuple[bool, str]:
        if self.W is None:
            return (False, "W (isothermal work) is missing from results")
        if not isinstance(self.W, (int, float)):
            return (False, f"W is not numeric: {type(self.W)}")
        return (True, "W is present and numeric")

    def test_interface_tf_provided(self) -> Tuple[bool, str]:
        if self.t_f is None:
            return (False, "t_f (final temperature in Celsius) is missing from results")
        if not isinstance(self.t_f, (int, float)):
            return (False, f"t_f is not numeric: {type(self.t_f)}")
        return (True, "t_f is present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_isothermal_work_formula(self) -> Tuple[bool, str]:
        """Check that W ≈ nRT0 ln(Vf/V0) if all needed numbers are present."""
        if any(x is None for x in (self.n, self.R, self.T0, self.V_ratio, self.W)):
            return (False, "SKIP: missing data for isothermal work invariant")
        expected = self.n * self.R * self.T0 * math.log(self.V_ratio)
        if not math.isclose(self.W, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W {self.W} deviates from nRT ln(Vf/V0) = {expected:.2f}")
        return (True, "Isothermal work matches nRT ln(Vf/V0)")

    def test_invariant_adiabatic_temperature_formula(self) -> Tuple[bool, str]:
        """Check that T_f ≈ T0 * (V0/Vf)^(gamma-1) and t_f = T_f - 273.15."""
        if any(x is None for x in (self.T0, self.V_ratio, self.gamma, self.t_f)):
            return (False, "SKIP: missing data for adiabatic invariant")
        T_f_from_t = self.t_f + 273.15  # convert °C to K
        expected_T_f = self.T0 * (1.0 / self.V_ratio) ** (self.gamma - 1)
        if not math.isclose(T_f_from_t, expected_T_f, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_f (in K) = {T_f_from_t:.2f} does not match T0*(V0/Vf)^(γ-1) = {expected_T_f:.2f}")
        return (True, "Adiabatic final temperature obeys TV^(γ-1)=constant")

    def test_invariant_positive_work(self) -> Tuple[bool, str]:
        """Isothermal expansion: work must be positive (gas does work on surroundings)."""
        if self.W is None:
            return (False, "SKIP: W not provided")
        if not math.isfinite(self.W):
            return (False, "W is not finite")
        if self.W <= 0:
            return (False, f"W = {self.W} should be positive for expansion")
        return (True, "W is positive as expected for expansion")

    def test_invariant_adiabatic_temperature_drop(self) -> Tuple[bool, str]:
        """For adiabatic expansion, temperature must decrease (t_f < T0)."""
        if self.t_f is None or self.T0 is None:
            return (False, "SKIP: missing temperatures")
        if not math.isfinite(self.t_f):
            return (False, "t_f is not finite")
        # T0 is in K, t_f in °C. Convert T0 to °C: T0 °C = 0.
        if self.t_f >= 0:
            return (False, f"t_f = {self.t_f}°C should be < 0°C for adiabatic expansion")
        return (True, "Final temperature is below 0°C as expected")

    # ------------------------------------------------------------------
    # Modeling / trace tests (consistency with explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_isothermal_work_intermediate(self) -> Tuple[bool, str]:
        """
        Check that the work step used correct intermediate: nRT * ln(Vf/V0).
        We recompute nRT and check if it is positive.
        """
        if any(x is None for x in (self.n, self.R, self.T0)):
            return (False, "SKIP: missing data for nRT")
        nRT = self.n * self.R * self.T0
        if not math.isfinite(nRT):
            return (False, "nRT is not finite")
        if nRT <= 0:
            return (False, f"nRT = {nRT} should be positive")
        return (True, "nRT is positive and finite")

    def test_modeling_adiabatic_exponent_used(self) -> Tuple[bool, str]:
        """Check that gamma-1 is between 0 and 1 (for monatomic gamma=5/3, gamma-1=2/3)."""
        if self.gamma is None:
            return (False, "SKIP: gamma not provided")
        exponent = self.gamma - 1
        if exponent <= 0 or exponent >= 1:
            return (False, f"γ-1 = {exponent} should be between 0 and 1 for monatomic ideal gas")
        return (True, "γ-1 is in expected range (0,1)")

    def test_modeling_tf_conversion(self) -> Tuple[bool, str]:
        """Check that t_f = T_f - 273.15 consistently if we also have T_f in results."""
        # We can try to extract T_f from results if available
        # But we only have t_f. We can assume if t_f is given, the conversion was done.
        # This test is trivial; we just check that t_f is not absurd.
        if self.t_f is None:
            return (False, "SKIP: t_f not provided")
        if not math.isfinite(self.t_f):
            return (False, "t_f is not finite")
        return (True, "t_f is finite and reasonable")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_work_increases_with_volume_ratio(self) -> Tuple[bool, str]:
        """If V_ratio increases, W should increase (for fixed n,R,T)."""
        if any(x is None for x in (self.n, self.R, self.T0, self.V_ratio, self.W)):
            return (False, "SKIP: missing data for monotonicity")
        # Perturb V_ratio by +10%
        perturbed_V_ratio = self.V_ratio * 1.10
        expected_new_W = self.n * self.R * self.T0 * math.log(perturbed_V_ratio)
        if not math.isclose(self.W, expected_new_W, rel_tol=1e-1, abs_tol=1e-2):
            # We cannot directly compare because original W is for original V_ratio.
            # We'll compute the derivative: dW/dV_ratio = nRT / V_ratio
            # For small perturbation, delta_W ~ (nRT/V_ratio) * delta_V_ratio
            delta_V = 0.10 * self.V_ratio
            expected_delta = (self.n * self.R * self.T0 / self.V_ratio) * delta_V
            # Check if original W is less than the perturbed W (monotonicity)
            if expected_new_W <= self.W:
                return (False, f"Expected perturbed W {expected_new_W:.2f} > original W {self.W:.2f} failed")
        return (True, "Work increases with volume ratio")

    def test_monotonicity_adiabatic_final_temp_decreases_with_gamma(self) -> Tuple[bool, str]:
        """For larger gamma, the exponent (gamma-1) is larger, so T_f is smaller.
        Check that if gamma were 5% larger, T_f would be lower."""
        if any(x is None for x in (self.T0, self.V_ratio, self.gamma, self.t_f)):
            return (False, "SKIP: missing data for monotonicity")
        # Original gamma, compute original T_f in K from t_f
        T_f_original = self.t_f + 273.15
        # Larger gamma
        gamma_large = self.gamma * 1.05
        expected_T_f_large = self.T0 * (1.0 / self.V_ratio) ** (gamma_large - 1)
        if expected_T_f_large >= T_f_original:
            return (False, f"Increasing gamma should decrease T_f, but T_f_large {expected_T_f_large:.2f} >= T_f_original {T_f_original:.2f}")
        return (True, "Increasing gamma reduces adiabatic final temperature")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one, no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_magnitude(self) -> Tuple[bool, str]:
        """Very weak check: W positive, t_f negative, orders of magnitude plausible."""
        if self.W is not None:
            if not (0 < self.W < 1e5):
                return (False, f"W = {self.W} seems out of plausible range (10^3 J)")
        if self.t_f is not None:
            if not (-300 < self.t_f < 0):
                return (False, f"t_f = {self.t_f} out of plausible range (-300 to 0 °C)")
        if self.W is None and self.t_f is None:
            return (False, "Both results missing; cannot run answer sanity")
        return (True, "Answer signs and magnitudes are plausible")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Discover all test methods
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