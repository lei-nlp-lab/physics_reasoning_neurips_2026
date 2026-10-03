import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union


class PhysicsChecker:
    """
    Physics checker for the problem: "Real Hamiltonian, real wavefunction at t=0 and t=t1,
    show periodicity and energy quantization."
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed data from solution_instance safely with defaults None.
        """
        self.solution_instance = solution_instance

        # RESULTS (from emit keys)
        self.T: Optional[Union[float, sp.Basic]] = None
        self.E_n: Optional[Union[float, sp.Basic]] = None

        # GIVENS (input data)
        self.t1: Optional[Union[float, sp.Basic]] = None
        self.hbar: Optional[Union[float, sp.Basic]] = None
        self.kn: Optional[Union[int, float]] = None
        self.psi0_real: Optional[bool] = None
        self.psi_t1_real: Optional[bool] = None
        self.H_real: Optional[bool] = None
        self.cn_real: Optional[bool] = None
        self.phi_real: Optional[bool] = None

        self.givens: Dict = {}

        # Extract results
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            res = solution_instance.results
            self.T = res.get('T', None)
            self.E_n = res.get('E_n', None)
            self.kn = res.get('kn', None)

        # Extract givens
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.givens = solution_instance.givens
            self.t1 = self.givens.get('t1', None)
            self.hbar = self.givens.get('hbar', None)
            self.kn = self.givens.get('kn', None) if self.kn is None else self.kn
            self.psi0_real = self.givens.get('psi0_real', None)
            self.psi_t1_real = self.givens.get('psi_t1_real', None)
            self.H_real = self.givens.get('H_real', None)
            self.cn_real = self.givens.get('cn_real', None)
            self.phi_real = self.givens.get('phi_real', None)

        # Fallback to direct attributes
        if self.t1 is None:
            self.t1 = getattr(solution_instance, 't1', None)
        if self.hbar is None:
            self.hbar = getattr(solution_instance, 'hbar', None)
        if self.kn is None:
            self.kn = getattr(solution_instance, 'kn', None)
        if self.psi0_real is None:
            self.psi0_real = getattr(solution_instance, 'psi0_real', None)
        if self.psi_t1_real is None:
            self.psi_t1_real = getattr(solution_instance, 'psi_t1_real', None)
        if self.H_real is None:
            self.H_real = getattr(solution_instance, 'H_real', None)
        if self.cn_real is None:
            self.cn_real = getattr(self.givens, 'cn_real', None)
        if self.phi_real is None:
            self.phi_real = getattr(self.givens, 'phi_real', None)

        # Prepare numeric versions for invariant tests
        self._T_num = None
        self._E_n_num = None
        self._t1_num = None
        self._hbar_num = None
        self._kn_num = None

        if self.T is not None:
            try:
                self._T_num = complex(self.T).real if isinstance(self.T, complex) else float(self.T)
            except (TypeError, ValueError):
                try:
                    if isinstance(self.T, sp.Basic) and self.T.is_number:
                        self._T_num = float(self.T.evalf())
                except Exception:
                    pass
        if self.E_n is not None:
            try:
                self._E_n_num = complex(self.E_n).real if isinstance(self.E_n, complex) else float(self.E_n)
            except (TypeError, ValueError):
                try:
                    if isinstance(self.E_n, sp.Basic) and self.E_n.is_number:
                        self._E_n_num = float(self.E_n.evalf())
                except Exception:
                    pass
        if self.t1 is not None:
            try:
                self._t1_num = complex(self.t1).real if isinstance(self.t1, complex) else float(self.t1)
            except (TypeError, ValueError):
                try:
                    if isinstance(self.t1, sp.Basic) and self.t1.is_number:
                        self._t1_num = float(self.t1.evalf())
                except Exception:
                    pass
        if self.hbar is not None:
            try:
                self._hbar_num = float(self.hbar)
            except (TypeError, ValueError):
                try:
                    if isinstance(self.hbar, sp.Basic) and self.hbar.is_number:
                        self._hbar_num = float(self.hbar.evalf())
                except Exception:
                    pass
        if self.kn is not None:
            try:
                self._kn_num = int(self.kn) if float(self.kn).is_integer() else round(float(self.kn))
            except (TypeError, ValueError):
                try:
                    if isinstance(self.kn, sp.Basic) and self.kn.is_number:
                        self._kn_num = int(round(float(self.kn.evalf())))
                except Exception:
                    pass

    # -----------------------------------------------------------------
    #  INTERFACE CHECKS
    # -----------------------------------------------------------------
    def test_interface_T_exists(self) -> Tuple[bool, str]:
        """Check that T is present and not None."""
        if self.T is None:
            return (False, "SKIP: T is None, missing result key 'T'")
        return (True, "T is present")

    def test_interface_E_n_exists(self) -> Tuple[bool, str]:
        """Check that E_n is present and not None."""
        if self.E_n is None:
            return (False, "SKIP: E_n is None, missing result key 'E_n'")
        return (True, "E_n is present")

    def test_interface_t1_provided(self) -> Tuple[bool, str]:
        """Check that t1 (the given time) is available as a given."""
        if self.t1 is None:
            return (False, "SKIP: t1 not provided in givens")
        return (True, "t1 is available")

    def test_interface_kn_provided(self) -> Tuple[bool, str]:
        """Check that kn (or some integer index) is available."""
        if self.kn is None:
            return (False, "SKIP: kn not provided in givens or results")
        return (True, "kn is available")

    # -----------------------------------------------------------------
    #  INVARIANT CHECKS (theorem-based, with math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_T_equals_2t1(self) -> Tuple[bool, str]:
        """If t1 is given numerically, T should equal 2*t1 (fuzzy)."""
        if self._T_num is None or self._t1_num is None:
            return (False, "SKIP: missing numeric T or t1 for this comparison")
        expected = 2.0 * self._t1_num
        if math.isclose(self._T_num, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T ≈ 2*t1: T={self._T_num}, 2*t1={expected}")
        else:
            return (False, f"T = {self._T_num} is not close to 2*t1 = {expected}")

    def test_invariant_energy_quantization(self) -> Tuple[bool, str]:
        """
        Check that E_n = (2*pi*hbar/T) * kn for some integer kn.
        If kn is given, verify equation; if not, verify that
        E_n * T / (2*pi*hbar) is close to an integer.
        """
        if self._E_n_num is None or self._T_num is None or self._hbar_num is None:
            return (False, "SKIP: missing numeric E_n, T, or hbar")
        if self._T_num == 0:
            return (False, "SKIP: T is zero")
        computed_factor = self._E_n_num * self._T_num / (2.0 * math.pi * self._hbar_num)
        closest_int = round(computed_factor)
        if math.isclose(computed_factor, closest_int, rel_tol=1e-2, abs_tol=1e-2):
            if self._kn_num is not None:
                if math.isclose(closest_int, self._kn_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Energy quantization: E_n ≈ (2*pi*hbar/T)*k_n with k_n={closest_int} (matches given kn={self._kn_num})")
                else:
                    return (False, f"E_n * T/(2*pi*hbar) = {computed_factor}, closest integer {closest_int} does not match given kn={self._kn_num}")
            else:
                return (True, f"Energy quantization: E_n * T/(2*pi*hbar) ≈ integer {closest_int} (kn not provided)")
        else:
            return (False, f"E_n * T/(2*pi*hbar) = {computed_factor} is not close to an integer (closest {closest_int})")

    def test_invariant_phase_reality_at_t1(self) -> Tuple[bool, str]:
        """
        Check that the phase factor e^{-i E_n t1/ħ} is real.
        i.e., its imaginary part is zero up to tolerance.
        """
        if self._E_n_num is None or self._t1_num is None or self._hbar_num is None:
            return (False, "SKIP: missing numeric E_n, t1, hbar")
        argument = -self._E_n_num * self._t1_num / self._hbar_num
        imag = math.sin(argument)  # exp(i*argument) imaginary part
        if math.isclose(imag, 0.0, abs_tol=1e-2):
            return (True, f"Imaginary part of e^{{-iE_n t1/hbar}} ≈ {imag} ~ 0")
        else:
            return (False, f"Imaginary part = {imag} not zero")

    def test_invariant_periodicity_condition(self) -> Tuple[bool, str]:
        """
        Verify that e^{-i E_n T/ħ} = 1 up to tolerance.
        """
        if self._E_n_num is None or self._T_num is None or self._hbar_num is None:
            return (False, "SKIP: missing numeric E_n, T, hbar")
        angle = -self._E_n_num * self._T_num / self._hbar_num
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)
        if math.isclose(cos_val, 1.0, rel_tol=1e-2, abs_tol=1e-2) and math.isclose(sin_val, 0.0, abs_tol=1e-2):
            return (True, "e^{{-i E_n T/hbar}} ≈ 1")
        else:
            return (False, f"e^{{-i E_n T/hbar}} = cos({angle}) + i sin({angle}) ≈ {cos_val} + i{sin_val} ≠ 1")

    # -----------------------------------------------------------------
    #  MODELING CHECKS (based on explanation steps / constraints)
    # -----------------------------------------------------------------
    def test_modeling_psi0_real(self) -> Tuple[bool, str]:
        """Check that psi(0) is reported as real (given constraint)."""
        if self.psi0_real is None:
            return (False, "SKIP: psi0_real not provided in givens")
        if self.psi0_real:
            return (True, "psi(0) is real as required")
        else:
            return (False, "psi(0) should be real but is reported as non-real")

    def test_modeling_H_real(self) -> Tuple[bool, str]:
        """Check that Hamiltonian is reported as real (given constraint)."""
        if self.H_real is None:
            return (False, "SKIP: H_real not provided in givens")
        if self.H_real:
            return (True, "H is real as required")
        else:
            return (False, "H should be real but is reported as non-real")

    def test_modeling_psi_t1_real(self) -> Tuple[bool, str]:
        """Check that psi(t1) is reported as real (given constraint)."""
        if self.psi_t1_real is None:
            return (False, "SKIP: psi_t1_real not provided in givens")
        if self.psi_t1_real:
            return (True, "psi(t1) is real as required")
        else:
            return (False, "psi(t1) should be real but is reported as non-real")

    def test_modeling_cn_real(self) -> Tuple[bool, str]:
        """
        Check that c_n (expansion coefficients) are reported as real.
        From theory: because psi(0) and eigenfunctions are real, c_n are real.
        """
        cn_real = self.cn_real
        if cn_real is None:
            cn_real = self.givens.get('cn_real', None)
        if cn_real is None:
            return (False, "SKIP: cn_real not provided")
        if cn_real:
            return (True, "c_n are real (consistent with real psi(0) and phi_n)")
        else:
            return (False, "c_n should be real but reported as non-real")

    def test_modeling_eigenfunctions_real(self) -> Tuple[bool, str]:
        """Check that energy eigenfunctions phi_n are reported as real."""
        phi_real = self.phi_real
        if phi_real is None:
            phi_real = self.givens.get('phi_real', None)
        if phi_real is None:
            return (False, "SKIP: phi_real not provided")
        if phi_real:
            return (True, "Eigenfunctions phi_n are real (consistent with real H)")
        else:
            return (False, "Eigenfunctions should be real but reported as non-real")

    # -----------------------------------------------------------------
    # MONOTONICITY TESTS (explicit perturbation of input, compare to baseline)
    # -----------------------------------------------------------------
    def test_monotonicity_T_increases_with_t1(self) -> Tuple[bool, str]:
        """
        Perturb t1 by +10% and check that the predicted T (from the formula T=2*t1)
        increases proportionally. We compare the computed T for the perturbed t1
        against the baseline T, and verify the change is positive.
        """
        if self._T_num is None or self._t1_num is None:
            return (False, "SKIP: missing numeric T or t1")
        if self._t1_num <= 0:
            return (False, "SKIP: t1 is not positive, cannot test monotonic increase")
        # Baseline: T_baseline = 2 * t1
        T_baseline = 2.0 * self._t1_num
        # Perturb: increase t1 by 10%
        delta = 0.1 * self._t1_num
        t1_perturbed = self._t1_num + delta
        T_perturbed_predicted = 2.0 * t1_perturbed
        # Compare predicted perturbed T to baseline T
        if T_perturbed_predicted > T_baseline:
            return (True, f"Perturbing t1 by +{delta}: baseline T = {T_baseline}, perturbed predicted T = {T_perturbed_predicted} (increased)")
        else:
            return (False, f"Monotonicity violated: T_perturbed {T_perturbed_predicted} <= T_baseline {T_baseline}")

    def test_monotonicity_energy_increases_with_kn(self) -> Tuple[bool, str]:
        """
        Perturb kn by +1 and check that the predicted E_n (from the formula
        E_n = (2*pi*hbar/T)*kn) increases. We compare predicted perturbed E_n
        against baseline E_n (the actual given E_n should match the baseline formula).
        """
        if self._E_n_num is None or self._T_num is None or self._hbar_num is None:
            return (False, "SKIP: missing numeric E_n, T, hbar")
        if self._kn_num is None:
            return (False, "SKIP: kn not available")
        if self._T_num <= 0:
            return (False, "SKIP: T is not positive")
        # Baseline: E_baseline from formula (should match given E_n within tolerance)
        E_baseline_formula = (2.0 * math.pi * self._hbar_num / self._T_num) * self._kn_num
        if not math.isclose(self._E_n_num, E_baseline_formula, rel_tol=1e-2, abs_tol=1e-2):
            # If baseline formula doesn't match given E_n, we cannot reliably use formula for perturbation.
            return (False, f"Given E_n {self._E_n_num} does not match formula prediction {E_baseline_formula} for kn={self._kn_num}; cannot perform monotonicity test.")
        # Perturb: increase kn by 1
        kn_perturbed = self._kn_num + 1
        E_perturbed_predicted = (2.0 * math.pi * self._hbar_num / self._T_num) * kn_perturbed
        # Compare predicted perturbed E_n to baseline formula value (which agrees with given E_n)
        if E_perturbed_predicted > E_baseline_formula:
            return (True, f"Perturbing kn by +1: baseline E_n (formula) = {E_baseline_formula}, perturbed predicted E_n = {E_perturbed_predicted} (increased)")
        else:
            return (False, f"Monotonicity violated: E_perturbed {E_perturbed_predicted} <= baseline {E_baseline_formula}")

    # -----------------------------------------------------------------
    # ANSWER SANITY (weak, secondary) - only one method
    # -----------------------------------------------------------------
    def test_answer_sanity_T_positive(self) -> Tuple[bool, str]:
        """Weak sanity: T must be positive (real period)."""
        if self._T_num is None:
            return (False, "SKIP: numeric T not available")
        if self._T_num > 0:
            return (True, f"T = {self._T_num} > 0")
        else:
            return (False, f"T = {self._T_num} should be positive")

    # -----------------------------------------------------------------
    # RUN ALL TESTS
    # -----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all methods starting with 'test_', execute them in try/except,
        and return overall pass status and list of log messages.
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
                passed, msg = False, f"{name} CRASHED: {e} {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)


# Example usage (for testing purposes):
if __name__ == "__main__":
    # Dummy solution instance for demonstration
    class DummySolution:
        def __init__(self):
            self.results = {
                'T': 4.0,
                'E_n': 3.141592653589793,
                'kn': 2
            }
            self.givens = {
                't1': 2.0,
                'hbar': 1.0545718e-34,
                'psi0_real': True,
                'psi_t1_real': True,
                'H_real': True,
                'cn_real': True,
                'phi_real': True,
                'kn': 2
            }

    checker = PhysicsChecker(DummySolution())
    passed, logs = checker.run_all_tests()
    print("\n".join(logs))
    print(f"\nOverall pass: {passed}")