import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits solutions for the coupled harmonic oscillator problem:
    H = p1^2/(2m) + p2^2/(2m) + (1/2)k(x1^2+x2^2) + ε x1 x2.
    Checks normal mode frequencies, energy levels, and consistency relations.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens safely from solution_instance attributes or results dict
        self.m = getattr(solution_instance, 'm', None)
        if self.m is None:
            self.m = solution_instance.results.get('m', None) if hasattr(solution_instance, 'results') else None
        self.k = getattr(solution_instance, 'k', None)
        if self.k is None:
            self.k = solution_instance.results.get('k', None) if hasattr(solution_instance, 'results') else None
        self.eps = getattr(solution_instance, 'eps', None)
        if self.eps is None:
            self.eps = solution_instance.results.get('eps', None) if hasattr(solution_instance, 'results') else None
        self.hbar = getattr(solution_instance, 'hbar', None)
        if self.hbar is None:
            self.hbar = solution_instance.results.get('hbar', None) if hasattr(solution_instance, 'results') else None
        self.n_plus = getattr(solution_instance, 'n_plus', None)
        if self.n_plus is None:
            self.n_plus = solution_instance.results.get('n_plus', None) if hasattr(solution_instance, 'results') else None
        self.n_minus = getattr(solution_instance, 'n_minus', None)
        if self.n_minus is None:
            self.n_minus = solution_instance.results.get('n_minus', None) if hasattr(solution_instance, 'results') else None

        # Extract result quantities from results dictionary
        results = {}
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        self.freq_plus = results.get('freq_plus', None)
        self.freq_minus = results.get('freq_minus', None)
        self.E_approx = results.get('E_approx', None)

        # Derived quantity (only used inside tests, not copied from explanation)
        self.omega = None
        if self.k is not None and self.m is not None:
            self.omega = math.sqrt(self.k / self.m) if self.m > 0 else None

    # ----------------------------------------------------------------------
    # Interface tests (required keys exist and have valid types)
    # ----------------------------------------------------------------------
    def test_interface_freq_plus_exists(self) -> Tuple[bool, str]:
        if self.freq_plus is None:
            return (False, "Missing required result: freq_plus")
        if not isinstance(self.freq_plus, (int, float)):
            return (False, f"freq_plus has invalid type: {type(self.freq_plus)}")
        if not math.isfinite(self.freq_plus):
            return (False, f"freq_plus is not finite: {self.freq_plus}")
        return (True, "freq_plus present and finite")

    def test_interface_freq_minus_exists(self) -> Tuple[bool, str]:
        if self.freq_minus is None:
            return (False, "Missing required result: freq_minus")
        if not isinstance(self.freq_minus, (int, float)):
            return (False, f"freq_minus has invalid type: {type(self.freq_minus)}")
        if not math.isfinite(self.freq_minus):
            return (False, f"freq_minus is not finite: {self.freq_minus}")
        return (True, "freq_minus present and finite")

    def test_interface_E_approx_exists(self) -> Tuple[bool, str]:
        if self.E_approx is None:
            return (False, "Missing required result: E_approx")
        if not isinstance(self.E_approx, (int, float)):
            return (False, f"E_approx has invalid type: {type(self.E_approx)}")
        if not math.isfinite(self.E_approx):
            return (False, f"E_approx is not finite: {self.E_approx}")
        return (True, "E_approx present and finite")

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        missing = []
        for name in ['m', 'k', 'eps', 'hbar', 'n_plus', 'n_minus']:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_freq_product(self) -> Tuple[bool, str]:
        if any(v is None for v in [self.freq_plus, self.freq_minus, self.k, self.m, self.eps]):
            return (False, "SKIP: insufficient data for freq product invariant")
        # product should be sqrt((k^2 - eps^2)/m^2) (since both frequencies positive)
        expected_product = math.sqrt((self.k**2 - self.eps**2) / (self.m**2))
        actual_product = self.freq_plus * self.freq_minus
        if math.isclose(actual_product, expected_product, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"freq_plus * freq_minus agrees: {actual_product:.4f} vs expected {expected_product:.4f}")
        else:
            return (False, f"freq_plus * freq_minus mismatch: {actual_product:.4f} vs expected {expected_product:.4f}")

    def test_invariant_energy_positivity(self) -> Tuple[bool, str]:
        if self.E_approx is None:
            return (False, "SKIP: E_approx not available")
        if self.E_approx > 0:
            return (True, f"E_approx is positive: {self.E_approx}")
        else:
            return (False, f"E_approx should be positive: {self.E_approx}")

    # ----------------------------------------------------------------------
    # Modeling/trace tests (check consistency with physics laws)
    # ----------------------------------------------------------------------
    def test_modeling_freq_plus_formula(self) -> Tuple[bool, str]:
        if any(v is None for v in [self.freq_plus, self.k, self.m, self.eps]):
            return (False, "SKIP: missing data for freq_plus formula check")
        expected = math.sqrt((self.k + self.eps) / self.m)
        if math.isclose(self.freq_plus, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"freq_plus = sqrt((k+eps)/m): {self.freq_plus:.4f} ≈ {expected:.4f}")
        else:
            return (False, f"freq_plus = {self.freq_plus:.4f} but expected sqrt((k+eps)/m) = {expected:.4f}")

    def test_modeling_freq_minus_formula(self) -> Tuple[bool, str]:
        if any(v is None for v in [self.freq_minus, self.k, self.m, self.eps]):
            return (False, "SKIP: missing data for freq_minus formula check")
        expected = math.sqrt((self.k - self.eps) / self.m)
        if math.isclose(self.freq_minus, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"freq_minus = sqrt((k-eps)/m): {self.freq_minus:.4f} ≈ {expected:.4f}")
        else:
            return (False, f"freq_minus = {self.freq_minus:.4f} but expected sqrt((k-eps)/m) = {expected:.4f}")

    def test_modeling_E_approx_from_expansion(self) -> Tuple[bool, str]:
        if any(v is None for v in [self.E_approx, self.omega, self.hbar, self.n_plus, self.n_minus, self.eps, self.k]):
            return (False, "SKIP: missing data for E_approx expansion check")
        if self.omega is None or self.omega <= 0:
            return (False, "SKIP: omega not valid")
        # Re-derive E_approx from first-order expansion (inside the checker)
        derived = (self.hbar * self.omega * (self.n_plus + self.n_minus + 1.0) +
                   self.hbar * self.omega * (self.n_plus - self.n_minus) * self.eps / (2.0 * self.k))
        if math.isclose(self.E_approx, derived, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_approx matches expansion: {self.E_approx:.4f} ≈ {derived:.4f}")
        else:
            return (False, f"E_approx = {self.E_approx:.4f} but derived from expansion = {derived:.4f}")

    # ----------------------------------------------------------------------
    # Monotonicity tests (trend checks via perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_energy_with_n_plus(self) -> Tuple[bool, str]:
        if any(v is None for v in [self.E_approx, self.hbar, self.omega, self.n_plus, self.n_minus, self.eps, self.k]):
            return (False, "SKIP: missing data for monotonicity test")
        if self.omega is None or self.omega <= 0:
            return (False, "SKIP: omega not valid")
        # Perturb n_plus upward by 1, other parameters same
        E_base = self.E_approx
        E_perturbed = (self.hbar * self.omega * ((self.n_plus+1) + self.n_minus + 1.0) +
                       self.hbar * self.omega * ((self.n_plus+1) - self.n_minus) * self.eps / (2.0 * self.k))
        if E_perturbed > E_base:
            return (True, f"Increasing n_plus increases E_approx: base={E_base:.4f}, perturbed={E_perturbed:.4f}")
        else:
            return (False, f"E_approx should increase with n_plus: base={E_base:.4f}, perturbed={E_perturbed:.4f}")

    def test_monotonicity_energy_with_n_minus(self) -> Tuple[bool, str]:
        if any(v is None for v in [self.E_approx, self.hbar, self.omega, self.n_plus, self.n_minus, self.eps, self.k]):
            return (False, "SKIP: missing data for monotonicity test")
        if self.omega is None or self.omega <= 0:
            return (False, "SKIP: omega not valid")
        E_base = self.E_approx
        E_perturbed = (self.hbar * self.omega * (self.n_plus + (self.n_minus+1) + 1.0) +
                       self.hbar * self.omega * (self.n_plus - (self.n_minus+1)) * self.eps / (2.0 * self.k))
        if E_perturbed > E_base:
            return (True, f"Increasing n_minus increases E_approx: base={E_base:.4f}, perturbed={E_perturbed:.4f}")
        else:
            return (False, f"E_approx should increase with n_minus: base={E_base:.4f}, perturbed={E_perturbed:.4f}")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (only one, no math.isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_E_approx_finite(self) -> Tuple[bool, str]:
        if self.E_approx is None:
            return (False, "SKIP: E_approx not available")
        if math.isfinite(self.E_approx):
            return (True, f"E_approx is finite: {self.E_approx}")
        else:
            return (False, f"E_approx is not finite: {self.E_approx}")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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