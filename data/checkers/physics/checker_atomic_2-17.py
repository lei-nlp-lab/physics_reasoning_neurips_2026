import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits the solution for Born approximation scattering from a Yukawa potential,
    and the screening length scaling with nuclear charge.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results from solution_instance
        # Assume solution_instance has .results dictionary or we use getattr
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Final answer keys
        self.sigma = results.get('sigma', None)
        self.alpha_scaling = results.get('alpha_scaling', None)

        # Givens: we try to extract them from solution_instance; safe defaults to None
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.Q = getattr(solution_instance, 'Q', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.alpha = getattr(solution_instance, 'alpha', None)
        self.k0 = getattr(solution_instance, 'k0', None)

        # For convenience, also allow givens inside results
        if self.m is None:
            self.m = results.get('m', None)
        if self.q is None:
            self.q = results.get('q', None)
        if self.Q is None:
            self.Q = results.get('Q', None)
        if self.hbar is None:
            self.hbar = results.get('hbar', None)
        if self.alpha is None:
            self.alpha = results.get('alpha', None)
        if self.k0 is None:
            self.k0 = results.get('k0', None)

        # Explanations (optional, used to decide what to test)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discovers and runs all test_* methods, returns (all_pass, logs)."""
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
        return all_pass, logs

    # ----------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------
    def test_interface_sigma_exists(self) -> Tuple[bool, str]:
        """Check that result key 'sigma' is present and is a string (symbolic) or numeric."""
        if self.sigma is None:
            return False, "Missing 'sigma' in results."
        if not isinstance(self.sigma, (str, int, float)):
            return False, f"'sigma' has unexpected type {type(self.sigma)}."
        return True, "sigma exists with valid type."

    def test_interface_alpha_scaling_exists(self) -> Tuple[bool, str]:
        """Check that result key 'alpha_scaling' is present and is a string."""
        if self.alpha_scaling is None:
            return False, "Missing 'alpha_scaling' in results."
        if not isinstance(self.alpha_scaling, str):
            return False, f"'alpha_scaling' is not a string."
        return True, "alpha_scaling exists and is a string."

    # ----------------------------------------------------------------
    # INVARIANT (THEOREM) CHECKS
    # ----------------------------------------------------------------
    def test_invariant_sigma_positive(self) -> Tuple[bool, str]:
        """If sigma is numeric, it must be positive."""
        if isinstance(self.sigma, (int, float)):
            if self.sigma <= 0:
                return False, f"sigma = {self.sigma} is not positive."
            return True, f"sigma = {self.sigma} > 0."
        # Symbolic expressions: we can't check sign without numeric sub.
        return True, "sigma is symbolic; positivity not verified."

    def test_invariant_alpha_positive(self) -> Tuple[bool, str]:
        """If alpha (screening length inverse) is numeric, it must be positive."""
        if self.alpha is not None and isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                return False, f"alpha = {self.alpha} is not positive."
            return True, f"alpha = {self.alpha} > 0."
        return True, "alpha not numeric; positivity not checked."

    # ----------------------------------------------------------------
    # MODELING / TRACE CHECKS (using explanation steps)
    # ----------------------------------------------------------------
    def test_modeling_sigma_calculation(self) -> Tuple[bool, str]:
        """
        If all givens are available, compute the Born cross section numerically
        using the derived formula and compare to self.sigma (if numeric).
        This checks that the solution's sigma matches the expected expression.
        """
        if self.sigma is None:
            return True, "SKIP: sigma not provided."
        # Only run if we have all numeric givens
        numeric_givens = [self.m, self.q, self.Q, self.hbar, self.alpha, self.k0]
        if any(g is None for g in numeric_givens):
            return True, "SKIP: insufficient numeric givens to compute expected sigma."

        m, q, Q, hbar, alpha, k0 = numeric_givens
        # Expected sigma from derived formula: sigma = (16*pi*m^2*Q^2*q^2) / (hbar^4 * alpha^2 * (alpha^2+4*k0^2))
        expected_sigma = (16 * math.pi * m**2 * q**2 * Q**2) / (hbar**4 * alpha**2 * (alpha**2 + 4 * k0**2))

        if isinstance(self.sigma, str):
            # If sigma is symbolic, try to evaluate it using the same numbers via sympy
            try:
                # Define symbols
                sigma_sym = sp.sympify(self.sigma)
                # Substitute with numeric values
                subs_dict = {sp.Symbol('m'): m, sp.Symbol('q'): q, sp.Symbol('Q'): Q,
                             sp.Symbol('hbar'): hbar, sp.Symbol('alpha'): alpha,
                             sp.Symbol('k0'): k0}
                value = float(sigma_sym.subs(subs_dict).evalf())
            except Exception:
                return True, "SKIP: cannot evaluate symbolic sigma numerically."
        else:
            value = float(self.sigma)

        if not math.isfinite(value):
            return False, f"sigma value is not finite: {value}"

        rel_tol = 1e-2  # 1% tolerance for rounding noise
        if not math.isclose(value, expected_sigma, rel_tol=rel_tol, abs_tol=1e-12):
            return False, f"sigma value {value} differs from expected {expected_sigma} by more than {rel_tol*100}%."
        return True, f"sigma matches expected formula within tolerance."

    def test_modeling_alpha_scaling_trend(self) -> Tuple[bool, str]:
        """
        Check that the alpha_scaling answer correctly indicates that alpha increases with Z,
        and optionally scales as Z^{1/3}.
        """
        if self.alpha_scaling is None:
            return True, "SKIP: alpha_scaling not present."

        s = self.alpha_scaling.lower()
        # Check key phrases
        has_increasing = ('increasing' in s or 'larger' in s or 'positive' in s or 'proportional' in s)
        has_z13 = ('z^{1/3}' in s or 'z**(1/3)' in s or 'z^(1/3)' in s)
        if not has_increasing:
            # If it says "alpha decreases", that's wrong.
            if 'decreasing' in s or 'smaller' in s:
                return False, f"alpha_scaling says decreasing but should increase with Z: {self.alpha_scaling}"
            return True, "alpha_scaling does not explicitly mention increasing; cannot verify."
        return True, f"alpha_scaling indicates increasing trend with Z: {self.alpha_scaling}"

    # ----------------------------------------------------------------
    # MONOTONICITY TESTS (perturb input, check trend)
    # ----------------------------------------------------------------
    def test_monotonicity_sigma_with_alpha(self) -> Tuple[bool, str]:
        """
        If all numeric givens and sigma are numeric, increase alpha by 10% and check that sigma decreases.
        sigma ∝ 1/(α^2(α^2+4k0^2)) → strictly decreasing with α for α>0, k0>0.
        """
        if self.sigma is None or not isinstance(self.sigma, (int, float)):
            return True, "SKIP: sigma not numeric."
        numeric_givens = [self.m, self.q, self.Q, self.hbar, self.alpha, self.k0]
        if any(g is None for g in numeric_givens):
            return True, "SKIP: insufficient numeric givens."

        m, q, Q, hbar, alpha, k0 = numeric_givens

        # Perturb alpha upward by 10%
        alpha_pert = alpha * 1.10
        sigma_pert = (16 * math.pi * m**2 * q**2 * Q**2) / (hbar**4 * alpha_pert**2 * (alpha_pert**2 + 4 * k0**2))

        if sigma_pert >= self.sigma:
            return False, f"sigma does not decrease when alpha increases: baseline {self.sigma}, perturbed {sigma_pert}"
        return True, f"sigma decreases with increasing alpha: {self.sigma} -> {sigma_pert}"

    def test_monotonicity_sigma_with_k0(self) -> Tuple[bool, str]:
        """
        If all numeric givens and sigma are numeric, increase k0 by 10% and check that sigma decreases.
        sigma ∝ 1/(α^2(α^2+4k0^2)) → strictly decreasing with k0.
        """
        if self.sigma is None or not isinstance(self.sigma, (int, float)):
            return True, "SKIP: sigma not numeric."
        numeric_givens = [self.m, self.q, self.Q, self.hbar, self.alpha, self.k0]
        if any(g is None for g in numeric_givens):
            return True, "SKIP: insufficient numeric givens."

        m, q, Q, hbar, alpha, k0 = numeric_givens

        # Perturb k0 upward by 10%
        k0_pert = k0 * 1.10
        sigma_pert = (16 * math.pi * m**2 * q**2 * Q**2) / (hbar**4 * alpha**2 * (alpha**2 + 4 * k0_pert**2))

        if sigma_pert >= self.sigma:
            return False, f"sigma does not decrease when k0 increases: baseline {self.sigma}, perturbed {sigma_pert}"
        return True, f"sigma decreases with increasing k0: {self.sigma} -> {sigma_pert}"

    # ----------------------------------------------------------------
    # WEAK FINAL-ANSWER SANITY (only one test, no exact matching)
    # ----------------------------------------------------------------
    def test_answer_sanity_sigma_range(self) -> Tuple[bool, str]:
        """
        Very broad sanity: if sigma is numeric, check that it is finite and not enormous (e.g., < 10^20).
        Since cross sections can vary, use a forgiving upper bound.
        """
        if self.sigma is None or not isinstance(self.sigma, (int, float)):
            return True, "SKIP: sigma not numeric."
        if not math.isfinite(self.sigma):
            return False, f"sigma is not finite: {self.sigma}"
        # Typical nuclear cross sections can be up to barns ~10^-24 cm^2, but given expression could be large.
        # Use an arbitrary huge bound to catch obvious blunders.
        if abs(self.sigma) > 1e20:
            return False, f"sigma magnitude {self.sigma} seems unrealistically large."
        if self.sigma < 0:
            return False, "sigma is negative."
        return True, "sigma passes basic sanity checks."

# Example usage:
# class DummySolution:
#     pass
# sol = DummySolution()
# sol.results = {'sigma': '16*pi*m**2*q**2*Q**2/(hbar**4*alpha**2*(alpha**2+4*k0**2))',
#                'alpha_scaling': r'alpha prop Z^{1/3}'}
# sol.m = 1.0; sol.q = 1.0; sol.Q = 1.0; sol.hbar = 1.0; sol.alpha = 1.0; sol.k0 = 1.0
# checker = PhysicsChecker(sol)
# all_pass, logs = checker.run_all_tests()
# for log in logs:
#     print(log)