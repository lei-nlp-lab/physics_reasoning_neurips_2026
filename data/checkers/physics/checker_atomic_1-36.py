import sympy as sp
import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits the solution of a semiclassical electron in a magnetic field,
    verifying that the Bohr-Sommerfeld quantization condition leads to the
    correct quantization of the orbit area S in k-space.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely, use None for missing keys
        results = getattr(solution_instance, 'results', {})
        self.S = results.get('S', None)
        self.Phi = results.get('Phi', None)          # optional intermediate
        self.n = results.get('n', None)             # Landau level index
        self.B = results.get('B', None)            # magnetic field
        self.e = results.get('e', None)            # elementary charge
        self.hbar = results.get('hbar', None)      # reduced Planck constant
        self.c = results.get('c', None)            # speed of light
        self.gamma = results.get('gamma', None)    # zero-point correction

        # Fallback: if not in results, try to extract from givens (still None if missing)
        givens = getattr(solution_instance, 'givens', {})
        if self.n is None:
            self.n = givens.get('n', None)
        if self.B is None:
            self.B = givens.get('B', None)
        if self.e is None:
            self.e = givens.get('e', None)
        if self.hbar is None:
            self.hbar = givens.get('hbar', None)
        if self.c is None:
            self.c = givens.get('c', None)
        if self.gamma is None:
            # DO NOT assign a numeric default; leave as None
            self.gamma = givens.get('gamma', None)

        # Extract explanation steps for trace consistency (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

    # ------------------------------------------------------------
    # Interface checks: required keys exist and have valid types
    # ------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the essential output key 'S' is present and numeric."""
        if self.S is None:
            return (False, "results['S'] is missing")
        if not isinstance(self.S, (int, float, sp.Expr, sp.Number)):
            return (False, f"results['S'] has unexpected type: {type(self.S)}")
        return (True, "results['S'] is present and numeric/symbolic")

    def test_interface_n_B_available(self) -> Tuple[bool, str]:
        """Check that the inputs n and B are available for numerical checks."""
        missing = []
        if self.n is None:
            missing.append("n")
        if self.B is None:
            missing.append("B")
        if self.e is None:
            missing.append("e")
        if self.hbar is None:
            missing.append("hbar")
        if self.c is None:
            missing.append("c")
        if self.gamma is None:
            missing.append("gamma")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens are present")

    # ------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_flux_quantization(self) -> Tuple[bool, str]:
        """
        If Phi (magnetic flux) is present in results, verify that it satisfies
        Phi = (n + gamma) * 2*pi*hbar*c / e.
        If Phi is not present, but we have all givens and S, we can deduce Phi
        from the relation S = (e^2*B/(hbar^2*c^2))*Phi and check consistency.
        """
        # Check for missing critical data
        if any(x is None for x in (self.n, self.e, self.hbar, self.c, self.gamma)):
            return (False, "SKIP: insufficient givens for flux check (n, e, hbar, c, gamma required)")
        try:
            n_val = float(self.n) if not isinstance(self.n, sp.Basic) else float(self.n.evalf())
            hbar_val = float(self.hbar)
            e_val = float(self.e)
            c_val = float(self.c)
            gamma_val = float(self.gamma)  # may be None, but we already checked
        except:
            return (False, "Cannot evaluate numeric values for flux check")
        expected_Phi = (n_val + gamma_val) * 2 * math.pi * hbar_val * c_val / e_val

        if self.Phi is not None:
            # Direct check
            try:
                Phi_val = float(self.Phi) if not isinstance(self.Phi, sp.Basic) else float(self.Phi.evalf())
            except:
                return (False, "Cannot evaluate Phi numerically")
            if math.isclose(Phi_val, expected_Phi, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Flux matches: Phi = {Phi_val:.3e}")
            else:
                return (False, f"Flux mismatch: got {Phi_val:.3e}, expected {expected_Phi:.3e}")
        elif self.S is not None and self.B is not None:
            # Derive Phi from S using relation S = (e^2*B/(hbar^2*c^2))*Phi
            try:
                B_val = float(self.B)
                S_val = float(self.S) if not isinstance(self.S, sp.Basic) else float(self.S.evalf())
            except:
                return (False, "Cannot evaluate S or B numerically")
            derived_Phi = S_val * (hbar_val**2 * c_val**2) / (e_val**2 * B_val)
            if math.isclose(derived_Phi, expected_Phi, rel_tol=2e-2, abs_tol=1e-2):
                return (True, f"Derived flux consistent: Phi = {derived_Phi:.3e}")
            else:
                return (False, f"Derived flux {derived_Phi:.3e} != expected {expected_Phi:.3e}")
        else:
            return (False, "SKIP: neither Phi nor S+B available for flux check")

    def test_invariant_k_space_area_formula(self) -> Tuple[bool, str]:
        """
        Verify that the relation S = (n + 1/2) * (2*pi*e/(hbar*c)) * B holds.
        This is the main invariant.
        """
        if any(x is None for x in (self.S, self.n, self.B, self.e, self.hbar, self.c, self.gamma)):
            return (False, "SKIP: missing data for S formula check")
        try:
            n_val = float(self.n) if not isinstance(self.n, sp.Basic) else float(self.n.evalf())
            B_val = float(self.B) if not isinstance(self.B, sp.Basic) else float(self.B.evalf())
            e_val = float(self.e)
            hbar_val = float(self.hbar)
            c_val = float(self.c)
            S_val = float(self.S) if not isinstance(self.S, sp.Basic) else float(self.S.evalf())
            gamma_val = float(self.gamma)
        except Exception as e:
            return (False, f"Cannot evaluate numeric values: {e}")
        expected_S = (n_val + gamma_val) * 2 * math.pi * e_val * B_val / (hbar_val * c_val)
        if math.isclose(S_val, expected_S, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S matches formula: {S_val:.3e} ≈ {expected_S:.3e}")
        else:
            return (False, f"S mismatch: got {S_val:.3e}, expected {expected_S:.3e}")

    # ------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ------------------------------------------------------------
    def test_trace_flux_from_Bohr_Sommerfeld(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps contain the key relation:
        Φ = (n + 1/2) * 2πħc / e. This is a weak qualitative check.
        """
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps available")
        flux_keywords = ['flux', 'Phi', 'Φ']
        half_keywords = ['1/2', 'half', '0.5', 'gamma']
        for step in self.explanation_steps:
            if not isinstance(step, str):
                continue
            step_lower = step.lower()
            has_flux = any(kw in step_lower for kw in flux_keywords)
            has_half = any(kw in step_lower for kw in half_keywords)
            if has_flux and has_half:
                return (True, "Explanation step mentions flux quantized with half-integer")
        return (False, "No explanation step explicitly connects flux and half-integer quantization")

    def test_trace_area_relation(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps relate real-space area A to k-space area S
        via A = (hc/(eB))^2 * S.
        """
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps")
        for step in self.explanation_steps:
            if not isinstance(step, str):
                continue
            step_lower = step.lower()
            # Flexible pattern: look for "hc/(eb)" or "h c / (e B)" etc.
            if 'hc/(eb)' in step_lower or ('h' in step_lower and 'c/(eb)' in step_lower) or 'hc/(eB)' in step_lower:
                return (True, "Found area scaling relation between A and S")
        return (False, "No step shows area scaling A ∝ S/B^2")

    # ------------------------------------------------------------
    # Monotonicity checks (trend analysis via perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_S_increases_with_B(self) -> Tuple[bool, str]:
        """
        For a fixed n, S should increase as B increases.
        Perturb B upward by 5% and compute expected S from formula.
        """
        if any(x is None for x in (self.S, self.B, self.n, self.e, self.hbar, self.c, self.gamma)):
            return (False, "SKIP: missing data for monotonicity")
        try:
            B0 = float(self.B)
            S0 = float(self.S) if not isinstance(self.S, sp.Basic) else float(self.S.evalf())
            n_val = float(self.n)
            gamma_val = float(self.gamma)
            e_val = float(self.e)
            hbar_val = float(self.hbar)
            c_val = float(self.c)
        except:
            return (False, "Cannot evaluate baseline values")
        # Perturb B upward by 5%
        B1 = B0 * 1.05
        S1_expected = (n_val + gamma_val) * 2 * math.pi * e_val * B1 / (hbar_val * c_val)
        if S1_expected > S0:
            return (True, f"S increases with B: baseline {S0:.3e}, 5% increase → {S1_expected:.3e}")
        else:
            return (False, f"S does not increase with B: baseline {S0:.3e}, predicted {S1_expected:.3e}")

    def test_monotonicity_S_increases_with_n(self) -> Tuple[bool, str]:
        """
        For fixed B, S should increase linearly with n.
        Perturb n by +1 (assuming integer index) and check trend.
        """
        if any(x is None for x in (self.S, self.n, self.B, self.e, self.hbar, self.c, self.gamma)):
            return (False, "SKIP: missing data for n monotonicity")
        try:
            n0 = float(self.n)
            B0 = float(self.B)
            S0 = float(self.S)
            gamma_val = float(self.gamma)
            e_val = float(self.e)
            hbar_val = float(self.hbar)
            c_val = float(self.c)
        except:
            return (False, "Cannot evaluate numeric values")
        # Perturb n by +1 (typical Landau level increment)
        n1 = n0 + 1
        S1_expected = (n1 + gamma_val) * 2 * math.pi * e_val * B0 / (hbar_val * c_val)
        if S1_expected > S0:
            return (True, f"S increases with n: baseline {S0:.3e}, n+1 → {S1_expected:.3e}")
        else:
            return (False, f"S does not increase with n: baseline {S0:.3e}, predicted {S1_expected:.3e}")

    # ------------------------------------------------------------
    # Weak final-answer sanity check (do NOT re-derive, only sign and order)
    # ------------------------------------------------------------
    def test_answer_sanity_S_positive_and_finite(self) -> Tuple[bool, str]:
        """
        Ensure S is positive and finite. This is a weak sanity check.
        """
        if self.S is None:
            return (False, "SKIP: S not available")
        try:
            S_val = float(self.S) if not isinstance(self.S, sp.Basic) else float(self.S.evalf())
        except:
            return (False, "Cannot evaluate S numerically")
        if math.isfinite(S_val) and S_val > 0:
            return (True, f"S = {S_val:.3e} is positive and finite")
        else:
            return (False, f"S = {S_val} is not positive and finite")

    # ------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Collect all test methods
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


# Example usage (if needed for local testing):
if __name__ == "__main__":
    # Create a mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'S': sp.sympify("(1/2 + 1)*2*pi*e*B/(hbar*c)"),  # symbolic expression
                'n': 1,
                'B': 1.0,
                'e': sp.symbols('e'),  # keep symbolic? We'll evaluate
                'hbar': sp.symbols('hbar'),
                'c': sp.symbols('c'),
                'gamma': 0.5,
                'Phi': None
            }
            self.givens = {'n': 1, 'B': 1.0, 'e': 1.602e-19, 'hbar': 1.055e-34, 'c': 3e8}
            self.explanation_steps = [
                "Use Bohr-Sommerfeld: ∮ (p - eA/c)·dr = (n+1/2)h",
                "From Lorentz force: ∮ p·dr = 2eΦ/c",
                "Thus Φ = (n+1/2)h c / e",
                "Area scaling: A = (hc/(eB))^2 S",
                "Combine to get S = (n+1/2) 2πeB/(ħc)"
            ]

    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)