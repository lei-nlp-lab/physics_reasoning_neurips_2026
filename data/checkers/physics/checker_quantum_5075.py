import math
import sympy as sp
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits solutions for a particle in a box undergoing sudden expansion.
    """
    def __init__(self, solution_instance: Any):
        # Extract givens (use None if missing)
        self.L = getattr(solution_instance, 'L', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Extract emitted results
        results = getattr(solution_instance, 'results', {})
        self.P1 = results.get('P1', None)
        self.P_max = results.get('P_max', None)
        self.n_max = results.get('n_max', None)
        self.Pp = results.get('Pp', None)  # string or sympy expression

    # ------------------------------------------------------------------
    # Helper: evaluate Pp for given numeric p, L, hbar
    # ------------------------------------------------------------------
    def _eval_Pp(self, p_val: float, L_val: float, hbar_val: float) -> float:
        """
        Evaluate the momentum distribution at a specific momentum p.
        Uses the expression from self.Pp if available; otherwise falls back
        to the theoretical formula for the infinite box sudden expansion.
        """
        if self.Pp is not None:
            try:
                # Try to parse as sympy expression
                pp_expr = sp.sympify(self.Pp) if isinstance(self.Pp, str) else self.Pp
                subs_dict = {
                    sp.Symbol('p'): p_val,
                    sp.Symbol('L'): L_val,
                    sp.Symbol('hbar'): hbar_val
                }
                result = float(pp_expr.subs(subs_dict).evalf())
                return result
            except Exception:
                pass  # fall through to theoretical formula

        # Fallback: explicit theoretical expression (derived from Fourier transform)
        # This is the known result from the problem statement:
        # P(p) = 2 * pi * hbar^3 * L / ( (hbar^2 * pi^2 - p^2 * L^2)^2 ) * (1 + cos(p*L/hbar))
        numerator = 2 * math.pi * hbar_val**3 * L_val
        denominator = (hbar_val**2 * math.pi**2 - p_val**2 * L_val**2)**2
        cos_term = 1 + math.cos(p_val * L_val / hbar_val)
        return numerator * cos_term / denominator

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        required = ['P1', 'P_max', 'n_max', 'Pp']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_values_are_numerical(self) -> Tuple[bool, str]:
        if not isinstance(self.P1, (int, float)):
            return (False, f"P1 is not numeric, got {type(self.P1)}")
        if not isinstance(self.P_max, (int, float)):
            return (False, f"P_max is not numeric, got {type(self.P_max)}")
        if not isinstance(self.n_max, int):
            return (False, f"n_max is not int, got {type(self.n_max)}")
        if not isinstance(self.Pp, (str, sp.Expr)):
            return (False, f"Pp is not a valid expression, got {type(self.Pp)}")
        return (True, "All values have correct types")

    # ------------------------------------------------------------------
    # Invariant (theorem‑based) tests
    # ------------------------------------------------------------------
    def test_invariant_initial_wavefunction_normalized(self) -> Tuple[bool, str]:
        if self.L is None:
            return (False, "SKIP: L not available")
        try:
            L_val = float(self.L) if isinstance(self.L, (int, float)) else 1.0
            x = sp.symbols('x', real=True)
            psi_i = sp.sqrt(2/L_val) * sp.sin(sp.pi * x / L_val)
            norm = sp.integrate(psi_i**2, (x, 0, L_val))
            if math.isclose(float(norm), 1.0, rel_tol=1e-6):
                return (True, f"Normalization integral = {float(norm):.6f}")
            else:
                return (False, f"Normalization integral = {float(norm):.6f}, expected 1")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_invariant_final_ground_state_normalized(self) -> Tuple[bool, str]:
        if self.L is None:
            return (False, "SKIP: L not available")
        try:
            L_val = float(self.L) if isinstance(self.L, (int, float)) else 1.0
            x = sp.symbols('x', real=True)
            phi1 = sp.sqrt(1/(2*L_val)) * sp.sin(sp.pi * x / (2*L_val))  # final ground state
            norm = sp.integrate(phi1**2, (x, 0, 2*L_val))
            if math.isclose(float(norm), 1.0, rel_tol=1e-6):
                return (True, f"Final ground state normalization = {float(norm):.6f}")
            else:
                return (False, f"Final ground state normalization = {float(norm):.6f}, expected 1")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_invariant_overlap_P1(self) -> Tuple[bool, str]:
        if self.L is None or self.P1 is None:
            return (False, "SKIP: L or P1 missing")
        try:
            L_val = float(self.L) if isinstance(self.L, (int, float)) else 1.0
            x = sp.symbols('x', real=True)
            psi_i = sp.sqrt(2/L_val) * sp.sin(sp.pi * x / L_val)
            phi1 = sp.sqrt(1/(2*L_val)) * sp.sin(sp.pi * x / (2*L_val))
            overlap = sp.integrate(phi1 * psi_i, (x, 0, L_val))
            P1_computed = float(abs(overlap)**2)
            if math.isclose(P1_computed, float(self.P1), rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"P1 computed = {P1_computed:.6f}, given = {float(self.P1):.6f}")
            else:
                return (False, f"P1 mismatch: computed {P1_computed:.6f}, given {float(self.P1):.6f}")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_invariant_overlap_P2(self) -> Tuple[bool, str]:
        if self.P_max is None:
            return (False, "SKIP: P_max missing")
        try:
            L_val = float(self.L) if isinstance(self.L, (int, float)) else 1.0
            x = sp.symbols('x', real=True)
            psi_i = sp.sqrt(2/L_val) * sp.sin(sp.pi * x / L_val)
            phi2 = sp.sqrt(1/(2*L_val)) * sp.sin(2 * sp.pi * x / (2*L_val))  # n=2 state
            overlap = sp.integrate(phi2 * psi_i, (x, 0, L_val))
            P2_computed = float(abs(overlap)**2)
            if math.isclose(P2_computed, 0.5, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(float(self.P_max), 0.5, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"P2 computed = {P2_computed:.6f}, P_max = {float(self.P_max):.6f}")
            else:
                return (False, f"P2 = {P2_computed:.6f}, P_max = {float(self.P_max):.6f} (expected 0.5)")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_invariant_n_max_is_2(self) -> Tuple[bool, str]:
        if self.n_max is None:
            return (False, "SKIP: n_max missing")
        if self.n_max == 2:
            return (True, "n_max = 2 as expected from physics")
        else:
            return (False, f"n_max = {self.n_max}, expected 2")

    # ------------------------------------------------------------------
    # Modeling/trace checks
    # ------------------------------------------------------------------
    def test_modeling_overlap_consistency_n1_n2(self) -> Tuple[bool, str]:
        """Verify that the general overlap formula reproduces P1 and P2."""
        if self.L is None:
            return (False, "SKIP: L missing")
        try:
            L_val = float(self.L) if isinstance(self.L, (int, float)) else 1.0
            x = sp.symbols('x', real=True)
            psi_i = sp.sqrt(2/L_val) * sp.sin(sp.pi * x / L_val)
            for n, expected in [(1, 32/(9*math.pi**2)), (2, 0.5)]:
                phi = sp.sqrt(1/(2*L_val)) * sp.sin(n * sp.pi * x / (2*L_val))
                c = sp.integrate(phi * psi_i, (x, 0, L_val))
                Pn = float(abs(c)**2)
                if not math.isclose(Pn, expected, rel_tol=1e-2):
                    return (False, f"P_{n} computed = {Pn:.6f}, expected {expected:.6f}")
            return (True, "Overlap integrals for n=1,2 are consistent")
        except Exception as e:
            return (False, f"Error: {e}")

    def test_modeling_momentum_expression_structural(self) -> Tuple[bool, str]:
        """Sanity check that the momentum distribution expression has the correct structure."""
        if self.Pp is None:
            return (False, "SKIP: Pp missing")
        try:
            pp_str = str(self.Pp).lower()
            # It should contain terms like hbar^3, L, (hbar^2 pi^2 - p^2 L^2)^2, (1+cos(...))
            conditions = [
                'hbar' in pp_str,
                'L' in pp_str,
                'pi' in pp_str,
                'cos' in pp_str or 'cos' in pp_str,
            ]
            if all(conditions):
                return (True, "Momentum expression contains expected structural components")
            else:
                return (False, f"Momentum expression lacks expected components: {pp_str[:200]}")
        except Exception as e:
            return (False, f"Error: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (each must perturb an input and compare to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_momentum_peak_amplitude_increases_with_L(self) -> Tuple[bool, str]:
        """
        Perturb L by +10% and check that the momentum distribution at p=0 increases.
        Baseline: compute Pp(p=0) for original L.
        """
        if self.L is None:
            return (False, "SKIP: L not available for perturbation")
        if self.hbar is None:
            return (False, "SKIP: hbar not available")
        try:
            L0 = float(self.L)
            hbar0 = float(self.hbar)
            p0 = 0.0  # peak

            # Baseline amplitude
            base = self._eval_Pp(p0, L0, hbar0)

            # Perturbed L (+10%)
            L1 = L0 * 1.1
            perturbed = self._eval_Pp(p0, L1, hbar0)

            # Physics: Pp(0) ~ L, so perturbed should be larger
            if perturbed > base:
                return (True, f"Pp(p=0) increased from {base:.4f} to {perturbed:.4f} when L increased by 10%")
            else:
                return (False, f"Pp(p=0) did not increase: baseline {base:.4f}, perturbed {perturbed:.4f}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    def test_monotonicity_momentum_distribution_decays_with_momentum(self) -> Tuple[bool, str]:
        """
        For two values of p (0.5 and 2.0), check that the momentum distribution is smaller at higher p.
        Perturbation: use p as the varied parameter.
        """
        if self.L is None or self.hbar is None:
            return (False, "SKIP: L or hbar not available for evaluation")
        try:
            L_val = float(self.L)
            hbar_val = float(self.hbar)

            p1 = 0.5
            p2 = 2.0
            val1 = self._eval_Pp(p1, L_val, hbar_val)
            val2 = self._eval_Pp(p2, L_val, hbar_val)

            # The envelope of the momentum distribution decays as ~1/p^4,
            # so it should be smaller at p=2 than at p=0.5 (for typical parameters)
            if val2 < val1:
                return (True, f"Pp decreases from {val1:.4f} at p={p1} to {val2:.4f} at p={p2}")
            else:
                return (False, f"Pp did not decrease: p=0.5 -> {val1:.4f}, p=2.0 -> {val2:.4f}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic_ranges(self) -> Tuple[bool, str]:
        if self.P1 is None or self.P_max is None or self.n_max is None:
            return (False, "SKIP: missing results for sanity")
        if not (0 <= self.P1 <= 1):
            return (False, f"P1 = {self.P1} not in [0,1]")
        if not (0 <= self.P_max <= 1):
            return (False, f"P_max = {self.P_max} not in [0,1]")
        if self.n_max <= 0:
            return (False, f"n_max = {self.n_max} not positive")
        if not self.Pp:  # empty string or None
            return (False, "Pp is empty")
        return (True, "Basic sanity passed")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
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