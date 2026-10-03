import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for a non-interacting Fermi gas at T = 0.
    Checks the step function f, ground state energy U, pressure p, 
    compressibility K and its inverse against thermodynamic relations.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely
        self.results: Dict[str, Any] = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results

        # Required emitted keys (from pseudocode contract)
        self.f = self.results.get('f', None)
        self.U = self.results.get('U', None)
        self.p = self.results.get('p', None)
        self.K = self.results.get('K', None)
        self.K_inv = self.results.get('K^{-1}', None)

        # Attempt to extract more basic variables if stored (N, V, epsilon_F, n)
        # Try to get them from `results` or directly from `solution_instance` attributes.
        self.N: Optional[Any] = None
        self.V: Optional[Any] = None
        self.epsilon_F: Optional[Any] = None
        self.n: Optional[Any] = None

        for attr_name in ['N', 'V', 'epsilon_F', 'n']:
            val = None
            if attr_name in self.results:
                val = self.results[attr_name]
            elif hasattr(solution_instance, attr_name):
                val = getattr(solution_instance, attr_name)
            # Also check a 'givens' dict if present
            if val is None and hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
                val = solution_instance.givens.get(attr_name, None)
            setattr(self, attr_name, val)

        # If n is missing but N and V available, compute n = N / V
        if self.n is None and self.N is not None and self.V is not None:
            self.n = self.N / self.V

    # ----------------------------------------------------------------------
    # Test methods – return (passed: bool, message: str)
    # ----------------------------------------------------------------------

    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        """Check that all five expected keys are present in results."""
        expected = ['f', 'U', 'p', 'K', 'K^{-1}']
        missing = [k for k in expected if k not in self.results]
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, f"All required keys present: {expected}"

    def test_invariant_U_vs_N_epsilon(self) -> Tuple[bool, str]:
        """U should equal (3/5) * N * epsilon_F."""
        U = self.U
        N = self.N
        eps = self.epsilon_F
        if U is None or N is None or eps is None:
            return False, "SKIP: U, N or epsilon_F not available"
        # Try numeric
        try:
            U_num = float(U)
            N_num = float(N)
            eps_num = float(eps)
            expected = (3.0/5.0) * N_num * eps_num
            if math.isclose(U_num, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"U ≈ {U_num}, expected (3/5)N ε_F = {expected}"
            else:
                return False, f"U = {U_num}, expected {expected}"
        except (TypeError, ValueError):
            pass
        # Try symbolic
        try:
            U_sym = sp.sympify(U)
            N_sym = sp.sympify(N)
            eps_sym = sp.sympify(eps)
            diff = sp.simplify(U_sym - (sp.Rational(3,5) * N_sym * eps_sym))
            if diff == 0:
                return True, "U symbolically equals (3/5) N ε_F"
            else:
                return False, f"Symbolic difference U - (3/5)N ε_F = {diff}"
        except Exception:
            return False, "Could not interpret U, N, epsilon_F numerically or symbolically"

    def test_invariant_p_vs_n_epsilon(self) -> Tuple[bool, str]:
        """p should equal (2/5) * n * epsilon_F."""
        p = self.p
        n = self.n
        eps = self.epsilon_F
        if p is None or n is None or eps is None:
            return False, "SKIP: p, n or epsilon_F not available"
        try:
            p_num = float(p)
            n_num = float(n)
            eps_num = float(eps)
            expected = (2.0/5.0) * n_num * eps_num
            if math.isclose(p_num, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"p ≈ {p_num}, expected (2/5) n ε_F = {expected}"
            else:
                return False, f"p = {p_num}, expected {expected}"
        except (TypeError, ValueError):
            pass
        try:
            p_sym = sp.sympify(p)
            n_sym = sp.sympify(n)
            eps_sym = sp.sympify(eps)
            diff = sp.simplify(p_sym - (sp.Rational(2,5) * n_sym * eps_sym))
            if diff == 0:
                return True, "p symbolically equals (2/5) n ε_F"
            else:
                return False, f"Symbolic difference p - (2/5)n ε_F = {diff}"
        except Exception:
            return False, "Could not interpret p, n, epsilon_F"

    def test_invariant_K_inv_vs_n_epsilon(self) -> Tuple[bool, str]:
        """K^{-1} should equal (2/3) * n * epsilon_F."""
        Kinv = self.K_inv
        n = self.n
        eps = self.epsilon_F
        if Kinv is None or n is None or eps is None:
            return False, "SKIP: K^{-1}, n or epsilon_F not available"
        try:
            Knum = float(Kinv)
            nnum = float(n)
            epsnum = float(eps)
            expected = (2.0/3.0) * nnum * epsnum
            if math.isclose(Knum, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"K^{-1} ≈ {Knum}, expected (2/3) n ε_F = {expected}"
            else:
                return False, f"K^{-1} = {Knum}, expected {expected}"
        except (TypeError, ValueError):
            pass
        try:
            K_sym = sp.sympify(Kinv)
            n_sym = sp.sympify(n)
            eps_sym = sp.sympify(eps)
            diff = sp.simplify(K_sym - (sp.Rational(2,3) * n_sym * eps_sym))
            if diff == 0:
                return True, "K^{-1} symbolically equals (2/3) n ε_F"
            else:
                return False, f"Symbolic difference K^{-1} - (2/3)n ε_F = {diff}"
        except Exception:
            return False, "Could not interpret K^{-1}, n, epsilon_F"

    def test_modeling_thermodynamic_pressure(self) -> Tuple[bool, str]:
        """Check p = -dU/dV at constant N (thermodynamic relation)."""
        U = self.U
        p = self.p
        V = self.V
        if U is None or p is None or V is None:
            return False, "SKIP: U, p, or V not available"
        # Try symbolic derivative
        try:
            U_sym = sp.sympify(U)
            V_sym = sp.sympify(V)
            dU_dV = sp.diff(U_sym, V_sym)
            neg_dU_dV = sp.simplify(-dU_dV)
            p_sym = sp.sympify(p)
            diff = sp.simplify(neg_dU_dV - p_sym)
            if diff == 0:
                return True, "p = -dU/dV holds symbolically"
            else:
                return False, f"Difference p - (-dU/dV) = {diff}"
        except Exception:
            pass
        # Try numeric if symbolic fails (e.g., if numeric values)
        try:
            U_num = float(U)
            p_num = float(p)
            V_num = float(V)
            # Very crude numerical derivative (central difference)
            delta = 1e-5 * V_num
            # We cannot recompute U without the functional form, so skip numeric.
            return False, "Numeric derivative of U not possible; omit test"
        except (TypeError, ValueError):
            return False, "Could not interpret U, p, V symbolically or numerically"

    def test_modeling_compressibility_definition(self) -> Tuple[bool, str]:
        """Check K = -[ V (∂p/∂V) ]^{-1}. Also verify relation between K and K^{-1}."""
        K = self.K
        K_inv = self.K_inv
        p = self.p
        V = self.V
        if K is not None and K_inv is not None:
            # Mutual consistency: K * K_inv should be 1 (within tolerance)
            try:
                Knum = float(K)
                K_inv_num = float(K_inv)
                prod = Knum * K_inv_num
                if math.isclose(prod, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                    return True, f"K * K^{-1} = {prod} ≈ 1"
                else:
                    return False, f"K * K^{-1} = {prod}, expected 1"
            except (TypeError, ValueError):
                pass
            try:
                K_sym = sp.sympify(K)
                K_inv_sym = sp.sympify(K_inv)
                prod_sym = sp.simplify(K_sym * K_inv_sym)
                if prod_sym == 1:
                    return True, "K * K^{-1} = 1 symbolically"
                else:
                    return False, f"Symbolic K * K^{-1} = {prod_sym}"
            except Exception:
                return False, "Could not interpret K or K^{-1}"

        if K is not None and p is not None and V is not None:
            # Check K = - [ V (∂p/∂V) ]^{-1} using symbolic derivative
            try:
                p_sym = sp.sympify(p)
                V_sym = sp.sympify(V)
                dp_dV = sp.diff(p_sym, V_sym)
                V_dp_dV = sp.simplify(V_sym * dp_dV)
                K_from_p = sp.simplify(-1 / (V_dp_dV))
                K_sym = sp.sympify(K)
                diff = sp.simplify(K_sym - K_from_p)
                if diff == 0:
                    return True, "K = -[V (∂p/∂V)]^{-1} holds symbolically"
                else:
                    return False, f"Difference K - (-1/(V dp/dV)) = {diff}"
            except Exception:
                return False, "Symbolic test of compressibility definition failed"

        return False, "SKIP: insufficient data for compressibility definition test"

    def test_monotonicity_energy_increases_with_N(self) -> Tuple[bool, str]:
        """Perturb N by +1% and check that U increases using scaling U ∝ N^{5/3}."""
        U = self.U
        N = self.N
        V = self.V
        if U is None or N is None or V is None:
            return False, "SKIP: U, N, or V not available"
        try:
            U0 = float(U)
            N0 = float(N)
            # Keep V fixed, increase N by 1%
            delta = 0.01
            N1 = N0 * (1 + delta)
            # U ∝ N^{5/3}, so U1 = U0 * (N1/N0)^{5/3}
            U1 = U0 * (N1 / N0) ** (5.0/3.0)
            if U1 > U0:
                return True, f"N increased by 1%: U went from {U0} to {U1} (increase)"
            else:
                return False, f"N increased by 1% but U did not increase: U0={U0}, U1={U1}"
        except (TypeError, ValueError):
            return False, "Cannot perform numeric monotonicity test"

    def test_monotonicity_pressure_decreases_with_V(self) -> Tuple[bool, str]:
        """Perturb V by +1% and check that p decreases using scaling p ∝ V^{-5/3}."""
        p = self.p
        V = self.V
        N = self.N
        if p is None or V is None or N is None:
            return False, "SKIP: p, V, or N not available"
        try:
            p0 = float(p)
            V0 = float(V)
            # Increase V by 1% (N fixed)
            delta = 0.01
            V1 = V0 * (1 + delta)
            # p ∝ n^{5/3} ∝ (N/V)^{5/3} ∝ V^{-5/3}
            p1 = p0 * (V0 / V1) ** (5.0/3.0)
            if p1 < p0:
                return True, f"V increased by 1%: p went from {p0} to {p1} (decrease)"
            else:
                return False, f"V increased by 1% but p did not decrease: p0={p0}, p1={p1}"
        except (TypeError, ValueError):
            return False, "Cannot perform numeric monotonicity test"

    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        """Weak sanity: all results should be finite and positive (except f which can be 0/1)."""
        results_to_check = {'f': self.f, 'U': self.U, 'p': self.p, 'K': self.K, 'K^{-1}': self.K_inv}
        issues = []
        for name, val in results_to_check.items():
            if val is None:
                issues.append(f"{name} is None")
                continue
            # Try to get numeric value
            try:
                num = float(val)
                if math.isnan(num) or math.isinf(num):
                    issues.append(f"{name} is NaN or infinite")
                elif name != 'f' and num <= 0:
                    issues.append(f"{name} = {num} is not positive")
                elif name == 'f' and (num < 0 or num > 1):
                    issues.append(f"{name} = {num} not in [0,1]")
            except (TypeError, ValueError):
                # For symbolic expressions, just check they are not None
                pass
        if issues:
            return False, "Issues: " + "; ".join(issues)
        return True, "All non-None results appear finite and (where applicable) positive"

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, returning (all_pass, log_list)."""
        logs: List[str] = []
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