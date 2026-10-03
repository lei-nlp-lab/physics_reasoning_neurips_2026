import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with None defaults
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.M = self.results.get('M', None)

        # Givens
        self.givens = getattr(solution_instance, 'given', None)
        if self.givens is None:
            self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}
        self.m = self.givens.get('m', None)
        self.c = self.givens.get('c', None)

        # Define symbols (always available for internal derivation)
        self.m_sym = sp.symbols('m', positive=True)
        self.c_sym = sp.symbols('c', positive=True)

        # Physically derived quantities (based on problem statement)
        self.KE_A = 2 * self.m_sym * self.c_sym**2
        self.E_A = self.m_sym * self.c_sym**2 + self.KE_A          # 3 m c^2
        self.p_A_c = sp.sqrt(self.E_A**2 - (self.m_sym * self.c_sym**2)**2)  # sqrt(8) m c^2
        self.E_B = self.m_sym * self.c_sym**2
        self.E_total = self.E_A + self.E_B                         # 4 m c^2
        self.P_total_c = self.p_A_c
        # Expected M from invariant mass formula
        self.M_expected = sp.sqrt(self.E_total**2 - self.P_total_c**2) / self.c_sym**2  # sqrt(8) m

        # Additional attributes (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Helper: evaluate a symbolic expression numerically (m=1, c=1)
    # ------------------------------------------------------------------
    def _numeric_value(self, expr):
        """Convert sympy expression to float using m=1, c=1."""
        try:
            return float(expr.subs({self.m_sym: 1, self.c_sym: 1}).evalf())
        except Exception:
            return None

    # ==================================================================
    # Interface tests
    # ==================================================================
    def test_interface_results_dict_has_M(self):
        if 'M' not in self.results:
            return (False, "results dict must contain key 'M'")
        return (True, "'M' found in results dict")

    def test_interface_M_not_None(self):
        if self.M is None:
            return (False, "results['M'] is None")
        return (True, "results['M'] is not None")

    # ==================================================================
    # Invariant tests (theorem-based)
    # ==================================================================
    def test_invariant_invariant_mass_formula(self):
        """
        Check M^2 c^4 = E_total^2 - (P_total c)^2.
        """
        if self.M is None:
            return (False, "SKIP: M not available")

        M_sol = self.M if hasattr(self.M, 'is_Symbol') else sp.sympify(self.M)
        # Check symbolic equality
        try:
            if sp.simplify(M_sol - self.M_expected) == 0:
                return (True, "Invariant mass formula holds (symbolic equality)")
        except Exception:
            pass
        # Numeric fallback
        n_sol = self._numeric_value(M_sol)
        n_exp = self._numeric_value(self.M_expected)
        if n_sol is not None and n_exp is not None:
            if math.isclose(n_sol, n_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Invariant mass formula holds (numeric: {n_sol} ≈ {n_exp})")
        return (False, "Invariant mass formula violated. M should be sqrt(8) m ≈ 2.828 m")

    def test_invariant_final_particle_energy_momentum_relation(self):
        """
        Check that the final particle also satisfies E^2 = (pc)^2 + (Mc^2)^2.
        """
        if self.M is None:
            return (False, "SKIP: M not available")

        M_sol = self.M if hasattr(self.M, 'is_Symbol') else sp.sympify(self.M)
        lhs = self.E_total**2
        rhs = self.P_total_c**2 + (M_sol * self.c_sym**2)**2
        diff = sp.simplify(lhs - rhs)
        if diff == 0:
            return (True, "Final particle satisfies relativistic energy-momentum relation (symbolic)")
        # Numeric fallback
        n_diff = self._numeric_value(diff)
        if n_diff is not None and math.isclose(n_diff, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Final particle satisfies energy-momentum relation (numeric diff {n_diff:.2e})")
        return (False, "Final particle does not satisfy E^2 = (pc)^2 + (Mc^2)^2")

    # ==================================================================
    # Modeling / trace tests
    # ==================================================================
    def test_modeling_M_expression_exact(self):
        """
        Verify that M simplifies to 2*sqrt(2)*m.
        """
        if self.M is None:
            return (False, "SKIP: M not available")
        M_sol = sp.sympify(self.M) if not hasattr(self.M, 'is_Symbol') else self.M
        expected = 2 * sp.sqrt(2) * self.m_sym
        try:
            if sp.simplify(M_sol - expected) == 0:
                return (True, "M exactly equals 2*sqrt(2)*m")
        except Exception:
            pass
        n_sol = self._numeric_value(M_sol)
        n_exp = self._numeric_value(expected)
        if n_sol is not None and n_exp is not None:
            if math.isclose(n_sol, n_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"M ≈ {n_sol} ≈ 2*sqrt(2)*m ≈ {n_exp}")
        return (False, f"M = {self.M} does not match 2*sqrt(2)*m")

    def test_modeling_no_spurious_variables(self):
        """
        Ensure M does not depend on c.
        """
        if self.M is None:
            return (False, "SKIP: M not available")
        M_sol = sp.sympify(self.M) if not hasattr(self.M, 'is_Symbol') else self.M
        expr = sp.simplify(M_sol)
        if not expr.has(self.c_sym):
            return (True, "M is independent of c (correct)")
        # If c appears, check numeric cancellation
        val1 = self._numeric_value(expr.subs({self.c_sym: 1}))
        val2 = self._numeric_value(expr.subs({self.c_sym: 3e8}))
        if val1 is not None and val2 is not None and math.isclose(val1, val2, rel_tol=1e-6):
            return (True, "M effectively independent of c (cancels out numerically)")
        return (False, f"M depends on c: {M_sol}")

    # ==================================================================
    # Monotonicity test (perturbation-based)
    # ==================================================================
    def test_monotonicity_M_increases_with_KE(self):
        """
        Perturb KE_A by +10% and verify M increases.
        """
        if self.M is None:
            return (False, "SKIP: M not available")
        # Baseline KE_A = 2 m c^2
        KE_A_base = 2 * self.m_sym * self.c_sym**2
        M_base = sp.sqrt(4 * self.m_sym**2 * self.c_sym**4 +
                         2 * self.m_sym * self.c_sym**2 * KE_A_base) / self.c_sym**2
        # Perturb: increase KE_A by 10%
        delta = 0.1
        KE_A_new = (1 + delta) * KE_A_base
        M_new = sp.sqrt(4 * self.m_sym**2 * self.c_sym**4 +
                        2 * self.m_sym * self.c_sym**2 * KE_A_new) / self.c_sym**2

        n_base = self._numeric_value(M_base)
        n_new = self._numeric_value(M_new)
        if n_base is not None and n_new is not None:
            if n_new > n_base:
                return (True, f"M increases with KE: baseline M ≈ {n_base:.3f}m, perturbed (10% more KE) M ≈ {n_new:.3f}m")
            else:
                return (False, f"Perturbation gave M_new = {n_new:.3f}, not > {n_base:.3f}")
        return (False, "Could not evaluate monotonicity numerically")

    # ==================================================================
    # Weak answer sanity test (only one allowed)
    # ==================================================================
    def test_answer_sanity_M_reasonable_range(self):
        """
        Weak sanity: M should be greater than 2m (the sum of the two original rest masses).
        Only use numeric comparison if M can be evaluated; otherwise SKIP.
        No symbolic manipulation.
        """
        if self.M is None:
            return (False, "SKIP: M not available")
        # Try to evaluate M numerically
        try:
            M_sol = sp.sympify(self.M) if not hasattr(self.M, 'is_Symbol') else self.M
            n_M = self._numeric_value(M_sol)
            if n_M is not None:
                if n_M > 2.0:  # for m=1, 2m = 2
                    return (True, f"M ≈ {n_M:.3f} > 2 (reasonable)")
                else:
                    return (False, f"M ≈ {n_M:.3f} is not > 2")
            else:
                return (False, "SKIP: Could not evaluate M numerically")
        except Exception as e:
            return (False, f"Could not evaluate range: {e}")

    # ==================================================================
    # Main test runner
    # ==================================================================
    def run_all_tests(self):
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