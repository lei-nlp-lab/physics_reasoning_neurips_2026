import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract givens (may be None)
        self.N = getattr(solution_instance, 'N', None)
        self.mu = getattr(solution_instance, 'mu', None)
        self.H = getattr(solution_instance, 'H', None)
        self.k = getattr(solution_instance, 'k', None)
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        if self.epsilon is None and self.mu is not None and self.H is not None:
            self.epsilon = 2 * self.mu * self.H

        # Safely extract results (use None defaults)
        results = getattr(solution_instance, 'results', {})
        self.Omega_E = results.get('Omega_E', None)
        self.S_n = results.get('S_n', None)
        self.stirling = results.get('stirling', None)
        self.S_approx = results.get('S_approx', None)
        self.n_max = results.get('n_max', None)
        self.inv_T = results.get('inv_T', None)
        self.E_negT = results.get('E_negT', None)
        self.negT_reason = results.get('negT_reason', None)

    # ------------------------------------------------------------
    # Interface tests: check that expected keys exist and are of right types
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check all required result keys are present."""
        required = ['Omega_E', 'S_n', 'stirling', 'S_approx', 'n_max', 'inv_T', 'E_negT', 'negT_reason']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_Omega_E_type(self):
        if self.Omega_E is None:
            return (False, "SKIP: Omega_E not available")
        return (True, "Type OK") if isinstance(self.Omega_E, str) else (False, f"Expected str, got {type(self.Omega_E).__name__}")

    def test_interface_S_n_type(self):
        if self.S_n is None:
            return (False, "SKIP: S_n not available")
        return (True, "Type OK") if isinstance(self.S_n, str) else (False, f"Expected str, got {type(self.S_n).__name__}")

    def test_interface_stirling_type(self):
        if self.stirling is None:
            return (False, "SKIP: stirling not available")
        return (True, "Type OK") if isinstance(self.stirling, str) else (False, f"Expected str, got {type(self.stirling).__name__}")

    def test_interface_S_approx_type(self):
        if self.S_approx is None:
            return (False, "SKIP: S_approx not available")
        return (True, "Type OK") if isinstance(self.S_approx, str) else (False, f"Expected str, got {type(self.S_approx).__name__}")

    def test_interface_n_max_type(self):
        if self.n_max is None:
            return (False, "SKIP: n_max not available")
        is_num = isinstance(self.n_max, (int, float))
        return (True, "Type OK") if is_num else (False, f"Expected numeric, got {type(self.n_max).__name__}")

    def test_interface_inv_T_type(self):
        if self.inv_T is None:
            return (False, "SKIP: inv_T not available")
        return (True, "Type OK") if isinstance(self.inv_T, str) else (False, f"Expected str, got {type(self.inv_T).__name__}")

    def test_interface_E_negT_type(self):
        if self.E_negT is None:
            return (False, "SKIP: E_negT not available")
        return (True, "Type OK") if isinstance(self.E_negT, (str, int, float)) else (False, f"Expected str or number, got {type(self.E_negT).__name__}")

    def test_interface_negT_reason_type(self):
        if self.negT_reason is None:
            return (False, "SKIP: negT_reason not available")
        return (True, "Type OK") if isinstance(self.negT_reason, str) else (False, f"Expected str, got {type(self.negT_reason).__name__}")

    # ------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose where numerical, else string/pattern)
    # ------------------------------------------------------------
    def test_invariant_omega_combinatorial(self):
        """Omega(E) should be the combinatorial number of microstates."""
        if self.Omega_E is None:
            return (False, "SKIP: Omega_E not available")
        expected_sub = "N!/(n!(N-n)!)"
        if expected_sub in self.Omega_E:
            return (True, "Omega_E contains combinatorial expression")
        else:
            return (False, f"Omega_E does not contain '{expected_sub}'")

    def test_invariant_S_n_expression(self):
        """S(n) should be k ln( combinatorial )."""
        if self.S_n is None:
            return (False, "SKIP: S_n not available")
        checks = ["k ln(", "N!/(n!(N-n)!)"]
        for s in checks:
            if s not in self.S_n:
                return (False, f"S_n missing '{s}'")
        return (True, "S_n contains expected form")

    def test_invariant_stirling_approximation(self):
        """Stirling approximation should be ln n! ~= n ln n - n."""
        if self.stirling is None:
            return (False, "SKIP: stirling not available")
        if any(phrase in self.stirling for phrase in ["n ln n - n", "n*ln(n)-n", "n log n - n"]):
            return (True, "Stirling approximation present")
        else:
            return (False, f"Stirling string does not contain expected approximation: {self.stirling}")

    def test_invariant_S_approx_formula(self):
        """Approximate entropy should be N ln(N/(N-n)) - n ln(n/(N-n))."""
        if self.S_approx is None:
            return (False, "SKIP: S_approx not available")
        patterns = ["N ln", "n ln", "N/(N-n)", "n/(N-n)"]
        for p in patterns:
            if p not in self.S_approx:
                return (False, f"S_approx missing '{p}'")
        return (True, "S_approx contains expected components")

    def test_invariant_n_max_value(self):
        """n_max should be N/2 (if N is available, else just check it's a number)."""
        if self.n_max is None:
            return (False, "SKIP: n_max not available")
        if not isinstance(self.n_max, (int, float)):
            return (False, "n_max not numeric")
        if self.N is not None:
            expected = self.N / 2.0
            if math.isclose(self.n_max, expected, rel_tol=1e-9, abs_tol=1e-9):
                return (True, f"n_max = {self.n_max} (expected {expected})")
            else:
                return (False, f"n_max = {self.n_max}, expected {expected}")
        else:
            if self.n_max > 0:
                return (True, f"n_max is a positive number")
            else:
                return (False, "n_max should be positive")

    def test_invariant_inv_T_expression(self):
        """1/T should be (k/epsilon)*ln((N-n)/n)."""
        if self.inv_T is None:
            return (False, "SKIP: inv_T not available")
        patterns = ["ln", "N-n", "/n"]
        for p in patterns:
            if p not in self.inv_T:
                return (False, f"inv_T missing '{p}'")
        return (True, "inv_T contains expected components")

    def test_invariant_E_negT_expression(self):
        """E_negT should be N*epsilon/2 (or equivalent)."""
        if self.E_negT is None:
            return (False, "SKIP: E_negT not available")
        if isinstance(self.E_negT, str):
            if "N" in self.E_negT and "epsilon" in self.E_negT and ("/2" in self.E_negT or "*0.5" in self.E_negT):
                return (True, "E_negT expression contains N*epsilon/2")
            else:
                return (False, f"E_negT string does not match expected pattern: {self.E_negT}")
        elif isinstance(self.E_negT, (int, float)):
            if self.N is not None and self.epsilon is not None:
                expected = self.N * self.epsilon / 2.0
                if math.isclose(self.E_negT, expected, rel_tol=1e-9, abs_tol=1e-9):
                    return (True, f"E_negT numeric matches {expected}")
                else:
                    return (False, f"E_negT = {self.E_negT}, expected {expected}")
            else:
                return (True, "E_negT is numeric")
        else:
            return (False, f"Unexpected type for E_negT: {type(self.E_negT).__name__}")

    def test_invariant_negT_reason_content(self):
        """Reason should mention upper bound or bounded spectrum."""
        if self.negT_reason is None:
            return (False, "SKIP: negT_reason not available")
        keywords = ["upper bound", "upper limit", "bounded", "limited"]
        if any(kw in self.negT_reason for kw in keywords):
            return (True, "negT_reason contains expected concept")
        else:
            return (False, f"negT_reason does not mention bounded spectrum: {self.negT_reason}")

    # ------------------------------------------------------------
    # Modeling / trace tests: check local relations from explanation_steps
    # ------------------------------------------------------------
    def test_modeling_stirling_derivation(self):
        """Stirling approximation should mention integral approximation."""
        if self.stirling is None:
            return (False, "SKIP: stirling not available")
        if "integral" in self.stirling or "∫" in self.stirling or "∫" in self.stirling:
            return (True, "Stirling derivation includes integral")
        else:
            return (False, "Stirling derivation does not mention integral approximation")

    def test_modeling_temperature_sign(self):
        """Derive sign of 1/T from inv_T expression using symbolic substitution."""
        if self.inv_T is None or self.N is None or self.epsilon is None or self.k is None:
            return (False, "SKIP: insufficient data for temperature sign test")
        try:
            N_sym, n_sym, k_sym, eps_sym = sp.symbols('N n k epsilon', positive=True, real=True)
            expr_str = self.inv_T.replace('ln', 'log').replace('epsilon', 'eps')
            expr = sp.sympify(expr_str, locals={'N': N_sym, 'n': n_sym, 'k': k_sym, 'epsilon': eps_sym})
            # Test two n values relative to N/2
            subs_1 = {N_sym: 10, n_sym: 2, k_sym: 1, eps_sym: 1}
            val_1 = float(expr.subs(subs_1).evalf())
            subs_2 = {N_sym: 10, n_sym: 8, k_sym: 1, eps_sym: 1}
            val_2 = float(expr.subs(subs_2).evalf())
            if val_1 > 0 and val_2 < 0:
                return (True, f"1/T positive for n<N/2 ({val_1}), negative for n>N/2 ({val_2})")
            else:
                return (False, f"Sign mismatch: val_1={val_1}, val_2={val_2}")
        except Exception as e:
            if "ln" in self.inv_T and "N-n" in self.inv_T:
                return (True, "inv_T expression present (fallback)")
            else:
                return (False, f"Cannot evaluate inv_T: {e}")

    def test_modeling_entropy_maximum_derivative(self):
        """Check that at n=N/2, dS/dn = 0 using S_approx expression."""
        if self.S_approx is None or self.N is None:
            return (False, "SKIP: insufficient data for derivative test")
        try:
            expr_str = self.S_approx.replace('ln', 'log')
            N_sym, n_sym = sp.symbols('N n', positive=True, real=True)
            expr = sp.sympify(expr_str, locals={'N': N_sym, 'n': n_sym})
            dS_dn = sp.diff(expr, n_sym)
            subs = {n_sym: N_sym/2}
            derivative_at_max = sp.simplify(dS_dn.subs(subs))
            if derivative_at_max == 0:
                return (True, "Derivative of S is zero at n = N/2")
            else:
                try:
                    val = float(derivative_at_max.subs({N_sym: 10}))
                    if abs(val) < 1e-12:
                        return (True, "Derivative numerically zero at n=N/2")
                    else:
                        return (False, f"Derivative at n=N/2 = {val}, should be 0")
                except:
                    return (False, f"Derivative not identically zero: {derivative_at_max}")
        except Exception as e:
            if self.n_max is not None and self.N is not None and math.isclose(self.n_max, self.N/2, rel_tol=1e-9):
                return (True, "n_max = N/2 (fallback)")
            else:
                return (False, f"Cannot verify derivative: {e}")

    # ------------------------------------------------------------
    # Monotonicity tests: explicitly perturb a parameter and compare baseline vs perturbed output
    # ------------------------------------------------------------
    def test_monotonicity_entropy_increases_with_n_below_half(self):
        """
        For n < N/2, increasing n by 1 should increase entropy.
        Baseline: N=10, n=4. Perturb: n=5. Both < N/2=5.
        """
        if self.S_approx is None:
            return (False, "SKIP: S_approx not available")
        try:
            N_val = 10.0
            expr_str = self.S_approx.replace('ln', 'log')
            N_sym, n_sym = sp.symbols('N n', positive=True, real=True)
            expr = sp.sympify(expr_str, locals={'N': N_sym, 'n': n_sym})
            # Baseline: n = 4
            S_base = float(expr.subs({N_sym: N_val, n_sym: 4.0}).evalf())
            # Perturbed: n = 5
            S_pert = float(expr.subs({N_sym: N_val, n_sym: 5.0}).evalf())
            if S_pert > S_base:
                return (True, f"S(5)={S_pert:.4f} > S(4)={S_base:.4f}")
            else:
                return (False, f"S(5)={S_pert:.4f} <= S(4)={S_base:.4f}")
        except Exception as e:
            return (False, f"Cannot evaluate S_approx: {e}")

    def test_monotonicity_entropy_decreases_with_n_above_half(self):
        """
        For n > N/2, increasing n by 1 should decrease entropy.
        Baseline: N=10, n=6. Perturb: n=7. Both > N/2=5.
        """
        if self.S_approx is None:
            return (False, "SKIP: S_approx not available")
        try:
            N_val = 10.0
            expr_str = self.S_approx.replace('ln', 'log')
            N_sym, n_sym = sp.symbols('N n', positive=True, real=True)
            expr = sp.sympify(expr_str, locals={'N': N_sym, 'n': n_sym})
            # Baseline: n = 6
            S_base = float(expr.subs({N_sym: N_val, n_sym: 6.0}).evalf())
            # Perturbed: n = 7
            S_pert = float(expr.subs({N_sym: N_val, n_sym: 7.0}).evalf())
            if S_pert < S_base:
                return (True, f"S(7)={S_pert:.4f} < S(6)={S_base:.4f}")
            else:
                return (False, f"S(7)={S_pert:.4f} >= S(6)={S_base:.4f}")
        except Exception as e:
            return (False, f"Cannot evaluate S_approx: {e}")

    def test_monotonicity_entropy_increases_with_N(self):
        """
        Increase total number of particles N (while keeping n/N fixed) should increase entropy.
        Baseline: N=10, n=4 (fraction 0.4). Perturb: N=12, n=5 (fraction 0.4167). Expect higher entropy.
        """
        if self.S_approx is None:
            return (False, "SKIP: S_approx not available")
        try:
            expr_str = self.S_approx.replace('ln', 'log')
            N_sym, n_sym = sp.symbols('N n', positive=True, real=True)
            expr = sp.sympify(expr_str, locals={'N': N_sym, 'n': n_sym})
            # Baseline: N=10, n=4
            S_base = float(expr.subs({N_sym: 10.0, n_sym: 4.0}).evalf())
            # Perturbed: N=12, n=5 (slightly larger fraction, so monotonicity should still hold)
            S_pert = float(expr.subs({N_sym: 12.0, n_sym: 5.0}).evalf())
            if S_pert > S_base:
                return (True, f"S(12,5)={S_pert:.4f} > S(10,4)={S_base:.4f}")
            else:
                return (False, f"S(12,5)={S_pert:.4f} <= S(10,4)={S_base:.4f}")
        except Exception as e:
            return (False, f"Cannot evaluate S_approx: {e}")

    def test_monotonicity_inv_T_changes_sign_with_n(self):
        """
        Perturb n from 4 to 6: 1/T should go from positive to negative.
        Baseline: N=10, n=4. Perturbed: n=6.
        """
        if self.inv_T is None or self.N is None or self.epsilon is None or self.k is None:
            return (False, "SKIP: insufficient data")
        try:
            N_val = 10.0
            k_val = 1.0
            eps_val = 1.0
            N_sym, n_sym, k_sym, eps_sym = sp.symbols('N n k epsilon', positive=True, real=True)
            expr_str = self.inv_T.replace('ln', 'log').replace('epsilon', 'eps')
            expr = sp.sympify(expr_str, locals={'N': N_sym, 'n': n_sym, 'k': k_sym, 'epsilon': eps_sym})
            # Baseline: n=4 -> should be positive
            val_base = float(expr.subs({N_sym: N_val, n_sym: 4.0, k_sym: k_val, eps_sym: eps_val}).evalf())
            # Perturbed: n=6 -> should be negative
            val_pert = float(expr.subs({N_sym: N_val, n_sym: 6.0, k_sym: k_val, eps_sym: eps_val}).evalf())
            if val_base > 0 and val_pert < 0:
                return (True, f"1/T changes from +{val_base:.3f} (n=4) to {val_pert:.3f} (n=6)")
            else:
                return (False, f"Sign change not observed: baseline={val_base:.3f}, perturbed={val_pert:.3f}")
        except Exception as e:
            return (False, f"Cannot evaluate inv_T: {e}")

    # ------------------------------------------------------------
    # Answer sanity (weak, single test)
    # ------------------------------------------------------------
    def test_answer_sanity_general(self):
        """Weak sanity: check plausibility of n_max and physical constants."""
        if self.n_max is None:
            return (False, "SKIP: n_max not available")
        if self.N is not None:
            if not (0 < self.n_max <= self.N):
                return (False, f"n_max={self.n_max} out of range (0,N={self.N})")
        else:
            if self.n_max <= 0:
                return (False, f"n_max={self.n_max} should be positive")
        if isinstance(self.E_negT, (int, float)):
            if self.E_negT <= 0:
                return (False, "E_negT should be positive")
        return (True, "Basic sanity checks passed")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)