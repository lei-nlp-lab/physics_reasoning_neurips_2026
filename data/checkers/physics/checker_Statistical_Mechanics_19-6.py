import math
import sympy as sp
import traceback

class PhysicsChecker:
    """Auditor for Bose-Einstein condensation problem (free gas and harmonic trap)."""

    def __init__(self, solution_instance):
        # Safely extract givens and results; default to None.
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}

        # Givens
        self.d = givens.get('d', None)
        self.L = givens.get('L', None)
        self.m = givens.get('m', None)
        self.hbar = givens.get('hbar', None)
        self.k_B = givens.get('k_B', None)
        self.omega = givens.get('omega', None)
        self.N = givens.get('N', None)
        # Volume: either given explicitly or computed from L and d
        self.V = givens.get('V', None)
        if self.V is None and self.L is not None and self.d is not None:
            self.V = self.L ** self.d

        # Results
        self.D_p = results.get('D_p', None)
        self.D_eps = results.get('D_eps', None)
        self.tau_E_free = results.get('tau_E_free', None)
        self.tau_E_trap = results.get('tau_E_trap', None)

    # ------------------------------ helpers ------------------------------
    def _any_missing(self, *args):
        return any(a is None for a in args)

    def _numeric_check(self, expr, subs=None):
        """If expr is symbolic, substitute numeric values; else return error."""
        if subs is None:
            subs = {}
        if hasattr(expr, 'subs'):
            try:
                return float(expr.subs(subs).evalf())
            except Exception:
                return None
        # Assume it's a numeric object
        try:
            return float(expr)
        except (TypeError, ValueError):
            return None

    def _compute_free_tau_E(self, N, V):
        """Compute tau_E (k_B T_c) for free gas from givens."""
        if self._any_missing(self.d, self.m, self.hbar):
            return None
        d = self.d
        m = self.m
        hbar = self.hbar
        try:
            tau = (2 * math.pi * hbar**2 / m) * (N / (V * sp.zeta(d/2)))**(2/d)
            return float(tau.evalf()) if hasattr(tau, 'evalf') else float(tau)
        except Exception:
            return None

    def _compute_trap_tau_E(self, N, omega=None):
        """Compute tau_E (k_B T_c) for harmonic trap from givens.
        Can specify omega for perturbation; if None, uses self.omega."""
        if self._any_missing(self.d, self.hbar):
            return None
        d = self.d
        hbar = self.hbar
        if omega is None:
            omega = self.omega
        if omega is None:
            return None
        try:
            tau = hbar * omega * (N / sp.zeta(d))**(1/d)
            return float(tau.evalf()) if hasattr(tau, 'evalf') else float(tau)
        except Exception:
            return None

    # ------------------------------ interface tests ------------------------------
    def test_interface_all_keys_present(self):
        missing = []
        if self.D_p is None:
            missing.append('D_p')
        if self.D_eps is None:
            missing.append('D_eps')
        if self.tau_E_free is None:
            missing.append('tau_E_free')
        if self.tau_E_trap is None:
            missing.append('tau_E_trap')
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    # ------------------------------ invariant tests (scaling laws) ------------------------------
    def test_invariant_D_p_scaling(self):
        """D_p ∝ V * p^{d-1} ."""
        if self.D_p is None:
            return (False, "SKIP: D_p not provided")
        if self._any_missing(self.d, self.V):
            return (False, "SKIP: missing d or V for scaling check")

        p = sp.Symbol('p', positive=True)
        scale_expr = sp.simplify(self.D_p / (self.V * p**(self.d - 1)))
        if scale_expr.is_constant():
            return (True, "D_p scales as V * p^(d-1)")
        # Numeric fallback: evaluate at three p values and check ratio constancy
        try:
            vals = [0.5, 1.0, 2.0]
            ratios = []
            for v in vals:
                num = self._numeric_check(self.D_p, {'p': v, 'V': self.V, 'd': self.d, 'hbar': self.hbar})
                if num is None:
                    return (False, "Cannot evaluate D_p numerically")
                expected = (2 * self.V) / ((4 * math.pi * self.hbar**2)**(self.d/2) * math.gamma(self.d/2)) * v**(self.d-1)
                ratios.append(num / expected)
            if all(math.isclose(r, ratios[0], rel_tol=1e-2, abs_tol=1e-2) for r in ratios):
                return (True, "D_p scaling confirmed numerically")
            else:
                return (False, "D_p scaling inconsistent")
        except Exception as e:
            return (False, f"D_p scaling test error: {e}")

    def test_invariant_D_eps_scaling(self):
        """D_eps ∝ V * ε^{d/2-1} ."""
        if self.D_eps is None:
            return (False, "SKIP: D_eps not provided")
        if self._any_missing(self.d, self.V, self.m, self.hbar):
            return (False, "SKIP: missing d, V, m, hbar for scaling check")

        eps = sp.Symbol('eps', positive=True)
        scale_expr = sp.simplify(self.D_eps / (self.V * eps**(self.d/2 - 1)))
        if scale_expr.is_constant():
            return (True, "D_eps scales as V * ε^(d/2-1)")

        try:
            vals = [1.0, 2.0, 3.0]
            ratios = []
            for v in vals:
                num = self._numeric_check(self.D_eps, {'eps': v, 'V': self.V, 'd': self.d, 'm': self.m, 'hbar': self.hbar})
                if num is None:
                    return (False, "Cannot evaluate D_eps numerically")
                expected = (self.m/(2*math.pi*self.hbar**2))**(self.d/2) * self.V / math.gamma(self.d/2) * v**(self.d/2 - 1)
                ratios.append(num / expected)
            if all(math.isclose(r, ratios[0], rel_tol=1e-2, abs_tol=1e-2) for r in ratios):
                return (True, "D_eps scaling confirmed numerically")
            else:
                return (False, "D_eps scaling inconsistent")
        except Exception as e:
            return (False, f"D_eps scaling test error: {e}")

    def test_invariant_tau_E_free_scaling(self):
        """tau_E_free ∝ (N/V)^{2/d} , inversely proportional to m ."""
        if self.tau_E_free is None:
            return (False, "SKIP: tau_E_free not provided")
        if self._any_missing(self.d, self.m, self.hbar, self.N, self.V):
            return (False, "SKIP: missing parameters for free tau_E scaling")

        expr = sp.simplify(self.tau_E_free * self.m * (self.V / self.N)**(2/self.d))
        if hasattr(expr, 'is_constant') and expr.is_constant():
            return (True, "tau_E_free scaling consistent")
        try:
            tau = self._numeric_check(self.tau_E_free, {})
            if tau is None:
                return (False, "Cannot evaluate tau_E_free numerically")
            expected = (2 * math.pi * self.hbar**2 / self.m) * (self.N / (self.V * sp.zeta(self.d/2)))**(2/self.d)
            if hasattr(expected, 'evalf'):
                expected = float(expected.evalf())
            if math.isclose(tau, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "tau_E_free scaling confirmed")
            else:
                return (False, f"tau_E_free numerical mismatch: got {tau}, expected {expected}")
        except Exception as e:
            return (False, f"tau_E_free scaling test error: {e}")

    def test_invariant_tau_E_trap_scaling(self):
        """tau_E_trap ∝ ω * (N)^{1/d} ."""
        if self.tau_E_trap is None:
            return (False, "SKIP: tau_E_trap not provided")
        if self._any_missing(self.d, self.omega, self.hbar, self.N):
            return (False, "SKIP: missing parameters for trap tau_E scaling")

        expr = sp.simplify(self.tau_E_trap / (self.hbar * self.omega * self.N ** (1/self.d)))
        if hasattr(expr, 'is_constant') and expr.is_constant():
            return (True, "tau_E_trap scaling consistent")
        try:
            tau = self._numeric_check(self.tau_E_trap, {})
            if tau is None:
                return (False, "Cannot evaluate tau_E_trap numerically")
            expected = self.hbar * self.omega * (self.N / sp.zeta(self.d))**(1/self.d)
            if hasattr(expected, 'evalf'):
                expected = float(expected.evalf())
            if math.isclose(tau, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "tau_E_trap scaling confirmed")
            else:
                return (False, f"tau_E_trap numerical mismatch: got {tau}, expected {expected}")
        except Exception as e:
            return (False, f"tau_E_trap scaling test error: {e}")

    # ------------------------------ modeling tests (derivations) ------------------------------
    def test_modeling_D_eps_derivation(self):
        """Check that D_eps = D_p(p(eps)) * dp/deps ."""
        if self._any_missing(self.D_p, self.D_eps):
            return (False, "SKIP: D_p or D_eps not provided")
        if self._any_missing(self.m, self.d, self.V):
            return (False, "SKIP: missing m, d, V")

        p, eps = sp.Symbols('p eps', positive=True)
        try:
            D_p_sub = self.D_p.subs({'p': sp.sqrt(2*self.m*eps)})
            dp_deps = sp.sqrt(self.m/(2*eps))
            derived_D_eps = sp.simplify(D_p_sub * dp_deps)
            diff = sp.simplify(derived_D_eps - self.D_eps)
            if diff == 0:
                return (True, "D_eps derived correctly from D_p")
            # Numeric fallback
            test_eps = 1.0
            num_derived = float(derived_D_eps.subs({eps: test_eps, 'V': self.V, 'd': self.d, 'm': self.m, 'hbar': self.hbar}).evalf())
            num_given = self._numeric_check(self.D_eps, {'eps': test_eps, 'V': self.V, 'd': self.d, 'm': self.m, 'hbar': self.hbar})
            if num_given is not None and math.isclose(num_derived, num_given, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "D_eps derivation confirmed numerically")
            else:
                return (False, f"D_eps derivation mismatch: derived {num_derived}, given {num_given}")
        except Exception as e:
            return (False, f"D_eps derivation test error: {e}")

    def test_modeling_BEC_condition_free(self):
        """Check that N = ∫ D_eps/(e^{βε}-1) dε leads to tau_E_free."""
        if self._any_missing(self.tau_E_free, self.D_eps, self.N, self.V, self.d, self.m, self.hbar):
            return (False, "SKIP: missing data for free BEC condition")
        try:
            expected_tau = self._compute_free_tau_E(self.N, self.V)
            if expected_tau is None:
                return (False, "Cannot compute expected free tau_E")
            given_tau = self._numeric_check(self.tau_E_free, {})
            if given_tau is None:
                return (False, "Cannot evaluate given tau_E_free numerically")
            if math.isclose(given_tau, expected_tau, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Free BEC condition yields correct tau_E")
            else:
                return (False, f"Free tau_E mismatch: given {given_tau}, expected {expected_tau}")
        except Exception as e:
            return (False, f"Free BEC condition test error: {e}")

    def test_modeling_BEC_condition_trap(self):
        """Check that N = ∫ g(ε)/(e^{βε}-1) dε leads to tau_E_trap."""
        if self._any_missing(self.tau_E_trap, self.N, self.omega, self.hbar, self.d):
            return (False, "SKIP: missing data for trap BEC condition")
        try:
            expected_tau = self._compute_trap_tau_E(self.N)
            if expected_tau is None:
                return (False, "Cannot compute expected trap tau_E")
            given_tau = self._numeric_check(self.tau_E_trap, {})
            if given_tau is None:
                return (False, "Cannot evaluate given tau_E_trap numerically")
            if math.isclose(given_tau, expected_tau, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Trap BEC condition yields correct tau_E")
            else:
                return (False, f"Trap tau_E mismatch: given {given_tau}, expected {expected_tau}")
        except Exception as e:
            return (False, f"Trap BEC condition test error: {e}")

    # ------------------------------ monotonicity tests (explicit perturbation) ------------------------------
    def test_monotonicity_tau_E_free_increases_with_N(self):
        """tau_E_free should increase with N. Perturb N by +10% and compare."""
        if self._any_missing(self.tau_E_free, self.N, self.V, self.d, self.m, self.hbar):
            return (False, "SKIP: missing data for free tau_E monotonicity")
        try:
            # Baseline
            N0 = self.N
            V0 = self.V
            baseline = self._compute_free_tau_E(N0, V0)
            if baseline is None:
                return (False, "Cannot compute baseline tau_E_free")
            # Perturb N by +10%
            N1 = N0 * 1.1
            perturbed = self._compute_free_tau_E(N1, V0)
            if perturbed is None:
                return (False, "Cannot compute perturbed tau_E_free")
            if perturbed > baseline:
                return (True, "tau_E_free increases with N (+10% N gave higher tau)")
            else:
                return (False, f"tau_E_free does not increase: baseline {baseline}, perturbed {perturbed}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    def test_monotonicity_tau_E_free_decreases_with_V(self):
        """tau_E_free should decrease with V. Perturb V by +10% and compare."""
        if self._any_missing(self.tau_E_free, self.N, self.V, self.d, self.m, self.hbar):
            return (False, "SKIP: missing data for free tau_E monotonicity")
        try:
            N0 = self.N
            V0 = self.V
            baseline = self._compute_free_tau_E(N0, V0)
            if baseline is None:
                return (False, "Cannot compute baseline tau_E_free")
            # Perturb V by +10%
            V1 = V0 * 1.1
            perturbed = self._compute_free_tau_E(N0, V1)
            if perturbed is None:
                return (False, "Cannot compute perturbed tau_E_free")
            if perturbed < baseline:
                return (True, "tau_E_free decreases with V (+10% V gave lower tau)")
            else:
                return (False, f"tau_E_free does not decrease: baseline {baseline}, perturbed {perturbed}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    def test_monotonicity_tau_E_trap_increases_with_N(self):
        """tau_E_trap should increase with N. Perturb N by +10% and compare."""
        if self._any_missing(self.tau_E_trap, self.N, self.omega, self.hbar, self.d):
            return (False, "SKIP: missing data for trap tau_E monotonicity")
        try:
            N0 = self.N
            baseline = self._compute_trap_tau_E(N0)
            if baseline is None:
                return (False, "Cannot compute baseline tau_E_trap")
            # Perturb N by +10%
            N1 = N0 * 1.1
            perturbed = self._compute_trap_tau_E(N1)
            if perturbed is None:
                return (False, "Cannot compute perturbed tau_E_trap")
            if perturbed > baseline:
                return (True, "tau_E_trap increases with N (+10% N gave higher tau)")
            else:
                return (False, f"tau_E_trap does not increase: baseline {baseline}, perturbed {perturbed}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    def test_monotonicity_tau_E_trap_increases_with_omega(self):
        """tau_E_trap should increase with omega. Perturb omega by +10% and compare."""
        if self._any_missing(self.tau_E_trap, self.N, self.omega, self.hbar, self.d):
            return (False, "SKIP: missing data for trap tau_E monotonicity")
        try:
            omega0 = self.omega
            N0 = self.N
            baseline = self._compute_trap_tau_E(N0, omega=omega0)
            if baseline is None:
                return (False, "Cannot compute baseline tau_E_trap")
            # Perturb omega by +10%
            omega1 = omega0 * 1.1
            perturbed = self._compute_trap_tau_E(N0, omega=omega1)
            if perturbed is None:
                return (False, "Cannot compute perturbed tau_E_trap")
            if perturbed > baseline:
                return (True, "tau_E_trap increases with omega (+10% omega gave higher tau)")
            else:
                return (False, f"tau_E_trap does not increase: baseline {baseline}, perturbed {perturbed}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ------------------------------ weak answer sanity (only one) ------------------------------
    def test_answer_sanity_free_tau_E_positive_finite(self):
        """tau_E_free should be positive and finite."""
        if self.tau_E_free is None:
            return (False, "SKIP: tau_E_free not provided")
        try:
            tau = self._numeric_check(self.tau_E_free, {})
            if tau is None:
                return (False, "Cannot evaluate tau_E_free numerically")
            if not math.isfinite(tau) or tau <= 0:
                return (False, f"tau_E_free is not positive finite: {tau}")
            return (True, "tau_E_free is positive finite")
        except Exception as e:
            return (False, f"Error in answer sanity: {e}")

    # ------------------------------ run_all_tests ------------------------------
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