import math
import traceback

class PhysicsChecker:
    """
    Auditor for quantum atom–particle interaction problem.
    Checks interface completeness, physics invariants, modeling consistency,
    monotonicity trends, and a weak final-answer sanity.
    """
    def __init__(self, solution_instance):
        # -------- extract givens safely --------
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.q = givens.get('q')
        self.r12 = givens.get('r12')
        self.hbar = givens.get('hbar')
        self.alpha = givens.get('alpha')
        self.omega0 = givens.get('omega0')
        self.tau = givens.get('tau')
        self.N = givens.get('N')
        # optional: t0, t (for dipole moment evaluation)
        self.t0 = givens.get('t0')
        self.t = givens.get('t')

        # -------- extract results safely --------
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.lambda_ = results.get('lambda')
        self.P2 = results.get('P2')
        self.langle_r = results.get('langle_r')
        self.P2_N = results.get('P2_N')

        # -------- extract explanation steps / pseudocode (optional) --------
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ---------- helper: safe numeric fetch ----------
    def _get_numeric(self, val, name):
        if val is None:
            raise ValueError(f"SKIP: {name} not available")
        return val

    # ========== INTERFACE TESTS ==========
    def test_interface_givens_exist(self):
        """All required givens must be present."""
        required = ['q', 'r12', 'hbar', 'alpha', 'omega0', 'tau', 'N']
        missing = [r for r in required if getattr(self, r) is None]
        if missing:
            return False, f"Missing givens: {missing}"
        return True, "All required givens present"

    def test_interface_results_exist(self):
        """All expected result keys must exist."""
        required = ['lambda', 'P2', 'langle_r', 'P2_N']
        missing = []
        for r in required:
            if getattr(self, r) is None:
                missing.append(r)
        if missing:
            return False, f"Missing results: {missing}"
        return True, "All required results present"

    def test_interface_givens_numeric(self):
        """All givens should be numbers (not strings or None)."""
        for name in ['q','r12','hbar','alpha','omega0','tau','N']:
            val = getattr(self, name)
            if val is None:
                return False, f"Given {name} is missing"
            if not isinstance(val, (int, float)):
                return False, f"Given {name} is not numeric: {type(val).__name__}"
        return True, "All givens are numeric"

    def test_interface_results_numeric(self):
        """All result values should be numbers."""
        for name in ['lambda','P2','langle_r','P2_N']:
            val = getattr(self, name)
            if val is None:
                continue   # skip missing – will be caught by existence test
            if not isinstance(val, (int, float)):
                return False, f"Result {name} is not numeric: {type(val).__name__}"
        return True, "All present results are numeric"

    # ========== INVARIANT TESTS (use math.isclose) ==========
    def test_invariant_lambda_positive(self):
        """lambda should be non‑negative (absolute value)."""
        try:
            lam = self._get_numeric(self.lambda_, 'lambda')
        except ValueError as e:
            return False, str(e)
        if lam < 0:
            return False, f"lambda = {lam} is negative"
        return True, f"lambda = {lam} >= 0"

    def test_invariant_P2_nonnegative(self):
        """P2 (probability) must be ≥ 0."""
        try:
            p2 = self._get_numeric(self.P2, 'P2')
        except ValueError as e:
            return False, str(e)
        if p2 < 0:
            return False, f"P2 = {p2} is negative"
        return True, f"P2 = {p2} >= 0"

    def test_invariant_P2_N_nonnegative(self):
        """P2_N (probability) must be ≥ 0."""
        try:
            p2n = self._get_numeric(self.P2_N, 'P2_N')
        except ValueError as e:
            return False, str(e)
        if p2n < 0:
            return False, f"P2_N = {p2n} is negative"
        return True, f"P2_N = {p2n} >= 0"

    # ========== MODELING TESTS (use math.isclose) ==========
    def test_modeling_lambda_formula(self):
        """Check lambda = |q * r12 * alpha| / hbar."""
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
        except ValueError as e:
            return False, str(e)
        expected = abs(q * r12 * alpha) / hbar
        try:
            lam = self._get_numeric(self.lambda_, 'lambda')
        except ValueError as e:
            return False, str(e)
        if math.isclose(lam, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"lambda={lam} matches expected={expected}"
        return False, f"lambda={lam} differs from expected={expected}"

    def test_modeling_P2_formula(self):
        """Check P2 = (q * r12 * alpha / hbar)^2."""
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
        except ValueError as e:
            return False, str(e)
        expected = (q * r12 * alpha / hbar) ** 2
        try:
            p2 = self._get_numeric(self.P2, 'P2')
        except ValueError as e:
            return False, str(e)
        if math.isclose(p2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"P2={p2} matches expected={expected}"
        return False, f"P2={p2} differs from expected={expected}"

    def test_modeling_langle_r_form(self):
        """
        Check that langle_r has the expected functional form:
        ⟨r⟩ = - (2 α q r12^2 / ℏ) sin(ω0 (t – t0)).
        This test is run only when t and t0 are provided.
        """
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
            omega0 = self._get_numeric(self.omega0, 'omega0')
            t = self._get_numeric(self.t, 't')
            t0 = self._get_numeric(self.t0, 't0')
        except ValueError as e:
            # t / t0 may not be provided; skip gracefully
            return False, "SKIP: t or t0 not available"
        expected = - (2 * alpha * q * r12**2 / hbar) * math.sin(omega0 * (t - t0))
        try:
            lr = self._get_numeric(self.langle_r, 'langle_r')
        except ValueError as e:
            return False, str(e)
        if math.isclose(lr, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"langle_r={lr} matches expected (t={t}, t0={t0})"
        return False, f"langle_r={lr} differs from expected={expected}"

    def test_modeling_P2_N_small_angle_approx(self):
        """
        When ω0 τ << 1, P2(Nτ) ≈ (N α q r12 / ℏ)^2.
        We check this approximation using the given values.
        """
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
            omega0 = self._get_numeric(self.omega0, 'omega0')
            tau = self._get_numeric(self.tau, 'tau')
            N = self._get_numeric(self.N, 'N')
        except ValueError as e:
            return False, str(e)

        # Check that small‑angle condition holds; if not, the test is less relevant.
        if omega0 * tau > 0.1:
            return False, f"SKIP: ω0 τ = {omega0*tau} not << 1, approximation not expected"

        expected_approx = (N * alpha * q * r12 / hbar) ** 2
        try:
            p2n = self._get_numeric(self.P2_N, 'P2_N')
        except ValueError as e:
            return False, str(e)
        if math.isclose(p2n, expected_approx, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"P2_N={p2n} matches approximated expected={expected_approx}"
        return False, f"P2_N={p2n} differs from approximated expected={expected_approx}"

    def test_modeling_P2_N_exact_from_sum(self):
        """
        Check the exact formula: P2_N = (α q r12 / ℏ)^2 * | Σ exp(i ω0 n τ) |^2.
        This is a stronger test but requires the exact geometric sum.
        Only run if we have numeric values and the small‑angle condition is not extreme.
        """
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
            omega0 = self._get_numeric(self.omega0, 'omega0')
            tau = self._get_numeric(self.tau, 'tau')
            N = self._get_numeric(self.N, 'N')
        except ValueError as e:
            return False, str(e)

        # Avoid division by zero for ω0 τ = 0
        if omega0 == 0 or tau == 0:
            return False, "SKIP: ω0 or τ zero, degenerate case"

        # Compute exact squared amplitude of the geometric sum
        import cmath
        S = sum(cmath.exp(1j * omega0 * n * tau) for n in range(N))   # n = 0..N-1
        exact = (alpha * q * r12 / hbar) ** 2 * abs(S) ** 2
        try:
            p2n = self._get_numeric(self.P2_N, 'P2_N')
        except ValueError as e:
            return False, str(e)
        if math.isclose(p2n, exact, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"P2_N={p2n} matches exact geometric sum value={exact}"
        return False, f"P2_N={p2n} differs from exact geometric sum value={exact}"

    # ========== MONOTONICITY TESTS ==========
    def test_monotonicity_P2_with_alpha(self):
        """Increasing alpha should increase P2 (monotonic increasing)."""
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
        except ValueError as e:
            return False, str(e)
        try:
            p2 = self._get_numeric(self.P2, 'P2')
        except ValueError as e:
            return False, str(e)

        # Perturb alpha by +10%
        delta = 0.1 * alpha
        alpha_pert = alpha + delta
        expected_pert = (q * r12 * alpha_pert / hbar) ** 2
        if expected_pert > p2:
            return True, f"P2 increases with alpha: {p2} -> {expected_pert} on +{delta:.4g}"
        else:
            return False, f"P2 did not increase with alpha: baseline {p2}, perturbed {expected_pert}"

    def test_monotonicity_lambda_with_alpha(self):
        """Increasing alpha should increase lambda."""
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
        except ValueError as e:
            return False, str(e)
        try:
            lam = self._get_numeric(self.lambda_, 'lambda')
        except ValueError as e:
            return False, str(e)

        delta = 0.1 * alpha
        alpha_pert = alpha + delta
        expected_pert = abs(q * r12 * alpha_pert / hbar)
        if expected_pert > lam:
            return True, f"lambda increases with alpha: {lam} -> {expected_pert} on +{delta:.4g}"
        else:
            return False, f"lambda did not increase with alpha: baseline {lam}, perturbed {expected_pert}"

    def test_monotonicity_P2_N_with_N(self):
        """Increasing N should increase P2_N (monotonic increasing) when ω0 τ small."""
        try:
            q = self._get_numeric(self.q, 'q')
            r12 = self._get_numeric(self.r12, 'r12')
            hbar = self._get_numeric(self.hbar, 'hbar')
            alpha = self._get_numeric(self.alpha, 'alpha')
            N = self._get_numeric(self.N, 'N')
        except ValueError as e:
            return False, str(e)
        try:
            p2n = self._get_numeric(self.P2_N, 'P2_N')
        except ValueError as e:
            return False, str(e)

        # Compute expected P2_N for N+1 (if N is integer)
        if not isinstance(N, int) and not isinstance(N, float) or N != int(N):
            return False, "SKIP: N is not an integer"
        N_int = int(N)
        N_pert = N_int + 1
        expected_pert = (N_pert * alpha * q * r12 / hbar) ** 2   # small‑angle approximation
        # but be robust: if omega0*tau not small, this approximation may break.
        # For safety, we compute using exact sum if possible.
        try:
            omega0 = self._get_numeric(self.omega0, 'omega0')
            tau = self._get_numeric(self.tau, 'tau')
            # Use exact sum for N_pert if omega0 not zero
            if omega0 != 0 and tau != 0:
                import cmath
                S_pert = sum(cmath.exp(1j * omega0 * n * tau) for n in range(N_pert))
                expected_pert = (alpha * q * r12 / hbar) ** 2 * abs(S_pert) ** 2
            else:
                expected_pert = (N_pert * alpha * q * r12 / hbar) ** 2
        except ValueError:
            # tau or omega0 missing -> use approximation
            expected_pert = (N_pert * alpha * q * r12 / hbar) ** 2

        if expected_pert > p2n:
            return True, f"P2_N increases with N: {p2n} -> {expected_pert} (N={N_int} -> {N_pert})"
        else:
            return False, f"P2_N did not increase with N: baseline {p2n}, perturbed {expected_pert}"

    # ========== FINAL ANSWER SANITY (weak) ==========
    def test_answer_sanity_general(self):
        """
        Weak sanity: all results should be finite real numbers.
        No math.isclose, no re‑derivation.
        """
        issues = []
        for name in ['lambda', 'P2', 'langle_r', 'P2_N']:
            val = getattr(self, name)
            if val is None:
                issues.append(f"{name} is missing")
                continue
            if not isinstance(val, (int, float)):
                issues.append(f"{name} is not a number")
                continue
            if math.isnan(val):
                issues.append(f"{name} is NaN")
            if math.isinf(val):
                issues.append(f"{name} is infinite")
        if issues:
            return False, " | ".join(issues)
        return True, "All present results are finite and numeric"

    # ========== MAIN RUNNER ==========
    def run_all_tests(self):
        """Discover and execute all test_ methods, return summary."""
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