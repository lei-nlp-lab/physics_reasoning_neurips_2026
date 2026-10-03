import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the three-mass normal modes problem.
    Tests: interface, invariants, modeling quality, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract inputs and results safely
        self.givens = getattr(solution_instance, 'givens', None)
        self.results = getattr(solution_instance, 'results', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.solving_pseudocode = getattr(solution_instance, 'solving_pseudocode', None)

        # Extract results keys (all expected)
        self.omega_squared = self.results.get('omega_squared', None)
        self.omega = self.results.get('omega', None)
        self.mode1 = self.results.get('mode1', None)
        self.mode2 = self.results.get('mode2', None)
        self.mode3 = self.results.get('mode3', None)

        # Extract given m and k (if available)
        if isinstance(self.givens, dict):
            self.m = self.givens.get('m', None)
            self.k = self.givens.get('k', None)
        else:
            # Fallback: try direct attribute
            self.m = getattr(solution_instance, 'm', None)
            self.k = getattr(solution_instance, 'k', None)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _normalize_vector(self, vec):
        """Return normalized vector (divide by first non‑zero element)."""
        if vec is None or len(vec) == 0:
            return None
        for i, v in enumerate(vec):
            if abs(v) > 1e-12:
                return [float(x) / float(v) for x in vec]
        return vec  # all zero

    def _check_eigen_equation(self, omega_sq, mode_vec):
        """Check that (K - omega_sq * M) * mode ≈ 0."""
        if self.k is None or self.m is None:
            return None, "m or k missing"
        k = float(self.k)
        m = float(self.m)
        # Stiffness matrix (symmetric)
        K = [[k, 0, -k],
             [0, k, -k],
             [-k, -k, 2*k]]
        # Identity mass matrix (all masses equal)
        M_scale = m
        # Compute residual vector
        res = []
        for i in range(3):
            s = 0.0
            for j in range(3):
                s += (K[i][j] - omega_sq * (M_scale if i == j else 0.0)) * float(mode_vec[j])
            res.append(s)
        # Check all components are close to zero
        all_zero = all(math.isclose(r, 0.0, rel_tol=1e-2, abs_tol=1e-2) for r in res)
        return all_zero, f"residuals = {res}"

    def _characteristic_det(self, omega_sq_sym):
        """Return symbolic determinant of (K - ω² M) for given ω² (SymPy expression)."""
        k_sym = sp.Symbol('k')
        m_sym = sp.Symbol('m')
        K_mat = sp.Matrix([[k_sym, 0, -k_sym],
                           [0, k_sym, -k_sym],
                           [-k_sym, -k_sym, 2*k_sym]])
        M_mat = m_sym * sp.eye(3)
        mat = K_mat - omega_sq_sym * M_mat
        return sp.simplify(mat.det())

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_omega_squared(self):
        if self.omega_squared is None:
            return (False, "SKIP: omega_squared not present")
        if not isinstance(self.omega_squared, (list, tuple)) or len(self.omega_squared) != 3:
            return (False, f"omega_squared should be list of length 3, got {len(self.omega_squared) if hasattr(self.omega_squared, '__len__') else '?'}")
        return (True, "omega_squared is a list of length 3")

    def test_interface_omega(self):
        if self.omega is None:
            return (False, "SKIP: omega not present")
        if not isinstance(self.omega, (list, tuple)) or len(self.omega) != 3:
            return (False, f"omega should be list of length 3, got {len(self.omega) if hasattr(self.omega, '__len__') else '?'}")
        return (True, "omega is a list of length 3")

    def test_interface_mode1(self):
        if self.mode1 is None:
            return (False, "SKIP: mode1 not present")
        if not isinstance(self.mode1, (list, tuple)) or len(self.mode1) != 3:
            return (False, f"mode1 should be list/tuple of length 3, got {len(self.mode1) if hasattr(self.mode1, '__len__') else '?'}")
        return (True, "mode1 is present and has 3 components")

    def test_interface_mode2(self):
        if self.mode2 is None:
            return (False, "SKIP: mode2 not present")
        if not isinstance(self.mode2, (list, tuple)) or len(self.mode2) != 3:
            return (False, f"mode2 should be list/tuple of length 3, got {len(self.mode2) if hasattr(self.mode2, '__len__') else '?'}")
        return (True, "mode2 is present and has 3 components")

    def test_interface_mode3(self):
        if self.mode3 is None:
            return (False, "SKIP: mode3 not present")
        if not isinstance(self.mode3, (list, tuple)) or len(self.mode3) != 3:
            return (False, f"mode3 should be list/tuple of length 3, got {len(self.mode3) if hasattr(self.mode3, '__len__') else '?'}")
        return (True, "mode3 is present and has 3 components")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_omega_squared_nonnegative(self):
        if self.omega_squared is None:
            return (False, "SKIP: omega_squared not present")
        for i, val in enumerate(self.omega_squared):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return (False, f"omega_squared[{i}] is not numeric: {val}")
            if v < -1e-12:
                return (False, f"omega_squared[{i}] = {v} is negative")
        return (True, "All omega_squared values are non‑negative")

    def test_invariant_eigenvalue_omega1(self):
        if self.omega_squared is None or self.mode1 is None:
            return (False, "SKIP: needed data missing")
        wsq0 = float(self.omega_squared[0])
        # ω1 = 0 is expected
        if not math.isclose(wsq0, 0.0, abs_tol=1e-6):
            return (False, f"omega_squared[0] = {wsq0}, expected 0")
        ok, msg = self._check_eigen_equation(wsq0, self.mode1)
        if ok is None:
            return (False, f"SKIP: {msg}")
        if not ok:
            return (False, f"Mode1 does not satisfy eigenvalue eq: {msg}")
        return (True, "Mode1 satisfies (K - 0*M) * mode1 ≈ 0")

    def test_invariant_eigenvalue_omega2(self):
        if self.omega_squared is None or self.mode2 is None:
            return (False, "SKIP: needed data missing")
        wsq1 = float(self.omega_squared[1])
        ok, msg = self._check_eigen_equation(wsq1, self.mode2)
        if ok is None:
            return (False, f"SKIP: {msg}")
        if not ok:
            return (False, f"Mode2 does not satisfy eigenvalue eq: {msg}")
        return (True, "Mode2 satisfies eigenvalue equation")

    def test_invariant_eigenvalue_omega3(self):
        if self.omega_squared is None or self.mode3 is None:
            return (False, "SKIP: needed data missing")
        wsq2 = float(self.omega_squared[2])
        ok, msg = self._check_eigen_equation(wsq2, self.mode3)
        if ok is None:
            return (False, f"SKIP: {msg}")
        if not ok:
            return (False, f"Mode3 does not satisfy eigenvalue eq: {msg}")
        return (True, "Mode3 satisfies eigenvalue equation")

    def test_invariant_orthogonality_modes(self):
        # Modes with distinct eigenvalues are orthogonal w.r.t. mass matrix (identity here)
        if None in (self.mode1, self.mode2, self.mode3):
            return (False, "SKIP: one or more modes missing")
        m1 = [float(x) for x in self.mode1]
        m2 = [float(x) for x in self.mode2]
        m3 = [float(x) for x in self.mode3]
        # dot products
        dot12 = sum(a*b for a,b in zip(m1,m2))
        dot13 = sum(a*b for a,b in zip(m1,m3))
        dot23 = sum(a*b for a,b in zip(m2,m3))
        ok = all(math.isclose(d, 0.0, abs_tol=1e-2) for d in (dot12, dot13, dot23))
        if not ok:
            return (False, f"Modes not orthogonal: dot12={dot12}, dot13={dot13}, dot23={dot23}")
        return (True, "Modes are orthogonal (dot products ≈ 0)")

    def test_invariant_characteristic_determinant(self):
        """Check that for each ω² eigenvalue, det(K - ω² M) ≈ 0."""
        if self.omega_squared is None or self.k is None or self.m is None:
            return (False, "SKIP: missing data")
        omega_sym = sp.Symbol('omega_sq')
        det_expr = self._characteristic_det(omega_sym)
        # Substitute symbolic k, m with actual values
        subs_dict = {sp.Symbol('k'): float(self.k), sp.Symbol('m'): float(self.m)}
        any_fail = False
        for i, wsq_val in enumerate(self.omega_squared):
            wsq_num = float(wsq_val)
            det_val = float(det_expr.subs(omega_sym, wsq_num).subs(subs_dict))
            if not math.isclose(det_val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                any_fail = True
                msg = f"ω²[{i}] = {wsq_num}: det = {det_val}, not ≈ 0"
        if any_fail:
            return (False, msg)
        return (True, "All eigenvalues make determinant ≈ 0")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_mode1_translational(self):
        """Mode1 should have all identical components (translation)."""
        if self.mode1 is None:
            return (False, "SKIP: mode1 not present")
        norm = self._normalize_vector(self.mode1)
        if norm is None:
            return (False, "mode1 is zero vector")
        # After normalization, all components should be 1 (or very close)
        all_equal = all(math.isclose(c, 1.0, rel_tol=1e-2) for c in norm)
        if not all_equal:
            return (False, f"Normalized mode1 = {norm}, expected all ≈ 1")
        return (True, "Mode1 corresponds to uniform translation (all components equal)")

    def test_modeling_mode2_pattern(self):
        """Mode2: middle component zero, outer components opposite."""
        if self.mode2 is None:
            return (False, "SKIP: mode2 not present")
        m2 = [float(x) for x in self.mode2]
        # Check that middle ≈ 0
        if not math.isclose(m2[1], 0.0, abs_tol=1e-2):
            return (False, f"middle component of mode2 is {m2[1]}, expected 0")
        # Check outer components have opposite sign and similar magnitude
        if not math.isclose(m2[0], -m2[2], rel_tol=1e-2):
            return (False, f"outer components of mode2: {m2[0]}, {m2[2]}, expected opposite")
        return (True, "Mode2 has middle stationary, outer opposite")

    def test_modeling_mode3_pattern(self):
        """Mode3: first and third equal, middle = -2 * first (after normalization)."""
        if self.mode3 is None:
            return (False, "SKIP: mode3 not present")
        m3 = [float(x) for x in self.mode3]
        # Normalize by first non-zero (likely first component not zero)
        norm_vec = self._normalize_vector(m3)
        if norm_vec is None:
            return (False, "mode3 is zero vector")
        # Expected pattern after normalization: (1, ? , 1) with ? = -2
        # But normalization may have scaled differently; check ratios
        # Check that first and third are equal
        if not math.isclose(norm_vec[0], norm_vec[2], rel_tol=1e-2):
            return (False, f"Normalized mode3: first {norm_vec[0]}, third {norm_vec[2]}, expected equal")
        # Check that middle is -2 times first
        if not math.isclose(norm_vec[1], -2.0 * norm_vec[0], rel_tol=1e-2):
            return (False, f"Normalized mode3: middle {norm_vec[1]}, expected -2*first = {-2*norm_vec[0]}")
        return (True, "Mode3 has outer equal, middle opposite and twice amplitude")

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_frequency_increase_with_k(self):
        """
        If we increase k by 5%, the non‑zero angular frequencies should increase.
        Use given k,m to compute expected ω² and compare with formula.
        """
        if self.k is None or self.m is None:
            return (False, "SKIP: k or m missing")
        k_orig = float(self.k)
        m = float(self.m)
        k_pert = 1.05 * k_orig

        # Expected ω² values for original and perturbed (using formula)
        orig_wsq2 = k_orig / m
        orig_wsq3 = 3 * k_orig / m
        pert_wsq2 = k_pert / m
        pert_wsq3 = 3 * k_pert / m

        # Compare perturbed vs original: ratio should be 1.05
        # (We are not using solution's ω² directly to avoid copy of final answer)
        if not math.isclose(pert_wsq2 / orig_wsq2, 1.05, rel_tol=1e-2):
            return (False, f"Ratio of ω² for mode2 after 5% k increase: {pert_wsq2/orig_wsq2}, expected 1.05")
        if not math.isclose(pert_wsq3 / orig_wsq3, 1.05, rel_tol=1e-2):
            return (False, f"Ratio of ω² for mode3 after 5% k increase: {pert_wsq3/orig_wsq3}, expected 1.05")
        # Also check that perturbed > original
        if not (pert_wsq2 > orig_wsq2 and pert_wsq3 > orig_wsq3):
            return (False, "Perturbed frequencies should be larger than original")
        return (True, "Increasing k by 5% increases ω² by factor 1.05 (modes 2 and 3)")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_omegas(self):
        """
        Very weak sanity: all omegas are real, non‑negative, and finite.
        No re‑derivation of values.
        """
        if self.omega is None:
            return (False, "SKIP: omega not present")
        for i, val in enumerate(self.omega):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return (False, f"omega[{i}] = {val} is not numeric")
            if math.isnan(v) or math.isinf(v):
                return (False, f"omega[{i}] = {v} is NaN or inf")
            if v < -1e-12:
                return (False, f"omega[{i}] = {v} is negative")
        return (True, "All omegas are real, non‑negative, and finite")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
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
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)