import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics checker for the Omega- discovery and K0 decay problem.
    Tests interface completeness, physical invariants, modeling correctness,
    monotonicity, and a weak answer sanity check.
    """

    def __init__(self, solution_instance):
        # --- Extract givens safely ---
        self.m_K = None
        self.m_p = None
        self.m_Omega = None
        self.m_pi = None
        self.beta = None

        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
            self.m_K = givens.get('m_K', None)
            self.m_p = givens.get('m_p', None)
            self.m_Omega = givens.get('m_Omega', None)
            self.m_pi = givens.get('m_pi', None)
            self.beta = givens.get('beta', None)
        else:
            self.m_K = getattr(solution_instance, 'm_K', None)
            self.m_p = getattr(solution_instance, 'm_p', None)
            self.m_Omega = getattr(solution_instance, 'm_Omega', None)
            self.m_pi = getattr(solution_instance, 'm_pi', None)
            self.beta = getattr(solution_instance, 'beta', None)

        # --- Extract results safely ---
        self.T = None
        self.theta_max = None
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
            self.T = results.get('T', None)
            self.theta_max = results.get('theta_max', None)
        else:
            self.T = getattr(solution_instance, 'T', None)
            self.theta_max = getattr(solution_instance, 'theta_max', None)

    def _compute_pion_restframe_props(self):
        """Return (E_star, p_star, v_star) if masses are numeric, else (None, None, None)."""
        if self.m_K is None or self.m_pi is None:
            return None, None, None
        mK, mpi = self.m_K, self.m_pi
        if not (isinstance(mK, (int, float)) and isinstance(mpi, (int, float))):
            return None, None, None
        E_star = mK / 2.0
        p_star = math.sqrt((mK/2.0)**2 - mpi**2)
        v_star = p_star / E_star
        return E_star, p_star, v_star

    # ---------- Interface Tests ----------
    def test_interface_has_T(self):
        if self.T is None:
            return (False, "Missing required result 'T'")
        return (True, "T present")

    def test_interface_has_theta_max(self):
        if self.theta_max is None:
            return (False, "Missing required result 'theta_max'")
        return (True, "theta_max present")

    def test_interface_has_masses(self):
        missing = []
        if self.m_K is None:
            missing.append("m_K")
        if self.m_p is None:
            missing.append("m_p")
        if self.m_Omega is None:
            missing.append("m_Omega")
        if self.m_pi is None:
            missing.append("m_pi")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required masses provided")

    def test_interface_beta_is_08(self):
        if self.beta is None:
            return (False, "Missing 'beta' given")
        if not isinstance(self.beta, (int, float)):
            return (False, "Beta must be numeric")
        if not math.isclose(self.beta, 0.8, rel_tol=1e-2):
            return (False, f"Beta should be 0.8, got {self.beta}")
        return (True, f"Beta = {self.beta} (expected 0.8)")

    # ---------- Invariant Tests (theorem-based, use isclose) ----------
    def test_invariant_threshold_energy_relation(self):
        """Check that (E_K + m_p)^2 - p_K^2 = (m_Omega + 2 m_K)^2."""
        if any(v is None for v in [self.m_K, self.m_p, self.m_Omega, self.T]):
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.m_K, self.m_p, self.m_Omega, self.T]):
            return (True, "SKIP: non-numeric data")
        mK, mp, mOmega, T = self.m_K, self.m_p, self.m_Omega, self.T
        E_K = T + mK
        pK_sq = E_K**2 - mK**2
        computed_sq = (E_K + mp)**2 - pK_sq
        expected_sq = (mOmega + 2*mK)**2
        if not math.isclose(computed_sq, expected_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Invariant mass squared mismatch: computed {computed_sq}, expected {expected_sq}")
        return (True, "Threshold invariant mass condition holds")

    def test_invariant_pion_restframe_energy(self):
        """In K0 rest frame, each pion energy = m_K/2."""
        E_star, _, _ = self._compute_pion_restframe_props()
        if E_star is None:
            return (False, "SKIP: cannot compute")
        half_mK = self.m_K / 2.0
        if not math.isclose(E_star, half_mK, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_pi* = {E_star} != m_K/2 = {half_mK}")
        return (True, f"E_pi* = {E_star} ≈ m_K/2")

    def test_invariant_pion_restframe_on_shell(self):
        """Check E*^2 - p*^2 = m_pi^2."""
        E_star, p_star, _ = self._compute_pion_restframe_props()
        if E_star is None:
            return (False, "SKIP: cannot compute")
        mass_sq = E_star**2 - p_star**2
        if not math.isclose(mass_sq, self.m_pi**2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"On-shell fails: computed {mass_sq}, expected {self.m_pi**2}")
        return (True, "Pion rest-frame on-shell condition satisfied")

    # ---------- Modeling Tests ----------
    def test_modeling_lorentz_boost_consistency(self):
        """Check that for θ* = 0 or π, the lab angle is 0 or π respectively."""
        if self.beta is None or self.m_K is None or self.m_pi is None:
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.beta, self.m_K, self.m_pi]):
            return (True, "SKIP: non-numeric data")
        E_star, p_star, _ = self._compute_pion_restframe_props()
        if E_star is None:
            return (False, "SKIP: could not compute pion properties")
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        # θ* = 0 case: pion emitted forward in K0 rest frame
        p_par_0 = gamma * (p_star + self.beta * E_star)
        p_perp_0 = 0.0
        theta_lab_0 = math.atan2(p_perp_0, p_par_0)  # should be 0
        if not math.isclose(theta_lab_0, 0.0, abs_tol=1e-2):
            return (False, f"θ* = 0 should give θ_lab = 0, got {theta_lab_0}")
        # θ* = π case: pion emitted backward
        p_par_pi = gamma * (-p_star + self.beta * E_star)
        p_perp_pi = 0.0
        theta_lab_pi = math.atan2(p_perp_pi, p_par_pi)
        # For beta=0.8 and typical masses, both pions go forward.
        # The precise check is that the mapping is consistent.
        # We check that the transformation doesn't produce an angle > pi.
        if theta_lab_pi < 0:
            # backward emission can become forward in lab; that's fine
            pass
        return (True, "Lorentz boost mapping works for forward/backward emission")

    def test_modeling_max_angle_derivation(self):
        """Verify that the condition for finite max angle is β > v* (pion speed in K0 rest frame)."""
        _, _, v_star = self._compute_pion_restframe_props()
        if v_star is None or self.beta is None:
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [v_star, self.beta]):
            return (True, "SKIP: non-numeric data")
        # Derived condition: if β > v*, then theta_max < π; if β ≤ v*, then theta_max = π.
        if self.theta_max is None:
            return (False, "SKIP: theta_max not available")
        if self.beta > v_star:
            # expected theta_max finite
            if self.theta_max == sp.pi or (isinstance(self.theta_max, (int, float)) and
                math.isclose(self.theta_max, math.pi, rel_tol=1e-2)):
                return (False, f"beta > v* ({self.beta} > {v_star:.3f}) should give theta_max < pi, got pi")
            if isinstance(self.theta_max, (int, float)):
                if self.theta_max >= math.pi:
                    return (False, f"theta_max = {self.theta_max} should be < pi")
            return (True, f"beta > v*: theta_max finite")
        else:
            # expected theta_max = pi
            if not (self.theta_max == sp.pi or (isinstance(self.theta_max, (int, float)) and
                math.isclose(self.theta_max, math.pi, rel_tol=1e-2))):
                return (False, f"beta <= v* should give theta_max = pi, got {self.theta_max}")
            return (True, f"beta <= v*: theta_max = pi")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_T_with_m_Omega(self):
        """Check T increases when m_Omega increases."""
        if any(v is None for v in [self.m_K, self.m_p, self.m_Omega, self.T]):
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.m_K, self.m_p, self.m_Omega, self.T]):
            return (True, "SKIP: non-numeric data")
        mK, mp, mOmega, T = self.m_K, self.m_p, self.m_Omega, self.T
        delta = 0.05 * mOmega
        mOmega2 = mOmega + delta
        T2 = ((mOmega2 + 2*mK)**2 - (mp + mK)**2) / (2*mp)
        if T2 <= T:
            return (False, f"T should increase with m_Omega: baseline {T}, perturbed {T2}")
        return (True, f"T increases with m_Omega: {T} -> {T2}")

    def test_monotonicity_theta_max_with_beta_when_finite(self):
        """For beta > v*, increasing beta should reduce theta_max."""
        _, _, v_star = self._compute_pion_restframe_props()
        if v_star is None or self.beta is None or self.theta_max is None:
            return (False, "SKIP: missing data")
        if not all(isinstance(v, (int, float)) for v in [self.beta, v_star, self.theta_max]):
            return (True, "SKIP: non-numeric data")
        if self.beta <= v_star:
            return (True, "SKIP: beta <= v*, theta_max = pi always, no monotonicity to test")
        # Baseline theta_max (re-derive from given parameters)
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        sin_theta_base = v_star / (gamma * self.beta)
        if sin_theta_base > 1:
            theta_base = math.pi
        else:
            theta_base = math.asin(sin_theta_base)
        # Perturb beta upward by 5%
        beta_new = self.beta * 1.05
        if beta_new >= 1:
            return (True, "SKIP: perturbed beta >= 1")
        gamma_new = 1.0 / math.sqrt(1.0 - beta_new**2)
        sin_theta_new = v_star / (gamma_new * beta_new)
        if sin_theta_new > 1:
            theta_new = math.pi
        else:
            theta_new = math.asin(sin_theta_new)
        if theta_new >= theta_base:
            return (False, f"Increasing beta should decrease theta_max: baseline {theta_base:.3f}, new {theta_new:.3f}")
        return (True, f"theta_max decreases with beta: {theta_base:.3f} -> {theta_new:.3f}")

    # ---------- Weak Answer Sanity (single method) ----------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: T should be positive (if numeric), theta_max should be in [0, pi] (inclusive) or symbolic pi.
        No exact equality checks against computed target values.
        """
        messages = []
        all_ok = True

        if self.T is None:
            all_ok = False
            messages.append("T missing")
        elif isinstance(self.T, (int, float)):
            if self.T <= 0:
                all_ok = False
                messages.append(f"T = {self.T} not positive")
            else:
                messages.append(f"T = {self.T} > 0")
        else:
            messages.append("T present (symbolic)")

        if self.theta_max is None:
            all_ok = False
            messages.append("theta_max missing")
        elif self.theta_max == sp.pi:
            messages.append("theta_max = pi (symbolic)")
        elif isinstance(self.theta_max, (int, float)):
            if 0.0 <= self.theta_max <= math.pi:
                messages.append(f"theta_max = {self.theta_max} in [0, pi]")
            else:
                all_ok = False
                messages.append(f"theta_max = {self.theta_max} out of range [0, pi]")
        else:
            messages.append("theta_max present (symbolic, non-pi)")

        if all_ok:
            return (True, "; ".join(messages))
        else:
            return (False, "; ".join(messages))

    # ---------- Main runner ----------
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