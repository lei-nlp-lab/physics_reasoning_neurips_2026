import sympy
import math
import traceback

class PhysicsChecker:
    """
    A robust physics checker for the p+n capture problem.
    Checks interface, invariants, modeling, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # ---------- Extract results safely ----------
        # solution_instance may have a 'results' dict or direct attributes
        self.results = getattr(solution_instance, 'results', {})
        # Or try direct attributes (None if missing)
        def _safe_get(key):
            val = self.results.get(key)
            if val is None:
                val = getattr(solution_instance, key, None)
            return val

        self.Q = _safe_get('Q')
        self.E_gamma = _safe_get('E_gamma')
        self.recoil_correction = _safe_get('recoil_correction')
        self.V_c_MeV = _safe_get('V_c_MeV')
        self.T_n = _safe_get('T_n')

        # ---------- Physical constants (from problem statement) ----------
        self.m_p = 1.00783          # amu
        self.m_n = 1.00867          # amu
        self.m_d = 2.01410          # amu
        self.amu_MeV = 931.0        # 1 amu = 931 MeV (as given)
        self.amu_g = 1.66e-24       # g
        self.hbar = 1.05e-27        # erg·s
        self.c = 3e10               # cm/s
        self.MeV_erg = 1.6e-6       # erg/MeV (1 MeV = 1.6e-6 erg)
        self.r0 = 4e-13             # cm

        # ---------- Derived quantities ----------
        # Reduced mass in grams
        self.mu_g = (self.m_n * self.m_p) / (self.m_n + self.m_p) * self.amu_g
        # m_d c^2 in MeV (used in some checks)
        self.m_d_c2_MeV = self.m_d * self.amu_MeV

    def run_all_tests(self):
        """Discover and run all test_* methods, collect results."""
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith('test_'):
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

    # ============================================================
    # Interface tests
    # ============================================================
    def test_interface_required_keys(self):
        """Check that all five expected result keys are present and not None."""
        missing = []
        for key in ['Q', 'E_gamma', 'recoil_correction', 'V_c_MeV', 'T_n']:
            if getattr(self, key, None) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    # ============================================================
    # Invariant tests (theorem‑based physical sanity)
    # ============================================================
    def test_invariant_Q_from_mass_defect(self):
        """Q must equal (m_p + m_n - m_d) * amu_MeV."""
        if self.Q is None:
            return (False, "SKIP: Q not available")
        Q_calc = (self.m_p + self.m_n - self.m_d) * self.amu_MeV
        if math.isclose(self.Q, Q_calc, rel_tol=1e-3):
            return (True, f"Q={self.Q} close to computed {Q_calc}")
        return (False, f"Q={self.Q} not close to computed {Q_calc}")

    def test_invariant_photon_energy_approx(self):
        """E_gamma should be approximately equal to Q (recoil negligible)."""
        if self.E_gamma is None or self.Q is None:
            return (False, "SKIP: E_gamma or Q missing")
        if math.isclose(self.E_gamma, self.Q, rel_tol=1e-2):
            return (True, f"E_gamma={self.E_gamma} approx Q={self.Q}")
        return (False, f"E_gamma={self.E_gamma} not close to Q={self.Q}")

    def test_invariant_recoil_correction_positive_small(self):
        """recoil_correction must be positive and small compared to Q."""
        if self.recoil_correction is None:
            return (False, "SKIP: recoil_correction missing")
        if self.recoil_correction <= 0:
            return (False, f"recoil_correction={self.recoil_correction} not positive")
        if self.Q is not None and self.recoil_correction > 0.01 * self.Q:
            return (False, f"recoil_correction={self.recoil_correction} too large relative to Q={self.Q}")
        return (True, "recoil_correction positive and reasonable")

    def test_invariant_centrifugal_barrier(self):
        """Centrifugal barrier V_c in MeV must match ℏ²/(μ r0²)."""
        if self.V_c_MeV is None:
            return (False, "SKIP: V_c_MeV missing")
        V_c_erg = self.hbar**2 / (self.mu_g * self.r0**2)
        V_c_MeV_calc = V_c_erg / self.MeV_erg
        if math.isclose(self.V_c_MeV, V_c_MeV_calc, rel_tol=1e-2):
            return (True, f"V_c_MeV={self.V_c_MeV} close to computed {V_c_MeV_calc}")
        return (False, f"V_c_MeV={self.V_c_MeV} not close to computed {V_c_MeV_calc}")

    # ============================================================
    # Modeling / trace tests (check internal consistency of derived quantities)
    # ============================================================
    def test_modeling_recoil_correction_formula(self):
        """Check that recoil_correction ≈ Q²/(2 m_d amu_MeV)."""
        if self.recoil_correction is None or self.Q is None:
            return (False, "SKIP: recoil_correction or Q missing")
        expected = self.Q**2 / (2 * self.m_d * self.amu_MeV)
        if math.isclose(self.recoil_correction, expected, rel_tol=1e-2):
            return (True, f"recoil_correction={self.recoil_correction} close to {expected}")
        return (False, f"recoil_correction={self.recoil_correction} not close to {expected}")

    def test_modeling_T_n_from_V_c(self):
        """Check that T_n = (m_n+m_p)/m_p * V_c_MeV (exact lab‑CM relation)."""
        if self.T_n is None or self.V_c_MeV is None:
            return (False, "SKIP: T_n or V_c_MeV missing")
        expected_Tn = (self.m_n + self.m_p) / self.m_p * self.V_c_MeV
        if math.isclose(self.T_n, expected_Tn, rel_tol=1e-2):
            return (True, f"T_n={self.T_n} close to computed {expected_Tn}")
        return (False, f"T_n={self.T_n} not close to computed {expected_Tn}")

    # ============================================================
    # Monotonicity tests (trend checks via explicit perturbation)
    # ============================================================
    def test_monotonicity_Tn_decreases_with_r0(self):
        """T_n should decrease when the deuteron radius r0 is increased."""
        if self.T_n is None:
            return (False, "SKIP: T_n missing")
        # Baseline: compute T_n from V_c using the same formulas
        V_c_erg_base = self.hbar**2 / (self.mu_g * self.r0**2)
        V_c_MeV_base = V_c_erg_base / self.MeV_erg
        T_n_base = (self.m_n + self.m_p) / self.m_p * V_c_MeV_base
        # Increase r0 by 10%
        r0_new = self.r0 * 1.1
        V_c_erg_new = self.hbar**2 / (self.mu_g * r0_new**2)
        V_c_MeV_new = V_c_erg_new / self.MeV_erg
        T_n_new = (self.m_n + self.m_p) / self.m_p * V_c_MeV_new
        if T_n_new < T_n_base:
            return (True, f"T_n decreases when r0 increases (base={T_n_base:.3f}, new={T_n_new:.3f})")
        return (False, f"Monotonicity violation: T_n did not decrease (base={T_n_base:.3f}, new={T_n_new:.3f})")

    def test_monotonicity_E_gamma_increases_with_Q(self):
        """E_gamma should increase when Q is increased (exact quadratic relation)."""
        if self.E_gamma is None or self.Q is None:
            return (False, "SKIP: E_gamma or Q missing")
        # Compute exact E_gamma from Q using the full quadratic solution:
        # E = m_d c² ( -1 + sqrt(1 + 2Q/(m_d c²)) )
        E_base = self.m_d_c2_MeV * (-1 + math.sqrt(1 + 2*self.Q/self.m_d_c2_MeV))
        Q_pert = self.Q * 1.1
        E_pert = self.m_d_c2_MeV * (-1 + math.sqrt(1 + 2*Q_pert/self.m_d_c2_MeV))
        if E_pert > E_base:
            return (True, f"E_gamma increases with Q (base={E_base:.4f}, pert={E_pert:.4f})")
        return (False, f"Monotonicity violation: E_gamma did not increase (base={E_base:.4f}, pert={E_pert:.4f})")

    # ============================================================
    # Weak answer sanity test (only one, secondary)
    # ============================================================
    def test_answer_sanity_positive_finite(self):
        """All five results must be positive finite numbers."""
        results = {
            'Q': self.Q,
            'E_gamma': self.E_gamma,
            'recoil_correction': self.recoil_correction,
            'V_c_MeV': self.V_c_MeV,
            'T_n': self.T_n
        }
        for name, val in results.items():
            if val is None:
                return (False, f"{name} is None")
            if not (isinstance(val, (int, float)) and val > 0 and math.isfinite(val)):
                return (False, f"{name}={val} not positive finite number")
        return (True, "All results positive and finite")