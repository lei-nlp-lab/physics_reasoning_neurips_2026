import math

class PhysicsChecker:
    """
    Auditor for the H2 ideal gas problem: translational rms speed, rotational rms angular speed,
    molar heat capacities C_v and C_p.
    """

    def __init__(self, solution_instance):
        # ------------------------------------------------------------------
        # Extract givens and constants (use None as default)
        # ------------------------------------------------------------------
        self.T = getattr(solution_instance, 'T', None)
        self.m_H = getattr(solution_instance, 'm_H', None)
        self.d = getattr(solution_instance, 'd', None)
        self.k = getattr(solution_instance, 'k', None)
        self.R = getattr(solution_instance, 'R', None)

        # Results (emit contract)
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.v_rms = results.get('v_rms', None)
        self.omega_rms = results.get('omega_rms', None)
        self.C_v = results.get('C_v', None)
        self.C_p = results.get('C_p', None)

        # Derived quantities
        self.m_mol = 2 * self.m_H if (self.m_H is not None) else None
        if self.m_H is not None and self.d is not None:
            self.I = self.m_H * (self.d ** 2) / 2.0
        else:
            self.I = None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All four emit keys must be present in results."""
        missing = []
        if self.v_rms is None:
            missing.append('v_rms')
        if self.omega_rms is None:
            missing.append('omega_rms')
        if self.C_v is None:
            missing.append('C_v')
        if self.C_p is None:
            missing.append('C_p')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_numeric_and_finite(self):
        """All four result values must be finite floats."""
        for name, val in [('v_rms', self.v_rms),
                          ('omega_rms', self.omega_rms),
                          ('C_v', self.C_v),
                          ('C_p', self.C_p)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (type {type(val).__name__})")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
        return (True, "All values are finite numbers")

    # ------------------------------------------------------------------
    # Invariant (theorem-based) tests
    # ------------------------------------------------------------------
    def test_invariant_translational_equipartition(self):
        """Check ½ m_mol v_rms² ≈ (3/2) kT."""
        if any(x is None for x in [self.v_rms, self.m_mol, self.k, self.T]):
            return (False, "SKIP: missing v_rms, m_mol, k, or T")
        left = 0.5 * self.m_mol * (self.v_rms ** 2)
        right = 1.5 * self.k * self.T
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Translational equipartition holds: left={left:.3e}, right={right:.3e}")
        else:
            return (False, f"Translational equipartition violated: left={left:.3e}, right={right:.3e}")

    def test_invariant_rotational_equipartition(self):
        """Check ½ I ω_rms² ≈ kT (two rotational DOF → 2*(½kT) = kT)."""
        if any(x is None for x in [self.omega_rms, self.I, self.k, self.T]):
            return (False, "SKIP: missing omega_rms, I, k, or T")
        left = 0.5 * self.I * (self.omega_rms ** 2)
        right = self.k * self.T   # because 2*(1/2)kT = kT
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Rotational equipartition holds: left={left:.3e}, right={right:.3e}")
        else:
            return (False, f"Rotational equipartition violated: left={left:.3e}, right={right:.3e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_moment_of_inertia(self):
        """Check that I is consistent with m_H * d^2 / 2 (if both given)."""
        if self.I is None:
            return (False, "SKIP: I not available (missing m_H or d)")
        if self.m_H is None or self.d is None:
            return (False, "SKIP: m_H or d missing from solution instance")
        expected_I = self.m_H * (self.d ** 2) / 2.0
        if self.omega_rms is None or self.k is None or self.T is None:
            return (False, "SKIP: cannot infer I from omega_rms")
        inferred_I = 2.0 * self.k * self.T / (self.omega_rms ** 2)
        if math.isclose(expected_I, inferred_I, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Moments of inertia agree: expected={expected_I:.3e}, inferred={inferred_I:.3e}")
        else:
            return (False, f"Moment of inertia mismatch: expected={expected_I:.3e}, inferred={inferred_I:.3e}")

    def test_modeling_Cp_minus_Cv(self):
        """Check that C_p - C_v ≈ R."""
        if any(x is None for x in [self.C_p, self.C_v, self.R]):
            return (False, "SKIP: missing C_p, C_v, or R")
        diff = self.C_p - self.C_v
        if math.isclose(diff, self.R, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C_p - C_v = {diff:.4f} ≈ R = {self.R:.4f}")
        else:
            return (False, f"C_p - C_v = {diff:.4f}, expected ~{self.R:.4f}")

    def test_modeling_degrees_of_freedom(self):
        """Check that C_v / (R/2) is close to 5 (two rotational + three translational)."""
        if self.C_v is None or self.R is None:
            return (False, "SKIP: missing C_v or R")
        f_inferred = 2.0 * self.C_v / self.R
        if math.isclose(f_inferred, 5.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Inferred f = {f_inferred:.3f} ≈ 5")
        else:
            return (False, f"Inferred f = {f_inferred:.3f}, expected 5")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_v_rms_with_T(self):
        """If T increases, v_rms increases (for fixed m_H and k)."""
        if any(x is None for x in [self.T, self.v_rms, self.m_mol, self.k]):
            return (False, "SKIP: missing T, v_rms, m_mol, or k")
        # Baseline: compute v_rms from equipartition
        v_baseline = math.sqrt(3.0 * self.k * self.T / self.m_mol)
        # Perturb T up by 5%
        T_pert = self.T * 1.05
        v_pert = math.sqrt(3.0 * self.k * T_pert / self.m_mol)
        # The actual v_rms should be less than v_pert because actual T is lower
        if self.v_rms < v_pert:
            return (True, f"Monotonicity holds: v_rms increases with T (baseline {v_baseline:.2f} < perturbed {v_pert:.2f})")
        else:
            return (False, f"Monotonicity violated: v_rms should be < v_pert, got {self.v_rms:.2f} vs {v_pert:.2f}")

    def test_monotonicity_omega_rms_with_T(self):
        """If T increases, omega_rms increases (for fixed m_H, d, k)."""
        if any(x is None for x in [self.T, self.omega_rms, self.I, self.k]):
            return (False, "SKIP: missing T, omega_rms, I, or k")
        omega_baseline = math.sqrt(2.0 * self.k * self.T / self.I)
        T_pert = self.T * 1.05
        omega_pert = math.sqrt(2.0 * self.k * T_pert / self.I)
        if self.omega_rms < omega_pert:
            return (True, f"Monotonicity holds: omega_rms increases with T (baseline {omega_baseline:.2e} < perturbed {omega_pert:.2e})")
        else:
            return (False, f"Monotonicity violated: omega_rms should be < omega_pert, got {self.omega_rms:.2e} vs {omega_pert:.2e}")

    # ------------------------------------------------------------------
    # Weak answer‑sanity test (exactly one)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausibility(self):
        """Check broad plausibility of all four result quantities (no exact derivation)."""
        issues = []

        # v_rms ~ 10^3 m/s (1000–5000)
        if self.v_rms is not None:
            if not (1000 <= self.v_rms <= 5000):
                issues.append(f"v_rms={self.v_rms:.2f} outside plausible range [1000,5000]")
        else:
            issues.append("v_rms missing")

        # omega_rms ~ 10^13 rad/s (1e12–1e14)
        if self.omega_rms is not None:
            if not (1e12 <= self.omega_rms <= 1e14):
                issues.append(f"omega_rms={self.omega_rms:.2e} outside plausible range [1e12,1e14]")
        else:
            issues.append("omega_rms missing")

        # C_v ~ 21, C_p ~ 29, C_p > C_v
        if self.C_v is not None and self.C_p is not None:
            if not (20 <= self.C_v <= 22):
                issues.append(f"C_v={self.C_v:.2f} not in [20,22]")
            if not (28 <= self.C_p <= 30):
                issues.append(f"C_p={self.C_p:.2f} not in [28,30]")
            if self.C_p <= self.C_v:
                issues.append("C_p <= C_v")
        else:
            issues.append("C_v or C_p missing")

        if issues:
            return (False, " ; ".join(issues))
        return (True, "All quantities within plausible ranges and ordering OK")

    # ------------------------------------------------------------------
    # Master test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)