import math
import traceback


class PhysicsChecker:
    """
    Audits a relativistic nucleus-nucleus collision solution for:
    - interface completeness (required result keys exist)
    - invariant conservation laws (4‑momentum, invariant mass)
    - modeling consistency (velocity addition, monotonicity)
    - one weak sanity check (numeric magnitudes)
    """

    def __init__(self, solution_instance):
        # ----- input givens (no numeric defaults) -----
        self.m1 = getattr(solution_instance, 'm1', None)          # in atomic mass units (u)
        self.m2 = getattr(solution_instance, 'm2', None)          # in u
        self.beta1 = getattr(solution_instance, 'beta1', None)    # dimensionless
        self.beta_c = getattr(solution_instance, 'beta_c', None)  # dimensionless
        self.theta_c_deg = getattr(solution_instance, 'theta_c_deg', None)
        self.c2 = getattr(solution_instance, 'c2', None)          # MeV/u conversion

        # ----- results dictionary -----
        self.results = getattr(solution_instance, 'results', {})

        # ----- derived helpers (computed on‑the‑fly in tests) -----
        self.gamma1 = None
        if self.beta1 is not None and self.beta1 < 1.0:
            self.gamma1 = 1.0 / math.sqrt(1.0 - self.beta1 * self.beta1)

    # -----------------------------------------------------------------
    #  Interface checks: required keys exist and are of valid type/range
    # -----------------------------------------------------------------
    def test_interface_required_keys(self):
        """All six expected result keys must be present and not None."""
        required = ['beta0_formula', 'epsilon0_formula',
                    'beta0_num', 'epsilon0_num', 'beta_l', 'theta_l_deg']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing or None keys: {missing}")
        return (True, "All required keys present")

    def test_interface_beta0_formula_string(self):
        """beta0_formula must be a non‑empty string containing LaTeX."""
        formula = self.results.get('beta0_formula')
        if not isinstance(formula, str) or not formula:
            return (False, "beta0_formula is not a string or is empty")
        return (True, "beta0_formula is a valid string")

    def test_interface_epsilon0_formula_string(self):
        """epsilon0_formula must be a non‑empty string containing LaTeX."""
        formula = self.results.get('epsilon0_formula')
        if not isinstance(formula, str) or not formula:
            return (False, "epsilon0_formula is not a string or is empty")
        return (True, "epsilon0_formula is a valid string")

    def test_interface_beta0_num_valid(self):
        """beta0_num must be a finite number between 0 and 1."""
        val = self.results.get('beta0_num')
        if val is None:
            return (False, "beta0_num is None")
        if not isinstance(val, (int, float)):
            return (False, "beta0_num is not numeric")
        if not (0 < val < 1):
            return (False, f"beta0_num = {val} not in (0,1)")
        if not math.isfinite(val):
            return (False, "beta0_num is not finite")
        return (True, "beta0_num is valid")

    def test_interface_epsilon0_num_valid(self):
        """epsilon0_num must be positive, finite, and in a reasonable range (10⁴–10⁶ MeV)."""
        val = self.results.get('epsilon0_num')
        if val is None:
            return (False, "epsilon0_num is None")
        if not isinstance(val, (int, float)):
            return (False, "epsilon0_num is not numeric")
        if not math.isfinite(val):
            return (False, "epsilon0_num is not finite")
        if val <= 0:
            return (False, f"epsilon0_num = {val} not positive")
        if not (1e4 < val < 1e6):
            return (False, f"epsilon0_num = {val} outside plausible range")
        return (True, "epsilon0_num is valid")

    def test_interface_beta_l_valid(self):
        """beta_l must be a finite number between 0 and 1."""
        val = self.results.get('beta_l')
        if val is None:
            return (False, "beta_l is None")
        if not isinstance(val, (int, float)):
            return (False, "beta_l is not numeric")
        if not (0 < val < 1):
            return (False, f"beta_l = {val} not in (0,1)")
        if not math.isfinite(val):
            return (False, "beta_l is not finite")
        return (True, "beta_l is valid")

    def test_interface_theta_l_deg_valid(self):
        """theta_l_deg must be a finite number between 0 and 180."""
        val = self.results.get('theta_l_deg')
        if val is None:
            return (False, "theta_l_deg is None")
        if not isinstance(val, (int, float)):
            return (False, "theta_l_deg is not numeric")
        if not math.isfinite(val):
            return (False, "theta_l_deg is not finite")
        if not (0 <= val <= 180):
            return (False, f"theta_l_deg = {val} out of [0,180]")
        return (True, "theta_l_deg is valid")

    # -----------------------------------------------------------------
    #  Invariant checks: physics laws (use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_beta0_conservation(self):
        """beta0 from 4‑momentum conservation must match the provided value."""
        b0_prov = self.results.get('beta0_num')
        if b0_prov is None or self.m1 is None or self.m2 is None or self.beta1 is None:
            return (False, "SKIP: missing required data for beta0 conservation")

        gamma1 = 1.0 / math.sqrt(1.0 - self.beta1 ** 2)
        b0_calc = (self.m1 * gamma1 * self.beta1) / (self.m1 * gamma1 + self.m2)
        if not math.isclose(b0_prov, b0_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"beta0 conservation: provided {b0_prov:.4f}, derived {b0_calc:.4f}")
        return (True, f"beta0 from conservation OK ({b0_prov:.4f})")

    def test_invariant_epsilon0_invariant_mass(self):
        """epsilon0 must satisfy invariant mass: s = (E1+E2)^2 - (p1+p2)^2."""
        eps_prov = self.results.get('epsilon0_num')
        if eps_prov is None or self.m1 is None or self.m2 is None or self.beta1 is None:
            return (False, "SKIP: missing required data for epsilon0 invariant mass")

        # conversion factor: use provided c2 or default to 931.5 MeV/u (constant)
        c2_val = self.c2 if self.c2 is not None else 931.5
        gamma1 = 1.0 / math.sqrt(1.0 - self.beta1 ** 2)
        m1_MeV = self.m1 * c2_val
        m2_MeV = self.m2 * c2_val
        s = (m1_MeV * gamma1 + m2_MeV) ** 2 - (m1_MeV * gamma1 * self.beta1) ** 2
        if s <= 0:
            return (False, "Invariant mass squared is not positive")
        eps_calc = math.sqrt(s)
        if not math.isclose(eps_prov, eps_calc, rel_tol=5e-2, abs_tol=5e-2):
            return (False,
                    f"epsilon0 invariant mass: provided {eps_prov:.2e}, "
                    f"derived {eps_calc:.2e} (diff {abs(eps_prov-eps_calc)/eps_calc*100:.1f}%)")
        return (True, f"epsilon0 from invariant mass OK ({eps_prov:.2e})")

    def test_invariant_velocity_addition_beta_l(self):
        """Beta_l from relativistic velocity addition must match the provided value."""
        bl_prov = self.results.get('beta_l')
        b0 = self.results.get('beta0_num')
        if (bl_prov is None or b0 is None or
                self.beta_c is None or self.theta_c_deg is None):
            return (False, "SKIP: missing required data for velocity addition")

        theta_c = math.radians(self.theta_c_deg)
        ux_prime = self.beta_c * math.cos(theta_c)
        uy_prime = self.beta_c * math.sin(theta_c)
        gamma0 = 1.0 / math.sqrt(1.0 - b0 * b0)
        denom = 1.0 + b0 * ux_prime
        ux = (ux_prime + b0) / denom
        uy = uy_prime / (gamma0 * denom)
        bl_calc = math.sqrt(ux * ux + uy * uy)
        if not math.isclose(bl_prov, bl_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"beta_l: provided {bl_prov:.4f}, derived {bl_calc:.4f}")
        return (True, f"beta_l from Lorentz addition OK ({bl_prov:.4f})")

    def test_invariant_velocity_addition_theta_l(self):
        """Theta_l from relativistic velocity addition must match the provided value."""
        th_prov = self.results.get('theta_l_deg')
        b0 = self.results.get('beta0_num')
        if (th_prov is None or b0 is None or
                self.beta_c is None or self.theta_c_deg is None):
            return (False, "SKIP: missing required data for velocity addition")

        theta_c = math.radians(self.theta_c_deg)
        ux_prime = self.beta_c * math.cos(theta_c)
        uy_prime = self.beta_c * math.sin(theta_c)
        gamma0 = 1.0 / math.sqrt(1.0 - b0 * b0)
        denom = 1.0 + b0 * ux_prime
        ux = (ux_prime + b0) / denom
        uy = uy_prime / (gamma0 * denom)
        th_calc = math.degrees(math.atan2(uy, ux))
        if not math.isclose(th_prov, th_calc, rel_tol=5e-2, abs_tol=1.0):
            return (False,
                    f"theta_l: provided {th_prov:.1f}°, derived {th_calc:.1f}°")
        return (True, f"theta_l from Lorentz addition OK ({th_prov:.1f}°)")

    # -----------------------------------------------------------------
    #  Modeling checks: quality of explanation and justifications
    # -----------------------------------------------------------------
    def test_modeling_beta0_formula_contains_expected_parts(self):
        """The formula string should contain essential mathematical elements."""
        formula = self.results.get('beta0_formula', '')
        needed = ['beta', 'm_1', 'm_2', 'sqrt', 'frac']
        missing = [el for el in needed if el not in formula]
        if missing:
            return (False, f"Formula missing LaTeX keywords: {missing}")
        return (True, "beta0 formula contains expected LaTeX")

    def test_modeling_epsilon0_formula_contains_expected_parts(self):
        """The formula string should contain essential mathematical elements."""
        formula = self.results.get('epsilon0_formula', '')
        needed = ['varepsilon', 'sqrt', 'm_1', 'm_2', 'frac', 'sqrt']
        missing = [el for el in needed if el not in formula]
        if missing:
            return (False, f"Formula missing LaTeX keywords: {missing}")
        return (True, "epsilon0 formula contains expected LaTeX")

    def test_modeling_nonrelativistic_approx_justified(self):
        """Check that gamma0 is close enough to 1 to justify NR approximation."""
        b0 = self.results.get('beta0_num')
        if b0 is None:
            return (False, "SKIP: beta0_num missing for NR justification check")
        gamma0 = 1.0 / math.sqrt(1.0 - b0 * b0)
        if gamma0 > 1.05:  # more than 5% deviation
            return (False, f"gamma0 = {gamma0:.3f} > 1.05, NR approx not justified")
        return (True, f"gamma0 = {gamma0:.3f} near 1, NR approx is reasonable")

    # -----------------------------------------------------------------
    #  Monotonicity checks (perturb inputs and compare to baseline)
    # -----------------------------------------------------------------
    def test_monotonicity_beta0_increases_with_beta1(self):
        """Increasing beta1 (while keeping masses fixed) should increase beta0."""
        b0_ref = self.results.get('beta0_num')
        if (b0_ref is None or self.m1 is None or self.m2 is None or
                self.beta1 is None):
            return (False, "SKIP: missing required data for monotonicity")

        # increase beta1 by 5% (but stay below 1)
        delta = 0.05
        beta1_pert = min(self.beta1 * (1 + delta), 0.99)
        if beta1_pert <= self.beta1:
            return (False, "Perturbation did not increase beta1")
        gamma1_pert = 1.0 / math.sqrt(1.0 - beta1_pert * beta1_pert)
        b0_pert = (self.m1 * gamma1_pert * beta1_pert) / \
                  (self.m1 * gamma1_pert + self.m2)
        if b0_pert <= b0_ref:
            return (False,
                    f"beta0 did not rise with beta1: ref={b0_ref:.4f}, "
                    f"pert={b0_pert:.4f}")
        return (True, f"beta0 increases with beta1 ({b0_ref:.4f} -> {b0_pert:.4f})")

    def test_monotonicity_epsilon0_increases_with_beta1(self):
        """Increasing beta1 should increase epsilon0."""
        eps_ref = self.results.get('epsilon0_num')
        if (eps_ref is None or self.m1 is None or self.m2 is None or
                self.beta1 is None):
            return (False, "SKIP: missing required data for monotonicity")

        delta = 0.05
        beta1_pert = min(self.beta1 * (1 + delta), 0.99)
        if beta1_pert <= self.beta1:
            return (False, "Perturbation did not increase beta1")
        c2_val = self.c2 if self.c2 is not None else 931.5
        gamma1_pert = 1.0 / math.sqrt(1.0 - beta1_pert * beta1_pert)
        m1_MeV = self.m1 * c2_val
        m2_MeV = self.m2 * c2_val
        s_pert = (m1_MeV * gamma1_pert + m2_MeV) ** 2 - \
                 (m1_MeV * gamma1_pert * beta1_pert) ** 2
        eps_pert = math.sqrt(s_pert)
        if eps_pert <= eps_ref:
            return (False,
                    f"epsilon0 did not rise with beta1: ref={eps_ref:.2e}, "
                    f"pert={eps_pert:.2e}")
        return (True, f"epsilon0 increases with beta1 ({eps_ref:.2e} -> {eps_pert:.2e})")

    # -----------------------------------------------------------------
    #  One weak answer sanity check (no isclose, no derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_sign_and_magnitude(self):
        """Final numeric answers must be positive, finite, and have sensible signs."""
        b0 = self.results.get('beta0_num')
        eps0 = self.results.get('epsilon0_num')
        bl = self.results.get('beta_l')
        th = self.results.get('theta_l_deg')

        checks = []
        if b0 is not None:
            if not (0 < b0 < 1):
                checks.append("beta0 out of (0,1)")
        if eps0 is not None:
            if not (eps0 > 0 and math.isfinite(eps0)):
                checks.append("epsilon0 invalid")
        if bl is not None:
            if not (0 < bl < 1):
                checks.append("beta_l out of (0,1)")
        if th is not None:
            if not (0 <= th <= 180):
                checks.append("theta_l out of [0,180]")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All answer signs and magnitudes are sensible")

    # -----------------------------------------------------------------
    #  Main test runner
    # -----------------------------------------------------------------
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
                passed, msg = (False, f"{name} CRASHED: {e}\n"
                                      f"{traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)