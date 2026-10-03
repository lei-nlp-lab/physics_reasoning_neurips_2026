import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely; default to None for all problem-specific quantities
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        # Fundamental physical constants (set to None if not provided)
        self.e = self.givens.get('e', None)          # elementary charge (C)
        self.m_e = self.givens.get('m_e', None)      # electron mass (kg)
        self.c = self.givens.get('c', None)          # speed of light (m/s)
        self.mu0 = self.givens.get('mu0', None)      # vacuum permeability (T·m/A)

        # Problem-specific givens (ALL default to None)
        self.E_eV = self.givens.get('E', None)          # beam energy in eV
        self.N = self.givens.get('N', None)             # number of particles per bunch
        self.L_lab = self.givens.get('L_lab', None)     # bunch length in lab frame (m)
        self.R = self.givens.get('R', None)             # bunch radius in lab frame (m)

        # Derived quantity: gamma (only if necessary data present)
        if (self.E_eV is not None and self.e is not None and
            self.m_e is not None and self.c is not None):
            self.gamma = (self.E_eV * self.e) / (self.m_e * self.c**2)
        else:
            self.gamma = None

        # Result values (ALL default to None)
        self.r_own = self.results.get('r_own', None)
        self.L_own = self.results.get('L_own', None)
        self.r_opp = self.results.get('r_opp', None)
        self.L_opp = self.results.get('L_opp', None)
        self.t_pass = self.results.get('t_pass', None)
        self.B_tot = self.results.get('B_tot', None)
        self.B_tot_G = self.results.get('B_tot_G', None)
        self.theta = self.results.get('theta', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All expected result keys must be present and not None."""
        required_keys = ['r_own', 'L_own', 'r_opp', 'L_opp', 't_pass', 'B_tot', 'B_tot_G', 'theta']
        missing = [k for k in required_keys if self.results.get(k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        for k in required_keys:
            val = self.results[k]
            if not isinstance(val, (int, float)):
                return (False, f"Key '{k}' is not numeric: type={type(val).__name__}")
        return (True, "All required keys present and numeric")

    def test_interface_positivity(self):
        """Basic positivity check for lengths, times, magnetic field, angle."""
        checks = [
            (self.r_own, 'r_own'),
            (self.L_own, 'L_own'),
            (self.r_opp, 'r_opp'),
            (self.L_opp, 'L_opp'),
            (self.t_pass, 't_pass'),
            (self.B_tot, 'B_tot'),
            (self.theta, 'theta')
        ]
        for val, name in checks:
            if val is None:
                continue
            if val <= 0:
                return (False, f"Non-positive value for {name}: {val}")
        return (True, "All present results are positive")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_length_contraction_own(self):
        """Own bunch length in its rest frame must be gamma * L_lab."""
        if self.L_own is None or self.gamma is None or self.L_lab is None:
            return (False, "SKIP: required givens missing")
        expected = self.gamma * self.L_lab
        if math.isclose(self.L_own, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L_own = {self.L_own:.3e} matches gamma*L_lab = {expected:.3e}")
        else:
            return (False, f"L_own = {self.L_own:.3e} but expected {expected:.3e} "
                          f"(gamma={self.gamma:.1e}, L_lab={self.L_lab:.1e})")

    def test_invariant_no_transverse_contraction(self):
        """Transverse dimensions (radius) must be unchanged in any frame."""
        if self.r_own is None or self.r_opp is None or self.R is None:
            return (False, "SKIP: radius values missing")
        ok_own = math.isclose(self.r_own, self.R, rel_tol=1e-2, abs_tol=1e-12)
        ok_opp = math.isclose(self.r_opp, self.R, rel_tol=1e-2, abs_tol=1e-12)
        if ok_own and ok_opp:
            return (True, f"r_own = {self.r_own:.2e}, r_opp = {self.r_opp:.2e} both equal to R = {self.R:.2e}")
        else:
            msg = ""
            if not ok_own:
                msg += f"r_own = {self.r_own:.2e} != R = {self.R:.2e}; "
            if not ok_opp:
                msg += f"r_opp = {self.r_opp:.2e} != R = {self.R:.2e}; "
            return (False, msg.rstrip('; '))

    def test_invariant_opposite_length_contraction(self):
        """Opposite bunch length in observer's frame should be L_lab/(2*gamma) (approx)."""
        if self.L_opp is None or self.gamma is None or self.L_lab is None:
            return (False, "SKIP: required givens missing")
        expected = self.L_lab / (2.0 * self.gamma)
        if math.isclose(self.L_opp, expected, rel_tol=5e-2, abs_tol=1e-2):
            return (True, f"L_opp = {self.L_opp:.3e} matches L_lab/(2*gamma) = {expected:.3e}")
        else:
            return (False, f"L_opp = {self.L_opp:.3e} but expected ~ {expected:.3e} "
                          f"(L_lab={self.L_lab:.1e}, gamma={self.gamma:.1e})")

    def test_invariant_passage_time(self):
        """Passage time in observer's frame should be (L_own + L_opp)/c."""
        if (self.t_pass is None or self.L_own is None or
            self.L_opp is None or self.c is None):
            return (False, "SKIP: required values missing")
        expected = (self.L_own + self.L_opp) / self.c
        if math.isclose(self.t_pass, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t_pass = {self.t_pass:.3e} matches (L_own+L_opp)/c = {expected:.3e}")
        else:
            return (False, f"t_pass = {self.t_pass:.3e} but expected {expected:.3e} "
                          f"(L_own={self.L_own:.3e}, L_opp={self.L_opp:.3e})")

    def test_invariant_magnetic_field_structure(self):
        """Magnetic field at r=R from one bunch: B_one = mu0*N*e*c/(2pi*R*L_lab).
           Total B_tot should be 2*B_one because both bunches contribute equally."""
        required = [self.B_tot, self.mu0, self.N, self.e, self.c, self.R, self.L_lab]
        if any(v is None for v in required):
            return (False, "SKIP: required values missing")
        B_one_expected = (self.mu0 * self.N * self.e * self.c) / (2.0 * math.pi * self.R * self.L_lab)
        B_tot_expected = 2.0 * B_one_expected
        if math.isclose(self.B_tot, B_tot_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"B_tot = {self.B_tot:.3e} T matches 2*B_one = {B_tot_expected:.3e} T")
        else:
            return (False, f"B_tot = {self.B_tot:.3e} T but expected {B_tot_expected:.3e} T "
                          f"(from mu0, N, e, c, R, L_lab)")

    def test_invariant_deflection_angle(self):
        """Deflection angle in impulse approximation: theta ≈ e*B_tot*L_lab/(gamma*m_e*c)."""
        required = [self.theta, self.e, self.B_tot, self.L_lab, self.gamma, self.m_e, self.c]
        if any(v is None for v in required):
            return (False, "SKIP: required values missing")
        expected = (self.e * self.B_tot * self.L_lab) / (self.gamma * self.m_e * self.c)
        if math.isclose(self.theta, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"theta = {self.theta:.3e} rad matches impulse approx = {expected:.3e}")
        else:
            return (False, f"theta = {self.theta:.3e} rad but expected {expected:.3e} rad "
                          f"(from B_tot, L_lab, gamma, m_e)")

    # ------------------------------------------------------------------
    # Modeling‑quality checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_B_tot_G_conversion(self):
        """B_tot_G should be B_tot * 1e4 (Tesla → Gauss)."""
        if self.B_tot is None or self.B_tot_G is None:
            return (False, "SKIP: B_tot or B_tot_G missing")
        expected_G = self.B_tot * 1e4
        if math.isclose(self.B_tot_G, expected_G, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"B_tot_G = {self.B_tot_G:.3e} G matches B_tot*1e4 = {expected_G:.3e} G")
        else:
            return (False, f"B_tot_G = {self.B_tot_G:.3e} G but expected {expected_G:.3e} G")

    def test_modeling_pseudocode_consistency_own_dimensions(self):
        """Own bunch dimensions must reflect rest frame: radius unchanged, length expanded by gamma."""
        if self.r_own is None or self.L_own is None or self.R is None or self.L_lab is None:
            return (False, "SKIP: dimensions missing")
        ok = True
        msg_parts = []
        if not math.isclose(self.r_own, self.R, rel_tol=1e-2, abs_tol=1e-12):
            ok = False
            msg_parts.append(f"r_own ({self.r_own:.2e}) != R ({self.R:.2e})")
        if not ok:
            return (False, "; ".join(msg_parts))
        return (True, "Own bunch dimensions correctly reflect rest frame (r unchanged, L expanded by gamma)")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_L_own_increases_with_gamma(self):
        """If gamma increases, L_own (= gamma * L_lab) must increase."""
        if self.L_own is None or self.gamma is None or self.L_lab is None:
            return (False, "SKIP: required values missing")
        # Baseline L_own from results
        L_own_baseline = self.L_own
        # Perturb gamma up by 10%
        gamma_perturbed = self.gamma * 1.1
        L_own_perturbed = gamma_perturbed * self.L_lab
        if L_own_perturbed > L_own_baseline:
            return (True, f"L_own increases with gamma: perturbed = {L_own_perturbed:.3e} > baseline = {L_own_baseline:.3e}")
        else:
            return (False, f"L_own does not increase: perturbed = {L_own_perturbed:.3e} <= baseline = {L_own_baseline:.3e}")

    def test_monotonicity_L_opp_decreases_with_gamma(self):
        """If gamma increases, L_opp (~ L_lab/(2*gamma)) must decrease."""
        if self.L_opp is None or self.gamma is None or self.L_lab is None:
            return (False, "SKIP: required values missing")
        L_opp_baseline = self.L_opp
        gamma_perturbed = self.gamma * 1.1
        L_opp_perturbed = self.L_lab / (2.0 * gamma_perturbed)
        if L_opp_perturbed < L_opp_baseline:
            return (True, f"L_opp decreases with gamma: perturbed = {L_opp_perturbed:.3e} < baseline = {L_opp_baseline:.3e}")
        else:
            return (False, f"L_opp does not decrease: perturbed = {L_opp_perturbed:.3e} >= baseline = {L_opp_baseline:.3e}")

    def test_monotonicity_B_tot_decreases_with_R(self):
        """If R increases, B_tot (∝ 1/(R)) must decrease."""
        required = [self.B_tot, self.mu0, self.N, self.e, self.c, self.R, self.L_lab]
        if any(v is None for v in required):
            return (False, "SKIP: required values missing")
        B_tot_baseline = self.B_tot
        # Perturb R up by 10%
        R_perturbed = self.R * 1.1
        # Recompute B_one with perturbed R, then double for total
        B_one_perturbed = (self.mu0 * self.N * self.e * self.c) / (2.0 * math.pi * R_perturbed * self.L_lab)
        B_tot_perturbed = 2.0 * B_one_perturbed
        if B_tot_perturbed < B_tot_baseline:
            return (True, f"B_tot decreases with R: perturbed = {B_tot_perturbed:.3e} < baseline = {B_tot_baseline:.3e}")
        else:
            return (False, f"B_tot does not decrease: perturbed = {B_tot_perturbed:.3e} >= baseline = {B_tot_baseline:.3e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only ONE, uses no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_theta_range(self):
        """Deflection angle should be a small positive angle (plausible range 1e-5 .. 1e-1 rad)."""
        if self.theta is None:
            return (False, "SKIP: theta missing")
        if self.theta <= 0:
            return (False, f"theta = {self.theta:.3e} should be positive")
        if not math.isfinite(self.theta):
            return (False, "theta is not finite")
        if self.theta < 1e-5 or self.theta > 1e-1:
            return (False, f"theta = {self.theta:.3e} is outside plausible range (1e-5 .. 1e-1)")
        return (True, f"theta = {self.theta:.3e} rad is plausible (positive, finite, small)")

    # ------------------------------------------------------------------
    # Master runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)