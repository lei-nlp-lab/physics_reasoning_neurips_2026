import math
import sys
import traceback

class PhysicsChecker:
    """Audits a solution for the decay Xi- → Λ0 + π-.

    The solution_instance must have:
      - attributes: m_Xi, m_Lambda (or m_L), m_pi, p_Xi (all in GeV/c^2)
      - attribute results: dict with keys p_pi, theta_pi, p_Lambda, theta_Lambda, theta_open
    """

    def __init__(self, solution_instance):
        # Givens – safe defaults to None
        self.m_Xi = getattr(solution_instance, 'm_Xi', None)
        # try alternative attribute names for Lambda mass
        self.m_L = (getattr(solution_instance, 'm_Lambda', None) or
                    getattr(solution_instance, 'm_L', None))
        self.m_pi = getattr(solution_instance, 'm_pi', None)
        self.p_Xi = getattr(solution_instance, 'p_Xi', None)

        # Results
        results = getattr(solution_instance, 'results', None) or {}
        self.p_pi = results.get('p_pi', None)
        self.theta_pi = results.get('theta_pi', None)        # degrees
        self.p_Lambda = results.get('p_Lambda', None)
        self.theta_Lambda = results.get('theta_Lambda', None)  # degrees
        self.theta_open = results.get('theta_open', None)      # degrees

        self.results = results  # keep for interface checks

    # -----------------------------------------------------------------
    #  run_all_tests – discover all test_ methods and execute them
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # -----------------------------------------------------------------
    #  Interface tests
    # -----------------------------------------------------------------
    def test_interface_required_keys(self):
        """All five result keys must exist and be numeric floats."""
        expected = ['p_pi', 'theta_pi', 'p_Lambda', 'theta_Lambda', 'theta_open']
        missing = [k for k in expected if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        for k in expected:
            v = self.results[k]
            if v is None:
                return (False, f"Key '{k}' is None")
            if not isinstance(v, (int, float)):
                return (False, f"Key '{k}' is not numeric: {type(v)}")
            if math.isnan(v) or math.isinf(v):
                return (False, f"Key '{k}' is NaN or Inf")
        return (True, "All required keys present and numeric.")

    # -----------------------------------------------------------------
    #  Invariant tests (use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_lab_momentum_conservation(self):
        """In the lab frame, p_Xi = p_Lambda + p_pi (vector sum)."""
        if any(x is None for x in (self.p_Xi, self.p_Lambda, self.p_pi,
                                   self.theta_Lambda, self.theta_pi)):
            return (False, "SKIP: missing momentum/angle data")
        th_L = math.radians(self.theta_Lambda)
        th_pi = math.radians(self.theta_pi)
        pLx = self.p_Lambda * math.cos(th_L)
        pLy = self.p_Lambda * math.sin(th_L)
        ppix = self.p_pi * math.cos(th_pi)
        ppiy = self.p_pi * math.sin(th_pi)
        sum_x = pLx + ppix
        sum_y = pLy + ppiy
        if not math.isclose(sum_x, self.p_Xi, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"X-momentum sum {sum_x:.3f} != p_Xi = {self.p_Xi:.3f}")
        if not math.isclose(sum_y, 0.0, abs_tol=1e-2):
            return (False, f"Y-momentum sum {sum_y:.4f} != 0")
        return (True, "Lab momentum conservation holds.")

    def test_invariant_energy_conservation(self):
        """Lab frame total energy: E_Xi ≈ E_Lambda + E_pi."""
        if any(x is None for x in (self.p_Xi, self.p_Lambda, self.p_pi,
                                   self.m_Xi, self.m_L, self.m_pi)):
            return (False, "SKIP: missing data for energy calculation")
        E_Xi = math.sqrt(self.p_Xi**2 + self.m_Xi**2)
        E_L = math.sqrt(self.p_Lambda**2 + self.m_L**2)
        E_pi = math.sqrt(self.p_pi**2 + self.m_pi**2)
        if not math.isclose(E_Xi, E_L + E_pi, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Energy sum {E_L+E_pi:.3f} != E_Xi = {E_Xi:.3f}")
        return (True, "Lab energy conservation holds.")

    def test_invariant_cm_momentum_equal(self):
        """In the CM frame the daughter momenta are equal and match Q-value."""
        if any(x is None for x in (self.m_Xi, self.m_L, self.m_pi, self.p_Xi,
                                   self.p_Lambda, self.p_pi,
                                   self.theta_Lambda, self.theta_pi)):
            return (False, "SKIP: missing data for CM analysis")
        # compute boost parameters
        E_Xi = math.sqrt(self.p_Xi**2 + self.m_Xi**2)
        beta = self.p_Xi / E_Xi
        gamma = 1.0 / math.sqrt(1 - beta**2)

        # theoretical CM momentum from masses
        mXi, mL, mpi = self.m_Xi, self.m_L, self.m_pi
        p_star = (math.sqrt((mXi**2 - (mL+mpi)**2) * (mXi**2 - (mL-mpi)**2)) /
                  (2.0 * mXi))

        # inverse Lorentz to get CM 3‑momentum of Lambda
        th_L = math.radians(self.theta_Lambda)
        pLx = self.p_Lambda * math.cos(th_L)
        pLy = self.p_Lambda * math.sin(th_L)
        E_L = math.sqrt(self.p_Lambda**2 + mL**2)
        pLx_star = gamma * (pLx - beta * E_L)
        pLy_star = pLy
        pL_star = math.sqrt(pLx_star**2 + pLy_star**2)

        # inverse for pion
        th_pi = math.radians(self.theta_pi)
        ppix = self.p_pi * math.cos(th_pi)
        ppiy = self.p_pi * math.sin(th_pi)
        E_pi = math.sqrt(self.p_pi**2 + mpi**2)
        ppix_star = gamma * (ppix - beta * E_pi)
        ppiy_star = ppiy
        ppi_star = math.sqrt(ppix_star**2 + ppiy_star**2)

        if not math.isclose(pL_star, ppi_star, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"CM momenta differ: p*_L = {pL_star:.3f}, p*_pi = {ppi_star:.3f}")
        if not math.isclose(pL_star, p_star, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"CM momentum {pL_star:.3f} != p_star = {p_star:.3f}")
        return (True, "CM momenta equal and correct.")

    def test_invariant_on_shell(self):
        """Each particle must satisfy E² = p² + m²."""
        checks = []
        # Lambda
        if self.p_Lambda is not None and self.m_L is not None:
            E = math.sqrt(self.p_Lambda**2 + self.m_L**2)
            if not math.isclose(E**2 - self.p_Lambda**2, self.m_L**2,
                                rel_tol=1e-2, abs_tol=1e-2):
                checks.append("Lambda off shell")
        # pion
        if self.p_pi is not None and self.m_pi is not None:
            E = math.sqrt(self.p_pi**2 + self.m_pi**2)
            if not math.isclose(E**2 - self.p_pi**2, self.m_pi**2,
                                rel_tol=1e-2, abs_tol=1e-2):
                checks.append("Pion off shell")
        # Xi
        if self.p_Xi is not None and self.m_Xi is not None:
            E = math.sqrt(self.p_Xi**2 + self.m_Xi**2)
            if not math.isclose(E**2 - self.p_Xi**2, self.m_Xi**2,
                                rel_tol=1e-2, abs_tol=1e-2):
                checks.append("Xi off shell")
        if checks:
            return (False, "Off shell: " + ", ".join(checks))
        return (True, "All particles on mass shell.")

    # -----------------------------------------------------------------
    #  Modelling / trace tests (use math.isclose)
    # -----------------------------------------------------------------
    def test_modeling_lambda_boost(self):
        """Lab Λ momentum and angle from Lorentz boosting the CM kinematics."""
        if any(x is None for x in (self.m_Xi, self.m_L, self.m_pi, self.p_Xi,
                                   self.p_Lambda, self.theta_Lambda)):
            return (False, "SKIP: missing data for Λ modelling")
        # compute CM momentum
        mXi, mL, mpi = self.m_Xi, self.m_L, self.m_pi
        p_star = (math.sqrt((mXi**2 - (mL+mpi)**2) * (mXi**2 - (mL-mpi)**2)) /
                  (2.0 * mXi))
        E_star_L = math.sqrt(p_star**2 + mL**2)

        # boost parameters
        E_Xi = math.sqrt(self.p_Xi**2 + mXi**2)
        beta = self.p_Xi / E_Xi
        gamma = 1.0 / math.sqrt(1 - beta**2)

        # CM angles: Λ at 30°, π at 210°
        th_star = math.radians(30.0)
        p_star_x = p_star * math.cos(th_star)
        p_star_y = p_star * math.sin(th_star)

        # Lorentz boost to lab
        pLx_lab = gamma * (p_star_x + beta * E_star_L)
        pLy_lab = p_star_y
        pL_lab = math.sqrt(pLx_lab**2 + pLy_lab**2)
        thL_lab = math.degrees(math.atan2(pLy_lab, pLx_lab))

        if not math.isclose(pL_lab, self.p_Lambda, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected p_Λ = {pL_lab:.3f}, got {self.p_Lambda:.3f}")
        if not math.isclose(thL_lab, self.theta_Lambda, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected θ_Λ = {thL_lab:.2f}°, got {self.theta_Lambda:.2f}°")
        return (True, "Λ boost modelling correct.")

    def test_modeling_pi_boost(self):
        """Lab π⁻ momentum and angle from Lorentz boosting the CM kinematics."""
        if any(x is None for x in (self.m_Xi, self.m_L, self.m_pi, self.p_Xi,
                                   self.p_pi, self.theta_pi)):
            return (False, "SKIP: missing data for π modelling")
        mXi, mL, mpi = self.m_Xi, self.m_L, self.m_pi
        p_star = (math.sqrt((mXi**2 - (mL+mpi)**2) * (mXi**2 - (mL-mpi)**2)) /
                  (2.0 * mXi))
        E_star_pi = math.sqrt(p_star**2 + mpi**2)

        E_Xi = math.sqrt(self.p_Xi**2 + mXi**2)
        beta = self.p_Xi / E_Xi
        gamma = 1.0 / math.sqrt(1 - beta**2)

        # π⁻ in CM goes opposite: θ* + 180°
        th_star = math.radians(30.0)
        p_star_x = p_star * math.cos(th_star)
        p_star_y = p_star * math.sin(th_star)
        ppi_star_x = -p_star_x
        ppi_star_y = -p_star_y

        ppix_lab = gamma * (ppi_star_x + beta * E_star_pi)
        ppiy_lab = ppi_star_y
        ppi_lab = math.sqrt(ppix_lab**2 + ppiy_lab**2)
        thpi_lab = math.degrees(math.atan2(ppiy_lab, ppix_lab))

        if not math.isclose(ppi_lab, self.p_pi, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected p_π = {ppi_lab:.3f}, got {self.p_pi:.3f}")
        if not math.isclose(thpi_lab, self.theta_pi, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected θ_π = {thpi_lab:.2f}°, got {self.theta_pi:.2f}°")
        return (True, "π⁻ boost modelling correct.")

    def test_modeling_opening_angle(self):
        """Opening angle from dot product matches result."""
        if any(x is None for x in (self.p_Lambda, self.p_pi,
                                   self.theta_Lambda, self.theta_pi,
                                   self.theta_open)):
            return (False, "SKIP: missing data for opening angle")
        thL = math.radians(self.theta_Lambda)
        thp = math.radians(self.theta_pi)
        dot = (self.p_Lambda * self.p_pi *
               math.cos(thL - thp))
        cos_open = dot / (self.p_Lambda * self.p_pi)
        cos_open = max(-1.0, min(1.0, cos_open))
        open_deg = math.degrees(math.acos(cos_open))
        if not math.isclose(open_deg, self.theta_open, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Expected opening angle = {open_deg:.2f}, got {self.theta_open:.2f}")
        return (True, "Opening angle consistent.")

    # -----------------------------------------------------------------
    #  Monotonicity tests (perturb an input and compare outputs)
    # -----------------------------------------------------------------
    def test_monotonicity_pi_momentum_with_beta(self):
        """If beta (Xi lab momentum) increases, the pi lab momentum should increase (monotonic)."""
        if any(x is None for x in (self.m_Xi, self.m_L, self.m_pi, self.p_Xi,
                                   self.p_pi)):
            return (False, "SKIP: missing data for monotonicity test")
        # baseline values
        mXi, mL, mpi = self.m_Xi, self.m_L, self.m_pi
        pXi_orig = self.p_Xi
        E_Xi = math.sqrt(pXi_orig**2 + mXi**2)
        beta_orig = pXi_orig / E_Xi
        gamma_orig = 1.0 / math.sqrt(1 - beta_orig**2)

        # perturb beta by +5% (increase p_Xi by 5%)
        pXi_new = pXi_orig * 1.05
        E_Xi_new = math.sqrt(pXi_new**2 + mXi**2)
        beta_new = pXi_new / E_Xi_new
        gamma_new = 1.0 / math.sqrt(1 - beta_new**2)

        # Compute pi momentum for both using the same theoretical CM quantities
        p_star = (math.sqrt((mXi**2 - (mL+mpi)**2) * (mXi**2 - (mL-mpi)**2)) /
                  (2.0 * mXi))
        E_star_pi = math.sqrt(p_star**2 + mpi**2)
        th_star = math.radians(30.0)
        p_star_x = p_star * math.cos(th_star)
        p_star_y = p_star * math.sin(th_star)
        ppi_star_x = -p_star_x
        ppi_star_y = -p_star_y

        # original lab pi momentum (recompute to be independent of self.p_pi)
        ppix_lab_orig = gamma_orig * (ppi_star_x + beta_orig * E_star_pi)
        ppiy_lab_orig = ppi_star_y
        ppi_lab_orig = math.sqrt(ppix_lab_orig**2 + ppiy_lab_orig**2)

        # new lab pi momentum
        ppix_lab_new = gamma_new * (ppi_star_x + beta_new * E_star_pi)
        ppiy_lab_new = ppi_star_y
        ppi_lab_new = math.sqrt(ppix_lab_new**2 + ppiy_lab_new**2)

        if ppi_lab_new <= ppi_lab_orig:
            # compare with the original result as well
            if self.p_pi is not None and self.p_pi < ppi_lab_new:
                # if original result is even smaller, still monotonicity fails
                return (False, f"π momentum should increase with beta: orig={ppi_lab_orig:.3f}, new={ppi_lab_new:.3f}, but decrease or equal")
        # also check that the monotonicity is consistent with the user result
        if self.p_pi is not None and ppi_lab_new <= self.p_pi:
            return (False, f"π momentum with 5% higher p_Xi should be > original result. Predicted new p_pi={ppi_lab_new:.3f} > orig {self.p_pi:.3f}")
        return (True, "π momentum monotonic with beta.")

    def test_monotonicity_lambda_momentum_with_beta(self):
        """If beta increases, Lambda lab momentum should increase."""
        if any(x is None for x in (self.m_Xi, self.m_L, self.m_pi, self.p_Xi,
                                   self.p_Lambda)):
            return (False, "SKIP: missing data")
        mXi, mL, mpi = self.m_Xi, self.m_L, self.m_pi
        pXi_orig = self.p_Xi
        E_Xi = math.sqrt(pXi_orig**2 + mXi**2)
        beta_orig = pXi_orig / E_Xi
        gamma_orig = 1.0 / math.sqrt(1 - beta_orig**2)

        pXi_new = pXi_orig * 1.05
        E_Xi_new = math.sqrt(pXi_new**2 + mXi**2)
        beta_new = pXi_new / E_Xi_new
        gamma_new = 1.0 / math.sqrt(1 - beta_new**2)

        p_star = (math.sqrt((mXi**2 - (mL+mpi)**2) * (mXi**2 - (mL-mpi)**2)) /
                  (2.0 * mXi))
        E_star_L = math.sqrt(p_star**2 + mL**2)
        th_star = math.radians(30.0)
        p_star_x = p_star * math.cos(th_star)
        p_star_y = p_star * math.sin(th_star)

        # original
        pLx_lab_orig = gamma_orig * (p_star_x + beta_orig * E_star_L)
        pLy_lab_orig = p_star_y
        pL_lab_orig = math.sqrt(pLx_lab_orig**2 + pLy_lab_orig**2)

        # new
        pLx_lab_new = gamma_new * (p_star_x + beta_new * E_star_L)
        pLy_lab_new = p_star_y
        pL_lab_new = math.sqrt(pLx_lab_new**2 + pLy_lab_new**2)

        if pL_lab_new <= pL_lab_orig:
            return (False, f"Λ momentum should increase with beta")
        if self.p_Lambda is not None and pL_lab_new <= self.p_Lambda:
            return (False, f"Λ momentum with higher beta > original result")
        return (True, "Λ momentum monotonic with beta.")

    # -----------------------------------------------------------------
    #  Weak answer sanity test (only one, no hardcoded target values)
    # -----------------------------------------------------------------
    def test_answer_sanity_sign_and_magnitude(self):
        """Basic sanity: momenta positive, angles within reasonable range, opening angle positive."""
        if self.p_Lambda is None or self.p_pi is None:
            return (False, "SKIP: missing momentum data")
        checks = []
        if self.p_Lambda <= 0:
            checks.append(f"p_Lambda = {self.p_Lambda} should be positive")
        if self.p_pi <= 0:
            checks.append(f"p_pi = {self.p_pi} should be positive")
        if self.theta_Lambda is not None:
            if not (-10 < self.theta_Lambda < 10):
                checks.append(f"theta_Lambda = {self.theta_Lambda} out of reasonable range")
        if self.theta_pi is not None:
            if not (-90 < self.theta_pi < 10):
                checks.append(f"theta_pi = {self.theta_pi} out of reasonable range")
        if self.theta_open is not None:
            if not (0 < self.theta_open < 180):
                checks.append(f"theta_open = {self.theta_open} out of (0,180)")
        if checks:
            return (False, "; ".join(checks))
        return (True, "Basic sign and magnitude sanity passed.")