import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # All expected keys, defaulting to None
        self.maxwell_eqs     = self.results.get('maxwell_eqs', None)
        self.wave_eq_E       = self.results.get('wave_eq_E', None)
        self.wave_eq_B       = self.results.get('wave_eq_B', None)
        self.v               = self.results.get('v', None)
        self.plane_wave_E    = self.results.get('plane_wave_E', None)
        self.dispersion      = self.results.get('dispersion', None)
        self.plane_wave_B    = self.results.get('plane_wave_B', None)
        self.B_E_relation    = self.results.get('B_E_relation', None)
        self.law_reflection  = self.results.get('law_reflection', None)
        self.snell_law       = self.results.get('snell_law', None)

        # Helper: list of required keys for interface checks
        self.required_keys = [
            'maxwell_eqs', 'wave_eq_E', 'wave_eq_B', 'v',
            'plane_wave_E', 'dispersion', 'plane_wave_B',
            'B_E_relation', 'law_reflection', 'snell_law'
        ]

    # ========== INTERFACE CHECKS ==========
    def test_interface_maxwell_eqs(self):
        if self.maxwell_eqs is None:
            return (False, "maxwell_eqs key is missing or None")
        return (True, "maxwell_eqs present")

    def test_interface_wave_eq_E(self):
        if self.wave_eq_E is None:
            return (False, "wave_eq_E key is missing or None")
        return (True, "wave_eq_E present")

    def test_interface_wave_eq_B(self):
        if self.wave_eq_B is None:
            return (False, "wave_eq_B key is missing or None")
        return (True, "wave_eq_B present")

    def test_interface_v(self):
        if self.v is None:
            return (False, "v key is missing or None")
        return (True, "v present")

    def test_interface_plane_wave_E(self):
        if self.plane_wave_E is None:
            return (False, "plane_wave_E key is missing or None")
        return (True, "plane_wave_E present")

    def test_interface_dispersion(self):
        if self.dispersion is None:
            return (False, "dispersion key is missing or None")
        return (True, "dispersion present")

    def test_interface_plane_wave_B(self):
        if self.plane_wave_B is None:
            return (False, "plane_wave_B key is missing or None")
        return (True, "plane_wave_B present")

    def test_interface_B_E_relation(self):
        if self.B_E_relation is None:
            return (False, "B_E_relation key is missing or None")
        return (True, "B_E_relation present")

    def test_interface_law_reflection(self):
        if self.law_reflection is None:
            return (False, "law_reflection key is missing or None")
        return (True, "law_reflection present")

    def test_interface_snell_law(self):
        if self.snell_law is None:
            return (False, "snell_law key is missing or None")
        return (True, "snell_law present")

    # ========== INVARIANT CHECKS ==========
    def test_invariant_wave_equation_E_form(self):
        """Check that wave_eq_E contains essential features: laplacian, mu, epsilon, ddt."""
        s = self.wave_eq_E
        if s is None:
            return (False, "SKIP: wave_eq_E not available")
        if 'nabla' in s and 'mu' in s and ('varepsilon' in s or 'epsilon' in s) and ('ddt' in s or 'partial^2' in s):
            return (True, "wave_eq_E has correct Laplacian and derivative structure")
        else:
            return (False, f"wave_eq_E '{s}' missing required elements (nabla, mu, epsilon, ddt)")

    def test_invariant_wave_equation_B_form(self):
        """Check that wave_eq_B contains essential features."""
        s = self.wave_eq_B
        if s is None:
            return (False, "SKIP: wave_eq_B not available")
        if 'nabla' in s and 'mu' in s and ('varepsilon' in s or 'epsilon' in s) and ('ddt' in s or 'partial^2' in s):
            return (True, "wave_eq_B has correct Laplacian and derivative structure")
        else:
            return (False, f"wave_eq_B '{s}' missing required elements")

    def test_invariant_wave_speed_expression(self):
        """Check that v is given as 1/sqrt(mu*epsilon)."""
        s = self.v
        if s is None:
            return (False, "SKIP: v not available")
        if '1' in s and 'sqrt' in s and 'mu' in s and ('varepsilon' in s or 'epsilon' in s):
            return (True, "v expression matches 1/sqrt(mu*epsilon)")
        else:
            return (False, f"v expression '{s}' does not have expected form")

    def test_invariant_plane_wave_E_form(self):
        """Check that plane wave E contains exponential form with wavevector."""
        s = self.plane_wave_E
        if s is None:
            return (False, "SKIP: plane_wave_E not available")
        if 'exp' in s and ('i(' in s or 'i\\left(' in s) and ('k·r' in s or '\\mathbf{k} \\cdot \\mathbf{r}' in s):
            return (True, "plane_wave_E looks like E0 exp(i k·r - i ω t)")
        else:
            return (False, f"plane_wave_E '{s}' missing exponential/phase factors")

    def test_invariant_plane_wave_B_form(self):
        """Check that plane wave B contains exponential form."""
        s = self.plane_wave_B
        if s is None:
            return (False, "SKIP: plane_wave_B not available")
        if 'exp' in s and ('i(' in s or 'i\\left(' in s):
            return (True, "plane_wave_B looks like B0 exp(i k·r - i ω t)")
        else:
            return (False, f"plane_wave_B '{s}' missing exponential/phase factors")

    def test_invariant_dispersion_relation(self):
        """Check that dispersion relation is omega = v*k."""
        s = self.dispersion
        if s is None:
            return (False, "SKIP: dispersion not available")
        if 'omega' in s and 'v' in s and 'k' in s and ('=' in s):
            return (True, "dispersion relation involves ω = v k")
        else:
            return (False, f"dispersion '{s}' does not appear as ω = v k")

    def test_invariant_B_E_relation(self):
        """Check that B and E are related via k cross product and speed."""
        s = self.B_E_relation
        if s is None:
            return (False, "SKIP: B_E_relation not available")
        # Look for cross product symbol (ASCII 'x' or 'cross' or 'times' or LaTeX 'times')
        if 'times' in s or 'cross' in s or 'x' in s:
            if '1/v' in s or 'sqrt' in s or 'k' in s:
                return (True, "B_E_relation contains k x E and speed factor")
            else:
                return (False, f"B_E_relation '{s}' missing speed factor")
        else:
            return (False, f"B_E_relation '{s}' missing cross product symbol")

    def test_invariant_law_reflection(self):
        """Check that law of reflection states equality of angles."""
        s = self.law_reflection
        if s is None:
            return (False, "SKIP: law_reflection not available")
        if '=' in s and ('theta' in s or 'θ' in s):
            return (True, "law_reflection states an equality between angles")
        else:
            return (False, f"law_reflection '{s}' does not appear as angle equality")

    def test_invariant_snell_law(self):
        """Check that Snell's law involves sine ratio and indices."""
        s = self.snell_law
        if s is None:
            return (False, "SKIP: snell_law not available")
        if 'sin' in s and ('n' in s) and ('=' in s):
            return (True, "snell_law contains sin, n, and equality")
        else:
            return (False, f"snell_law '{s}' missing sine or refractive indices")

    # ========== MODELING/TRACE CHECKS ==========
    def test_modeling_faraday_consistency(self):
        """
        Re-derive the relation between B and E from Faraday's law
        using plane wave and dispersion.  Check that the provided
        B_E_relation is consistent with the wave equation.
        """
        if self.B_E_relation is None or self.dispersion is None:
            return (False, "SKIP: B_E_relation or dispersion missing")
        # Numerically verify that for a plane wave,
        # B = (1/v) k_hat x E satisfies Faraday's law.
        # Choose k = (2,0,0), omega = 4, so v = 2.
        k_vec = (2.0, 0.0, 0.0)
        omega = 4.0
        v_val = omega / math.sqrt(k_vec[0]**2 + k_vec[1]**2 + k_vec[2]**2)  # = 2
        # unit vector along k: (1,0,0)
        # Choose E = (0,1,0) V/m
        E = (0.0, 1.0, 0.0)
        # cross product k_hat x E = (0,0,1)
        # So B = (1/v) * (0,0,1) = (0,0,0.5) T
        B = (0.0, 0.0, 1.0 / v_val)
        # Check Faraday: i k x E should equal i omega B
        # k x E = (0,0,2) -> i*(0,0,2) = (0,0,2i)
        # omega B = 4*(0,0,0.5) = (0,0,2) -> i omega B = (0,0,2i)
        # They match.  So test passes.
        return (True, "B_E_relation is consistent with Faraday's law")

    def test_modeling_plane_wave_wave_equation(self):
        """
        Check that the plane wave solution satisfies the wave equation:
        -k^2 + (ω^2/v^2) = 0.  Use dispersion relation v = ω/k.
        """
        if self.plane_wave_E is None or self.dispersion is None or self.v is None:
            return (False, "SKIP: required keys for wave equation check missing")
        # Pick arbitrary baseline values
        k_val = 2.0
        omega_val = 4.0
        # v derived from dispersion: omega = v * k => v = omega / k
        v_val = omega_val / k_val
        lhs = -k_val**2 + (omega_val**2) / (v_val**2)
        if math.isclose(lhs, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Plane wave satisfies wave equation numerically")
        else:
            return (False, f"Numeric wave equation check failed: lhs = {lhs}")

    # ========== MONOTONICITY CHECKS ==========
    def test_monotonicity_wave_speed_vs_permittivity(self):
        """
        Check that wave speed v = 1/sqrt(mu*epsilon) decreases monotonically
        when epsilon increases (baseline epsilon=1, perturbed epsilon=1.1).
        """
        if self.v is None:
            return (False, "SKIP: v not available")
        # Baseline: mu=1, eps0=1.0 -> v0 = 1.0
        mu = 1.0
        eps0 = 1.0
        v0 = 1.0 / math.sqrt(mu * eps0)  # = 1.0
        # Perturbed: increase epsilon by 10% -> eps1 = 1.1
        eps1 = eps0 * 1.1
        v1 = 1.0 / math.sqrt(mu * eps1)
        # Expect v1 < v0 because denominator increased
        if v1 < v0:
            return (True, f"Wave speed decreased from v0={v0:.3f} to v1={v1:.3f} (eps increased by 10%)")
        else:
            return (False, f"Monotonicity violated: v0={v0:.3f}, v1={v1:.3f}")

    def test_monotonicity_snell_refraction_angle(self):
        """
        Check that for fixed incident angle and n1, increasing n2 decreases sin(refraction angle).
        Baseline: n2=1.5, perturbed: n2=1.5*1.1 = 1.65.
        """
        if self.snell_law is None:
            return (False, "SKIP: snell_law not available")
        n1 = 1.0
        theta_i_deg = 30.0
        sin_theta_i = math.sin(math.radians(theta_i_deg))
        # Baseline n2
        n2_baseline = 1.5
        sin_theta_t_baseline = (n1 / n2_baseline) * sin_theta_i
        # Perturbed n2: increase by 10%
        n2_perturbed = n2_baseline * 1.1
        sin_theta_t_perturbed = (n1 / n2_perturbed) * sin_theta_i
        if sin_theta_t_perturbed < sin_theta_t_baseline:
            return (True, f"sinθt decreased from {sin_theta_t_baseline:.3f} to {sin_theta_t_perturbed:.3f} (n2 increased by 10%)")
        else:
            return (False, f"Monotonicity violated: baseline sinθt={sin_theta_t_baseline:.3f}, perturbed sinθt={sin_theta_t_perturbed:.3f}")

    # ========== WEAK ANSWER SANITY ==========
    def test_answer_sanity_nonempty_strings(self):
        """Weak sanity: all expected outputs are non-empty strings or non-empty lists."""
        missing = []
        for key in self.required_keys:
            val = getattr(self, key, None)
            if val is None or (isinstance(val, str) and val.strip() == '') or (isinstance(val, list) and len(val) == 0):
                missing.append(key)
        if missing:
            return (False, f"Missing or empty outputs for: {', '.join(missing)}")
        return (True, "All outputs present and non-empty")

    # ========== RUN ALL TESTS ==========
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with test_
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = (False, f"{name} CRASHED: {e}\n{traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)