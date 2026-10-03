import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution to the neutron star radius estimation problem.
    Checks interface, physical invariants, modeling consistency, monotonicity,
    and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        """
        Extract all relevant data safely.  No numeric defaults.
        """
        # Try to access results dictionary first, then givens.
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}

        # Required result quantities
        self.R_cm = results.get('R_cm', None)
        self.R_m  = results.get('R_m',  None)

        # Input parameters (from givens or results)
        self.M    = (results.get('M', None) or givens.get('M', None))
        self.m_n  = (results.get('m_n', None) or givens.get('m_n', None))
        self.hbar = (results.get('hbar', None) or givens.get('hbar', None))
        self.G    = (results.get('G', None) or givens.get('G', None))

        # Explanation steps (for reference, not directly used in basic checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', []) or []

        # Physical constants needed for derived checks
        self.c_cgs = 3.0e10   # speed of light in cm/s

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_R_cm_exists(self):
        """Check that R_cm is provided and is numeric."""
        if self.R_cm is None:
            return (False, "R_cm not provided.")
        try:
            val = float(self.R_cm)
            return (True, f"R_cm = {val}")
        except (TypeError, ValueError) as e:
            return (False, f"R_cm is not numeric: {e}")

    def test_interface_R_m_exists(self):
        """Check that R_m is provided and is numeric."""
        if self.R_m is None:
            return (False, "R_m not provided.")
        try:
            val = float(self.R_m)
            return (True, f"R_m = {val}")
        except (TypeError, ValueError) as e:
            return (False, f"R_m is not numeric: {e}")

    def test_interface_R_m_converted_correctly(self):
        """R_m should equal R_cm divided by 100 (unit conversion)."""
        if self.R_cm is None or self.R_m is None:
            return (False, "SKIP: missing R_cm or R_m")
        expected_m = float(self.R_cm) / 100.0
        if math.isclose(float(self.R_m), expected_m, rel_tol=1e-5):
            return (True, f"R_m = {self.R_m} ≈ R_cm/100")
        else:
            return (False, f"R_m = {self.R_m} but expected approx {expected_m}")

    # ------------------------------------------------------------------
    # Invariant tests (physics-based, lenient tolerance)
    # ------------------------------------------------------------------
    def test_invariant_hydrostatic_equilibrium(self):
        """Degeneracy pressure ≃ gravitational pressure at given radius."""
        if any(v is None for v in (self.R_cm, self.M, self.m_n, self.hbar, self.G)):
            return (False, "SKIP: missing one or more required parameters")

        R = float(self.R_cm)
        M = float(self.M)
        m_n = float(self.m_n)
        hbar = float(self.hbar)
        G = float(self.G)

        # Number density (uniform)
        V = (4.0/3.0) * math.pi * R**3
        n = M / (m_n * V)

        # Degeneracy pressure (nonrelativistic)
        P_deg = (hbar**2 / (5 * m_n)) * (3 * math.pi**2)**(2.0/3.0) * n**(5.0/3.0)

        # Gravitational pressure (order-of-magnitude from uniform sphere)
        P_grav = (3.0 / (8.0 * math.pi)) * G * M**2 / R**4

        if math.isclose(P_deg, P_grav, rel_tol=1e-2):
            return (True, f"P_deg ≈ P_grav: {P_deg:.2e} vs {P_grav:.2e} dyn/cm^2")
        else:
            return (False, f"P_deg = {P_deg:.2e}, P_grav = {P_grav:.2e} – not within 1%")

    def test_invariant_radius_formula(self):
        """The derived radius should match the theoretical expression (order of magnitude)."""
        if any(v is None for v in (self.R_cm, self.M, self.m_n, self.hbar, self.G)):
            return (False, "SKIP: missing parameter(s)")

        R = float(self.R_cm)
        M = float(self.M)
        m_n = float(self.m_n)
        hbar = float(self.hbar)
        G = float(self.G)

        # Formula from explanation: R = (9π/4)^{2/3} * ħ²/(G m_n³) * (m_n/M)^{1/3}
        coeff = (9 * math.pi / 4)**(2.0/3.0)
        R_expected = coeff * (hbar**2) / (G * m_n**3) * (m_n / M)**(1.0/3.0)

        if math.isclose(R, R_expected, rel_tol=1e-2):
            return (True, f"R ≈ expected: {R:.4e} vs {R_expected:.4e} cm")
        else:
            return (False, f"R = {R:.4e}, expected = {R_expected:.4e} cm – not within 1%")

    # ------------------------------------------------------------------
    # Modeling tests (check physical assumptions)
    # ------------------------------------------------------------------
    def test_modeling_uniform_density(self):
        """Density must be positive and finite."""
        if self.R_cm is None or self.M is None:
            return (False, "SKIP: missing R_cm or M")
        R = float(self.R_cm)
        M = float(self.M)
        if R <= 0 or M <= 0:
            return (False, "R or M non-positive")
        V = (4.0/3.0) * math.pi * R**3
        density = M / V
        if density > 0 and math.isfinite(density):
            return (True, f"Density = {density:.3e} g/cm^3 (positive)")
        else:
            return (False, f"Invalid density: {density}")

    def test_modeling_fermi_velocity_nonrelativistic(self):
        """Fermi velocity must be clearly nonrelativistic (<< c)."""
        if any(v is None for v in (self.R_cm, self.M, self.m_n, self.hbar)):
            return (False, "SKIP: missing parameters for Fermi velocity")

        R = float(self.R_cm)
        M = float(self.M)
        m_n = float(self.m_n)
        hbar = float(self.hbar)

        V = (4.0/3.0) * math.pi * R**3
        n = M / (m_n * V)
        p_F = hbar * (3 * math.pi**2 * n)**(1.0/3.0)
        v_F = p_F / m_n

        threshold = 0.1 * self.c_cgs   # 10% of speed of light
        if v_F < threshold:
            return (True, f"v_F = {v_F:.3e} cm/s (<< c)")
        else:
            return (False, f"v_F = {v_F:.3e} cm/s – may be relativistic (threshold = {threshold:.2e})")

    # ------------------------------------------------------------------
    # Monotonicity tests (directional consistency)
    # ------------------------------------------------------------------
    def test_monotonicity_radius_decreases_with_mass(self):
        """R ~ M^{-1/3}: increasing mass should decrease radius (derivative check)."""
        if any(v is None for v in (self.R_cm, self.M, self.m_n, self.hbar, self.G)):
            return (False, "SKIP: missing parameters for monotonicity check")

        R = float(self.R_cm)
        M = float(self.M)
        m_n = float(self.m_n)
        hbar = float(self.hbar)
        G = float(self.G)

        # Use the theoretical expression to compute the derivative (semi-analytical)
        # For a small increase dM (10%), we expect dR negative.
        dM_fraction = 0.1
        M_perturbed = M * (1 + dM_fraction)
        # Compute R from formula again
        coeff = (9 * math.pi / 4)**(2.0/3.0)
        R_perturbed = coeff * (hbar**2) / (G * m_n**3) * (m_n / M_perturbed)**(1.0/3.0)

        if R_perturbed < R:
            return (True, f"R decreases from {R:.4e} to {R_perturbed:.4e} when M increases 10%")
        else:
            return (False, f"R does not decrease (original {R:.4e}, perturbed {R_perturbed:.4e})")

    # ------------------------------------------------------------------
    # Answer sanity (weak, only sign / order of magnitude)
    # ------------------------------------------------------------------
    def test_answer_sanity_order_of_magnitude(self):
        """Final radius (in meters) should be ~1.6×10⁴ m, order of magnitude O(10⁴)."""
        if self.R_m is None:
            return (False, "SKIP: R_m not available")
        R_m = float(self.R_m)
        # Accept between 1e3 and 1e5 (one order of magnitude around 1.6e4)
        if 1e3 <= R_m <= 1e5:
            return (True, f"R_m = {R_m:.2e} m – acceptable order of magnitude")
        else:
            return (False, f"R_m = {R_m:.2e} m – outside expected range [1e3, 1e5]")

    # ------------------------------------------------------------------
    # Test orchestrator
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, catch exceptions."""
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