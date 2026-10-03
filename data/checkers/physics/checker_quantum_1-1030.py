import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics checker for the harmonic oscillator electron excitation problem.
    Audits the solution output delta_E_eV for consistency with the virial theorem
    and the ground-state variance relation.
    """

    # Standard physical constants (SI)
    STANDARD_CONSTANTS = {
        'hbar': 1.054571817e-34,   # J·s
        'm_e': 9.1093837015e-31,  # kg
        'e': 1.602176634e-19,     # J/eV
    }

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        self.delta_E_eV = results.get('delta_E_eV', None)

        # Extract givens (problem data)
        self.dx = getattr(solution_instance, 'dx', None) or results.get('dx', None)
        self.hbar = getattr(solution_instance, 'hbar', None) or results.get('hbar', None)
        self.m_e = getattr(solution_instance, 'm', None) or results.get('m', None)
        self.eV = getattr(solution_instance, 'e', None) or results.get('e', None)

        # Fallback to solution_instance.givens dict if available
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
            if self.dx is None:
                self.dx = givens.get('dx', None)
            if self.hbar is None:
                self.hbar = givens.get('hbar', None)
            if self.m_e is None:
                self.m_e = givens.get('m', None) or givens.get('m_e', None)
            if self.eV is None:
                self.eV = givens.get('e', None)

        # Store intermediate results if available (not required)
        self.omega = results.get('omega', None)
        self.delta_E_J = results.get('delta_E_J', None)

    # --------------------------------------------------------------------------
    # Helper to access constants with standard fallback
    # --------------------------------------------------------------------------
    def _get_const(self, name):
        # Return the value from the solution if present, else from standard constants
        v = getattr(self, name, None)
        if v is not None:
            return v
        return self.STANDARD_CONSTANTS.get(name, None)

    # ==========================================================================
    # Interface tests: check that required keys exist and have valid types
    # ==========================================================================
    def test_interface_delta_E_eV_exists(self):
        if self.delta_E_eV is None:
            return (False, "SKIP: delta_E_eV not found in results")
        return (True, "delta_E_eV is present")

    def test_interface_delta_E_eV_is_numeric(self):
        if self.delta_E_eV is None:
            return (False, "SKIP: delta_E_eV missing")
        if not isinstance(self.delta_E_eV, (int, float)):
            return (False, f"delta_E_eV is not numeric: {type(self.delta_E_eV)}")
        return (True, "delta_E_eV is numeric")

    # ==========================================================================
    # Invariant tests (theorem-based physical sanity; use math.isclose)
    # ==========================================================================
    def test_invariant_energy_from_variance(self):
        """
        Check that ΔE = ħ^2 / (2 m Δx^2) holds.
        """
        dx = self.dx
        delta_E = self.delta_E_eV
        if dx is None:
            return (False, "SKIP: dx (standard deviation) not provided")
        if delta_E is None:
            return (False, "SKIP: delta_E_eV not available")

        hbar = self._get_const('hbar')
        m_e = self._get_const('m_e')
        eV = self._get_const('eV')
        if None in (hbar, m_e, eV):
            return (False, "SKIP: needed physical constants missing")

        expected_delta_E_J = hbar**2 / (2 * m_e * dx**2)
        expected_delta_E_eV = expected_delta_E_J / eV

        if math.isclose(delta_E, expected_delta_E_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ΔE = {delta_E} eV matches expected {expected_delta_E_eV:.3f} eV")
        else:
            return (False, f"ΔE = {delta_E} eV, expected ≈ {expected_delta_E_eV:.3f} eV (rel. diff={abs(delta_E-expected_delta_E_eV)/expected_delta_E_eV:.2%})")

    def test_invariant_omega_consistency(self):
        """
        If ω is provided, check that ΔE = ħ ω.
        """
        omega = self.omega
        delta_E = self.delta_E_eV
        if omega is None or delta_E is None:
            return (False, "SKIP: omega or delta_E_eV not available")
        hbar = self._get_const('hbar')
        eV = self._get_const('eV')
        if hbar is None or eV is None:
            return (False, "SKIP: constants missing")
        expected_delta_E_J = hbar * omega
        expected_delta_E_eV = expected_delta_E_J / eV
        if math.isclose(delta_E, expected_delta_E_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ΔE = ħω holds (ΔE={delta_E} eV, ħω={expected_delta_E_eV:.3f} eV)")
        else:
            return (False, f"ΔE = {delta_E} eV, but ħω ≈ {expected_delta_E_eV:.3f} eV")

    def test_invariant_virial_ground_state(self):
        """
        Check that the virial relation for the ground state is satisfied:
        ħω/2 = m ω^2 ⟨x^2⟩.  Here we check the derived ω against dx.
        """
        dx = self.dx
        if dx is None:
            return (False, "SKIP: dx missing")
        hbar = self._get_const('hbar')
        m_e = self._get_const('m_e')
        if hbar is None or m_e is None:
            return (False, "SKIP: constants missing")

        # From virial: ω = ħ/(2 m Δx^2)
        omega_from_virial = hbar / (2 * m_e * dx**2)
        # If solution provided ω, compare
        if self.omega is not None:
            omega = self.omega
            if math.isclose(omega, omega_from_virial, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Virial relation holds: ω = {omega} rad/s matches predicted {omega_from_virial:.3e} rad/s")
            else:
                return (False, f"ω = {omega} rad/s, but virial predicts {omega_from_virial:.3e} rad/s")
        else:
            # If no ω, just report the predicted value (not a failure)
            return (True, f"SKIP (no ω provided), virial predicts ω = {omega_from_virial:.3e} rad/s")

    # ==========================================================================
    # Modeling / trace tests (use explanation_steps to decide what to audit)
    # ==========================================================================
    def test_modeling_conversion_consistency(self):
        """
        If delta_E_J is present, check that delta_E_eV = delta_E_J / eV.
        """
        delta_E_J = self.delta_E_J
        delta_E_eV = self.delta_E_eV
        if delta_E_J is None or delta_E_eV is None:
            return (False, "SKIP: delta_E_J or delta_E_eV not available")
        eV = self._get_const('eV')
        if eV is None:
            return (False, "SKIP: eV conversion factor missing")
        derived_eV = delta_E_J / eV
        if math.isclose(delta_E_eV, derived_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Joule-to-eV conversion is correct")
        else:
            return (False, f"delta_E_eV = {delta_E_eV} but delta_E_J/eV = {derived_eV:.3f}")

    def test_modeling_positive_finite(self):
        """
        Basic quality: result must be positive, finite, not NaN.
        """
        if self.delta_E_eV is None:
            return (False, "SKIP: delta_E_eV missing")
        if math.isfinite(self.delta_E_eV) and self.delta_E_eV > 0:
            return (True, "Result is positive and finite")
        else:
            return (False, f"Result not positive/finite: {self.delta_E_eV}")

    # ==========================================================================
    # Monotonicity test (perturb dx and check directional trend)
    # ==========================================================================
    def test_monotonicity_energy_decreases_with_increasing_dx(self):
        """
        If dx increases, ΔE should decrease (derivative negative).
        We check by perturbing dx upward by 1% and computing expected ΔE from formula.
        """
        dx = self.dx
        delta_E = self.delta_E_eV
        if dx is None or delta_E is None:
            return (False, "SKIP: dx or delta_E_eV not available")
        hbar = self._get_const('hbar')
        m_e = self._get_const('m_e')
        eV = self._get_const('eV')
        if None in (hbar, m_e, eV):
            return (False, "SKIP: constants missing")

        # Perturb dx upward by 1%
        dx_pert = dx * 1.01
        expected_delta_E_pert = hbar**2 / (2 * m_e * dx_pert**2) / eV
        # Compare perturbed expected value to original solver's output (baseline)
        if expected_delta_E_pert < delta_E:
            return (True, f"Monotonicity holds: increasing dx by 1% decreases ΔE ({delta_E:.3f} → {expected_delta_E_pert:.3f} eV)")
        else:
            return (False, f"Monotonicity violated: increasing dx should reduce ΔE, but expected perturbed ΔE = {expected_delta_E_pert:.3f} >= {delta_E:.3f} eV")

    # ==========================================================================
    # Answer sanity (weak, at most one, no math.isclose)
    # ==========================================================================
    def test_answer_sanity_broad_range(self):
        """
        The answer should be in the ballpark: between 0.1 eV and 100 eV.
        """
        if self.delta_E_eV is None:
            return (False, "SKIP: delta_E_eV missing")
        if 0.1 < self.delta_E_eV < 100:
            return (True, f"Answer {self.delta_E_eV:.3f} eV lies in plausible range (0.1–100 eV)")
        else:
            return (False, f"Answer {self.delta_E_eV:.3f} eV outside plausible range (0.1–100 eV)")

    # ==========================================================================
    # Run all tests
    # ==========================================================================
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods whose names start with 'test_'
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