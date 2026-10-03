import math
import traceback

class PhysicsChecker:
    """
    Auditor for the neutrino detector problem.
    Checks the estimated daily production of Ar-37 atoms (R_day).
    Includes interface, invariant, modeling/trace, and monotonicity tests.
    """

    def __init__(self, solution_instance):
        # Safely extract main result
        results = getattr(solution_instance, 'results', {})
        self.R_day = results.get('R_day', None)

        # Safely extract givens if they exist (attributes on solution_instance or results)
        self.V = getattr(solution_instance, 'V', None)           # detector volume in L
        self.rho = getattr(solution_instance, 'rho', None)       # density g/cm^3
        self.M = getattr(solution_instance, 'M', None)           # molar mass g/mol
        self.f_Cl37 = getattr(solution_instance, 'f_Cl37', None) # isotopic fraction
        self.N_A = getattr(solution_instance, 'N_A', None)       # Avogadro number
        self.Phi = getattr(solution_instance, 'Phi', None)       # neutrino flux cm^-2 s^-1
        self.sigma = getattr(solution_instance, 'sigma', None)   # cross section cm^2

        # Additional intermediate results that might be present
        self.N_target = results.get('N_target', None)

    # ------------------------------------------------------------------
    # Helper: compute N_target from given parameters
    # ------------------------------------------------------------------
    def _compute_N_target(self, V=None, rho=None, M=None, f=None, NA=None):
        """Return N_target or None if any required given is missing."""
        # Use stored defaults if not provided
        V = V if V is not None else self.V
        rho = rho if rho is not None else self.rho
        M = M if M is not None else self.M
        f = f if f is not None else self.f_Cl37
        NA = NA if NA is not None else self.N_A
        if any(x is None for x in [V, rho, M, f, NA]):
            return None
        V_cm3 = V * 1000  # 1 L = 1000 cm^3
        n_mol = V_cm3 * rho / M
        N_molecules = n_mol * NA
        N_Cl = 4 * N_molecules
        return N_Cl * f

    def _compute_R_day(self, V=None, rho=None, M=None, f=None, NA=None, Phi=None, sigma=None):
        """Return R_day computed from given parameters, or None if missing."""
        N_target = self._compute_N_target(V, rho, M, f, NA)
        if N_target is None:
            return None
        Phi = Phi if Phi is not None else self.Phi
        sigma = sigma if sigma is not None else self.sigma
        if Phi is None or sigma is None:
            return None
        R_per_s = Phi * sigma * N_target
        return R_per_s * 86400

    # ------------------------------------------------------------------
    # Interfaces
    # ------------------------------------------------------------------
    def test_interface_R_day_exists(self):
        """R_day must be present in results."""
        if self.R_day is None:
            return (False, "R_day is missing from results")
        return (True, "R_day found in results")

    def test_interface_R_day_numeric(self):
        """R_day must be a number (int or float)."""
        if self.R_day is None:
            return (False, "SKIP: R_day not available")
        if not isinstance(self.R_day, (int, float)):
            return (False, f"R_day is {type(self.R_day).__name__}, not numeric")
        return (True, "R_day is numeric")

    # ------------------------------------------------------------------
    # Invariants (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_R_day_positive(self):
        """Production rate must be positive."""
        if self.R_day is None:
            return (False, "SKIP: R_day not available")
        if self.R_day <= 0:
            return (False, f"R_day is non-positive: {self.R_day}")
        return (True, "R_day is positive")

    def test_invariant_R_day_finite(self):
        """R_day must be finite (not inf or nan)."""
        if self.R_day is None:
            return (False, "SKIP: R_day not available")
        if math.isinf(self.R_day) or math.isnan(self.R_day):
            return (False, f"R_day is not finite: {self.R_day}")
        return (True, "R_day is finite")

    def test_invariant_production_rate_formula(self):
        """Check fundamental relation: R_day = Phi * sigma * N_target * 86400.
           Use math.isclose with lenient tolerance.
        """
        if self.R_day is None:
            return (False, "SKIP: R_day not available")
        # Try to recompute from stored givens
        expected = self._compute_R_day()
        if expected is None:
            return (False, "SKIP: insufficient givens to compute expectation")
        if math.isclose(self.R_day, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R_day {self.R_day} matches production formula (expected {expected:.2e})")
        if math.isclose(self.R_day, expected, rel_tol=0.5, abs_tol=0.5):
            return (True, f"R_day {self.R_day} within 50% of formula (expected {expected:.2e})")
        return (False, f"R_day {self.R_day} deviates from formula value {expected:.2e}")

    # ------------------------------------------------------------------
    # Modeling / trace quality checks
    # ------------------------------------------------------------------
    def test_modeling_plausible_range(self):
        """R_day should be in the ballpark of 10^2 - 10^3 (typical estimate ~480)."""
        if self.R_day is None:
            return (False, "SKIP: R_day not available")
        if not (100 <= self.R_day <= 1000):
            return (False, f"R_day {self.R_day} outside plausible range [100, 1000]")
        return (True, f"R_day {self.R_day} lies in plausible range [100,1000]")

    def test_modeling_consistency_with_derived_N_target(self):
        """If both R_day and N_target are present, check that R_day/(N_target) gives
           expected product Phi*sigma*86400 (about 2e-27)."""
        if self.R_day is None or self.N_target is None:
            return (False, "SKIP: R_day or N_target not available")
        if self.N_target <= 0:
            return (False, "N_target not positive")
        product = self.R_day / self.N_target
        if product <= 0:
            return (False, f"R_day/N_target = {product:.3e} not positive")
        if 1e-40 <= product <= 1e-20:
            return (True, f"R_day/N_target = {product:.3e} in plausible range")
        else:
            return (False, f"R_day/N_target = {product:.3e} outside plausible range [1e-40, 1e-20]")

    def test_modeling_N_target_consistency(self):
        """If N_target is provided, verify its computation from givens (if available)."""
        if self.N_target is None:
            return (False, "SKIP: N_target not available")
        N_computed = self._compute_N_target()
        if N_computed is None:
            return (False, "SKIP: cannot compute N_target from givens")
        if math.isclose(self.N_target, N_computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_target {self.N_target:.2e} matches computed value {N_computed:.2e}")
        if math.isclose(self.N_target, N_computed, rel_tol=0.5, abs_tol=0.5):
            return (True, f"N_target {self.N_target:.2e} within 50% of computed {N_computed:.2e}")
        return (False, f"N_target {self.N_target:.2e} deviates from computed {N_computed:.2e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and check output direction)
    # ------------------------------------------------------------------
    def test_monotonicity_volume_increases_R_day(self):
        """Increasing the detector volume should increase R_day."""
        # Baseline R_day from stored givens
        baseline = self._compute_R_day()
        if baseline is None:
            return (False, "SKIP: insufficient givens to compute baseline")
        # Increase volume by 10%
        if self.V is None:
            return (False, "SKIP: V not available for perturbation")
        perturbed_V = self.V * 1.10
        perturbed = self._compute_R_day(V=perturbed_V)
        if perturbed is None:
            return (False, "SKIP: perturbation failed to compute R_day")
        if perturbed > baseline * 0.99:   # allow 1% rounding down
            return (True, f"Increasing V by 10% increases R_day: {baseline:.2e} -> {perturbed:.2e}")
        return (False, f"R_day did not increase when V increased: baseline {baseline:.2e}, perturbed {perturbed:.2e}")

    def test_monotonicity_flux_increases_R_day(self):
        """Increasing neutrino flux should increase R_day."""
        baseline = self._compute_R_day()
        if baseline is None:
            return (False, "SKIP: insufficient givens to compute baseline")
        if self.Phi is None:
            return (False, "SKIP: Phi not available for perturbation")
        perturbed_Phi = self.Phi * 1.05  # 5% increase
        perturbed = self._compute_R_day(Phi=perturbed_Phi)
        if perturbed is None:
            return (False, "SKIP: perturbation failed to compute R_day")
        if perturbed > baseline * 0.99:   # allow 1% rounding down
            return (True, f"Increasing Phi by 5% increases R_day: {baseline:.2e} -> {perturbed:.2e}")
        return (False, f"R_day did not increase when Phi increased")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_R_day_not_exact(self):
        """Simple sanity: R_day must not be exactly some common wrong number like 0, 1e6, etc."""
        if self.R_day is None:
            return (False, "SKIP: R_day not available")
        if self.R_day == 0:
            return (False, "R_day is zero – impossible")
        if self.R_day > 1e6:
            return (False, f"R_day {self.R_day} is too large (order 10^6)")
        if self.R_day < 0.1:
            return (False, f"R_day {self.R_day} is too small (< 0.1)")
        return (True, f"R_day {self.R_day} passes simple sanity checks")

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
                tb = traceback.format_exc()
                msg = f"{name} CRASHED: {e}\n{tb}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)