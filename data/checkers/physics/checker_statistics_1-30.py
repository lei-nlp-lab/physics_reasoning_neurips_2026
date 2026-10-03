import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for estimating Neptune's surface temperature.
    Checks invariants: inverse-square law, blackbody equilibrium, 
    monotonicity, and modeling assumptions.
    """
    def __init__(self, solution_instance):
        # Extract results (always present in a valid solution)
        self.S_N = getattr(solution_instance, 'S_N', 
                           solution_instance.results.get('S_N', None)
                           if hasattr(solution_instance, 'results') else None)
        self.T_N = getattr(solution_instance, 'T_N',
                           solution_instance.results.get('T_N', None)
                           if hasattr(solution_instance, 'results') else None)

        # Try to extract givens (may be missing)
        self.d_E = getattr(solution_instance, 'd_E', None)
        self.d_N = getattr(solution_instance, 'd_N', None)
        self.S_E = getattr(solution_instance, 'S_E', None)
        self.sigma = getattr(solution_instance, 'sigma', None)

        # If givens are not direct attributes, check for 'givens' dict
        if self.d_E is None and hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            self.d_E = givens.get('d_E', None)
            self.d_N = givens.get('d_N', None)
            self.S_E = givens.get('S_E', None)
            self.sigma = givens.get('sigma', None)

        # Fallback: default to None for all missing values (no numeric defaults)
        if self.d_E is None:
            self.d_E = None
        if self.d_N is None:
            self.d_N = None
        if self.S_E is None:
            self.S_E = None
        if self.sigma is None:
            self.sigma = None

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        """Check that S_N and T_N exist and are numeric."""
        if self.S_N is None:
            return (False, "Missing required result: S_N")
        if self.T_N is None:
            return (False, "Missing required result: T_N")
        try:
            float(self.S_N)
            float(self.T_N)
        except (TypeError, ValueError) as e:
            return (False, f"Non-numeric values: {e}")
        return (True, "Both S_N and T_N are present and numeric.")

    # ---------- Invariant tests (theorem-based, use math.isclose) ----------
    def test_invariant_inverse_square(self):
        """Check that S_N ≈ S_E * (d_E / d_N)^2."""
        if any(v is None for v in (self.S_N, self.S_E, self.d_E, self.d_N)):
            return (False, "SKIP: Missing givens for inverse-square check")
        try:
            expected = self.S_E * (self.d_E / self.d_N) ** 2
        except Exception as e:
            return (False, f"Computation error: {e}")
        if not math.isclose(self.S_N, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S_N {self.S_N} differs from expected {expected}")
        return (True, f"S_N = {self.S_N} matches inverse-square law.")

    def test_invariant_blackbody_equilibrium(self):
        """Check that T_N^4 * 4σ ≈ S_N (emitted power density equals absorbed)."""
        if self.T_N is None or self.sigma is None:
            return (False, "SKIP: Missing T_N or sigma")
        if self.S_N is None:
            return (False, "SKIP: Missing S_N")
        try:
            emitted = 4 * self.sigma * (self.T_N ** 4)
            incoming = self.S_N  # absorbed power per unit area (A=0)
        except Exception as e:
            return (False, f"Computation error: {e}")
        if not math.isclose(incoming, emitted, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Emitted power {emitted} != absorbed {incoming}")
        return (True, "Blackbody equilibrium holds (T_N^4 * 4σ ≈ S_N).")

    def test_invariant_flux_positive(self):
        """S_N must be positive (solar flux at Neptune > 0)."""
        if self.S_N is None:
            return (False, "SKIP: S_N not available")
        if self.S_N <= 0:
            return (False, f"S_N = {self.S_N} is not positive")
        return (True, f"S_N = {self.S_N} positive.")

    def test_invariant_temperature_finite(self):
        """T_N must be positive and finite."""
        if self.T_N is None:
            return (False, "SKIP: T_N not available")
        if not (0 < self.T_N < float('inf')):
            return (False, f"T_N = {self.T_N} is not positive finite")
        return (True, f"T_N = {self.T_N} is positive finite.")

    def test_invariant_sigma_positive(self):
        """Stefan-Boltzmann constant must be positive (given)."""
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        if self.sigma <= 0:
            return (False, f"sigma = {self.sigma} non-positive")
        return (True, "sigma positive.")

    # ---------- Modeling / trace checks ----------
    def test_modeling_distance_ratio(self):
        """Check that the distance ratio used for S_N matches (d_E/d_N)."""
        if self.d_E is None or self.d_N is None:
            return (False, "SKIP: d_E or d_N missing")
        try:
            actual_ratio = self.d_E / self.d_N
        except Exception as e:
            return (False, f"Could not compute ratio: {e}")
        # The expected ratio is 1/30 from Earth to Neptune
        expected_ratio = 1.0 / 30.0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.05):
            return (False, f"Distance ratio {actual_ratio} != expected ~1/30")
        return (True, f"Distance ratio {actual_ratio} consistent.")

    def test_modeling_flux_ratio(self):
        """Check that S_E / S_N ≈ (d_N/d_E)^2 (inverse-square)."""
        if any(v is None for v in (self.S_E, self.S_N, self.d_E, self.d_N)):
            return (False, "SKIP: Missing data for flux ratio")
        try:
            ratio_flux = self.S_E / self.S_N
            ratio_dist = (self.d_N / self.d_E) ** 2
        except Exception as e:
            return (False, f"Computation error: {e}")
        if not math.isclose(ratio_flux, ratio_dist, rel_tol=0.05):
            return (False, f"Flux ratio {ratio_flux} != distance ratio squared {ratio_dist}")
        return (True, f"Flux ratio matches inverse-square law.")

    def test_modeling_epsilon_given(self):
        """
        Check that the solution does not use a non‑zero albedo or emissivity
        that would violate the blackbody assumption. (Assumption check.)
        We cannot directly inspect the solution's internal logic, but we can
        verify that T_N is consistent with A=0 as expected.
        """
        if self.T_N is None or self.S_N is None or self.sigma is None:
            return (False, "SKIP: Missing data for assumption check")
        # If the solution had used a different albedo, T_N would change.
        # With A=0, T_N^4 = S_N/(4σ)
        expected_T = (self.S_N / (4 * self.sigma)) ** 0.25
        if not math.isclose(self.T_N, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_N {self.T_N} does not match blackbody (A=0) prediction {expected_T}")
        return (True, "T_N consistent with blackbody assumption (A=0).")

    # ---------- Monotonicity tests (directional) ----------
    def test_monotonicity_distance_increase(self):
        """
        If Neptune were farther from the Sun, S_N would decrease and T_N would decrease.
        We check this by computing what the flux and temperature would be for a
        perturbed distance (10% larger) and verify the trend against the original values.
        """
        if any(v is None for v in (self.S_E, self.d_E, self.d_N, self.sigma)):
            return (False, "SKIP: Missing givens for monotonicity test")
        if self.S_N is None or self.T_N is None:
            return (False, "SKIP: Missing results for comparison")

        # Baseline values
        S_N_baseline = self.S_N
        T_N_baseline = self.T_N

        # Perturb: increase d_N by 10%
        d_N_perturbed = self.d_N * 1.10
        S_N_perturbed = self.S_E * (self.d_E / d_N_perturbed) ** 2
        # Under blackbody, T_N ∝ S_N^{1/4}
        T_N_perturbed = (S_N_perturbed / (4 * self.sigma)) ** 0.25

        # Trends: perturbed values should be less than baseline
        if not (S_N_perturbed < S_N_baseline):
            return (False, f"Perturbed flux {S_N_perturbed} not less than baseline {S_N_baseline}")
        if not (T_N_perturbed < T_N_baseline):
            return (False, f"Perturbed temp {T_N_perturbed} not less than baseline {T_N_baseline}")

        return (True, "Monotonicity: increasing distance decreases both flux and temperature.")

    # ---------- Weak answer sanity (non‑dominant) ----------
    def test_answer_sanity_temperature_range(self):
        """
        Very weak check: Neptune's blackbody temperature must be between
        30 K and 100 K (reasonable for outer solar system).
        """
        if self.T_N is None:
            return (False, "SKIP: T_N not available")
        if not (30 <= self.T_N <= 100):
            return (False, f"T_N = {self.T_N} outside plausible range (30-100 K)")
        return (True, "T_N within plausible range for Neptune blackbody.")

    # ---------- Main test runner ----------
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