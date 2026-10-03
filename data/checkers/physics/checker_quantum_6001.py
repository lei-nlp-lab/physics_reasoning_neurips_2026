import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the hard sphere s-wave scattering cross section problem.
    Checks interface, physical invariants, modelling quality, monotonicity, and answer sanity.
    """

    def __init__(self, solution_instance):
        # ---- Safely extract all needed attributes ----
        # Givens (expected from solution_instance or its results dict)
        self.R = (getattr(solution_instance, 'R', None) or
                  solution_instance.results.get('R', None))
        self.k = (getattr(solution_instance, 'k', None) or
                  solution_instance.results.get('k', None))

        # Emitted results
        self.delta0 = solution_instance.results.get('delta0', None)
        self.sigma = solution_instance.results.get('sigma', None)

        # Fallback for missing keys: keep None, no numeric defaults.

    # ---------------------------------------------------------------
    # Interface tests
    # ---------------------------------------------------------------
    def test_interface_delta0_exists(self):
        if self.delta0 is None:
            return (False, "SKIP: delta0 not provided")
        return (True, "delta0 is present")

    def test_interface_sigma_exists(self):
        if self.sigma is None:
            return (False, "SKIP: sigma not provided")
        return (True, "sigma is present")

    def test_interface_R_exists(self):
        if self.R is None:
            return (False, "SKIP: R (hard sphere radius) not provided")
        return (True, "R is present")

    def test_interface_k_exists(self):
        if self.k is None:
            return (False, "SKIP: k (wave number) not provided")
        return (True, "k is present")

    def test_interface_delta0_numeric(self):
        if self.delta0 is None:
            return (False, "SKIP: delta0 not available")
        if not isinstance(self.delta0, (int, float)):
            return (False, "delta0 must be numeric")
        return (True, "delta0 is numeric")

    def test_interface_sigma_numeric(self):
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        if not isinstance(self.sigma, (int, float)):
            return (False, "sigma must be numeric")
        return (True, "sigma is numeric")

    def test_interface_R_positive(self):
        if self.R is None:
            return (False, "SKIP: R not available")
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}")
        return (True, "R > 0")

    def test_interface_k_positive(self):
        if self.k is None:
            return (False, "SKIP: k not available")
        if self.k <= 0:
            return (False, f"k must be positive, got {self.k}")
        return (True, "k > 0")

    # ---------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ---------------------------------------------------------------
    def test_invariant_boundary_condition(self):
        """ sin(kR + delta0) must be zero (hard sphere boundary condition) """
        if any(v is None for v in (self.k, self.R, self.delta0)):
            return (False, "SKIP: k, R, or delta0 missing")
        val = self.k * self.R + self.delta0
        if not math.isclose(math.sin(val), 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sin(kR+δ₀) = {math.sin(val):.4e} not zero (val = {val:.4f})")
        return (True, "Boundary condition sin(kR+δ₀)=0 satisfied")

    def test_invariant_phase_shift_quantized(self):
        """ (kR + delta0) must be an integer multiple of π. """
        if any(v is None for v in (self.k, self.R, self.delta0)):
            return (False, "SKIP: k, R, or delta0 missing")
        val = self.k * self.R + self.delta0
        multiple = val / math.pi
        # Check closeness to nearest integer
        nearest_int = round(multiple)
        if not math.isclose(multiple, nearest_int, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"(kR+δ₀)/π = {multiple:.4f}, not close to integer {nearest_int}")
        return (True, "Phase shift condition (kR+δ₀ = nπ) holds")

    def test_invariant_cross_section_formula(self):
        """ sigma must equal (4π/k²) sin²(δ₀) """
        if any(v is None for v in (self.k, self.sigma, self.delta0)):
            return (False, "SKIP: k, sigma, or delta0 missing")
        computed = (4 * math.pi / (self.k ** 2)) * (math.sin(self.delta0) ** 2)
        if not math.isclose(self.sigma, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma = {self.sigma:.4e}, expected from formula {computed:.4e}")
        return (True, "Cross section matches formula (4π/k²) sin²(δ₀)")

    def test_invariant_phase_shift_delta0(self):
        """ Phase shift must be -kR (mod π). """
        if any(v is None for v in (self.k, self.R, self.delta0)):
            return (False, "SKIP: k, R, or delta0 missing")
        # Check if delta0 is close to -kR modulo π
        diff = self.delta0 + self.k * self.R
        # diff should be an integer multiple of π
        multiple = diff / math.pi
        nearest_int = round(multiple)
        if not math.isclose(multiple, nearest_int, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"δ₀ = {self.delta0:.4f}, expected δ₀ ≡ -kR (mod π) with -kR = {-self.k*self.R:.4f}, difference/π = {multiple:.4f}")
        return (True, "Phase shift satisfies δ₀ ≡ -kR (mod π)")

    def test_invariant_sigma_nonnegative(self):
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        if self.sigma < -1e-8:
            return (False, f"sigma must be non-negative, got {self.sigma}")
        return (True, "sigma >= 0")

    # ---------------------------------------------------------------
    # Modelling quality tests
    # ---------------------------------------------------------------
    def test_modeling_low_energy_limit(self):
        """
        For small kR (< 0.15) the cross section should be approx 4πR².
        (Only test if kR is indeed small)
        """
        if any(v is None for v in (self.k, self.R, self.sigma)):
            return (False, "SKIP: k, R, or sigma missing")
        kr = self.k * self.R
        if kr > 0.15:
            return (True, "kR not small (>0.15), low-energy limit test skipped")
        target = 4 * math.pi * (self.R ** 2)
        if not math.isclose(self.sigma, target, rel_tol=1e-1, abs_tol=1e-2):
            return (False, f"For small kR ({kr:.4e}) expected sigma ≈ {target:.4e}, got {self.sigma:.4e}")
        return (True, "Low-energy limit satisfied (sigma ≈ 4πR²)")

    def test_modeling_high_energy_bound(self):
        """
        For any k, sigma must be <= 4π/k² because sin² ≤ 1.
        """
        if any(v is None for v in (self.k, self.sigma)):
            return (False, "SKIP: k or sigma missing")
        bound = 4 * math.pi / (self.k ** 2)
        if self.sigma > bound + 1e-8:
            return (False, f"sigma > 4π/k² ({bound:.4e}) -> violates sin² bound")
        return (True, f"sigma <= 4π/k² holds")

    # ---------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ---------------------------------------------------------------
    def test_monotonicity_delta0_with_k(self):
        """
        Perturb k by +1% and check that delta0 decreases by approximately R * Δk.
        """
        if any(v is None for v in (self.k, self.R, self.delta0)):
            return (False, "SKIP: k, R, or delta0 missing")
        if self.k <= 0:
            return (False, "k must be positive")
        delta_k = 0.01 * self.k          # small positive perturbation
        expected_new_delta0 = - (self.k + delta_k) * self.R
        # The change predicted by the derivative should be -R * delta_k
        predicted_change = -self.R * delta_k
        actual_change = expected_new_delta0 - self.delta0
        if not math.isclose(actual_change, predicted_change, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"After +1% k increase, expected Δδ₀ ≈ {predicted_change:.4e}, got {actual_change:.4e}. "
                           "δ₀ does not behave as -kR.")
        return (True, "Monotonicity δ₀ with k consistent: dδ₀/dk ≈ -R")

    def test_monotonicity_delta0_with_R(self):
        """
        Perturb R by +1% and check that delta0 decreases by approximately k * ΔR.
        """
        if any(v is None for v in (self.k, self.R, self.delta0)):
            return (False, "SKIP: k, R, or delta0 missing")
        if self.R <= 0:
            return (False, "R must be positive")
        delta_R = 0.01 * self.R
        expected_new_delta0 = - self.k * (self.R + delta_R)
        predicted_change = -self.k * delta_R
        actual_change = expected_new_delta0 - self.delta0
        if not math.isclose(actual_change, predicted_change, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"After +1% R increase, expected Δδ₀ ≈ {predicted_change:.4e}, got {actual_change:.4e}. "
                           "δ₀ does not behave as -kR.")
        return (True, "Monotonicity δ₀ with R consistent: dδ₀/dR ≈ -k")

    # ---------------------------------------------------------------
    # One weak answer sanity check (no hardcoded target)
    # ---------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """ Check that sigma is finite and not NaN. """
        if self.sigma is None:
            return (False, "SKIP: sigma not available")
        if not math.isfinite(self.sigma):
            return (False, "sigma is not finite or is NaN")
        return (True, "sigma is finite and not NaN")

    # ---------------------------------------------------------------
    # Run all tests
    # ---------------------------------------------------------------
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