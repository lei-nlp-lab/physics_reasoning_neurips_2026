import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract given constants with None defaults
        self.G = getattr(solution_instance, 'G', None)
        self.M = getattr(solution_instance, 'M', None)
        self.f = getattr(solution_instance, 'f', None)
        self.rho_nuc = getattr(solution_instance, 'rho_nuc', None)

        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Safely extract expected result keys
        self.rho_min = self.results.get('rho_min', None)
        self.R_max = self.results.get('R_max', None)
        self.R = self.results.get('R', None)

        # For convenience, store explanation steps if present
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # --- Interface tests ---
    def test_interface_required_keys_exist(self):
        """Check that all expected result keys are present."""
        missing = []
        for key in ['rho_min', 'R_max', 'R']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_given_constants_exist(self):
        """Check that required given constants are provided."""
        missing = []
        for attr in ['G', 'M', 'f', 'rho_nuc']:
            if getattr(self, attr) is None:
                missing.append(attr)
        if missing:
            return (False, f"Missing given constants: {missing}")
        return (True, "All required constants present")

    def test_interface_results_numeric_and_positive(self):
        """Check that results are numeric and positive."""
        for name, val in [('rho_min', self.rho_min),
                          ('R_max', self.R_max),
                          ('R', self.R)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "All results positive numeric")

    # --- Invariant tests (theorem-based) ---
    def test_invariant_min_density_formula(self):
        """Check rho_min derived from formula: rho_min = 3*(2πf)^2/(4πG)."""
        if any(x is None for x in [self.f, self.G, self.rho_min]):
            return (False, "SKIP: missing f/G/rho_min")
        omega = 2 * math.pi * self.f
        rho_expected = 3 * omega**2 / (4 * math.pi * self.G)
        if math.isclose(self.rho_min, rho_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"rho_min {self.rho_min:.3e} matches expected {rho_expected:.3e}")
        else:
            return (False, f"rho_min {self.rho_min:.3e} deviates from expected {rho_expected:.3e}")

    def test_invariant_max_radius_from_density(self):
        """Check R_max derived from rho_min: R_max = (3M/(4π rho_min))^(1/3)."""
        if any(x is None for x in [self.M, self.rho_min, self.R_max]):
            return (False, "SKIP: missing M/rho_min/R_max")
        R_expected = (3 * self.M / (4 * math.pi * self.rho_min)) ** (1/3)
        if math.isclose(self.R_max, R_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R_max {self.R_max:.3e} matches expected {R_expected:.3e}")
        else:
            return (False, f"R_max {self.R_max:.3e} deviates from expected {R_expected:.3e}")

    def test_invariant_nuclear_radius_formula(self):
        """Check R from nuclear density: R = (3M/(4π rho_nuc))^(1/3)."""
        if any(x is None for x in [self.M, self.rho_nuc, self.R]):
            return (False, "SKIP: missing M/rho_nuc/R")
        R_expected = (3 * self.M / (4 * math.pi * self.rho_nuc)) ** (1/3)
        if math.isclose(self.R, R_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R {self.R:.3e} matches expected {R_expected:.3e}")
        else:
            return (False, f"R {self.R:.3e} deviates from expected {R_expected:.3e}")

    def test_invariant_stability_inequality(self):
        """Check that centrifugal acceleration ≤ gravitational acceleration at equator."""
        if any(x is None for x in [self.f, self.G, self.rho_min]):
            return (False, "SKIP: missing f/G/rho_min")
        omega = 2 * math.pi * self.f
        left = (4 * math.pi / 3) * self.rho_min
        right = omega**2 / self.G
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Stability condition holds: (4π/3)ρ_min = {left:.3e} ≈ ω²/G = {right:.3e}")
        else:
            return (False, f"Stability condition violated: (4π/3)ρ_min = {left:.3e} vs ω²/G = {right:.3e}")

    # --- Modeling tests (based on explanation steps and pseudocode) ---
    def test_modeling_omega_calculation(self):
        """Check that omega = 2πf is computed consistently."""
        if self.f is None:
            return (False, "SKIP: missing f")
        omega_explicit = 2 * math.pi * self.f
        if self.rho_min is not None:
            # Infer omega from rho_min and G
            omega_inferred = math.sqrt( (4 * math.pi * self.G * self.rho_min) / 3 )
            if math.isclose(omega_inferred, omega_explicit, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Omega consistent with rho_min")
            else:
                return (False, f"Inferred omega {omega_inferred:.3f} differs from 2πf {omega_explicit:.3f}")
        return (True, "No rho_min to cross-check omega")

    def test_modeling_radius_scaling(self):
        """Check that R_max and R are calculated with cube root scaling."""
        if any(x is None for x in [self.rho_min, self.rho_nuc, self.R_max, self.R]):
            return (False, "SKIP: missing densities or radii")
        ratio_expected = (self.rho_nuc / self.rho_min) ** (1/3)
        ratio_actual = self.R_max / self.R
        if math.isclose(ratio_actual, ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Radius scaling correct: R_max/R = {ratio_actual:.3f} ≈ (ρ_nuc/ρ_min)^{1/3} = {ratio_expected:.3f}")
        else:
            return (False, f"Radius scaling off: R_max/R = {ratio_actual:.3f} vs expected {ratio_expected:.3f}")

    def test_modeling_consistency_with_explanation(self):
        """Check if explanation steps contain the key formulas (optional check)."""
        if not self.explanation_steps:
            return (True, "No explanation steps to check")
        key_phrases = ['ρ_min', 'R_max', 'R', 'density', 'radius']
        found = [phrase for phrase in key_phrases if any(phrase in str(step) for step in self.explanation_steps)]
        if len(found) >= 2:
            return (True, f"Explanation steps mention key concepts: {found}")
        else:
            return (False, f"Explanation steps missing key terms: only found {found}")

    # --- Monotonicity tests (each must perturb an input and compare perturbed output to baseline) ---
    def test_monotonicity_density_with_frequency(self):
        """Perturb f: increase it by 10% and check that rho_min increases."""
        if any(x is None for x in [self.f, self.G]):
            return (False, "SKIP: missing f or G")
        # Baseline rho_min using formula
        omega_base = 2 * math.pi * self.f
        rho_base = 3 * omega_base**2 / (4 * math.pi * self.G)
        # Perturb f by +10%
        f_pert = self.f * 1.1
        omega_pert = 2 * math.pi * f_pert
        rho_pert = 3 * omega_pert**2 / (4 * math.pi * self.G)
        if rho_pert > rho_base:
            return (True, f"rho_min increased from {rho_base:.3e} to {rho_pert:.3e} when f increased by 10%")
        else:
            return (False, f"rho_min did not increase: base {rho_base:.3e}, perturbed {rho_pert:.3e}")

    def test_monotonicity_radius_with_density(self):
        """Perturb rho_min: increase it by 10% and check that R_max decreases."""
        if any(x is None for x in [self.M, self.rho_min]):
            return (False, "SKIP: missing M or rho_min")
        # Baseline R_max using formula
        R_base = (3 * self.M / (4 * math.pi * self.rho_min)) ** (1/3)
        # Perturb rho_min by +10%
        rho_pert = self.rho_min * 1.1
        R_pert = (3 * self.M / (4 * math.pi * rho_pert)) ** (1/3)
        if R_pert < R_base:
            return (True, f"R_max decreased from {R_base:.3e} to {R_pert:.3e} when rho_min increased by 10%")
        else:
            return (False, f"R_max did not decrease: base {R_base:.3e}, perturbed {R_pert:.3e}")

    # --- Answer sanity (weak, only one method) ---
    def test_answer_sanity_final(self):
        """Weak sanity: final results should be finite, positive, and within realistic orders of magnitude."""
        if any(x is None for x in [self.rho_min, self.R_max, self.R]):
            return (False, "SKIP: missing results")
        checks = []
        # Check rho_min: should be huge (~10^14) but not negative
        if self.rho_min <= 0 or math.isnan(self.rho_min) or math.isinf(self.rho_min):
            checks.append(f"rho_min invalid: {self.rho_min}")
        else:
            if self.rho_min < 1e10 or self.rho_min > 1e16:
                checks.append(f"rho_min {self.rho_min:.3e} outside plausible range [1e10, 1e16]")
        # Check R_max: should be about 150 km = 1.5e5 m
        if self.R_max <= 0 or math.isnan(self.R_max) or math.isinf(self.R_max):
            checks.append(f"R_max invalid: {self.R_max}")
        else:
            if self.R_max < 1e3 or self.R_max > 1e7:
                checks.append(f"R_max {self.R_max:.3e} outside plausible range [1e3, 1e7]")
        # Check R: should be about 17 km = 1.7e4 m
        if self.R <= 0 or math.isnan(self.R) or math.isinf(self.R):
            checks.append(f"R invalid: {self.R}")
        else:
            if self.R < 1e3 or self.R > 1e6:
                checks.append(f"R {self.R:.3e} outside plausible range [1e3, 1e6]")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All results finite, positive, and within plausible orders")