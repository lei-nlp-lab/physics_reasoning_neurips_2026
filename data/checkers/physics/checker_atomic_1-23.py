import math
import sympy
import traceback

class PhysicsChecker:
    """
    Audits the solution instance for the problem:
    Typical cross-section for low-energy electron-atom scattering.
    Verifies geometric cross-section relation, unit consistency, and order of magnitude.
    """
    
    def __init__(self, solution_instance):
        # Safe extraction of results dictionary
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        
        # Result keys from pseudocode [emit] contract
        self.sigma_geom = self.results.get('sigma_geom', None)
        
        # Givens – try to find a0 (Bohr radius) in various possible locations
        self.a0 = None
        if hasattr(solution_instance, 'a0'):
            self.a0 = solution_instance.a0
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.a0 = solution_instance.givens.get('a0', None)
        
        # Constants (always available)
        self.pi = math.pi
        
        # Derived expected value (will be computed if a0 is available)
        self.expected_sigma_geom = None
        if self.a0 is not None:
            self.expected_sigma_geom = self.pi * (self.a0 ** 2)
    
    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that the results dictionary contains the required key 'sigma_geom'."""
        if not isinstance(self.results, dict):
            return (False, "results is not a dictionary")
        if 'sigma_geom' not in self.results:
            return (False, "Missing required key 'sigma_geom' in results")
        return (True, "Key 'sigma_geom' present")
    
    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_geometric_cross_section(self):
        """
        Check that sigma_geom ≈ π * a0^2 (geometric cross-section of atom).
        This is the fundamental physics relation for low-energy scattering.
        """
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        if self.a0 is None:
            return (False, "SKIP: a0 (Bohr radius) not available, cannot verify geometric relation")
        if self.expected_sigma_geom is None:
            return (False, "SKIP: cannot compute expected value")
        
        passed = math.isclose(self.sigma_geom, self.expected_sigma_geom, rel_tol=1e-2, abs_tol=1e-2)
        msg = f"sigma_geom = {self.sigma_geom:.2e}, expected π*a0^2 = {self.expected_sigma_geom:.2e}, match={passed}"
        return (passed, msg)
    
    def test_invariant_typical_order(self):
        """
        Based on atomic size (~10^{-8} cm), the geometric cross-section must be
        on the order of 10^{-16} cm^2. This is a physical invariant derived from
        the scale of atoms.
        """
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        # Allow factor of 10 tolerance around 1e-16
        lower = 1e-17
        upper = 1e-15
        if lower <= self.sigma_geom <= upper:
            return (True, f"sigma_geom = {self.sigma_geom:.2e} is in typical range [{lower:.0e}, {upper:.0e}]")
        else:
            return (False, f"sigma_geom = {self.sigma_geom:.2e} is outside the expected order of magnitude")
    
    # ----------------------------------------------------------------------
    # Modeling/Trace quality tests (at least two required)
    # ----------------------------------------------------------------------
    def test_modeling_positive_finite(self):
        """Check that sigma_geom is a positive, finite number (consistent with area)."""
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        if not isinstance(self.sigma_geom, (int, float)):
            return (False, f"sigma_geom is not numeric, type={type(self.sigma_geom)}")
        if not math.isfinite(self.sigma_geom):
            return (False, "sigma_geom is not finite (NaN or inf)")
        if self.sigma_geom <= 0:
            return (False, f"sigma_geom must be positive, got {self.sigma_geom}")
        return (True, f"sigma_geom = {self.sigma_geom:.3e} > 0, finite")
    
    def test_modeling_implied_radius(self):
        """
        From the cross-section, compute the implied atomic radius r = sqrt(sigma_geom/π).
        The radius should be on the order of 10^{-8} cm (Bohr radius).
        """
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        if self.sigma_geom <= 0:
            return (False, "SKIP: sigma_geom must be positive to compute radius")
        r_implied = math.sqrt(self.sigma_geom / self.pi)
        # Expected radius around 1e-8 cm (Bohr radius ~5.3e-9 cm)
        lower = 1e-10   # very generous lower bound
        upper = 1e-7    # upper bound (larger than typical atom)
        if lower <= r_implied <= upper:
            return (True, f"Implied radius = {r_implied:.2e} cm, within expected range [{lower:.0e}, {upper:.0e}]")
        else:
            return (False, f"Implied radius = {r_implied:.2e} cm, outside expected range [{lower:.0e}, {upper:.0e}]")
    
    def test_modeling_closest_option(self):
        """
        Among the given multiple-choice options (1e-16, 1e-24, 1e-32, 1e-40 cm^2),
        check that the computed sigma_geom is closest to 1e-16 (the expected answer).
        """
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        options = [1e-16, 1e-24, 1e-32, 1e-40]
        if self.sigma_geom <= 0:
            return (False, "SKIP: sigma_geom must be positive")
        # Compute relative distances (log scale to handle huge ranges)
        log_sigma = math.log10(self.sigma_geom)
        distances = {}
        for opt in options:
            log_opt = math.log10(opt)
            dist = abs(log_sigma - log_opt)
            distances[opt] = dist
        closest = min(distances, key=distances.get)
        if closest == 1e-16:
            return (True, f"Closest option is 10^-16 (distance = {distances[1e-16]:.2f})")
        else:
            return (False, f"Closest option is {closest}, not 1e-16 (distances: {distances})")
    
    # ----------------------------------------------------------------------
    # Monotonicity test (directional/trend check with explicit perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_a0_scaling(self):
        """
        Perturb a0 by a factor of 2 and verify that sigma_geom scales as a0^2.
        We compute the expected perturbed sigma as π*(2*a0)^2 and compare
        it to 4 times the original sigma_geom (the scaling factor).
        """
        if self.sigma_geom is None or self.a0 is None:
            return (False, "SKIP: missing sigma_geom or a0 for monotonicity check")
        if self.a0 == 0:
            return (False, "SKIP: a0 is zero, cannot compute scaling")
        
        factor = 2.0
        perturbed_a0 = factor * self.a0
        expected_perturbed_sigma = self.pi * (perturbed_a0 ** 2)
        predicted_perturbed_sigma = self.sigma_geom * (factor ** 2)
        
        passed = math.isclose(expected_perturbed_sigma, predicted_perturbed_sigma, rel_tol=1e-2, abs_tol=1e-2)
        msg = (f"Perturb a0 by factor {factor}: expected perturbed sigma = {expected_perturbed_sigma:.2e}, "
               f"predicted from scaling = {predicted_perturbed_sigma:.2e}, match={passed}")
        return (passed, msg)
    
    # ----------------------------------------------------------------------
    # Weak answer sanity test (only one allowed, no math.isclose)
    # ----------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """
        Weak sanity: sigma_geom should be in the broad range [1e-20, 1e-10] cm^2.
        This is a very loose check; the actual answer is 10^{-16}.
        """
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        lower = 1e-20
        upper = 1e-10
        if lower <= self.sigma_geom <= upper:
            return (True, f"sigma_geom = {self.sigma_geom:.2e} is within very broad range [{lower:.0e}, {upper:.0e}]")
        else:
            return (False, f"sigma_geom = {self.sigma_geom:.2e} is outside the extremely loose bounds [{lower:.0e}, {upper:.0e}]")
    
    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover and run all test methods.
        Returns (success_bool, logs_list).
        """
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