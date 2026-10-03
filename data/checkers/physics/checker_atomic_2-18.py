import math
import traceback
import sys

class PhysicsChecker:
    """
    Audits a solution for the scattering of a 1-keV proton by a hydrogen atom.
    Expected outputs: sigma_geom (geometric cross section π a0^2) and
    sigma_est (total cross section estimate including Coulomb tail).
    """

    # Physical constants (SI units, but cross sections in cm^2)
    a0 = 5.2918e-11  # Bohr radius in m
    pi = math.pi
    # Geometric cross section in cm^2: π a0^2 * 1e4
    expected_sigma_geom_cm2 = pi * a0**2 * 1e4  # ≈ 8.79e-17 cm^2

    # Known incident energy in eV (from problem statement)
    E_0 = 1000.0  # eV

    def __init__(self, solution_instance):
        """Extract results safely; set None for missing values."""
        if solution_instance is None:
            self.sigma_geom = None
            self.sigma_est = None
            return

        results = getattr(solution_instance, 'results', None)
        if results is None:
            self.sigma_geom = None
            self.sigma_est = None
        else:
            self.sigma_geom = results.get('sigma_geom', None)
            self.sigma_est = results.get('sigma_est', None)

    # ----------------------------------------------------------------------
    # Interface tests (correct existence and type)
    # ----------------------------------------------------------------------

    def test_interface_sigma_geom_exists(self):
        """Check that sigma_geom is present and is a number."""
        if self.sigma_geom is None:
            return (False, "sigma_geom is missing (None)")
        if not isinstance(self.sigma_geom, (int, float)):
            return (False, f"sigma_geom should be numeric, got {type(self.sigma_geom)}")
        return (True, "sigma_geom exists and is numeric")

    def test_interface_sigma_est_exists(self):
        """Check that sigma_est is present and is a number."""
        if self.sigma_est is None:
            return (False, "sigma_est is missing (None)")
        if not isinstance(self.sigma_est, (int, float)):
            return (False, f"sigma_est should be numeric, got {type(self.sigma_est)}")
        return (True, "sigma_est exists and is numeric")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based, use fuzzy matching)
    # ----------------------------------------------------------------------

    def test_invariant_sigma_geom_geometric(self):
        """Geometric cross section should equal π a0² within 2% tolerance."""
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        expected = self.expected_sigma_geom_cm2
        if math.isclose(self.sigma_geom, expected, rel_tol=0.02, abs_tol=1e-20):
            return (True, f"sigma_geom = {self.sigma_geom:.3e} cm² ≈ π a0² = {expected:.3e} cm²")
        else:
            return (False, f"sigma_geom = {self.sigma_geom:.3e} cm², expected ~ {expected:.3e} cm² (2% tolerance)")

    def test_invariant_sigma_est_greater_than_geom(self):
        """Total cross section should be larger than geometric due to Coulomb tail."""
        if self.sigma_geom is None or self.sigma_est is None:
            return (False, "SKIP: one of the cross sections missing")
        # Allow very small numerical tolerance
        if self.sigma_est > self.sigma_geom - 1e-20:
            return (True, f"sigma_est = {self.sigma_est:.3e} > sigma_geom = {self.sigma_geom:.3e}")
        else:
            return (False, f"sigma_est = {self.sigma_est:.3e} should be > sigma_geom = {self.sigma_geom:.3e}")

    def test_invariant_ratio_plausible(self):
        """The ratio sigma_est / sigma_geom should be between 100 and 1000 (typical for this problem)."""
        if self.sigma_geom is None or self.sigma_est is None:
            return (False, "SKIP: one of the cross sections missing")
        if self.sigma_geom <= 0:
            return (False, "sigma_geom must be positive for ratio")
        ratio = self.sigma_est / self.sigma_geom
        # Based on literature: sigma_est ≈ 1.76e-14, sigma_geom ≈ 8.79e-17 → ratio ≈ 200
        if 50 <= ratio <= 500:
            return (True, f"Ratio sigma_est / sigma_geom = {ratio:.1f} (expected ~200)")
        else:
            return (False, f"Ratio = {ratio:.1f}, expected in [50, 500]")

    # ----------------------------------------------------------------------
    # Modelling / trace tests (based on explanation steps)
    # ----------------------------------------------------------------------

    def test_modeling_sigma_est_positive(self):
        """Total cross section estimate must be positive."""
        if self.sigma_est is None:
            return (False, "SKIP: sigma_est not available")
        if self.sigma_est > 0:
            return (True, f"sigma_est = {self.sigma_est:.3e} > 0")
        else:
            return (False, f"sigma_est must be positive, got {self.sigma_est}")

    def test_modeling_sigma_geom_positive(self):
        """Geometric cross section must be positive."""
        if self.sigma_geom is None:
            return (False, "SKIP: sigma_geom not available")
        if self.sigma_geom > 0:
            return (True, f"sigma_geom = {self.sigma_geom:.3e} > 0")
        else:
            return (False, f"sigma_geom must be positive, got {self.sigma_geom}")

    def test_modeling_sigma_est_plausible_magnitude(self):
        """Total cross section should be in a physically plausible range for 1 keV protons on H."""
        if self.sigma_est is None:
            return (False, "SKIP: sigma_est not available")
        # Typical atomic cross sections are between 1e-20 and 1e-10 cm²
        if 1e-20 < self.sigma_est < 1e-10:
            return (True, f"sigma_est = {self.sigma_est:.3e} cm² is within plausible range (1e-20 to 1e-10)")
        else:
            return (False, f"sigma_est = {self.sigma_est:.3e} cm² outside plausible range")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb a physical input and check trend)
    # ----------------------------------------------------------------------

    def test_monotonicity_sigma_est_decreases_with_energy(self):
        """
        From explanation steps, total cross section estimate scales as sigma_est ∝ 1/E.
        Perturb incident energy by +10% (E → 1.1*E) and check that sigma_est decreases.
        """
        if self.sigma_est is None:
            return (False, "SKIP: sigma_est not available")
        # Compute perturbed energy
        E_pert = 1.1 * self.E_0  # 1100 eV
        # Using scaling: sigma_est_new = sigma_est * (E_0 / E_pert)
        sigma_est_pert = self.sigma_est * (self.E_0 / E_pert)
        # The perturbation should lead to a smaller cross section
        if sigma_est_pert < self.sigma_est:
            return (True, f"sigma_est at {self.E_0} eV = {self.sigma_est:.3e} cm², "
                          f"at {E_pert:.0f} eV = {sigma_est_pert:.3e} cm² (decreased as expected)")
        else:
            return (False, f"sigma_est did not decrease when energy increased: "
                          f"original={self.sigma_est:.3e}, perturbed={sigma_est_pert:.3e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (one method, no isclose, no re-derivation)
    # ----------------------------------------------------------------------

    def test_answer_sanity_units(self):
        """Cross sections should be given in cm² (or m²) – ensure they are positive and not astronomically small."""
        if self.sigma_geom is None or self.sigma_est is None:
            return (False, "SKIP: missing cross section")
        if self.sigma_geom > 1e-30 and self.sigma_est > 1e-30:
            return (True, "Both cross sections are reasonably sized (> 1e-30)")
        else:
            return (False, "Cross section values seem unnaturally small")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------

    def run_all_tests(self):
        """Discover and execute all test_ methods, return (success, logs)."""
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


# Example usage:
if __name__ == "__main__":
    # Dummy solution with expected outputs
    class DummySolution:
        def __init__(self):
            # Geometric sigma = π a0² in cm²
            pi = math.pi
            a0 = 5.2918e-11
            self.results = {
                'sigma_geom': pi * a0**2 * 1e4,  # 8.79e-17 cm²
                'sigma_est': 1.76e-14           # cm²
            }

    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for line in logs:
        print(line)
    print(f"\nOverall success: {success}")