import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Physics checker for the reflection phase change problem:
    plane polarized EM wave at normal incidence from dielectric (n) to conductor (n(1+ip)).
    """
    
    def __init__(self, solution_instance: Any) -> None:
        """
        Extract inputs and results safely. Default to None for missing values.
        """
        # Givens from the problem
        self.n = getattr(solution_instance, 'n', None)
        self.p = getattr(solution_instance, 'p', None)
        
        # Results (emitted by pseudocode)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.phi = self.results.get('phi', None)
    
    # ----------------------------------------------------------------------
    # Interface tests – check required keys and types
    # ----------------------------------------------------------------------
    def test_interface_phi_exists(self) -> Tuple[bool, str]:
        """Check that phi is present in results and is a number."""
        if self.phi is None:
            return (False, "Missing key 'phi' in results")
        if not isinstance(self.phi, (int, float)):
            return (False, f"'phi' is not a number, got {type(self.phi).__name__}")
        if math.isnan(self.phi) or math.isinf(self.phi):
            return (False, f"'phi' is NaN or infinite: {self.phi}")
        return (True, f"'phi' = {self.phi} (valid number)")
    
    def test_interface_givens_n_p(self) -> Tuple[bool, str]:
        """Check that n and p are provided."""
        if self.n is None:
            return (False, "Missing given 'n' (refractive index of dielectric)")
        if self.p is None:
            return (False, "Missing given 'p' (loss parameter of conductor, n2 = n(1+ip))")
        if not isinstance(self.n, (int, float)) or not isinstance(self.p, (int, float)):
            return (False, "n and p must be numeric")
        if self.p <= 0:
            return (False, "p must be positive for a conductor (absorption)")
        return (True, f"n = {self.n}, p = {self.p}")
    
    # ----------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_phase_formula(self) -> Tuple[bool, str]:
        """
        Check that phi satisfies the derived Fresnel formula:
            phi = arctan(2/p)
        with p>0.  The formula is derived inside the checker from givens.
        """
        if self.phi is None or self.p is None:
            return (False, "SKIP: phi or p missing")
        if self.p <= 0:
            return (False, f"Non-positive p = {self.p}. Cannot compute arctan(2/p).")
        
        # Expected value derived from physics mapping and givens
        expected_phi = math.atan(2.0 / self.p)   # principal value, in [0, pi/2]
        if not math.isclose(self.phi, expected_phi, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"phi = {self.phi} does not match expected {expected_phi} "
                    f"(difference = {abs(self.phi - expected_phi):.4f})")
        return (True, f"phi = {self.phi} is consistent with arctan(2/{self.p})")
    
    def test_invariant_phase_range(self) -> Tuple[bool, str]:
        """
        Physical constraint: for p>0, phi should be in (0, π/2).
        (As arctan(2/p) goes to π/2 when p→0⁺ and to 0 when p→∞.)
        """
        if self.phi is None:
            return (False, "SKIP: phi missing")
        if not (0 < self.phi < math.pi/2):
            return (False, f"phi = {self.phi} is outside (0, π/2)")
        return (True, f"phi = {self.phi} lies in (0, π/2) as expected")
    
    # ----------------------------------------------------------------------
    # Modeling / trace tests – consistency with explanation steps
    # ----------------------------------------------------------------------
    def test_modeling_positive_phi(self) -> Tuple[bool, str]:
        """Phase change should be positive (reflection with a phase advance)."""
        if self.phi is None:
            return (False, "SKIP: phi missing")
        if self.phi <= 0:
            return (False, f"phi = {self.phi} is not positive")
        return (True, f"phi = {self.phi} > 0")
    
    def test_modeling_finite_phi(self) -> Tuple[bool, str]:
        """phi must be finite (no infinities or NaN)."""
        if self.phi is None:
            return (False, "SKIP: phi missing")
        if not math.isfinite(self.phi):
            return (False, f"phi = {self.phi} is not finite")
        return (True, "phi is finite")
    
    # ----------------------------------------------------------------------
    # Monotonicity tests – trend check via input perturbation
    # ----------------------------------------------------------------------
    def test_monotonicity_decreasing_with_p(self) -> Tuple[bool, str]:
        """
        For a conductor, as p increases (more conductive), 
        the phase shift phi should decrease monotonically.
        Derive the expected behaviour from the formula and check that
        a small percentage increase in p leads to a smaller phi.
        """
        if self.phi is None or self.p is None:
            return (False, "SKIP: phi or p missing")
        if self.p <= 0:
            return (False, f"Non-positive p = {self.p}.")
        
        # Baseline p and phi
        p0 = self.p
        phi0 = self.phi
        
        # Perturb p upward by 10%
        p1 = p0 * 1.1
        # Expected phi after increase in p (using the formula)
        phi1 = math.atan(2.0 / p1)
        
        # Monotonicity: larger p -> smaller phi
        if not (phi1 < phi0):
            return (False,
                    f"With p increased from {p0:.4f} to {p1:.4f}, "
                    f"phi should decrease from {phi0:.4f} but expected phi_perturbed = {phi1:.4f}")
        return (True, f"Monotonicity holds: p increased → phi decreases "
                      f"({phi0:.4f} → {phi1:.4f})")
    
    # ----------------------------------------------------------------------
    # Optional weak answer sanity – only sign, finiteness, broad range
    # ----------------------------------------------------------------------
    def test_answer_sanity_phi_sign(self) -> Tuple[bool, str]:
        """
        Weak sanity: the phase change should be a positive acute angle.
        No derivation, no comparison to expected numeric value.
        """
        if self.phi is None:
            return (False, "SKIP: phi missing")
        # Must be a finite number between 0 and π (physical range)
        if not math.isfinite(self.phi):
            return (False, "phi is not finite")
        if self.phi <= 0 or self.phi > math.pi:
            return (False, f"phi = {self.phi} is not in (0, π]")
        # Allow up to π (though formula gives < π/2)
        return (True, f"phi = {self.phi} passes weak sanity (finite, in (0,π])")
    
    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them,
        collect messages, and return (all_pass, logs).
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)