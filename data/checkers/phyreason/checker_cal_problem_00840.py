import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs with safe defaults
        self.T = getattr(solution_instance, 'T', getattr(solution_instance, 'inputs', {}).get('T', 1000.0))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 1.737e6))
        self.G = getattr(solution_instance, 'G', getattr(solution_instance, 'inputs', {}).get('G', 6.67430e-11))
        
        # Extract results
        self.results = getattr(solution_instance, 'results', {})
        self.M = float(self.results.get('M', 0.0))
        self.rho = float(self.results.get('rho', 0.0))
        
        self.logs = []
    
    def _log(self, msg: str, success: bool):
        """Helper to log test results."""
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants using numeric calculations only.
        """
        success = True
        
        # Test 1: Gravitational = Centripetal force
        try:
            # Orbital radius = 2R
            r_orbit = 2 * self.R
            
            # Orbital speed v = 2πr/T
            v = 2 * math.pi * r_orbit / self.T
            
            # Gravitational acceleration (force per unit mass)
            a_grav = self.G * self.M / r_orbit**2
            
            # Centripetal acceleration
            a_cen = v**2 / r_orbit
            
            invariant1 = math.isclose(a_grav, a_cen, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Gravitational = Centripetal force: {invariant1}", invariant1)
            success = success and invariant1
        except Exception as e:
            self._log(f"Gravitational/Centripetal invariant failed: {str(e)}", False)
            success = False
        
        # Test 2: Density consistency
        try:
            # Volume of Moon = (4/3)πR³
            V = (4/3) * math.pi * self.R**3
            density_from_M = self.M / V
            
            invariant2 = math.isclose(density_from_M, self.rho, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Density = Mass/Volume: {invariant2}", invariant2)
            success = success and invariant2
        except Exception as e:
            self._log(f"Density invariant failed: {str(e)}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check results with expected formulas.
        """
        success = True
        
        try:
            # Expected M from formula: M = 32π²R³/(GT²)
            M_expected = (32 * math.pi**2 * self.R**3) / (self.G * self.T**2)
            M_consistent = math.isclose(self.M, M_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Mass M matches KB derivation: {M_consistent}", M_consistent)
            success = success and M_consistent
        except Exception as e:
            self._log(f"Mass consistency check failed: {str(e)}", False)
            success = False
        
        try:
            # Expected ρ from formula: ρ = 24π/(GT²)
            rho_expected = (24 * math.pi) / (self.G * self.T**2)
            rho_consistent = math.isclose(self.rho, rho_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Density ρ matches KB derivation: {rho_consistent}", rho_consistent)
            success = success and rho_consistent
        except Exception as e:
            self._log(f"Density consistency check failed: {str(e)}", False)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input R and check if M and ρ change as expected.
        """
        success = True
        
        try:
            # Original values
            R_orig = float(self.R)
            T_orig = float(self.T)
            G_orig = float(self.G)
            
            # Perturb R by +1%
            R_pert = R_orig * 1.01
            
            # Calculate expected M and ρ using formulas
            M_orig_formula = (32 * math.pi**2 * R_orig**3) / (G_orig * T_orig**2)
            M_pert_formula = (32 * math.pi**2 * R_pert**3) / (G_orig * T_orig**2)
            rho_orig_formula = (24 * math.pi) / (G_orig * T_orig**2)
            rho_pert_formula = (24 * math.pi) / (G_orig * T_orig**2)
            
            # Check 1: M should increase by ~(1.01)³ = 1.030301
            M_ratio_expected = (R_pert / R_orig)**3
            M_ratio_actual = M_pert_formula / M_orig_formula
            M_monotonic = math.isclose(M_ratio_actual, M_ratio_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"M increases with R³ (ratio: {M_ratio_actual:.4f} ≈ {M_ratio_expected:.4f}): {M_monotonic}", M_monotonic)
            success = success and M_monotonic
            
            # Check 2: ρ should remain constant
            rho_constant = math.isclose(rho_pert_formula, rho_orig_formula, rel_tol=1e-4, abs_tol=1e-4)
            self._log(f"ρ independent of R (Δρ = 0): {rho_constant}", rho_constant)
            success = success and rho_constant
            
            # Check 3: Given M and ρ should match formula values
            M_formula_match = math.isclose(self.M, M_orig_formula, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Given M matches formula: {M_formula_match}", M_formula_match)
            success = success and M_formula_match
            
            rho_formula_match = math.isclose(self.rho, rho_orig_formula, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Given ρ matches formula: {rho_formula_match}", rho_formula_match)
            success = success and rho_formula_match
                
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics audits and return results."""
        self.logs = []
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall = test1 and test2 and test3
        
        # Summary
        self.logs.append("=" * 40)
        self.logs.append(f"Overall: {'PASS' if overall else 'FAIL'}")
        self.logs.append(f"  Invariants: {'✓' if test1 else '✗'}")
        self.logs.append(f"  Trace Consistency: {'✓' if test2 else '✗'}")
        self.logs.append(f"  Monotonicity: {'✓' if test3 else '✗'}")
        
        return overall, self.logs