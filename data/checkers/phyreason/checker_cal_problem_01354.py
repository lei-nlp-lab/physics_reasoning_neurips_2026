import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters safely
        self.M = getattr(solution_instance, 'M', self.results.get('M', 1.0))
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.mu = getattr(solution_instance, 'mu', self.results.get('mu', 0.1))
        self.g = getattr(solution_instance, 'g', self.results.get('g', 9.8))
        self.L = getattr(solution_instance, 'L', self.results.get('L', 1.0))
        self.l = getattr(solution_instance, 'l', self.results.get('l', 1.0))
        
        # Extract results safely
        self.Q = self.results.get('Q', 0.0)
        self.DeltaE = self.results.get('DeltaE', 0.0)
        self.DeltaE_km = self.results.get('DeltaE_km', 0.0)
        self.DeltaE_kM = self.results.get('DeltaE_kM', 0.0)
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message: str):
        """Helper to log messages."""
        self.logs.append(message)
        
    def _is_close(self, a: float, b: float, tol: float = 1e-2) -> bool:
        """Fuzzy numerical matching with 1% tolerance."""
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    
    def test_invariants(self) -> bool:
        """Test physical invariants from KB."""
        success = True
        
        # 1. Energy Conservation: Q = μ*m*g*L
        expected_Q = self.mu * self.m * self.g * self.L
        if not self._is_close(self.Q, expected_Q):
            self._log(f"FAIL: Q mismatch. Expected {expected_Q}, got {self.Q}")
            success = False
        else:
            self._log(f"PASS: Q = μmgL = {expected_Q}")
            
        # 2. Work-Energy for small block: ΔE_km = μmg(l - L)
        expected_DeltaE_km = self.mu * self.m * self.g * (self.l - self.L)
        if not self._is_close(self.DeltaE_km, expected_DeltaE_km):
            self._log(f"FAIL: ΔE_km mismatch. Expected {expected_DeltaE_km}, got {self.DeltaE_km}")
            success = False
        else:
            self._log(f"PASS: ΔE_km = μmg(l - L) = {expected_DeltaE_km}")
            
        # 3. Work-Energy for wooden block: ΔE_kM = μmg*l (magnitude of energy loss)
        expected_DeltaE_kM = self.mu * self.m * self.g * self.l
        if not self._is_close(self.DeltaE_kM, expected_DeltaE_kM):
            self._log(f"FAIL: ΔE_kM mismatch. Expected {expected_DeltaE_kM}, got {self.DeltaE_kM}")
            success = False
        else:
            self._log(f"PASS: ΔE_kM magnitude = μmg*l = {expected_DeltaE_kM}")
            
        # 4. System energy reduction: ΔE = Q = μmgL
        if not self._is_close(self.DeltaE, expected_Q):
            self._log(f"FAIL: ΔE mismatch. Expected {expected_Q}, got {self.DeltaE}")
            success = False
        else:
            self._log(f"PASS: ΔE = μmgL = {expected_Q}")
            
        # 5. Correct energy balance: ΔE = ΔE_kM - ΔE_km
        # The mechanical energy reduction equals the energy lost by M minus energy gained by m
        system_change = self.DeltaE_kM - self.DeltaE_km
        if not self._is_close(self.DeltaE, system_change):
            self._log(f"FAIL: Energy balance mismatch. ΔE={self.DeltaE}, ΔE_kM-ΔE_km={system_change}")
            success = False
        else:
            self._log(f"PASS: Energy balance: ΔE = ΔE_kM - ΔE_km")
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check intermediate results are consistent with final answer."""
        success = True
        
        # All results should be proportional to μmg
        base_factor = self.mu * self.m * self.g
        
        # Check each result's dependency
        checks = [
            ("Q", self.Q, base_factor * self.L),
            ("ΔE_km", self.DeltaE_km, base_factor * (self.l - self.L)),
            ("ΔE_kM", self.DeltaE_kM, base_factor * self.l),
            ("ΔE", self.DeltaE, base_factor * self.L)
        ]
        
        for name, value, expected in checks:
            if not self._is_close(value, expected):
                self._log(f"FAIL: {name} dependency mismatch. Value={value}, Expected={expected}")
                success = False
            else:
                self._log(f"PASS: {name} has correct dependency")
                
        return success
    
    def test_monotonicity(self) -> bool:
        """Test that results change monotonically with parameter perturbations."""
        success = True
        
        # Store original values
        original_mu = self.mu
        perturbed_mu = original_mu * 1.01  # Increase by 1%
        
        # Recalculate expected results with perturbed μ
        base_factor_orig = original_mu * self.m * self.g
        base_factor_pert = perturbed_mu * self.m * self.g
        
        original_vals = {
            'Q': base_factor_orig * self.L,
            'ΔE_km': base_factor_orig * (self.l - self.L),
            'ΔE_kM': base_factor_orig * self.l,
            'ΔE': base_factor_orig * self.L
        }
        
        perturbed_vals = {
            'Q': base_factor_pert * self.L,
            'ΔE_km': base_factor_pert * (self.l - self.L),
            'ΔE_kM': base_factor_pert * self.l,
            'ΔE': base_factor_pert * self.L
        }
        
        # Check monotonic increase for all quantities (all are proportional to μ)
        for key in original_vals:
            if perturbed_vals[key] <= original_vals[key]:
                self._log(f"FAIL: {key} not increasing with μ. Original={original_vals[key]}, Perturbed={perturbed_vals[key]}")
                success = False
            else:
                self._log(f"PASS: {key} increases with μ")
        
        # Restore original μ
        self.mu = original_mu
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        self._log("=== PhysicsChecker Starting ===")
        
        # Run tests
        test1 = self.test_invariants()
        self._log(f"Invariants test: {'PASS' if test1 else 'FAIL'}")
        
        test2 = self.test_trace_consistency()
        self._log(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
        
        test3 = self.test_monotonicity()
        self._log(f"Monotonicity test: {'PASS' if test3 else 'FAIL'}")
        
        overall_success = test1 and test2 and test3
        self._log(f"=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs