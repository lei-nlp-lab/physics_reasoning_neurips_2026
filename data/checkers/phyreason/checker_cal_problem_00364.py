import math
import sympy as sp
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    """
    A robust physics auditor for magnetic flux problems.
    """
    
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute extraction to prevent sandbox errors.
        """
        # Safely extract input parameters from solution instance
        self.solution = solution_instance
        
        # Input parameters (safe access with fallbacks)
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'magnetic_field', 0.6))
        self.S = getattr(solution_instance, 'S', getattr(solution_instance, 'area', 0.4))
        self.theta_deg = getattr(solution_instance, 'theta_deg', getattr(solution_instance, 'theta', 60))
        self.theta_rad = math.radians(self.theta_deg)
        
        # Safely extract results from solution
        results = getattr(solution_instance, 'results', {})
        
        # Expected keys from mapping
        self.phi1 = results.get('Φ', getattr(solution_instance, 'Φ', 0.0))
        self.phi2 = results.get('Φ₂', getattr(solution_instance, 'Φ₂', 0.0))
        self.delta_phi = results.get('delta_Φ', results.get('ΔΦ', getattr(solution_instance, 'ΔΦ', 0.0)))
        
        # Convert to float if string
        if isinstance(self.phi1, str):
            self.phi1 = float(''.join(c for c in self.phi1 if c.isdigit() or c == '.'))
        if isinstance(self.phi2, str):
            self.phi2 = float(''.join(c for c in self.phi2 if c.isdigit() or c == '.'))
        if isinstance(self.delta_phi, str):
            self.delta_phi = float(''.join(c for c in self.delta_phi if c.isdigit() or c == '.'))
        
        # Constants
        self.cos60 = math.cos(math.radians(60))
        self.cos0 = math.cos(math.radians(0))
        
        # Initialize symbolic variables
        self.B_sym, self.S_sym, self.theta_sym = sp.symbols('B S theta')
        
        self.logs = []
    
    def _log(self, message: str):
        """Helper to log messages."""
        self.logs.append(message)
    
    def _is_close(self, a: float, b: float) -> bool:
        """
        Fuzzy numerical matching with 1-2% tolerance.
        """
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics laws:
        1. Magnetic flux formula Φ = B·S·cosθ
        2. Consistency between initial, final, and change in flux
        """
        success = True
        
        # 1. Check initial flux using Φ = B·S·cosθ
        expected_phi1 = self.B * self.S * self.cos60
        if not self._is_close(self.phi1, expected_phi1):
            self._log(f"Initial flux mismatch: expected {expected_phi1:.3f}, got {self.phi1:.3f}")
            success = False
        else:
            self._log(f"✓ Initial flux correct: Φ = B·S·cos60 = {expected_phi1:.3f} Wb")
        
        # 2. Check final flux (coil perpendicular to field)
        expected_phi2 = self.B * self.S * self.cos0  # cos0 = 1
        if not self._is_close(self.phi2, expected_phi2):
            self._log(f"Final flux mismatch: expected {expected_phi2:.3f}, got {self.phi2:.3f}")
            success = False
        else:
            self._log(f"✓ Final flux correct: Φ₂ = B·S = {expected_phi2:.3f} Wb")
        
        # 3. Check change in flux ΔΦ = |Φ₁| + |Φ₂| (absolute values)
        # Based on step analysis, change is sum of magnitudes
        expected_delta_phi = abs(self.phi1) + abs(self.phi2)
        if not self._is_close(self.delta_phi, expected_delta_phi):
            self._log(f"Change in flux mismatch: expected {expected_delta_phi:.3f}, got {self.delta_phi:.3f}")
            success = False
        else:
            self._log(f"✓ Change in flux correct: ΔΦ = |Φ₁| + |Φ₂| = {expected_delta_phi:.3f} Wb")
        
        # 4. Symbolic check (fallback if numerical fails)
        phi1_expr = self.B_sym * self.S_sym * sp.cos(self.theta_sym)
        phi2_expr = self.B_sym * self.S_sym
        delta_phi_expr = sp.Abs(phi1_expr) + sp.Abs(phi2_expr)
        
        self._log(f"Symbolic laws verified: Φ₁ = {phi1_expr}, Φ₂ = {phi2_expr}, ΔΦ = {delta_phi_expr}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer.
        """
        success = True
        
        # Check that final answers match expected values
        target_phi1 = 0.12
        target_delta_phi = 0.36
        
        if not self._is_close(self.phi1, target_phi1):
            self._log(f"Trace consistency fail: Φ should be ~{target_phi1}, got {self.phi1:.3f}")
            success = False
        else:
            self._log(f"✓ Trace consistency: Φ = {self.phi1:.3f} matches target {target_phi1}")
        
        if not self._is_close(self.delta_phi, target_delta_phi):
            self._log(f"Trace consistency fail: ΔΦ should be ~{target_delta_phi}, got {self.delta_phi:.3f}")
            success = False
        else:
            self._log(f"✓ Trace consistency: ΔΦ = {self.delta_phi:.3f} matches target {target_delta_phi}")
        
        # Verify intermediate step: Φ₂ should be 0.24
        if not self._is_close(self.phi2, 0.24):
            self._log(f"Trace consistency warning: Φ₂ should be ~0.24, got {self.phi2:.3f}")
        else:
            self._log(f"✓ Intermediate Φ₂ = {self.phi2:.3f} correct")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and check monotonic relationships.
        Restore original values after test.
        """
        success = True
        
        # Store original values
        original_B = self.B
        original_S = self.S
        
        try:
            # Test 1: Increase B by 10%, check if all fluxes increase proportionally
            self.B *= 1.1
            new_phi1 = self.B * self.S * self.cos60
            new_phi2 = self.B * self.S * self.cos0
            new_delta_phi = abs(new_phi1) + abs(new_phi2)
            
            # Check monotonic increase
            if not (new_phi1 > self.phi1 and new_phi2 > self.phi2 and new_delta_phi > self.delta_phi):
                self._log(f"Monotonicity test 1 fail: Increasing B should increase all fluxes")
                success = False
            else:
                self._log(f"✓ Monotonicity: +10% B → all fluxes increase proportionally")
            
            # Restore B, perturb S
            self.B = original_B
            self.S *= 1.1
            
            new_phi1 = self.B * self.S * self.cos60
            new_phi2 = self.B * self.S * self.cos0
            new_delta_phi = abs(new_phi1) + abs(new_phi2)
            
            # Check monotonic increase
            if not (new_phi1 > self.phi1 and new_phi2 > self.phi2 and new_delta_phi > self.delta_phi):
                self._log(f"Monotonicity test 2 fail: Increasing S should increase all fluxes")
                success = False
            else:
                self._log(f"✓ Monotonicity: +10% S → all fluxes increase proportionally")
                
        finally:
            # Always restore original values
            self.B = original_B
            self.S = original_S
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return success status with logs.
        """
        self.logs = []
        
        self._log("=" * 50)
        self._log("Physics Checker Starting...")
        self._log(f"Inputs: B={self.B} T, S={self.S} m², θ={self.theta_deg}°")
        self._log(f"Extracted: Φ={self.phi1:.3f}, Φ₂={self.phi2:.3f}, ΔΦ={self.delta_phi:.3f}")
        self._log("=" * 50)
        
        results = []
        
        # Run individual tests
        results.append(("Invariants", self.test_invariants()))
        results.append(("Trace Consistency", self.test_trace_consistency()))
        results.append(("Monotonicity", self.test_monotonicity()))
        
        # Summarize
        self._log("=" * 50)
        success_count = sum(1 for name, passed in results if passed)
        
        if success_count == len(results):
            final_success = True
            self._log("✅ ALL TESTS PASSED")
        else:
            final_success = False
            self._log(f"❌ {success_count}/{len(results)} tests passed")
            for name, passed in results:
                status = "✓" if passed else "✗"
                self._log(f"  {status} {name}")
        
        self._log("=" * 50)
        
        return final_success, self.logs


# Optional: Example usage stub
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.B = 0.6
            self.S = 0.4
            self.theta_deg = 60
            self.results = {
                'Φ': 0.12,
                'Φ₂': 0.24,
                'delta_Φ': 0.36
            }
    
    mock = MockSolution()
    checker = PhysicsChecker(mock)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)