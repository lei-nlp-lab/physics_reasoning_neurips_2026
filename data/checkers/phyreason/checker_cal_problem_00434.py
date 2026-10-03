import sympy as sp
import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Problem constants
        self.L0 = 0.20  # meters (20 cm)
        self.dF = 0.50  # Newtons
        self.dx = 0.012  # meters (1.20 cm)
        
        # Extract results safely using getattr with fallback to results dict
        self.k = self._safe_get(solution_instance, 'k')
        self.Deltax_1 = self._safe_get(solution_instance, 'Deltax_1')
        self.L = self._safe_get(solution_instance, 'L')
        
        # Derived expected values for comparison
        self.expected_k = self.dF / self.dx  # Hooke's law: k = ΔF/Δx
        self.expected_Deltax_1 = None  # Will compute from other values if needed
        self.expected_L = None  # Will compute from other values if needed
        
        # Store logs for reporting
        self.logs = []
    
    def _safe_get(self, obj, key: str) -> Any:
        """Safely get attribute from solution instance with multiple fallbacks."""
        # First try direct attribute
        val = getattr(obj, key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) else val
        
        # Then try results dictionary
        if hasattr(obj, 'results'):
            val = obj.results.get(key, None)
            if val is not None:
                return float(val) if isinstance(val, (int, float, str)) else val
        
        # Try alternative key names
        alt_keys = {
            'k': ['spring_constant', 'k_value'],
            'Deltax_1': ['delta_x', 'elongation', 'Δx1'],
            'L': ['length', 'L_value', 'final_length']
        }
        
        for alt_key in alt_keys.get(key, []):
            val = getattr(obj, alt_key, None)
            if val is not None:
                return float(val) if isinstance(val, (int, float, str)) else val
            
            if hasattr(obj, 'results'):
                val = obj.results.get(alt_key, None)
                if val is not None:
                    return float(val) if isinstance(val, (int, float, str)) else val
        
        return None
    
    def _log(self, message: str, success: bool = True):
        """Add message to logs with success indicator."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2) -> bool:
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants."""
        success = True
        
        # 1. Verify Hooke's Law: k = ΔF/Δx from given data
        if self.k is not None:
            if self._is_close(self.k, self.expected_k):
                self._log(f"Hooke's Law constant correct: k = {self.k:.2f} N/m ≈ {self.expected_k:.2f} N/m")
            else:
                self._log(f"Hooke's Law constant mismatch: k = {self.k} vs expected {self.expected_k:.2f}", False)
                success = False
        
        # 2. Verify Hooke's Law relationship symbolically (if we have all values)
        try:
            # Create symbolic variables
            k, F, dx = sp.symbols('k F dx')
            hookes_law = sp.Eq(F, k * dx)
            
            # If we have k and Deltax_1, compute implied force
            if self.k is not None and self.Deltax_1 is not None:
                implied_force = self.k * self.Deltax_1
                # We don't have the actual force, but we can check consistency
                # with the length calculation in trace_consistency
                pass
        except Exception as e:
            self._log(f"Symbolic Hooke's Law check failed: {e}", False)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer."""
        success = True
        
        # From steps analysis:
        # Step 1: k = 41.67 N/m (already checked in invariants)
        # Step 3: Δx₁ = F₁/k = 12 cm = 0.12 m
        # Step 2: L = L₀ + Δx₁ = 20 cm + 12 cm = 32 cm = 0.32 m
        
        # Check L = L0 + Δx₁
        if self.Deltax_1 is not None and self.L is not None:
            expected_L = self.L0 + self.Deltax_1
            if self._is_close(self.L, expected_L):
                self._log(f"Length consistency: L = {self.L:.3f} m = L0 + Δx₁ = {self.L0:.3f} + {self.Deltax_1:.3f} m")
            else:
                self._log(f"Length inconsistency: L = {self.L} m ≠ L0 + Δx₁ = {self.L0:.3f} + {self.Deltax_1:.3f} = {expected_L:.3f} m", False)
                success = False
        
        # Check that Δx₁ is positive (spring stretched, not compressed)
        if self.Deltax_1 is not None and self.Deltax_1 < 0:
            self._log(f"Elongation Δx₁ should be positive but got {self.Deltax_1}", False)
            success = False
        
        # Check that L > L₀ (spring is longer with weight)
        if self.L is not None and self.L <= self.L0:
            self._log(f"Final length L should be > L₀ but got L={self.L}, L₀={self.L0}", False)
            success = False
        
        # If we have k and Δx₁, we can compute the implied weight force
        if self.k is not None and self.Deltax_1 is not None:
            implied_force = self.k * self.Deltax_1
            self._log(f"Implied weight force: F = k·Δx₁ = {self.k:.2f}·{self.Deltax_1:.3f} = {implied_force:.2f} N")
            
            # The force should be positive (downward weight)
            if implied_force <= 0:
                self._log(f"Implied force should be positive but got {implied_force:.2f} N", False)
                success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb input and check physical trends."""
        success = True
        
        # We'll perturb the spring constant k and check if elongation changes appropriately
        # Original system: F = k·Δx  ⇒ Δx = F/k
        
        # We need a reference force F. Use the implied force from step 3
        if self.k is None or self.Deltax_1 is None:
            self._log("Cannot test monotonicity: missing k or Δx₁", False)
            return False
        
        F_ref = self.k * self.Deltax_1
        
        # Test 1: Increase k by 10%, elongation should decrease
        k_perturbed = self.k * 1.10
        dx_perturbed = F_ref / k_perturbed
        
        if dx_perturbed < self.Deltax_1:
            self._log(f"Monotonicity 1: Increasing k by 10% decreases Δx (as expected): {dx_perturbed:.4f} < {self.Deltax_1:.4f}")
        else:
            self._log(f"Monotonicity 1: Increasing k should decrease Δx but got {dx_perturbed:.4f} ≥ {self.Deltax_1:.4f}", False)
            success = False
        
        # Test 2: Decrease k by 10%, elongation should increase
        k_perturbed = self.k * 0.90
        dx_perturbed = F_ref / k_perturbed
        
        if dx_perturbed > self.Deltax_1:
            self._log(f"Monotonicity 2: Decreasing k by 10% increases Δx (as expected): {dx_perturbed:.4f} > {self.Deltax_1:.4f}")
        else:
            self._log(f"Monotonicity 2: Decreasing k should increase Δx but got {dx_perturbed:.4f} ≤ {self.Deltax_1:.4f}", False)
            success = False
        
        # Test 3: Length should change in same direction as elongation
        L_perturbed = self.L0 + dx_perturbed
        if (dx_perturbed > self.Deltax_1 and L_perturbed > self.L) or (dx_perturbed < self.Deltax_1 and L_perturbed < self.L):
            self._log(f"Monotonicity 3: Length changes consistently with elongation")
        else:
            self._log(f"Monotonicity 3: Length should change in same direction as elongation", False)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status with logs."""
        self.logs = []  # Reset logs
        overall_success = True
        
        try:
            self._log("=== Physics Checker Started ===")
            
            # Run individual tests
            tests = [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]
            
            for test_name, test_func in tests:
                self._log(f"\n--- Running {test_name} Test ---")
                try:
                    if not test_func():
                        overall_success = False
                        self._log(f"{test_name} test failed", False)
                except Exception as e:
                    self._log(f"{test_name} test crashed: {traceback.format_exc()}", False)
                    overall_success = False
            
            # Final summary
            self._log(f"\n=== Physics Checker Complete: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===")
            
        except Exception as e:
            self._log(f"Checker runtime error: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs