import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallbacks
        self.solution = solution_instance
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key quantities with safe defaults
        self.E = self._safe_get('E', 0.0)
        self.F_original = self._safe_get('F', 0.0)  # Original force from step 1
        self.F_prime = self._safe_get("F'", self._safe_get('F_prime', 0.0))  # Force at point B
        self.E_all = self._safe_get('E_all', 0.0)
        self.direction = self._safe_get('Direction', '')
        
        # Extract inputs if available
        self.q = self._safe_get_input('q', 0.0)  # Test charge
        self.k = self._safe_get_input('k', 9e9)  # Coulomb constant
        self.r_A = self._safe_get_input('r_A', 0.0)  # Distance to point A
        self.r_B = self._safe_get_input('r_B', 0.0)  # Distance to point B
        
        # Reference values from steps analysis
        self.ref_E = 4.0/3.0  # 1.33 N/C ≈ 4/3
        self.ref_F_prime = 1e-4  # 0.0001 N
        self.ref_E_all = 8.0/3.0  # 2.67 N/C ≈ 8/3
        
        # Tolerance for comparisons
        self.rel_tol = 1e-2  # 1% relative tolerance
        self.abs_tol = 1e-2  # 0.01 absolute tolerance
        
    def _safe_get(self, key: str, default: float) -> float:
        """Safely get a result value with multiple fallbacks."""
        # Try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return float(val)
        
        # Try results dict
        val = self.results.get(key, None)
        if val is not None:
            return float(val)
        
        # Try variations for special keys
        if key == "F'":
            val = self.results.get('F_prime', None)
            if val is not None:
                return float(val)
        
        return default
    
    def _safe_get_input(self, key: str, default: float) -> float:
        """Safely get an input value."""
        if hasattr(self.solution, 'inputs'):
            val = self.solution.inputs.get(key, default)
            return float(val)
        return default
    
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self) -> List[str]:
        """Test fundamental physics invariants."""
        logs = []
        
        # 1. Coulomb's Law: F = k*q1*q2/r²
        logs.append("Testing Coulomb's Law invariants...")
        
        if self.q != 0 and self.r_A != 0 and self.F_original != 0:
            # Compute expected force using Coulomb's law
            # Assuming q1 = Q (source charge), q2 = q (test charge)
            # We need to find Q from E = k*Q/r_A²
            if self.E != 0:
                Q = (self.E * self.r_A**2) / self.k
                expected_F = self.k * abs(Q * self.q) / self.r_A**2
                
                if not self._is_close(expected_F, self.F_original):
                    logs.append(f"  WARNING: Coulomb's law mismatch. Expected F ≈ {expected_F:.4e}, got {self.F_original:.4e}")
                else:
                    logs.append(f"  OK: Coulomb's law satisfied (F = kQq/r²)")
        
        # 2. Field-Force Relationship: E = F/q (definition of electric field)
        if self.q != 0 and self.F_original != 0:
            expected_E = self.F_original / self.q
            if not self._is_close(expected_E, self.E):
                logs.append(f"  WARNING: E = F/q mismatch. Expected E ≈ {expected_E:.4f}, got {self.E:.4f}")
            else:
                logs.append(f"  OK: Field definition satisfied (E = F/q)")
        
        # 3. Force ratio from distance ratio: F_B/F_A = (r_A/r_B)²
        if self.r_A != 0 and self.r_B != 0 and self.F_original != 0:
            expected_ratio = (self.r_A / self.r_B)**2
            actual_ratio = self.F_prime / self.F_original
            
            if not self._is_close(expected_ratio, actual_ratio):
                logs.append(f"  WARNING: Force-distance inverse square law mismatch. Expected ratio {expected_ratio:.4f}, got {actual_ratio:.4f}")
            else:
                logs.append(f"  OK: Inverse square law satisfied")
        
        # 4. Superposition: E_all = 2E (if two equal charges at symmetric positions)
        if self.E != 0:
            expected_E_all = 2 * self.E
            if not self._is_close(expected_E_all, self.E_all):
                logs.append(f"  WARNING: Superposition mismatch. Expected E_all ≈ {expected_E_all:.4f}, got {self.E_all:.4f}")
            else:
                logs.append(f"  OK: Superposition principle satisfied (E_all = 2E)")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Check intermediate steps are consistent with final answers."""
        logs = []
        logs.append("Testing trace consistency...")
        
        # Check against reference values from steps analysis
        if not self._is_close(self.E, self.ref_E):
            logs.append(f"  WARNING: E mismatch. Reference {self.ref_E:.4f}, got {self.E:.4f}")
        else:
            logs.append(f"  OK: E ≈ {self.E:.4f} matches reference {self.ref_E:.4f}")
        
        if not self._is_close(self.F_prime, self.ref_F_prime):
            logs.append(f"  WARNING: F' mismatch. Reference {self.ref_F_prime:.4e}, got {self.F_prime:.4e}")
        else:
            logs.append(f"  OK: F' ≈ {self.F_prime:.4e} matches reference {self.ref_F_prime:.4e}")
        
        if not self._is_close(self.E_all, self.ref_E_all):
            logs.append(f"  WARNING: E_all mismatch. Reference {self.ref_E_all:.4f}, got {self.E_all:.4f}")
        else:
            logs.append(f"  OK: E_all ≈ {self.E_all:.4f} matches reference {self.ref_E_all:.4f}")
        
        # Check logical consistency: F' should be 1/4 of original F (from distance doubling)
        expected_F_prime = self.F_original / 4.0
        if self.F_original != 0:
            if not self._is_close(self.F_prime, expected_F_prime):
                logs.append(f"  WARNING: F' should be F/4 ≈ {expected_F_prime:.4e}, got {self.F_prime:.4e}")
            else:
                logs.append(f"  OK: F' = F/4 relationship holds")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Test that perturbing inputs produces expected trends."""
        logs = []
        logs.append("Testing monotonicity via perturbation...")
        
        # Save original state
        original_q = self.q
        
        # Test 1: Increase test charge magnitude
        if original_q != 0:
            perturbed_q = original_q * 1.1  # 10% increase
            
            # Predict new force using F = q * E (E field at point A is fixed by source)
            expected_F_new = perturbed_q * self.E
            
            # Check trend: Force should increase proportionally with charge
            if expected_F_new > self.F_original:
                logs.append(f"  OK: Force increases with charge (F ∝ q)")
            else:
                logs.append(f"  WARNING: Force should increase with charge, but predicted {expected_F_new:.4e} ≤ {self.F_original:.4e}")
        
        # Test 2: Increase distance (if distances are available)
        if self.r_A != 0 and self.r_B != 0:
            perturbed_r_A = self.r_A * 1.1  # 10% increase
            
            # Predict new E field using E = kQ/r²
            # We need Q first
            if self.E != 0:
                Q = (self.E * self.r_A**2) / self.k
                expected_E_new = self.k * Q / (perturbed_r_A**2)
                
                # Check trend: E field should decrease with distance
                if expected_E_new < self.E:
                    logs.append(f"  OK: E field decreases with distance (E ∝ 1/r²)")
                else:
                    logs.append(f"  WARNING: E should decrease with distance, but predicted {expected_E_new:.4f} ≥ {self.E:.4f}")
        
        # Test 3: Check superposition scaling
        if self.E != 0:
            # If we double the source charges (not changing positions), E_all should double
            expected_E_all_new = 2 * 2 * self.E  # Double the charges → double each contribution → 4E total?
            # Actually careful: if E_all = 2E (from two equal charges), doubling each charge gives E_all_new = 4E
            # But we can't test without charge values
            
            logs.append(f"  INFO: Monotonicity tests require charge values for full verification")
        
        return logs
    
    def test_direction_logic(self) -> List[str]:
        """Check force and field direction consistency."""
        logs = []
        logs.append("Testing direction logic...")
        
        # Direction can be stored as string or numeric flag
        # If it's a string, check for expected text
        # If it's numeric, just verify it's not zero/default
        if isinstance(self.direction, str):
            if self.direction:
                if "A to B" in self.direction or "from A to B" in self.direction.lower():
                    logs.append(f"  OK: Direction '{self.direction}' matches expected 'From A to B'")
                else:
                    logs.append(f"  WARNING: Unexpected direction '{self.direction}', expected 'From A to B'")
            else:
                logs.append(f"  WARNING: No direction specified in results")
        else:
            # Numeric representation - just check it's not the default
            if self.direction != 0.0:
                logs.append(f"  OK: Direction represented numerically as {self.direction}")
            else:
                logs.append(f"  WARNING: Direction is zero/default value")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run each test
            for test_name, test_method in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Direction Logic", self.test_direction_logic),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== {test_name} ===")
                test_logs = test_method()
                all_logs.extend(test_logs)
                
                # Check for warnings in test logs
                if any("WARNING" in log for log in test_logs):
                    success = False
        
        except Exception as e:
            all_logs.append(f"\nERROR during testing: {str(e)}")
            all_logs.append(f"Traceback: {traceback.format_exc()}")
            success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if success:
            all_logs.append("All physics checks passed within tolerance.")
        else:
            all_logs.append("WARNING: Some physics checks failed or raised concerns.")
        
        return success, all_logs