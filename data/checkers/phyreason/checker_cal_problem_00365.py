import math
import sympy as sp
from typing import Dict, Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely
        self.B = getattr(solution_instance, 'B', 1.0)  # Default to 1.0 if not provided
        self.L = getattr(solution_instance, 'L', 1.0)  # Default to 1.0 if not provided
        
        # Extract symbolic expressions from attributes (if available)
        self.Phi1_sym = getattr(solution_instance, 'Phi_1_sym', None)
        self.Phi2_sym = getattr(solution_instance, 'Phi_2_sym', None)
        self.r_sym = getattr(solution_instance, 'r_sym', None)
        
        # Extract numeric results from results dict
        self.Phi1_num = self.results.get('Φ_1', None)
        self.Phi2_num = self.results.get('Φ_2', None)
        self.r_num = self.results.get('r', None)
        
        # For symbolic calculations
        self.B_sym, self.L_sym = sp.symbols('B L', positive=True)
        
    def _compare_numeric(self, val1, val2, description: str) -> Tuple[bool, str]:
        """Compare two numeric values with fuzzy matching."""
        try:
            if val1 is None or val2 is None:
                return False, f"✗ {description}: Missing value(s)"
            if math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"✓ {description}: {val1:.4g} ≈ {val2:.4g}"
            else:
                return False, f"✗ {description}: {val1:.4g} ≠ {val2:.4g}"
        except Exception as e:
            return False, f"✗ {description}: Comparison failed with error: {e}"

    def test_invariants(self) -> List[str]:
        """Test fundamental physics invariants using numeric values."""
        logs = []
        
        # 1. Magnetic flux formula check for square coil (numeric)
        try:
            expected_Phi1_num = self.B * self.L**2
            check1, msg1 = self._compare_numeric(self.Phi1_num, expected_Phi1_num, 
                                               "Square flux Φ₁ = B·L² (numeric)")
            logs.append(msg1)
        except Exception as e:
            logs.append(f"✗ Square flux invariant failed: {e}")
        
        # 2. Perimeter conservation: r = 2L/π (numeric)
        try:
            if self.r_num is not None:
                expected_r_num = 2 * self.L / math.pi
                check2, msg2 = self._compare_numeric(self.r_num, expected_r_num, 
                                                   "Radius r = 2L/π (numeric)")
                logs.append(msg2)
        except Exception as e:
            logs.append(f"✗ Perimeter conservation failed: {e}")
        
        # 3. Magnetic flux formula check for circular coil (numeric)
        try:
            if self.r_num is not None and self.Phi2_num is not None:
                expected_Phi2_num = self.B * math.pi * self.r_num**2
                check3, msg3 = self._compare_numeric(self.Phi2_num, expected_Phi2_num, 
                                                   "Circular flux Φ₂ = B·π·r² (numeric)")
                logs.append(msg3)
        except Exception as e:
            logs.append(f"✗ Circular flux invariant failed: {e}")
        
        # 4. Alternative Φ₂ formula check (numeric)
        try:
            if self.Phi2_num is not None:
                expected_Phi2_alt_num = 4 * self.B * self.L**2 / math.pi
                check4, msg4 = self._compare_numeric(self.Phi2_num, expected_Phi2_alt_num, 
                                                   "Circular flux Φ₂ = 4B·L²/π (numeric)")
                logs.append(msg4)
        except Exception as e:
            logs.append(f"✗ Alternative circular flux formula failed: {e}")
        
        return logs

    def test_trace_consistency(self) -> List[str]:
        """Test consistency between intermediate steps and final results (numeric)."""
        logs = []
        
        # Check that using the computed radius gives the correct Φ₂ (numeric)
        try:
            if self.r_num is not None and self.Phi2_num is not None:
                # Recompute Φ₂ from r
                recomputed_Phi2_num = self.B * math.pi * self.r_num**2
                check, msg = self._compare_numeric(self.Phi2_num, recomputed_Phi2_num, 
                                                 "Φ₂ recomputed from radius r matches stored Φ₂ (numeric)")
                logs.append(msg)
        except Exception as e:
            logs.append(f"✗ Trace consistency check failed: {e}")
        
        # Check ratio between Φ₂ and Φ₁ (numeric)
        try:
            if self.Phi1_num is not None and self.Phi2_num is not None:
                expected_ratio = 4 / math.pi
                actual_ratio = self.Phi2_num / self.Phi1_num
                check, msg = self._compare_numeric(actual_ratio, expected_ratio, 
                                                 "Ratio Φ₂/Φ₁ = 4/π (numeric)")
                logs.append(msg)
        except Exception as e:
            logs.append(f"✗ Ratio check failed: {e}")
        
        return logs

    def test_monotonicity(self) -> List[str]:
        """Test that flux increases with increasing L (positive B)."""
        logs = []
        
        # Only run if we have numerical values
        if self.B is not None and self.L is not None:
            try:
                # Perturb L by +1%
                L_perturbed = self.L * 1.01
                
                # Compute perturbed fluxes
                Phi1_original = self.B * self.L**2
                Phi1_perturbed = self.B * L_perturbed**2
                
                Phi2_original = 4 * self.B * self.L**2 / math.pi
                Phi2_perturbed = 4 * self.B * L_perturbed**2 / math.pi
                
                # Check monotonic increase (for positive B)
                if self.B > 0:
                    if Phi1_perturbed > Phi1_original:
                        logs.append(f"✓ Φ₁ increases with L: {Phi1_original:.4g} → {Phi1_perturbed:.4g}")
                    else:
                        logs.append(f"✗ Φ₁ monotonicity failed: {Phi1_original:.4g} → {Phi1_perturbed:.4g}")
                    
                    if Phi2_perturbed > Phi2_original:
                        logs.append(f"✓ Φ₂ increases with L: {Phi2_original:.4g} → {Phi2_perturbed:.4g}")
                    else:
                        logs.append(f"✗ Φ₂ monotonicity failed: {Phi2_original:.4g} → {Phi2_perturbed:.4g}")
                else:
                    logs.append("⚠ Monotonicity test skipped: B is non-positive")
                    
            except Exception as e:
                logs.append(f"✗ Monotonicity test failed: {e}")
        else:
            logs.append("⚠ Monotonicity test skipped: B or L not available for numerical evaluation")
        
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        
        # Run tests
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(self.test_monotonicity())
        
        # Determine success: count failures
        failures = sum(1 for log in all_logs if log.startswith("✗"))
        success = failures == 0
        
        return success, all_logs