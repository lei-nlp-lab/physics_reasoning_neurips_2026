import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe extraction with fallbacks
        self.c = 3.0e8  # Speed of light in vacuum (m/s)
        self.f = 37.5e3  # Frequency (Hz) from problem context
        
        # Extract results safely - FIRST check results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        self.lam = results_dict.get('lambda', None)
        self.C = results_dict.get('C', None)
        
        # For LC circuit check, we need inductance L if available
        self.L = results_dict.get('L', 4e-3)  # Default from problem
        
        # Store original values for monotonicity test restoration
        self._orig_f = self.f
        self._orig_lam = self.lam
        self._orig_C = self.C
        self._orig_L = self.L
        
        # Target answers for validation
        self.target_lam = 8000.0  # meters
        self.target_C = 4.5e-9  # Farads
        
        self.logs = []

    def _log(self, message: str) -> None:
        """Helper to add timestamped log entries"""
        self.logs.append(message)

    def test_invariants(self) -> bool:
        """Test fundamental physics invariants"""
        success = True
        
        # 1. Wave equation: c = f * λ
        if self.lam is not None:
            expected_lam = self.c / self.f
            if not math.isclose(self.lam, expected_lam, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Wave equation violation. λ_actual={self.lam}, λ_expected={expected_lam}")
                success = False
            else:
                self._log(f"PASS: Wave equation c = fλ satisfied")
        
        # 2. LC resonant frequency formula (if L is available)
        if self.L is not None and self.C is not None and self.C > 0:
            # Numerical check only - no SymPy
            try:
                f_calculated = 1/(2*math.pi*math.sqrt(self.L*self.C))
                if not math.isclose(self.f, f_calculated, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: LC resonance violation. f_actual={self.f}, f_calc={f_calculated}")
                    success = False
                else:
                    self._log(f"PASS: LC resonance formula satisfied")
            except Exception as e:
                self._log(f"WARN: LC resonance check failed with error: {e}")
        
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers"""
        success = True
        
        # Check wavelength against target
        if self.lam is not None:
            if not math.isclose(self.lam, self.target_lam, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Wavelength mismatch. λ={self.lam}, target={self.target_lam}")
                success = False
            else:
                self._log(f"PASS: Wavelength matches target (8000m)")
        
        # Check capacitance against target
        if self.C is not None and self.C > 0:
            if not math.isclose(self.C, self.target_C, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Capacitance mismatch. C={self.C}, target={self.target_C}")
                success = False
            else:
                self._log(f"PASS: Capacitance matches target (4.5e-9F)")
        
        # Verify that if we compute C from wavelength formula (if L available), it's consistent
        if self.L is not None and self.lam is not None:
            # From LC resonance: C = 1/(4π²f²L)
            C_calc = 1/(4*(math.pi**2)*(self.f**2)*self.L)
            if self.C is not None and self.C > 0:
                if not math.isclose(self.C, C_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: C from LC formula inconsistent. C={self.C}, C_calc={C_calc}")
                    success = False
                else:
                    self._log(f"PASS: C consistent with LC formula")
        
        return success

    def test_monotonicity(self) -> bool:
        """Test expected trends by perturbing inputs"""
        success = True
        
        # Test 1: Frequency vs Wavelength (inverse relationship)
        if self.lam is not None:
            # Increase frequency by 10%
            f_perturbed = self.f * 1.1
            lam_perturbed = self.c / f_perturbed
            
            # Expected: higher frequency → shorter wavelength
            if not (lam_perturbed < self.lam):
                self._log(f"FAIL: Monotonicity violated: f↑ should cause λ↓")
                success = False
            else:
                self._log(f"PASS: λ decreases when f increases (inverse relationship)")
            
            # Restore original values
            self.lam = self._orig_lam
        
        # Test 2: Frequency vs Capacitance (if L available and C is valid)
        if self.L is not None and self.C is not None and self.C > 0:
            # Increase frequency by 10%
            f_perturbed = self.f * 1.1
            C_perturbed = 1/(4*(math.pi**2)*(f_perturbed**2)*self.L)
            
            # Expected: higher frequency → smaller capacitance for resonance
            if not (C_perturbed < self.C):
                self._log(f"FAIL: Monotonicity violated: f↑ should cause C↓ for LC resonance")
                success = False
            else:
                self._log(f"PASS: C decreases when f increases (LC resonance)")
            
            # Restore original values
            self.C = self._orig_C
            self.f = self._orig_f
        
        # Test 3: Inductance vs Capacitance (if L available and C is valid)
        if self.L is not None and self.C is not None and self.C > 0:
            # Increase inductance by 10%
            L_perturbed = self.L * 1.1
            C_perturbed = 1/(4*(math.pi**2)*(self.f**2)*L_perturbed)
            
            # Expected: higher inductance → smaller capacitance for same frequency
            if not (C_perturbed < self.C):
                self._log(f"FAIL: Monotonicity violated: L↑ should cause C↓ for same f")
                success = False
            else:
                self._log(f"PASS: C decreases when L increases (inverse square relationship)")
            
            # Restore original values
            self.C = self._orig_C
            self.L = self._orig_L
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all validation tests and return results"""
        self.logs = []
        self.logs.append("=== Physics Checker Start ===")
        
        # Initialize success flags
        invariants_ok = True
        trace_ok = True
        monotonicity_ok = True
        
        # Run tests
        try:
            self.logs.append("\n--- Testing Physics Invariants ---")
            invariants_ok = self.test_invariants()
        except Exception as e:
            self._log(f"ERROR in test_invariants: {e}")
            invariants_ok = False
        
        try:
            self.logs.append("\n--- Testing Trace Consistency ---")
            trace_ok = self.test_trace_consistency()
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {e}")
            trace_ok = False
        
        try:
            self.logs.append("\n--- Testing Monotonicity Trends ---")
            monotonicity_ok = self.test_monotonicity()
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {e}")
            monotonicity_ok = False
        
        # Overall result
        overall_success = invariants_ok and trace_ok and monotonicity_ok
        
        # Summary
        self.logs.append("\n=== Summary ===")
        self.logs.append(f"Invariants: {'PASS' if invariants_ok else 'FAIL'}")
        self.logs.append(f"Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if monotonicity_ok else 'FAIL'}")
        self.logs.append(f"OVERALL: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs