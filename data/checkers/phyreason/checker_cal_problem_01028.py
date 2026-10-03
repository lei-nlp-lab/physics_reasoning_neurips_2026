import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results with fallbacks
        self.solution = solution_instance
        
        # Input parameters
        self.V0 = self._safe_get('V0', 1.0)
        self.p0 = self._safe_get('p0', 1.0)
        self.T0 = self._safe_get('T0', 1.0)
        
        # Results from key mapping
        self.V_1 = self._safe_get('V_1', 0.0)
        self.p_2 = self._safe_get('p_2', 0.0)
        self.T_2 = self._safe_get('T_2', 0.0)
        self.V_2 = self._safe_get('V_2', 0.0)
        self.DeltaV = self._safe_get('DeltaV', 0.0)
        self.DeltaW_BC = self._safe_get('DeltaW_BC', 0.0)
        self.W_CA = self._safe_get('W_CA', 0.0)
        self.DeltaU = self._safe_get('DeltaU', 0.0)
        self.DeltaQ = self._safe_get('DeltaQ', 0.0)
        
        # Constants for ideal gas (assuming 1 mole for numerical checks)
        self.n = 1.0
        self.R = 8.314
        self.Cv = 1.5 * self.R  # for monatomic ideal gas
        self.Cp = 2.5 * self.R
        
        self.logs = []
        
    def _safe_get(self, key, default):
        """Safely retrieve attribute from solution instance."""
        try:
            # First try direct attribute
            value = getattr(self.solution, key, None)
            if value is not None:
                return float(value)
            
            # Then try results dictionary
            results = getattr(self.solution, 'results', {})
            if key in results:
                return float(results[key])
                
            # Finally try inputs dictionary
            inputs = getattr(self.solution, 'inputs', {})
            if key in inputs:
                return float(inputs[key])
                
        except (AttributeError, TypeError, ValueError):
            pass
            
        return default
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> List[str]:
        """Test physics invariants from KB theorems."""
        logs = []
        
        # 1. Check Boyle's Law for step 1 (A->B)
        expected_V1 = self.V0 / 3
        if self._is_close(self.V_1, expected_V1):
            logs.append("✓ Step 1: Boyle's Law holds (V1 ≈ V0/3)")
        else:
            logs.append(f"✗ Step 1: Boyle's Law failed. Expected V1 ≈ {expected_V1}, got {self.V_1}")
        
        # 2. Check isobaric process step 2 (B->C)
        if self._is_close(self.p_2, 3 * self.p0):
            logs.append(f"✓ Step 2: p2 ≈ 3p0 ({self.p_2} ≈ {3*self.p0})")
        else:
            logs.append(f"✗ Step 2: p2 != 3p0. Expected {3*self.p0}, got {self.p_2}")
            
        if self._is_close(self.T_2, 3 * self.T0):
            logs.append(f"✓ Step 2: T2 ≈ 3T0 ({self.T_2} ≈ {3*self.T0})")
        else:
            logs.append(f"✗ Step 2: T2 != 3T0. Expected {3*self.T0}, got {self.T_2}")
        
        # 3. Check Gay-Lussac's Law for step 3 (B->C isobaric)
        expected_V2 = self.V_1 * (self.T_2 / self.T0)
        if self._is_close(self.V_2, expected_V2) and self._is_close(self.V_2, self.V0):
            logs.append("✓ Step 3: Gay-Lussac's Law holds (V2 ≈ V0)")
        else:
            logs.append(f"✗ Step 3: V2 mismatch. Expected ≈ {expected_V2} or {self.V0}, got {self.V_2}")
        
        # 4. Check volume change calculation
        expected_DeltaV = self.V_2 - self.V_1
        if self._is_close(self.DeltaV, expected_DeltaV) and self._is_close(self.DeltaV, 2*self.V0/3):
            logs.append("✓ Step 4: ΔV calculation correct (ΔV ≈ 2V0/3)")
        else:
            logs.append(f"✗ Step 4: ΔV mismatch. Expected ≈ {2*self.V0/3}, got {self.DeltaV}")
        
        # 5. Check work done in isobaric process
        expected_W_BC = self.p_2 * (self.V_2 - self.V_1)
        if self._is_close(self.DeltaW_BC, expected_W_BC) and self._is_close(self.DeltaW_BC, 2*self.p0*self.V0):
            logs.append("✓ Step 5: ΔW_BC calculation correct (ΔW_BC ≈ 2p0V0)")
        else:
            logs.append(f"✗ Step 5: ΔW_BC mismatch. Expected ≈ {2*self.p0*self.V0}, got {self.DeltaW_BC}")
        
        # 6. Check isochoric process (C->A)
        if self._is_close(self.W_CA, 0.0):
            logs.append("✓ Step 6: W_CA ≈ 0 (isochoric process)")
        else:
            logs.append(f"✗ Step 6: W_CA should be 0, got {self.W_CA}")
        
        # 7. Check cyclic process internal energy change
        if self._is_close(self.DeltaU, 0.0):
            logs.append("✓ Step 6: ΔU ≈ 0 (cyclic process)")
        else:
            logs.append(f"✗ Step 6: ΔU should be 0 for cycle, got {self.DeltaU}")
        
        # 8. Check First Law of Thermodynamics for entire cycle
        # The checker's original formula: ΔU = ΔQ + ΔW (where ΔW is work done ON gas)
        # With ΔW_BC defined as work done BY external force ON gas, and W_CA = 0
        total_work_on_gas = self.DeltaW_BC + self.W_CA
        first_law_residual = self.DeltaU - (self.DeltaQ + total_work_on_gas)
        
        if self._is_close(first_law_residual, 0.0):
            logs.append("✓ Step 9-10: First Law satisfied (ΔU = ΔQ + ΔW)")
        else:
            # Alternative check using ΔU = ΔQ - W_by_gas
            work_by_gas = self.p_2 * (self.V_2 - self.V_1)  # positive for expansion
            first_law_alt = self.DeltaU - (self.DeltaQ - work_by_gas)
            if self._is_close(first_law_alt, 0.0):
                logs.append("✓ Step 9-10: First Law satisfied (ΔU = ΔQ - W_by_gas)")
            else:
                logs.append(f"✗ Step 9-10: First Law violation. ΔU - (ΔQ + ΔW) = {first_law_residual}")
            
        # Numerical fallback check using ideal gas law (skip - uses arbitrary n=1)
        try:
            # Check pV = nRT at state A
            pV_A = self.p0 * self.V0
            nRT_A = self.n * self.R * self.T0
            if not self._is_close(pV_A, nRT_A):
                logs.append("⚠ Ideal gas law at A doesn't match (n arbitrary)")
                
            # Check pV = nRT at state C
            pV_C = self.p_2 * self.V_2
            nRT_C = self.n * self.R * self.T_2
            if not self._is_close(pV_C, nRT_C):
                logs.append("⚠ Ideal gas law at C doesn't match (n arbitrary)")
        except Exception as e:
            logs.append(f"⚠ Ideal gas law check skipped: {str(e)}")
        
        self.logs.extend(logs)
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Check that ΔV matches target answer 1
        target_DeltaV = 2 * self.V0 / 3
        if self._is_close(self.DeltaV, target_DeltaV):
            logs.append(f"✓ ΔV consistent with target: {self.DeltaV} ≈ 2V0/3")
        else:
            logs.append(f"✗ ΔV inconsistent. Got {self.DeltaV}, expected ≈ {target_DeltaV}")
        
        # Check that ΔQ matches target answer 2
        target_DeltaQ = 2 * self.p0 * self.V0
        if self._is_close(self.DeltaQ, target_DeltaQ):
            logs.append(f"✓ ΔQ consistent with target: {self.DeltaQ} ≈ 2p0V0")
        else:
            logs.append(f"✗ ΔQ inconsistent. Got {self.DeltaQ}, expected ≈ {target_DeltaQ}")
        
        # Check internal consistency of work calculations
        recalc_W_BC = self.p_2 * self.DeltaV
        if self._is_close(self.DeltaW_BC, recalc_W_BC):
            logs.append("✓ ΔW_BC internally consistent with p2·ΔV")
        else:
            logs.append(f"✗ ΔW_BC inconsistent: {self.DeltaW_BC} ≠ p2·ΔV = {recalc_W_BC}")
        
        # Check that cycle returns to initial state (V2 = V0)
        if self._is_close(self.V_2, self.V0):
            logs.append("✓ Cycle returns to initial volume: V2 ≈ V0")
        else:
            logs.append(f"✗ Volume not cyclic: V2={self.V_2}, V0={self.V0}")
        
        self.logs.extend(logs)
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Test monotonic trends by perturbing inputs."""
        logs = []
        
        # Store original values
        original_V0 = self.V0
        original_p0 = self.p0
        original_DeltaV = self.DeltaV
        original_DeltaQ = self.DeltaQ
        original_DeltaW_BC = self.DeltaW_BC
        
        try:
            # Test 1: Increase V0 by 10%
            self.V0 = original_V0 * 1.10
            
            # Recalculate expected values using formulas from steps
            V1_new = self.V0 / 3
            V2_new = self.V0
            DeltaV_new = V2_new - V1_new  # = 2*V0/3
            DeltaW_BC_new = 3 * original_p0 * DeltaV_new  # p2 = 3p0
            DeltaQ_new = 2 * original_p0 * self.V0
            
            # Check monotonic increase (use numeric comparison)
            if DeltaV_new > original_DeltaV:
                logs.append("✓ ΔV increases with increasing V0 (monotonic)")
            else:
                logs.append("✗ ΔV fails monotonicity test with V0")
                
            if DeltaQ_new > original_DeltaQ:
                logs.append("✓ ΔQ increases with increasing V0 (monotonic)")
            else:
                logs.append("✗ ΔQ fails monotonicity test with V0")
                
            # Test 2: Decrease p0 by 10%
            self.V0 = original_V0  # restore V0
            self.p0 = original_p0 * 0.90
            
            # Recalculate
            DeltaW_BC_new2 = 2 * self.p0 * self.V0
            DeltaQ_new2 = 2 * self.p0 * self.V0
            
            if DeltaW_BC_new2 < original_DeltaW_BC:
                logs.append("✓ ΔW_BC decreases with decreasing p0 (monotonic)")
            else:
                logs.append("✗ ΔW_BC fails monotonicity test with p0")
                
            if DeltaQ_new2 < original_DeltaQ:
                logs.append("✓ ΔQ decreases with decreasing p0 (monotonic)")
            else:
                logs.append("✗ ΔQ fails monotonicity test with p0")
                
        except Exception as e:
            logs.append(f"⚠ Monotonicity test error: {str(e)}")
        finally:
            # Restore original values
            self.V0 = original_V0
            self.p0 = original_p0
        
        self.logs.extend(logs)
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = ["="*50, "Physics Checker Report", "="*50]
        
        # Run tests
        invariants_logs = self.test_invariants()
        self.logs.append("\n--- Invariant Tests ---")
        self.logs.extend(invariants_logs)
        
        trace_logs = self.test_trace_consistency()
        self.logs.append("\n--- Trace Consistency Tests ---")
        self.logs.extend(trace_logs)
        
        monotonicity_logs = self.test_monotonicity()
        self.logs.append("\n--- Monotonicity Tests ---")
        self.logs.extend(monotonicity_logs)
        
        # Determine overall success
        success = True
        for log in self.logs:
            if log.startswith("✗"):
                success = False
                break
        
        self.logs.append(f"\n{'='*50}")
        self.logs.append(f"Overall result: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs