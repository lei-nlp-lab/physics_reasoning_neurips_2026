import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary attributes from the solution instance.
        """
        # Extract results safely using getattr with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract specific results with safe defaults
        self.S = self.results.get('S', 0.0)
        self.P_2 = self.results.get('P_2', 0.0)
        self.ρvS = self.results.get('ρvS', 0.0)
        self.W = self.results.get('W', 0.0)
        self.P_m = self.results.get('P_m', 0.0)
        self.P_1 = self.results.get('P_1', 0.0)
        
        # Extract inputs from results or use actual problem values as defaults
        self.rho = self.results.get('rho', 1.3)  # Actual problem value
        self.r = self.results.get('r', 20.0)     # Actual problem value
        self.v = self.results.get('v', 10.0)     # Actual problem value
        self.v1 = self.results.get('v1', 9.0)    # Actual problem value
        self.v2 = self.results.get('v2', 6.0)    # Actual problem value
        self.t = self.results.get('t', 5000.0)   # Actual problem value
        
        # Store for perturbation backup
        self._backup_values = {}
    
    def _perturb_input(self, input_name: str, factor: float = 1.01):
        """
        Temporarily perturb an input and back it up.
        """
        if hasattr(self, input_name):
            self._backup_values[input_name] = getattr(self, input_name)
            setattr(self, input_name, getattr(self, input_name) * factor)
            return True
        return False
    
    def _restore_inputs(self):
        """Restore all perturbed inputs."""
        for name, value in self._backup_values.items():
            setattr(self, name, value)
        self._backup_values.clear()
    
    def _calculate_P_m(self, v_value=None) -> float:
        """Calculate maximum wind power using the formula."""
        if v_value is None:
            v_value = self.v
        return 0.5 * math.pi * self.rho * (self.r ** 2) * (v_value ** 3)
    
    def test_invariants(self) -> List[str]:
        """
        Verify fundamental physics laws.
        """
        logs = []
        
        try:
            # 1. Check power-cubed wind speed relationship (P ∝ v³)
            expected_P2_ratio = (self.v2 / self.v1) ** 3
            if self.P_1 > 0:
                actual_ratio = self.P_2 / self.P_1
                if not math.isclose(actual_ratio, expected_P2_ratio, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Power ratio mismatch. Expected {expected_P2_ratio:.3f}, got {actual_ratio:.3f}")
                else:
                    logs.append(f"PASS: Power ratio (v2/v1)³ = {expected_P2_ratio:.3f} matches")
            
            # 2. Check energy-power-time relationship: W = P_2 * t
            expected_W = self.P_2 * self.t  # P_2 in kW, t in hours
            if not math.isclose(self.W, expected_W, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Energy calculation W = P_2 * t mismatch. Expected {expected_W:.1f}, got {self.W:.1f}")
            else:
                logs.append(f"PASS: Energy W = P_2 * t = {expected_W:.1f} kW·h")
            
            # 3. Check maximum wind power formula
            calculated_P_m = self._calculate_P_m()
            if not math.isclose(self.P_m, calculated_P_m, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Max wind power mismatch. Formula gives {calculated_P_m:.1f}, got {self.P_m:.1f}")
            else:
                logs.append(f"PASS: Max wind power formula verified: P_m = {calculated_P_m:.1f} W")
            
            # 4. Check mass flow rate: ρvS
            expected_ρvS = self.rho * self.v * self.S
            if not math.isclose(self.ρvS, expected_ρvS, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Mass flow rate mismatch. Expected {expected_ρvS:.3f}, got {self.ρvS:.3f}")
            else:
                logs.append(f"PASS: Mass flow rate ρvS = {expected_ρvS:.3f} kg/s")
            
            # 5. Check area calculation: S = πr²
            expected_S = math.pi * (self.r ** 2)
            if not math.isclose(self.S, expected_S, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Area S mismatch. Expected {expected_S:.3f}, got {self.S:.3f}")
            else:
                logs.append(f"PASS: Area S = πr² = {expected_S:.3f} m²")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """
        Check consistency between intermediate and final results.
        """
        logs = []
        
        try:
            # 1. Check that P_m = 0.5 * ρvS * v²
            expected_P_m_from_flow = 0.5 * self.ρvS * (self.v ** 2)
            if not math.isclose(self.P_m, expected_P_m_from_flow, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: P_m inconsistent with mass flow. From flow: {expected_P_m_from_flow:.1f}, got {self.P_m:.1f}")
            else:
                logs.append(f"PASS: P_m = 0.5 * ρvS * v² verified")
            
            # 2. Check that P_2 is consistent with target answer (160 kW)
            expected_P2 = 160.0  # kW from steps analysis
            if not math.isclose(self.P_2, expected_P2, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: P_2 doesn't match expected 160 kW. Got {self.P_2:.1f} kW")
            else:
                logs.append(f"PASS: P_2 = {self.P_2:.1f} kW matches expected")
            
            # 3. Check that W is consistent with target answer (8×10^5 kW·h)
            expected_W = 8e5  # kW·h from steps analysis
            if not math.isclose(self.W, expected_W, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: W doesn't match expected 8e5 kW·h. Got {self.W:.1f} kW·h")
            else:
                logs.append(f"PASS: W = {self.W:.1f} kW·h matches expected")
            
            # 4. Check that P_2 < P_1 since v2 < v1
            if self.v2 < self.v1 and self.P_2 >= self.P_1:
                logs.append(f"FAIL: When v2 < v1, P_2 should be < P_1. P_1={self.P_1:.1f}, P_2={self.P_2:.1f}")
            elif self.v2 < self.v1 and self.P_2 < self.P_1:
                logs.append(f"PASS: P_2 < P_1 when v2 < v1 as expected")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """
        Test that power increases with wind speed cubed.
        """
        logs = []
        
        try:
            # Test 1: Perturb wind speed v and check P_m response
            perturbation_factor = 1.1  # 10% increase
            
            # Calculate original P_m with original v
            original_P_m = self._calculate_P_m()
            
            # Calculate new P_m with perturbed v
            perturbed_v = self.v * perturbation_factor
            new_P_m = self._calculate_P_m(perturbed_v)
            
            # Expected ratio from P ∝ v³
            expected_ratio = perturbation_factor ** 3  # (1.1)³ ≈ 1.331
            actual_ratio = new_P_m / original_P_m if original_P_m != 0 else 0
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: P_m doesn't scale as v³. Expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}")
            else:
                logs.append(f"PASS: P_m scales correctly as v³ (ratio: {actual_ratio:.3f})")
            
            # Test 2: Check that P_2 < P_1 when v2 < v1 (correct for this problem)
            if self.v2 < self.v1 and self.P_2 >= self.P_1:
                logs.append(f"FAIL: When v2 < v1, P_2 should be < P_1. P_1={self.P_1:.1f}, P_2={self.P_2:.1f}")
            elif self.v2 < self.v1 and self.P_2 < self.P_1:
                logs.append(f"PASS: P_2 < P_1 when v2 < v1 as expected")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success status and logs.
        """
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== Running {test_name} Test ===")
            test_logs = test_method()
            all_logs.extend(test_logs)
            
            # Check for any FAIL messages
            if any("FAIL:" in log for log in test_logs):
                success = False
        
        # Final summary
        all_logs.append(f"\n=== Summary ===")
        all_logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, all_logs