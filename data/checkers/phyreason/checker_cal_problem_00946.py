import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely
        self.n_turns = getattr(solution_instance, 'n_turns', 100)
        self.area = getattr(solution_instance, 'area', 0.5)
        self.B = getattr(solution_instance, 'B', math.sqrt(2)/10)
        self.omega = getattr(solution_instance, 'omega', 10.0)
        self.lamp_voltage = 10.0
        self.lamp_power = 10.0
        self.fuse_current = 1.5
        
        # Extract results safely from KB mapping
        self.frac_ratio = self.results.get('fracn_1n_2', None)  # Expected as fraction or string
        self.E = self.results.get('E', None)
        self.U1 = self.results.get('U_1', None)
        self.E_m = self.results.get('E_m', None)
        self.P1m = self.results.get('P_1m', None)
        self.P2m = self.results.get('P_2m', None)
        self.n_lamps = self.results.get('n', None)
        self.U2 = self.results.get('U_2', None)
        
        # Derived constants
        self.lamp_resistance = self.lamp_voltage**2 / self.lamp_power  # 10 Ω
        self.sqrt2 = math.sqrt(2)
        
        # Logs for test results
        self.logs = []
        
    def _log(self, message: str, success: Optional[bool] = None):
        """Helper to log test messages"""
        prefix = "[PASS]" if success is True else "[FAIL]" if success is False else "[INFO]"
        self.logs.append(f"{prefix} {message}")
        
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy comparison for physics values"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Test Faraday's Law, transformer principles, and power conservation"""
        success = True
        
        # 1. Faraday's Law: Maximum EMF
        expected_E_m = self.n_turns * self.B * self.area * self.omega
        if self.E_m is not None:
            if not self._isclose(self.E_m, expected_E_m):
                success = False
                self._log(f"Faraday's Law failed: E_m = {self.E_m} != {expected_E_m}", False)
            else:
                self._log(f"Faraday's Law (max EMF): {self.E_m} ≈ {expected_E_m}", True)
        
        # 2. Effective EMF
        expected_E = expected_E_m / self.sqrt2
        if self.E is not None:
            if not self._isclose(self.E, expected_E):
                success = False
                self._log(f"Effective EMF failed: E = {self.E} != {expected_E}", False)
            else:
                self._log(f"Effective EMF: {self.E} ≈ {expected_E}", True)
        
        # 3. Transformer voltage ratio
        if self.U1 is not None and self.U2 is not None:
            expected_U2 = 10.0  # Given in problem
            if not self._isclose(self.U2, expected_U2):
                success = False
                self._log(f"U2 mismatch: {self.U2} != {expected_U2}", False)
            else:
                self._log(f"U2 matches given: {self.U2} ≈ {expected_U2}", True)
            
            # Check U1 = E
            if self.E is not None:
                if not self._isclose(self.U1, self.E):
                    success = False
                    self._log(f"U1 = E failed: {self.U1} != {self.E}", False)
                else:
                    self._log(f"U1 = E: {self.U1} ≈ {self.E}", True)
            
            # Check turns ratio
            expected_ratio = self.U1 / self.U2
            if self.frac_ratio is not None:
                # Parse ratio (could be string like "5:1" or float)
                if isinstance(self.frac_ratio, str):
                    if ":" in self.frac_ratio:
                        num, den = map(float, self.frac_ratio.split(":"))
                        ratio_value = num / den
                    else:
                        ratio_value = float(self.frac_ratio)
                else:
                    ratio_value = float(self.frac_ratio)
                
                if not self._isclose(ratio_value, expected_ratio):
                    success = False
                    self._log(f"Turns ratio failed: {ratio_value} != {expected_ratio}", False)
                else:
                    self._log(f"Turns ratio: {ratio_value} ≈ {expected_ratio}", True)
        
        # 4. Power conservation with fuse limit
        # Maximum primary current is fuse current (1.5A effective)
        expected_P1m = self.fuse_current * self.U1 if self.U1 else None
        if self.P1m is not None and expected_P1m is not None:
            if not self._isclose(self.P1m, expected_P1m):
                success = False
                self._log(f"Primary power failed: P1m = {self.P1m} != {expected_P1m}", False)
            else:
                self._log(f"Primary power (fuse limit): {self.P1m} ≈ {expected_P1m}", True)
        
        # 5. Power equality P1m = P2m (ideal transformer)
        if self.P1m is not None and self.P2m is not None:
            if not self._isclose(self.P1m, self.P2m):
                success = False
                self._log(f"Power conservation failed: P1m = {self.P1m} != P2m = {self.P2m}", False)
            else:
                self._log(f"Power conservation: P1m = P2m = {self.P1m}", True)
        
        # 6. Number of lamps calculation
        if self.P2m is not None:
            expected_n_lamps_float = self.P2m / self.lamp_power
            expected_n_lamps = int(expected_n_lamps_float)  # Integer number of lamps
            
            if self.n_lamps is not None:
                if not self._isclose(self.n_lamps, expected_n_lamps, abs_tol=0.5):
                    success = False
                    self._log(f"Number of lamps failed: n = {self.n_lamps} != {expected_n_lamps}", False)
                else:
                    self._log(f"Number of lamps: n = {self.n_lamps} ≈ {expected_n_lamps}", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results from steps analysis"""
        success = True
        
        # Check step_1 values
        expected_E_m = 50 * self.sqrt2  # 50√2 V
        expected_E = 50.0  # 50 V
        
        if self.E_m is not None:
            if not self._isclose(self.E_m, expected_E_m):
                success = False
                self._log(f"Step 1 E_m mismatch: {self.E_m} != {expected_E_m}", False)
            else:
                self._log(f"Step 1 E_m: {self.E_m} ≈ {expected_E_m}", True)
        
        if self.E is not None:
            if not self._isclose(self.E, expected_E):
                success = False
                self._log(f"Step 1 E mismatch: {self.E} != {expected_E}", False)
            else:
                self._log(f"Step 1 E: {self.E} ≈ {expected_E}", True)
        
        # Check step_2 values
        if self.U1 is not None and self.E is not None:
            if not self._isclose(self.U1, self.E):
                success = False
                self._log(f"Step 2 U1 = E failed: {self.U1} != {self.E}", False)
            else:
                self._log(f"Step 2 U1 = E: {self.U1} ≈ {self.E}", True)
        
        expected_U2 = 10.0
        if self.U2 is not None:
            if not self._isclose(self.U2, expected_U2):
                success = False
                self._log(f"Step 2 U2 mismatch: {self.U2} != {expected_U2}", False)
            else:
                self._log(f"Step 2 U2: {self.U2} ≈ {expected_U2}", True)
        
        # Check step_3 ratio
        if self.U1 is not None and self.U2 is not None:
            expected_ratio_val = self.U1 / self.U2  # Should be 5.0
            if self.frac_ratio is not None:
                if isinstance(self.frac_ratio, str):
                    if self.frac_ratio == "5:1":
                        ratio_val = 5.0
                    else:
                        try:
                            ratio_val = float(self.frac_ratio)
                        except:
                            ratio_val = None
                else:
                    ratio_val = float(self.frac_ratio)
                
                if ratio_val is not None and not self._isclose(ratio_val, expected_ratio_val):
                    success = False
                    self._log(f"Step 3 ratio mismatch: {ratio_val} != {expected_ratio_val}", False)
                else:
                    self._log(f"Step 3 ratio: {self.frac_ratio} ≈ 5:1", True)
        
        # Check step_4 & step_5 power calculations
        if self.U1 is not None:
            expected_P1m = self.fuse_current * self.U1  # 1.5 * 50 = 75 W
            
            if self.P1m is not None:
                if not self._isclose(self.P1m, expected_P1m):
                    success = False
                    self._log(f"Step 4 P1m mismatch: {self.P1m} != {expected_P1m}", False)
                else:
                    self._log(f"Step 4 P1m: {self.P1m} ≈ {expected_P1m}", True)
            
            if self.P2m is not None and self.P1m is not None:
                if not self._isclose(self.P2m, self.P1m):
                    success = False
                    self._log(f"Step 5 P2m mismatch: {self.P2m} != {self.P1m}", False)
                else:
                    self._log(f"Step 5 P2m = P1m: {self.P2m} ≈ {self.P1m}", True)
            
            if self.P2m is not None:
                expected_n = int(self.P2m / self.lamp_power)  # Should be 7
                if self.n_lamps is not None:
                    if not self._isclose(self.n_lamps, expected_n, abs_tol=0.5):
                        success = False
                        self._log(f"Step 5 n mismatch: {self.n_lamps} != {expected_n}", False)
                    else:
                        self._log(f"Step 5 n: {self.n_lamps} ≈ {expected_n}", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Corrected monotonicity test: n_lamps ∝ ω when fuse current is fixed.
        Actually, with fixed fuse current I_fuse, P1m = I_fuse * U1, and U1 = E ∝ ω.
        So P1m ∝ ω, and thus n ∝ ω.
        """
        success = True
        
        # Store original values
        original_omega = self.omega
        original_n_lamps = self.n_lamps
        
        # Perturb omega by +10%
        perturbed_omega = original_omega * 1.1
        
        # Recalculate using correct physics
        # New E_m and E (proportional to ω)
        new_E_m = self.n_turns * self.B * self.area * perturbed_omega
        new_E = new_E_m / self.sqrt2
        
        # New U1 = new_E
        new_U1 = new_E
        
        # New P1m with same fuse current
        new_P1m = self.fuse_current * new_U1
        
        # New number of lamps
        new_n_lamps_float = new_P1m / self.lamp_power
        new_n_lamps = int(new_n_lamps_float)
        
        # Check monotonic trend: increasing ω should INCREASE n_lamps (n ∝ ω)
        if original_n_lamps is not None:
            if perturbed_omega > original_omega:
                expected_trend = new_n_lamps > original_n_lamps
            else:
                expected_trend = new_n_lamps < original_n_lamps
            
            if not expected_trend:
                success = False
                self._log(f"Monotonicity failed: ω↑ ({original_omega}→{perturbed_omega}) but n ({original_n_lamps}→{new_n_lamps}) not ↑", False)
            else:
                self._log(f"Monotonicity: ω↑ ({original_omega}→{perturbed_omega}) ⇒ n↑ ({original_n_lamps}→{new_n_lamps})", True)
        
        # Restore original values
        self.omega = original_omega
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs"""
        self.logs = []
        overall_success = True
        
        self._log("Starting Physics Auditor...")
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} Test ---")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"{test_name} test failed!", False)
                else:
                    self._log(f"{test_name} test passed.", True)
            except Exception as e:
                overall_success = False
                self._log(f"{test_name} test raised exception: {str(e)}", False)
        
        # Final summary
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs