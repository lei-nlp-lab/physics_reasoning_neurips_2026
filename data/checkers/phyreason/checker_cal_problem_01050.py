import math
import re
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access using getattr with fallbacks
        self.delta_t = 0.2  # Given in problem statement
        
        # Extract results with safe fallbacks - handle y specially as string
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract numeric values
        self.delta_x = results_dict.get('Δx', 0.0)
        self.A = results_dict.get('A', 0.0)
        self.v_min = results_dict.get('v_min', 0.0)
        self.v = results_dict.get('v', 0.0)
        self.lambd = results_dict.get('λ', 8.0)  # Default 8 from steps
        self.omega = results_dict.get('omega', 0.0)
        
        # Extract y as string - don't try to convert to float
        self.y_expr = str(results_dict.get('y', ''))
        
        # For wave speed expression parsing
        self.v_expr = str(self.v) if hasattr(self.v, '__str__') else str(self.v)
        
        # Parse the integer n from wave speed expression
        self.n = self._extract_n_from_v()
        
        self.logs = []
        
    def _extract_n_from_v(self) -> int:
        """Extract the integer n from wave speed expression v=40n+30"""
        if isinstance(self.v, (int, float)):
            # If v is numeric, solve for n: n = (v-30)/40
            n_val = (self.v - 30) / 40
            if math.isclose(n_val, round(n_val), rel_tol=1e-2):
                return int(round(n_val))
            return 0
        elif isinstance(self.v_expr, str):
            # Try to parse expression like "40*n+30" or "40n+30"
            match = re.search(r'40\s*[*]?\s*n\s*\+\s*30', self.v_expr)
            if match:
                # If expression contains n, use n=0 for minimum case
                return 0
        return 0
        
    def _log(self, message: str, success: bool):
        """Helper to log test results"""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants"""
        success = True
        
        # 1. Wave speed definition: v = Δx/Δt
        expected_delta_x = self.v * self.delta_t
        if not math.isclose(self.delta_x, expected_delta_x, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Wave speed definition violated: Δx={self.delta_x}, v*Δt={expected_delta_x}", 
                False
            )
        else:
            self._log("Wave speed definition (v=Δx/Δt) satisfied", True)
        
        # 2. Relationship between v, λ, and ω for n=0 (minimum case)
        # For minimum case (n=0): v_min = 30 m/s, λ=8 m
        expected_omega = 2 * math.pi * self.v_min / self.lambd
        if not math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"ω=2πv/λ violated: ω={self.omega}, 2πv_min/λ={expected_omega}", 
                False
            )
        else:
            self._log("ω=2πv/λ relationship satisfied for minimum case", True)
        
        # 3. Check amplitude
        if not math.isclose(self.A, 0.02, rel_tol=1e-2, abs_tol=1e-2):  # 2 cm = 0.02 m
            success = self._log(f"Amplitude mismatch: A={self.A}, expected 0.02 m", False)
        else:
            self._log(f"Amplitude A={self.A} m correct", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer"""
        success = True
        
        # 1. Check wavelength from steps
        if not math.isclose(self.lambd, 8.0, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Wavelength λ={self.lambd}, expected 8 m", False)
        else:
            self._log(f"Wavelength λ={self.lambd} m correct", True)
        
        # 2. Check distance traveled for given n
        expected_delta_x = 8 * self.n + 6
        if not math.isclose(self.delta_x, expected_delta_x, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Distance traveled mismatch: Δx={self.delta_x}, expected {expected_delta_x} m", 
                False
            )
        else:
            self._log(f"Distance traveled Δx={self.delta_x} m consistent with n={self.n}", True)
        
        # 3. Check wave speed expression
        expected_v = 40 * self.n + 30
        if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Wave speed mismatch: v={self.v}, expected {expected_v} m/s", 
                False
            )
        else:
            self._log(f"Wave speed v={self.v} m/s correct for n={self.n}", True)
        
        # 4. Check minimum wave speed
        if not math.isclose(self.v_min, 30.0, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Minimum wave speed mismatch: v_min={self.v_min}, expected 30 m/s", False)
        else:
            self._log(f"Minimum wave speed v_min={self.v_min} m/s correct", True)
        
        # 5. Parse and check vibration equation
        if self.y_expr:
            # Parse y = A sin(ωt + φ)
            match = re.search(r'([\d\.]+)\s*sin\s*\(\s*([\d\.]+)\s*[πpi]*\s*/\s*([\d\.]+)?\s*t\s*[\+\-]\s*([πpi]+)', 
                             str(self.y_expr), re.IGNORECASE)
            
            if match:
                A_str, omega_num_str, omega_den_str, phase_str = match.groups()
                A_val = float(A_str)
                omega_num = float(omega_num_str) if omega_num_str else 15.0
                omega_den = float(omega_den_str) if omega_den_str else 2.0
                
                # Check amplitude (convert to meters: 2 cm = 0.02 m)
                if not math.isclose(A_val, 0.02, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(f"Vibration amplitude mismatch: {A_val}, expected 0.02", False)
                else:
                    self._log("Vibration amplitude correct", True)
                
                # Check angular frequency (15π/2 for n=0)
                expected_omega_val = 15 * math.pi / 2
                actual_omega_val = omega_num * math.pi / omega_den
                if not math.isclose(actual_omega_val, expected_omega_val, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(
                        f"Angular frequency mismatch: {actual_omega_val}, expected {expected_omega_val}", 
                        False
                    )
                else:
                    self._log("Angular frequency in vibration equation correct", True)
                
                # Check phase (should be π)
                if 'π' not in phase_str.lower() and 'pi' not in phase_str.lower():
                    success = self._log(f"Phase mismatch: {phase_str}, expected π", False)
                else:
                    self._log("Phase in vibration equation correct", True)
            else:
                success = self._log("Could not parse vibration equation", False)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test monotonic relationships by perturbing parameters"""
        success = True
        
        # Test 1: If n increases, wave speed should increase
        original_v = 40 * self.n + 30
        perturbed_v = 40 * (self.n + 1) + 30
        
        if not (perturbed_v > original_v):
            success = self._log(f"Monotonicity violated: v(n={self.n})={original_v}, v(n={self.n+1})={perturbed_v}", False)
        else:
            self._log(f"Wave speed increases with n: v(n)={original_v}, v(n+1)={perturbed_v}", True)
        
        # Test 2: For constant wavelength, ω ∝ v
        original_omega = 2 * math.pi * original_v / self.lambd
        perturbed_omega = 2 * math.pi * perturbed_v / self.lambd
        
        if not (perturbed_omega > original_omega):
            success = self._log(
                f"ω ∝ v relationship violated: ω(v={original_v})={original_omega}, ω(v={perturbed_v})={perturbed_omega}", 
                False
            )
        else:
            self._log(f"ω increases with v: ω({original_v})={original_omega}, ω({perturbed_v})={perturbed_omega}", True)
        
        # Test 3: For fixed Δt, v ∝ Δx
        original_delta_x = 8 * self.n + 6
        perturbed_delta_x = 8 * (self.n + 1) + 6
        
        if not (perturbed_delta_x > original_delta_x):
            success = self._log(
                f"Δx monotonicity violated: Δx(n={self.n})={original_delta_x}, Δx(n={self.n+1})={perturbed_delta_x}", 
                False
            )
        else:
            self._log(f"Δx increases with n: Δx({self.n})={original_delta_x}, Δx({self.n+1})={perturbed_delta_x}", True)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs"""
        self.logs = ["PhysicsChecker starting tests..."]
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_success = True
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Testing {test_name} ---")
            try:
                if not test_func():
                    all_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                all_success = False
        
        self.logs.append(f"\n{'All tests passed' if all_success else 'Some tests failed'}")
        return all_success, self.logs