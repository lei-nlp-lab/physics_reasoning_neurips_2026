import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs from results dict (they should be stored there)
        self.m = self.results.get('m', 1.0)
        self.q = self.results.get('q', 1.0)
        self.B = self.results.get('B', 1.0)
        self.v = self.results.get('v', 2.0)
        self.d = self.results.get('d', 1.0)
        self.n = self.results.get('n', 2)
        
        # First pass quantities
        self.W_1 = self.results.get('W_1', None)
        self.v_1 = self.results.get('v_1', None)
        
        # General n-th pass quantities
        self.d_n = self.results.get('d_n', None)
        self.E_n = self.results.get('E_n', None)
        self.v_n = self.results.get('v_n', None)
        self.v_n_plus_1 = self.results.get('v_n+1', None)
        self.t_n = self.results.get('t_n', None)
        
        # Time-related quantities
        self.t_e = self.results.get('t_e', None)
        self.t_c = self.results.get('t_c', None)
        self.T = self.results.get('T', None)
        self.t_total = self.results.get('t', None)
        
        # Magnetic motion
        self.r = self.results.get('r', None)
        
        self.logs = []
        
    def _log(self, msg: str, success: bool):
        """Helper to log test results"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {msg}")
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison"""
        try:
            if a is None or b is None:
                return False
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
            
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test conservation laws and physical invariants"""
        sub_tests = []
        
        # 1. Check Lorentz force as centripetal force
        if self.r is not None:
            expected_r = self.m * self.v / (self.q * self.B)
            if self._is_close(self.r, expected_r):
                self._log(f"Centripetal force invariant: r = {self.r:.3e}, expected {expected_r:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Centripetal force invariant FAILED: r = {self.r:.3e}, expected {expected_r:.3e}", False)
                sub_tests.append(False)
                
        # 2. Check work-energy theorem for first pass
        if self.W_1 is not None and self.v_1 is not None:
            expected_W1 = 0.5 * self.m * (self.v_1**2 - self.v**2)
            if self._is_close(self.W_1, expected_W1):
                self._log(f"Work-energy theorem (1st pass): W_1 = {self.W_1:.3e}, expected {expected_W1:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Work-energy theorem (1st pass) FAILED: W_1 = {self.W_1:.3e}, expected {expected_W1:.3e}", False)
                sub_tests.append(False)
                
        # 3. Check impulse-momentum theorem for n-th pass
        if all(x is not None for x in [self.E_n, self.t_n, self.v_n, self.v_n_plus_1]):
            lhs = self.q * self.E_n * self.t_n
            rhs = self.m * (self.v_n_plus_1 - self.v_n)
            if self._is_close(lhs, rhs):
                self._log(f"Impulse-momentum theorem (n-th pass): qE_n t_n = {lhs:.3e}, mΔv = {rhs:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Impulse-momentum theorem FAILED: qE_n t_n = {lhs:.3e}, mΔv = {rhs:.3e}", False)
                sub_tests.append(False)
                
        # 4. Check magnetic period
        if self.T is not None:
            expected_T = 2 * math.pi * self.m / (self.q * self.B)
            if self._is_close(self.T, expected_T):
                self._log(f"Magnetic period: T = {self.T:.3e}, expected {expected_T:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Magnetic period FAILED: T = {self.T:.3e}, expected {expected_T:.3e}", False)
                sub_tests.append(False)
                
        return all(sub_tests), self.logs[-len(sub_tests):]
        
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate results"""
        sub_tests = []
        
        # 1. Check velocity progression pattern
        if self.v_1 is not None:
            expected_v1 = self.v * self.v  # v * v = v^2
            if self._is_close(self.v_1, expected_v1):
                self._log(f"Velocity progression: v_1 = {self.v_1:.3e}, expected v^2 = {expected_v1:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Velocity progression FAILED: v_1 = {self.v_1:.3e}, expected v^2 = {expected_v1:.3e}", False)
                sub_tests.append(False)
                
        # 2. Check electric field width progression
        if self.d_n is not None:
            expected_dn = self.n * self.d
            if self._is_close(self.d_n, expected_dn):
                self._log(f"Electric field width: d_n = {self.d_n:.3e}, expected n*d = {expected_dn:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Electric field width FAILED: d_n = {self.d_n:.3e}, expected n*d = {expected_dn:.3e}", False)
                sub_tests.append(False)
                
        # 3. Check work-energy for n-th pass
        if all(x is not None for x in [self.E_n, self.d_n, self.v_n, self.v_n_plus_1]):
            lhs = self.q * self.E_n * self.d_n
            rhs = 0.5 * self.m * (self.v_n_plus_1**2 - self.v_n**2)
            if self._is_close(lhs, rhs):
                self._log(f"Work-energy (n-th pass): qE_n d_n = {lhs:.3e}, ΔK = {rhs:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Work-energy (n-th pass) FAILED: qE_n d_n = {lhs:.3e}, ΔK = {rhs:.3e}", False)
                sub_tests.append(False)
                
        # 4. Check time in electric field is positive
        if self.t_e is not None:
            if self.t_e > 0:
                self._log(f"Total electric field time positive: t_e = {self.t_e:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Total electric field time invalid: t_e = {self.t_e:.3e}", False)
                sub_tests.append(False)
                
        # 5. Check total time formula
        if all(x is not None for x in [self.t_total, self.t_e, self.t_c]):
            expected_total = self.t_e + self.t_c
            if self._is_close(self.t_total, expected_total):
                self._log(f"Total time: t = {self.t_total:.3e}, t_e + t_c = {expected_total:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Total time FAILED: t = {self.t_total:.3e}, t_e + t_c = {expected_total:.3e}", False)
                sub_tests.append(False)
                
        return all(sub_tests), self.logs[-len(sub_tests):]
        
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test how results change with parameter perturbations"""
        sub_tests = []
        
        # Only run if we have all necessary values
        required = [self.m, self.q, self.B, self.v, self.d, self.W_1, self.E_n, self.t_total]
        if any(x is None for x in required):
            self._log("Monotonicity test skipped: insufficient numerical data", True)
            return True, self.logs[-1:]
            
        try:
            # Test 1: Increase mass m -> increase W_1
            perturbed_m = self.m * 1.1
            W1_orig = self.W_1
            W1_perturbed = 0.5 * perturbed_m * self.v**2 * (self.v**2 - 1)
            
            if W1_perturbed > W1_orig:
                self._log(f"Mass monotonicity: m↑ {self.m:.3e}→{perturbed_m:.3e}, W_1↑ {W1_orig:.3e}→{W1_perturbed:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Mass monotonicity FAILED: m↑ but W_1 not increased", False)
                sub_tests.append(False)
                
            # Test 2: Increase velocity v -> increase E_n
            perturbed_v = self.v * 1.1
            E_n_orig = self.E_n
            E_n_perturbed = (self.m * perturbed_v**(2*self.n) * (perturbed_v**2 - 1)) / (2 * self.q * self.n * self.d)
            
            if E_n_perturbed > E_n_orig:
                self._log(f"Velocity monotonicity: v↑ {self.v:.3e}→{perturbed_v:.3e}, E_n↑ {E_n_orig:.3e}→{E_n_perturbed:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Velocity monotonicity FAILED: v↑ but E_n not increased", False)
                sub_tests.append(False)
                
            # Test 3: Increase magnetic field B -> decrease total time (t_c component decreases)
            perturbed_B = self.B * 1.1
            t_c_orig = self.t_c
            t_c_perturbed = self.n * math.pi * self.m / (self.q * perturbed_B)
            
            if t_c_perturbed < t_c_orig:
                self._log(f"Magnetic field monotonicity: B↑ {self.B:.3e}→{perturbed_B:.3e}, t_c↓ {t_c_orig:.3e}→{t_c_perturbed:.3e}", True)
                sub_tests.append(True)
            else:
                self._log(f"Magnetic field monotonicity FAILED: B↑ but t_c not decreased", False)
                sub_tests.append(False)
                
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", False)
            sub_tests.append(False)
            
        return all(sub_tests), self.logs[-len(sub_tests):]
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs"""
        all_logs = []
        overall_success = True
        
        try:
            # Test 1: Invariants
            success1, logs1 = self.test_invariants()
            all_logs.extend(logs1)
            if not success1:
                overall_success = False
                
            # Test 2: Trace consistency
            success2, logs2 = self.test_trace_consistency()
            all_logs.extend(logs2)
            if not success2:
                overall_success = False
                
            # Test 3: Monotonicity
            success3, logs3 = self.test_monotonicity()
            all_logs.extend(logs3)
            if not success3:
                overall_success = False
                
        except Exception as e:
            error_msg = f"Test runner error: {traceback.format_exc()}"
            all_logs.append(f"ERROR: {error_msg}")
            overall_success = False
            
        summary = f"Overall: {'PASS' if overall_success else 'FAIL'} ({sum(1 for log in all_logs if 'PASS' in log)}/{len(all_logs)} sub-tests passed)"
        all_logs.append(summary)
        
        return overall_success, all_logs