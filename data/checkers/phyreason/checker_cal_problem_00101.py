import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize with safe attribute access.
        solution_instance should have:
            - results: Dict with keys like 'f', 'F_elastic', 'v'', 'F_net', 'r0', 'r1', 'x_1'', 'v_B', 'W', 'W_all'
            - inputs: Dict with keys 'm', 'q', 'E', 'k', 'v0', 'v1'
        """
        # Safe extraction of inputs with defaults
        self.inputs = {}
        input_keys = ['m', 'q', 'E', 'k', 'v0', 'v1']
        for key in input_keys:
            self.inputs[key] = getattr(solution_instance, key, None)
            if self.inputs[key] is None:
                self.inputs[key] = getattr(solution_instance, 'inputs', {}).get(key, None)
        
        # Safe extraction of results with defaults
        self.results = {}
        result_keys = ['f', 'F_elastic', 'v\'', 'F_net', 'r0', 'r1', 'x_1\'', 'v_B', 'W', 'W_all']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, None)
            if self.results[key] is None:
                self.results[key] = getattr(solution_instance, 'results', {}).get(key, None)
        
        # Create symbols for symbolic calculations
        self.m, self.q, self.E, self.k, self.v0, self.v1 = sp.symbols('m q E k v0 v1', positive=True)
        self.r0_sym, self.r1_sym, self.W_sym, self.W_all_sym, self.v_prime_sym = sp.symbols('r0 r1 W W_all v_prime')
        
        # For numerical checks, convert to floats if possible
        self.num_inputs = {}
        for key, val in self.inputs.items():
            try:
                self.num_inputs[key] = float(val) if val is not None else None
            except (TypeError, ValueError):
                self.num_inputs[key] = None
        
        self.num_results = {}
        for key, val in self.results.items():
            try:
                self.num_results[key] = float(val) if val is not None else None
            except (TypeError, ValueError):
                self.num_results[key] = None
        
        self.logs = []
    
    def _log(self, message: str, success: bool = True):
        """Log a message with success status."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Safe numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Test physical invariants: force balance at critical points."""
        success = True
        
        try:
            # Test 1: Force balance at t1 (minimum velocity)
            # According to step 2: F_net = 0 at t1
            # Force balance: k*q^2/r0^2 + q*E = f, with f = 2qE (from step 3)
            if self.num_results.get('F_net') is not None:
                if not self._is_close(self.num_results['F_net'], 0.0):
                    self._log(f"F_net at t1 should be 0, got {self.num_results['F_net']}", False)
                    success = False
                else:
                    self._log("F_net at t1 = 0 (correct)")
            
            # Check the force balance equation symbolically
            if self.num_results.get('r0') is not None and all(v is not None for v in [self.num_inputs['k'], self.num_inputs['q'], self.num_inputs['E']]):
                lhs = (self.num_inputs['k'] * self.num_inputs['q']**2) / (self.num_results['r0']**2) + self.num_inputs['q'] * self.num_inputs['E']
                rhs = 2 * self.num_inputs['q'] * self.num_inputs['E']
                if not self._is_close(lhs, rhs):
                    self._log(f"Force balance at t1 fails: k*q²/r0² + qE = {lhs}, 2qE = {rhs}", False)
                    success = False
                else:
                    self._log("Force balance at t1 satisfied: k*q²/r0² + qE = 2qE")
            
            # Test 2: Force balance at t2 (maximum velocity)
            # According to step 5: F_net = 0 at t2
            # Force balance: F_elastic + f = q*E + k*q^2/r1^2
            # With F_elastic = 3qE (step 6) and f = 2qE (step 6)
            if self.num_results.get('F_net') is not None:  # Same key, different time
                # Note: We can't distinguish between t1 and t2 F_net from the data structure
                # So we'll check the specific force balance instead
                pass
            
            if (self.num_results.get('r1') is not None and 
                self.num_results.get('F_elastic') is not None and
                all(v is not None for v in [self.num_inputs['k'], self.num_inputs['q'], self.num_inputs['E']])):
                
                lhs = self.num_results['F_elastic'] + 2 * self.num_inputs['q'] * self.num_inputs['E']
                rhs = self.num_inputs['q'] * self.num_inputs['E'] + (self.num_inputs['k'] * self.num_inputs['q']**2) / (self.num_results['r1']**2)
                
                if not self._is_close(lhs, rhs):
                    self._log(f"Force balance at t2 fails: F_elastic + 2qE = {lhs}, qE + k*q²/r1² = {rhs}", False)
                    success = False
                else:
                    self._log("Force balance at t2 satisfied: F_elastic + 2qE = qE + k*q²/r1²")
            
            # Test 3: Check that v1 is indeed the minimum velocity (given in problem)
            # We can't verify this from static data, but we can check consistency
            # with work-energy theorem from step 1
            if (self.num_results.get('W') is not None and
                all(v is not None for v in [self.num_inputs['m'], self.num_inputs['v0'], self.num_inputs['v1']])):
                
                delta_KE = 0.5 * self.num_inputs['m'] * (self.num_inputs['v1']**2 - self.num_inputs['v0']**2)
                if not self._is_close(self.num_results['W'], delta_KE):
                    self._log(f"Work-Energy theorem mismatch: W = {self.num_results['W']}, ΔKE = {delta_KE}", False)
                    success = False
                else:
                    self._log("Work-Energy theorem satisfied for 0~t1")
            
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate and final results."""
        success = True
        
        try:
            # Test 1: r0 = sqrt(k*q/E) from step 4
            if (self.num_results.get('r0') is not None and
                all(v is not None for v in [self.num_inputs['k'], self.num_inputs['q'], self.num_inputs['E']])):
                
                expected_r0 = math.sqrt(self.num_inputs['k'] * self.num_inputs['q'] / self.num_inputs['E'])
                if not self._is_close(self.num_results['r0'], expected_r0):
                    self._log(f"r0 mismatch: got {self.num_results['r0']}, expected sqrt(k*q/E) = {expected_r0}", False)
                    success = False
                else:
                    self._log("r0 = sqrt(k*q/E) satisfied")
            
            # Test 2: r1 = 0.5 * sqrt(k*q/E) from step 7
            if (self.num_results.get('r1') is not None and
                all(v is not None for v in [self.num_inputs['k'], self.num_inputs['q'], self.num_inputs['E']])):
                
                expected_r1 = 0.5 * math.sqrt(self.num_inputs['k'] * self.num_inputs['q'] / self.num_inputs['E'])
                if not self._is_close(self.num_results['r1'], expected_r1):
                    self._log(f"r1 mismatch: got {self.num_results['r1']}, expected 0.5*sqrt(k*q/E) = {expected_r1}", False)
                    success = False
                else:
                    self._log("r1 = 0.5*sqrt(k*q/E) satisfied")
            
            # Test 3: W_all = q*sqrt(k*q*E)/2 from step 8
            if (self.num_results.get('W_all') is not None and
                all(v is not None for v in [self.num_inputs['k'], self.num_inputs['q'], self.num_inputs['E']])):
                
                expected_W_all = self.num_inputs['q'] * math.sqrt(self.num_inputs['k'] * self.num_inputs['q'] * self.num_inputs['E']) / 2
                if not self._is_close(self.num_results['W_all'], expected_W_all):
                    self._log(f"W_all mismatch: got {self.num_results['W_all']}, expected q*sqrt(k*q*E)/2 = {expected_W_all}", False)
                    success = False
                else:
                    self._log("W_all = q*sqrt(k*q*E)/2 satisfied")
            
            # Test 4: Also check W_all = q*E*(r0 - r1) from step 8
            if (self.num_results.get('W_all') is not None and
                self.num_results.get('r0') is not None and
                self.num_results.get('r1') is not None and
                all(v is not None for v in [self.num_inputs['q'], self.num_inputs['E']])):
                
                expected_W_all_alt = self.num_inputs['q'] * self.num_inputs['E'] * (self.num_results['r0'] - self.num_results['r1'])
                if not self._is_close(self.num_results['W_all'], expected_W_all_alt):
                    self._log(f"W_all alternative form mismatch: got {self.num_results['W_all']}, expected q*E*(r0-r1) = {expected_W_all_alt}", False)
                    success = False
                else:
                    self._log("W_all = q*E*(r0-r1) satisfied")
            
            # Test 5: v' = (1+sqrt(3))*v1 from step 16
            v_prime_key = "v'"
            if (self.num_results.get(v_prime_key) is not None and
                self.num_inputs.get('v1') is not None):
                
                expected_v_prime = (1 + math.sqrt(3)) * self.num_inputs['v1']
                if not self._is_close(self.num_results[v_prime_key], expected_v_prime):
                    self._log(f"v' mismatch: got {self.num_results[v_prime_key]}, expected (1+sqrt(3))*v1 = {expected_v_prime}", False)
                    success = False
                else:
                    self._log("v' = (1+sqrt(3))*v1 satisfied")
            
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}", False)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input and check if the outputs change in the expected direction.
        We'll perturb q and check r0, r1, and W_all.
        """
        success = True
        
        try:
            # We need numerical values for this test
            if any(v is None for v in [
                self.num_inputs.get('k'), self.num_inputs.get('q'), 
                self.num_inputs.get('E'), self.num_inputs.get('v1'),
                self.num_results.get('r0'), self.num_results.get('r1'),
                self.num_results.get('W_all')
            ]):
                self._log("Skipping monotonicity test due to missing numerical values")
                return True  # Skip, not a failure
            
            original_q = self.num_inputs['q']
            perturbed_q = original_q * 1.01  # +1% perturbation
            
            # Calculate expected perturbed values using formulas
            expected_r0_pert = math.sqrt(self.num_inputs['k'] * perturbed_q / self.num_inputs['E'])
            expected_r1_pert = 0.5 * math.sqrt(self.num_inputs['k'] * perturbed_q / self.num_inputs['E'])
            expected_W_all_pert = perturbed_q * math.sqrt(self.num_inputs['k'] * perturbed_q * self.num_inputs['E']) / 2
            
            # Check monotonicity: all should increase with q (since q>0)
            if not (expected_r0_pert > self.num_results['r0']):
                self._log(f"r0 not monotonic with q: {expected_r0_pert} <= {self.num_results['r0']}", False)
                success = False
            else:
                self._log("r0 increases with q (monotonic)")
            
            if not (expected_r1_pert > self.num_results['r1']):
                self._log(f"r1 not monotonic with q: {expected_r1_pert} <= {self.num_results['r1']}", False)
                success = False
            else:
                self._log("r1 increases with q (monotonic)")
            
            if not (expected_W_all_pert > self.num_results['W_all']):
                self._log(f"W_all not monotonic with q: {expected_W_all_pert} <= {self.num_results['W_all']}", False)
                success = False
            else:
                self._log("W_all increases with q (monotonic)")
            
            # Check proportionality relationships
            # r0 ∝ sqrt(q), so (r0_pert/r0_orig) should be sqrt(1.01) ≈ 1.004987...
            r0_ratio = expected_r0_pert / self.num_results['r0']
            if not self._is_close(r0_ratio, math.sqrt(1.01), rel_tol=1e-3):
                self._log(f"r0 scaling with q not proportional to sqrt(q): ratio = {r0_ratio}, expected {math.sqrt(1.01)}", False)
                success = False
            
            # W_all ∝ q^(3/2), so ratio should be (1.01)^(3/2) ≈ 1.01494...
            W_all_ratio = expected_W_all_pert / self.num_results['W_all']
            expected_ratio = 1.01 ** 1.5
            if not self._is_close(W_all_ratio, expected_ratio, rel_tol=1e-3):
                self._log(f"W_all scaling with q not proportional to q^(3/2): ratio = {W_all_ratio}, expected {expected_ratio}", False)
                success = False
            
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}", False)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        
        self._log("Starting Physics Audit")
        
        # Run tests
        test_results = []
        test_results.append(("Invariants", self.test_invariants()))
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        test_results.append(("Monotonicity", self.test_monotonicity()))
        
        # Count successes
        success_count = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        
        # Log summary
        for test_name, result in test_results:
            self._log(f"{test_name}: {'PASS' if result else 'FAIL'}")
        
        overall_success = success_count == total_tests
        self._log(f"Overall: {success_count}/{total_tests} tests passed", overall_success)
        
        return overall_success, self.logs