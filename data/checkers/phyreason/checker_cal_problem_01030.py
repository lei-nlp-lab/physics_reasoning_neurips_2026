import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution instance.
        Never access attributes directly; use getattr with fallbacks.
        """
        # Extract inputs safely
        inputs = getattr(solution_instance, 'inputs', {})
        self.V0 = inputs.get('V0', 1.0)  # Default values for safety
        self.T0 = inputs.get('T0', 300.0)
        self.N = inputs.get('N', 6.022e23)
        self.p = inputs.get('p', 101325.0)
        self.delta_U = inputs.get('delta_U', 0.0)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.Q = self.results.get('Q', 0.0)
        self.d = self.results.get('d', 0.0)
        self.W = self.results.get('W', 0.0)
        self.V_B = self.results.get('V_B', 0.0)
        
        self.logs = []
    
    def _log(self, msg: str):
        """Helper to append log messages."""
        self.logs.append(msg)
    
    def _is_close(self, a, b) -> bool:
        """
        Robust numerical comparison with 2% tolerance.
        """
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """
        Verify thermodynamic laws and relationships from KB.
        Returns True if all invariants hold.
        """
        try:
            all_pass = True
            
            # 1. Check isobaric process: V_B = 2*V0 (from V/T constant)
            if not self._is_close(self.V_B, 2 * self.V0):
                self._log(f"FAIL: V_B = {self.V_B} ≠ 2*V0 = {2*self.V0}")
                all_pass = False
            else:
                self._log(f"PASS: V_B = 2*V0 (isobaric process)")
            
            # 2. Check molecular distance: d = (V_B/N)^(1/3)
            d_calc = (self.V_B / self.N) ** (1/3)
            if not self._is_close(self.d, d_calc):
                self._log(f"FAIL: d = {self.d} ≠ (V_B/N)^(1/3) = {d_calc}")
                all_pass = False
            else:
                self._log(f"PASS: d = (V_B/N)^(1/3)")
            
            # 3. Check work from isobaric expansion: W = -p*(V_B - V0)
            W_calc = -self.p * (self.V_B - self.V0)
            if not self._is_close(self.W, W_calc):
                self._log(f"FAIL: W = {self.W} ≠ -p*(V_B-V0) = {W_calc}")
                all_pass = False
            else:
                self._log(f"PASS: W = -p*(V_B-V0) (isobaric work)")
            
            # 4. Check first law: Q = ΔU - W
            Q_calc = self.delta_U - self.W
            if not self._is_close(self.Q, Q_calc):
                self._log(f"FAIL: Q = {self.Q} ≠ ΔU - W = {Q_calc}")
                all_pass = False
            else:
                self._log(f"PASS: Q = ΔU - W (first law)")
            
            # 5. Check consistency between steps
            # From STEPS: d should also equal (2*V0/N)^(1/3)
            d_alt = (2 * self.V0 / self.N) ** (1/3)
            if not self._is_close(self.d, d_alt):
                self._log(f"FAIL: d = {self.d} ≠ (2*V0/N)^(1/3) = {d_alt}")
                all_pass = False
            else:
                self._log(f"PASS: d consistent with V0 and N")
            
            # 6. Check W = -p*V0 (since V_B - V0 = V0)
            if not self._is_close(self.W, -self.p * self.V0):
                self._log(f"FAIL: W = {self.W} ≠ -p*V0 = {-self.p*self.V0}")
                all_pass = False
            else:
                self._log(f"PASS: W = -p*V0 (simplified)")
            
            # 7. Check Q = ΔU + p*V0 (combined result)
            Q_combined = self.delta_U + self.p * self.V0
            if not self._is_close(self.Q, Q_combined):
                self._log(f"FAIL: Q = {self.Q} ≠ ΔU + p*V0 = {Q_combined}")
                all_pass = False
            else:
                self._log(f"PASS: Q = ΔU + p*V0 (final answer)")
            
            return all_pass
            
        except Exception as e:
            self._log(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        """
        try:
            all_pass = True
            
            # Expected formulas from physics
            expected_V_B = 2 * self.V0
            expected_d = (2 * self.V0 / self.N) ** (1/3)
            expected_W = -self.p * self.V0
            expected_Q = self.delta_U + self.p * self.V0
            
            # Check V_B
            if not self._is_close(self.V_B, expected_V_B):
                self._log(f"TRACE FAIL: V_B = {self.V_B} ≠ expected {expected_V_B}")
                all_pass = False
            
            # Check d
            if not self._is_close(self.d, expected_d):
                self._log(f"TRACE FAIL: d = {self.d} ≠ expected {expected_d}")
                all_pass = False
            
            # Check W
            if not self._is_close(self.W, expected_W):
                self._log(f"TRACE FAIL: W = {self.W} ≠ expected {expected_W}")
                all_pass = False
            
            # Check Q
            if not self._is_close(self.Q, expected_Q):
                self._log(f"TRACE FAIL: Q = {self.Q} ≠ expected {expected_Q}")
                all_pass = False
            
            if all_pass:
                self._log("PASS: All trace checks consistent")
            
            return all_pass
            
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check trends match physical expectations.
        Restores original values after test.
        """
        try:
            # Store original values
            orig_V0 = self.V0
            orig_N = self.N
            orig_p = self.p
            orig_d = self.d  # Use computed d value
            
            # Test 1: Increase V0 by 5% → d should increase
            self.V0 *= 1.05
            d_new = (2 * self.V0 / self.N) ** (1/3)
            if not d_new > orig_d:  # Compare with original d, not original V0
                self._log(f"MONO FAIL: d did not increase when V0 increased (d_new={d_new}, orig_d={orig_d})")
                self.V0 = orig_V0
                return False
            
            # Test 2: Increase N by 5% → d should decrease
            self.N *= 1.05
            d_new2 = (2 * self.V0 / self.N) ** (1/3)
            if not d_new2 < d_new:
                self._log(f"MONO FAIL: d did not decrease when N increased")
                self.V0 = orig_V0
                self.N = orig_N
                return False
            
            # Test 3: Increase p by 5% → |W| should increase
            self.p *= 1.05
            W_new = -self.p * self.V0
            if not abs(W_new) > abs(-orig_p * orig_V0):
                self._log(f"MONO FAIL: |W| did not increase when p increased")
                self.V0 = orig_V0
                self.N = orig_N
                self.p = orig_p
                return False
            
            # Test 4: Increase V0 → Q should increase (if ΔU fixed)
            Q_new = self.delta_U + self.p * self.V0
            Q_orig = self.delta_U + orig_p * orig_V0
            if not Q_new > Q_orig:
                self._log(f"MONO FAIL: Q did not increase when V0 increased")
                self.V0 = orig_V0
                self.N = orig_N
                self.p = orig_p
                return False
            
            # Restore original values
            self.V0 = orig_V0
            self.N = orig_N
            self.p = orig_p
            
            self._log("PASS: All monotonicity trends correct")
            return True
            
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {str(e)}")
            # Try to restore on error
            try:
                self.V0 = orig_V0
                self.N = orig_N
                self.p = orig_p
            except:
                pass
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return success status with logs.
        """
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        results = []
        try:
            results.append(("Invariants", self.test_invariants()))
            results.append(("Trace Consistency", self.test_trace_consistency()))
            results.append(("Monotonicity", self.test_monotonicity()))
            
            all_pass = all(r[1] for r in results)
            
            # Summary
            self.logs.append("\n=== Summary ===")
            for test_name, passed in results:
                status = "PASS" if passed else "FAIL"
                self.logs.append(f"{test_name}: {status}")
            
            self.logs.append(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            all_pass = False
        
        return all_pass, self.logs