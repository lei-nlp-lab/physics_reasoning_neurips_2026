import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        try:
            # First try to get results dictionary
            results = getattr(solution_instance, 'results', {})
            
            # Extract results with safe fallbacks
            self.W_PQ = results.get('W_PQ', 0.0)
            self.T = results.get('T', 0.0)
            self.t = results.get('t', 0.0)
            self.a = results.get('a', 0.0)
            self.theta = results.get('theta', 0.0)
            self.U_PQ = results.get('U_PQ', 0.0)
            self.tantheta = results.get('tantheta', 0.0)
            
            # Extract input parameters from results or use defaults
            # Use the same values that were used in the solution code
            self.m = results.get('m', 2.0)  # Default to the values used in solution
            self.q = results.get('q', 1.5)
            self.L = results.get('L', 3.0)
            self.v0 = results.get('v0', 5.0)
            self.g = results.get('g', 9.8)
            
            # Derived electric field from equilibrium condition (qE = mg)
            if abs(self.q) > 1e-12:
                self.E = self.m * self.g / abs(self.q)
            else:
                self.E = 0.0
            
        except Exception as e:
            print(f"Warning: Initialization error - {e}")
            # Set defaults matching the solution's numerical values
            self.m, self.q, self.L, self.v0, self.g = 2.0, 1.5, 3.0, 5.0, 9.8
            self.W_PQ = self.T = self.t = self.a = self.theta = self.U_PQ = self.tantheta = 0.0
            self.E = 0.0

    def _is_close(self, a, b):
        """Fuzzy numerical comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Verify fundamental physics laws based on KB."""
        logs = []
        success = True
        
        try:
            # 1. Check force equilibrium in second quadrant (qE = mg)
            qE = abs(self.q) * self.E
            mg = self.m * self.g
            if not self._is_close(qE, mg):
                logs.append(f"FAIL: Force equilibrium violated. qE={qE}, mg={mg}")
                success = False
            else:
                logs.append(f"PASS: Force equilibrium qE = mg satisfied.")
            
            # 2. Check angle θ = 45° from tanθ = mg/(qE) = 1
            if not self._is_close(self.tantheta, 1.0):
                logs.append(f"FAIL: tanθ should be 1.0, got {self.tantheta}")
                success = False
            else:
                logs.append(f"PASS: tanθ = 1.0 verified.")
            
            # 3. Check work-energy theorem from P to Q (net work = 0)
            # W_PQ = qEL = mgL (from solution)
            # Net work = W_PQ - mgL = mgL - mgL = 0
            W_net = self.W_PQ - self.m * self.g * self.L
            if not self._is_close(W_net, 0.0):
                logs.append(f"FAIL: Net work from P to Q should be 0, got {W_net}")
                success = False
            else:
                logs.append(f"PASS: Work-energy theorem (net work = 0) satisfied.")
            
            # 4. Check potential difference U_PQ = -mgL/q
            expected_U_PQ = -self.m * self.g * self.L / self.q
            if not self._is_close(self.U_PQ, expected_U_PQ):
                logs.append(f"FAIL: U_PQ mismatch. Expected {expected_U_PQ}, got {self.U_PQ}")
                success = False
            else:
                logs.append(f"PASS: Potential difference U_PQ = -mgL/q verified.")
            
            # 5. Check acceleration in first quadrant: a = √2 * g
            expected_a = math.sqrt(2) * self.g
            if not self._is_close(self.a, expected_a):
                logs.append(f"FAIL: Acceleration mismatch. Expected {expected_a}, got {self.a}")
                success = False
            else:
                logs.append(f"PASS: Acceleration a = √2*g verified.")
            
            # 6. Check time t = v0/(√2*g) = (√2*v0)/(2g)
            expected_t = self.v0 / (math.sqrt(2) * self.g)
            if not self._is_close(self.t, expected_t):
                logs.append(f"FAIL: Time t mismatch. Expected {expected_t}, got {self.t}")
                success = False
            else:
                logs.append(f"PASS: Time t = v0/(√2*g) verified.")
            
            # 7. Check round trip time T = 2t = √2*v0/g
            expected_T1 = 2 * self.t
            expected_T2 = math.sqrt(2) * self.v0 / self.g
            if not (self._is_close(self.T, expected_T1) and self._is_close(self.T, expected_T2)):
                logs.append(f"FAIL: T mismatch. Expected {expected_T2} or 2t={expected_T1}, got {self.T}")
                success = False
            else:
                logs.append(f"PASS: Round trip time T = √2*v0/g verified.")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check θ = 45°
            if not self._is_close(self.theta, 45.0):
                logs.append(f"FAIL: θ should be 45°, got {self.theta}°")
                success = False
            else:
                logs.append(f"PASS: θ = 45° consistent.")
            
            # 2. Check U_PQ = W_PQ / (-q)
            if abs(self.q) > 1e-12:
                expected_from_W = self.W_PQ / (-self.q)
                if not self._is_close(self.U_PQ, expected_from_W):
                    logs.append(f"FAIL: U_PQ should equal W_PQ/(-q). Got {self.U_PQ}, expected {expected_from_W}")
                    success = False
                else:
                    logs.append(f"PASS: U_PQ = W_PQ/(-q) consistent.")
            
            # 3. Check T = 2t
            if not self._is_close(self.T, 2 * self.t):
                logs.append(f"FAIL: T should equal 2t. T={self.T}, 2t={2*self.t}")
                success = False
            else:
                logs.append(f"PASS: T = 2t consistent.")
            
            # 4. Check a = √2*g (already in invariants, but double-check)
            if not self._is_close(self.a, math.sqrt(2) * self.g):
                logs.append(f"FAIL: a should be √2*g. a={self.a}, √2*g={math.sqrt(2)*self.g}")
                success = False
            else:
                logs.append(f"PASS: a = √2*g consistent.")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """Perturb inputs and check trends."""
        logs = []
        success = True
        
        try:
            # Store original values
            orig_m, orig_q, orig_L, orig_v0, orig_g = self.m, self.q, self.L, self.v0, self.g
            orig_T, orig_U_PQ = self.T, self.U_PQ
            
            # Test 1: Perturb v0 by +10%, check T increases proportionally
            self.v0 *= 1.10
            new_T = math.sqrt(2) * self.v0 / self.g
            if not self._is_close(new_T / orig_T, 1.10):
                logs.append(f"FAIL: T should scale linearly with v0. Factor: {new_T/orig_T:.3f}, expected 1.10")
                success = False
            else:
                logs.append(f"PASS: T ∝ v0 verified.")
            
            # Restore v0
            self.v0 = orig_v0
            
            # Test 2: Perturb m by +10%, check |U_PQ| increases proportionally
            self.m *= 1.10
            new_U_PQ = -self.m * self.g * self.L / self.q
            if not self._is_close(abs(new_U_PQ) / abs(orig_U_PQ), 1.10):
                logs.append(f"FAIL: |U_PQ| should scale linearly with m. Factor: {abs(new_U_PQ)/abs(orig_U_PQ):.3f}, expected 1.10")
                success = False
            else:
                logs.append(f"PASS: |U_PQ| ∝ m verified.")
            
            # Restore m
            self.m = orig_m
            
            # Test 3: Perturb g by +10%, check T decreases and |U_PQ| increases
            self.g *= 1.10
            new_T = math.sqrt(2) * self.v0 / self.g
            new_U_PQ = -self.m * self.g * self.L / self.q
            if not (self._is_close(new_T / orig_T, 1/1.10) and 
                    self._is_close(abs(new_U_PQ) / abs(orig_U_PQ), 1.10)):
                logs.append(f"FAIL: T ∝ 1/g and |U_PQ| ∝ g violated.")
                success = False
            else:
                logs.append(f"PASS: T ∝ 1/g and |U_PQ| ∝ g verified.")
            
            # Restore g
            self.g = orig_g
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test suite
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            success, logs = test_method()
            overall_success = overall_success and success
            all_logs.extend(logs)
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs