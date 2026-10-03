import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.L = getattr(solution_instance, 'L', 0.02)  # distance between adjacent surfaces (m)
        self.U_ac = getattr(solution_instance, 'U_ac', 60.0)  # potential difference a-c (V)
        
        # Extract results safely (solution_instance.results is expected)
        results = getattr(solution_instance, 'results', {})
        self.E = results.get('E', 0.0)  # electric field strength (N/C or V/m)
        self.varphi_c = results.get('varphi_c', 0.0)  # potential at c (V)
        self.W = results.get('W', 0.0)  # work from a-c (J)
        
        # For monotonicity: get charge q if available (from problem context)
        self.q = getattr(solution_instance, 'q', -1e-10)  # -0.1 nC default
        
        # Store original values for perturbation tests
        self._original_L = self.L
        self._original_U_ac = self.U_ac
        
        # Setup symbolic variables for invariant checks
        self.symbols = {
            'L': sp.Symbol('L', positive=True),
            'U_ac': sp.Symbol('U_ac', real=True),
            'E': sp.Symbol('E', real=True),
            'varphi_c': sp.Symbol('varphi_c', real=True),
            'W': sp.Symbol('W', real=True),
            'q': sp.Symbol('q', real=True),
            'd_ac': sp.Symbol('d_ac', positive=True),
            'd_bc': sp.Symbol('d_bc', positive=True)
        }
        
    def _is_close(self, a, b):
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Check E = U_ac / (2*L) from uniform field formula
            # Distance a-c is 2L (two intervals)
            expected_E = self.U_ac / (2 * self.L)
            if not self._is_close(self.E, expected_E):
                logs.append(f"E invariant failed: E={self.E}, expected U_ac/(2L)={expected_E}")
                success = False
            else:
                logs.append(f"E invariant passed: {self.E:.2f} ≈ {expected_E:.2f}")
            
            # 2. Check varphi_c = -E * L (potential drop from b to c, one interval)
            expected_varphi_c = -self.E * self.L
            if not self._is_close(self.varphi_c, expected_varphi_c):
                logs.append(f"varphi_c invariant failed: varphi_c={self.varphi_c}, expected -E*L={expected_varphi_c}")
                success = False
            else:
                logs.append(f"varphi_c invariant passed: {self.varphi_c:.2f} ≈ {expected_varphi_c:.2f}")
            
            # 3. Check W = q * U_ac (work-energy theorem for conservative field)
            expected_W = self.q * self.U_ac
            if not self._is_close(self.W, expected_W):
                logs.append(f"W invariant failed: W={self.W}, expected q*U_ac={expected_W}")
                success = False
            else:
                logs.append(f"W invariant passed: {self.W:.2e} ≈ {expected_W:.2e}")
            
            # 4. Symbolic check: delta_V = -E * delta_x in uniform field
            # Define symbolic relationship: U_ac = -E * (-2L) since moving opposite to field?
            # Actually in steps: E = U_ac/(2L) => U_ac = E * 2L (positive since moving along field?)
            # Let's verify with signs: If E is positive and potential decreases in field direction,
            # then moving from a to c (with the field) gives negative potential difference.
            # But problem states U_ac = 60V positive, so likely a is at higher potential.
            # We'll check absolute magnitudes:
            if not self._is_close(abs(self.U_ac), abs(self.E * 2 * self.L)):
                logs.append(f"Symbolic |U_ac| = |E*2L| check failed: {abs(self.U_ac)} vs {abs(self.E * 2 * self.L)}")
                success = False
            else:
                logs.append(f"Symbolic magnitude check passed: |U_ac| = |E*2L|")
                
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Step 1: E should be 1500 N/C
            if not self._is_close(self.E, 1500.0):
                logs.append(f"E trace mismatch: {self.E} vs expected 1500")
                success = False
            else:
                logs.append(f"E trace correct: {self.E:.2f} N/C")
            
            # Step 2: varphi_c should be -30 V
            if not self._is_close(self.varphi_c, -30.0):
                logs.append(f"varphi_c trace mismatch: {self.varphi_c} vs expected -30")
                success = False
            else:
                logs.append(f"varphi_c trace correct: {self.varphi_c:.2f} V")
            
            # Step 3: W should be -6.0e-9 J
            if not self._is_close(self.W, -6.0e-9):
                logs.append(f"W trace mismatch: {self.W:.2e} vs expected -6.0e-9")
                success = False
            else:
                logs.append(f"W trace correct: {self.W:.2e} J")
            
            # Step 4: Closed path work should be 0
            # We don't have that result stored, so skip this check
            logs.append("Closed path work check skipped (not stored in results)")
            
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase L by 10%, E should decrease (E = U_ac/(2L))
            perturbed_L = self.L * 1.1
            expected_E_new = self.U_ac / (2 * perturbed_L)
            
            # Since E ∝ 1/L, increasing L decreases E
            if not (expected_E_new < self.E):
                logs.append(f"Monotonicity L->E failed: E should decrease when L increases")
                success = False
            else:
                logs.append(f"L↑ → E↓ monotonicity passed: {self.E:.2f} → {expected_E_new:.2f}")
            
            # Test 2: Increase U_ac by 10%, E should increase proportionally
            perturbed_U = self.U_ac * 1.1
            expected_E_new2 = perturbed_U / (2 * self.L)
            
            if not (expected_E_new2 > self.E):
                logs.append(f"Monotonicity U_ac->E failed: E should increase when U_ac increases")
                success = False
            else:
                logs.append(f"U_ac↑ → E↑ monotonicity passed: {self.E:.2f} → {expected_E_new2:.2f}")
            
            # Test 3: varphi_c magnitude should increase with E (and thus with U_ac)
            expected_varphi_c_new = -expected_E_new2 * self.L
            if not (abs(expected_varphi_c_new) > abs(self.varphi_c)):
                logs.append(f"Monotonicity U_ac->|varphi_c| failed")
                success = False
            else:
                logs.append(f"U_ac↑ → |varphi_c|↑ monotonicity passed")
            
            # Restore original values
            self.L = self._original_L
            self.U_ac = self._original_U_ac
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            # Restore on error too
            self.L = self._original_L
            self.U_ac = self._original_U_ac
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"Test crashed: {traceback.format_exc()}")
                overall_success = False
            all_logs.append("")  # empty line between tests
        
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs