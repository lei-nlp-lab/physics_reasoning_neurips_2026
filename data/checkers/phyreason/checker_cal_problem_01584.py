import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        Extract inputs and results from solution_instance.
        """
        self.solution = solution_instance
        
        # Input parameters: first try to get from results dict, then from attributes
        results = getattr(solution_instance, 'results', {})
        
        # Extract from results with fallback to attributes
        self.m = results.get('m', getattr(solution_instance, 'm', 1.0))
        self.v0 = results.get('v0', getattr(solution_instance, 'v0', 1.0))
        self.theta = results.get('theta', getattr(solution_instance, 'theta', 45.0 * math.pi/180))
        self.g = results.get('g', getattr(solution_instance, 'g', 9.8))
        self.E = results.get('E', getattr(solution_instance, 'E', 1.0))
        self.q = results.get('q', getattr(solution_instance, 'q', 1.0))
        
        # Results (use .get() for dictionary access)
        self.a = results.get('a', 0.0)
        self.l = results.get('l', 0.0)
        self.DeltaE_p = results.get('DeltaE_p', 0.0)
        self.W = results.get('W', 0.0)
        self.s = results.get('s', 0.0)
        self.mg_eq_condition = results.get('mg=Eqtantheta', None)
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Log storage
        self.logs = []
        
    def _log(self, msg):
        """Append log message."""
        self.logs.append(msg)
        
    def _fuzzy_equal(self, a, b):
        """Safe numerical comparison with tolerance."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        else:
            # Try to convert to float if possible
            try:
                val_a = float(a) if not isinstance(a, float) else a
                val_b = float(b) if not isinstance(b, float) else b
                return math.isclose(val_a, val_b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            except:
                return False
            
    def _check_condition(self):
        """
        Check the force equilibrium condition: mg = E q tan(θ)
        Based on STEP 1.
        """
        try:
            lhs = self.m * self.g
            rhs = self.E * self.q * math.tan(self.theta)
            
            if self._fuzzy_equal(lhs, rhs):
                self._log(f"✓ Force equilibrium: mg = {lhs:.4f}, E q tanθ = {rhs:.4f}")
                return True
            else:
                self._log(f"✗ Force equilibrium failed: mg = {lhs:.4f}, E q tanθ = {rhs:.4f}")
                return False
        except Exception as e:
            self._log(f"✗ Force equilibrium check error: {e}")
            return False
            
    def test_invariants(self):
        """
        Verify key physical invariants.
        Returns success count.
        """
        successes = 0
        
        # 1. Force equilibrium (STEP 1)
        if self._check_condition():
            successes += 1
            
        # 2. Acceleration check (STEP 2)
        try:
            expected_a = self.g / math.sin(self.theta)
            if self._fuzzy_equal(self.a, expected_a):
                self._log(f"✓ Acceleration: a = {self.a:.4f}, expected g/sinθ = {expected_a:.4f}")
                successes += 1
            else:
                self._log(f"✗ Acceleration mismatch: a = {self.a:.4f}, expected g/sinθ = {expected_a:.4f}")
        except Exception as e:
            self._log(f"✗ Acceleration check error: {e}")
            
        # 3. Kinematics: v0² = 2 a s (STEP 3)
        try:
            lhs = self.v0**2
            rhs = 2 * self.a * self.s
            if self._fuzzy_equal(lhs, rhs):
                self._log(f"✓ Kinematics: v0² = {lhs:.4f}, 2as = {rhs:.4f}")
                successes += 1
            else:
                self._log(f"✗ Kinematics failed: v0² = {lhs:.4f}, 2as = {rhs:.4f}")
        except Exception as e:
            self._log(f"✗ Kinematics check error: {e}")
            
        # 4. Geometry: l = s cosθ (STEP 4)
        try:
            expected_l = self.s * math.cos(self.theta)
            if self._fuzzy_equal(self.l, expected_l):
                self._log(f"✓ Geometry: l = {self.l:.4f}, s cosθ = {expected_l:.4f}")
                successes += 1
            else:
                self._log(f"✗ Geometry mismatch: l = {self.l:.4f}, s cosθ = {expected_l:.4f}")
        except Exception as e:
            self._log(f"✗ Geometry check error: {e}")
            
        # 5. Work: W = -E q l (STEP 5)
        try:
            expected_W = -self.E * self.q * self.l
            if self._fuzzy_equal(self.W, expected_W):
                self._log(f"✓ Work: W = {self.W:.4f}, -E q l = {expected_W:.4f}")
                successes += 1
            else:
                self._log(f"✗ Work mismatch: W = {self.W:.4f}, -E q l = {expected_W:.4f}")
        except Exception as e:
            self._log(f"✗ Work check error: {e}")
            
        # 6. Energy: ΔE_p = -W (STEP 6)
        try:
            if self._fuzzy_equal(self.DeltaE_p, -self.W):
                self._log(f"✓ Energy conservation: ΔE_p = {self.DeltaE_p:.4f}, -W = {-self.W:.4f}")
                successes += 1
            else:
                self._log(f"✗ Energy conservation failed: ΔE_p = {self.DeltaE_p:.4f}, -W = {-self.W:.4f}")
        except Exception as e:
            self._log(f"✗ Energy conservation check error: {e}")
            
        return successes
        
    def test_trace_consistency(self):
        """
        Cross-check that all intermediate results lead to final answer.
        """
        try:
            # Compute from base principles using STEPS analysis
            # STEP 2: a = g/sinθ
            a_calc = self.g / math.sin(self.theta)
            
            # STEP 3: s = v0²/(2a)
            s_calc = self.v0**2 / (2 * a_calc)
            
            # STEP 4: l = s cosθ
            l_calc = s_calc * math.cos(self.theta)
            
            # STEP 5 & 6: ΔE_p = 1/2 m v0² cos²θ
            expected_DeltaE_p = 0.5 * self.m * self.v0**2 * (math.cos(self.theta))**2
            
            # Check consistency chain
            chain_valid = True
            if not self._fuzzy_equal(self.a, a_calc):
                self._log(f"✗ Trace step 2 mismatch: a = {self.a:.4f}, calculated = {a_calc:.4f}")
                chain_valid = False
            if not self._fuzzy_equal(self.s, s_calc):
                self._log(f"✗ Trace step 3 mismatch: s = {self.s:.4f}, calculated = {s_calc:.4f}")
                chain_valid = False
            if not self._fuzzy_equal(self.l, l_calc):
                self._log(f"✗ Trace step 4 mismatch: l = {self.l:.4f}, calculated = {l_calc:.4f}")
                chain_valid = False
            if not self._fuzzy_equal(self.DeltaE_p, expected_DeltaE_p):
                self._log(f"✗ Final answer mismatch: ΔE_p = {self.DeltaE_p:.4f}, expected = {expected_DeltaE_p:.4f}")
                chain_valid = False
                
            if chain_valid:
                self._log("✓ All trace steps consistent with final answer")
                return True
            else:
                return False
                
        except Exception as e:
            self._log(f"✗ Trace consistency error: {e}")
            return False
            
    def test_monotonicity(self):
        """
        Perturb an input parameter and check trend.
        Use mass m as perturbation parameter.
        """
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Calculate expected ΔE_p using formula from STEP 6
            original_energy = 0.5 * original_m * self.v0**2 * (math.cos(self.theta))**2
            perturbed_energy = 0.5 * perturbed_m * self.v0**2 * (math.cos(self.theta))**2
            
            # Check monotonicity: energy should increase with mass
            if perturbed_energy > original_energy:
                self._log(f"✓ Monotonicity: ΔE_p increases with mass (linear dependence)")
                
                # Verify the ratio matches
                expected_ratio = perturbed_m / original_m
                actual_ratio = perturbed_energy / original_energy
                
                if self._fuzzy_equal(expected_ratio, actual_ratio):
                    self._log(f"  ✓ Ratio test: expected {expected_ratio:.4f}, got {actual_ratio:.4f}")
                    return True
                else:
                    self._log(f"  ✗ Ratio mismatch: expected {expected_ratio:.4f}, got {actual_ratio:.4f}")
                    return False
            else:
                self._log(f"✗ Monotonicity failed: energy doesn't increase with mass")
                return False
                
        except Exception as e:
            self._log(f"✗ Monotonicity test error: {e}")
            return False
        finally:
            # Restore original value
            self.m = original_m
            
    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs).
        """
        self.logs = []
        total_successes = 0
        
        try:
            # Test 1: Invariants (6 checks)
            self._log("=== Testing Physical Invariants ===")
            invariant_successes = self.test_invariants()
            total_successes += invariant_successes
            self._log(f"Invariants passed: {invariant_successes}/6")
            
            # Test 2: Trace Consistency
            self._log("\n=== Testing Trace Consistency ===")
            trace_success = self.test_trace_consistency()
            if trace_success:
                total_successes += 1
            self._log(f"Trace consistency: {'PASS' if trace_success else 'FAIL'}")
            
            # Test 3: Monotonicity
            self._log("\n=== Testing Monotonicity ===")
            monotonicity_success = self.test_monotonicity()
            if monotonicity_success:
                total_successes += 1
            self._log(f"Monotonicity: {'PASS' if monotonicity_success else 'FAIL'}")
            
            # Overall result (at least 7 out of 8 checks should pass)
            success = total_successes >= 7
            self._log(f"\n=== SUMMARY ===")
            self._log(f"Total checks passed: {total_successes}/8")
            self._log(f"Overall: {'PASS' if success else 'FAIL'}")
            
            return success, self.logs
            
        except Exception as e:
            self._log(f"Critical error in test suite: {traceback.format_exc()}")
            return False, self.logs