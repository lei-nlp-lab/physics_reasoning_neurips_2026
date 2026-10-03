import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dictionary
        self.inputs = {}
        self.results = {}
        
        # Extract inputs with safe defaults
        self.inputs['theta'] = getattr(solution_instance, 'theta', 37.0)
        self.inputs['sin37'] = getattr(solution_instance, 'sin37', 0.6)
        self.inputs['cos37'] = getattr(solution_instance, 'cos37', 0.8)
        self.inputs['g'] = getattr(solution_instance, 'g', 10.0)
        self.inputs['v0'] = getattr(solution_instance, 'v0', 4.0)
        self.inputs['L1'] = getattr(solution_instance, 'L1', 6.0)
        self.inputs['L2'] = getattr(solution_instance, 'L2', 4.0)
        self.inputs['mu2'] = getattr(solution_instance, 'mu2', 0.4)
        
        # Extract results safely - using getattr first, then results dict fallback
        sol_results = getattr(solution_instance, 'results', {})
        keys_to_extract = ['a_1', 'Deltax', 't_2', 't', 'a_2', 'x_2', 
                          't_1', 'mu_1', 's', 'x_1']
        
        for key in keys_to_extract:
            self.results[key] = getattr(solution_instance, key, sol_results.get(key, 0.0))
        
        # Initialize logs
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
    
    def fuzzy_equal(self, a, b):
        """Fuzzy numerical comparison with 1% tolerance."""
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Test physical invariants based on KB theorems."""
        success = True
        
        try:
            # Test 1: Kinematics on incline (step 1)
            v0 = self.inputs['v0']
            L1 = self.inputs['L1']
            expected_a1 = (v0**2) / (2 * L1)
            computed_a1 = self.results.get('a_1', 0.0)
            
            if not self.fuzzy_equal(computed_a1, expected_a1):
                self.log(f"Invariant 1 failed: a1={computed_a1}, expected={expected_a1}")
                success = False
            else:
                self.log(f"Invariant 1 passed: a1={computed_a1}")
            
            # Test 2: Newton's 2nd Law on incline (step 2)
            g = self.inputs['g']
            sin37 = self.inputs['sin37']
            cos37 = self.inputs['cos37']
            a1_val = float(computed_a1)
            expected_mu1 = (a1_val + g * sin37) / (g * cos37)
            computed_mu1 = self.results.get('mu_1', 0.0)
            
            if not self.fuzzy_equal(computed_mu1, expected_mu1):
                self.log(f"Invariant 2 failed: mu1={computed_mu1}, expected={expected_mu1}")
                success = False
            else:
                self.log(f"Invariant 2 passed: mu1={computed_mu1}")
            
            # Test 3: Friction on conveyor belt (step 3)
            mu2 = self.inputs['mu2']
            expected_a2 = mu2 * g
            computed_a2 = self.results.get('a_2', 0.0)
            
            if not self.fuzzy_equal(computed_a2, expected_a2):
                self.log(f"Invariant 3 failed: a2={computed_a2}, expected={expected_a2}")
                success = False
            else:
                self.log(f"Invariant 3 passed: a2={computed_a2}")
            
            # Test 4: Energy/work theorem alternative
            # Using kinematic relationship for uniformly accelerated motion
            computed_t1 = self.results.get('t_1', 0.0)
            computed_a2_val = float(computed_a2)
            expected_t1 = v0 / computed_a2_val if abs(computed_a2_val) > 1e-10 else 0.0
            
            if computed_a2_val > 0 and not self.fuzzy_equal(computed_t1, expected_t1):
                self.log(f"Invariant 4 failed: t1={computed_t1}, expected={expected_t1}")
                success = False
            elif computed_a2_val > 0:
                self.log(f"Invariant 4 passed: t1={computed_t1}")
            
            # Test 5: Displacement during acceleration (step 5)
            computed_x1 = self.results.get('x_1', 0.0)
            expected_x1 = (v0 / 2) * computed_t1
            
            if not self.fuzzy_equal(computed_x1, expected_x1):
                self.log(f"Invariant 5 failed: x1={computed_x1}, expected={expected_x1}")
                success = False
            else:
                self.log(f"Invariant 5 passed: x1={computed_x1}")
            
            # Test 6: Belt displacement during t1 (step 6)
            computed_s = self.results.get('s', 0.0)
            expected_s = v0 * computed_t1
            
            if not self.fuzzy_equal(computed_s, expected_s):
                self.log(f"Invariant 6 failed: s={computed_s}, expected={expected_s}")
                success = False
            else:
                self.log(f"Invariant 6 passed: s={computed_s}")
            
            # Test 7: Remaining distance (step 7)
            computed_x2 = self.results.get('x_2', 0.0)
            L2 = self.inputs['L2']
            expected_x2 = L2 - computed_x1
            
            if not self.fuzzy_equal(computed_x2, expected_x2):
                self.log(f"Invariant 7 failed: x2={computed_x2}, expected={expected_x2}")
                success = False
            else:
                self.log(f"Invariant 7 passed: x2={computed_x2}")
            
            # Test 8: Uniform motion time (step 8)
            computed_t2 = self.results.get('t_2', 0.0)
            expected_t2 = computed_x2 / v0 if abs(v0) > 1e-10 else 0.0
            
            if not self.fuzzy_equal(computed_t2, expected_t2):
                self.log(f"Invariant 8 failed: t2={computed_t2}, expected={expected_t2}")
                success = False
            else:
                self.log(f"Invariant 8 passed: t2={computed_t2}")
            
            # Test 9: Total time (step 9)
            computed_t = self.results.get('t', 0.0)
            expected_t = computed_t1 + computed_t2
            
            if not self.fuzzy_equal(computed_t, expected_t):
                self.log(f"Invariant 9 failed: t={computed_t}, expected={expected_t}")
                success = False
            else:
                self.log(f"Invariant 9 passed: t={computed_t}")
            
            # Test 10: Relative displacement/scratch (step 10)
            computed_dx = self.results.get('Deltax', 0.0)
            expected_dx = computed_s - computed_x1
            
            if not self.fuzzy_equal(computed_dx, expected_dx):
                self.log(f"Invariant 10 failed: Δx={computed_dx}, expected={expected_dx}")
                success = False
            else:
                self.log(f"Invariant 10 passed: Δx={computed_dx}")
                
        except Exception as e:
            self.log(f"Exception in test_invariants: {str(e)}")
            success = False
            
        return success
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        success = True
        
        try:
            # Check that target answers are consistent with computed results
            # Target answers: ["$\frac{11}{12}$", "$1.5\mathrm{m}$; $2\mathrm{m}$"]
            
            # First target: μ1 should be approximately 11/12
            computed_mu1 = self.results.get('mu_1', 0.0)
            expected_mu1 = 11/12  # ≈ 0.9167
            
            if not self.fuzzy_equal(computed_mu1, expected_mu1):
                self.log(f"Trace 1 failed: μ1={computed_mu1}, target={expected_mu1}")
                success = False
            else:
                self.log(f"Trace 1 passed: μ1 matches target 11/12")
            
            # Second target: t=1.5s and Δx=2m
            computed_t = self.results.get('t', 0.0)
            computed_dx = self.results.get('Deltax', 0.0)
            
            if not self.fuzzy_equal(computed_t, 1.5):
                self.log(f"Trace 2 failed: t={computed_t}, target=1.5")
                success = False
            else:
                self.log(f"Trace 2 passed: t=1.5s")
            
            if not self.fuzzy_equal(computed_dx, 2.0):
                self.log(f"Trace 3 failed: Δx={computed_dx}, target=2.0")
                success = False
            else:
                self.log(f"Trace 3 passed: Δx=2.0m")
            
            # Check consistency between steps
            computed_t1 = self.results.get('t_1', 0.0)
            computed_t2 = self.results.get('t_2', 0.0)
            
            if not self.fuzzy_equal(computed_t, computed_t1 + computed_t2):
                self.log(f"Trace 4 failed: t={computed_t} ≠ t1+t2={computed_t1}+{computed_t2}")
                success = False
            else:
                self.log(f"Trace 4 passed: t = t1 + t2")
            
            computed_x1 = self.results.get('x_1', 0.0)
            computed_x2 = self.results.get('x_2', 0.0)
            L2 = self.inputs['L2']
            
            if not self.fuzzy_equal(L2, computed_x1 + computed_x2):
                self.log(f"Trace 5 failed: L2={L2} ≠ x1+x2={computed_x1}+{computed_x2}")
                success = False
            else:
                self.log(f"Trace 5 passed: L2 = x1 + x2")
                
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {str(e)}")
            success = False
            
        return success
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs."""
        success = True
        
        try:
            # Save original state
            orig_mu2 = self.inputs['mu2']
            orig_results = self.results.copy()
            
            # Test 1: Increase μ2 should decrease total time t
            # Since higher friction on belt accelerates block faster to v0
            perturbed_mu2 = orig_mu2 * 1.1  # Increase by 10%
            
            # Recalculate affected quantities using formulas from steps
            g = self.inputs['g']
            v0 = self.inputs['v0']
            L2 = self.inputs['L2']
            
            # New a2
            new_a2 = perturbed_mu2 * g
            
            # New t1
            new_t1 = v0 / new_a2
            
            # New x1
            new_x1 = (v0 / 2) * new_t1
            
            # New s
            new_s = v0 * new_t1
            
            # New x2
            new_x2 = L2 - new_x1
            
            # New t2
            new_t2 = new_x2 / v0
            
            # New total time
            new_t = new_t1 + new_t2
            
            # Compare with original
            orig_t = self.results.get('t', 0.0)
            
            if perturbed_mu2 > orig_mu2 and new_t > orig_t:
                self.log(f"Monotonicity 1 failed: μ2↑ ({orig_mu2}→{perturbed_mu2}) but t↑ ({orig_t}→{new_t})")
                success = False
            elif perturbed_mu2 > orig_mu2 and new_t < orig_t:
                self.log(f"Monotonicity 1 passed: μ2↑→t↓ (expected for faster acceleration)")
            else:
                self.log(f"Monotonicity 1 inconclusive")
            
            # Test 2: Increase L1 should DECREASE μ1 (corrected physics)
            # Since longer incline with same v0→0 requires smaller deceleration, thus less friction
            orig_L1 = self.inputs['L1']
            perturbed_L1 = orig_L1 * 1.1  # Increase by 10%
            
            # Recalculate a1 and μ1
            new_a1 = (v0**2) / (2 * perturbed_L1)
            sin37 = self.inputs['sin37']
            cos37 = self.inputs['cos37']
            new_mu1 = (new_a1 + g * sin37) / (g * cos37)
            
            orig_mu1 = self.results.get('mu_1', 0.0)
            
            if perturbed_L1 > orig_L1 and new_mu1 < orig_mu1:
                self.log(f"Monotonicity 2 passed: L1↑→μ1↓ (expected: longer incline needs less friction)")
            elif perturbed_L1 > orig_L1 and new_mu1 > orig_mu1:
                self.log(f"Monotonicity 2 failed: L1↑ ({orig_L1}→{perturbed_L1}) but μ1↑ ({orig_mu1}→{new_mu1})")
                success = False
            else:
                self.log(f"Monotonicity 2 inconclusive")
            
            # Restore original state
            self.inputs['mu2'] = orig_mu2
            self.results = orig_results.copy()
            
        except Exception as e:
            self.log(f"Exception in test_monotonicity: {str(e)}")
            success = False
            # Restore on exception
            self.inputs['mu2'] = orig_mu2
            self.results = orig_results.copy()
            
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        
        # Run tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        # Overall success
        success = invariants_ok and trace_ok and monotonicity_ok
        
        # Summary
        self.log(f"\n=== SUMMARY ===")
        self.log(f"Invariants: {'PASS' if invariants_ok else 'FAIL'}")
        self.log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.log(f"Monotonicity: {'PASS' if monotonicity_ok else 'FAIL'}")
        self.log(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs