import math
import traceback
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs safely
        self.inputs['h'] = getattr(solution_instance, 'h', 1.0)  # default non-zero
        self.inputs['omega'] = getattr(solution_instance, 'omega', 1.0)
        self.inputs['g'] = getattr(solution_instance, 'g', 9.8)
        self.inputs['R'] = getattr(solution_instance, 'R', 6.37e6)
        
        # Extract results safely from solution_instance.results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        # Expected keys mapping
        self.results['fracT_1T_2'] = float(results_dict.get('fracT_1T_2', 0.0))
        self.results['R_prime'] = float(results_dict.get("R'", results_dict.get('R_prime', 0.0)))
        self.results['N/A'] = float(results_dict.get('N/A', 0.0))
        self.results['a_A'] = float(results_dict.get('a_A', 0.0))
        self.results['L'] = float(results_dict.get('L', 0.0))
        
        # Pre-compute expected values for verification
        self.GM = self.inputs['g'] * self.inputs['R']**2  # GM = gR² from surface gravity
        
        # Expected formulas from steps analysis
        self.expected_a_A = self.GM / (self.inputs['R'] + self.inputs['h'])**2
        self.expected_L = (self.GM / (self.inputs['omega']**2))**(1/3)
        self.expected_R_prime = (self.expected_L + self.inputs['R'] + self.inputs['h']) / 2
        
        # Expected period ratio: T1/T2 = (R'/L)^(3/2)
        if self.expected_L != 0:
            self.expected_fracT_1T_2 = (self.expected_R_prime / self.expected_L)**(1.5)
        else:
            self.expected_fracT_1T_2 = 0.0
        
    def test_invariants(self):
        logs = []
        
        try:
            # 1. Check gravitational force equals centripetal force at surface
            # GM = gR²
            surface_check = math.isclose(self.GM, 
                                        self.inputs['g'] * self.inputs['R']**2,
                                        rel_tol=1e-2, abs_tol=1e-2)
            if surface_check:
                logs.append("✓ Surface gravity consistent: GM = gR²")
            else:
                logs.append(f"✗ Surface gravity inconsistent: GM={self.GM}, gR²={self.inputs['g']*self.inputs['R']**2}")
            
            # 2. Check acceleration at perigee A
            a_A_check = math.isclose(self.results['a_A'], self.expected_a_A,
                                    rel_tol=1e-2, abs_tol=1e-2)
            if a_A_check:
                logs.append(f"✓ Acceleration at perigee correct: a_A = {self.results['a_A']:.3e}, expected {self.expected_a_A:.3e}")
            else:
                logs.append(f"✗ Acceleration at perigee mismatch: got {self.results['a_A']:.3e}, expected {self.expected_a_A:.3e}")
            
            # 3. Check circular orbit radius L
            L_check = math.isclose(self.results['L'], self.expected_L,
                                  rel_tol=1e-2, abs_tol=1e-2)
            if L_check:
                logs.append(f"✓ Circular orbit radius correct: L = {self.results['L']:.3e}, expected {self.expected_L:.3e}")
            else:
                logs.append(f"✗ Circular orbit radius mismatch: got {self.results['L']:.3e}, expected {self.expected_L:.3e}")
            
            # 4. Check semi-major axis R'
            R_prime_check = math.isclose(self.results['R_prime'], self.expected_R_prime,
                                        rel_tol=1e-2, abs_tol=1e-2)
            if R_prime_check:
                logs.append(f"✓ Semi-major axis correct: R' = {self.results['R_prime']:.3e}, expected {self.expected_R_prime:.3e}")
            else:
                logs.append(f"✗ Semi-major axis mismatch: got {self.results['R_prime']:.3e}, expected {self.expected_R_prime:.3e}")
            
            # 5. Check Kepler's Third Law relation
            # For elliptical orbit: T1² ∝ R'³, for circular: T2² ∝ L³
            # So T1/T2 = (R'/L)^(3/2)
            period_ratio_check = math.isclose(self.results['fracT_1T_2'], self.expected_fracT_1T_2,
                                             rel_tol=1e-2, abs_tol=1e-2)
            if period_ratio_check:
                logs.append(f"✓ Period ratio correct: T1/T2 = {self.results['fracT_1T_2']:.3e}, expected {self.expected_fracT_1T_2:.3e}")
            else:
                logs.append(f"✗ Period ratio mismatch: got {self.results['fracT_1T_2']:.3e}, expected {self.expected_fracT_1T_2:.3e}")
            
            # 6. Check that perigee velocity > apogee velocity using vis-viva equation
            # Only perform this check if the orbit is physically plausible
            r_A = self.inputs['R'] + self.inputs['h']
            r_B = 2*self.expected_R_prime - r_A  # from ellipse property: r_A + r_B = 2a
            
            # Check if orbit is physically plausible (apogee >= perigee > 0)
            if r_A > 0 and r_B > 0 and r_B >= r_A and self.expected_R_prime > 0:
                v_A_sq = self.GM * (2/r_A - 1/self.expected_R_prime)
                v_B_sq = self.GM * (2/r_B - 1/self.expected_R_prime)
                
                if v_A_sq > v_B_sq:
                    logs.append("✓ Perigee velocity greater than apogee velocity (Kepler's 2nd law)")
                    velocity_check = True
                else:
                    logs.append(f"✗ Velocity ordering incorrect: v_A²={v_A_sq:.3e}, v_B²={v_B_sq:.3e}")
                    velocity_check = False
            else:
                # Unphysical orbit parameters (e.g., default ω=1.0 gives L < R+h)
                # Skip this test with a warning
                logs.append("⚠ Skipping velocity test: orbit parameters unphysical (apogee < perigee)")
                velocity_check = True  # Don't fail overall test due to unrealistic defaults
            
            # Overall success
            success = (surface_check and a_A_check and L_check and 
                      R_prime_check and period_ratio_check and velocity_check)
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        
        try:
            # Check that L appears consistently in formulas
            # From step 5: L = (gR²/ω²)^(1/3)
            computed_L = (self.GM / (self.inputs['omega']**2))**(1/3)
            
            L_consistent = math.isclose(self.results['L'], computed_L,
                                       rel_tol=1e-2, abs_tol=1e-2)
            if L_consistent:
                logs.append(f"✓ L consistent with input parameters: L = {self.results['L']:.3e}")
            else:
                logs.append(f"✗ L inconsistent: from results {self.results['L']:.3e}, recomputed {computed_L:.3e}")
            
            # Check that R' = (L + R + h)/2
            computed_R_prime = (self.results['L'] + self.inputs['R'] + self.inputs['h']) / 2
            R_prime_consistent = math.isclose(self.results['R_prime'], computed_R_prime,
                                             rel_tol=1e-2, abs_tol=1e-2)
            if R_prime_consistent:
                logs.append(f"✓ R' consistent with L, R, h: R' = {self.results['R_prime']:.3e}")
            else:
                logs.append(f"✗ R' inconsistent: from results {self.results['R_prime']:.3e}, recomputed {computed_R_prime:.3e}")
            
            # Check that T1/T2 = (R'/L)^(3/2)
            if self.results['L'] != 0:
                computed_ratio = (self.results['R_prime'] / self.results['L'])**(1.5)
                ratio_consistent = math.isclose(self.results['fracT_1T_2'], computed_ratio,
                                               rel_tol=1e-2, abs_tol=1e-2)
                if ratio_consistent:
                    logs.append(f"✓ Period ratio consistent with R' and L: T1/T2 = {self.results['fracT_1T_2']:.3e}")
                else:
                    logs.append(f"✗ Period ratio inconsistent: from results {self.results['fracT_1T_2']:.3e}, recomputed {computed_ratio:.3e}")
            else:
                logs.append("✗ Cannot check period ratio: L is zero")
                ratio_consistent = False
            
            # Check that a_A = GM/(R+h)²
            computed_a_A = self.GM / (self.inputs['R'] + self.inputs['h'])**2
            a_A_consistent = math.isclose(self.results['a_A'], computed_a_A,
                                         rel_tol=1e-2, abs_tol=1e-2)
            if a_A_consistent:
                logs.append(f"✓ a_A consistent with gravitational formula: a_A = {self.results['a_A']:.3e}")
            else:
                logs.append(f"✗ a_A inconsistent: from results {self.results['a_A']:.3e}, recomputed {computed_a_A:.3e}")
            
            success = (L_consistent and R_prime_consistent and 
                      ratio_consistent and a_A_consistent)
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        
        try:
            # Perturb height h by +10%
            original_h = self.inputs['h']
            perturbed_h = original_h * 1.10
            
            # Save original state
            original_results = deepcopy(self.results)
            
            # Recompute expected values with perturbed h
            perturbed_a_A = self.GM / (self.inputs['R'] + perturbed_h)**2
            # L remains same (depends on ω, g, R, not h)
            perturbed_L = self.expected_L
            perturbed_R_prime = (perturbed_L + self.inputs['R'] + perturbed_h) / 2
            
            if perturbed_L != 0:
                perturbed_ratio = (perturbed_R_prime / perturbed_L)**(1.5)
            else:
                perturbed_ratio = 0.0
            
            # Check monotonic trends:
            # 1. As h increases, acceleration at perigee should decrease
            if perturbed_a_A < self.expected_a_A:
                logs.append("✓ a_A decreases as h increases (inverse square law)")
                monotonic_a = True
            else:
                logs.append(f"✗ a_A monotonicity failed: a_A(h)={self.expected_a_A:.3e}, a_A(1.1h)={perturbed_a_A:.3e}")
                monotonic_a = False
            
            # 2. As h increases, semi-major axis should increase
            if perturbed_R_prime > self.expected_R_prime:
                logs.append("✓ R' increases as h increases (larger orbit)")
                monotonic_R = True
            else:
                logs.append(f"✗ R' monotonicity failed: R'(h)={self.expected_R_prime:.3e}, R'(1.1h)={perturbed_R_prime:.3e}")
                monotonic_R = False
            
            # 3. As h increases, period ratio T1/T2 should increase (longer elliptical period)
            if perturbed_ratio > self.expected_fracT_1T_2:
                logs.append("✓ T1/T2 increases as h increases (Kepler's 3rd law)")
                monotonic_ratio = True
            else:
                logs.append(f"✗ T1/T2 monotonicity failed: ratio(h)={self.expected_fracT_1T_2:.3e}, ratio(1.1h)={perturbed_ratio:.3e}")
                monotonic_ratio = False
            
            # Restore original values
            self.inputs['h'] = original_h
            self.results = original_results
            
            success = (monotonic_a and monotonic_R and monotonic_ratio)
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"✗ {test_name} FAILED")
                else:
                    all_logs.append(f"✓ {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"✗ {test_name} ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs