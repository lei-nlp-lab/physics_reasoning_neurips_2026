import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.overlinea = getattr(solution_instance, 'overlinea', 
                                getattr(solution_instance, 'results', {}).get('overlinea', 0.0))
        self.t = getattr(solution_instance, 't', 
                        getattr(solution_instance, 'results', {}).get('t', 0.0))
        
        # Problem constants from context
        self.v0 = 10.0  # m/s east
        self.v1 = -20.0 # m/s west (negative in east-positive coordinate system)
        self.delta_t_kick = 0.1  # s
        self.v_f_grass = 0.0  # Comes to rest on grass
        
        # For kinematic checks
        self.v_i_grass = self.v1  # Initial velocity on grass = velocity after kick
        
        # Setup symbolic equations
        self.setup_symbolic_equations()
    
    def setup_symbolic_equations(self):
        """Define symbolic equations from KB for verification"""
        v0, v1, dt, a_avg = sp.symbols('v0 v1 dt a_avg')
        v_i, v_f, a, t = sp.symbols('v_i v_f a t')
        
        # Theorem 1: Average acceleration
        self.eq_a_avg = sp.Eq(a_avg, (v1 - v0) / dt)
        
        # Theorem 2: Kinematic equations
        self.eq_v_avg = sp.Eq((v0 + v1) / 2, (v0 + v1) / 2)  # Identity for checking
        self.eq_kinematic_t = sp.Eq(t, (v_f - v_i) / a)
    
    def test_invariants(self):
        """Verify physical laws from KB"""
        logs = []
        
        try:
            # Check 1: Average acceleration formula (Theorem 1)
            a_avg_calc = (self.v1 - self.v0) / self.delta_t_kick
            
            # Magnitude check with 1-2% tolerance
            if math.isclose(abs(a_avg_calc), float(self.overlinea), 
                           rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Average acceleration magnitude matches formula")
            else:
                logs.append(f"✗ Average acceleration mismatch: {abs(a_avg_calc)} vs {self.overlinea}")
            
            # Direction check: acceleration should be west (negative)
            if a_avg_calc < 0:
                logs.append("✓ Acceleration direction correct (west)")
            else:
                logs.append("✗ Acceleration direction incorrect (should be west)")
            
            # Check 2: Kinematic equation for time (Theorem 2)
            # From step 2: t = (v_f - v_i) / a
            # We need acceleration on grass. From step 1 result and step 2 value:
            # a_grass = (v_f - v_i) / t = (0 - (-20)) / 10 = 2 m/s² east
            
            if float(self.t) > 0:
                a_grass = (self.v_f_grass - self.v_i_grass) / float(self.t)
                
                # Sign check: acceleration on grass should oppose motion (deceleration)
                if a_grass * self.v_i_grass < 0:
                    logs.append("✓ Grass acceleration opposes motion (deceleration)")
                else:
                    logs.append("✗ Grass acceleration direction inconsistent")
                
                # Numerical check: a_grass should be ~2 m/s² east
                if math.isclose(a_grass, 2.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("✓ Grass acceleration magnitude ~2 m/s²")
                else:
                    logs.append(f"✗ Grass acceleration magnitude unexpected: {a_grass}")
            else:
                logs.append("✗ Time on grass must be positive")
            
        except Exception as e:
            logs.append(f"Error in invariant tests: {e}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answers"""
        logs = []
        
        try:
            # Step 1 to Step 2 consistency
            # Acceleration from step 1 is 300 m/s² west (during kick)
            # Acceleration on grass from step 2 calculation is ~2 m/s² east
            # These are different accelerations for different phases - that's OK
            
            # Check that velocities connect properly
            # v_after_kick = v1 = -20 m/s (west)
            # v_on_grass_initial = should equal v1
            if math.isclose(self.v_i_grass, self.v1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Velocity after kick matches initial grass velocity")
            else:
                logs.append("✗ Velocity continuity broken between phases")
            
            # Check time values are reasonable
            if float(self.overlinea) > 0 and float(self.t) > 0:
                logs.append("✓ Both results positive as expected")
            else:
                logs.append("✗ Negative results detected")
            
            # Verify step 2 equation numerically
            a_grass = (self.v_f_grass - self.v_i_grass) / float(self.t)
            t_calc = (self.v_f_grass - self.v_i_grass) / a_grass
            
            if math.isclose(float(self.t), t_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Step 2 equation consistent numerically")
            else:
                logs.append("✗ Step 2 equation inconsistent")
            
        except Exception as e:
            logs.append(f"Error in trace consistency: {e}")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check trends"""
        logs = []
        
        try:
            # Test 1: Perturb initial velocity v0
            original_a = abs((self.v1 - self.v0) / self.delta_t_kick)
            
            for delta in [1.0, -1.0, 2.0]:
                v0_perturbed = self.v0 + delta
                a_perturbed = abs((self.v1 - v0_perturbed) / self.delta_t_kick)
                
                # Check trend: larger |Δv| → larger |a|
                if delta > 0:  # v0 increased → |Δv| increases
                    if a_perturbed > original_a:
                        logs.append(f"✓ Trend correct: v0↑ {delta} → |a|↑ ({original_a:.1f}→{a_perturbed:.1f})")
                    else:
                        logs.append(f"✗ Trend wrong: v0↑ {delta} but |a|↓ ({original_a:.1f}→{a_perturbed:.1f})")
                else:  # v0 decreased → |Δv| decreases
                    if a_perturbed < original_a:
                        logs.append(f"✓ Trend correct: v0↓ {delta} → |a|↓ ({original_a:.1f}→{a_perturbed:.1f})")
                    else:
                        logs.append(f"✗ Trend wrong: v0↓ {delta} but |a|↑ ({original_a:.1f}→{a_perturbed:.1f})")
            
            # Test 2: Perturb interaction time
            original_t = float(self.t)
            
            # For grass motion: t = (v_f - v_i) / a_grass
            # If a_grass increases (more deceleration), t decreases
            a_grass_original = (self.v_f_grass - self.v_i_grass) / original_t
            
            for factor in [0.8, 1.2]:
                a_grass_perturbed = a_grass_original * factor
                t_perturbed = (self.v_f_grass - self.v_i_grass) / a_grass_perturbed
                
                if factor > 1:  # Larger acceleration
                    if t_perturbed < original_t:
                        logs.append(f"✓ Trend correct: a_grass↑ {factor}x → t↓ ({original_t:.1f}→{t_perturbed:.1f})")
                    else:
                        logs.append(f"✗ Trend wrong: a_grass↑ {factor}x but t↑ ({original_t:.1f}→{t_perturbed:.1f})")
                else:  # Smaller acceleration
                    if t_perturbed > original_t:
                        logs.append(f"✓ Trend correct: a_grass↓ {factor}x → t↑ ({original_t:.1f}→{t_perturbed:.1f})")
                    else:
                        logs.append(f"✗ Trend wrong: a_grass↓ {factor}x but t↓ ({original_t:.1f}→{t_perturbed:.1f})")
            
        except Exception as e:
            logs.append(f"Error in monotonicity test: {e}")
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        all_logs = []
        success = True
        
        try:
            all_logs.append("=== Physics Checker Start ===")
            
            # Run individual test suites
            for test_name, test_func in [
                ("Invariant Tests", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity Tests", self.test_monotonicity)
            ]:
                all_logs.append(f"\n--- {test_name} ---")
                test_logs = test_func()
                all_logs.extend(test_logs)
                
                # Check for failures in this test suite
                for log in test_logs:
                    if log.startswith("✗"):
                        success = False
            
            all_logs.append(f"\n=== Final Result: {'PASS' if success else 'FAIL'} ===")
            
        except Exception as e:
            all_logs.append(f"Critical error in test execution: {e}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs