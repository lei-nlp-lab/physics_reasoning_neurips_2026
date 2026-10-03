import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safe initialization with defaults to avoid attribute errors."""
        # Inputs (use getattr with fallback from KB)
        self.m_squid = getattr(solution_instance, 'm_squid', 1.4)
        self.m_water = getattr(solution_instance, 'm_water', 0.1)
        self.v_squid_final = getattr(solution_instance, 'v_squid_final', 2.0)
        
        # Results (fallback to 0.0 if missing)
        self.v_water = getattr(solution_instance, 'v_2', 
                               getattr(solution_instance, 'results', {}).get('v_2', 0.0))
        
        # Derived quantities
        self.M = self.m_squid  # squid mass after expulsion
        self.m = self.m_water  # expelled water mass
        self.v1 = self.v_squid_final  # squid's final velocity
        
        # Expected answer from KB
        self.expected_v_water = 28.0
        
        # Logs
        self.logs = []
        
    def log(self, msg):
        """Helper to append log messages."""
        self.logs.append(msg)
    
    def test_invariants(self):
        """Test conservation of momentum (primary physics law)."""
        try:
            # Momentum conservation: initial = 0, final = M*v1 + m*v2_actual
            # v2_actual should be negative since water is ejected backward
            # The stored v_water is magnitude, so actual velocity = -v_water
            v2_actual = -self.v_water
            
            # Calculate final momentum
            rhs_val = self.M * self.v1 + self.m * v2_actual
            
            # Check if momentum is conserved (should be zero)
            if not math.isclose(0, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL momentum conservation: 0 ≠ {rhs_val:.3f}")
                return False
            
            # Check magnitude equation v2 = (M*v1)/m from steps analysis
            expected_from_eq = (self.M * self.v1) / self.m
            if not math.isclose(self.v_water, expected_from_eq, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL magnitude equation: v_water={self.v_water:.3f}, expected {expected_from_eq:.3f}")
                return False
            
            self.log("PASS momentum conservation and magnitude equation")
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate step with final answer."""
        try:
            # From step analysis: v_2 = (M * v1) / m = (1.4*2)/0.1 = 28
            step_value = 28.0
            if math.isclose(self.v_water, step_value, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"PASS trace consistency: v_water={self.v_water:.3f} matches step analysis {step_value}")
                return True
            else:
                self.log(f"FAIL trace consistency: v_water={self.v_water:.3f} ≠ step analysis {step_value}")
                return False
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb water mass and check trend, then restore."""
        original_m = self.m
        original_v = self.v_water
        
        try:
            # Increase water mass by 10%
            perturbed_m = original_m * 1.1
            # Recalculate expected v2 using v2 = (M*v1)/m
            new_v = (self.M * self.v1) / perturbed_m
            
            # v2 should decrease (inverse relationship)
            if new_v < original_v:
                self.log(f"PASS monotonicity: v_water decreases ({original_v:.3f} → {new_v:.3f}) when m_water increases")
                trend_ok = True
            else:
                self.log(f"FAIL monotonicity: v_water did not decrease when m_water increased")
                trend_ok = False
            
            # Restore original values
            self.m = original_m
            self.v_water = original_v
            
            return trend_ok
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            # Restore on error
            self.m = original_m
            self.v_water = original_v
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Momentum Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            if not test_func():
                success = False
        
        return success, self.logs