import math
import sympy as sp
from typing import Tuple, List, Optional, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get()
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.g = 10.0  # Given in problem
        self.T = 0.05  # Stroboscope flash period
        self.ratio_numer = 3  # s1/s2 ratio numerator
        self.ratio_denom = 7  # s1/s2 ratio denominator
        
        # Extract results with safe access
        self.t = self._safe_get('t', 0.2)  # Time between adjacent balls (4T)
        self.x = self._safe_get('x', 0.0)  # Horizontal displacement between adjacent balls
        self.v0 = self._safe_get('v_0', 0.0)  # Initial horizontal velocity
        self.y1 = self._safe_get('y_1', 0.0)  # Vertical displacement for first interval
        self.y2 = self._safe_get('y_2', 0.0)  # Vertical displacement for second interval
        
        # Expected symbolic answer
        self.expected_v0 = 2*sp.sqrt(5)/5
        
        # Initialize sympy symbols
        self.v0_sym, self.t_sym, self.x_sym, self.y1_sym, self.y2_sym = sp.symbols('v0 t x y1 y2')
        self.g_sym = sp.symbols('g')
        
    def _safe_get(self, key, default):
        """Safely get value from results or attributes."""
        if hasattr(self.solution, key):
            return getattr(self.solution, key)
        return self.results.get(key, default)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants."""
        logs = []
        all_pass = True
        
        # 1. Check time interval calculation: t = 4T
        expected_t = 4 * self.T
        if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Time interval t={self.t} doesn't match 4T={expected_t}")
            all_pass = False
        else:
            logs.append(f"PASS: Time interval t={self.t:.3f}s matches 4T")
        
        # 2. Check vertical displacements from free fall
        # y1 = 0.5*g*t^2
        expected_y1 = 0.5 * self.g * self.t**2
        if not math.isclose(self.y1, expected_y1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: y1={self.y1} doesn't match 0.5*g*t²={expected_y1:.3f}")
            all_pass = False
        else:
            logs.append(f"PASS: y1={self.y1:.3f}m matches free fall formula")
        
        # y2 should be 3*y1 (from step analysis: y2 = 3*y1)
        if not math.isclose(self.y2, 3*self.y1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: y2={self.y2} not equal to 3*y1={3*self.y1:.3f}")
            all_pass = False
        else:
            logs.append(f"PASS: y2={self.y2:.3f}m = 3*y1 (consistent with kinematics)")
        
        # 3. Check horizontal motion: x = v0*t
        expected_x_from_v0 = self.v0 * self.t
        if not math.isclose(self.x, expected_x_from_v0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: x={self.x} doesn't match v0*t={expected_x_from_v0:.3f}")
            all_pass = False
        else:
            logs.append(f"PASS: x={self.x:.3f}m = v0*t (uniform horizontal motion)")
        
        # 4. Check the ratio condition symbolically
        # s1 = sqrt(x² + y1²), s2 = sqrt(x² + y2²)
        # s1/s2 = 3/7
        s1_expr = sp.sqrt(self.x_sym**2 + self.y1_sym**2)
        s2_expr = sp.sqrt(self.x_sym**2 + self.y2_sym**2)
        ratio_eq = sp.Eq(s1_expr/s2_expr, self.ratio_numer/self.ratio_denom)
        
        # Substitute known values
        ratio_eq_sub = ratio_eq.subs({
            self.x_sym: self.x,
            self.y1_sym: self.y1,
            self.y2_sym: self.y2
        })
        
        # Evaluate numerically
        left_val = float(s1_expr.subs({self.x_sym: self.x, self.y1_sym: self.y1}))
        right_val = float(s2_expr.subs({self.x_sym: self.x, self.y2_sym: self.y2}))
        ratio_val = left_val / right_val if right_val != 0 else 0
        
        if not math.isclose(ratio_val, self.ratio_numer/self.ratio_denom, 
                          rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Ratio s1/s2={ratio_val:.3f} ≠ {self.ratio_numer}/{self.ratio_denom}")
            all_pass = False
        else:
            logs.append(f"PASS: Ratio condition satisfied: s1/s2={ratio_val:.3f}")
        
        return all_pass, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate and final results."""
        logs = []
        all_pass = True
        
        # 1. Recompute x from ratio condition algebraically
        # From ratio equation: (x² + y1²)/(x² + y2²) = (3/7)²
        # y2 = 3*y1 from kinematics
        ratio_sq = (self.ratio_numer/self.ratio_denom)**2
        expected_x_sq = (self.y1**2 - ratio_sq * (3*self.y1)**2) / (ratio_sq - 1)
        
        if expected_x_sq < 0:
            logs.append("FAIL: Negative x² from ratio equation")
            all_pass = False
        else:
            expected_x = math.sqrt(expected_x_sq)
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x={self.x} doesn't match computed x={expected_x:.3f} from ratio")
                all_pass = False
            else:
                logs.append(f"PASS: x consistent with ratio equation")
        
        # 2. Recompute v0 from x and t
        expected_v0_from_x = self.x / self.t if self.t != 0 else 0
        if not math.isclose(self.v0, expected_v0_from_x, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v0={self.v0} doesn't match x/t={expected_v0_from_x:.3f}")
            all_pass = False
        else:
            logs.append(f"PASS: v0 consistent with x/t relation")
        
        # 3. Check against expected symbolic answer
        expected_v0_val = float(self.expected_v0)
        if not math.isclose(self.v0, expected_v0_val, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v0={self.v0} doesn't match expected {self.expected_v0}≈{expected_v0_val:.3f}")
            all_pass = False
        else:
            logs.append(f"PASS: v0 matches expected symbolic answer")
        
        return all_pass, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test how results change with perturbations."""
        logs = []
        all_pass = True
        
        # Save original values
        orig_y1, orig_y2, orig_x, orig_v0 = self.y1, self.y2, self.x, self.v0
        
        try:
            # Test 1: Perturb gravitational acceleration g
            # Increase g by 10%, check that y1 and y2 increase proportionally
            g_perturbed = self.g * 1.1
            y1_perturbed = 0.5 * g_perturbed * self.t**2
            y2_perturbed = 3 * y1_perturbed  # Still maintains y2 = 3*y1
            
            # Recompute x from ratio condition with perturbed y1
            ratio_sq = (self.ratio_numer/self.ratio_denom)**2
            x_sq_perturbed = (y1_perturbed**2 - ratio_sq * (3*y1_perturbed)**2) / (ratio_sq - 1)
            
            if x_sq_perturbed > 0:
                x_perturbed = math.sqrt(x_sq_perturbed)
                v0_perturbed = x_perturbed / self.t
                
                # Check monotonic trend: if g increases, vertical displacements increase,
                # so horizontal displacement needed to maintain ratio decreases? Let's verify
                if y1_perturbed > orig_y1 and y2_perturbed > orig_y2:
                    logs.append(f"PASS: y1 and y2 increase with increased g (as expected)")
                    
                    # The effect on v0 depends on the ratio equation
                    # Log the change for manual inspection
                    v0_change = (v0_perturbed - orig_v0) / orig_v0 * 100
                    logs.append(f"INFO: With g increased by 10%, v0 changes by {v0_change:.1f}%")
                else:
                    logs.append(f"WARN: Unexpected trend with g perturbation")
            else:
                logs.append(f"SKIP: Invalid x² with perturbed g")
            
            # Test 2: Perturb the time interval (simulating different stroboscope settings)
            # Increase t by 5%
            t_perturbed = self.t * 1.05
            y1_t_perturbed = 0.5 * self.g * t_perturbed**2
            
            # Recompute everything
            x_sq_t_perturbed = (y1_t_perturbed**2 - ratio_sq * (3*y1_t_perturbed)**2) / (ratio_sq - 1)
            
            if x_sq_t_perturbed > 0:
                x_t_perturbed = math.sqrt(x_sq_t_perturbed)
                v0_t_perturbed = x_t_perturbed / t_perturbed
                
                # Check if v0 changes as expected
                if t_perturbed > self.t:
                    # For fixed ratio, if t increases, y1 increases quadratically,
                    # so x must adjust to maintain ratio, and v0 = x/t
                    # The relationship is complex, so we just log the change
                    v0_t_change = (v0_t_perturbed - orig_v0) / orig_v0 * 100
                    logs.append(f"INFO: With t increased by 5%, v0 changes by {v0_t_change:.1f}%")
            
            # Test 3: Perturb the ratio (simulating different photographic measurements)
            # Change ratio from 3:7 to 2:7 (smaller s1 relative to s2)
            new_numer = 2
            new_ratio_sq = (new_numer/self.ratio_denom)**2
            
            # Keep same y1 (since g and t unchanged)
            x_sq_ratio_perturbed = (orig_y1**2 - new_ratio_sq * (3*orig_y1)**2) / (new_ratio_sq - 1)
            
            if x_sq_ratio_perturbed > 0:
                x_ratio_perturbed = math.sqrt(x_sq_ratio_perturbed)
                v0_ratio_perturbed = x_ratio_perturbed / self.t
                
                # Check monotonicity: if ratio decreases, x should increase
                if new_numer < self.ratio_numer and x_ratio_perturbed > orig_x:
                    logs.append(f"PASS: x increases when ratio decreases (monotonic trend)")
                elif new_numer > self.ratio_numer and x_ratio_perturbed < orig_x:
                    logs.append(f"PASS: x decreases when ratio increases (monotonic trend)")
                else:
                    logs.append(f"WARN: Unexpected trend with ratio perturbation")
            
            logs.append("PASS: Monotonicity tests completed")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            all_pass = False
        
        # Restore original values
        self.y1, self.y2, self.x, self.v0 = orig_y1, orig_y2, orig_x, orig_v0
        
        return all_pass, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall result."""
        all_logs = []
        overall_success = True
        
        all_logs.append("="*60)
        all_logs.append("Physics Checker for Projectile Motion Problem")
        all_logs.append("="*60)
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n1. PHYSICS INVARIANTS:")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n2. TRACE CONSISTENCY:")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n3. MONOTONICITY TESTS:")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        all_logs.append("\n" + "="*60)
        all_logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append("="*60)
        
        return overall_success, all_logs

# Example usage (for testing purposes only)
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                't': 0.2,
                'x': 2*math.sqrt(5)/25,  # Approximately 0.1789
                'v_0': 2*math.sqrt(5)/5,  # Approximately 0.8944
                'y_1': 0.2,
                'y_2': 0.6
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)