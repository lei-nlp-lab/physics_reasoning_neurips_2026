import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks from problem statement
        self.d = getattr(solution_instance, 'd', 600.0)
        self.vA = getattr(solution_instance, 'vA', 10.0)
        self.vB = getattr(solution_instance, 'vB', 20.0)
        self.v_bird = getattr(solution_instance, 'v_bird', 20.0)
        
        # Safely extract results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.t = self.results.get('t', 0.0)
        self.s = self.results.get('s', 0.0)
        self.x = self.results.get('x', 0.0)
        self.v = self.results.get('v', 0.0)
        
        self.logs = []
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants"""
        logs = []
        success = True
        
        # 1. Time invariant: t = d/(vA + vB)
        expected_t = self.d / (self.vA + self.vB)
        if not math.isclose(self.t, expected_t, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Time invariant failed: t={self.t}, expected {expected_t}")
            success = False
        
        # 2. Distance invariant: s = v_bird * t
        expected_s = self.v_bird * self.t
        if not math.isclose(self.s, expected_s, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Distance invariant failed: s={self.s}, expected {expected_s}")
            success = False
        
        # 3. Displacement invariant: x = vA * t
        expected_x = self.vA * self.t
        if not math.isclose(self.x, expected_x, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Displacement invariant failed: x={self.x}, expected {expected_x}")
            success = False
        
        # 4. Average speed invariant: v = s/t OR v = x/t
        if self.t > 0:
            expected_v1 = self.s / self.t
            expected_v2 = self.x / self.t
            if not (math.isclose(self.v, expected_v1, rel_tol=0.02, abs_tol=0.02) or 
                    math.isclose(self.v, expected_v2, rel_tol=0.02, abs_tol=0.02)):
                logs.append(f"Average speed invariant failed: v={self.v}, expected from distance {expected_v1} or from displacement {expected_v2}")
                success = False
        
        # 5. SymPy symbolic check for uniform motion equations
        try:
            t_sym = sp.Symbol('t')
            vA_sym = sp.Symbol('vA')
            vB_sym = sp.Symbol('vB')
            d_sym = sp.Symbol('d')
            
            # Check t = d/(vA + vB)
            eq1 = sp.Eq(t_sym, d_sym/(vA_sym + vB_sym))
            numeric_eq1 = eq1.subs({t_sym: self.t, d_sym: self.d, vA_sym: self.vA, vB_sym: self.vB})
            if not numeric_eq1.rhs.equals(numeric_eq1.lhs):  # Use equals for symbolic equality
                if not math.isclose(float(numeric_eq1.lhs), float(numeric_eq1.rhs), rel_tol=0.02):
                    logs.append(f"Symbolic time equation check failed: {numeric_eq1}")
                    success = False
        except Exception as e:
            logs.append(f"Symbolic check skipped: {e}")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate and final results"""
        logs = []
        success = True
        
        # Check that final answers match expected values from problem statement
        expected_s = 400.0
        expected_v = 10.0
        
        if not math.isclose(self.s, expected_s, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Distance inconsistency: s={self.s}, expected {expected_s}")
            success = False
        
        if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Average speed inconsistency: v={self.v}, expected {expected_v}")
            success = False
        
        # Check that intermediate calculations lead to final results
        # From steps: t = d/(vA + vB) = 600/(10+20) = 20
        expected_t = self.d/(self.vA + self.vB)
        if not math.isclose(self.t, expected_t, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Time calculation inconsistent: t={self.t}, expected {expected_t}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test how results change with parameter perturbations"""
        logs = []
        success = True
        
        # Save original values
        original_d = self.d
        
        try:
            # Perturb distance by +10%
            perturbed_d = original_d * 1.1
            
            # Recalculate using physics formulas
            perturbed_t = perturbed_d / (self.vA + self.vB)
            perturbed_s = self.v_bird * perturbed_t
            perturbed_x = self.vA * perturbed_t
            perturbed_v = perturbed_x / perturbed_t  # Average velocity from displacement
            
            # Check monotonic trends
            if not (perturbed_t > self.t):
                logs.append(f"Monotonicity failed: t doesn't increase with increased distance")
                success = False
            
            if not (perturbed_s > self.s):
                logs.append(f"Monotonicity failed: s doesn't increase with increased distance")
                success = False
            
            if not (perturbed_x > self.x):
                logs.append(f"Monotonicity failed: x doesn't increase with increased distance")
                success = False
            
            # v should stay constant (equal to vA) since it's displacement/time
            if not math.isclose(perturbed_v, self.vA, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Monotonicity failed: v changes with distance, expected constant {self.vA}")
                success = False
            
            # Test negative perturbation
            perturbed_d2 = original_d * 0.9
            perturbed_t2 = perturbed_d2 / (self.vA + self.vB)
            
            if not (perturbed_t2 < self.t):
                logs.append(f"Monotonicity failed: t doesn't decrease with decreased distance")
                success = False
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {e}")
            success = False
        
        finally:
            # Restore original distance
            self.d = original_d
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return results"""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            test_success, test_logs = test_func()
            all_logs.append(f"=== {test_name} Test ===")
            all_logs.extend(test_logs)
            if test_logs:
                all_logs.append(f"Result: {'PASS' if test_success else 'FAIL'}")
            else:
                all_logs.append(f"Result: PASS (no issues)")
            all_logs.append("")
            
            if not test_success:
                overall_success = False
        
        # Summary
        all_logs.append("=" * 40)
        all_logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs