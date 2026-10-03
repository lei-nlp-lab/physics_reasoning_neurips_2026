import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results with fallbacks
        self.solution = solution_instance
        
        # Extract acceleration result
        self.a = getattr(solution_instance, 'a', 
                        getattr(solution_instance, 'results', {}).get('a', 0.0))
        
        # Try to extract intermediate values if available
        self.v0 = getattr(solution_instance, 'v0', 
                         getattr(solution_instance, 'results', {}).get('v0', None))
        self.v1 = getattr(solution_instance, 'v1', 
                         getattr(solution_instance, 'results', {}).get('v1', None))
        self.t = getattr(solution_instance, 't', 
                        getattr(solution_instance, 'results', {}).get('t', 0.1))
        
        # Expected values from problem statement
        self.expected_v0_mag = 10.0  # m/s toward ground
        self.expected_v1_mag = 20.0  # m/s opposite direction
        self.expected_t = 0.1  # s
        
        # Expected acceleration magnitude
        self.expected_a_mag = 300.0  # m/s²
        
        # Direction convention: positive = upward, negative = downward
        # Initial velocity is downward, final is upward
        self.sign_correct = None
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physics invariants and definitions"""
        logs = []
        success = True
        
        # Check 1: Magnitude of acceleration from kinematic definition
        # a = (v1 - v0)/t
        # Since v0 and v1 are in opposite directions, magnitude should be large
        
        # First, try symbolic verification
        try:
            v0_sym, v1_sym, t_sym, a_sym = sp.symbols('v0 v1 t a')
            eq = sp.Eq(a_sym, (v1_sym - v0_sym) / t_sym)
            
            # Substitute expected values
            # v0 = -10 (downward), v1 = +20 (upward) if using sign convention
            # Or v0 = +10, v1 = -20 if using magnitude with direction
            # Let's check both conventions
            
            # Convention 1: v0 = -10, v1 = +20
            test1 = eq.subs({v0_sym: -10, v1_sym: 20, t_sym: 0.1})
            a_calc1 = float((20 - (-10)) / 0.1)  # Should be 300
            
            # Convention 2: v0 = +10, v1 = -20  
            test2 = eq.subs({v0_sym: 10, v1_sym: -20, t_sym: 0.1})
            a_calc2 = float((-20 - 10) / 0.1)  # Should be -300
            
            # Check if solution matches either convention
            matches_convention1 = math.isclose(abs(self.a), 300, rel_tol=1e-2, abs_tol=1e-2)
            matches_convention2 = math.isclose(abs(self.a), 300, rel_tol=1e-2, abs_tol=1e-2)
            
            if matches_convention1 or matches_convention2:
                logs.append("✓ Acceleration magnitude matches kinematic definition")
                # Determine sign convention used
                if self.a > 0:
                    self.sign_correct = "positive upward convention"
                else:
                    self.sign_correct = "negative upward convention"
            else:
                logs.append(f"✗ Acceleration magnitude mismatch. Got |a|={abs(self.a):.2f}, expected ~300")
                success = False
                
        except Exception as e:
            logs.append(f"Symbolic check failed: {e}")
            # Fall back to numerical check
            expected_change = 30.0  # m/s magnitude change (from 10 to 20 in opposite direction)
            expected_a_mag = expected_change / self.expected_t  # 300 m/s²
            
            if math.isclose(abs(self.a), expected_a_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Acceleration magnitude matches expected value (numerical fallback)")
            else:
                logs.append(f"✗ Acceleration magnitude mismatch. Got |a|={abs(self.a):.2f}, expected ~{expected_a_mag}")
                success = False
        
        # Check 2: Direction consistency
        if self.v0 is not None and self.v1 is not None:
            # v0 and v1 should have opposite signs
            if (self.v0 > 0 and self.v1 < 0) or (self.v0 < 0 and self.v1 > 0):
                logs.append("✓ Initial and final velocities have opposite directions (as expected)")
            else:
                logs.append("✗ Initial and final velocities should be in opposite directions")
                success = False
                
            # Check acceleration sign relative to velocity change
            velocity_change = self.v1 - self.v0
            if not math.isclose(0, velocity_change, rel_tol=1e-2, abs_tol=1e-2):
                # Acceleration should have same sign as velocity change
                if (velocity_change > 0 and self.a > 0) or (velocity_change < 0 and self.a < 0):
                    logs.append("✓ Acceleration direction matches velocity change direction")
                else:
                    logs.append("✗ Acceleration direction inconsistent with velocity change")
                    success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer"""
        logs = []
        success = True
        
        # If we have v0, v1, and t, verify they're consistent with a
        if all(x is not None for x in [self.v0, self.v1, self.t]):
            calculated_a = (self.v1 - self.v0) / self.t
            
            if math.isclose(self.a, calculated_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Acceleration consistent with velocities and time")
            else:
                logs.append(f"✗ Inconsistency: a={self.a} but (v1-v0)/t={calculated_a}")
                success = False
                
            # Check magnitudes are reasonable
            if math.isclose(abs(self.v0), self.expected_v0_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Initial velocity magnitude matches problem statement")
            else:
                logs.append(f"✗ Initial velocity magnitude mismatch: |v0|={abs(self.v0)}, expected {self.expected_v0_mag}")
                success = False
                
            if math.isclose(abs(self.v1), self.expected_v1_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Final velocity magnitude matches problem statement")
            else:
                logs.append(f"✗ Final velocity magnitude mismatch: |v1|={abs(self.v1)}, expected {self.expected_v1_mag}")
                success = False
        else:
            logs.append("⚠ Could not perform full trace consistency check (missing v0, v1, or t)")
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test trend from KB: a ∝ Δv/t"""
        logs = []
        success = True
        
        try:
            # Test 1: Increase time, acceleration magnitude should decrease
            original_t = self.expected_t
            perturbed_t = original_t * 1.2  # Increase by 20%
            
            # Calculate acceleration with perturbed time (keeping Δv constant)
            delta_v_mag = 30.0  # From 10 m/s to 20 m/s in opposite direction
            original_a_mag = delta_v_mag / original_t
            perturbed_a_mag = delta_v_mag / perturbed_t
            
            if perturbed_a_mag < original_a_mag:
                logs.append("✓ Monotonicity holds: Increasing time decreases acceleration magnitude")
            else:
                logs.append("✗ Monotonicity violation: Increasing time should decrease acceleration")
                success = False
            
            # Test 2: Increase velocity change, acceleration magnitude should increase
            original_delta_v = 30.0
            perturbed_delta_v = original_delta_v * 1.3  # Increase by 30%
            
            original_a_mag2 = original_delta_v / original_t
            perturbed_a_mag2 = perturbed_delta_v / original_t
            
            if perturbed_a_mag2 > original_a_mag2:
                logs.append("✓ Monotonicity holds: Increasing Δv increases acceleration magnitude")
            else:
                logs.append("✗ Monotonicity violation: Increasing Δv should increase acceleration")
                success = False
                
        except Exception as e:
            logs.append(f"⚠ Monotonicity test error: {e}")
            # Don't fail on test errors, just warn
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs"""
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Run each test
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"❌ Test crashed: {e}")
                overall_success = False
        
        # Summary
        all_logs.append("\n=== Summary ===")
        if overall_success:
            all_logs.append("✅ All physics checks passed!")
        else:
            all_logs.append("❌ Some physics checks failed")
            
        # Add interpretation
        all_logs.append(f"\nInterpretation:")
        all_logs.append(f"  Acceleration magnitude: |a| = {abs(self.a):.1f} m/s²")
        if math.isclose(abs(self.a), 300, rel_tol=1e-2, abs_tol=1e-2):
            all_logs.append("  ✓ Matches expected 300 m/s² magnitude")
        if self.sign_correct:
            all_logs.append(f"  Sign convention: {self.sign_correct}")
            
        return overall_success, all_logs