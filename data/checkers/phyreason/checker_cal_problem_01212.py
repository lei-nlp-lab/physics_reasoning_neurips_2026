import math
import sympy as sp
from sympy import symbols, Piecewise, Eq

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.g = 10.0
        self.dt = 1.2e-3
        self.lethal_g = 500.0
        
        # Extract results safely
        self.v1 = getattr(solution_instance, 'v_1', 
                         solution_instance.results.get('v_1', 0.0))
        self.a = getattr(solution_instance, 'a', 
                        solution_instance.results.get('a', 0.0))
        self.a_l = getattr(solution_instance, 'a_l', 
                          solution_instance.results.get('a_l', 0.0))
        
        # Extract any other needed attributes
        self.solution_instance = solution_instance
        
    def test_invariants(self):
        logs = []
        success = True
        
        # 1. Check velocity conversion (36 km/h = 10 m/s)
        expected_v1 = 36 * (1000/3600)  # 10 m/s
        if not math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Velocity conversion failed: v1={self.v1}, expected={expected_v1}")
            success = False
        else:
            logs.append("✓ Velocity conversion correct")
        
        # 2. Check lethal acceleration calculation (500g)
        expected_a_l = self.lethal_g * self.g
        if not math.isclose(self.a_l, expected_a_l, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Lethal acceleration calculation failed: a_l={self.a_l}, expected={expected_a_l}")
            success = False
        else:
            logs.append("✓ Lethal acceleration calculation correct")
        
        # 3. Check average acceleration formula: a = (v2 - v1)/dt
        # For head-on collision: v2 = -v1 (same magnitude, opposite direction)
        v2 = -self.v1
        delta_v = v2 - self.v1
        expected_a = delta_v / self.dt
        
        if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Average acceleration calculation failed: a={self.a}, expected={expected_a}")
            logs.append(f"  delta_v={delta_v}, dt={self.dt}")
            success = False
        else:
            logs.append("✓ Average acceleration calculation correct")
        
        # 4. Check lethality condition: |a| > a_l
        is_lethal = abs(self.a) > self.a_l
        if not is_lethal:
            logs.append(f"Lethality condition failed: |a|={abs(self.a)} ≤ a_l={self.a_l}")
            success = False
        else:
            logs.append("✓ Lethality condition satisfied (acceleration exceeds lethal threshold)")
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check intermediate results consistency
        # The calculation should be traceable: v1 → a → comparison with a_l
        
        # Step 1: Recalculate a from v1
        v2 = -self.v1
        recalculated_a = (v2 - self.v1) / self.dt
        
        if not math.isclose(self.a, recalculated_a, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency failed: a={self.a}, recalculated={recalculated_a}")
            success = False
        else:
            logs.append("✓ Acceleration trace consistent")
        
        # Step 2: Recalculate a_l from g
        recalculated_a_l = self.lethal_g * self.g
        if not math.isclose(self.a_l, recalculated_a_l, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Lethal acceleration trace failed: a_l={self.a_l}, recalculated={recalculated_a_l}")
            success = False
        else:
            logs.append("✓ Lethal acceleration trace consistent")
        
        # Step 3: Check final conclusion matches calculated values
        is_lethal_calculated = abs(self.a) > self.a_l
        logs.append(f"Calculated: |a|={abs(self.a):.1f} {'>' if is_lethal_calculated else '≤'} a_l={self.a_l}")
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        # Store original values
        original_v1 = self.v1
        original_a = self.a
        
        # Test 1: Increase velocity by 10%
        perturbed_v1 = original_v1 * 1.1
        v2 = -perturbed_v1
        perturbed_a = (v2 - perturbed_v1) / self.dt
        
        # |a| should increase with increased velocity
        if abs(perturbed_a) <= abs(original_a):
            logs.append(f"Monotonicity test 1 failed: |a|={abs(perturbed_a)} ≤ {abs(original_a)} when v1 increased")
            success = False
        else:
            logs.append("✓ Monotonicity test 1 passed: |a| increases with increased velocity")
        
        # Test 2: Decrease collision time by 10%
        perturbed_dt = self.dt * 0.9
        perturbed_a = (v2 - original_v1) / perturbed_dt
        
        # |a| should increase with decreased collision time
        if abs(perturbed_a) <= abs(original_a):
            logs.append(f"Monotonicity test 2 failed: |a|={abs(perturbed_a)} ≤ {abs(original_a)} when dt decreased")
            success = False
        else:
            logs.append("✓ Monotonicity test 2 passed: |a| increases with decreased collision time")
        
        # Restore original values
        self.v1 = original_v1
        self.a = original_a
        
        return success, logs
    
    def test_energy_conservation(self):
        """For collision problems, check if momentum/energy considerations are reasonable"""
        logs = []
        success = True
        
        # For perfectly inelastic collision, kinetic energy is not conserved
        # But we can check that the deceleration makes physical sense
        
        # Calculate stopping distance if constant deceleration
        # v_f^2 = v_i^2 + 2*a*d  => d = -v_i^2/(2*a)
        d = -(self.v1**2) / (2 * self.a)
        
        if d < 0:
            logs.append(f"Stopping distance negative: d={d}")
            success = False
        elif d > 1.0:  # Unrealistically large stopping distance for such high acceleration
            logs.append(f"Stopping distance unusually large: d={d} m")
            success = False
        else:
            logs.append(f"✓ Stopping distance reasonable: d={d:.4f} m")
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = ["=== Physics Checker ==="]
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Energy Conservation", self.test_energy_conservation)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} Test ---")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"✗ {test_name} test failed")
                else:
                    all_logs.append(f"✓ {test_name} test passed")
            except Exception as e:
                all_logs.append(f"✗ {test_name} test error: {str(e)}")
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== Summary: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===")
        
        return overall_success, all_logs