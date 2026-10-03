import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results with fallback to empty dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results using mapping
        self.Deltav_mag = self.results.get('|Deltav|', 0.0)  # Magnitude of velocity change
        self.a1 = self.results.get('a_1', 0.0)  # Acceleration during kick
        self.Deltav = self.results.get('Deltav', 0.0)  # Velocity change (signed)
        self.t = self.results.get('t', 0.0)  # Time in air
        self.a2 = self.results.get('a_2', 0.0)  # Acceleration during catch
        
        # Problem constants from context
        self.v0 = 0.0  # Initial velocity (ball at rest)
        self.v1 = 30.0  # Velocity after kick
        self.v2 = -10.0  # Velocity after catch (negative for opposite direction)
        self.t1 = 0.1  # Contact time during kick
        self.t2 = 0.1  # Contact time during catch
        self.s = 12.0  # Distance to goalkeeper
        
        # Expected values from steps analysis
        self.expected_a1 = 300.0
        self.expected_t = 0.4
        self.expected_Deltav = -40.0
        self.expected_Deltav_mag = 40.0
        self.expected_a2 = -400.0

    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Check acceleration during kick: a1 = (v1 - v0)/t1
            calculated_a1 = (self.v1 - self.v0) / self.t1
            if not math.isclose(self.a1, calculated_a1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: a1 mismatch. Got {self.a1}, expected {calculated_a1}")
                success = False
            else:
                logs.append(f"PASS: a1 = {self.a1} matches calculation")
            
            # 2. Check time in air: t = s / v1
            calculated_t = self.s / self.v1
            if not math.isclose(self.t, calculated_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t mismatch. Got {self.t}, expected {calculated_t}")
                success = False
            else:
                logs.append(f"PASS: t = {self.t} matches calculation")
            
            # 3. Check velocity change: Δv = v2 - v1
            calculated_Deltav = self.v2 - self.v1
            if not math.isclose(self.Deltav, calculated_Deltav, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δv mismatch. Got {self.Deltav}, expected {calculated_Deltav}")
                success = False
            else:
                logs.append(f"PASS: Δv = {self.Deltav} matches calculation")
            
            # 4. Check magnitude of velocity change: |Δv| = |v2 - v1|
            calculated_Deltav_mag = abs(calculated_Deltav)
            if not math.isclose(self.Deltav_mag, calculated_Deltav_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: |Δv| mismatch. Got {self.Deltav_mag}, expected {calculated_Deltav_mag}")
                success = False
            else:
                logs.append(f"PASS: |Δv| = {self.Deltav_mag} matches calculation")
            
            # 5. Check acceleration during catch: a2 = (v2 - v1)/t2
            calculated_a2 = (self.v2 - self.v1) / self.t2
            if not math.isclose(self.a2, calculated_a2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: a2 mismatch. Got {self.a2}, expected {calculated_a2}")
                success = False
            else:
                logs.append(f"PASS: a2 = {self.a2} matches calculation")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Compare with expected values from steps analysis
            checks = [
                ("a1", self.a1, self.expected_a1),
                ("t", self.t, self.expected_t),
                ("Δv", self.Deltav, self.expected_Deltav),
                ("|Δv|", self.Deltav_mag, self.expected_Deltav_mag),
                ("a2", self.a2, self.expected_a2)
            ]
            
            for name, actual, expected in checks:
                if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: {name} = {actual} doesn't match expected {expected}")
                    success = False
                else:
                    logs.append(f"PASS: {name} = {actual} matches expected value")
                    
            # Cross-check: a2 should have same magnitude as |Δv|/t2
            expected_a2_mag = self.expected_Deltav_mag / self.t2
            if not math.isclose(abs(self.a2), expected_a2_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: |a2| = {abs(self.a2)} doesn't match |Δv|/t2 = {expected_a2_mag}")
                success = False
            else:
                logs.append(f"PASS: |a2| = {abs(self.a2)} consistent with |Δv|/t2")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb the distance s (increase by 10%)
            original_s = self.s
            perturbed_s = original_s * 1.1
            
            # Calculate time with perturbed distance
            original_t = original_s / self.v1
            perturbed_t = perturbed_s / self.v1
            
            # Check monotonicity: larger distance should give larger time
            if perturbed_t <= original_t:
                logs.append(f"FAIL: Monotonicity violated. t({original_s}) = {original_t}, t({perturbed_s}) = {perturbed_t}")
                success = False
            else:
                logs.append(f"PASS: Monotonicity holds. t({original_s}) = {original_t}, t({perturbed_s}) = {perturbed_t}")
            
            # Restore original value (for completeness)
            self.s = original_s
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRITICAL: Test {test_name} crashed: {str(e)}")
                overall_success = False
        
        return overall_success, all_logs