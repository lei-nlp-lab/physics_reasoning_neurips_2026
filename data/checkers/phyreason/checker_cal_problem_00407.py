import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract all values from results dictionary
        self.t1 = solution_instance.results.get('t_1', None)
        self.B = solution_instance.results.get('B', None)
        self.E = solution_instance.results.get('E', None)
        self.v0 = solution_instance.results.get('v0', None)
        self.t0 = solution_instance.results.get('t0', None)
        # Constants
        self.sin30 = 0.5  # sin(30°)
        self.sqrt3 = math.sqrt(3)
        self.pi = math.pi

    def test_invariants(self):
        logs = []
        try:
            # All values must be numeric
            if not all(isinstance(x, (int, float)) for x in [self.E, self.v0, self.t1, self.t0, self.B]):
                logs.append("Skipping invariants test: some values are not numeric.")
                return False, logs
            
            # 1. Kinematics: (v0/2)*t1 = d_AB * sin30
            # Solve for d_AB
            d_AB = (self.v0 * self.t1) / (2 * self.sin30)
            logs.append(f"d_AB from kinematics: {d_AB}")
            
            # 2. Work-Energy: q*E*d_AB*sin30 = 0.5*m*v0^2
            # Assume m=1, q=1 for numeric check (they cancel in ratio)
            work_lhs = self.E * d_AB * self.sin30
            work_rhs = 0.5 * 1.0 * self.v0**2
            if math.isclose(work_lhs, work_rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Work-Energy holds: {work_lhs} ≈ {work_rhs}")
            else:
                logs.append(f"✗ Work-Energy mismatch: {work_lhs} vs {work_rhs}")
                return False, logs
            
            # 3. Centripetal force: q*v0*B = m*v0^2/R -> R = m*v0/(q*B)
            # Assume m=1, q=1
            R = self.v0 / self.B
            logs.append(f"Radius R: {R}")
            
            # 4. Geometry: chord length AB = 2R sin(φ/2)
            # We don't have φ, but we can check consistency with d_AB
            # From geometry: d_AB = AB (since d_AB is distance between A and B)
            # Actually d_AB computed above is the straight-line distance AB.
            # The chord equation should hold: d_AB = 2R sin(φ/2)
            # We can compute sin(φ/2) = d_AB/(2R)
            sin_half_phi = d_AB / (2 * R)
            logs.append(f"sin(φ/2) = {sin_half_phi}")
            # Expected sin(φ/2) = √3/2 ≈ 0.866025 (for φ=2π/3 or 4π/3)
            if math.isclose(sin_half_phi, self.sqrt3/2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Geometry consistent: sin(φ/2) ≈ √3/2")
            else:
                logs.append(f"✗ Geometry mismatch: sin(φ/2) = {sin_half_phi}, expected {self.sqrt3/2}")
                return False, logs
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
            return False, logs
        return True, logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Check that t1 = (3√3/(4π)) * t0
            expected_t1 = (3 * self.sqrt3) / (4 * self.pi) * self.t0
            if math.isclose(self.t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ t1 matches expected: {self.t1} ≈ {expected_t1}")
            else:
                logs.append(f"✗ t1 mismatch: {self.t1} vs {expected_t1}")
                return False, logs
            
            # Check B = √3 * E / v0
            expected_B = self.sqrt3 * self.E / self.v0
            if math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ B matches expected: {self.B} ≈ {expected_B}")
            else:
                logs.append(f"✗ B mismatch: {self.B} vs {expected_B}")
                return False, logs
                    
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False, logs
        return True, logs

    def test_monotonicity(self):
        logs = []
        try:
            # Perturb t0 and check t1 changes proportionally
            perturbed_t0 = self.t0 * 1.01  # +1%
            original_ratio = self.t1 / self.t0
            expected_t1_perturbed = (3 * self.sqrt3) / (4 * self.pi) * perturbed_t0
            perturbed_ratio = expected_t1_perturbed / perturbed_t0
            if math.isclose(original_ratio, perturbed_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ t1/t0 ratio constant under perturbation: {original_ratio} ≈ {perturbed_ratio}")
            else:
                logs.append(f"✗ t1/t0 ratio changed: {original_ratio} vs {perturbed_ratio}")
                return False, logs
            
            # Perturb E and check B changes proportionally (v0 fixed)
            perturbed_E = self.E * 1.01
            original_B = self.B
            expected_B_perturbed = self.sqrt3 * perturbed_E / self.v0
            if math.isclose(expected_B_perturbed / perturbed_E, original_B / self.E, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ B/E ratio constant under perturbation")
            else:
                logs.append(f"✗ B/E ratio changed")
                return False, logs
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            return False, logs
        return True, logs

    def run_all_tests(self):
        all_logs = []
        success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"--- {test_name} ---")
            try:
                test_success, logs = test_func()
                all_logs.extend(logs)
                if not test_success:
                    success = False
                    all_logs.append(f"{test_name} FAILED.")
                else:
                    all_logs.append(f"{test_name} passed.")
            except Exception as e:
                all_logs.append(f"Error during {test_name}: {traceback.format_exc()}")
                success = False
            all_logs.append("")
        
        return success, all_logs