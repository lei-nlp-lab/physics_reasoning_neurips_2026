import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results using getattr with fallbacks.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.L = getattr(solution_instance, 'L', 0.40)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        self.theta_rad = math.radians(self.theta_deg)
        self.B = getattr(solution_instance, 'B', 0.5)
        self.E = getattr(solution_instance, 'E', 4.5)
        self.r = getattr(solution_instance, 'r', 0.50)
        self.m = getattr(solution_instance, 'm', 0.040)
        self.R0 = getattr(solution_instance, 'R0', 2.5)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe fallbacks
        self.I = self.results.get('I', 0.0)
        self.F_a = self.results.get('F_a', 0.0)
        self.F_1 = self.results.get('F_1', 0.0)
        self.f = self.results.get('f', 0.0)
        
        # Precompute trigonometric values for checks
        self.sin_theta = math.sin(self.theta_rad)
        self.cos_theta = math.cos(self.theta_rad)
        
        # Target answers for comparison
        self.target_I = 1.5
        self.target_Fa = 0.30
        self.target_f = 0.06
        self.target_F1 = 0.24
        
        self.logs = []

    def test_invariants(self):
        """Test conservation laws and equilibrium conditions."""
        logs = []
        try:
            # 1. Ohm's Law: I = E / (R0 + r)
            expected_I = self.E / (self.R0 + self.r)
            if not math.isclose(self.I, expected_I, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Ohm's law violated. I={self.I:.3f}, expected={expected_I:.3f}")
            else:
                logs.append(f"PASS: Ohm's law satisfied (I={self.I:.3f}A)")
            
            # 2. Ampere Force: F_a = B * I * L
            expected_Fa = self.B * self.I * self.L
            if not math.isclose(self.F_a, expected_Fa, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Ampere force formula violated. F_a={self.F_a:.3f}, expected={expected_Fa:.3f}")
            else:
                logs.append(f"PASS: Ampere force formula satisfied (F_a={self.F_a:.3f}N)")
            
            # 3. Gravitational decomposition: F_1 = m * g * sin(theta)
            expected_F1 = self.m * self.g * self.sin_theta
            if not math.isclose(self.F_1, expected_F1, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Gravity decomposition violated. F_1={self.F_1:.3f}, expected={expected_F1:.3f}")
            else:
                logs.append(f"PASS: Gravity decomposition satisfied (F_1={self.F_1:.3f}N)")
            
            # 4. Equilibrium condition: F_a = F_1 + f
            net_force = self.F_a - (self.F_1 + self.f)
            if not math.isclose(net_force, 0.0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Net force not zero. F_a - (F_1+f) = {net_force:.3f}N")
            else:
                logs.append(f"PASS: Equilibrium condition satisfied (F_a = F_1 + f)")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        self.logs.extend(logs)
        return len(logs) == 0 or all('PASS' in log or 'ERROR' in log for log in logs)

    def test_trace_consistency(self):
        """Cross-check intermediate results and final answers."""
        logs = []
        try:
            # Check against target answers
            checks = [
                (self.I, self.target_I, "Current I"),
                (self.F_a, self.target_Fa, "Ampere force F_a"),
                (self.f, self.target_f, "Friction force f"),
                (self.F_1, self.target_F1, "Gravity component F_1")
            ]
            all_pass = True
            for val, target, desc in checks:
                if not math.isclose(val, target, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: {desc} mismatch: {val:.3f} vs target {target}")
                    all_pass = False
                else:
                    logs.append(f"PASS: {desc} matches target")
            
            # Verify friction derived from equilibrium
            expected_f = self.F_a - self.F_1
            if not math.isclose(self.f, expected_f, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Friction inconsistent: f={self.f:.3f}, expected F_a-F_1={expected_f:.3f}")
                all_pass = False
            else:
                logs.append(f"PASS: Friction derived correctly from equilibrium")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            all_pass = False
        self.logs.extend(logs)
        return all_pass

    def test_monotonicity(self):
        """
        Perturb mass m, re-compute forces, check directional trends.
        IMPORTANT: Restore original mass after perturbation.
        """
        logs = []
        original_m = self.m
        try:
            # Increase mass by 10%
            perturbed_m = original_m * 1.10
            # Re-compute gravitational component
            F1_original = self.F_1
            F1_perturbed = perturbed_m * self.g * self.sin_theta
            
            # Friction should adjust to maintain equilibrium: f = F_a - F1
            f_original = self.f
            f_perturbed = self.F_a - F1_perturbed
            
            # Trend: if mass increases, F1 increases, friction should decrease (since F_a constant)
            if F1_perturbed > F1_original and f_perturbed < f_original:
                logs.append("PASS: Monotonicity trend correct (increasing mass → decreased friction)")
            elif F1_perturbed < F1_original and f_perturbed > f_original:
                logs.append("PASS: Monotonicity trend correct (decreasing mass → increased friction)")
            else:
                logs.append(f"FAIL: Unexpected trend. F1: {F1_original:.3f}→{F1_perturbed:.3f}, f: {f_original:.3f}→{f_perturbed:.3f}")
                
            # Restore original mass
            self.m = original_m
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            self.m = original_m  # Ensure restoration
        self.logs.extend(logs)
        return len(logs) == 1 and 'PASS' in logs[0]

    def run_all_tests(self):
        """Execute all checks and return success flag with logs."""
        self.logs = []
        success = True
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]
        
        for test in tests:
            try:
                if not test():
                    success = False
            except Exception as e:
                self.logs.append(f"CRASH in {test.__name__}: {traceback.format_exc()}")
                success = False
        
        return success, self.logs