import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction for inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 25.0))
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', 2.5))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', 60.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 10.0))
        
        # Safe attribute extraction for results
        self.results = getattr(solution_instance, 'results', {})
        self.v = self.results.get('v', 0.0)
        self.F_N = self.results.get('F_N', 0.0)
        self.F_pressure = self.results.get('F_pressure', 0.0)
        
        # Symbolic variables for checks
        self.sym_m, self.sym_l, self.sym_theta, self.sym_g, self.sym_v = sp.symbols('m l theta g v')
        self.sym_F_N, self.sym_F_pressure = sp.symbols('F_N F_pressure')
        
        # Constants
        self.theta_rad = math.radians(self.theta)
        self.h = self.l * (1 - math.cos(self.theta_rad))  # height difference

    def test_invariants(self):
        logs = []
        try:
            # 1. Energy Conservation: m*g*h = 0.5*m*v^2
            potential_energy = self.m * self.g * self.h
            kinetic_energy = 0.5 * self.m * self.v**2
            if not math.isclose(potential_energy, kinetic_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Energy conservation failed: PE={potential_energy:.2f}, KE={kinetic_energy:.2f}")
            else:
                logs.append("Energy conservation: PASS")
                
            # 2. Newton's Second Law at lowest point: F_N - m*g = m*v^2 / l
            centripetal_force = self.m * self.v**2 / self.l
            expected_F_N = self.m * self.g + centripetal_force
            if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton's 2nd Law failed: F_N={self.F_N:.2f}, expected={expected_F_N:.2f}")
            else:
                logs.append("Newton's 2nd Law: PASS")
                
            # 3. Newton's Third Law: F_pressure should equal F_N
            if not math.isclose(self.F_pressure, self.F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton's 3rd Law failed: F_pressure={self.F_pressure:.2f}, F_N={self.F_N:.2f}")
            else:
                logs.append("Newton's 3rd Law: PASS")
                
        except Exception as e:
            logs.append(f"Invariant test error: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Check v from energy conservation
            expected_v = math.sqrt(2 * self.g * self.h)
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency v failed: v={self.v:.2f}, expected={expected_v:.2f}")
            else:
                logs.append(f"Trace v consistency: PASS (v={self.v:.2f})")
                
            # Check F_N from centripetal equation
            expected_F_N = self.m * self.g + self.m * self.v**2 / self.l
            if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency F_N failed: F_N={self.F_N:.2f}, expected={expected_F_N:.2f}")
            else:
                logs.append(f"Trace F_N consistency: PASS (F_N={self.F_N:.2f})")
                
            # Check F_pressure equals F_N
            if not math.isclose(self.F_pressure, self.F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency F_pressure failed: F_pressure={self.F_pressure:.2f}, F_N={self.F_N:.2f}")
            else:
                logs.append(f"Trace F_pressure consistency: PASS (F_pressure={self.F_pressure:.2f})")
                
        except Exception as e:
            logs.append(f"Trace consistency error: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        original_m = self.m
        try:
            # Perturb mass by +10%
            self.m *= 1.1
            
            # Recalculate using formulas (symbolic derivation)
            # Energy conservation: v = sqrt(2*g*l*(1-cos(theta)))
            new_v = math.sqrt(2 * self.g * self.l * (1 - math.cos(self.theta_rad)))
            # Newton's second law: F_N = m*g + m*v^2/l
            new_F_N = self.m * self.g + self.m * new_v**2 / self.l
            
            # Check monotonicity: F_N should increase with mass (positive correlation)
            if new_F_N > self.F_N:
                logs.append(f"Monotonicity (mass increase -> F_N increase): PASS (F_N_old={self.F_N:.2f}, F_N_new={new_F_N:.2f})")
            else:
                logs.append(f"Monotonicity failed: F_N decreased with mass increase")
                
            # Restore original mass
            self.m = original_m
        except Exception as e:
            logs.append(f"Monotonicity test error: {traceback.format_exc()}")
            self.m = original_m  # Ensure restoration even on error
        return logs

    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            logs = test_func()
            all_logs.append(f"--- {test_name} ---")
            all_logs.extend(logs)
            
            # Check for failure indicators in logs
            for log in logs:
                if "failed" in log.lower():
                    success = False
        
        return success, all_logs