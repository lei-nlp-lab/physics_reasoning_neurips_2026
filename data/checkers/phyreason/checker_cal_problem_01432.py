import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: extract all needed attributes without direct access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.h = getattr(solution_instance, 'h', 3.0)
        self.L = getattr(solution_instance, 'L', 6.0)
        self.m = getattr(solution_instance, 'm', 60.0)
        self.F_f = getattr(solution_instance, 'F_f', 240.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe fallbacks
        self.a = self.results.get('a', 0.0)
        self.t = self.results.get('t', 0.0)
        self.v = self.results.get('v', 0.0)
        
        # Constants for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def test_invariants(self):
        """
        Verify Newton's 2nd Law and Work-Energy Theorem based on KB reference.
        Returns: (bool, list_of_logs)
        """
        logs = []
        success = True
        
        try:
            # Test 1: Newton's 2nd Law (using step_1 equation from analysis)
            sin_theta = self.h / self.L  # sin(theta) = h/L
            F_gravity_parallel = self.m * self.g * sin_theta
            net_force = F_gravity_parallel - self.F_f
            expected_accel = net_force / self.m
            
            if not math.isclose(self.a, expected_accel, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"FAIL Newton's 2nd Law: a={self.a:.3f}, expected={expected_accel:.3f}")
                success = False
            else:
                logs.append(f"PASS Newton's 2nd Law: a={self.a:.3f} matches expected")
                
            # Test 2: Work-Energy Theorem (alternative verification)
            # Initial energy = m*g*h (potential)
            # Work done by friction = -F_f * L
            # Final kinetic energy = 0.5*m*v^2
            energy_initial = self.m * self.g * self.h
            work_friction = -self.F_f * self.L
            energy_final = 0.5 * self.m * self.v**2
            
            # Energy conservation with friction: ΔKE = W_gravity + W_friction
            delta_KE = energy_final - 0
            work_total = self.m * self.g * self.h - self.F_f * self.L
            
            if not math.isclose(delta_KE, work_total, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"FAIL Work-Energy: ΔKE={delta_KE:.3f}, W_total={work_total:.3f}")
                success = False
            else:
                logs.append(f"PASS Work-Energy: ΔKE matches total work")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Verify kinematic consistency between intermediate results.
        Checks: v = a*t and L = 0.5*a*t^2
        """
        logs = []
        success = True
        
        try:
            # Check v = a*t
            v_from_kinematics = self.a * self.t
            if not math.isclose(self.v, v_from_kinematics, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"FAIL v=a*t: v={self.v:.3f}, a*t={v_from_kinematics:.3f}")
                success = False
            else:
                logs.append(f"PASS v=a*t consistent")
                
            # Check L = 0.5*a*t^2
            L_from_kinematics = 0.5 * self.a * self.t**2
            if not math.isclose(self.L, L_from_kinematics, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"FAIL L=0.5*a*t^2: L={self.L:.3f}, computed={L_from_kinematics:.3f}")
                success = False
            else:
                logs.append(f"PASS L=0.5*a*t^2 consistent")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb mass and check if trends match expectations.
        Important: Restore original values after testing.
        """
        logs = []
        success = True
        
        # Store original values
        original_m = self.m
        original_a = self.a
        original_t = self.t
        original_v = self.v
        
        try:
            # Perturbation: increase mass by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected values using physics formulas
            sin_theta = self.h / self.L
            perturbed_a = self.g * sin_theta - self.F_f / perturbed_m
            perturbed_t = math.sqrt(2 * self.L / perturbed_a) if perturbed_a > 0 else 0
            perturbed_v = perturbed_a * perturbed_t
            
            # Expected trends based on formulas:
            # 1. Acceleration should INCREASE when mass increases (since F_f/m decreases)
            # 2. Time should DECREASE when acceleration increases (t = sqrt(2L/a))
            # 3. Velocity: v = a*t = sqrt(2L*a) → should increase if a increases
            
            # Check acceleration trend
            if not (perturbed_a > original_a):
                logs.append(f"FAIL monotonicity: a did not increase with mass increase")
                success = False
            else:
                logs.append(f"PASS acceleration increases with mass (expected)")
                
            # Check time trend
            if not (perturbed_t < original_t):
                logs.append(f"FAIL monotonicity: t did not decrease with a increase")
                success = False
            else:
                logs.append(f"PASS time decreases with acceleration increase")
                
            # Check velocity trend
            if not (perturbed_v > original_v):
                logs.append(f"WARNING monotonicity: v trend unexpected")
                logs.append(f"  v_original={original_v:.3f}, v_perturbed={perturbed_v:.3f}")
            else:
                logs.append(f"PASS velocity increases with acceleration increase")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        finally:
            # Restore original values
            self.m = original_m
            self.a = original_a
            self.t = original_t
            self.v = original_v
            
        return success, logs
    
    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"=== Running {test_name} Test ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRITICAL ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
            all_logs.append("")  # Empty line for readability
            
        return overall_success, all_logs