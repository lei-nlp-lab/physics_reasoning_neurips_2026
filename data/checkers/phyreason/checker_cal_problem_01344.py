import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Use getattr with fallback to .results dictionary to avoid AttributeError.
        """
        self.solution = solution_instance
        self.logs = []
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 1.8e4))
        self.x_AB = 180.0  # meters
        self.x_BC = 50.0   # meters
        self.theta = 15.0  # degrees
        self.sin_theta = 0.3
        self.g = 10.0      # m/s²
        self.v_C = getattr(solution_instance, 'v_C', getattr(solution_instance, 'results', {}).get('v_C', 100.0))
        self.v_B_min = 0.6 * self.v_C  # minimum speed at B
        
        # Extract computed results safely
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 0.0))
        self.P = getattr(solution_instance, 'P', getattr(solution_instance, 'results', {}).get('P', 0.0))
        self.t_total = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.t1 = getattr(solution_instance, 't_1', getattr(solution_instance, 'results', {}).get('t_1', 0.0))
        self.t2 = self.t_total - self.t1 if self.t_total > 0 else 0.0
        
        # Extract intermediate work/energy terms
        self.FxAB = getattr(solution_instance, 'F*x_AB', getattr(solution_instance, 'results', {}).get('F*x_AB', 0.0))
        self.Pt2 = getattr(solution_instance, 'P*t_2', getattr(solution_instance, 'results', {}).get('P*t_2', 0.0))
        
    def log(self, msg, success=None):
        """Helper to log messages and test results."""
        self.logs.append(msg)
        if success is not None:
            self.logs[-1] += f" - {'PASS' if success else 'FAIL'}"
    
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics laws: work-energy theorem and Newton's 2nd Law."""
        try:
            # 1. Work-energy theorem for segment AB (horizontal, no friction/gravity work)
            #    W_AB = ΔK_AB = 0.5*m*v_B^2 - 0 (assuming starts from rest at A)
            W_AB_calc = 0.5 * self.m * self.v_B_min**2
            if self.fuzzy_equal(self.FxAB, W_AB_calc):
                self.log(f"Work-energy theorem for AB: F*x_AB = {self.FxAB:.2e} J, ΔK = {W_AB_calc:.2e} J", True)
            else:
                self.log(f"Work-energy theorem for AB mismatch: F*x_AB = {self.FxAB:.2e} J, expected ΔK = {W_AB_calc:.2e} J", False)
                return False
            
            # 2. Work-energy theorem for segment BC (inclined, with gravity and power)
            #    W_net = ΔK_BC = 0.5*m*v_C^2 - 0.5*m*v_B^2
            #    W_net = P*t_2 - m*g*sinθ*x_BC
            ΔK_BC = 0.5 * self.m * (self.v_C**2 - self.v_B_min**2)
            W_net_BC = self.Pt2 - self.m * self.g * self.sin_theta * self.x_BC
            if self.fuzzy_equal(W_net_BC, ΔK_BC):
                self.log(f"Work-energy theorem for BC: W_net = {W_net_BC:.2e} J, ΔK = {ΔK_BC:.2e} J", True)
            else:
                self.log(f"Work-energy theorem for BC mismatch: W_net = {W_net_BC:.2e} J, expected ΔK = {ΔK_BC:.2e} J", False)
                return False
            
            # 3. Newton's 2nd Law on horizontal segment (constant force F, mass m)
            #    a_AB = F/m
            #    From kinematics: v_B^2 = 2*a_AB*x_AB  (starting from rest)
            a_AB = self.F / self.m
            v_B_from_kinematics = math.sqrt(2 * a_AB * self.x_AB)
            if self.fuzzy_equal(v_B_from_kinematics, self.v_B_min):
                self.log(f"Newton's 2nd Law consistency for AB: v_B from kinematics = {v_B_from_kinematics:.2f} m/s, required v_B = {self.v_B_min:.2f} m/s", True)
            else:
                self.log(f"Newton's 2nd Law inconsistency for AB: v_B from kinematics = {v_B_from_kinematics:.2f} m/s, required v_B = {self.v_B_min:.2f} m/s", False)
                return False
            
            return True
        except Exception as e:
            self.log(f"Error in test_invariants: {e}\n{traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Force F should match: F = (0.5*m*v_B^2) / x_AB
            F_expected = (0.5 * self.m * self.v_B_min**2) / self.x_AB
            if self.fuzzy_equal(self.F, F_expected):
                self.log(f"Force F consistency: F = {self.F:.2e} N, expected = {F_expected:.2e} N", True)
            else:
                self.log(f"Force F mismatch: F = {self.F:.2e} N, expected = {F_expected:.2e} N", False)
                return False
            
            # 2. Power P should match: P = F * v_B (maximum at B)
            P_expected = self.F * self.v_B_min
            if self.fuzzy_equal(self.P, P_expected):
                self.log(f"Power P consistency: P = {self.P:.2e} W, expected = {P_expected:.2e} W", True)
            else:
                self.log(f"Power P mismatch: P = {self.P:.2e} W, expected = {P_expected:.2e} W", False)
                return False
            
            # 3. Time t1 should match: t1 = 2*x_AB / v_B (for uniform acceleration from rest)
            t1_expected = 2 * self.x_AB / self.v_B_min
            if self.fuzzy_equal(self.t1, t1_expected):
                self.log(f"Time t1 consistency: t1 = {self.t1:.2f} s, expected = {t1_expected:.2f} s", True)
            else:
                self.log(f"Time t1 mismatch: t1 = {self.t1:.2f} s, expected = {t1_expected:.2f} s", False)
                return False
            
            # 4. Total time t should satisfy energy equation for BC
            #    From work-energy: P*t2 = ΔK_BC + m*g*sinθ*x_BC
            #    Compute ΔK_BC locally
            ΔK_BC = 0.5 * self.m * (self.v_C**2 - self.v_B_min**2)
            t2_from_energy = (ΔK_BC + self.m * self.g * self.sin_theta * self.x_BC) / self.P
            t_total_expected = self.t1 + t2_from_energy
            if self.fuzzy_equal(self.t_total, t_total_expected):
                self.log(f"Total time t consistency: t = {self.t_total:.2f} s, expected = {t_total_expected:.2f} s", True)
            else:
                self.log(f"Total time t mismatch: t = {self.t_total:.2f} s, expected = {t_total_expected:.2f} s", False)
                return False
            
            return True
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input parameter, re-calculate using physics formulas, 
        check expected trend, and restore original values.
        """
        original_m = self.m
        try:
            # Increase mass by 10% and check that required force F increases
            self.m *= 1.10
            v_B_min = 0.6 * self.v_C  # recompute with same v_C
            F_new = (0.5 * self.m * v_B_min**2) / self.x_AB
            
            if F_new > self.F:
                self.log(f"Monotonicity: Increasing mass by 10% increases force F (new F = {F_new:.2e} N > original {self.F:.2e} N)", True)
            else:
                self.log(f"Monotonicity violation: Increased mass but force did not increase (new F = {F_new:.2e} N, original {self.F:.2e} N)", False)
                self.m = original_m
                return False
            
            # Restore original mass
            self.m = original_m
            return True
        except Exception as e:
            self.m = original_m
            self.log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        self.logs = []
        success = True
        
        # Run each test, accumulate results
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} failed", False)
            except Exception as e:
                success = False
                self.log(f"{test_name} crashed: {e}\n{traceback.format_exc()}", False)
        
        # Final summary
        self.log(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs