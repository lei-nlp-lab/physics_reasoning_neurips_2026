import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.m = getattr(solution_instance, 'm', 1.0)  # kg
        self.M = getattr(solution_instance, 'M', 9.0)  # kg
        self.theta = getattr(solution_instance, 'theta', 37.0)  # degrees
        self.F_buoyancy = getattr(solution_instance, 'F_buoyancy', 18.0)  # N
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²
        
        # Trigonometric values
        self.sin_th = math.sin(math.radians(self.theta))
        self.cos_th = math.cos(math.radians(self.theta))
        
        # Safe extraction of results (with fallback to 0.0)
        self.results = getattr(solution_instance, 'results', {})
        self.T = self.results.get('T', 0.0)
        self.F_wind = self.results.get('F_wind', 0.0)
        self.N_ground = self.results.get('N_ground', 0.0)
        self.f_friction = self.results.get('f_friction', 0.0)
        
        # Log storage
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append({"test": message, "passed": success})
        
    def test_invariants(self):
        """Check static equilibrium conditions for balloon and box"""
        try:
            # 1. Balloon horizontal equilibrium: F_wind = T * sinθ
            lhs_bal_x = self.F_wind
            rhs_bal_x = self.T * self.sin_th
            bal_x_ok = math.isclose(lhs_bal_x, rhs_bal_x, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Balloon horizontal: F_wind ({lhs_bal_x:.3f}) = T*sinθ ({rhs_bal_x:.3f})", bal_x_ok)
            
            # 2. Balloon vertical equilibrium: F_buoyancy = m*g + T*cosθ
            lhs_bal_y = self.F_buoyancy
            rhs_bal_y = self.m * self.g + self.T * self.cos_th
            bal_y_ok = math.isclose(lhs_bal_y, rhs_bal_y, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Balloon vertical: F_buoyancy ({lhs_bal_y:.3f}) = m*g + T*cosθ ({rhs_bal_y:.3f})", bal_y_ok)
            
            # 3. Box horizontal equilibrium: f_friction = T * sinθ
            lhs_box_x = self.f_friction
            rhs_box_x = self.T * self.sin_th
            box_x_ok = math.isclose(lhs_box_x, rhs_box_x, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Box horizontal: f_friction ({lhs_box_x:.3f}) = T*sinθ ({rhs_box_x:.3f})", box_x_ok)
            
            # 4. Box vertical equilibrium: M*g = N_ground + T*cosθ
            lhs_box_y = self.M * self.g
            rhs_box_y = self.N_ground + self.T * self.cos_th
            box_y_ok = math.isclose(lhs_box_y, rhs_box_y, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Box vertical: M*g ({lhs_box_y:.3f}) = N_ground + T*cosθ ({rhs_box_y:.3f})", box_y_ok)
            
            return all([bal_x_ok, bal_y_ok, box_x_ok, box_y_ok])
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
            
    def test_trace_consistency(self):
        """Check that intermediate calculations are consistent with final answers"""
        try:
            # From Step 3: F_wind and T should satisfy balloon equilibria
            # We already checked these in test_invariants, but verify numerical values match expectations
            T_expected = (self.F_buoyancy - self.m * self.g) / self.cos_th
            F_wind_expected = T_expected * self.sin_th
            
            # Check if computed T and F_wind are close to expected
            T_ok = math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2)
            F_wind_ok = math.isclose(self.F_wind, F_wind_expected, rel_tol=1e-2, abs_tol=1e-2)
            
            self._log(f"T consistency: computed {self.T:.3f} ≈ expected {T_expected:.3f}", T_ok)
            self._log(f"F_wind consistency: computed {self.F_wind:.3f} ≈ expected {F_wind_expected:.3f}", F_wind_ok)
            
            # From Steps 4-6: Check box forces
            f_expected = T_expected * self.sin_th
            N_expected = self.M * self.g - T_expected * self.cos_th
            
            f_ok = math.isclose(self.f_friction, f_expected, rel_tol=1e-2, abs_tol=1e-2)
            N_ok = math.isclose(self.N_ground, N_expected, rel_tol=1e-2, abs_tol=1e-2)
            
            self._log(f"f_friction consistency: computed {self.f_friction:.3f} ≈ expected {f_expected:.3f}", f_ok)
            self._log(f"N_ground consistency: computed {self.N_ground:.3f} ≈ expected {N_expected:.3f}", N_ok)
            
            return all([T_ok, F_wind_ok, f_ok, N_ok])
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
            
    def test_monotonicity(self):
        """Perturb mass m and check physical trends"""
        try:
            # Store original values
            m_original = self.m
            T_original = self.T
            F_wind_original = self.F_wind
            N_ground_original = self.N_ground
            f_friction_original = self.f_friction
            
            # Perturb m by +10%
            self.m *= 1.1
            
            # Recalculate expected values using physics equations
            T_new = (self.F_buoyancy - self.m * self.g) / self.cos_th
            F_wind_new = T_new * self.sin_th
            N_ground_new = self.M * self.g - T_new * self.cos_th
            f_friction_new = T_new * self.sin_th
            
            # Check monotonic trends:
            # As m increases, T should decrease (since buoyancy is fixed)
            trend1 = T_new < T_original or math.isclose(T_new, T_original, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity: T decreases when m increases (new: {T_new:.3f}, orig: {T_original:.3f})", trend1)
            
            # F_wind should also decrease (proportional to T)
            trend2 = F_wind_new < F_wind_original or math.isclose(F_wind_new, F_wind_original, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity: F_wind decreases when m increases (new: {F_wind_new:.3f}, orig: {F_wind_original:.3f})", trend2)
            
            # N_ground should increase (less upward pull from rope)
            trend3 = N_ground_new > N_ground_original or math.isclose(N_ground_new, N_ground_original, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity: N_ground increases when m increases (new: {N_ground_new:.3f}, orig: {N_ground_original:.3f})", trend3)
            
            # f_friction should decrease (proportional to T)
            trend4 = f_friction_new < f_friction_original or math.isclose(f_friction_new, f_friction_original, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Monotonicity: f_friction decreases when m increases (new: {f_friction_new:.3f}, orig: {f_friction_original:.3f})", trend4)
            
            # Restore original values
            self.m = m_original
            
            return all([trend1, trend2, trend3, trend4])
            
        except Exception as e:
            # Restore original m on error
            if 'm_original' in locals():
                self.m = m_original
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
            
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []  # Reset logs
        
        tests = [
            ("Static Equilibrium Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append(success)
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                results.append(False)
        
        overall_success = all(results)
        return overall_success, self.logs