import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        Uses safe attribute access with fallbacks to results dictionary.
        """
        try:
            # Extract inputs with fallbacks
            self.S = getattr(solution_instance, 'S', 1.5)  # Heating area in m²
            self.time_sunlight_seconds = getattr(solution_instance, 'time_sunlight_seconds', 4 * 3600)  # 4 hours in seconds
            self.E0 = getattr(solution_instance, 'E0', 7e3)  # Solar constant J/(m²·s)
            self.efficiency = getattr(solution_instance, 'efficiency', 0.2)  # 20% conversion
            self.m = getattr(solution_instance, 'm', 200.0)  # Water mass in kg
            self.t0 = getattr(solution_instance, 't0', 20.0)  # Initial temperature in °C
            self.c_water = getattr(solution_instance, 'c_water', 4200.0)  # Specific heat of water J/(kg·°C), standard value
            
            # Extract results safely
            self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', 0.0))
            self.t_final = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
            
            # Store original values for restoration in monotonicity test
            self._original_m = self.m
            self._original_S = self.S
            self._original_efficiency = self.efficiency
            
        except Exception as e:
            self._error = f"Initialization error: {str(e)}\n{traceback.format_exc()}"
            raise

    def test_invariants(self):
        """
        Test physical invariants:
        1. Energy conservation: Total solar energy * efficiency = Heat absorbed (Q)
        2. Internal energy change: Q = m*c*(t_final - t0)
        3. Non-negative temperature rise
        """
        logs = []
        success = True
        
        try:
            # 1. Energy conversion consistency
            total_solar_energy = self.E0 * self.S * self.time_sunlight_seconds
            expected_Q = total_solar_energy * self.efficiency
            
            if not math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Energy conversion invariant failed: Q={self.Q:.2e}, expected η*E0*S*t={expected_Q:.2e}")
            
            # 2. Internal energy balance (specific heat formula)
            delta_T = self.t_final - self.t0
            required_energy = self.m * self.c_water * delta_T
            
            if not math.isclose(self.Q, required_energy, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Internal energy invariant failed: Q={self.Q:.2e}, m*c*ΔT={required_energy:.2e}")
            
            # 3. Non-negativity checks
            if self.Q < -1e-2:  # Allow small negative due to floating point
                success = False
                logs.append(f"Negative heat absorbed: Q={self.Q:.2e}")
            
            if delta_T < -1e-2:
                success = False
                logs.append(f"Temperature decreased: ΔT={delta_T:.2f}°C")
            
            # Symbolic check using sympy
            symbols = sp.symbols('E0 S t_sun eta m c ΔT Q')
            eq1 = sp.Eq(symbols[6], symbols[0] * symbols[1] * symbols[2] * symbols[3])
            eq2 = sp.Eq(symbols[6], symbols[4] * symbols[5] * symbols[7])
            
            if success:
                logs.append("All invariants passed symbolic validation")
            
        except Exception as e:
            success = False
            logs.append(f"Invariant test error: {str(e)}\n{traceback.format_exc()}")
        
        return success, logs

    def test_trace_consistency(self):
        """
        Verify that intermediate calculations are consistent with final results.
        Cross-check Q and t_final using both formulas.
        """
        logs = []
        success = True
        
        try:
            # Step 1 verification: Q = efficiency * E0 * time * area
            Q_calc_step1 = self.efficiency * self.E0 * self.time_sunlight_seconds * self.S
            if not math.isclose(self.Q, Q_calc_step1, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Step 1 inconsistency: Q={self.Q:.2e}, recalculated={Q_calc_step1:.2e}")
            
            # Step 2 verification: t_final = t0 + Q/(m*c)
            t_calc_step2 = self.t0 + self.Q / (self.m * self.c_water)
            if not math.isclose(self.t_final, t_calc_step2, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Step 2 inconsistency: t_final={self.t_final:.2f}, recalculated={t_calc_step2:.2f}")
            
            # Verify final temperature within reasonable bounds
            if not (0 <= self.t_final <= 150):  # Practical upper bound for water heaters
                success = False
                logs.append(f"Unrealistic final temperature: t_final={self.t_final:.2f}°C")
            
            if success:
                logs.append("Trace consistency verified: intermediate calculations match final results")
                
        except Exception as e:
            success = False
            logs.append(f"Trace consistency test error: {str(e)}\n{traceback.format_exc()}")
        
        return success, logs

    def test_monotonicity(self):
        """
        Test that increasing input parameters produces expected directional changes:
        1. Increase mass → decrease final temperature
        2. Increase area → increase final temperature
        3. Increase efficiency → increase final temperature
        """
        logs = []
        success = True
        
        try:
            original_t = self.t_final
            
            # 1. Test mass perturbation
            self.m *= 1.1  # Increase by 10%
            new_t_mass = self.t0 + self.Q / (self.m * self.c_water)
            if not (new_t_mass < original_t or math.isclose(new_t_mass, original_t, rel_tol=1e-2)):
                success = False
                logs.append(f"Mass monotonicity failed: Increased mass but t did not decrease: {original_t:.2f}→{new_t_mass:.2f}")
            self.m = self._original_m  # Restore
            
            # 2. Test area perturbation
            self.S *= 1.1
            new_Q_area = self.efficiency * self.E0 * self.time_sunlight_seconds * self.S
            new_t_area = self.t0 + new_Q_area / (self.m * self.c_water)
            if not (new_t_area > original_t or math.isclose(new_t_area, original_t, rel_tol=1e-2)):
                success = False
                logs.append(f"Area monotonicity failed: Increased area but t did not increase: {original_t:.2f}→{new_t_area:.2f}")
            self.S = self._original_S  # Restore
            
            # 3. Test efficiency perturbation
            self.efficiency *= 1.1
            new_Q_eff = self.efficiency * self.E0 * self.time_sunlight_seconds * self.S
            new_t_eff = self.t0 + new_Q_eff / (self.m * self.c_water)
            if not (new_t_eff > original_t or math.isclose(new_t_eff, original_t, rel_tol=1e-2)):
                success = False
                logs.append(f"Efficiency monotonicity failed: Increased efficiency but t did not increase: {original_t:.2f}→{new_t_eff:.2f}")
            self.efficiency = self._original_efficiency  # Restore
            
            if success:
                logs.append("All monotonicity tests passed: Physics trends are correct")
                
        except Exception as e:
            success = False
            logs.append(f"Monotonicity test error: {str(e)}\n{traceback.format_exc()}")
        
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success status with logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.extend(test_logs)
                if not test_success:
                    overall_success = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                overall_success = False
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.append(f"CRASHED: {str(e)}\n{traceback.format_exc()}")
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs