import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Problem constants (given in problem statement)
        self.mass_vehicle = 6000.0  # kg
        # Don't assume g value - extract from solution if available, otherwise use 9.8
        self.g = self._safe_get(solution_instance, 'g', 9.8)  # m/s^2
        self.resistance_coef = 0.05
        self.max_power = 60000.0  # W (60 kW)
        
        # Compute derived constant using the g value from solution
        self.resistance_force = self.resistance_coef * self.mass_vehicle * self.g
        
        # Extract results with safe fallbacks
        self.t = self._safe_get(solution_instance, 't')  # acceleration time
        self.F1 = self._safe_get(solution_instance, 'F1')  # traction force during acceleration
        self.m_fuel = self._safe_get(solution_instance, 'm')  # hydrogen fuel mass
        self.t1 = self._safe_get(solution_instance, 't1')  # acceleration time (alternative)
        self.vm = self._safe_get(solution_instance, 'vm')  # max speed at constant velocity
        self.v1 = self._safe_get(solution_instance, 'v1')  # speed at end of acceleration
        
        # For monotonicity testing
        self.original_mass = self.mass_vehicle
        
    def _safe_get(self, solution_instance, key, default=None):
        """Safe attribute access with fallback to results dict."""
        val = getattr(solution_instance, key, None)
        if val is None and hasattr(solution_instance, 'results'):
            val = solution_instance.results.get(key, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        return default
    
    def _is_close(self, a, b):
        """Robust numerical comparison with physics tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        
        # 1. Check Newton's 2nd Law for acceleration phase
        if self.F1 is not None and self.v1 is not None and self.t is not None:
            # Compute acceleration from kinematics: a = v1 / t
            if self.t != 0:
                a_calculated = self.v1 / self.t
            else:
                a_calculated = 0
                logs.append("Warning: t=0, cannot compute acceleration")
            
            # From Newton's 2nd Law: F_net = m*a = F1 - resistance
            F_net_calculated = self.mass_vehicle * a_calculated
            F_net_expected = self.F1 - self.resistance_force
            
            if not self._is_close(F_net_calculated, F_net_expected):
                logs.append(f"Newton's 2nd Law violation: F_net={F_net_calculated:.2f}N, expected {F_net_expected:.2f}N")
        
        # 2. Check equilibrium condition at max speed
        if self.vm is not None and self.vm != 0:
            # At constant velocity, traction force = resistance force
            # From power: F_traction = P / v
            F_traction_at_vm = self.max_power / self.vm
            
            if not self._is_close(F_traction_at_vm, self.resistance_force):
                logs.append(f"Equilibrium violation at vm: F_traction={F_traction_at_vm:.2f}N, resistance={self.resistance_force:.2f}N (using g={self.g:.1f}m/s²)")
        elif self.vm == 0:
            logs.append("Warning: vm=0, cannot check equilibrium")
        
        # 3. Check energy-power relationship during acceleration
        if self.F1 is not None and self.v1 is not None:
            # Instantaneous power at end of acceleration: P = F1 * v1
            power_at_v1 = self.F1 * self.v1
            
            # Should not exceed max power (could be equal at v1 if constant power acceleration)
            # Allow small tolerance for floating point errors
            if power_at_v1 > self.max_power * 1.02:  # Allow 2% tolerance
                logs.append(f"Power exceedance: P(v1)={power_at_v1:.2f}W > P_max={self.max_power:.2f}W")
        
        return len(logs) == 0, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Target answers: 20s, 20m/s, 12.6kg (with tolerance)
        target_t = 20.0
        target_vm = 20.0
        target_m_fuel = 12.6
        
        # Check t1 (acceleration time) - use t1 if available, otherwise t
        time_to_check = self.t1 if self.t1 is not None else self.t
        if time_to_check is not None and not self._is_close(time_to_check, target_t):
            logs.append(f"Acceleration time mismatch: t={time_to_check:.2f}s, target={target_t}s")
        
        # Check vm (max speed)
        if self.vm is not None and not self._is_close(self.vm, target_vm):
            logs.append(f"Max speed mismatch: vm={self.vm:.2f}m/s, target={target_vm}m/s")
        
        # Check m_fuel (hydrogen mass)
        if self.m_fuel is not None and not self._is_close(self.m_fuel, target_m_fuel):
            logs.append(f"Fuel mass mismatch: m={self.m_fuel:.2f}kg, target={target_m_fuel}kg")
        
        # Internal consistency: if we have all three, check energy balance
        if all(x is not None for x in [self.t, self.v1, self.F1]) and self.t != 0:
            # Distance during acceleration: s = 0.5 * a * t^2
            a_accel = self.v1 / self.t
            s_accel = 0.5 * a_accel * self.t**2
            
            # Work done by traction force: W = F1 * s
            work_traction = self.F1 * s_accel
            
            # Work against resistance: W_res = resistance * s
            work_resistance = self.resistance_force * s_accel
            
            # Kinetic energy gained: ΔKE = 0.5 * m * v1^2
            delta_ke = 0.5 * self.mass_vehicle * self.v1**2
            
            # Energy balance: W_traction = ΔKE + W_resistance
            if not self._is_close(work_traction, delta_ke + work_resistance):
                logs.append(f"Energy balance violation during acceleration: W_trac={work_traction:.2f}J, ΔKE+W_res={delta_ke+work_resistance:.2f}J")
        
        return len(logs) == 0, logs
    
    def test_monotonicity(self):
        """Perturb vehicle mass and check trend."""
        logs = []
        
        try:
            if self.vm is None:
                return True, logs
                
            # Perturb mass by +10%
            perturbed_mass = self.original_mass * 1.10
            
            # Recalculate resistance with perturbed mass (using same g)
            perturbed_resistance = self.resistance_coef * perturbed_mass * self.g
            
            # Max speed should DECREASE with increased mass (since resistance increases)
            if perturbed_resistance != 0:
                perturbed_vm = self.max_power / perturbed_resistance
                
                if perturbed_vm >= self.vm * 1.001:  # Allow 0.1% tolerance for floating point
                    logs.append(f"Monotonicity violation: Increased mass to {perturbed_mass:.0f}kg but vm increased to {perturbed_vm:.2f}m/s (original {self.vm:.2f}m/s)")
            else:
                logs.append("Warning: perturbed_resistance=0, cannot compute perturbed_vm")
            
            # Restore original mass
            self.mass_vehicle = self.original_mass
            
        except Exception as e:
            logs.append(f"Monotonicity test failed: {str(e)}")
        
        return len(logs) == 0, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                if not test_success:
                    success = False
                    all_logs.append(f"{test_name} failed:")
                    all_logs.extend([f"  - {log}" for log in logs])
                else:
                    all_logs.append(f"{test_name}: PASSED")
            except Exception as e:
                success = False
                all_logs.append(f"{test_name} ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
        
        return success, all_logs