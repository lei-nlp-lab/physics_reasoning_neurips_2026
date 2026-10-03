import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        
        # Extract inputs with defaults from problem statement
        self.L = getattr(solution_instance, 'L', 0.2)
        self.alpha_deg = getattr(solution_instance, 'alpha', 60)
        self.alpha = math.radians(self.alpha_deg)
        self.x0 = getattr(solution_instance, 'x0', 0.1)
        self.k = getattr(solution_instance, 'k', 100)
        self.m = getattr(solution_instance, 'm', 1)
        self.g = getattr(solution_instance, 'g', 10)
        
        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        
        self.T0 = getattr(solution_instance, 'T_0', results.get('T_0', 0.0))
        self.Deltax0 = getattr(solution_instance, 'Deltax_0', results.get('Deltax_0', 0.0))
        self.x1 = getattr(solution_instance, 'x_1', results.get('x_1', 0.0))
        self.r = getattr(solution_instance, 'r', results.get('r', 0.0))
        self.omega0 = getattr(solution_instance, 'omega_0', results.get('omega_0', 0.0))
        self.T = getattr(solution_instance, 'T', results.get('T', 0.0))
        self.rprime = getattr(solution_instance, 'rprime', results.get('rprime', 0.0))
        self.omega = getattr(solution_instance, 'omega', results.get('omega', 0.0))
        
        # Precompute derived values for convenience
        self.cos_alpha = math.cos(self.alpha)
        self.sin_alpha = math.sin(self.alpha)
        self.tan_alpha = math.tan(self.alpha)
        
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def test_invariants(self):
        """Verify physical conservation laws and equilibrium conditions."""
        logs = []
        success = True
        
        # 1. Check stationary equilibrium (step 1)
        expected_T0 = self.m * self.g * self.cos_alpha
        if not math.isclose(self.T0, expected_T0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Stationary equilibrium failed: T0={self.T0}, expected={expected_T0}")
            success = False
        else:
            logs.append(f"Stationary equilibrium OK: T0={self.T0:.3f} N")
        
        # 2. Check Hooke's Law for stationary compression (step 2)
        expected_Deltax0 = self.T0 / self.k
        if not math.isclose(self.Deltax0, expected_Deltax0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Hooke's Law failed: Δx0={self.Deltax0}, expected={expected_Deltax0}")
            success = False
        else:
            logs.append(f"Hooke's Law OK: Δx0={self.Deltax0:.3f} m")
        
        # 3. Check geometric consistency (step 3)
        expected_x1 = self.x0 - self.Deltax0
        if not math.isclose(self.x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Geometry failed: x1={self.x1}, expected={expected_x1}")
            success = False
        else:
            logs.append(f"Geometry OK: x1={self.x1:.3f} m")
        
        # 4. Check rotational equilibrium for ω0 (step 4, 5, 6)
        # r should be x0*sin(α) when spring is at original length
        expected_r = self.x0 * self.sin_alpha
        if not math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Radius mismatch: r={self.r}, expected={expected_r}")
            success = False
        
        # Centripetal force should equal horizontal component
        centripetal = self.m * self.omega0**2 * self.r
        horizontal_force = (self.m * self.g) / self.tan_alpha
        
        if not math.isclose(centripetal, horizontal_force, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"ω0 equilibrium failed: mω²r={centripetal:.3f}, mg/tanα={horizontal_force:.3f}")
            success = False
        else:
            logs.append(f"ω0 equilibrium OK: ω0={self.omega0:.3f} rad/s")
        
        # 5. Check forces at point P (steps 7-11)
        # Spring force at P (step 7)
        expected_T = self.k * (self.L - self.x0)
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Spring force at P failed: T={self.T}, expected={expected_T}")
            success = False
        
        # Radius at P (step 10)
        expected_rprime = self.L * self.sin_alpha
        if not math.isclose(self.rprime, expected_rprime, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Radius at P failed: r'={self.rprime}, expected={expected_rprime}")
            success = False
        
        # Solve for normal force FN from vertical equilibrium (step 8)
        # mg + Tcosα = FN sinα
        FN = (self.m * self.g + self.T * self.cos_alpha) / self.sin_alpha
        
        # Check horizontal equilibrium (step 9)
        left_side = self.T * self.sin_alpha + FN * self.cos_alpha
        right_side = self.m * self.omega**2 * self.rprime
        
        if not math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"ω equilibrium failed: Tsinα+FNcosα={left_side:.3f}, mω²r'={right_side:.3f}")
            success = False
        else:
            logs.append(f"ω equilibrium OK: ω={self.omega:.3f} rad/s")
        
        # Also verify ω against target answer 10 rad/s
        target_omega = 10.0
        if not math.isclose(self.omega, target_omega, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"ω doesn't match target: {self.omega} vs {target_omega}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check x1 matches target answer 0.05m
        if not math.isclose(self.x1, 0.05, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"x1 inconsistent: {self.x1} vs 0.05m")
            success = False
        else:
            logs.append(f"x1 consistent with target: {self.x1:.3f} m")
        
        # Check ω0 matches target answer 10√6/3 ≈ 8.165 rad/s
        target_omega0 = 10 * math.sqrt(6) / 3
        if not math.isclose(self.omega0, target_omega0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"ω0 inconsistent: {self.omega0:.3f} vs {target_omega0:.3f} rad/s")
            success = False
        else:
            logs.append(f"ω0 consistent with target: {self.omega0:.3f} rad/s")
        
        # Check ω matches target answer 10 rad/s
        if not math.isclose(self.omega, 10.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"ω inconsistent: {self.omega} vs 10 rad/s")
            success = False
        else:
            logs.append(f"ω consistent with target: {self.omega:.3f} rad/s")
        
        # Verify that x1 = x0 - Δx0 (already in invariants, but double-check)
        if not math.isclose(self.x1, self.x0 - self.Deltax0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Chain x0→Δx0→x1 broken: x1={self.x1}, x0-Δx0={self.x0 - self.Deltax0}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb mass and check trends, then restore."""
        logs = []
        success = True
        
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate T0 with perturbed mass
            new_T0 = perturbed_m * self.g * self.cos_alpha
            
            # T0 should increase with mass
            if not (new_T0 > self.T0):
                logs.append(f"T0 monotonicity failed: {new_T0:.3f} ≤ {self.T0:.3f}")
                success = False
            else:
                logs.append(f"T0 monotonicity OK: increases with mass")
            
            # Recalculate Δx0
            new_Deltax0 = new_T0 / self.k
            
            # Δx0 should increase with T0 (and thus with mass)
            if not (new_Deltax0 > self.Deltax0):
                logs.append(f"Δx0 monotonicity failed: {new_Deltax0:.3f} ≤ {self.Deltax0:.3f}")
                success = False
            else:
                logs.append(f"Δx0 monotonicity OK: increases with mass")
            
            # Recalculate ω0 - should be INDEPENDENT of mass (cancels in equation)
            # From: mg/tanα = mω0²r → ω0² = g/(tanα * r)
            # r = x0*sinα (unchanged)
            new_omega0 = math.sqrt(self.g / (self.tan_alpha * (self.x0 * self.sin_alpha)))
            
            # ω0 should remain nearly the same (within 0.1% tolerance)
            if not math.isclose(new_omega0, self.omega0, rel_tol=1e-3, abs_tol=1e-3):
                logs.append(f"ω0 should be mass-independent: {new_omega0:.3f} vs {self.omega0:.3f}")
                success = False
            else:
                logs.append(f"ω0 mass-independence OK")
            
            # Recalculate ω at point P
            # T is unchanged (k(L-x0))
            T = self.k * (self.L - self.x0)
            rprime = self.L * self.sin_alpha
            
            # Solve for ω from equations:
            # 1) mg + Tcosα = FN sinα → FN = (mg + Tcosα)/sinα
            # 2) Tsinα + FN cosα = mω²r'
            # Substitute FN:
            # Tsinα + [(mg + Tcosα)/sinα] * cosα = mω²r'
            # Multiply by sinα: Tsin²α + (mg + Tcosα)cosα = mω²r' sinα
            # ω² = [Tsin²α + (mg + Tcosα)cosα] / (m r' sinα)
            
            numerator = (T * self.sin_alpha**2 + 
                        (perturbed_m * self.g + T * self.cos_alpha) * self.cos_alpha)
            denominator = perturbed_m * rprime * self.sin_alpha
            new_omega = math.sqrt(numerator / denominator)
            
            # As mass increases, ω should DECREASE (from derived formula)
            # Let's verify: ω² = [Tsin²α + Tcos²α + mgcosα] / (m r' sinα)
            # = [T(sin²α+cos²α) + mgcosα] / (m r' sinα) = [T + mgcosα] / (m r' sinα)
            # = T/(m r' sinα) + (gcosα)/(r' sinα)
            # First term decreases with m, second is constant
            if not (new_omega < self.omega):
                logs.append(f"ω monotonicity failed: {new_omega:.3f} ≥ {self.omega:.3f}")
                success = False
            else:
                logs.append(f"ω monotonicity OK: decreases with increased mass")
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
        finally:
            # Restore original mass (though we only used local variables)
            self.m = original_m
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} Error ===")
                all_logs.append(f"Exception: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs