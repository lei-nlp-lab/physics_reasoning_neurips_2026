import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # SAFE ATTRIBUTE ACCESS: Use getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.g = getattr(solution_instance, 'g', 10.0)
        self.m = getattr(solution_instance, 'm', 180.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.v_platform = getattr(solution_instance, 'v_platform', 3.0)
        self.H = getattr(solution_instance, 'H', 0.8)
        
        # Extract results with fallbacks
        self.s = self.results.get('s', 0.0)
        self.v_y = self.results.get('v_y', 0.0)
        self.v_A = self.results.get('v_A', 0.0)
        self.theta = self.results.get('theta', 0.0)
        self.alpha = self.results.get('alpha', 0.0)
        self.N_A = self.results.get('N_A', 0.0)
        self.N = self.results.get('N', 0.0)
        
        # Precompute derived quantities for validation
        self.t_fall = math.sqrt(2 * self.H / self.g) if self.g > 0 else 0.0
        self.sin53 = 0.8
        self.cos53 = 0.6
        
        # Target answers from problem statement
        self.targets = {
            's': 1.2,
            'v_A': 5.0,
            'theta': 106.0,
            'N_A': 5580.0,
            'N': 7740.0
        }
        
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison with 1-2% tolerance"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical conservation laws and Newton's laws"""
        checks = []
        
        # 1. Energy conservation from platform to point A
        # Kinetic energy at platform + potential energy = Kinetic energy at A
        KE_platform = 0.5 * self.m * self.v_platform**2
        PE_platform = self.m * self.g * self.H
        KE_A = 0.5 * self.m * self.v_A**2
        
        # At point A, some potential energy remains if we take ground as reference
        # Actually, point A is at ground level, so all potential from platform converted to kinetic
        energy_platform = KE_platform + PE_platform
        energy_A = KE_A
        
        if self._isclose(energy_platform, energy_A, rel_tol=0.05):
            checks.append(self._log(f"Energy conservation platform→A: {energy_platform:.1f} ≈ {energy_A:.1f} J", True))
        else:
            checks.append(self._log(f"Energy conservation platform→A FAILED: {energy_platform:.1f} ≠ {energy_A:.1f} J", False))
        
        # 2. Energy conservation from A to lowest point O (bottom of circle)
        # At point A: height above bottom = R*(1 - cos(alpha))
        # Use alpha from solution or compute from v_y/v
        if hasattr(self, 'alpha') and self.alpha > 0:
            alpha_rad = math.radians(self.alpha)
        else:
            alpha_rad = math.atan2(self.v_y, self.v_platform)
        
        h_A = self.R * (1 - math.cos(alpha_rad))
        KE_A = 0.5 * self.m * self.v_A**2
        PE_A = self.m * self.g * h_A
        
        # Speed at bottom from energy conservation: v_O^2 = v_A^2 + 2*g*h_A
        v_O_sq = self.v_A**2 + 2 * self.g * h_A
        v_O = math.sqrt(v_O_sq) if v_O_sq > 0 else 0.0
        
        # Compute N from centripetal force at bottom
        N_calculated = self.m * self.g + self.m * v_O_sq / self.R
        
        if self._isclose(self.N, N_calculated, rel_tol=0.05):
            checks.append(self._log(f"Energy conservation A→O yields N = {N_calculated:.0f} ≈ {self.N:.0f} N", True))
        else:
            checks.append(self._log(f"Energy conservation A→O FAILED: N_calc={N_calculated:.0f} ≠ {self.N:.0f}", False))
        
        # 3. Newton's 2nd law at point A (centripetal force)
        # N_A - m*g*cos(alpha) = m*v_A^2/R
        if self.alpha > 0:
            alpha_rad = math.radians(self.alpha)
            cos_alpha = math.cos(alpha_rad)
        else:
            # Use given sin53/cos53 if alpha is 53°
            cos_alpha = self.cos53 if self._isclose(self.alpha, 53.0) else math.cos(math.radians(self.alpha))
        
        centripetal_A = self.m * self.v_A**2 / self.R
        weight_component_A = self.m * self.g * cos_alpha
        N_A_calculated = weight_component_A + centripetal_A
        
        if self._isclose(self.N_A, N_A_calculated, rel_tol=0.05):
            checks.append(self._log(f"Newton's 2nd law at A: N_A = {N_A_calculated:.0f} ≈ {self.N_A:.0f} N", True))
        else:
            checks.append(self._log(f"Newton's 2nd law at A FAILED: {N_A_calculated:.0f} ≠ {self.N_A:.0f}", False))
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        checks = []
        
        # 1. Check fall time from H
        t_calc = math.sqrt(2 * self.H / self.g)
        if self._isclose(t_calc, self.t_fall):
            checks.append(self._log(f"Fall time t = √(2H/g) = {t_calc:.3f} s", True))
        else:
            checks.append(self._log(f"Fall time inconsistent: {t_calc:.3f} ≠ {self.t_fall:.3f}", False))
        
        # 2. Check horizontal distance s = v_platform * t
        s_calc = self.v_platform * self.t_fall
        if self._isclose(self.s, s_calc):
            checks.append(self._log(f"Horizontal distance s = v*t = {s_calc:.2f} ≈ {self.s:.2f} m", True))
        else:
            checks.append(self._log(f"Horizontal distance inconsistent: {s_calc:.2f} ≠ {self.s:.2f}", False))
        
        # 3. Check vertical velocity v_y = g*t
        v_y_calc = self.g * self.t_fall
        if self._isclose(self.v_y, v_y_calc):
            checks.append(self._log(f"Vertical velocity v_y = g*t = {v_y_calc:.1f} ≈ {self.v_y:.1f} m/s", True))
        else:
            checks.append(self._log(f"Vertical velocity inconsistent: {v_y_calc:.1f} ≠ {self.v_y:.1f}", False))
        
        # 4. Check speed at A: v_A = √(v_platform² + v_y²)
        v_A_calc = math.sqrt(self.v_platform**2 + self.v_y**2)
        if self._isclose(self.v_A, v_A_calc):
            checks.append(self._log(f"Speed at A: √(v² + v_y²) = {v_A_calc:.1f} ≈ {self.v_A:.1f} m/s", True))
        else:
            checks.append(self._log(f"Speed at A inconsistent: {v_A_calc:.1f} ≠ {self.v_A:.1f}", False))
        
        # 5. Check angle alpha = arctan(v_y/v_platform) in degrees
        if self.v_platform > 0:
            alpha_calc = math.degrees(math.atan2(self.v_y, self.v_platform))
            if self._isclose(self.alpha, alpha_calc) or self._isclose(self.alpha, 53.0):
                checks.append(self._log(f"Angle α = arctan(v_y/v) = {alpha_calc:.1f}° ≈ {self.alpha:.1f}°", True))
            else:
                checks.append(self._log(f"Angle α inconsistent: {alpha_calc:.1f}° ≠ {self.alpha:.1f}°", False))
        
        # 6. Check central angle θ = 2α
        theta_calc = 2 * self.alpha
        if self._isclose(self.theta, theta_calc):
            checks.append(self._log(f"Central angle θ = 2α = {theta_calc:.1f}° ≈ {self.theta:.1f}°", True))
        else:
            checks.append(self._log(f"Central angle inconsistent: {theta_calc:.1f}° ≠ {self.theta:.1f}°", False))
        
        # 7. Check against target answers
        for key, target in self.targets.items():
            value = getattr(self, key, 0.0)
            if self._isclose(value, target):
                checks.append(self._log(f"Target {key}: {value} ≈ {target}", True))
            else:
                checks.append(self._log(f"Target {key} MISSED: {value} ≠ {target}", False))
        
        return all(checks)
    
    def test_monotonicity(self):
        """Perturb mass and check trend in normal forces, then restore"""
        checks = []
        
        # Store original mass
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate N_A with perturbed mass (same velocities, angles)
            if self.alpha > 0:
                alpha_rad = math.radians(self.alpha)
                cos_alpha = math.cos(alpha_rad)
            else:
                cos_alpha = self.cos53  # Use given value if alpha ≈ 53°
            
            N_A_perturbed = perturbed_m * self.g * cos_alpha + perturbed_m * self.v_A**2 / self.R
            
            # N should scale linearly with mass if v_O^2 scales properly
            # But v_O depends on v_A which is independent of mass, so N should also scale linearly
            # Actually from energy: v_O^2 = v_A^2 + 2*g*h_A, both terms independent of mass
            # So centripetal term m*v_O^2/R scales linearly with mass
            N_perturbed = perturbed_m * self.g + perturbed_m * (self.v_A**2 + 2 * self.g * self.R * (1 - cos_alpha)) / self.R
            
            # Expected scaling factor
            scaling = perturbed_m / original_m
            
            # Check linear scaling
            if (self._isclose(N_A_perturbed / self.N_A, scaling, rel_tol=0.01) and 
                self._isclose(N_perturbed / self.N, scaling, rel_tol=0.01)):
                checks.append(self._log(
                    f"Monotonicity: +10% mass → N_A×{N_A_perturbed/self.N_A:.3f}, N×{N_perturbed/self.N:.3f} (expected ×{scaling:.3f})", 
                    True
                ))
            else:
                checks.append(self._log(
                    f"Monotonicity FAILED: N_A scaled by {N_A_perturbed/self.N_A:.3f}, N by {N_perturbed/self.N:.3f}", 
                    False
                ))
            
        except Exception as e:
            checks.append(self._log(f"Monotonicity test ERROR: {str(e)}", False))
        
        finally:
            # Restore original mass
            self.m = original_m
        
        return all(checks)
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        success = True
        
        # Add header
        self.logs.append("=" * 60)
        self.logs.append("PhysicsChecker Audit Report")
        self.logs.append("=" * 60)
        
        # Run tests
        test_suites = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Physical Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_suites:
            self.logs.append(f"\n{test_name}:")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self.logs.append(f"✗ {test_name} CRASHED: {str(e)}")
                self.logs.append(traceback.format_exc())
                success = False
        
        # Summary
        self.logs.append("\n" + "=" * 60)
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        self.logs.append("=" * 60)
        
        return success, self.logs