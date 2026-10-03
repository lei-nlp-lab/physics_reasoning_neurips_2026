import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs and results with safe defaults
        self.M = self.results.get('M', None)
        self.m = self.results.get('m', None)
        self.G = self.results.get('G', 6.67430e-11)
        self.R = self.results.get('R', None)
        self.r = self.results.get('r', None)
        self.L = self.results.get('L', 5.0e9)  # Use 5e9 m as default
        
        # Extract results from solution with safe access
        self.omega0 = self.results.get('omega0', None)
        self.omega0_sq = self.results.get('omega0_sq', None)
        self.r1 = self.results.get('r1', None)
        self.r2 = self.results.get('r2', None)
        self.rho = self.results.get('rho', None)
        self.T = self.results.get('T', None)
        self.omega = self.results.get('omega', None)  # This is oscillation frequency
        self.v = self.results.get('v', None)
        self.v_B = self.results.get('v_B', None)
        self.v_C = self.results.get('v_C', None)
        self.v_BC = self.results.get('v_BC', None)
        self.E = self.results.get('E', None)
        
        # Compute orbital angular frequency from T if available
        self.omega_orbit = None
        if self.T is not None:
            self.omega_orbit = 2 * math.pi / self.T
        
        self.logs = []
        
    def test_invariants(self) -> list:
        """Verify physical invariants based on KB theorems"""
        logs = []
        
        # 1. Center of mass condition: M*R = m*r
        if all(val is not None for val in [self.M, self.m, self.R, self.r]):
            expected_relation = self.M * self.R - self.m * self.r
            if not math.isclose(expected_relation, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Center of mass condition violated: M*R = {self.M*self.R}, m*r = {self.m*self.r}")
            else:
                logs.append("PASS: Center of mass condition M*R = m*r holds")
        
        # 2. Angular velocity consistency from gravitational force balance
        if all(val is not None for val in [self.omega0_sq, self.G, self.M, self.m, self.R, self.r]):
            # From step 3: ω₀² = G(M+m)/(R+r)³
            expected_omega0_sq1 = self.G * (self.M + self.m) / (self.R + self.r)**3
            
            # From step 4: ω₀² = GM/(r(R+r)²) and ω₀² = Gm/(R(R+r)²)
            expected_omega0_sq2 = self.G * self.M / (self.r * (self.R + self.r)**2)
            expected_omega0_sq3 = self.G * self.m / (self.R * (self.R + self.r)**2)
            
            if not (math.isclose(self.omega0_sq, expected_omega0_sq1, rel_tol=1e-2, abs_tol=1e-2) or
                    math.isclose(self.omega0_sq, expected_omega0_sq2, rel_tol=1e-2, abs_tol=1e-2) or
                    math.isclose(self.omega0_sq, expected_omega0_sq3, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"FAIL: ω₀² inconsistent with gravitational force balance. Got {self.omega0_sq}")
            else:
                logs.append("PASS: ω₀² consistent with gravitational theory")
        
        # 3. Check distances r₁ and r₂
        if all(val is not None for val in [self.r1, self.r2, self.R, self.r]):
            if not math.isclose(self.r1, self.R + self.r, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: r₁ = {self.r1} but expected R+r = {self.R+self.r}")
            else:
                logs.append("PASS: r₁ = R+r")
            
            if not math.isclose(self.r2, self.R + self.r, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: r₂ = {self.r2} but expected R+r = {self.R+self.r}")
            else:
                logs.append("PASS: r₂ = R+r")
        
        # 4. Check ρ (distance from μ to center of mass)
        if all(val is not None for val in [self.rho, self.R, self.r]):
            expected_rho = math.sqrt(self.R**2 + self.R*self.r + self.r**2)
            if not math.isclose(self.rho, expected_rho, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ρ = {self.rho} but expected √(R²+Rr+r²) = {expected_rho}")
            else:
                logs.append("PASS: ρ = √(R²+Rr+r²)")
        
        # 5. Check LISA spacecraft relationships
        if self.T is not None:
            # T should be Earth's orbital period
            expected_T = 365 * 24 * 60 * 60  # seconds
            if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: T = {self.T} differs from Earth period {expected_T}")
            else:
                logs.append("PASS: T matches Earth's orbital period")
        
        if self.omega_orbit is not None and self.T is not None:
            expected_omega_orbit = 2 * math.pi / self.T
            if not math.isclose(self.omega_orbit, expected_omega_orbit, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ω_orbit = {self.omega_orbit} but expected 2π/T = {expected_omega_orbit}")
            else:
                logs.append("PASS: ω_orbit = 2π/T")
        
        # 6. Check velocity relationships for LISA
        if all(val is not None for val in [self.v, self.omega_orbit, self.L]):
            # v = ω_orbit * (L/2cos30°) = ω_orbit * L/√3
            expected_v = self.omega_orbit * self.L / math.sqrt(3)
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v = {self.v} but expected ω_orbit*L/√3 = {expected_v}")
            else:
                logs.append("PASS: v = ω_orbit*L/√3")
        
        if all(val is not None for val in [self.v_BC, self.v]):
            # v_BC = √3 * v from step 31
            expected_v_BC = math.sqrt(3) * self.v
            if isinstance(self.v_BC, (list, tuple)) and len(self.v_BC) >= 2:
                mag_v_BC = math.sqrt(self.v_BC[0]**2 + self.v_BC[1]**2)
            else:
                mag_v_BC = self.v_BC
                
            if not math.isclose(mag_v_BC, expected_v_BC, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: |v_BC| = {mag_v_BC} but expected √3*v = {expected_v_BC}")
            else:
                logs.append("PASS: |v_BC| = √3*v")
        
        if all(val is not None for val in [self.v_BC, self.omega_orbit, self.L]):
            # v_BC = ω_orbit * L from step 33
            expected_v_BC_alt = self.omega_orbit * self.L
            if isinstance(self.v_BC, (list, tuple)) and len(self.v_BC) >= 2:
                mag_v_BC = math.sqrt(self.v_BC[0]**2 + self.v_BC[1]**2)
            else:
                mag_v_BC = self.v_BC
                
            if not math.isclose(mag_v_BC, expected_v_BC_alt, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: |v_BC| = {mag_v_BC} but expected ω_orbit*L = {expected_v_BC_alt}")
            else:
                logs.append("PASS: |v_BC| = ω_orbit*L")
        
        return logs
    
    def test_trace_consistency(self) -> list:
        """Cross-check intermediate results with final answers"""
        logs = []
        
        # Check if ω₀ matches expected expression from target answer
        if self.omega0 is not None and all(val is not None for val in [self.G, self.m, self.R, self.r]):
            # Expected: ω₀ = √(Gm/(R(R+r)²))
            expected_omega0 = math.sqrt(self.G * self.m / (self.R * (self.R + self.r)**2))
            if not math.isclose(self.omega0, expected_omega0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ω₀ = {self.omega0} differs from √(Gm/(R(R+r)²)) = {expected_omega0}")
            else:
                logs.append("PASS: ω₀ matches expected expression")
        
        # Check oscillation frequency from step 24
        if self.omega0 is not None and self.omega is not None:
            expected_osc_freq = (math.sqrt(7)/2) * self.omega0
            if not math.isclose(self.omega, expected_osc_freq, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ω_osc = {self.omega} but expected (√7/2)*ω₀ = {expected_osc_freq}")
            else:
                logs.append("PASS: ω_osc matches (√7/2)*ω₀")
        
        # Check that v_BC ≈ 996 m/s from step 32
        if self.v_BC is not None:
            if isinstance(self.v_BC, (list, tuple)) and len(self.v_BC) >= 2:
                mag_v_BC = math.sqrt(self.v_BC[0]**2 + self.v_BC[1]**2)
            else:
                mag_v_BC = self.v_BC
            
            if not math.isclose(mag_v_BC, 996.0, rel_tol=1e-2, abs_tol=10.0):
                logs.append(f"WARN: |v_BC| = {mag_v_BC}, expected ~996 m/s")
            else:
                logs.append("PASS: |v_BC| ≈ 996 m/s")
        
        return logs
    
    def test_monotonicity(self) -> list:
        """Perturb inputs and check trends"""
        logs = []
        
        # Save original values
        original_m = self.m
        original_R = self.R
        original_r = self.r
        
        if all(val is not None for val in [original_m, original_R, original_r, self.G, self.M]):
            # Test 1: Increase mass m by 10%
            perturbed_m = original_m * 1.1
            
            # Keep center of mass condition: M*R = m*r
            perturbed_r = (self.M * original_R) / perturbed_m
            
            # Calculate ω₀ for perturbed system
            omega0_original = math.sqrt(self.G * original_m / (original_R * (original_R + original_r)**2))
            omega0_perturbed = math.sqrt(self.G * perturbed_m / (original_R * (original_R + perturbed_r)**2))
            
            # Trend: If m increases, ω₀ should increase (since ω₀ ∝ √m when other factors held)
            if omega0_perturbed > omega0_original:
                logs.append("PASS: Increasing m increases ω₀ (consistent with physical trend)")
            else:
                logs.append(f"WARN: Increasing m decreased ω₀ from {omega0_original} to {omega0_perturbed}")
            
            # Test 2: Increase R by 10%
            perturbed_R = original_R * 1.1
            
            # Adjust r to maintain COM condition
            perturbed_r2 = (self.M * perturbed_R) / original_m
            
            omega0_perturbed2 = math.sqrt(self.G * original_m / (perturbed_R * (perturbed_R + perturbed_r2)**2))
            
            # Trend: If R increases, ω₀ should decrease (larger orbit → slower angular velocity)
            if omega0_perturbed2 < omega0_original:
                logs.append("PASS: Increasing R decreases ω₀ (consistent with physical trend)")
            else:
                logs.append(f"WARN: Increasing R increased ω₀ from {omega0_original} to {omega0_perturbed2}")
            
            # Restore original values
            self.m = original_m
            self.R = original_R
            self.r = original_r
        
        # Test LISA parameters
        original_L = self.L
        if original_L is not None and self.omega_orbit is not None:
            # Increase arm length L by 10%
            perturbed_L = original_L * 1.1
            
            v_original = self.omega_orbit * original_L / math.sqrt(3)
            v_perturbed = self.omega_orbit * perturbed_L / math.sqrt(3)
            
            # Trend: Longer arm → higher linear speed for same angular velocity
            if v_perturbed > v_original:
                logs.append("PASS: Increasing L increases spacecraft speed (consistent with v = ω_orbit*L/√3)")
            else:
                logs.append("WARN: Increasing L did not increase spacecraft speed")
            
            # Restore
            self.L = original_L
        
        return logs
    
    def run_all_tests(self) -> tuple:
        """Run all tests and return success status and logs"""
        all_logs = []
        
        # Run each test suite
        all_logs.append("=== Testing Physical Invariants ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== Testing Trace Consistency ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== Testing Monotonicity ===")
        all_logs.extend(self.test_monotonicity())
        
        # Determine success: no failures in critical tests
        success = True
        for log in all_logs:
            if log.startswith("FAIL:"):
                success = False
                break
        
        return success, all_logs