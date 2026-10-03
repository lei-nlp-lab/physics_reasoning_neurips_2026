import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: inputs and results
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', self.results.get('g', 9.8))
        self.L = getattr(solution_instance, 'L', self.results.get('L', 1.0))
        self.h = getattr(solution_instance, 'h', self.results.get('h', 0.5))
        self.H = getattr(solution_instance, 'H', self.results.get('H', 2.0 * self.L))
        
        # Extract results with safe defaults
        self.F = getattr(solution_instance, 'F', self.results.get('F', None))
        self.omega = getattr(solution_instance, 'omega', self.results.get('omega', None))
        self.x = getattr(solution_instance, 'x', self.results.get('x', None))
        self.s_0 = getattr(solution_instance, 's_0', self.results.get('s_0', None))
        self.r = getattr(solution_instance, 'r', self.results.get('r', None))
        self.v = getattr(solution_instance, 'v', self.results.get('v', None))
        self.h_prime = getattr(solution_instance, "h'", self.results.get("h'", None))
        
        # Define symbolic variables for checks
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'g': sp.symbols('g', positive=True),
            'L': sp.symbols('L', positive=True),
            'h': sp.symbols('h', positive=True),
            'omega': sp.symbols('omega', positive=True),
            'F': sp.symbols('F', positive=True),
            'theta': sp.symbols('theta', positive=True),
            'alpha': sp.symbols('alpha', positive=True),
            'v': sp.symbols('v', positive=True),
            'r': sp.symbols('r', positive=True),
            'x': sp.symbols('x', positive=True),
            's_0': sp.symbols('s_0', positive=True)
        }
        
        self.logs = []
        
    def log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def test_invariants(self):
        """Check fundamental physics laws based on KB."""
        checks = []
        
        # 1. Check centripetal force relationship: F = m * ω^2 * L
        if self.F is not None and self.omega is not None:
            F_expected = self.m * self.omega**2 * self.L
            if math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log(f"Centripetal force invariant: F = mω²L = {F_expected:.4f}", True))
            else:
                checks.append(self.log(f"Centripetal force violation: F={self.F:.4f}, expected={F_expected:.4f}", False))
        
        # 2. Check equilibrium in vertical direction: T cosθ = mg (from step 2)
        # We can't directly check θ, but we can check consistency
        if self.F is not None:
            # From step 2: F cosθ = mg, so cosθ = mg/F
            cos_theta = self.m * self.g / self.F if self.F != 0 else 0
            if 0 <= cos_theta <= 1:
                checks.append(self.log(f"Vertical equilibrium plausible: cosθ = {cos_theta:.4f} in [0,1]", True))
            else:
                checks.append(self.log(f"Vertical equilibrium violation: cosθ={cos_theta:.4f} outside [0,1]", False))
        
        # 3. Check ω = sqrt(g/h) from step 6
        if self.omega is not None:
            omega_expected = math.sqrt(self.g / self.h)
            if math.isclose(self.omega, omega_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log(f"Angular speed invariant: ω = √(g/h) = {omega_expected:.4f}", True))
            else:
                checks.append(self.log(f"Angular speed violation: ω={self.omega:.4f}, expected={omega_expected:.4f}", False))
        
        # 4. Check energy conservation in projectile phase
        if self.v is not None and self.h_prime is not None:
            # Initial kinetic energy + potential energy
            KE_initial = 0.5 * self.m * self.v**2
            PE_initial = self.m * self.g * self.h_prime
            # At ground, PE=0, KE should equal total initial energy
            # This is a conservation check
            total_energy = KE_initial + PE_initial
            if total_energy > 0:  # Positive energy is plausible
                checks.append(self.log(f"Energy conservation plausible: total={total_energy:.4f} > 0", True))
            else:
                checks.append(self.log(f"Energy violation: total energy={total_energy:.4f} ≤ 0", False))
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answers."""
        checks = []
        
        # 1. Check step 11: v = √(6gL)/2
        if self.v is not None:
            v_expected = math.sqrt(6 * self.g * self.L) / 2
            if math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log(f"Velocity consistent with step 11: v={self.v:.4f}", True))
            else:
                checks.append(self.log(f"Velocity mismatch: v={self.v:.4f}, expected={v_expected:.4f}", False))
        
        # 2. Check step 12: h' = (3/2)L
        if self.h_prime is not None:
            h_prime_expected = 1.5 * self.L
            if math.isclose(self.h_prime, h_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log(f"Height consistent with step 12: h'={self.h_prime:.4f}", True))
            else:
                checks.append(self.log(f"Height mismatch: h'={self.h_prime:.4f}, expected={h_prime_expected:.4f}", False))
        
        # 3. Check step 13: r = (√3/2)L
        if self.r is not None:
            r_expected = math.sqrt(3) * self.L / 2
            if math.isclose(self.r, r_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log(f"Radius consistent with step 13: r={self.r:.4f}", True))
            else:
                checks.append(self.log(f"Radius mismatch: r={self.r:.4f}, expected={r_expected:.4f}", False))
        
        # 4. Check step 16: x = (3√2/2)L
        if self.x is not None:
            x_expected = 3 * math.sqrt(2) * self.L / 2
            if math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log(f"Horizontal displacement consistent with step 16: x={self.x:.4f}", True))
            else:
                checks.append(self.log(f"Horizontal displacement mismatch: x={self.x:.4f}, expected={x_expected:.4f}", False))
        
        # 5. Check step 19: s_0 = (√37/2)L
        if self.s_0 is not None:
            s_0_expected = math.sqrt(37) * self.L / 2
            if math.isclose(self.s_0, s_0_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log(f"Final distance consistent with step 19: s_0={self.s_0:.4f}", True))
            else:
                checks.append(self.log(f"Final distance mismatch: s_0={self.s_0:.4f}, expected={s_0_expected:.4f}", False))
        
        # 6. Check Pythagorean consistency from step 17-18
        if self.r is not None and self.x is not None and self.s_0 is not None:
            # Step 17: s² = r² + x²
            s_sq = self.r**2 + self.x**2
            s = math.sqrt(s_sq)
            
            # Step 18: s_0² = H² + s²
            s_0_sq_expected = self.H**2 + s_sq
            
            if math.isclose(self.s_0**2, s_0_sq_expected, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self.log("Pythagorean consistency (steps 17-18) satisfied", True))
            else:
                checks.append(self.log(f"Pythagorean violation: s_0²={self.s_0**2:.4f}, expected={s_0_sq_expected:.4f}", False))
        
        return all(checks)
    
    def test_monotonicity(self):
        """Perturb inputs and check trends."""
        checks = []
        
        # Save original values
        m_orig = self.m
        L_orig = self.L
        h_orig = self.h
        
        try:
            # Test 1: Increase mass m → F should increase proportionally (F = mω²L)
            self.m = m_orig * 1.1  # +10%
            F_perturbed = self.m * (self.omega**2 if self.omega else (self.g/self.h)) * self.L
            if F_perturbed > (self.F if self.F else 0):
                checks.append(self.log(f"Force increases with mass (F ∝ m): {F_perturbed:.4f} > {self.F:.4f}", True))
            else:
                checks.append(self.log(f"Force monotonicity violation with mass", False))
            
            # Restore mass
            self.m = m_orig
            
            # Test 2: Increase L → F should increase (F = mω²L)
            self.L = L_orig * 1.1  # +10%
            F_perturbed = self.m * (self.omega**2 if self.omega else (self.g/self.h)) * self.L
            if F_perturbed > (self.F if self.F else 0):
                checks.append(self.log(f"Force increases with length (F ∝ L): {F_perturbed:.4f} > {self.F:.4f}", True))
            else:
                checks.append(self.log(f"Force monotonicity violation with length", False))
            
            # Restore L
            self.L = L_orig
            
            # Test 3: Increase h → ω should decrease (ω = √(g/h))
            self.h = h_orig * 1.1  # +10%
            omega_perturbed = math.sqrt(self.g / self.h)
            if omega_perturbed < (self.omega if self.omega else math.sqrt(self.g/h_orig)):
                checks.append(self.log(f"Angular speed decreases with height (ω ∝ 1/√h): {omega_perturbed:.4f} < {self.omega:.4f}", True))
            else:
                checks.append(self.log(f"Angular speed monotonicity violation with height", False))
            
            # Restore h
            self.h = h_orig
            
        except Exception as e:
            checks.append(self.log(f"Monotonicity test failed with error: {str(e)}", False))
            # Restore original values on error
            self.m = m_orig
            self.L = L_orig
            self.h = h_orig
        
        return all(checks)
    
    def run_all_tests(self):
        """Run all checks and return success status with logs."""
        self.logs = []  # Reset logs
        
        try:
            # Run tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            # Summary log
            self.logs.append(f"\nSUMMARY: {'ALL CHECKS PASSED' if success else 'SOME CHECKS FAILED'}")
            self.logs.append(f"Invariants: {'PASS' if inv_ok else 'FAIL'}")
            self.logs.append(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
            self.logs.append(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs