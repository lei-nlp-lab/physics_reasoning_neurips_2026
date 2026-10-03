import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely
        self.m = getattr(solution_instance, 'm', 0.4)
        self.mu = getattr(solution_instance, 'mu', 0.1)
        self.P = getattr(solution_instance, 'P', 10.0)
        self.L = getattr(solution_instance, 'L', 2.0)
        self.R = getattr(solution_instance, 'R', 0.5)
        self.alpha = getattr(solution_instance, 'alpha', 37.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        
        # Extract results safely with fallbacks
        results = getattr(solution_instance, 'results', {})
        self.F_N = results.get('F_N', 0.0)
        self.x = results.get('x', 0.0)
        self.v_y = results.get('v_y', 0.0)
        self.t = results.get('t', 0.0)
        self.v_C = results.get('v_C', 0.0)
        self.F_pressure = results.get('F_pressure', 0.0)
        self.h = results.get('h', 0.0)
        self.v_B = results.get('v_B', 0.0)
        self.t_1 = results.get('t_1', 0.0)
        
        # Given v_D from problem
        self.v_D = math.sqrt(27)
        
        # Store logs
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify conservation laws and Newton's laws."""
        logs = []
        try:
            # 1. Newton's Second Law at D: F_N = m*g + m*v_D^2/R
            expected_F_N = self.m * self.g + self.m * self.v_D**2 / self.R
            if math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Newton's 2nd Law at D: F_N = m*g + m*v_D²/R is satisfied")
            else:
                logs.append(f"✗ Newton's 2nd Law at D failed: F_N={self.F_N:.3f}, expected {expected_F_N:.3f}")
            
            # 2. Newton's Third Law at D: F_pressure = F_N
            if math.isclose(self.F_pressure, self.F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Newton's 3rd Law at D: F_pressure = F_N")
            else:
                logs.append(f"✗ Newton's 3rd Law at D failed: F_pressure={self.F_pressure:.3f}, F_N={self.F_N:.3f}")
            
            # 3. Conservation of Mechanical Energy between C and D
            # Energy at C: E_C = 1/2*m*v_C^2 + m*g*(R + R*cos(alpha))  # height from D to C is R(1-cos(alpha))
            # Actually, taking D as reference (h=0), then at C: h_C = R - R*cos(alpha) = R(1-cos(alpha))
            # So: E_C = 1/2*m*v_C^2 + m*g*R*(1-cos(alpha))
            # Energy at D: E_D = 1/2*m*v_D^2
            # Should satisfy: E_C = E_D
            E_C = 0.5 * self.m * self.v_C**2 + self.m * self.g * self.R * (1 - self.cos37)
            E_D = 0.5 * self.m * self.v_D**2
            if math.isclose(E_C, E_D, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Conservation of Mechanical Energy between C and D")
            else:
                logs.append(f"✗ Energy conservation C->D failed: E_C={E_C:.3f}, E_D={E_D:.3f}")
            
            # 4. Projectile motion decomposition at C
            v_y_expected = self.v_C * self.sin37
            if math.isclose(self.v_y, v_y_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Vertical velocity component at C: v_y = v_C*sin(α)")
            else:
                logs.append(f"✗ v_y decomposition failed: v_y={self.v_y:.3f}, expected {v_y_expected:.3f}")
            
            v_B_expected = self.v_C * self.cos37
            if math.isclose(self.v_B, v_B_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Horizontal velocity at B: v_B = v_C*cos(α)")
            else:
                logs.append(f"✗ v_B decomposition failed: v_B={self.v_B:.3f}, expected {v_B_expected:.3f}")
            
            # 5. Kinematic relations for projectile B->C
            h_expected = self.v_y**2 / (2 * self.g)
            if math.isclose(self.h, h_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Height difference B->C: h = v_y²/(2g)")
            else:
                logs.append(f"✗ Height h failed: h={self.h:.3f}, expected {h_expected:.3f}")
            
            t1_expected = self.v_y / self.g
            if math.isclose(self.t_1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Time B->C: t₁ = v_y/g")
            else:
                logs.append(f"✗ Time t₁ failed: t₁={self.t_1:.3f}, expected {t1_expected:.3f}")
            
            x_expected = self.v_B * self.t_1
            if math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Horizontal distance B->C: x = v_B*t₁")
            else:
                logs.append(f"✗ Distance x failed: x={self.x:.3f}, expected {x_expected:.3f}")
            
            # 6. Work-Energy theorem for A->B
            # Work done by motor: W_motor = P*t
            # Work done by friction: W_fric = -μ*m*g*L
            # Net work = ΔK = 1/2*m*v_B^2 - 0
            W_net = 0.5 * self.m * self.v_B**2
            W_total = self.P * self.t - self.mu * self.m * self.g * self.L
            if math.isclose(W_net, W_total, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Work-Energy theorem for A->B: P*t - μmgL = ½m v_B²")
            else:
                logs.append(f"✗ Work-Energy theorem failed: W_net={W_net:.3f}, W_total={W_total:.3f}")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('✗')]) == 0
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Target answers from problem
            # Note: The problem expects three groups of answers
            # We'll check each group against our calculated values
            
            # Group 1: F_pressure (should be 25.6) and v_C (should be 5)
            if math.isclose(self.F_pressure, 25.6, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ F_pressure matches target 25.6N")
            else:
                logs.append(f"✗ F_pressure mismatch: {self.F_pressure:.3f} vs 25.6")
            
            if math.isclose(self.v_C, 5.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_C matches target 5 m/s")
            else:
                logs.append(f"✗ v_C mismatch: {self.v_C:.3f} vs 5.0")
            
            # Group 2: h (0.45) and x (1.2)
            if math.isclose(self.h, 0.45, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ h matches target 0.45 m")
            else:
                logs.append(f"✗ h mismatch: {self.h:.3f} vs 0.45")
            
            if math.isclose(self.x, 1.2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ x matches target 1.2 m")
            else:
                logs.append(f"✗ x mismatch: {self.x:.3f} vs 1.2")
            
            # Group 3: t (0.4)
            if math.isclose(self.t, 0.4, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ t matches target 0.4 s")
            else:
                logs.append(f"✗ t mismatch: {self.t:.3f} vs 0.4")
            
            # Internal consistency: v_C from energy conservation
            v_C_energy = math.sqrt(self.v_D**2 - 2*self.g*self.R*(1 - self.cos37))
            if math.isclose(self.v_C, v_C_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_C consistent with energy conservation")
            else:
                logs.append(f"✗ v_C inconsistent: {self.v_C:.3f} vs energy-derived {v_C_energy:.3f}")
            
            # t from work-energy
            t_we = (0.5*self.m*self.v_B**2 + self.mu*self.m*self.g*self.L) / self.P
            if math.isclose(self.t, t_we, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ t consistent with work-energy theorem")
            else:
                logs.append(f"✗ t inconsistent: {self.t:.3f} vs work-energy-derived {t_we:.3f}")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('✗')]) == 0
    
    def test_monotonicity(self):
        """Perturb mass and check trends."""
        logs = []
        try:
            # Save original values
            m_orig = self.m
            perturb = 0.01  # 1% increase
            
            # Increase mass by 1%
            self.m = m_orig * (1 + perturb)
            
            # Recalculate expected changes using formulas (no need for full simulation)
            # F_N = m*g + m*v_D^2/R -> proportional to m
            F_N_new = self.m * self.g + self.m * self.v_D**2 / self.R
            F_N_ratio = F_N_new / self.F_N if self.F_N != 0 else float('inf')
            
            # v_C from energy: v_C^2 = v_D^2 - 2gR(1-cosα) independent of m
            v_C_new = math.sqrt(self.v_D**2 - 2*self.g*self.R*(1 - self.cos37))
            v_C_ratio = v_C_new / self.v_C if self.v_C != 0 else float('inf')
            
            # v_B = v_C*cosα -> independent of m
            v_B_new = v_C_new * self.cos37
            v_B_ratio = v_B_new / self.v_B if self.v_B != 0 else float('inf')
            
            # t from work-energy: t = (½m v_B^2 + μmgL)/P
            t_new = (0.5*self.m*v_B_new**2 + self.mu*self.m*self.g*self.L) / self.P
            t_ratio = t_new / self.t if self.t != 0 else float('inf')
            
            # Expected trends:
            # 1. F_N should increase with m (positive correlation)
            # 2. v_C, v_B should remain constant (independent of m)
            # 3. t should increase with m (positive correlation)
            
            # Check F_N
            if F_N_ratio > 1.0:
                logs.append("✓ F_N increases with mass (monotonic)")
            else:
                logs.append(f"✗ F_N monotonicity violated: ratio {F_N_ratio:.3f}")
            
            # v_C should be nearly constant (ratio ~1.0)
            if math.isclose(v_C_ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_C independent of mass (as expected)")
            else:
                logs.append(f"✗ v_C unexpectedly changed: ratio {v_C_ratio:.3f}")
            
            # v_B should be nearly constant
            if math.isclose(v_B_ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_B independent of mass (as expected)")
            else:
                logs.append(f"✗ v_B unexpectedly changed: ratio {v_B_ratio:.3f}")
            
            # t should increase with mass
            if t_ratio > 1.0:
                logs.append("✓ t increases with mass (monotonic)")
            else:
                logs.append(f"✗ t monotonicity violated: ratio {t_ratio:.3f}")
            
            # Restore original mass
            self.m = m_orig
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            # Restore mass on error
            self.m = getattr(self, 'm_orig_backup', 0.4)
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('✗')]) == 0
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        success = True
        
        # Run each test
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        success = inv_ok and trace_ok and mono_ok
        
        # Summary
        self.logs.append("\n=== SUMMARY ===")
        self.logs.append(f"Invariants: {'PASS' if inv_ok else 'FAIL'}")
        self.logs.append(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs