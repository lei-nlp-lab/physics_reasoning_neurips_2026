import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants from problem statement (with safe defaults)
        self.g = 9.8  # m/s^2
        self.H = 0.4  # m
        self.theta = math.radians(37)  # rad
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        self.r = 0.1  # m
        self.R = 0.2  # m
        self.mu = 1.0/12.0
        
        # Extract key results with safe fallbacks
        self.v0 = getattr(solution_instance, 'v0', self.results.get('v_0', 0.0))
        self.v_Emin = getattr(solution_instance, 'v_Emin', self.results.get('v_Emin', 0.0))
        self.v_Bmin = getattr(solution_instance, 'v_Bmin', self.results.get('v_Bmin', 0.0))
        self.v_G = getattr(solution_instance, 'v_G', self.results.get('v_G', 0.0))
        self.h_min = getattr(solution_instance, 'h_min', self.results.get('h_min', 0.0))
        self.x_max = getattr(solution_instance, 'x_max', self.results.get('x_max', 0.0))
        self.v_A_prime = getattr(solution_instance, 'v_A_prime', self.results.get('v_A\'', 0.0))
        self.v_A_min = getattr(solution_instance, 'v_A_min', self.results.get('v_A,min', 0.0))
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        self.H_plus_r_minus_y = getattr(solution_instance, 'H_plus_r_minus_y', 
                                        self.results.get('H+r-y', 0.0))
    
    def test_invariants(self):
        logs = []
        
        try:
            # 1. Check work-energy theorem for slider on AB (step 1)
            # Equation: mgh - μmgcosθ*(h/sinθ) = 0.5*m*v0²
            # All masses cancel, so: g*h - μ*g*cosθ*(h/sinθ) = 0.5*v0²
            h = self.inputs.get('h', 0.0)
            if h > 0:
                lhs = self.g * h - self.mu * self.g * self.cos_theta * (h / self.sin_theta)
                rhs = 0.5 * (self.v0 ** 2)
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("✓ Work-energy theorem holds for slider on AB")
                else:
                    logs.append(f"✗ Work-energy theorem violation on AB: lhs={lhs:.4f}, rhs={rhs:.4f}")
            
            # 2. Check centripetal condition at point E (step 2)
            # Equation: mg = m*v_Emin²/R → v_Emin = sqrt(g*R)
            v_Emin_expected = math.sqrt(self.g * self.R)
            if math.isclose(self.v_Emin, v_Emin_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Centripetal condition at point E satisfied")
            else:
                logs.append(f"✗ Centripetal condition at E: v_Emin={self.v_Emin:.4f}, expected={v_Emin_expected:.4f}")
            
            # 3. Check energy conservation from B to E (step 3)
            # Equation: 0.5*m*v_Bmin² = 0.5*m*v_Emin² + mg(R+r)
            lhs_energy = 0.5 * (self.v_Bmin ** 2)
            rhs_energy = 0.5 * (self.v_Emin ** 2) + self.g * (self.R + self.r)
            if math.isclose(lhs_energy, rhs_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Energy conservation from B to E holds")
            else:
                logs.append(f"✗ Energy conservation B→E: lhs={lhs_energy:.4f}, rhs={rhs_energy:.4f}")
            
            # 4. Check elastic collision conditions (steps 5 & 6)
            # Momentum: v_A_min = v_A_prime + v_Bmin
            momentum_eq = self.v_A_min - (self.v_A_prime + self.v_Bmin)
            if math.isclose(momentum_eq, 0, abs_tol=1e-2):
                logs.append("✓ Momentum conservation in collision")
            else:
                logs.append(f"✗ Momentum conservation: diff={momentum_eq:.4f}")
            
            # Energy: 0.5*v_A_min² = 0.5*v_A_prime² + 0.5*v_Bmin²
            energy_eq = 0.5*(self.v_A_min**2) - (0.5*(self.v_A_prime**2) + 0.5*(self.v_Bmin**2))
            if math.isclose(energy_eq, 0, abs_tol=1e-2):
                logs.append("✓ Energy conservation in elastic collision")
            else:
                logs.append(f"✗ Energy in collision: diff={energy_eq:.4f}")
            
            # 5. Check v_A_prime = 0 (from step 7 solution)
            if math.isclose(self.v_A_prime, 0, abs_tol=1e-2):
                logs.append("✓ Slider comes to rest after collision (v_A' = 0)")
            else:
                logs.append(f"✗ v_A' should be 0, got {self.v_A_prime:.4f}")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        
        try:
            # 1. Check h_min consistency (steps 8-9)
            # From step 8: m*g*h_min - μ*m*g*cosθ*(h_min/sinθ) = 0.5*m*v_A_min²
            # All masses cancel
            lhs_h = self.g * self.h_min - self.mu * self.g * self.cos_theta * (self.h_min / self.sin_theta)
            rhs_h = 0.5 * (self.v_A_min ** 2)
            if math.isclose(lhs_h, rhs_h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ h_min consistent with work-energy theorem")
            else:
                logs.append(f"✗ h_min inconsistent: lhs={lhs_h:.4f}, rhs={rhs_h:.4f}")
            
            # 2. Check v_Bmin from v_A_min and v_A' (step 7)
            # With v_A' = 0, v_Bmin should equal v_A_min from momentum conservation
            if math.isclose(self.v_Bmin, self.v_A_min, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_Bmin = v_A_min when v_A' = 0")
            else:
                logs.append(f"✗ v_Bmin={self.v_Bmin:.4f} ≠ v_A_min={self.v_A_min:.4f}")
            
            # 3. Check projectile optimization (steps 10-13)
            # From step 10: 0.5*v_G² = 0.5*v_Emin² + g(R+y)  where y is vertical distance from D to G
            # But we don't have y directly. Instead check that x_max = 0.8 is consistent with
            # optimization of x = 2√((0.5-y)(0.3+y))
            
            # The function x(y) = 2*sqrt((0.5-y)*(0.3+y)) has maximum when derivative = 0
            # Derivative: d/dy [2*sqrt((0.5-y)*(0.3+y))] = 0 → y = 0.1
            # Then x_max = 2*sqrt((0.5-0.1)*(0.3+0.1)) = 2*sqrt(0.4*0.4) = 2*0.4 = 0.8
            if math.isclose(self.x_max, 0.8, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ x_max = 0.8 m from optimization")
            else:
                logs.append(f"✗ x_max={self.x_max:.4f}, expected 0.8")
            
            # 4. Check projectile equations (step 11)
            # From: H+r-y = 0.5*g*t² and x = v_G*t
            # Eliminate t: t = x/v_G, substitute → H+r-y = 0.5*g*(x/v_G)²
            # Rearranged: v_G = x * sqrt(g/(2*(H+r-y)))
            # We don't have y, but we can check that at optimal y=0.1, x=0.8, H=0.4, r=0.1
            # H+r-y = 0.4+0.1-0.1 = 0.4
            # Then v_G = 0.8 * sqrt(9.8/(2*0.4)) = 0.8 * sqrt(9.8/0.8) = 0.8 * sqrt(12.25) = 0.8*3.5 = 2.8
            # Let's compute expected v_G from energy (step 10): 0.5*v_G² = 0.5*v_Emin² + g(R+y)
            # With y=0.1, R=0.2: 0.5*v_G² = 0.5*(√(gR))² + g*(0.3) = 0.5*gR + 0.3g = 0.5*9.8*0.2 + 2.94 = 0.98 + 2.94 = 3.92
            # So v_G = √(7.84) = 2.8
            v_G_expected = math.sqrt(self.v_Emin**2 + 2*self.g*(self.R + 0.1))  # using y=0.1
            if math.isclose(self.v_G, v_G_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_G consistent with energy conservation")
            else:
                logs.append(f"✗ v_G={self.v_G:.4f}, expected={v_G_expected:.4f}")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_h = self.inputs.get('h', 0.0)
        
        try:
            if original_h > 0:
                # Test 1: Increase h by 10%, v0 should increase
                h_perturbed = original_h * 1.1
                v0_original = self.v0
                
                # Recalculate v0 using work-energy theorem: v0² = 2g[h(1 - μ*cotθ)]
                # where cotθ = cosθ/sinθ
                cot_theta = self.cos_theta / self.sin_theta
                factor = 1 - self.mu * cot_theta
                v0_new = math.sqrt(2 * self.g * h_perturbed * factor)
                
                if v0_new > v0_original:
                    logs.append("✓ v0 increases with increasing h (monotonic)")
                else:
                    logs.append(f"✗ v0 monotonicity failed: v0_orig={v0_original:.4f}, v0_new={v0_new:.4f}")
                
                # Test 2: Decrease h by 10%, v0 should decrease
                h_perturbed2 = original_h * 0.9
                v0_new2 = math.sqrt(2 * self.g * h_perturbed2 * factor)
                
                if v0_new2 < v0_original:
                    logs.append("✓ v0 decreases with decreasing h (monotonic)")
                else:
                    logs.append(f"✗ v0 monotonicity failed on decrease: v0_orig={v0_original:.4f}, v0_new={v0_new2:.4f}")
            
            # Test 3: h_min should increase if v_Bmin increases
            # From collision: v_Bmin = v_A_min when v_A' = 0
            # From work-energy: v_A_min² = 2gh_min(1 - μ*cotθ)
            # So h_min ∝ v_Bmin²
            original_v_Bmin = self.v_Bmin
            perturbed_v_Bmin = original_v_Bmin * 1.1
            cot_theta = self.cos_theta / self.sin_theta
            factor = 1 - self.mu * cot_theta
            
            h_min_original = self.h_min
            h_min_perturbed = (perturbed_v_Bmin**2) / (2 * self.g * factor)
            
            if h_min_perturbed > h_min_original:
                logs.append("✓ h_min increases with v_Bmin (monotonic)")
            else:
                logs.append(f"✗ h_min monotonicity failed: h_min_orig={h_min_original:.4f}, h_min_pert={h_min_perturbed:.4f}")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        try:
            # Run each test suite
            invariants_logs = self.test_invariants()
            trace_logs = self.test_trace_consistency()
            monotonicity_logs = self.test_monotonicity()
            
            all_logs.extend(["=== Invariants Test ==="] + invariants_logs)
            all_logs.extend(["", "=== Trace Consistency Test ==="] + trace_logs)
            all_logs.extend(["", "=== Monotonicity Test ==="] + monotonicity_logs)
            
            # Check for failures (logs starting with ✗)
            for log in invariants_logs + trace_logs + monotonicity_logs:
                if log.startswith("✗"):
                    success = False
                    break
            
        except Exception as e:
            all_logs.append(f"Critical error in run_all_tests: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs