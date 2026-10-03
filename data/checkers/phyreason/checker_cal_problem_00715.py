import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.R = getattr(solution_instance, 'R', 0.125)
        self.d = getattr(solution_instance, 'd', 1.0)  # separation between rings
        self.E = getattr(solution_instance, 'E', 3.0)
        self.m = getattr(solution_instance, 'm', 0.06)
        self.r = getattr(solution_instance, 'r', 1.5)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.sin_theta = getattr(solution_instance, 'sin_theta', 0.6)
        self.cos_theta = getattr(solution_instance, 'cos_theta', 0.8)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safe extraction of results from solution_instance
        results = getattr(solution_instance, 'results', {})
        self.F = getattr(solution_instance, 'F', results.get('F', 0.0))
        self.I = getattr(solution_instance, 'I', results.get('I', 0.0))
        self.B = getattr(solution_instance, 'B', results.get('B', 0.0))
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', 0.0))
        
        # Initialize symbolic variables
        self._init_symbols()
        
    def _init_symbols(self):
        """Define symbolic variables for physical laws."""
        self.sym_m, self.sym_g, self.sym_theta = sp.symbols('m g theta', positive=True)
        self.sym_F, self.sym_I, self.sym_B, self.sym_L, self.sym_v, self.sym_R, self.sym_F_N = sp.symbols(
            'F I B L v R F_N', positive=True
        )
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        
        try:
            # 1. Force equilibrium at θ=37°: mg = F tanθ (from step_1)
            lhs_eq = self.m * self.g
            rhs_eq = self.F * (self.sin_theta / self.cos_theta)  # tanθ = sinθ/cosθ
            if self._is_close(lhs_eq, rhs_eq):
                logs.append("✓ Force equilibrium satisfied: mg = F tanθ")
            else:
                logs.append(f"✗ Force equilibrium violated: mg={lhs_eq:.3f}, F tanθ={rhs_eq:.3f}")
            
            # 2. Ohm's law: I = E / r (from step_2)
            I_expected = self.E / self.r
            if self._is_close(self.I, I_expected):
                logs.append("✓ Ohm's law satisfied: I = E/r")
            else:
                logs.append(f"✗ Ohm's law violation: I={self.I:.3f}, E/r={I_expected:.3f}")
            
            # 3. Ampere's force law: F = B I L (L = d = 1m) (from step_3)
            F_expected = self.B * self.I * self.d
            if self._is_close(self.F, F_expected):
                logs.append("✓ Ampere's force law satisfied: F = B I d")
            else:
                logs.append(f"✗ Ampere's force law violation: F={self.F:.3f}, B I d={F_expected:.3f}")
            
            # 4. Work-energy theorem: mgR(1 - sinθ) = ½mv² (from step_4)
            energy_lhs = self.m * self.g * self.R * (1 - self.sin_theta)
            energy_rhs = 0.5 * self.m * (self.v ** 2)
            if self._is_close(energy_lhs, energy_rhs):
                logs.append("✓ Work-energy theorem satisfied: mgR(1-sinθ) = ½mv²")
            else:
                logs.append(f"✗ Work-energy violation: LHS={energy_lhs:.3f}, RHS={energy_rhs:.3f}")
            
            # 5. Newton's second law (circular motion): 2F_N - mg = m v²/R (from step_5)
            newton_lhs = 2 * self.F_N - self.m * self.g
            newton_rhs = self.m * (self.v ** 2) / self.R
            if self._is_close(newton_lhs, newton_rhs):
                logs.append("✓ Newton's second law (circular) satisfied: 2F_N - mg = mv²/R")
            else:
                logs.append(f"✗ Newton's second law violation: 2F_N - mg={newton_lhs:.3f}, mv²/R={newton_rhs:.3f}")
                
        except Exception as e:
            logs.append(f"✗ Invariant test failed with error: {str(e)}")
            
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # Check step 1 -> step 3: F = 0.8N should lead to B = F/(I*d)
            B_from_F = self.F / (self.I * self.d)
            if self._is_close(self.B, B_from_F):
                logs.append("✓ B consistent with F and I: B = F/(I d)")
            else:
                logs.append(f"✗ B inconsistency: B={self.B:.3f}, F/(I d)={B_from_F:.3f}")
            
            # Check step 4 -> step 5: v = 1 m/s should satisfy energy and circular motion
            v_expected = math.sqrt(2 * self.g * self.R * (1 - self.sin_theta))
            if self._is_close(self.v, v_expected):
                logs.append("✓ v consistent with energy conservation")
            else:
                logs.append(f"✗ v inconsistency: v={self.v:.3f}, √(2gR(1-sinθ))={v_expected:.3f}")
            
            # Check step 5: F_N should satisfy both circular motion and equilibrium
            F_N_expected = (self.m * (self.v ** 2) / self.R + self.m * self.g) / 2
            if self._is_close(self.F_N, F_N_expected):
                logs.append("✓ F_N consistent with circular motion formula")
            else:
                logs.append(f"✗ F_N inconsistency: F_N={self.F_N:.3f}, expected={F_N_expected:.3f}")
                
        except Exception as e:
            logs.append(f"✗ Trace consistency test failed with error: {str(e)}")
            
        return logs
    
    def test_monotonicity(self):
        """Perturb mass m and check physical trends."""
        logs = []
        original_m = self.m
        
        try:
            # Perturb mass upward by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate F using equilibrium: F = mg / tanθ
            perturbed_F = perturbed_m * self.g / (self.sin_theta / self.cos_theta)
            
            # Recalculate v using energy: v = √(2gR(1-sinθ)) [independent of m!]
            # Actually v formula doesn't depend on m, but let's check the relationship
            perturbed_v = math.sqrt(2 * self.g * self.R * (1 - self.sin_theta))
            
            # Recalculate F_N using circular motion: F_N = (m v²/R + mg)/2
            perturbed_F_N = (perturbed_m * perturbed_v**2 / self.R + perturbed_m * self.g) / 2
            
            # Check trends: increasing m should increase both F and F_N
            if perturbed_F > self.F and perturbed_F_N > self.F_N:
                logs.append("✓ Monotonicity: increasing mass increases both F and F_N as expected")
            else:
                logs.append(f"✗ Monotonicity violation: F={self.F:.3f}->{perturbed_F:.3f}, F_N={self.F_N:.3f}->{perturbed_F_N:.3f}")
            
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed with error: {str(e)}")
        finally:
            # Restore original mass
            self.m = original_m
            
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        logs = ["=== Physics Checker Report ==="]
        success = True
        
        # Run individual test suites
        invariant_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        monotonicity_logs = self.test_monotonicity()
        
        # Check for any failures in invariant tests (critical)
        if any("✗" in log for log in invariant_logs):
            success = False
        
        # Combine logs
        logs.append("\n--- Invariant Tests ---")
        logs.extend(invariant_logs)
        
        logs.append("\n--- Trace Consistency Tests ---")
        logs.extend(trace_logs)
        
        logs.append("\n--- Monotonicity Tests ---")
        logs.extend(monotonicity_logs)
        
        logs.append(f"\n=== Overall Status: {'PASS' if success else 'FAIL'} ===")
        return success, logs