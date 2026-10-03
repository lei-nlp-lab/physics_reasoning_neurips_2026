import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.logs = []
        
        # Extract input parameters with safe defaults
        self.L = getattr(solution_instance, 'L', 8.0)
        self.B = getattr(solution_instance, 'B', 2.0)
        self.F = getattr(solution_instance, 'F', 32.0)
        self.theta = getattr(solution_instance, 'theta', math.pi/2)  # perpendicular = 90°
        
        # Extract results safely - from results dict or direct attribute
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.I = solution_instance.results.get('I', 0.0)
        else:
            self.I = getattr(solution_instance, 'I', 0.0)
        
        # For symbolic calculations
        self.symbols = {
            'I': sp.symbols('I'),
            'L': sp.symbols('L'),
            'B': sp.symbols('B'),
            'theta': sp.symbols('theta'),
            'F_mag': sp.symbols('F_mag')
        }
    
    def _log(self, message, success=True):
        """Helper for logging test results"""
        self.logs.append({
            'message': message,
            'success': success
        })
        return success
    
    def test_invariants(self):
        """Test physical laws: F = I*L*B*sin(theta)"""
        try:
            # Symbolic law
            law = sp.Eq(self.symbols['F_mag'], 
                       self.symbols['I'] * self.symbols['L'] * 
                       self.symbols['B'] * sp.sin(self.symbols['theta']))
            
            # Numerical check with fuzzy matching
            sin_theta = math.sin(self.theta)
            expected_F = self.I * self.L * self.B * sin_theta
            
            if math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"✓ Force law invariant holds: F={self.F:.2f} ≈ I*L*B*sinθ={expected_F:.2f}")
            else:
                return self._log(f"✗ Force law violation: F={self.F:.2f} ≠ I*L*B*sinθ={expected_F:.2f}", False)
                
        except Exception as e:
            return self._log(f"✗ Invariant test failed: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Check step-by-step consistency from STEPS ANALYSIS"""
        try:
            # From STEPS ANALYSIS: step_1 computes I = F/(B*L) = 32/(2*8) = 2.0
            sin_theta = math.sin(self.theta)
            expected_I = self.F / (self.B * self.L * sin_theta)
            
            if math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"✓ Trace consistent: I={self.I:.2f}A matches derived I=F/(B*L)={expected_I:.2f}A")
            else:
                return self._log(f"✗ Trace inconsistent: I={self.I:.2f}A ≠ F/(B*L)={expected_I:.2f}A", False)
                
        except ZeroDivisionError:
            return self._log(f"✗ Trace test failed: division by zero (B={self.B}, L={self.L}, sinθ={sin_theta})", False)
        except Exception as e:
            return self._log(f"✗ Trace test failed: {str(e)}", False)
    
    def test_monotonicity(self):
        """Test proportional relationships from KB Trend"""
        try:
            original_I = float(self.I)
            original_F = float(self.F)
            
            # Test 1: Increase B by 10%, F should increase proportionally
            B_perturbed = self.B * 1.1
            F_new_B = original_I * self.L * B_perturbed * math.sin(self.theta)
            
            if not math.isclose(F_new_B, original_F * 1.1, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"✗ Monotonicity violated: 10% B increase → {F_new_B:.2f}N, expected {original_F*1.1:.2f}N", False)
            
            # Test 2: Increase L by 10%, F should increase proportionally
            L_perturbed = self.L * 1.1
            F_new_L = original_I * L_perturbed * self.B * math.sin(self.theta)
            
            if not math.isclose(F_new_L, original_F * 1.1, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"✗ Monotonicity violated: 10% L increase → {F_new_L:.2f}N, expected {original_F*1.1:.2f}N", False)
            
            # Test 3: Increase I by 10%, F should increase proportionally
            I_perturbed = original_I * 1.1
            F_new_I = I_perturbed * self.L * self.B * math.sin(self.theta)
            
            if not math.isclose(F_new_I, original_F * 1.1, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"✗ Monotonicity violated: 10% I increase → {F_new_I:.2f}N, expected {original_F*1.1:.2f}N", False)
            
            # Test 4: Angle dependence (sinθ monotonic in [0,π/2])
            if 0 <= self.theta <= math.pi/2:
                theta_perturbed = self.theta * 1.1
                if theta_perturbed <= math.pi/2:
                    F_new_theta = original_I * self.L * self.B * math.sin(theta_perturbed)
                    if F_new_theta < original_F:
                        return self._log(f"✗ Angle monotonicity violated: θ↑{theta_perturbed:.3f} → F↓{F_new_theta:.2f}N", False)
            
            return self._log(f"✓ Monotonicity checks passed: F ∝ I, ∝ B, ∝ L, and follows sinθ trend")
            
        except Exception as e:
            return self._log(f"✗ Monotonicity test failed: {str(e)}", False)
    
    def run_all_tests(self):
        """Execute all physics checks"""
        self.logs = []
        success = True
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]
        
        for test in tests:
            try:
                if not test():
                    success = False
            except Exception as e:
                self._log(f"✗ Test crashed: {str(e)}\n{traceback.format_exc()}", False)
                success = False
        
        return success, self.logs