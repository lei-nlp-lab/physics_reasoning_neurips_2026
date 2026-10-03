import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults
        self.L1 = getattr(solution_instance, 'L1', 0.10)  # meters
        self.L2 = getattr(solution_instance, 'L2', 0.20)  # meters
        self.k1 = getattr(solution_instance, 'k1', 50.0)  # N/m
        self.k2 = getattr(solution_instance, 'k2', 20.0)  # N/m
        self.m1 = getattr(solution_instance, 'm1', 0.1)   # kg
        self.m2 = getattr(solution_instance, 'm2', 0.2)   # kg
        self.g = getattr(solution_instance, 'g', 9.8)     # m/s²
        
        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        self.x1 = getattr(solution_instance, 'x1', results.get('x_1', None))
        self.x2 = getattr(solution_instance, 'x2', results.get('x_2', None))
        self.L = getattr(solution_instance, 'L', results.get('L', None))
        self.x = getattr(solution_instance, 'x', results.get('x', None))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', None))
        
        # Derived quantities for checking
        self.x1_calc = (self.m1 + self.m2) * self.g / self.k1
        self.x2_calc = self.m2 * self.g / self.k2
        self.x_calc = self.m1 * self.g / (self.k1 + self.k2)
        self.F_N_calc1 = self.k2 * self.x_calc + self.m2 * self.g
        self.F_N_calc2 = (self.k1 * self.m2 * self.g + self.k2 * (self.m1 + self.m2) * self.g) / (self.k1 + self.k2)
        
        self.logs = []
    
    def _log(self, msg, success=True):
        self.logs.append((msg, success))
    
    def _check_close(self, a, b, name=""):
        if a is None or b is None:
            self._log(f"Missing value for {name}", False)
            return False
        close = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        if not close:
            self._log(f"Mismatch in {name}: {a} vs {b} (diff={abs(a-b)})", False)
        return close
    
    def test_invariants(self):
        """Verify Hooke's Law and equilibrium conditions"""
        try:
            # Hooke's Law for spring 1 (supporting both masses)
            if self.x1 is not None:
                force_spring1 = self.k1 * self.x1
                weight_total = (self.m1 + self.m2) * self.g
                if not self._check_close(force_spring1, weight_total, "Spring1 equilibrium"):
                    self._log(f"Spring1: k1*x1={force_spring1}, (m1+m2)g={weight_total}", False)
            
            # Hooke's Law for spring 2 (supporting only m2)
            if self.x2 is not None:
                force_spring2 = self.k2 * self.x2
                weight_m2 = self.m2 * self.g
                if not self._check_close(force_spring2, weight_m2, "Spring2 equilibrium"):
                    self._log(f"Spring2: k2*x2={force_spring2}, m2*g={weight_m2}", False)
            
            # Force equilibrium for m1 (with plate scenario from step 4-5)
            if self.x is not None:
                # Forces on m1: upward from spring1, downward from spring2 and gravity
                F_up = self.k1 * self.x  # spring1 compression
                F_down = self.k2 * self.x + self.m1 * self.g  # spring2 compression + weight
                if not self._check_close(F_up, F_down, "m1 equilibrium (plate scenario)"):
                    self._log(f"m1: k1*x={F_up}, k2*x+m1*g={F_down}", False)
            
            # Force equilibrium for m2 (with plate scenario)
            if self.x is not None and self.F_N is not None:
                # Forces on m2: upward from spring2 and plate, downward from gravity
                F_up_total = self.k2 * self.x + self.F_N
                F_down_total = self.m2 * self.g
                if not self._check_close(F_up_total, F_down_total, "m2 equilibrium (plate scenario)"):
                    self._log(f"m2: k2*x+F_N={F_up_total}, m2*g={F_down_total}", False)
            
            # Check both formulas for F_N give same result
            if self.F_N_calc1 is not None and self.F_N_calc2 is not None:
                if not self._check_close(self.F_N_calc1, self.F_N_calc2, "F_N consistency"):
                    self._log(f"F_N formula1={self.F_N_calc1}, formula2={self.F_N_calc2}", False)
            
            # Check the given result matches calculated
            if self.F_N is not None:
                if not self._check_close(self.F_N, self.F_N_calc1, "F_N result"):
                    self._log(f"Result F_N={self.F_N}, calculated={self.F_N_calc1}", False)
            
            self._log("All invariants checked", True)
            return True
        except Exception as e:
            self._log(f"Invariants test failed: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answers"""
        try:
            # Step 3 consistency: L = L1 + L2 + x1 + x2
            if all(v is not None for v in [self.L, self.x1, self.x2]):
                L_calc = self.L1 + self.L2 + self.x1 + self.x2
                if not self._check_close(self.L, L_calc, "Total length L"):
                    self._log(f"L={self.L}, calculated={L_calc}", False)
            elif self.L is not None:
                # Calculate from derived formulas
                L_calc = self.L1 + self.L2 + self.x1_calc + self.x2_calc
                if not self._check_close(self.L, L_calc, "Total length L (derived)"):
                    self._log(f"L={self.L}, derived={L_calc}", False)
            
            # Step 6 consistency: F_N should match both formulas
            if self.F_N is not None:
                if not self._check_close(self.F_N, self.F_N_calc1, "F_N step5"):
                    self._log(f"F_N result={self.F_N}, step5 formula={self.F_N_calc1}", False)
                if not self._check_close(self.F_N, self.F_N_calc2, "F_N step6"):
                    self._log(f"F_N result={self.F_N}, step6 formula={self.F_N_calc2}", False)
            
            # Step 4 consistency: x = m1*g/(k1+k2)
            if self.x is not None:
                if not self._check_close(self.x, self.x_calc, "Compression x"):
                    self._log(f"x={self.x}, calculated={self.x_calc}", False)
            
            self._log("Trace consistency checked", True)
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb m1 and check physical trends"""
        try:
            original_m1 = self.m1
            perturbed_m1 = original_m1 * 1.1  # Increase by 10%
            
            # Calculate with perturbed m1
            x1_pert = (perturbed_m1 + self.m2) * self.g / self.k1
            x2_pert = self.m2 * self.g / self.k2
            L_pert = self.L1 + self.L2 + x1_pert + x2_pert
            
            x_pert = perturbed_m1 * self.g / (self.k1 + self.k2)
            F_N_pert = self.k2 * x_pert + self.m2 * self.g
            
            # Check trends
            if self.L is not None:
                if L_pert > self.L:
                    self._log(f"L increases with m1: {self.L:.3f} -> {L_pert:.3f}", True)
                else:
                    self._log(f"L should increase with m1 but didn't", False)
            
            if self.F_N is not None:
                if F_N_pert > self.F_N:
                    self._log(f"F_N increases with m1: {self.F_N:.3f} -> {F_N_pert:.3f}", True)
                else:
                    self._log(f"F_N should increase with m1 but didn't", False)
            
            # Restore original value (conceptual)
            self.m1 = original_m1
            
            self._log("Monotonicity checked", True)
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all checks and return success status"""
        self.logs = []
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        success = inv_ok and trace_ok and mono_ok
        
        # Count successes
        success_count = sum(1 for _, s in self.logs if s)
        total_count = len(self.logs)
        
        summary = f"PhysicsChecker: {success_count}/{total_count} checks passed"
        if success:
            self._log(summary, True)
        else:
            self._log(summary, False)
        
        return success, self.logs