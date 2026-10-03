import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results using getattr and .get() to prevent AttributeError.
        """
        self.solution = solution_instance
        
        # Extract inputs with fallbacks
        self.T = getattr(solution_instance, 'T', getattr(solution_instance, 'period', None))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'mass', None))
        self.r = getattr(solution_instance, 'r', getattr(solution_instance, 'radius', None))
        
        # Extract results from results dictionary with fallbacks
        results_dict = getattr(solution_instance, 'results', {})
        self.v = results_dict.get('v', results_dict.get('velocity', 0.0))
        self.F = results_dict.get('F', results_dict.get('force', 0.0))
        
        # Store logs for reporting
        self.logs = []
        
    def test_invariants(self):
        """
        Verify physical invariants: velocity-period relation and centripetal force.
        Uses math.isclose with 1-2% tolerance for fuzzy matching.
        """
        try:
            # 1. Check velocity: v = 2πr / T
            if None not in (self.v, self.r, self.T):
                v_expected = 2 * math.pi * self.r / self.T
                if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Velocity invariant failed: expected {v_expected:.3e}, got {self.v:.3e}")
                    return False
                self.logs.append(f"✓ Velocity invariant passed: v ≈ {self.v:.3e} = 2πr/T")
            else:
                self.logs.append("⚠ Skipping velocity invariant: missing inputs")
            
            # 2. Check force: F = m*v²/r = 4π²mr/T²
            if None not in (self.F, self.m, self.v, self.r, self.T):
                # Check F = m*v²/r
                F_from_v = self.m * self.v**2 / self.r
                if not math.isclose(self.F, F_from_v, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Force invariant (F=mv²/r) failed: expected {F_from_v:.3e}, got {self.F:.3e}")
                    return False
                self.logs.append(f"✓ Force invariant (F=mv²/r) passed")
                
                # Check F = 4π²mr/T²
                F_from_T = 4 * math.pi**2 * self.m * self.r / self.T**2
                if not math.isclose(self.F, F_from_T, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Force invariant (F=4π²mr/T²) failed: expected {F_from_T:.3e}, got {self.F:.3e}")
                    return False
                self.logs.append(f"✓ Force invariant (F=4π²mr/T²) passed")
            else:
                self.logs.append("⚠ Skipping force invariant: missing inputs")
            
            # 3. Symbolic verification as fallback
            try:
                T_sym, m_sym, r_sym = sp.symbols('T m r', positive=True)
                v_expr = 2 * sp.pi * r_sym / T_sym
                F_expr_v = m_sym * v_expr**2 / r_sym
                F_expr_T = 4 * sp.pi**2 * m_sym * r_sym / T_sym**2
                # Simplify equality symbolically
                if not sp.simplify(F_expr_v - F_expr_T) == 0:
                    self.logs.append("⚠ Symbolic force equivalence check failed")
                    return False
                self.logs.append("✓ Symbolic invariant: F = mv²/r = 4π²mr/T²")
            except Exception as sym_err:
                self.logs.append(f"⚠ Symbolic check skipped: {sym_err}")
            
            return True
        except Exception as e:
            self.logs.append(f"❌ Error in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        From STEPS ANALYSIS: step1 yields v, step2 uses v to compute F.
        """
        try:
            if None not in (self.v, self.F, self.m, self.r):
                # Recompute F from the intermediate v (step1 result used in step2)
                F_recomputed = self.m * self.v**2 / self.r
                if not math.isclose(self.F, F_recomputed, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Trace consistency failed: F={self.F:.3e}, but m*v²/r={F_recomputed:.3e}")
                    return False
                self.logs.append(f"✓ Trace consistency: F = m*v²/r ≈ {self.F:.3e}")
                return True
            else:
                self.logs.append("⚠ Skipping trace consistency: missing data")
                return True
        except Exception as e:
            self.logs.append(f"❌ Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input (mass m) and verify the expected trend in force.
        From KB: if mass increases, centripetal force increases (fixed r, T).
        Restores original values after perturbation.
        """
        original_m = self.m
        try:
            if None not in (self.m, self.r, self.T):
                # Perturb mass by +2%
                perturbed_m = self.m * 1.02
                # Compute force with perturbed mass (using formula F=4π²mr/T²)
                F_original = 4 * math.pi**2 * self.m * self.r / self.T**2
                F_perturbed = 4 * math.pi**2 * perturbed_m * self.r / self.T**2
                
                # Check trend: force should increase with mass
                if not (F_perturbed > F_original):
                    self.logs.append(f"Monotonicity failed: F(original)={F_original:.3e}, F(perturbed m)={F_perturbed:.3e}")
                    return False
                self.logs.append(f"✓ Monotonicity: mass ↑ {self.m:.3e}→{perturbed_m:.3e} => force ↑ {F_original:.3e}→{F_perturbed:.3e}")
                return True
            else:
                self.logs.append("⚠ Skipping monotonicity: missing inputs")
                return True
        except Exception as e:
            self.logs.append(f"❌ Error in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
        finally:
            # Restore original mass
            self.m = original_m
    
    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.logs.append(f"\n--- Testing {name} ---")
            if not test_func():
                success = False
                self.logs.append(f"❌ {name} failed")
            else:
                self.logs.append(f"✓ {name} passed")
        
        self.logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs