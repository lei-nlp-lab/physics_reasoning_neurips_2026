import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults from problem statement
        self.M = getattr(solution_instance, 'M', 2.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.L = getattr(solution_instance, 'L', 0.5)
        self.v0 = getattr(solution_instance, 'v0', 4.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safely extract results from solution_instance.results with fallback defaults
        self.results = getattr(solution_instance, 'results', {})
        self.v1 = self.results.get('v_1', 0.0)
        self.F_plus_mg = self.results.get('F+mg', 0.0)
        self.F = self.results.get('F', 0.0)
        self.F_prime = self.results.get("F'", 0.0)
        
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
        
    def test_invariants(self):
        """Verify energy conservation and Newton's laws."""
        try:
            # 1. Conservation of mechanical energy for the ball
            K0 = 0.5 * self.m * self.v0**2
            K1 = 0.5 * self.m * self.v1**2
            U0 = 0.0  # Initial height = 0 (horizontal position)
            U1 = self.m * self.g * self.L  # Highest point height = L
            E0 = K0 + U0
            E1 = K1 + U1
            if not math.isclose(E0, E1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Energy conservation failed: E0={E0:.3f}, E1={E1:.3f}")
                return False
            self.log(f"Energy conservation passed: E0={E0:.3f} ≈ E1={E1:.3f}")
            
            # 2. Newton's second law at highest point (centripetal force)
            F_net_calc = self.m * self.v1**2 / self.L
            if not math.isclose(self.F_plus_mg, F_net_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Newton's second law failed: F+mg={self.F_plus_mg:.3f}, m*v1²/L={F_net_calc:.3f}")
                return False
            self.log(f"Newton's second law passed: F+mg={self.F_plus_mg:.3f} ≈ m*v1²/L={F_net_calc:.3f}")
            
            # 3. Force decomposition: F_plus_mg = F + m*g
            F_calc = self.F_plus_mg - self.m * self.g
            if not math.isclose(self.F, F_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Force decomposition failed: F={self.F:.3f}, (F+mg)-mg={F_calc:.3f}")
                return False
            self.log(f"Force decomposition passed: F={self.F:.3f} ≈ (F+mg)-mg={F_calc:.3f}")
            
            # 4. Newton's third law: F' = F in magnitude
            if not math.isclose(self.F_prime, self.F, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Newton's third law failed: F'={self.F_prime:.3f}, F={self.F:.3f}")
                return False
            self.log(f"Newton's third law passed: F'={self.F_prime:.3f} ≈ F={self.F:.3f}")
            
            return True
        except Exception as e:
            self.log(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Step 1: Energy conservation → v1
            v1_calc = math.sqrt(self.v0**2 - 2 * self.g * self.L)
            if not math.isclose(self.v1, v1_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace consistency step 1 failed: v1={self.v1:.3f}, sqrt(v0²-2gL)={v1_calc:.3f}")
                return False
            self.log(f"Step 1 passed: v1={self.v1:.3f} ≈ sqrt(v0²-2gL)={v1_calc:.3f}")
            
            # Step 2: Newton's second law → F+mg
            F_plus_mg_calc = self.m * v1_calc**2 / self.L
            if not math.isclose(self.F_plus_mg, F_plus_mg_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace consistency step 2 failed: F+mg={self.F_plus_mg:.3f}, m*v1²/L={F_plus_mg_calc:.3f}")
                return False
            self.log(f"Step 2 passed: F+mg={self.F_plus_mg:.3f} ≈ m*v1²/L={F_plus_mg_calc:.3f}")
            
            # Step 3: Solve for F
            F_calc = F_plus_mg_calc - self.m * self.g
            if not math.isclose(self.F, F_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace consistency step 3 failed: F={self.F:.3f}, (F+mg)-mg={F_calc:.3f}")
                return False
            self.log(f"Step 3 passed: F={self.F:.3f} ≈ (F+mg)-mg={F_calc:.3f}")
            
            # Step 4: Newton's third law → F'
            if not math.isclose(self.F_prime, F_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace consistency step 4 failed: F'={self.F_prime:.3f}, F={F_calc:.3f}")
                return False
            self.log(f"Step 4 passed: F'={self.F_prime:.3f} ≈ F={F_calc:.3f}")
            
            return True
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and check force F trend, then restore."""
        try:
            # Original values
            m_original = self.m
            # Perturb m by +10%
            self.m *= 1.10
            
            # Recompute v1 (independent of m due to energy conservation form)
            v1_perturbed = math.sqrt(self.v0**2 - 2 * self.g * self.L)
            # Recompute F using formula F = m*(v1²/L - g)
            F_perturbed = self.m * (v1_perturbed**2 / self.L - self.g)
            
            # Expected trend: F ∝ m (from formula)
            ratio_F = F_perturbed / self.F if self.F != 0 else float('inf')
            ratio_m = self.m / m_original
            
            # Check if ratio is close (within 2%)
            if not math.isclose(ratio_F, ratio_m, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Monotonicity failed: F ratio={ratio_F:.3f}, m ratio={ratio_m:.3f}")
                self.m = m_original  # Restore original
                return False
            
            self.log(f"Monotonicity passed: F∝m, F ratio={ratio_F:.3f} ≈ m ratio={ratio_m:.3f}")
            self.m = m_original  # Restore original
            return True
        except Exception as e:
            self.log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
            self.m = getattr(self, 'm_original', 1.0)  # Attempt to restore
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs.clear()
        success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} ---")
            if not test_func():
                success = False
                self.log(f"{test_name} FAILED")
            else:
                self.log(f"{test_name} PASSED")
        
        return success, self.logs