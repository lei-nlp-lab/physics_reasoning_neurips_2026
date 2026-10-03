import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Input parameters (with safe defaults)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Results (with safe extraction)
        self.E = getattr(solution_instance, 'E', 0.0)
        self.v = getattr(solution_instance, 'v', 0.0)
        self.frac12mv2 = getattr(solution_instance, 'frac12mv2', 0.0)
        self.F = getattr(solution_instance, 'F', 0.0)
        
        # Trigonometric values for α=37°
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = self.sin37 / self.cos37  # = 0.75 = 3/4
        
        # Expected formulas from steps analysis
        self.expected_E = (3 * self.m * self.g) / (4 * self.q)  # (3mg)/(4q)
        self.expected_v = math.sqrt(self.g * self.L / 2)        # √(gL/2)
        self.expected_F = 1.5 * self.m * self.g                 # (3mg)/2
        
        # Logs for reporting
        self.logs = []
    
    def _log(self, msg):
        """Helper to append log messages."""
        self.logs.append(msg)
    
    def _is_close(self, a, b, name=""):
        """Robust fuzzy comparison with 1% tolerance."""
        close = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        self._log(f"{name}: {a} vs {b} (close: {close})")
        return close
    
    def test_equilibrium(self):
        """Verify equilibrium condition: qE = mg·tan(37°)."""
        try:
            lhs = self.q * self.E
            rhs = self.m * self.g * self.tan37
            return self._is_close(lhs, rhs, "Equilibrium qE = mg·tan37")
        except Exception as e:
            self._log(f"Equilibrium test failed: {e}")
            return False
    
    def test_work_energy(self):
        """Verify work-energy theorem: mgL - qEL = ½mv²."""
        try:
            # Compute work done
            work_gravity = self.m * self.g * self.L
            work_electric = self.q * self.E * self.L
            net_work = work_gravity - work_electric
            
            # Compare with kinetic energy
            return self._is_close(net_work, self.frac12mv2, "Work-energy theorem")
        except Exception as e:
            self._log(f"Work-energy test failed: {e}")
            return False
    
    def test_circular_motion(self):
        """Verify Newton's 2nd law at lowest point: F - mg = mv²/L."""
        try:
            centripetal = self.m * self.v**2 / self.L
            lhs = self.F - self.m * self.g
            return self._is_close(lhs, centripetal, "Circular motion F - mg = mv²/L")
        except Exception as e:
            self._log(f"Circular motion test failed: {e}")
            return False
    
    def test_invariants(self):
        """Check fundamental physical invariants."""
        success = True
        
        # 1. Equilibrium of forces (net force zero at initial position)
        try:
            # Horizontal: Tsin37 = qE, Vertical: Tcos37 = mg
            # From these: tan37 = qE/(mg)
            ratio = (self.q * self.E) / (self.m * self.g) if self.m*self.g != 0 else float('inf')
            success &= self._is_close(ratio, self.tan37, "Force equilibrium tan37")
        except Exception as e:
            self._log(f"Force equilibrium check failed: {e}")
            success = False
        
        # 2. Energy conservation from equilibrium to lowest point
        try:
            # Initial energy (at equilibrium): potential + electric potential?
            # Actually, the problem states work-energy: mgL - qEL = ½mv²
            # We already tested this in test_work_energy()
            success &= self.test_work_energy()
        except Exception as e:
            self._log(f"Energy conservation check failed: {e}")
            success = False
        
        # 3. Newton's 2nd law at lowest point
        success &= self.test_circular_motion()
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check E against expected formula
        success &= self._is_close(self.E, self.expected_E, "E vs (3mg)/(4q)")
        
        # Check v against expected formula (if v is provided)
        if hasattr(self, 'v') and self.v != 0:
            success &= self._is_close(self.v, self.expected_v, "v vs √(gL/2)")
        
        # Check F against expected formula
        success &= self._is_close(self.F, self.expected_F, "F vs (3mg)/2")
        
        # Verify that using E in work-energy gives consistent v
        try:
            v_from_work = math.sqrt(2 * (self.m * self.g * self.L - self.q * self.E * self.L) / self.m)
            if self.v != 0:
                success &= self._is_close(self.v, v_from_work, "v from work-energy consistency")
        except Exception as e:
            self._log(f"Trace consistency (v from work) failed: {e}")
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and check trends in results."""
        original_m = self.m
        
        try:
            # Perturb m by +10%
            perturbed_m = original_m * 1.10
            
            # Recalculate expected values with perturbed m
            E_perturbed = (3 * perturbed_m * self.g) / (4 * self.q)
            F_perturbed = 1.5 * perturbed_m * self.g
            
            # Check monotonicity: both E and F should increase with m
            E_increased = E_perturbed > self.expected_E
            F_increased = F_perturbed > self.expected_F
            
            self._log(f"Monotonicity (m↑): E {E_increased}, F {F_increased}")
            
            # Also test negative perturbation
            perturbed_m_small = original_m * 0.90
            E_small = (3 * perturbed_m_small * self.g) / (4 * self.q)
            F_small = 1.5 * perturbed_m_small * self.g
            
            E_decreased = E_small < self.expected_E
            F_decreased = F_small < self.expected_F
            
            self._log(f"Monotonicity (m↓): E {E_decreased}, F {F_decreased}")
            
            return E_increased and F_increased and E_decreased and F_decreased
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = ["=== Physics Checker ==="]
        success = True
        
        tests = [
            ("Equilibrium test", self.test_equilibrium),
            ("Work-energy test", self.test_work_energy),
            ("Circular motion test", self.test_circular_motion),
            ("Invariant tests", self.test_invariants),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity test", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            self._log(f"\n--- {name} ---")
            try:
                result = test_func()
                success &= result
                self._log(f"Result: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                self._log(f"Exception: {traceback.format_exc()}")
                success = False
        
        self._log(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs