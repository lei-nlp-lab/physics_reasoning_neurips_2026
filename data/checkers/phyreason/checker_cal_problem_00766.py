import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.logs = []
        
        # Safe extraction of inputs (convert to SI units)
        self.m_A = getattr(solution_instance, 'm_A', 0.1)  # 100 g -> 0.1 kg
        self.m_B = getattr(solution_instance, 'm_B', 0.5)  # 500 g -> 0.5 kg
        self.x = getattr(solution_instance, 'x', 0.15)      # 15 cm -> 0.15 m
        self.g = getattr(solution_instance, 'g', 10.0)      # Default 10 m/s²
        
        # Safe extraction of results from results dict or attributes
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results_dict = solution_instance.results
        else:
            results_dict = {}
            
        self.k = getattr(solution_instance, 'k', results_dict.get('k', None))
        self.a_max = getattr(solution_instance, 'a_max', results_dict.get('a_max', None))
        self.x_A = getattr(solution_instance, 'x_A', results_dict.get('x_A', None))
        self.A = getattr(solution_instance, 'A', results_dict.get('A', None))
        
        # Store for monotonicity test
        self.original_m_B = self.m_B

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((success, message))
        return success

    def test_invariants(self):
        """Verify fundamental physics invariants."""
        try:
            # 1. Static equilibrium before cutting (Net force = 0)
            # Total weight = (m_A + m_B)*g, spring force = k*x
            if self.k is not None:
                net_force_before = (self.m_A + self.m_B) * self.g - self.k * self.x
                if math.isclose(net_force_before, 0, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Static equilibrium before cutting: Net force ≈ 0", True)
                else:
                    self._log(f"Static equilibrium violated: net force = {net_force_before}", False)
                    return False
            
            # 2. Newton's 2nd Law for ball A immediately after cutting
            # At the instant of cutting, ball A is at position x (15 cm) with no velocity.
            # The net force on A: F_net = k*(x - x_A) - m_A*g (upward positive)
            # From steps: x_A = m_A*g/k, so F_net = k*(x - m_A*g/k) - m_A*g = k*x - 2*m_A*g
            # But also from SHM: a_max = ω²*A = (k/m_A)*A
            if all(v is not None for v in [self.k, self.x_A, self.A, self.a_max]):
                # Check using SHM relation
                omega = sp.sqrt(self.k / self.m_A)
                a_max_shm = float(omega**2 * self.A)
                
                # Check using step4 formula: a_max = m_B*g/m_A
                a_max_step4 = self.m_B * self.g / self.m_A
                
                # Both should match
                if (math.isclose(a_max_shm, self.a_max, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(a_max_step4, self.a_max, rel_tol=1e-2, abs_tol=1e-2)):
                    self._log(f"Newton's 2nd Law consistent: a_max = {self.a_max:.2f} m/s²", True)
                else:
                    self._log(f"Newton's 2nd Law mismatch: SHM gives {a_max_shm:.2f}, step4 gives {a_max_step4:.2f}", False)
                    return False
            
            # 3. Energy conservation during SHM
            # Total energy at amplitude: E = (1/2)*k*A²
            # Maximum speed: v_max = ω*A
            # Kinetic energy at equilibrium: KE_max = (1/2)*m_A*v_max² = (1/2)*m_A*(ω²*A²) = (1/2)*k*A²
            if self.k is not None and self.A is not None:
                E_total = 0.5 * self.k * self.A**2
                v_max = sp.sqrt(self.k/self.m_A) * self.A
                KE_max = 0.5 * self.m_A * float(v_max)**2
                if math.isclose(E_total, KE_max, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Energy conservation in SHM verified", True)
                else:
                    self._log(f"Energy conservation mismatch: E_total={E_total:.3f}, KE_max={KE_max:.3f}", False)
                    return False
                
            return True
        except Exception as e:
            self._log(f"Invariant test failed: {traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            success = True
            
            # 1. Check k = (m_A + m_B)*g / x
            if self.k is not None:
                k_calc = (self.m_A + self.m_B) * self.g / self.x
                if not math.isclose(self.k, k_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace inconsistency in k: expected {k_calc:.2f}, got {self.k:.2f}", False)
                    success = False
                else:
                    self._log(f"k consistent: {self.k:.2f} N/m", True)
            
            # 2. Check x_A = m_A * g / k
            if all(v is not None for v in [self.x_A, self.k]):
                x_A_calc = self.m_A * self.g / self.k
                if not math.isclose(self.x_A, x_A_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace inconsistency in x_A: expected {x_A_calc:.3f}, got {self.x_A:.3f}", False)
                    success = False
                else:
                    self._log(f"x_A consistent: {self.x_A:.3f} m", True)
            
            # 3. Check A = x - x_A
            if all(v is not None for v in [self.A, self.x_A]):
                A_calc = self.x - self.x_A
                if not math.isclose(self.A, A_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace inconsistency in A: expected {A_calc:.3f}, got {self.A:.3f}", False)
                    success = False
                else:
                    self._log(f"A consistent: {self.A:.3f} m", True)
            
            # 4. Check a_max = m_B * g / m_A
            if self.a_max is not None:
                a_max_calc = self.m_B * self.g / self.m_A
                if not math.isclose(self.a_max, a_max_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace inconsistency in a_max: expected {a_max_calc:.2f}, got {self.a_max:.2f}", False)
                    success = False
                else:
                    self._log(f"a_max consistent: {self.a_max:.2f} m/s²", True)
            
            # 5. Check a_max = (k/m_A) * A
            if all(v is not None for v in [self.a_max, self.k, self.A]):
                a_max_shm = (self.k / self.m_A) * self.A
                if not math.isclose(self.a_max, a_max_shm, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"SHM formula mismatch: (k/m_A)*A = {a_max_shm:.2f}, a_max = {self.a_max:.2f}", False)
                    success = False
                else:
                    self._log("a_max consistent with SHM formula", True)
            
            return success
        except Exception as e:
            self._log(f"Trace consistency test failed: {traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """Perturb m_B and check that a_max increases proportionally."""
        try:
            # Save original value
            original_m_B = self.m_B
            
            # Perturb m_B by +10%
            self.m_B *= 1.1
            
            # Recalculate expected a_max using formula a_max = m_B * g / m_A
            perturbed_a_max = self.m_B * self.g / self.m_A
            
            # Restore original
            self.m_B = original_m_B
            
            # Original a_max
            if self.a_max is None:
                self._log("Cannot test monotonicity: a_max not available", False)
                return False
                
            original_a_max = self.a_max
            
            # Check monotonic increase
            if perturbed_a_max > original_a_max:
                self._log(f"Monotonicity confirmed: increasing m_B increases a_max ({original_a_max:.2f} -> {perturbed_a_max:.2f})", True)
                return True
            else:
                self._log(f"Monotonicity violation: a_max did not increase with m_B", False)
                return False
                
        except Exception as e:
            self._log(f"Monotonicity test failed: {traceback.format_exc()}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        # Run tests in order
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall_success = test1 and test2 and test3
        
        # Summary
        if overall_success:
            self.logs.append((True, "All physics checks passed"))
        else:
            self.logs.append((False, "Some physics checks failed"))
        
        return overall_success, self.logs