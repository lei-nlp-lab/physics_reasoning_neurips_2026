import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Given constants (with safe defaults)
        self.Q = getattr(solution_instance, 'Q', 2.4e-6)
        self.l = getattr(solution_instance, 'l', 0.02)
        self.varphi_P = getattr(solution_instance, 'varphi_P', 3000.0)
        self.W = getattr(solution_instance, 'W', -1.8e-3)
        self.a = getattr(solution_instance, 'a', 9.0e9)
        self.k = getattr(solution_instance, 'k', 9.0e9)
        
        # Extracted results (with safe defaults)
        self.fracqm = self.results.get('fracqm', 0.0)
        self.U = self.results.get('U', 3000.0)
        self.v = self.results.get('v', 1000.0)
        
        # Derived constants
        self.r = self.l  # distance from charge A to point P
        
        # Initialize test logs
        self.logs = []
        
    def log(self, message, success=None):
        """Helper to log test results."""
        entry = f"[{message}]"
        if success is not None:
            entry = f"{'PASS' if success else 'FAIL'}: {entry}"
        self.logs.append(entry)
        return success
        
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Coulomb's law and Newton's second law: a = k*Q*q/(m*r²) -> q/m = a*r²/(k*Q)
            qm_expected = (self.a * self.r**2) / (self.k * self.Q)
            if not math.isclose(self.fracqm, qm_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"q/m invariant violated: expected {qm_expected:.3e}, got {self.fracqm:.3e}", False)
            self.log(f"q/m from Coulomb+Newton: {self.fracqm:.3e} ≈ {qm_expected:.3e}", True)
            
            # 2. Work-energy theorem: W = q*(φ_inf - φ_P) = -q*φ_P (since φ_inf=0)
            #    Note: W is negative work done BY electric force (so force opposes motion)
            q_from_work = -self.W / self.varphi_P  # Should be positive since W negative and φ_P positive
            if q_from_work < 0:
                return self.log(f"Charge sign inconsistency: q from work={q_from_work:.3e} C", False)
            self.log(f"Charge q from work: {q_from_work:.3e} C", True)
            
            # 3. Acceleration from Coulomb's law: a = k*Q*q/(m*r²) = k*Q*(q/m)/r²
            a_expected = self.k * self.Q * self.fracqm / (self.r**2)
            if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Acceleration invariant violated: expected {a_expected:.3e}, got {self.a:.3e}", False)
            self.log(f"Acceleration consistent: {self.a:.3e} m/s² ≈ {a_expected:.3e} m/s²", True)
            
            # 4. Energy conservation during motion from P to infinity
            #    At P: potential energy U_pot = q*φ_P, kinetic energy = 0
            #    At infinity: potential energy = 0, kinetic energy = ½mv²_max
            #    So: q*φ_P = ½mv²_max -> v_max = sqrt(2*(q/m)*φ_P)
            v_expected = math.sqrt(2 * self.fracqm * self.varphi_P)
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"Energy conservation violated: expected v={v_expected:.3e}, got {self.v:.3e}", False)
            self.log(f"Max speed from energy conservation: {self.v:.3e} m/s ≈ {v_expected:.3e} m/s", True)
            
            return True
        except Exception as e:
            self.log(f"Invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Step 3 formula: q/m = a*r²/(k*Q)
            qm_calculated = (self.a * self.r**2) / (self.k * self.Q)
            if not math.isclose(self.fracqm, qm_calculated, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"q/m mismatch: trace says {self.fracqm:.3e}, calculated {qm_calculated:.3e}", False)
            self.log(f"q/m consistent with step 3 formula", True)
            
            # Step 8 formula: v = sqrt(2*a*r²*U/(k*Q)) where U = φ_P
            v_calculated = math.sqrt(2 * self.a * self.r**2 * self.varphi_P / (self.k * self.Q))
            if not math.isclose(self.v, v_calculated, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"v mismatch: trace says {self.v:.3e}, calculated {v_calculated:.3e}", False)
            self.log(f"v consistent with step 8 formula", True)
            
            # Check potential difference U = φ_P - 0
            if not math.isclose(self.U, self.varphi_P, rel_tol=1e-2, abs_tol=1e-2):
                return self.log(f"U mismatch: {self.U:.3e} vs φ_P={self.varphi_P:.3e}", False)
            self.log(f"Potential difference U = φ_P = {self.U:.3e} V", True)
            
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check trend sensitivity."""
        try:
            original_Q = self.Q
            perturbed_Q = original_Q * 1.1  # Increase by 10%
            
            # v = sqrt(2*a*r²*φ_P/(k*Q))
            # Since v ∝ 1/√Q, increasing Q should decrease v
            original_v = self.v
            perturbed_v = math.sqrt(2 * self.a * self.r**2 * self.varphi_P / (self.k * perturbed_Q))
            
            if perturbed_v >= original_v:
                return self.log(f"Monotonicity violated: increasing Q should decrease v", False)
            self.log(f"Monotonicity holds: Q↑ {original_Q:.3e}→{perturbed_Q:.3e} => v↓ {original_v:.3e}→{perturbed_v:.3e}", True)
            
            # Restore original value
            self.Q = original_Q
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        self.logs = []
        results = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                success = test_func()
                results.append((name, success))
                if success is None:
                    self.logs.append(f"INFO: {name} completed without explicit pass/fail")
            except Exception as e:
                self.logs.append(f"ERROR in {name}: {str(e)}")
                results.append((name, False))
        
        # Overall success if all tests pass
        overall_success = all(success for _, success in results)
        
        # Add final summary
        summary = f"PhysicsChecker: {sum(1 for _, s in results if s)}/{len(results)} tests passed"
        self.logs.append(summary)
        
        return overall_success, self.logs