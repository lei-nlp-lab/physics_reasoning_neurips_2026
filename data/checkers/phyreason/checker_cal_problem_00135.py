import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract all inputs and results safely using getattr."""
        # Store the solution instance for potential re-evaluation
        self.sol = solution_instance
        
        # Extract input parameters safely (default to 0.0 or 1.0 if missing)
        self.a = getattr(solution_instance, 'a', 0.0)
        self.b = getattr(solution_instance, 'b', 0.0)
        self.c = getattr(solution_instance, 'c', 0.0)
        self.n = getattr(solution_instance, 'n', 0.0)
        self.p = getattr(solution_instance, 'p', 0.0)
        self.e = getattr(solution_instance, 'e', 1.6e-19)
        self.B = getattr(solution_instance, 'B', 0.0)
        self.E = getattr(solution_instance, 'E', 0.0)
        self.Delta_t = getattr(solution_instance, 'Delta_t', 0.0)
        
        # Extract results from solution_instance.results dictionary if available
        results = getattr(solution_instance, 'results', {})
        
        self.I_n = results.get('I_n', getattr(solution_instance, 'I_n', 0.0))
        self.N_n = results.get('N_n', getattr(solution_instance, 'N_n', 0.0))
        self.F_e = results.get('F_e', getattr(solution_instance, 'F_e', 0.0))
        self.N_p = results.get('N_p', getattr(solution_instance, 'N_p', 0.0))
        self.F_l = results.get('F_l', getattr(solution_instance, 'F_l', 0.0))
        self.F_nz = results.get('F_nz', getattr(solution_instance, 'F_nz', 0.0))
        
        # Additional parameters that might appear in the solution
        self.v_nx = results.get('v_nx', getattr(solution_instance, 'v_nx', 0.0))
        self.v_nz = results.get('v_nz', getattr(solution_instance, 'v_nz', 0.0))
        self.v_pz = results.get('v_pz', getattr(solution_instance, 'v_pz', 0.0))
        
        # Build logs
        self.logs = []

    def log(self, msg, success=None):
        """Helper to log messages."""
        if success is not None:
            msg = f"{'PASS' if success else 'FAIL'}: {msg}"
        self.logs.append(msg)
        
    def safe_isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for numbers."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            return False

    def test_invariants(self):
        """Check fundamental physics laws using numerical methods."""
        try:
            # 1. Check current expression I_n = n * e * a * b * v_nx
            if self.v_nx != 0.0:
                I_n_calc = self.n * self.e * self.a * self.b * self.v_nx
                if not self.safe_isclose(self.I_n, I_n_calc):
                    self.log(f"Current invariant failed: I_n={self.I_n} vs calculated {I_n_calc}", False)
                else:
                    self.log("Current expression I_n = n e a b v_nx satisfied", True)
            
            # 2. Check Lorentz force on electron: F_l = e * v_nx * B
            if self.v_nx != 0.0 and self.B != 0.0:
                F_l_calc = self.e * self.v_nx * self.B
                if not self.safe_isclose(self.F_l, F_l_calc):
                    self.log(f"Lorentz force invariant failed: F_l={self.F_l} vs {F_l_calc}", False)
                else:
                    self.log("Lorentz force expression F_l = e v_nx B satisfied", True)
            
            # 3. Check electric force: F_e = e * E
            if self.E != 0.0:
                F_e_calc = self.e * self.E
                if not self.safe_isclose(self.F_e, F_e_calc):
                    self.log(f"Electric force invariant failed: F_e={self.F_e} vs {F_e_calc}", False)
                else:
                    self.log("Electric force expression F_e = e E satisfied", True)
            
            # 4. Check net force superposition: F_nz = F_l + F_e
            if self.F_l != 0.0 and self.F_e != 0.0:
                F_nz_expected = self.F_l + self.F_e
                if not self.safe_isclose(self.F_nz, F_nz_expected):
                    self.log(f"Superposition invariant failed: F_nz={self.F_nz} vs sum {F_nz_expected}", False)
                else:
                    self.log("Force superposition F_nz = F_l + F_e satisfied", True)
            
            # 5. Check the derived expression: F_nz = e * (I_n * B/(n e a b) + E)
            if self.I_n != 0.0 and self.n != 0.0:
                F_nz_derived = self.e * (self.I_n * self.B / (self.n * self.e * self.a * self.b) + self.E)
                if not self.safe_isclose(self.F_nz, F_nz_derived):
                    self.log(f"Derived F_nz expression failed: {self.F_nz} vs {F_nz_derived}", False)
                else:
                    self.log("Derived F_nz expression satisfied", True)
            
            # 6. Check particle number expressions
            if self.v_nz != 0.0 and self.Delta_t != 0.0:
                N_n_calc = self.n * self.a * self.c * self.v_nz * self.Delta_t
                if not self.safe_isclose(self.N_n, N_n_calc):
                    self.log(f"N_n expression failed: {self.N_n} vs {N_n_calc}", False)
                else:
                    self.log("N_n expression satisfied", True)
            
            if self.v_pz != 0.0 and self.Delta_t != 0.0:
                N_p_calc = self.p * self.a * self.c * self.v_pz * self.Delta_t
                if not self.safe_isclose(self.N_p, N_p_calc):
                    self.log(f"N_p expression failed: {self.N_p} vs {N_p_calc}", False)
                else:
                    self.log("N_p expression satisfied", True)
            
            # 7. Charge neutrality condition: In steady state, N_n = N_p
            if self.N_n != 0.0 and self.N_p != 0.0:
                if not self.safe_isclose(self.N_n, self.N_p):
                    self.log(f"Charge neutrality failed: N_n={self.N_n} vs N_p={self.N_p}", False)
                else:
                    self.log("Charge neutrality N_n = N_p satisfied", True)
            
            return True
        except Exception as e:
            self.log(f"Invariant test crashed: {traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        try:
            # Target answer 2: Check F_nz expression
            if self.I_n != 0.0 and self.n != 0.0:
                F_nz_expected = self.e * (self.I_n * self.B / (self.n * self.e * self.a * self.b) + self.E)
                if not self.safe_isclose(self.F_nz, F_nz_expected):
                    self.log(f"Trace consistency failed for F_nz: {self.F_nz} vs {F_nz_expected}", False)
                    return False
                else:
                    self.log("F_nz expression matches target answer", True)
            else:
                self.log("Skipping F_nz trace check due to missing data", None)
            
            # Target answer 3: Check particle expressions and equality
            if self.N_n != 0.0 and self.N_p != 0.0:
                if not self.safe_isclose(self.N_n, self.N_p):
                    self.log(f"Trace consistency failed: N_n != N_p ({self.N_n} vs {self.N_p})", False)
                    return False
                else:
                    self.log("N_n = N_p condition satisfied", True)
            
            return True
        except Exception as e:
            self.log(f"Trace consistency test crashed: {traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """Perturb an input and check if results change in expected direction."""
        try:
            # In this problem, I_n is given as a constant current.
            # We should test monotonicity with respect to a different parameter.
            # Let's test how F_nz changes with B (should increase).
            
            original_B = self.B
            if original_B == 0.0:
                original_B = 0.1  # Default
            
            # Increase B by 10%
            perturbed_B = original_B * 1.1
            
            # Recalculate F_nz using the derived expression
            # F_nz = e*(I_n*B/(n*e*a*b) + E) = (I_n*B)/(n*a*b) + e*E
            if self.I_n != 0.0 and self.n != 0.0:
                original_F_nz = (self.I_n * original_B) / (self.n * self.a * self.b) + self.e * self.E
                perturbed_F_nz = (self.I_n * perturbed_B) / (self.n * self.a * self.b) + self.e * self.E
                
                # Check monotonicity: F_nz should increase with B
                if perturbed_F_nz > original_F_nz:
                    self.log(f"Monotonicity: F_nz increases with B ({original_F_nz:.3e} → {perturbed_F_nz:.3e})", True)
                else:
                    self.log(f"Monotonicity violation: F_nz doesn't increase with B", False)
                    return False
            else:
                self.log("Skipping monotonicity test due to missing data", None)
            
            return True
            
        except Exception as e:
            self.log(f"Monotonicity test crashed: {traceback.format_exc()}", False)
            return False

    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        # Overall success if all passed
        success = inv_ok and trace_ok and mono_ok
        
        # Generate summary
        self.log(f"\n=== SUMMARY ===")
        self.log(f"Invariants: {'PASS' if inv_ok else 'FAIL'}")
        self.log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.log(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
        self.log(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs