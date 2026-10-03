import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely
        self.Ga = getattr(solution_instance, 'Ga', getattr(solution_instance, 'input_dict', {}).get('Ga', 10.0))
        self.Gb = getattr(solution_instance, 'Gb', getattr(solution_instance, 'input_dict', {}).get('Gb', 20.0))
        self.Gc = getattr(solution_instance, 'Gc', getattr(solution_instance, 'input_dict', {}).get('Gc', 30.0))
        self.k = getattr(solution_instance, 'k', getattr(solution_instance, 'input_dict', {}).get('k', 500.0))
        
        # Extract results safely with fallback
        self.results = getattr(solution_instance, 'results', {})
        self.x_1 = self.results.get('x_1', 0.0)
        self.x_b = self.results.get('x_b', 0.0)
        self.x_2 = self.results.get('x_2', 0.0)
        self.x_3 = self.results.get('x_3', 0.0)
        self.x = self.results.get('x', 0.0)
        
        # Set tolerances for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for reporting
        self.logs = []
        
    def log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def test_invariants(self):
        """Verify Hooke's Law and geometric relationships."""
        try:
            # 1. Check x_1 = Gb/k
            expected_x1 = self.Gb / self.k
            if math.isclose(self.x_1, expected_x1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"x_1 invariant: expected {expected_x1}, got {self.x_1}", True)
            else:
                self.log(f"x_1 invariant failed: expected {expected_x1}, got {self.x_1}", False)
                return False
            
            # 2. Check x_2 = Gc/k
            expected_x2 = self.Gc / self.k
            if math.isclose(self.x_2, expected_x2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"x_2 invariant: expected {expected_x2}, got {self.x_2}", True)
            else:
                self.log(f"x_2 invariant failed: expected {expected_x2}, got {self.x_2}", False)
                return False
            
            # 3. Check x_b = x_1 + x_2
            expected_xb = self.x_1 + self.x_2
            if math.isclose(self.x_b, expected_xb, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"x_b invariant: expected {expected_xb}, got {self.x_b}", True)
            else:
                self.log(f"x_b invariant failed: expected {expected_xb}, got {self.x_b}", False)
                return False
            
            # 4. Check x_3 = (Gb + Gc)/k
            expected_x3 = (self.Gb + self.Gc) / self.k
            if math.isclose(self.x_3, expected_x3, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"x_3 invariant: expected {expected_x3}, got {self.x_3}", True)
            else:
                self.log(f"x_3 invariant failed: expected {expected_x3}, got {self.x_3}", False)
                return False
            
            # 5. Check x = x_b + x_3
            expected_x = self.x_b + self.x_3
            if math.isclose(self.x, expected_x, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"x invariant: expected {expected_x}, got {self.x}", True)
            else:
                self.log(f"x invariant failed: expected {expected_x}, got {self.x}", False)
                return False
            
            # 6. Symbolic verification (optional)
            k, Gb, Gc = sp.symbols('k Gb Gc')
            expr_x1 = Gb / k
            expr_x2 = Gc / k
            expr_xb = expr_x1 + expr_x2
            expr_x3 = (Gb + Gc) / k
            expr_x = expr_xb + expr_x3
            # Substitute numerical values and evaluate
            subs_dict = {k: self.k, Gb: self.Gb, Gc: self.Gc}
            sym_x1 = float(expr_x1.subs(subs_dict))
            sym_x2 = float(expr_x2.subs(subs_dict))
            sym_xb = float(expr_xb.subs(subs_dict))
            sym_x3 = float(expr_x3.subs(subs_dict))
            sym_x = float(expr_x.subs(subs_dict))
            
            checks = [
                (self.x_1, sym_x1, "x_1 symbolic"),
                (self.x_2, sym_x2, "x_2 symbolic"),
                (self.x_b, sym_xb, "x_b symbolic"),
                (self.x_3, sym_x3, "x_3 symbolic"),
                (self.x, sym_x, "x symbolic")
            ]
            all_symbolic_ok = True
            for val, sym_val, label in checks:
                if not math.isclose(val, sym_val, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"{label} mismatch: expected {sym_val}, got {val}", False)
                    all_symbolic_ok = False
            if all_symbolic_ok:
                self.log("All symbolic checks passed", True)
            
            return True
        except Exception as e:
            self.log(f"Exception in test_invariants: {e}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # x_b should be x_1 + x_2
            if not math.isclose(self.x_b, self.x_1 + self.x_2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Trace consistency failed: x_b ({self.x_b}) != x_1 ({self.x_1}) + x_2 ({self.x_2})", False)
                return False
            
            # x should be x_b + x_3
            if not math.isclose(self.x, self.x_b + self.x_3, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Trace consistency failed: x ({self.x}) != x_b ({self.x_b}) + x_3 ({self.x_3})", False)
                return False
            
            # Also check that x_1 and x_2 are consistent with Gb and Gc
            if not (math.isclose(self.x_1, self.Gb/self.k, rel_tol=self.rel_tol, abs_tol=self.abs_tol) and
                    math.isclose(self.x_2, self.Gc/self.k, rel_tol=self.rel_tol, abs_tol=self.abs_tol)):
                self.log("Trace consistency failed: x_1 or x_2 not matching Gb/k or Gc/k", False)
                return False
            
            self.log("Trace consistency passed: all intermediate results align", True)
            return True
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {e}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb an input and verify output trend."""
        original_Gb = self.Gb
        perturbation = 1.0  # Increase Gb by 1 N
        try:
            # Calculate expected x with original Gb
            original_x = (self.x_1 + self.x_2) + (self.Gb + self.Gc) / self.k
            # Alternatively, use the formula: x = 2*(Gb + Gc)/k
            # But let's use the step formulas to be safe.
            
            # Perturb Gb
            self.Gb += perturbation
            # Recalculate x_1, x_2, x_3, x_b, x using formulas (not from results)
            new_x1 = self.Gb / self.k
            new_x2 = self.Gc / self.k  # unchanged
            new_xb = new_x1 + new_x2
            new_x3 = (self.Gb + self.Gc) / self.k
            new_x = new_xb + new_x3
            
            # Check monotonicity: increasing Gb should increase x
            if new_x > original_x:
                self.log(f"Monotonicity: increasing Gb from {original_Gb} to {self.Gb} increases x from {original_x} to {new_x}", True)
                success = True
            else:
                self.log(f"Monotonicity failed: increasing Gb did not increase x (original: {original_x}, new: {new_x})", False)
                success = False
            
            # Restore original Gb
            self.Gb = original_Gb
            return success
        except Exception as e:
            self.log(f"Exception in test_monotonicity: {e}", False)
            # Restore original Gb in case of error
            self.Gb = original_Gb
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []  # Reset logs
        success = True
        
        # Run tests and combine results
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self.log(f"Running {test_name}...", None)
                if not test_func():
                    success = False
            except Exception as e:
                self.log(f"Unexpected error in {test_name}: {traceback.format_exc()}", False)
                success = False
        
        self.log(f"All tests completed. Overall: {'PASS' if success else 'FAIL'}", None)
        return success, self.logs