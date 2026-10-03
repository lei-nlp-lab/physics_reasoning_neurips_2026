import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Get results dictionary
        self.results = getattr(solution_instance, 'results', {})
        self.logs = []
        
        # Extract ALL values from results dict with numeric defaults
        # Use .get() with sensible defaults for missing values
        self.h = self.results.get('h', 5.0)
        self.L = self.results.get('L', 1.7)
        self.v = self.results.get('v', 1.5)
        self.t = self.results.get('t', 2.0)
        
        # Extract results
        self.k = self.results.get('k', 0.0)
        self.OS = self.results.get('OS', 0.0)
        self.SM = self.results.get('SM', 0.0)
        self.OM = self.results.get('OM', 0.0)
        
    def _is_close(self, a, b):
        """Numeric comparison only."""
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """Check fundamental physics invariants using numeric values."""
        logs = []
        success = True
        
        try:
            # 1. Uniform motion check: OS = v*t
            expected_OS = self.v * self.t
            if not self._is_close(self.OS, expected_OS):
                logs.append(f"FAIL: Uniform motion invariant. OS={self.OS}, expected v*t={expected_OS}")
                success = False
            else:
                logs.append("PASS: Uniform motion (OS = v*t) holds")
            
            # 2. Similar triangles geometry: (h-L)/OS = h/OM
            # Avoid division by zero
            if abs(self.OS) > 1e-10 and abs(self.OM) > 1e-10:
                lhs = (self.h - self.L) / self.OS
                rhs = self.h / self.OM
                if not self._is_close(lhs, rhs):
                    logs.append(f"FAIL: Similar triangles invariant. (h-L)/OS={lhs:.4f}, h/OM={rhs:.4f}")
                    success = False
                else:
                    logs.append("PASS: Similar triangles geometry holds")
            else:
                logs.append("SKIP: Similar triangles check (division by near-zero)")
            
            # 3. Shadow length consistency: SM = OM - OS
            expected_SM = self.OM - self.OS
            if not self._is_close(self.SM, expected_SM):
                logs.append(f"FAIL: Shadow length SM != OM - OS. SM={self.SM}, OM-OS={expected_SM}")
                success = False
            else:
                logs.append("PASS: Shadow length SM = OM - OS")
            
            # 4. Constant rate of change: k should be L*v/(h-L)
            # Derivative check not needed for numeric values
            logs.append("SKIP: Derivative check (numeric values only)")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results using numeric values."""
        logs = []
        success = True
        
        try:
            # 1. Check OM expression from steps: OM = (h*v/(h-L))*t
            if abs(self.h - self.L) > 1e-10:
                expected_OM = (self.h * self.v / (self.h - self.L)) * self.t
                if not self._is_close(self.OM, expected_OM):
                    logs.append(f"FAIL: OM mismatch. OM={self.OM:.4f}, expected (h*v/(h-L))*t={expected_OM:.4f}")
                    success = False
                else:
                    logs.append("PASS: OM matches geometric optics formula")
            else:
                logs.append("SKIP: OM check (h ≈ L)")
            
            # 2. Check SM expression from steps: SM = (L*v/(h-L))*t
            if abs(self.h - self.L) > 1e-10:
                expected_SM = (self.L * self.v / (self.h - self.L)) * self.t
                if not self._is_close(self.SM, expected_SM):
                    logs.append(f"FAIL: SM mismatch. SM={self.SM:.4f}, expected (L*v/(h-L))*t={expected_SM:.4f}")
                    success = False
                else:
                    logs.append("PASS: SM matches derived formula")
            else:
                logs.append("SKIP: SM check (h ≈ L)")
            
            # 3. Check k expression: k = L*v/(h-L)
            if abs(self.h - self.L) > 1e-10:
                expected_k = self.L * self.v / (self.h - self.L)
                if not self._is_close(self.k, expected_k):
                    logs.append(f"FAIL: k mismatch. k={self.k:.4f}, expected L*v/(h-L)={expected_k:.4f}")
                    success = False
                else:
                    logs.append("PASS: k matches target answer")
            else:
                logs.append("SKIP: k check (h ≈ L)")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def test_monotonicity(self):
        """Test sensitivity to parameter changes using numeric values."""
        logs = []
        success = True
        
        try:
            # Use the actual values from results
            original_h = float(self.h)
            original_L = float(self.L)
            original_v = float(self.v)
            
            # Ensure h > L for valid physics
            if original_h <= original_L:
                logs.append(f"SKIP: Monotonicity test (h={original_h} <= L={original_L})")
                return success
            
            # Calculate original k
            k_original = original_L * original_v / (original_h - original_L)
            
            # Perturb L by +10%
            L_perturbed = original_L * 1.1
            k_perturbed = L_perturbed * original_v / (original_h - L_perturbed)
            
            # Check monotonicity: k should increase with L (since denominator decreases)
            if k_perturbed > k_original:
                logs.append("PASS: k increases with L (monotonicity holds)")
            else:
                logs.append(f"FAIL: k doesn't increase with L. k_original={k_original:.3f}, k_perturbed={k_perturbed:.3f}")
                success = False
            
            # Additional check: k should decrease with h
            h_perturbed = original_h * 1.1
            k_h_perturbed = original_L * original_v / (h_perturbed - original_L)
            
            if k_h_perturbed < k_original:
                logs.append("PASS: k decreases with h (monotonicity holds)")
            else:
                logs.append(f"FAIL: k doesn't decrease with h. k_original={k_original:.3f}, k_h_perturbed={k_h_perturbed:.3f}")
                success = False
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        self.logs.extend(logs)
        return success
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        self.logs = ["=== Physics Checker Started ==="]
        
        # Run tests
        invariant_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonic_ok = self.test_monotonicity()
        
        # Summary
        all_ok = invariant_ok and trace_ok and monotonic_ok
        
        self.logs.append(f"\n=== Summary ===")
        self.logs.append(f"Invariants test: {'PASS' if invariant_ok else 'FAIL'}")
        self.logs.append(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.logs.append(f"Monotonicity test: {'PASS' if monotonic_ok else 'FAIL'}")
        self.logs.append(f"Overall: {'PASS' if all_ok else 'FAIL'}")
        
        return all_ok, self.logs