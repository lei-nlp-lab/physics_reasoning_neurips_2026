import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        SAFETY: Never access solution_instance.attr directly. Use getattr with fallbacks.
        """
        self.solution = solution_instance
        
        # Extract final result safely - only from results dictionary
        self.V = self._safe_extract_numeric('V', 0.0)
        
        # Extract potential inputs (if available in solution)
        # These might be stored in results or as attributes
        self.p0 = self._safe_extract_numeric('p0', 1.013e5)  # atmospheric pressure default
        self.rho = self._safe_extract_numeric('rho', 13.6e3)  # mercury density default
        self.g = self._safe_extract_numeric('g', 9.8)  # gravity default
        self.h1 = self._safe_extract_numeric('h1', 0.20)
        self.h2 = self._safe_extract_numeric('h2', 0.25)
        self.V_B = self._safe_extract_numeric('V_B', None)
        self.V_C = self._safe_extract_numeric('V_C', None)
        
        # Store raw results for trace checking
        self.results = getattr(solution_instance, 'results', {})
        
        # Tolerance for fuzzy matching (1-2% as per robustness rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        # Special tolerance for small volume
        self.vol_abs_tol = 1e-6
    
    def _safe_extract_numeric(self, key, default):
        """Safe extraction of numeric values ONLY from results dictionary."""
        # Only look in results dictionary, not in attributes
        if hasattr(self.solution, 'results'):
            val = self.solution.results.get(key, None)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default
        return default
    
    def test_invariants(self):
        """
        Test Boyle's Law invariants from the KB.
        KB Theorem: Boyle's Law - P₁V₁ = P₂V₂ for isothermal process.
        """
        logs = []
        success = True
        
        try:
            # Test 1: Check if V is positive and reasonable
            if self.V <= 0:
                logs.append(f"FAIL: Volume V={self.V} is non-positive")
                success = False
            else:
                logs.append(f"PASS: Volume V={self.V} is positive")
            
            # Test 2: Boyle's Law consistency check if we have all parameters
            # Note: V_B and V_C might not be in results, so we skip if missing
            if self.V_B is not None and self.V_C is not None:
                # Equation from step 1: p0*(V_B + V_C) = (p0 + rho*g*h1)*V_C
                lhs1 = self.p0 * (self.V_B + self.V_C)
                rhs1 = (self.p0 + self.rho * self.g * self.h1) * self.V_C
                
                if not math.isclose(lhs1, rhs1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    logs.append(f"FAIL: Boyle's Law step 1 violated: {lhs1} ≠ {rhs1}")
                    success = False
                else:
                    logs.append(f"PASS: Boyle's Law step 1 satisfied: {lhs1} ≈ {rhs1}")
                
                # Equation from step 2: p0*(V_B + V_C - V) = (p0 + rho*g*h2)*(V_C - V)
                lhs2 = self.p0 * (self.V_B + self.V_C - self.V)
                rhs2 = (self.p0 + self.rho * self.g * self.h2) * (self.V_C - self.V)
                
                if not math.isclose(lhs2, rhs2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    logs.append(f"FAIL: Boyle's Law step 2 violated: {lhs2} ≠ {rhs2}")
                    success = False
                else:
                    logs.append(f"PASS: Boyle's Law step 2 satisfied: {lhs2} ≈ {rhs2}")
            else:
                logs.append("SKIP: Boyle's Law check - missing V_B or V_C in results")
        
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Check that intermediate results (if available) are consistent with final V.
        From steps analysis, we can derive V from the two equations.
        """
        logs = []
        success = True
        
        try:
            # If we have all parameters, we can solve for V analytically
            if self.V_C is not None:
                # From the derivation in steps: V = V_C * (1 - h1/h2)
                V_derived = self.V_C * (1 - self.h1/self.h2)
                if not math.isclose(self.V, V_derived, rel_tol=self.rel_tol, abs_tol=self.vol_abs_tol):
                    logs.append(f"FAIL: Derived V={V_derived} doesn't match result V={self.V}")
                    logs.append(f"INFO: Using formula V = V_C*(1 - h1/h2)")
                    success = False
                else:
                    logs.append(f"PASS: Derived formula gives consistent V={V_derived}")
            else:
                logs.append("SKIP: Trace consistency check - missing V_C in results")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Test monotonic behavior: if h1 increases, V should decrease (and vice versa).
        Perturb input, recalculate, check trend, then restore.
        """
        logs = []
        success = True
        
        try:
            if self.V_C is not None:
                # Store original values
                original_h1 = self.h1
                original_V = self.V
                
                # Perturb h1 by 1% increase
                perturbed_h1 = original_h1 * 1.01
                
                # Recalculate V using derived formula: V = V_C * (1 - h1/h2)
                perturbed_V = self.V_C * (1 - perturbed_h1/self.h2)
                
                # Check monotonicity: if h1 increases, V should decrease
                if perturbed_h1 > original_h1 and perturbed_V >= original_V:
                    logs.append(f"FAIL: Monotonicity violation: h1↑ ({original_h1}→{perturbed_h1}) but V not↓ ({original_V}→{perturbed_V})")
                    success = False
                elif perturbed_h1 > original_h1 and perturbed_V < original_V:
                    logs.append(f"PASS: Monotonicity: h1↑ → V↓ ({original_V}→{perturbed_V})")
                
                # Test opposite direction: decrease h1 by 1%
                perturbed_h1 = original_h1 * 0.99
                perturbed_V = self.V_C * (1 - perturbed_h1/self.h2)
                
                if perturbed_h1 < original_h1 and perturbed_V <= original_V:
                    logs.append(f"FAIL: Monotonicity violation: h1↓ ({original_h1}→{perturbed_h1}) but V not↑ ({original_V}→{perturbed_V})")
                    success = False
                elif perturbed_h1 < original_h1 and perturbed_V > original_V:
                    logs.append(f"PASS: Monotonicity: h1↓ → V↑ ({original_V}→{perturbed_V})")
                
            else:
                logs.append("SKIP: Monotonicity test - missing V_C in results")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("\n1. Boyle's Law Invariants:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n2. Trace Consistency:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n3. Monotonicity Test:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs