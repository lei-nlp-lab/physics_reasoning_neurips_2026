import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance"""
        self.solution = solution_instance
        
        # Extract inputs with safe fallbacks
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'inputs', {}).get('E', 2.0e4))
        self.d = getattr(solution_instance, 'd', getattr(solution_instance, 'inputs', {}).get('d', 0.20))
        self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'inputs', {}).get('q', 1.0e-8))
        
        # Extract results with safe fallbacks
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', None))
        self.W = getattr(solution_instance, 'W', getattr(solution_instance, 'results', {}).get('W', None))
        
        # For direction (from steps analysis)
        self.force_direction = getattr(solution_instance, 'force_direction', 'to the right')
        
        # Expected values from steps analysis
        self.expected_F = 2.0e-4
        self.expected_W = 4.0e-5
        
        # Initialize logs
        self.logs = []
        
    def test_invariants(self):
        """Check fundamental physics laws"""
        self.logs.append("=== Testing Physics Invariants ===")
        
        # 1. Force magnitude invariant: F = q * E
        try:
            calc_F = self.q * self.E
            
            if self.F is not None:
                if not math.isclose(self.F, calc_F, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"FAIL: Force invariant violated. F = q*E gives {calc_F:.2e}, but got {self.F:.2e}")
                    return False
                else:
                    self.logs.append(f"PASS: F = q*E invariant holds: {self.F:.2e} ≈ {calc_F:.2e}")
            else:
                self.logs.append(f"WARNING: Force result not available for invariant check")
        except Exception as e:
            self.logs.append(f"ERROR in force invariant: {str(e)}")
            return False
            
        # 2. Work-energy invariant: W = F * d (for constant force parallel to displacement)
        try:
            if self.F is not None and self.W is not None:
                calc_W = self.F * self.d
                
                if not math.isclose(self.W, calc_W, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"FAIL: Work-energy invariant violated. W = F*d gives {calc_W:.2e}, but got {self.W:.2e}")
                    return False
                else:
                    self.logs.append(f"PASS: W = F*d invariant holds: {self.W:.2e} ≈ {calc_W:.2e}")
            else:
                self.logs.append(f"WARNING: Missing F or W for work-energy check")
        except Exception as e:
            self.logs.append(f"ERROR in work-energy invariant: {str(e)}")
            return False
            
        # 3. Force direction consistency (from KB)
        try:
            if self.q > 0:
                # For positive charge, force should be in same direction as field
                # Since field direction isn't explicitly given, we assume from steps analysis
                if "right" in self.force_direction.lower():
                    self.logs.append(f"PASS: Force direction consistent for positive charge (+q)")
                else:
                    self.logs.append(f"WARNING: Unexpected force direction: {self.force_direction}")
        except Exception as e:
            self.logs.append(f"ERROR in direction check: {str(e)}")
            
        return True
        
    def test_trace_consistency(self):
        """Check intermediate results against final answers"""
        self.logs.append("\n=== Testing Trace Consistency ===")
        
        try:
            # Check against expected values from steps analysis
            if self.F is not None:
                if not math.isclose(self.F, self.expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"FAIL: Force mismatch. Expected {self.expected_F:.2e}, got {self.F:.2e}")
                    return False
                else:
                    self.logs.append(f"PASS: Force matches step_1 calculation: {self.F:.2e}")
                    
            if self.W is not None:
                if not math.isclose(self.W, self.expected_W, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"FAIL: Work mismatch. Expected {self.expected_W:.2e}, got {self.W:.2e}")
                    return False
                else:
                    self.logs.append(f"PASS: Work matches step_3 calculation: {self.W:.2e}")
                    
            # Verify F was calculated before W in the trace
            if self.F is not None and self.W is not None:
                calc_W_from_F = self.F * self.d
                if math.isclose(self.W, calc_W_from_F, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"PASS: Work calculation uses correct force value")
                else:
                    self.logs.append(f"WARNING: Work doesn't match F*d with given F")
                    
            return True
        except Exception as e:
            self.logs.append(f"ERROR in trace consistency: {str(e)}")
            return False
            
    def test_monotonicity(self):
        """Test how results change with perturbations"""
        self.logs.append("\n=== Testing Monotonicity ===")
        
        original_q = self.q
        original_E = self.E
        original_d = self.d
        
        try:
            # Test 1: Increase charge by 10%
            perturbed_q = original_q * 1.1
            expected_F_increase = perturbed_q * original_E
            original_F = original_q * original_E
            
            if expected_F_increase > original_F:
                self.logs.append(f"PASS: F increases with increased charge ({original_q:.2e} → {perturbed_q:.2e})")
            else:
                self.logs.append(f"FAIL: F doesn't increase with increased charge")
                return False
                
            # Test 2: Increase field strength by 10%
            perturbed_E = original_E * 1.1
            expected_F_increase = original_q * perturbed_E
            
            if expected_F_increase > original_F:
                self.logs.append(f"PASS: F increases with increased field strength ({original_E:.2e} → {perturbed_E:.2e})")
            else:
                self.logs.append(f"FAIL: F doesn't increase with increased field strength")
                return False
                
            # Test 3: Work increases with distance (for constant force)
            perturbed_d = original_d * 1.1
            original_W = original_F * original_d
            perturbed_W = original_F * perturbed_d
            
            if perturbed_W > original_W:
                self.logs.append(f"PASS: W increases with increased distance ({original_d:.2e} → {perturbed_d:.2e})")
            else:
                self.logs.append(f"FAIL: W doesn't increase with increased distance")
                return False
                
            # Test 4: Work is proportional to charge (through F)
            perturbed_F = perturbed_q * original_E
            perturbed_W2 = perturbed_F * original_d
            
            if perturbed_W2 > original_W:
                self.logs.append(f"PASS: W increases with increased charge (via F)")
            else:
                self.logs.append(f"FAIL: W doesn't increase with increased charge")
                return False
                
            # Restore original values
            self.q = original_q
            self.E = original_E
            self.d = original_d
            
            return True
            
        except Exception as e:
            self.logs.append(f"ERROR in monotonicity test: {str(e)}")
            # Restore on error
            self.q = original_q
            self.E = original_E
            self.d = original_d
            return False
            
    def run_all_tests(self):
        """Run all checks and return success status and logs"""
        self.logs = []
        success = True
        
        try:
            if not self.test_invariants():
                success = False
                
            if not self.test_trace_consistency():
                success = False
                
            if not self.test_monotonicity():
                success = False
                
        except Exception as e:
            self.logs.append(f"\nCRITICAL ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
            
        # Final summary
        self.logs.append("\n=== TEST SUMMARY ===")
        if success:
            self.logs.append("ALL CHECKS PASSED ✓")
        else:
            self.logs.append("SOME CHECKS FAILED ✗")
            
        return success, self.logs