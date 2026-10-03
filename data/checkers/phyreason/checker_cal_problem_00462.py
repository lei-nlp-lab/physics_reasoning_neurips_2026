import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        NEVER access solution_instance.attr directly - use getattr with fallback.
        """
        # Physics constants
        self.g = getattr(solution_instance, 'g', 9.81)  # Standard gravity
        
        # Input parameters (from problem statement)
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'mass', 70.0))
        self.h = getattr(solution_instance, 'h', getattr(solution_instance, 'height', 10.0))
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'initial_velocity', 10.0))
        
        # Results (from steps analysis)
        # Use safe extraction: first try direct attribute, then results dict, then default
        results_dict = getattr(solution_instance, 'results', {})
        
        self.E_k_A = getattr(solution_instance, 'E_k', 
                           getattr(solution_instance, 'E_{k}', 
                                  results_dict.get('E_k', results_dict.get('E_{k}', 0.0))))
        self.E_p_A = getattr(solution_instance, 'E_p',
                           getattr(solution_instance, 'E_{p}',
                                  results_dict.get('E_p', results_dict.get('E_{p}', 0.0))))
        self.E_total = getattr(solution_instance, 'E',
                             getattr(solution_instance, 'E_{total}',
                                    results_dict.get('E', results_dict.get('E_{total}', 0.0))))
        
        # v_B might be string or numeric, handle both
        v_B_raw = getattr(solution_instance, 'v_B',
                         getattr(solution_instance, 'v_{B}',
                                results_dict.get('v_B', results_dict.get('v_{B}', 0.0))))
        
        # Convert string representation to float if needed
        if isinstance(v_B_raw, str):
            # Handle formats like "10*sqrt(3)" or "$10\sqrt{3}\mathrm{m/s}$"
            clean_str = v_B_raw.replace('$', '').replace('\\mathrm{m/s}', '').replace('\\sqrt', 'sqrt')
            try:
                # Use sympy to evaluate expressions like 10*sqrt(3)
                self.v_B = float(sp.sympify(clean_str))
            except:
                self.v_B = 0.0
        else:
            self.v_B = float(v_B_raw)
        
        # Additional computed values for checking
        self.v_B_expected = None
        self.E_B_kinetic = None
        
    def test_invariants(self):
        """Test physics invariants: Energy Conservation and formula consistency."""
        logs = []
        success = True
        
        try:
            # 1. Check energy conservation between A and B
            # At A: E_total = E_k_A + E_p_A
            # At B: E_total = 0.5*m*v_B² + 0 (since h=0 at B)
            E_total_A = self.E_k_A + self.E_p_A
            self.E_B_kinetic = 0.5 * self.m * self.v_B**2
            
            if not math.isclose(E_total_A, self.E_B_kinetic, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Energy not conserved. E_total_A={E_total_A:.2f} J, "
                           f"E_kinetic_B={self.E_B_kinetic:.2f} J")
                success = False
            else:
                logs.append(f"PASS: Energy conserved within 2% tolerance")
            
            # 2. Check kinetic energy formula at A
            E_k_calculated = 0.5 * self.m * self.v0**2
            if not math.isclose(self.E_k_A, E_k_calculated, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Kinetic energy at A incorrect. Expected {E_k_calculated:.2f} J, "
                           f"got {self.E_k_A:.2f} J")
                success = False
            else:
                logs.append(f"PASS: Kinetic energy at A formula correct")
            
            # 3. Check potential energy formula at A
            E_p_calculated = self.m * self.g * self.h
            if not math.isclose(self.E_p_A, E_p_calculated, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Potential energy at A incorrect. Expected {E_p_calculated:.2f} J, "
                           f"got {self.E_p_A:.2f} J (using g={self.g})")
                success = False
            else:
                logs.append(f"PASS: Potential energy at A formula correct")
            
            # 4. Check total energy at A equals sum of parts
            if not math.isclose(self.E_total, self.E_k_A + self.E_p_A, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Total energy at A not sum of components. "
                           f"Expected {self.E_k_A + self.E_p_A:.2f} J, got {self.E_total:.2f} J")
                success = False
            else:
                logs.append(f"PASS: Total energy at A equals sum of kinetic and potential")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        logs = []
        success = True
        
        try:
            # 1. From steps analysis, E_total should be 10500 J
            if not math.isclose(self.E_total, 10500.0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"WARNING: Total energy {self.E_total:.2f} J differs from "
                           f"expected 10500 J (but may be using different g)")
            
            # 2. Calculate v_B from energy conservation: v_B = sqrt(2*E_total/m)
            self.v_B_expected = math.sqrt(2 * self.E_total / self.m)
            
            if not math.isclose(self.v_B, self.v_B_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_B inconsistent. From energy: {self.v_B_expected:.2f} m/s, "
                           f"reported: {self.v_B:.2f} m/s")
                success = False
            else:
                logs.append(f"PASS: v_B consistent with energy conservation")
            
            # 3. Check if v_B matches 10*sqrt(3) ≈ 17.32 m/s (from steps)
            v_B_theoretical = 10 * math.sqrt(3)
            if not math.isclose(self.v_B, v_B_theoretical, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"WARNING: v_B {self.v_B:.2f} m/s differs from theoretical "
                           f"10√3 ≈ {v_B_theoretical:.2f} m/s (check g value)")
            
            # 4. Verify individual steps from the analysis
            # Step 1: E_k at A = 3500 J
            if math.isclose(self.E_k_A, 3500.0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"PASS: Step 1 (E_k at A = 3500 J) verified")
            else:
                logs.append(f"WARNING: E_k at A {self.E_k_A:.2f} J differs from step 1 value 3500 J")
            
            # Step 2: E_p at A = 7000 J (assuming g=10 m/s²)
            if math.isclose(self.E_p_A, 7000.0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"PASS: Step 2 (E_p at A = 7000 J) verified")
            else:
                logs.append(f"WARNING: E_p at A {self.E_p_A:.2f} J differs from step 2 value 7000 J")
            
            # Step 3: E_total at A = 10500 J
            if math.isclose(self.E_total, 10500.0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"PASS: Step 3 (E_total at A = 10500 J) verified")
            else:
                logs.append(f"WARNING: E_total at A {self.E_total:.2f} J differs from step 3 value 10500 J")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs and checking trends."""
        logs = []
        success = True
        
        try:
            # Store original values
            original_m = self.m
            original_v0 = self.v0
            original_h = self.h
            
            # Test 1: Increase mass by 10%
            self.m *= 1.1
            
            # Recalculate energies
            new_E_k = 0.5 * self.m * original_v0**2
            new_E_p = self.m * self.g * original_h
            new_E_total = new_E_k + new_E_p
            new_v_B = math.sqrt(2 * new_E_total / self.m)
            
            # Check trends
            # Energy should increase proportionally with mass
            E_k_ratio = new_E_k / self.E_k_A if self.E_k_A != 0 else 1
            E_p_ratio = new_E_p / self.E_p_A if self.E_p_A != 0 else 1
            E_total_ratio = new_E_total / self.E_total if self.E_total != 0 else 1
            
            # Mass increased by 10%, so energies should increase by ~10%
            expected_ratio = 1.1
            
            if not math.isclose(E_k_ratio, expected_ratio, rel_tol=0.05, abs_tol=0.05):
                logs.append(f"FAIL: Kinetic energy doesn't scale linearly with mass. "
                           f"Ratio: {E_k_ratio:.3f}, expected: {expected_ratio:.3f}")
                success = False
            else:
                logs.append(f"PASS: Kinetic energy scales linearly with mass")
            
            if not math.isclose(E_p_ratio, expected_ratio, rel_tol=0.05, abs_tol=0.05):
                logs.append(f"FAIL: Potential energy doesn't scale linearly with mass. "
                           f"Ratio: {E_p_ratio:.3f}, expected: {expected_ratio:.3f}")
                success = False
            else:
                logs.append(f"PASS: Potential energy scales linearly with mass")
            
            if not math.isclose(E_total_ratio, expected_ratio, rel_tol=0.05, abs_tol=0.05):
                logs.append(f"FAIL: Total energy doesn't scale linearly with mass. "
                           f"Ratio: {E_total_ratio:.3f}, expected: {expected_ratio:.3f}")
                success = False
            else:
                logs.append(f"PASS: Total energy scales linearly with mass")
            
            # v_B should be independent of mass (from energy conservation)
            if not math.isclose(new_v_B, self.v_B, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_B depends on mass. Original: {self.v_B:.2f} m/s, "
                           f"new: {new_v_B:.2f} m/s")
                success = False
            else:
                logs.append(f"PASS: v_B is independent of mass (as expected)")
            
            # Restore mass
            self.m = original_m
            
            # Test 2: Increase initial height by 10%
            self.h *= 1.1
            
            new_E_p = self.m * self.g * self.h
            new_E_total = self.E_k_A + new_E_p  # E_k_A unchanged
            new_v_B = math.sqrt(2 * new_E_total / self.m)
            
            # v_B should increase with height
            if new_v_B <= self.v_B:
                logs.append(f"FAIL: v_B doesn't increase with height. "
                           f"Original: {self.v_B:.2f} m/s, new: {new_v_B:.2f} m/s")
                success = False
            else:
                logs.append(f"PASS: v_B increases with height")
            
            # Restore height
            self.h = original_h
            
            # Test 3: Increase initial velocity by 10%
            self.v0 *= 1.1
            
            new_E_k = 0.5 * self.m * self.v0**2
            new_E_total = new_E_k + self.E_p_A
            new_v_B = math.sqrt(2 * new_E_total / self.m)
            
            # v_B should increase with initial velocity
            if new_v_B <= self.v_B:
                logs.append(f"FAIL: v_B doesn't increase with initial velocity. "
                           f"Original: {self.v_B:.2f} m/s, new: {new_v_B:.2f} m/s")
                success = False
            else:
                logs.append(f"PASS: v_B increases with initial velocity")
            
            # Restore velocity
            self.v0 = original_v0
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            # Restore original values on error
            self.m = original_m
            self.h = original_h
            self.v0 = original_v0
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker for Sliding Athlete Problem ==="]
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("\n--- Test 1: Physics Invariants ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- Test 2: Trace Consistency ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- Test 3: Monotonicity Tests ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Summary
        all_logs.append("\n=== Summary ===")
        if overall_success:
            all_logs.append("ALL TESTS PASSED: Physics is consistent!")
        else:
            all_logs.append("SOME TESTS FAILED: Check physics assumptions and calculations")
        
        return overall_success, all_logs