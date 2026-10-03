import math
import sympy as sp
from typing import Tuple, List, Dict, Any
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs
        self.L = getattr(solution_instance, 'L', 0.5)
        self.v = getattr(solution_instance, 'v', 0.2)
        self.B = getattr(solution_instance, 'B', 0.5)
        self.R = getattr(solution_instance, 'R', None)  # Circuit resistance
        self.r = getattr(solution_instance, 'r', 0.0)  # Internal resistance (if any)
        
        # Extract results safely
        self.I_ind_dir = getattr(solution_instance, 'I_ind_dir', self.results.get('I_ind_dir', None))
        self.E = getattr(solution_instance, 'E', self.results.get('E', None))
        self.I = getattr(solution_instance, 'I', self.results.get('I', None))
        self.F = getattr(solution_instance, 'F', self.results.get('F', None))
        
        # Store original values for monotonicity test
        self._original_B = self.B
        self._original_v = self.v
        self._original_L = self.L
        
        # Validation flags
        self.valid = all(x is not None for x in [self.E, self.I, self.F])
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify conservation laws and fundamental physics invariants."""
        logs = []
        success = True
        
        if not self.valid:
            logs.append("WARNING: Missing required result values for invariant tests")
            return False, logs
        
        try:
            # 1. Check motional EMF formula: E = B * L * v
            expected_E = self.B * self.L * self.v
            if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: Motional EMF mismatch. Got E={self.E}, expected B*L*v={expected_E}")
                success = False
            else:
                logs.append(f"PASS: E = B*L*v holds ({self.E} ≈ {expected_E})")
            
            # 2. Check total resistance (R_total = R + r) if available
            if self.R is not None:
                R_total = self.R + self.r
                # Verify Ohm's Law: I = E / (R + r)
                expected_I = self.E / R_total
                if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"FAIL: Ohm's Law mismatch. Got I={self.I}, expected E/(R+r)={expected_I}")
                    success = False
                else:
                    logs.append(f"PASS: Ohm's Law holds with R_total={R_total} Ω")
            
            # 3. Check magnetic force formula: F = B * I * L
            expected_F = self.B * self.I * self.L
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-6):
                logs.append(f"FAIL: Magnetic force mismatch. Got F={self.F}, expected B*I*L={expected_F}")
                success = False
            else:
                logs.append(f"PASS: F = B*I*L holds ({self.F} ≈ {expected_F})")
            
            # 4. Check force balance for constant velocity (Newton's 1st Law)
            # For constant velocity, external force F_ext should equal magnetic force F_mag
            # From the problem: the external force magnitude equals the magnetic force
            # So check F (external) equals B*I*L (magnetic force on rod)
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-6):
                logs.append(f"FAIL: Force balance mismatch. External F={self.F}, magnetic force={expected_F}")
                success = False
            else:
                logs.append("PASS: Force balance for constant velocity holds")
            
            # 5. Energy conservation check: Power input = Power dissipated
            if self.R is not None:
                R_total = self.R + self.r
                power_input = self.F * self.v  # External power
                power_dissipated = (self.I ** 2) * R_total  # Joule heating
                if not math.isclose(power_input, power_dissipated, rel_tol=1e-2, abs_tol=1e-6):
                    logs.append(f"FAIL: Energy conservation mismatch. Power in={power_input}, Power dissipated={power_dissipated}")
                    success = False
                else:
                    logs.append(f"PASS: Energy conservation holds (Power balance)")
            
        except Exception as e:
            logs.append(f"ERROR in invariant tests: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Verify intermediate results are consistent with final answer."""
        logs = []
        success = True
        
        if not self.valid:
            logs.append("WARNING: Missing required result values for trace consistency")
            return False, logs
        
        try:
            # Step 2 → Step 3: E should lead to I via Ohm's Law
            if self.R is not None:
                R_total = self.R + self.r
                expected_I = self.E / R_total
                if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-4):
                    logs.append(f"FAIL: Step 2→3 inconsistency. E={self.E}V, R_total={R_total}Ω should give I={expected_I}A, got {self.I}A")
                    success = False
                else:
                    logs.append("PASS: E → I transition consistent")
            
            # Step 3 → Step 4: I should lead to F via F = B*I*L
            expected_F = self.B * self.I * self.L
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-6):
                logs.append(f"FAIL: Step 3→4 inconsistency. I={self.I}A should give F={expected_F}N, got {self.F}N")
                success = False
            else:
                logs.append("PASS: I → F transition consistent")
            
            # Check against target answers
            target_E = 0.05
            target_F = 6.25e-3
            
            if not math.isclose(self.E, target_E, rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: E={self.E}V doesn't match target {target_E}V")
                success = False
            else:
                logs.append(f"PASS: E matches target {target_E}V")
            
            if not math.isclose(self.F, target_F, rel_tol=1e-2, abs_tol=1e-6):
                logs.append(f"FAIL: F={self.F}N doesn't match target {target_F}N")
                success = False
            else:
                logs.append(f"PASS: F matches target {target_F}N")
            
            # Check direction consistency (qualitative)
            if self.I_ind_dir and "b to a" in str(self.I_ind_dir).lower():
                logs.append("PASS: Induced current direction 'b to a' is physically plausible")
            elif self.I_ind_dir:
                logs.append(f"WARNING: Unverified current direction: {self.I_ind_dir}")
            else:
                logs.append("WARNING: No current direction specified")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test if perturbing inputs produces expected trends."""
        logs = []
        success = True
        
        if not self.valid:
            logs.append("WARNING: Missing required result values for monotonicity test")
            return False, logs
        
        try:
            # Test 1: Increase B by 10%
            perturbed_B = self._original_B * 1.1
            
            # Recalculate using formulas
            new_E = perturbed_B * self.L * self.v
            if self.R is not None:
                R_total = self.R + self.r
                new_I = new_E / R_total
                new_F = perturbed_B * new_I * self.L
                
                # Check trends
                if not (new_E > self.E):
                    logs.append(f"FAIL: Increasing B should increase E, but {new_E} ≤ {self.E}")
                    success = False
                else:
                    logs.append(f"PASS: E increases with B ({self.E} → {new_E})")
                
                if not (new_I > self.I):
                    logs.append(f"FAIL: Increasing B should increase I, but {new_I} ≤ {self.I}")
                    success = False
                else:
                    logs.append(f"PASS: I increases with B ({self.I} → {new_I})")
                
                if not (new_F > self.F):
                    logs.append(f"FAIL: Increasing B should increase F, but {new_F} ≤ {self.F}")
                    success = False
                else:
                    logs.append(f"PASS: F increases with B ({self.F} → {new_F})")
            
            # Test 2: Increase v by 10%
            perturbed_v = self._original_v * 1.1
            new_E = self.B * self.L * perturbed_v
            if self.R is not None:
                new_I = new_E / (self.R + self.r)
                new_F = self.B * new_I * self.L
                
                if not (new_E > self.E):
                    logs.append(f"FAIL: Increasing v should increase E, but {new_E} ≤ {self.E}")
                    success = False
                else:
                    logs.append(f"PASS: E increases with v ({self.E} → {new_E})")
                
                if not (new_I > self.I):
                    logs.append(f"FAIL: Increasing v should increase I, but {new_I} ≤ {self.I}")
                    success = False
                else:
                    logs.append(f"PASS: I increases with v ({self.I} → {new_I})")
                
                if not (new_F > self.F):
                    logs.append(f"FAIL: Increasing v should increase F, but {new_F} ≤ {self.F}")
                    success = False
                else:
                    logs.append(f"PASS: F increases with v ({self.F} → {new_F})")
            
            # Restore original values
            self.B = self._original_B
            self.v = self._original_v
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            success = False
            # Restore on error
            self.B = self._original_B
            self.v = self._original_v
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics consistency checks."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*50}")
            all_logs.append(f"Running {test_name}:")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
                all_logs.append(f"{test_name}: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\n{'='*50}")
        all_logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs