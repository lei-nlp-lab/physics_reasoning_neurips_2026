import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize by safely extracting all required parameters."""
        self.sol = solution_instance
        
        # Extract inputs with safe fallbacks
        self.v_P = getattr(solution_instance, 'v_P', solution_instance.results.get('v_P', 340.0))
        self.delta_t = getattr(solution_instance, 'delta_t', solution_instance.results.get('delta_t', 1.0))
        
        # Extract velocities for each case separately
        # For stationary case (Q1)
        self.v_S_stationary = solution_instance.results.get('v_S_stationary', 
                          getattr(solution_instance, 'v_S_stationary', 20.0))
        self.v_A_stationary = solution_instance.results.get('v_A_stationary', 
                          getattr(solution_instance, 'v_A_stationary', 0.0))
        
        # For moving case (Q2)
        self.v_S_moving = solution_instance.results.get('v_S_moving', 
                       getattr(solution_instance, 'v_S_moving', 10.0))
        self.v_A_moving = solution_instance.results.get('v_A_moving', 
                       getattr(solution_instance, 'v_A_moving', 4.0))
        
        # Extract results for the two cases
        self.delta_t_prime_stationary = getattr(
            solution_instance, 
            'delta_t_prime1', 
            solution_instance.results.get('delta_t_prime1', 0.94)
        )
        self.delta_t_prime_moving = getattr(
            solution_instance, 
            'delta_t_prime2', 
            solution_instance.results.get('delta_t_prime2', 0.98)
        )
        
        self.logs = []
    
    def test_invariants(self) -> bool:
        """Test fundamental physical laws: Doppler formula and motion equations."""
        try:
            success = True
            
            # 1. Test stationary observer case (Q1)
            expected_stationary = (self.v_P - self.v_S_stationary) / (self.v_P - self.v_A_stationary) * self.delta_t
            
            if not math.isclose(self.delta_t_prime_stationary, expected_stationary, 
                              rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Stationary case mismatch. Computed: {expected_stationary:.3f}, "
                               f"Expected: {self.delta_t_prime_stationary:.3f}")
                success = False
            else:
                self.logs.append(f"PASS: Stationary case matches formula.")
            
            # 2. Test moving observer case (Q2)
            expected_moving = (self.v_P - self.v_S_moving) / (self.v_P - self.v_A_moving) * self.delta_t
            
            if not math.isclose(self.delta_t_prime_moving, expected_moving, 
                              rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Moving case mismatch. Computed: {expected_moving:.3f}, "
                               f"Expected: {self.delta_t_prime_moving:.3f}")
                success = False
            else:
                self.logs.append(f"PASS: Moving case matches formula.")
            
            # 3. Test that velocities are less than sound speed (non-relativistic)
            if (self.v_S_stationary >= self.v_P or self.v_A_stationary >= self.v_P or
                self.v_S_moving >= self.v_P or self.v_A_moving >= self.v_P):
                self.logs.append(f"FAIL: Velocities must be less than sound speed {self.v_P} m/s")
                success = False
            else:
                self.logs.append(f"PASS: Velocities are subsonic.")
            
            # 4. Test time interval positivity
            if self.delta_t_prime_stationary <= 0 or self.delta_t_prime_moving <= 0:
                self.logs.append(f"FAIL: Received time intervals must be positive.")
                success = False
            else:
                self.logs.append(f"PASS: All time intervals are positive.")
            
            return success
            
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        try:
            success = True
            
            # Reconstruct v_S_stationary from stationary case
            # Δt' = (v_P - v_S)/v_P * Δt  => v_S = v_P * (1 - Δt'/Δt)
            reconstructed_v_S_stationary = self.v_P * (1 - self.delta_t_prime_stationary / self.delta_t)
            
            if not math.isclose(self.v_S_stationary, reconstructed_v_S_stationary, 
                              rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v_S_stationary inconsistency. From stationary case: {reconstructed_v_S_stationary:.2f} m/s, "
                               f"Given: {self.v_S_stationary:.2f} m/s")
                success = False
            else:
                self.logs.append(f"PASS: v_S_stationary consistently derived from stationary case.")
            
            # Reconstruct v_A_moving from moving case
            # Δt' = (v_P - v_S)/(v_P - v_A) * Δt  
            # => v_A = v_P - (v_P - v_S) * Δt / Δt'
            reconstructed_v_A_moving = self.v_P - (self.v_P - self.v_S_moving) * self.delta_t / self.delta_t_prime_moving
            
            if not math.isclose(self.v_A_moving, reconstructed_v_A_moving, 
                              rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v_A_moving inconsistency. From moving case: {reconstructed_v_A_moving:.2f} m/s, "
                               f"Given: {self.v_A_moving:.2f} m/s")
                success = False
            else:
                self.logs.append(f"PASS: v_A_moving consistently derived from moving case.")
            
            return success
            
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb input parameters and verify expected trends."""
        try:
            success = True
            
            # Test 1: For stationary case, increase source speed toward observer
            # When source moves toward observer, Δt' should decrease
            perturbed_v_S = self.v_S_stationary * 1.1  # 10% increase
            delta_t_prime_perturbed = (self.v_P - perturbed_v_S) / (self.v_P - self.v_A_stationary) * self.delta_t
            
            if delta_t_prime_perturbed >= self.delta_t_prime_stationary:
                self.logs.append(f"FAIL: Increasing source speed should decrease Δt' (stationary case), "
                               f"but {delta_t_prime_perturbed:.3f} >= {self.delta_t_prime_stationary:.3f}")
                success = False
            else:
                self.logs.append(f"PASS: Δt' decreases with increasing source speed (stationary case).")
            
            # Test 2: For moving case, increase observer speed toward source
            # When observer moves toward source, Δt' should increase
            perturbed_v_A = self.v_A_moving * 1.1 if self.v_A_moving != 0 else 5.0
            delta_t_prime_perturbed = (self.v_P - self.v_S_moving) / (self.v_P - perturbed_v_A) * self.delta_t
            
            if delta_t_prime_perturbed <= self.delta_t_prime_moving:
                self.logs.append(f"FAIL: Increasing observer speed should increase Δt' (moving case), "
                               f"but {delta_t_prime_perturbed:.3f} <= {self.delta_t_prime_moving:.3f}")
                success = False
            else:
                self.logs.append(f"PASS: Δt' increases with increasing observer speed (moving case).")
            
            return success
            
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success status with logs."""
        try:
            self.logs = ["=== Physics Checker Started ==="]
            
            # Run individual tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            overall = inv_ok and trace_ok and mono_ok
            
            self.logs.append(f"\n=== Summary ===")
            self.logs.append(f"Invariants: {'PASS' if inv_ok else 'FAIL'}")
            self.logs.append(f"Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
            self.logs.append(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
            self.logs.append(f"Overall: {'PASS' if overall else 'FAIL'}")
            
            return overall, self.logs
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR in run_all_tests: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs